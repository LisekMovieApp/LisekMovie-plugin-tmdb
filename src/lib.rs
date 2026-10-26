//! TMDB (The Movie Database) metadata provider extension for LisekMovie.
//!
//! Implements `lisek_plugin_api::MetadataProvider` (see that crate's own
//! doc comment for why this is a normal Cargo dependency compiled into
//! `lisek-server`, not a `.dll`/`.so` loaded at runtime) on top of a plain
//! HTTP client for TMDB's API v3: searches for a movie or TV series by
//! name and returns the best match's poster/backdrop URLs and overview.
//!
//! `base_url` is a constructor parameter on [`TmdbClient`] rather than a
//! hardcoded constant specifically so tests can point this client at a
//! local `wiremock` server instead of the real TMDB API — every test in
//! this crate runs against a mock and asserts on the exact request TMDB
//! would see (path, query params, auth) and on how a realistic TMDB
//! response gets mapped onto [`TmdbMovieMatch`]/[`TmdbSeriesMatch`] and
//! then onto [`lisek_plugin_api::ProviderMatch`].

use std::sync::LazyLock;
use std::time::Duration;

use async_trait::async_trait;
use lisek_plugin_api::{
    MediaKind, MetadataProvider, MovieQuery, PersonRole, ProviderCandidate, ProviderEpisode,
    ProviderError, ProviderMatch, ProviderPerson, ProviderPersonDetails, ProviderSeason,
    SeriesQuery,
};
use regex::Regex;
use serde::Deserialize;
use thiserror::Error;

const DEFAULT_BASE_URL: &str = "https://api.themoviedb.org/3";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);

/// Poster/backdrop image *paths* TMDB returns (e.g. `/abc123.jpg`) are not
/// full URLs — the actual image lives behind TMDB's separate image CDN,
/// sized on request. This builds a ready-to-download URL from one of those
/// paths; `size` is any size token TMDB's `/configuration` endpoint lists
/// for the relevant image type (`"w500"`, `"original"`, ...) — callers pick
/// one rather than this module hardcoding a single size, since a poster
/// thumbnail and a full backdrop download want different sizes.
pub fn image_url(image_path: &str, size: &str) -> String {
    format!("https://image.tmdb.org/t/p/{size}{image_path}")
}

#[derive(Debug, Error)]
pub enum TmdbError {
    #[error("request to TMDB failed: {0}")]
    Request(String),
    #[error("TMDB rejected the API key (unauthorized)")]
    Unauthorized,
    #[error("TMDB returned an unexpected response: {0}")]
    UnexpectedResponse(String),
    #[error("TMDB has no such item")]
    NotFound,
}

impl From<reqwest::Error> for TmdbError {
    fn from(err: reqwest::Error) -> Self {
        TmdbError::Request(err.to_string())
    }
}

impl From<TmdbError> for ProviderError {
    fn from(err: TmdbError) -> Self {
        match err {
            TmdbError::Request(msg) => ProviderError::Request(msg),
            TmdbError::Unauthorized => ProviderError::Unauthorized,
            TmdbError::UnexpectedResponse(msg) => ProviderError::UnexpectedResponse(msg),
            TmdbError::NotFound => ProviderError::UnexpectedResponse("not found".to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TmdbMovieMatch {
    pub tmdb_id: u64,
    pub title: String,
    pub year: Option<i32>,
    pub overview: String,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TmdbSeriesMatch {
    pub tmdb_id: u64,
    pub title: String,
    pub year: Option<i32>,
    pub overview: String,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
}

/// The poster/backdrop/logo `/images` picks out for one movie or series,
/// once [`pick_best_image`] has already applied the language-priority chain
/// — see [`TmdbClient::movie_images`]/[`TmdbClient::series_images`]. Any
/// field can still be `None`: TMDB genuinely has no image of that kind for
/// some titles, language priority or not — `logo_path` specifically is also
/// `None` whenever every logo TMDB has cataloged is a non-PNG format (see
/// [`TmdbClient::fetch_images`] for why only `.png` logos are considered).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TmdbImages {
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    pub logo_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SearchResponse<T> {
    results: Vec<T>,
}

#[derive(Debug, Clone, Deserialize)]
struct RawImage {
    file_path: String,
    iso_639_1: Option<String>,
    #[serde(default)]
    vote_average: f64,
}

#[derive(Debug, Deserialize)]
struct ImagesResponse {
    #[serde(default)]
    posters: Vec<RawImage>,
    #[serde(default)]
    backdrops: Vec<RawImage>,
    /// Transparent title-logo artwork — the asset the admin asked for so
    /// the Hero Banner can show a real logo image instead of plain HTML
    /// text. TMDB has no "banner" (wide artwork) image type at all
    /// alongside this one — see [`lisek_plugin_api::ProviderMatch`]'s own
    /// doc comment for why `banner_url` stays unset from this provider.
    #[serde(default)]
    logos: Vec<RawImage>,
}

/// Picks the best image out of one `/images` response array (`posters` or
/// `backdrops`), following the same three-step language chain regardless
/// of which server's primary UI language `primary_language` is: an image
/// tagged with the server's own language first, then English, then an
/// image with no language tag at all (TMDB's convention for a purely
/// visual poster/backdrop with no embedded text — the safest generic
/// choice), and only if none of those exist, whatever TMDB has at all —
/// covering a title whose only cataloged artwork is in some other
/// language entirely, which is still a real image and a better outcome
/// than showing nothing.
///
/// Deliberately does *not* ask TMDB's own `include_image_language` query
/// param to do this filtering server-side: that param can only narrow the
/// *set* of images returned, and TMDB sorts the combined set by
/// `vote_average` across every included language rather than grouping by
/// language first — which would let a highly-voted English poster outrank
/// a lower-voted (but still present) poster in the server's own language,
/// exactly backwards from what "prefer my language" should mean. Fetching
/// every image once and doing the priority selection here instead makes
/// the chain a hard, testable priority order rather than a vote-average
/// tiebreak that only usually goes the right way.
fn pick_best_image<'a>(images: &'a [RawImage], primary_language: &str) -> Option<&'a RawImage> {
    let best_in = |lang: &str| -> Option<&'a RawImage> {
        images
            .iter()
            .filter(|image| image.iso_639_1.as_deref() == Some(lang))
            .max_by(|a, b| {
                a.vote_average
                    .partial_cmp(&b.vote_average)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    };
    let best_untagged = || -> Option<&'a RawImage> {
        images
            .iter()
            .filter(|image| image.iso_639_1.is_none())
            .max_by(|a, b| {
                a.vote_average
                    .partial_cmp(&b.vote_average)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    };
    let best_overall = || -> Option<&'a RawImage> {
        images.iter().max_by(|a, b| {
            a.vote_average
                .partial_cmp(&b.vote_average)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    };

    best_in(primary_language)
        .or_else(|| (primary_language != "en").then(|| best_in("en")).flatten())
        .or_else(best_untagged)
        .or_else(best_overall)
}

#[derive(Debug, Deserialize)]
struct RawMovieResult {
    id: u64,
    title: String,
    #[serde(default)]
    release_date: String,
    #[serde(default)]
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawSeriesResult {
    id: u64,
    name: String,
    #[serde(default)]
    first_air_date: String,
    #[serde(default)]
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
}

/// The collection ("box set") a movie belongs to, from
/// `/movie/{id}`'s `belongs_to_collection`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TmdbCollectionRef {
    pub id: u64,
    pub name: String,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
}

/// What [`TmdbClient::movie_details`] adds to a search match.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TmdbMovieDetails {
    pub title: String,
    pub year: Option<i32>,
    pub overview: String,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    pub collection: Option<TmdbCollectionRef>,
    /// `(country, certification)` pairs, e.g. `("US", "PG-13")`.
    pub certifications: Vec<(String, String)>,
    /// Main cast, directors and writers; `None` without `credits`.
    pub people: Option<Vec<ProviderPerson>>,
    /// Original title, tagline, genres, studios, keywords, rating, ids.
    pub extras: TmdbExtras,
}

/// The detail-page facts (Faza 9) a movie or series page carries:
/// written into `provider_data` by [`add_extras`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TmdbExtras {
    pub original_title: Option<String>,
    pub tagline: Option<String>,
    pub runtime_minutes: Option<u32>,
    /// TMDB's user score (0-10), only with at least one vote.
    pub rating: Option<f64>,
    pub genres: Vec<String>,
    pub studios: Vec<String>,
    /// TMDB keywords ("car race", "sequel"), shown as tags.
    pub keywords: Vec<String>,
    pub imdb_id: Option<String>,
    pub tvdb_id: Option<u64>,
}

/// Keywords kept per title: TMDB lists dozens for popular films.
pub const MAX_KEYWORDS: usize = 20;

#[derive(Debug, Default, Deserialize)]
struct RawNamed {
    #[serde(default)]
    name: String,
}

/// Movies wrap keywords in `keywords`, series in `results`.
#[derive(Debug, Default, Deserialize)]
struct RawKeywords {
    #[serde(default)]
    keywords: Vec<RawNamed>,
    #[serde(default)]
    results: Vec<RawNamed>,
}

#[derive(Debug, Default, Deserialize)]
struct RawExternalIds {
    imdb_id: Option<String>,
    tvdb_id: Option<u64>,
}

fn names(list: Vec<RawNamed>, limit: usize) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for name in list.into_iter().map(|n| n.name.trim().to_string()) {
        if !name.is_empty() && !out.contains(&name) && out.len() < limit {
            out.push(name);
        }
    }
    out
}

fn rating_from(vote_average: Option<f64>, vote_count: Option<u64>) -> Option<f64> {
    let average = vote_average.filter(|v| v.is_finite() && *v > 0.0)?;
    (vote_count.unwrap_or(0) > 0).then(|| (average * 10.0).round() / 10.0)
}

/// An IMDb title id (`tt0322259`); anything else is dropped.
fn imdb_title_id(value: Option<String>) -> Option<String> {
    let id = blank_to_none(value)?;
    (id.starts_with("tt") && id.len() > 2 && id[2..].chars().all(|c| c.is_ascii_digit())).then_some(id)
}

/// `fields.original_title`, `tagline`, `runtime_minutes`, `rating`,
/// `genres`, `studios`, `tags`, `imdb_id`, `tvdb_id`. Lists are always
/// written (empty when TMDB has none); single values only when known.
fn add_extras(fields: &mut serde_json::Value, extras: &TmdbExtras) {
    let mut set = |key: &str, value: serde_json::Value| fields[key] = value;
    if let Some(v) = &extras.original_title {
        set("original_title", serde_json::json!(v));
    }
    if let Some(v) = &extras.tagline {
        set("tagline", serde_json::json!(v));
    }
    if let Some(v) = extras.runtime_minutes {
        set("runtime_minutes", serde_json::json!(v));
    }
    if let Some(v) = extras.rating {
        set("rating", serde_json::json!(v));
    }
    set("genres", serde_json::json!(extras.genres));
    set("studios", serde_json::json!(extras.studios));
    set("tags", serde_json::json!(extras.keywords));
    if let Some(v) = &extras.imdb_id {
        set("imdb_id", serde_json::json!(v));
    }
    if let Some(v) = extras.tvdb_id {
        set("tvdb_id", serde_json::json!(v));
    }
}

#[derive(Debug, Deserialize)]
struct RawMovieDetails {
    #[serde(default)]
    title: String,
    #[serde(default)]
    release_date: String,
    #[serde(default)]
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
    belongs_to_collection: Option<TmdbCollectionRef>,
    #[serde(default)]
    release_dates: Option<RawReleaseDates>,
    #[serde(default)]
    credits: Option<RawCredits>,
    original_title: Option<String>,
    tagline: Option<String>,
    runtime: Option<u32>,
    vote_average: Option<f64>,
    vote_count: Option<u64>,
    #[serde(default)]
    genres: Vec<RawNamed>,
    #[serde(default)]
    production_companies: Vec<RawNamed>,
    imdb_id: Option<String>,
    #[serde(default)]
    keywords: Option<RawKeywords>,
}

/// Actors billed beyond this are left out (TMDB lists every extra).
pub const MAX_ACTORS: usize = 20;
/// Directors and writers kept per title.
pub const MAX_CREW_PER_ROLE: usize = 4;

/// `/movie/{id}/credits` (appended as `credits`).
#[derive(Debug, Default, Deserialize)]
struct RawCredits {
    #[serde(default)]
    cast: Vec<RawCast>,
    #[serde(default)]
    crew: Vec<RawCrew>,
}

#[derive(Debug, Deserialize)]
struct RawCast {
    id: u64,
    #[serde(default)]
    name: String,
    character: Option<String>,
    #[serde(default)]
    order: i32,
    profile_path: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RawCrew {
    id: u64,
    #[serde(default)]
    name: String,
    #[serde(default)]
    job: String,
    profile_path: Option<String>,
}

/// `/tv/{id}/aggregate_credits`: every season's credits folded per person.
#[derive(Debug, Default, Deserialize)]
struct RawAggregateCredits {
    #[serde(default)]
    cast: Vec<RawAggregateCast>,
    #[serde(default)]
    crew: Vec<RawAggregateCrew>,
}

#[derive(Debug, Deserialize)]
struct RawAggregateCast {
    id: u64,
    #[serde(default)]
    name: String,
    #[serde(default)]
    order: i32,
    profile_path: Option<String>,
    #[serde(default)]
    roles: Vec<RawAggregateRole>,
}

#[derive(Debug, Deserialize)]
struct RawAggregateRole {
    character: Option<String>,
    #[serde(default)]
    episode_count: u32,
}

#[derive(Debug, Deserialize)]
struct RawAggregateCrew {
    id: u64,
    #[serde(default)]
    name: String,
    profile_path: Option<String>,
    #[serde(default)]
    jobs: Vec<RawAggregateJob>,
}

#[derive(Debug, Deserialize)]
struct RawAggregateJob {
    #[serde(default)]
    job: String,
    #[serde(default)]
    episode_count: u32,
}

/// The crew role a TMDB job counts as, if any.
fn crew_role(job: &str) -> Option<PersonRole> {
    match job {
        "Director" => Some(PersonRole::Director),
        "Screenplay" | "Writer" | "Teleplay" | "Story" | "Novel" => Some(PersonRole::Writer),
        _ => None,
    }
}

fn blank_to_none(value: Option<String>) -> Option<String> {
    value.map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

/// Credits in the order TMDB bills them: the first [`MAX_ACTORS`] actors,
/// then up to [`MAX_CREW_PER_ROLE`] directors and writers each (one entry
/// per person and role). Nameless entries are dropped.
fn people_from_credits(raw: RawCredits) -> Vec<ProviderPerson> {
    let mut cast = raw.cast;
    cast.sort_by_key(|c| c.order);
    let mut people: Vec<ProviderPerson> = cast
        .into_iter()
        .filter(|c| !c.name.trim().is_empty())
        .take(MAX_ACTORS)
        .enumerate()
        .map(|(order, c)| ProviderPerson {
            provider_id: c.id.to_string(),
            name: c.name.trim().to_string(),
            role: PersonRole::Actor,
            character: blank_to_none(c.character),
            order: order as i32,
            profile_url: thumb(c.profile_path.as_deref()),
        })
        .collect();
    for role in [PersonRole::Director, PersonRole::Writer] {
        let mut seen = std::collections::HashSet::new();
        let crew = raw
            .crew
            .iter()
            .filter(|c| crew_role(&c.job) == Some(role) && !c.name.trim().is_empty())
            .filter(|c| seen.insert(c.id))
            .take(MAX_CREW_PER_ROLE);
        people.extend(crew.enumerate().map(|(order, c)| ProviderPerson {
            provider_id: c.id.to_string(),
            name: c.name.trim().to_string(),
            role,
            character: None,
            order: order as i32,
            profile_url: thumb(c.profile_path.as_deref()),
        }));
    }
    people
}

/// The same for a series: an actor's character is the one they play in
/// most episodes; directors and writers are ranked by episode count.
fn people_from_aggregate(raw: RawAggregateCredits) -> Vec<ProviderPerson> {
    let mut cast = raw.cast;
    cast.sort_by_key(|c| c.order);
    let mut people: Vec<ProviderPerson> = cast
        .into_iter()
        .filter(|c| !c.name.trim().is_empty())
        .take(MAX_ACTORS)
        .enumerate()
        .map(|(order, c)| {
            let character = c
                .roles
                .into_iter()
                .max_by_key(|r| r.episode_count)
                .and_then(|r| blank_to_none(r.character));
            ProviderPerson {
                provider_id: c.id.to_string(),
                name: c.name.trim().to_string(),
                role: PersonRole::Actor,
                character,
                order: order as i32,
                profile_url: thumb(c.profile_path.as_deref()),
            }
        })
        .collect();
    for role in [PersonRole::Director, PersonRole::Writer] {
        let mut ranked: Vec<(u32, &RawAggregateCrew)> = raw
            .crew
            .iter()
            .filter(|c| !c.name.trim().is_empty())
            .filter_map(|c| {
                let episodes: u32 = c
                    .jobs
                    .iter()
                    .filter(|j| crew_role(&j.job) == Some(role))
                    .map(|j| j.episode_count)
                    .sum();
                (episodes > 0).then_some((episodes, c))
            })
            .collect();
        ranked.sort_by_key(|(episodes, _)| std::cmp::Reverse(*episodes));
        people.extend(ranked.into_iter().take(MAX_CREW_PER_ROLE).enumerate().map(
            |(order, (_, c))| ProviderPerson {
                provider_id: c.id.to_string(),
                name: c.name.trim().to_string(),
                role,
                character: None,
                order: order as i32,
                profile_url: thumb(c.profile_path.as_deref()),
            },
        ));
    }
    people
}

/// `/person/{id}`.
#[derive(Debug, Deserialize)]
struct RawPerson {
    #[serde(default)]
    name: String,
    #[serde(default)]
    biography: String,
    birthday: Option<String>,
    deathday: Option<String>,
    place_of_birth: Option<String>,
    imdb_id: Option<String>,
    profile_path: Option<String>,
}

/// A `YYYY-MM-DD` date as TMDB writes it; anything else is dropped.
fn tmdb_date(value: Option<String>) -> Option<String> {
    let value = blank_to_none(value)?;
    let bytes = value.as_bytes();
    let shaped = value.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && value.chars().enumerate().all(|(i, c)| i == 4 || i == 7 || c.is_ascii_digit());
    shaped.then_some(value)
}

#[derive(Debug, Deserialize)]
struct RawReleaseDates {
    #[serde(default)]
    results: Vec<RawCountryReleases>,
}

#[derive(Debug, Deserialize)]
struct RawCountryReleases {
    iso_3166_1: String,
    #[serde(default)]
    release_dates: Vec<RawRelease>,
}

#[derive(Debug, Deserialize)]
struct RawRelease {
    #[serde(default)]
    certification: String,
    /// 1 premiere, 2 limited theatrical, 3 theatrical, 4 digital,
    /// 5 physical, 6 TV.
    #[serde(default, rename = "type")]
    release_type: u8,
}

#[derive(Debug, Deserialize)]
struct RawContentRatings {
    #[serde(default)]
    results: Vec<RawContentRating>,
}

/// `/tv/{id}` with `content_ratings` appended.
#[derive(Debug, Deserialize)]
struct RawSeriesDetails {
    #[serde(default)]
    name: String,
    #[serde(default)]
    first_air_date: String,
    #[serde(default)]
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
    #[serde(default)]
    content_ratings: Option<RawContentRatings>,
    original_name: Option<String>,
    tagline: Option<String>,
    #[serde(default)]
    episode_run_time: Vec<u32>,
    vote_average: Option<f64>,
    vote_count: Option<u64>,
    #[serde(default)]
    genres: Vec<RawNamed>,
    #[serde(default)]
    production_companies: Vec<RawNamed>,
    #[serde(default)]
    keywords: Option<RawKeywords>,
    #[serde(default)]
    external_ids: Option<RawExternalIds>,
    #[serde(default)]
    seasons: Vec<RawSeason>,
}

/// `/tv/{id}`'s `seasons`.
#[derive(Debug, Deserialize)]
struct RawSeason {
    season_number: Option<i32>,
    name: Option<String>,
    overview: Option<String>,
    poster_path: Option<String>,
    episode_count: Option<i32>,
    air_date: Option<String>,
}

/// `/tv/{id}/season/{n}`.
#[derive(Debug, Deserialize)]
struct RawSeasonDetails {
    #[serde(default)]
    episodes: Vec<RawEpisode>,
}

#[derive(Debug, Deserialize)]
struct RawEpisode {
    season_number: Option<i32>,
    episode_number: Option<i32>,
    name: Option<String>,
    overview: Option<String>,
    runtime: Option<u32>,
    vote_average: Option<f64>,
    vote_count: Option<u64>,
    still_path: Option<String>,
    air_date: Option<String>,
}

fn seasons_from(raw: Vec<RawSeason>) -> Vec<ProviderSeason> {
    let mut seasons: Vec<ProviderSeason> = raw
        .into_iter()
        .filter_map(|s| {
            Some(ProviderSeason {
                number: s.season_number.filter(|n| *n >= 0)?,
                name: blank_to_none(s.name),
                overview: blank_to_none(s.overview),
                poster_url: s.poster_path.as_deref().map(|p| image_url(p, "w500")),
                episode_count: s.episode_count.filter(|c| *c > 0),
                air_year: s.air_date.as_deref().and_then(year_from_release_date),
            })
        })
        .collect();
    seasons.sort_by_key(|s| s.number);
    seasons.dedup_by_key(|s| s.number);
    seasons
}

fn episodes_from(raw: RawSeasonDetails, season: i32) -> Vec<ProviderEpisode> {
    raw.episodes
        .into_iter()
        .filter_map(|e| {
            Some(ProviderEpisode {
                season: e.season_number.unwrap_or(season),
                episode: e.episode_number.filter(|n| *n >= 0)?,
                title: blank_to_none(e.name),
                overview: blank_to_none(e.overview),
                runtime_minutes: e.runtime.filter(|r| *r > 0),
                rating: rating_from(e.vote_average, e.vote_count),
                still_url: e.still_path.as_deref().map(|p| image_url(p, "w780")),
                air_date: tmdb_date(e.air_date),
            })
        })
        .collect()
}

/// What [`TmdbClient::series_details`] returns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TmdbSeriesDetails {
    pub title: String,
    pub year: Option<i32>,
    pub overview: String,
    pub poster_path: Option<String>,
    pub backdrop_path: Option<String>,
    /// `(country, rating)` pairs.
    pub ratings: Vec<(String, String)>,
    pub extras: TmdbExtras,
    pub seasons: Vec<ProviderSeason>,
}

fn ratings_from(raw: RawContentRatings) -> Vec<(String, String)> {
    raw.results
        .into_iter()
        .filter(|r| !r.rating.trim().is_empty())
        .map(|r| (r.iso_3166_1, r.rating.trim().to_string()))
        .collect()
}

#[derive(Debug, Deserialize)]
struct RawContentRating {
    iso_3166_1: String,
    #[serde(default)]
    rating: String,
}

/// One non-empty certification per country. A country lists one per
/// release; the theatrical one (type 3) wins, then the lowest type.
fn certifications_from(raw: RawReleaseDates) -> Vec<(String, String)> {
    raw.results
        .into_iter()
        .filter_map(|country| {
            let mut releases: Vec<RawRelease> = country
                .release_dates
                .into_iter()
                .filter(|r| !r.certification.trim().is_empty())
                .collect();
            releases.sort_by_key(|r| (r.release_type != 3, r.release_type));
            releases
                .into_iter()
                .next()
                .map(|r| (country.iso_3166_1, r.certification.trim().to_string()))
        })
        .collect()
}

/// Countries whose certification is used for the age rating, in order.
/// Fixed rather than derived from the scan language, so a rescan in
/// another language never changes what a restricted account can see.
const RATING_COUNTRIES: &[&str] = &["US", "GB", "DE", "FR", "PL", "NL", "AU", "CA"];

/// Minimum viewer age for one country's certification, `None` when the
/// value means "not rated" or is unknown.
pub fn age_from_certification(country: &str, certification: &str) -> Option<i32> {
    let cert = certification.trim().to_ascii_uppercase();
    let mapped = match (country, cert.as_str()) {
        // MPA (films) and TV Parental Guidelines.
        ("US", "G" | "TV-Y" | "TV-G") => Some(0),
        ("US", "TV-Y7" | "TV-Y7-FV") => Some(7),
        ("US", "PG" | "TV-PG") => Some(10),
        ("US", "PG-13") => Some(13),
        ("US", "TV-14") => Some(14),
        ("US", "R" | "TV-MA") => Some(17),
        ("US", "NC-17") => Some(18),
        ("US", _) => None,
        // BBFC.
        ("GB", "U") => Some(0),
        ("GB", "PG") => Some(8),
        ("GB", "12A") => Some(12),
        ("GB", "R18") => Some(18),
        // Australia (ACB).
        ("AU", "G") => Some(0),
        ("AU", "PG") => Some(8),
        ("AU", "M") => Some(15),
        ("AU", "MA15+" | "MA 15+") => Some(15),
        ("AU", "R18+" | "X18+") => Some(18),
        // "All ages" in several systems.
        (_, "U" | "TP" | "AL" | "ATP" | "L" | "BTL" | "0+") => Some(0),
        _ => None,
    };
    mapped.or_else(|| {
        // Numeric systems: FSK 12, 16, 12+, -12, M/16, K-7.
        let digits: String = cert
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit())
            .collect();
        digits.parse::<i32>().ok().filter(|age| (0..=21).contains(age))
    })
}

/// `(age, certification, country)` from the first country in
/// [`RATING_COUNTRIES`] with a usable certification, then any other.
pub fn pick_age_rating(certifications: &[(String, String)]) -> Option<(i32, String, String)> {
    let rated = |country: &str, cert: &str| {
        age_from_certification(country, cert).map(|age| (age, cert.to_string(), country.to_string()))
    };
    RATING_COUNTRIES
        .iter()
        .find_map(|preferred| {
            certifications
                .iter()
                .filter(|(country, _)| country == preferred)
                .find_map(|(country, cert)| rated(country, cert))
        })
        .or_else(|| {
            let mut others: Vec<&(String, String)> = certifications.iter().collect();
            others.sort();
            others.into_iter().find_map(|(country, cert)| rated(country, cert))
        })
}

/// Adds `age_rating`/`certification`/`certification_country` to a match's
/// fields when a certification maps to an age.
fn add_age_rating(fields: &mut serde_json::Value, certifications: &[(String, String)]) {
    if let Some((age, certification, country)) = pick_age_rating(certifications) {
        fields["age_rating"] = serde_json::json!(age);
        fields["certification"] = serde_json::json!(certification);
        fields["certification_country"] = serde_json::json!(country);
    }
}

static RELEASE_YEAR: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4})-\d{2}-\d{2}$").unwrap());

fn year_from_release_date(release_date: &str) -> Option<i32> {
    RELEASE_YEAR
        .captures(release_date)?
        .get(1)?
        .as_str()
        .parse()
        .ok()
}

impl From<RawMovieResult> for TmdbMovieMatch {
    fn from(raw: RawMovieResult) -> Self {
        TmdbMovieMatch {
            tmdb_id: raw.id,
            title: raw.title,
            year: year_from_release_date(&raw.release_date),
            overview: raw.overview,
            poster_path: raw.poster_path,
            backdrop_path: raw.backdrop_path,
        }
    }
}

impl From<RawSeriesResult> for TmdbSeriesMatch {
    fn from(raw: RawSeriesResult) -> Self {
        TmdbSeriesMatch {
            tmdb_id: raw.id,
            year: year_from_release_date(&raw.first_air_date),
            title: raw.name,
            overview: raw.overview,
            poster_path: raw.poster_path,
            backdrop_path: raw.backdrop_path,
        }
    }
}

pub struct TmdbClient {
    api_key: String,
    base_url: String,
    http: reqwest::Client,
}

impl TmdbClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Same as [`TmdbClient::new`] but pointed at a caller-chosen base URL —
    /// how this module's own tests substitute a local mock server; real
    /// callers should always use [`TmdbClient::new`].
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            http: reqwest::Client::builder().timeout(REQUEST_TIMEOUT).build().expect(
                "building the TMDB HTTP client failed — this only happens if the platform's TLS \
                 backend is broken, which is a startup-time configuration problem, not a normal \
                 runtime error worth threading a Result through every call site for",
            ),
        }
    }

    /// Searches TMDB for a movie by title (and, when known, release year —
    /// TMDB's `year` search param narrows results considerably for common
    /// titles, which matters for avoiding a remake/reboot mismatch),
    /// returning the best (first, TMDB's own relevance-ranked) match across
    /// an increasingly loose cascade of query candidates (see
    /// [`lisek_plugin_api::query_candidates`]) — the plain normalized
    /// title first, then a diacritics-stripped variant, then just its
    /// first two words — stopping at the first candidate that finds
    /// anything. `primary_language` is passed to TMDB as the `language`
    /// search param on every attempt: a query string in the server's own
    /// language (a Polish title, say) needs TMDB told what language it's
    /// in to match well against that language's translated/alternative
    /// titles, not just the English-default title TMDB assumes otherwise.
    pub async fn search_movie(
        &self,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Option<TmdbMovieMatch>, TmdbError> {
        for candidate in lisek_plugin_api::query_candidates(title) {
            if let Some(found) = self
                .search_movie_candidate(&candidate, year, primary_language)
                .await?
            {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    /// One query-candidate's worth of searching: the year-scoped attempt,
    /// then — only if that comes up empty and a year was actually given —
    /// a retry with the year dropped. Factored out of [`Self::search_movie`]
    /// so the query-candidate cascade above can call this once per
    /// candidate without duplicating the year-retry logic at each step.
    async fn search_movie_candidate(
        &self,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Option<TmdbMovieMatch>, TmdbError> {
        let result = self.search_movie_raw(title, year, primary_language).await?;
        if result.is_some() || year.is_none() {
            return Ok(result);
        }

        // A year-scoped search can come up empty even for a real match:
        // the year parsed from a filename can be off by one from TMDB's
        // own primary-release-date year (common for festival/international
        // releases, or a scene release dated by its home rather than
        // theatrical debut), and TMDB's `year` param is a strict filter,
        // not just a relevance hint. Retrying once without it trades a
        // little extra request noise for meaningfully fewer false "no
        // match" outcomes — still exactly as specific as a title with no
        // parsed year at all would have searched from the start.
        tracing::debug!(
            title,
            year,
            "year-scoped TMDB search found nothing — retrying without the year"
        );
        self.search_movie_raw(title, None, primary_language).await
    }

    async fn search_movie_raw(
        &self,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Option<TmdbMovieMatch>, TmdbError> {
        let mut query = vec![
            ("api_key", self.api_key.clone()),
            ("query", title.to_string()),
            ("language", primary_language.to_string()),
        ];
        if let Some(year) = year {
            query.push(("year", year.to_string()));
        }

        let response = self
            .http
            .get(format!("{}/search/movie", self.base_url))
            .query(&query)
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TmdbError::Unauthorized);
        }
        let response = response.error_for_status()?;

        let body: SearchResponse<RawMovieResult> = response
            .json()
            .await
            .map_err(|err| TmdbError::UnexpectedResponse(err.to_string()))?;

        Ok(body.results.into_iter().next().map(Into::into))
    }

    /// Searches TMDB for a TV series by title, across the same
    /// query-candidate cascade [`Self::search_movie`] uses (see
    /// [`lisek_plugin_api::query_candidates`]) — no year param exists for
    /// a series search here (the scanner's series detection doesn't parse
    /// one out of a series-level folder name), so there's no year-retry
    /// step, just the candidate loop. TMDB's own database covers TV well
    /// enough (not just movies) that this is a usable alternative to TVDB
    /// for series metadata, not just a movies-only client.
    pub async fn search_series(
        &self,
        title: &str,
        primary_language: &str,
    ) -> Result<Option<TmdbSeriesMatch>, TmdbError> {
        for candidate in lisek_plugin_api::query_candidates(title) {
            if let Some(found) = self.search_series_raw(&candidate, primary_language).await? {
                return Ok(Some(found));
            }
        }
        Ok(None)
    }

    async fn search_series_raw(
        &self,
        title: &str,
        primary_language: &str,
    ) -> Result<Option<TmdbSeriesMatch>, TmdbError> {
        let response = self
            .http
            .get(format!("{}/search/tv", self.base_url))
            .query(&[
                ("api_key", self.api_key.as_str()),
                ("query", title),
                ("language", primary_language),
            ])
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TmdbError::Unauthorized);
        }
        let response = response.error_for_status()?;

        let body: SearchResponse<RawSeriesResult> = response
            .json()
            .await
            .map_err(|err| TmdbError::UnexpectedResponse(err.to_string()))?;

        Ok(body.results.into_iter().next().map(Into::into))
    }

    /// Fetches every poster/backdrop TMDB has cataloged for one movie and
    /// picks the best one in each category by language priority — see
    /// [`pick_best_image`] for exactly what "best" means and why this
    /// deliberately fetches the unfiltered list rather than asking TMDB's
    /// `include_image_language` param to narrow it server-side.
    pub async fn movie_images(
        &self,
        tmdb_id: u64,
        primary_language: &str,
    ) -> Result<TmdbImages, TmdbError> {
        self.fetch_images(&format!("/movie/{tmdb_id}/images"), primary_language)
            .await
    }

    /// Same as [`TmdbClient::movie_images`] for a TV series.
    pub async fn series_images(
        &self,
        tmdb_id: u64,
        primary_language: &str,
    ) -> Result<TmdbImages, TmdbError> {
        self.fetch_images(&format!("/tv/{tmdb_id}/images"), primary_language)
            .await
    }

    /// `/movie/{id}` with its release dates: the collection the movie
    /// belongs to and every country's certification. `language` localizes
    /// the collection name and artwork.
    pub async fn movie_details(
        &self,
        tmdb_id: u64,
        primary_language: &str,
    ) -> Result<TmdbMovieDetails, TmdbError> {
        let body: RawMovieDetails = self
            .get_json(
                &format!("/movie/{tmdb_id}"),
                &[
                    ("language", primary_language),
                    ("append_to_response", "release_dates,credits,keywords"),
                ],
            )
            .await?;
        let extras = TmdbExtras {
            original_title: blank_to_none(body.original_title),
            tagline: blank_to_none(body.tagline),
            runtime_minutes: body.runtime.filter(|r| *r > 0),
            rating: rating_from(body.vote_average, body.vote_count),
            genres: names(body.genres, 10),
            studios: names(body.production_companies, 8),
            keywords: names(body.keywords.map(|k| k.keywords).unwrap_or_default(), MAX_KEYWORDS),
            imdb_id: imdb_title_id(body.imdb_id),
            tvdb_id: None,
        };
        Ok(TmdbMovieDetails {
            extras,
            people: body.credits.map(people_from_credits),
            title: body.title,
            year: year_from_release_date(&body.release_date),
            overview: body.overview,
            poster_path: body.poster_path,
            backdrop_path: body.backdrop_path,
            collection: body.belongs_to_collection,
            certifications: body.release_dates.map(certifications_from).unwrap_or_default(),
        })
    }

    /// `/tv/{id}/aggregate_credits`: the series' main cast and crew.
    pub async fn series_people(&self, tmdb_id: u64) -> Result<Vec<ProviderPerson>, TmdbError> {
        let body: RawAggregateCredits = self
            .get_json(&format!("/tv/{tmdb_id}/aggregate_credits"), &[])
            .await?;
        Ok(people_from_aggregate(body))
    }

    /// `/tv/{id}/season/{n}`: the season's episodes in `primary_language`.
    /// [`TmdbError::NotFound`] for an unknown series or season.
    pub async fn season_episodes(
        &self,
        tmdb_id: u64,
        season: i32,
        primary_language: &str,
    ) -> Result<Vec<ProviderEpisode>, TmdbError> {
        let body: RawSeasonDetails = self
            .get_json(
                &format!("/tv/{tmdb_id}/season/{season}"),
                &[("language", primary_language)],
            )
            .await?;
        Ok(episodes_from(body, season))
    }

    /// `/person/{id}` in `primary_language`; an empty biography is read
    /// again in English (TMDB rarely has one in every language).
    /// [`TmdbError::NotFound`] for an unknown id.
    pub async fn person(
        &self,
        tmdb_id: u64,
        primary_language: &str,
    ) -> Result<ProviderPersonDetails, TmdbError> {
        let path = format!("/person/{tmdb_id}");
        let body: RawPerson = self.get_json(&path, &[("language", primary_language)]).await?;
        let mut biography = body.biography.trim().to_string();
        if biography.is_empty() && !primary_language.starts_with("en") {
            match self.get_json::<RawPerson>(&path, &[("language", "en-US")]).await {
                Ok(english) => biography = english.biography.trim().to_string(),
                Err(err) => tracing::debug!(provider = "tmdb", %err, "english biography lookup failed"),
            }
        }
        Ok(ProviderPersonDetails {
            name: body.name.trim().to_string(),
            biography,
            birth_date: tmdb_date(body.birthday),
            death_date: tmdb_date(body.deathday),
            birth_place: blank_to_none(body.place_of_birth),
            imdb_id: blank_to_none(body.imdb_id),
            profile_url: body.profile_path.as_deref().map(|p| image_url(p, "h632")),
        })
    }

    /// `/tv/{id}/content_ratings`: `(country, rating)` pairs.
    pub async fn series_content_ratings(
        &self,
        tmdb_id: u64,
    ) -> Result<Vec<(String, String)>, TmdbError> {
        let body: RawContentRatings = self
            .get_json(&format!("/tv/{tmdb_id}/content_ratings"), &[])
            .await?;
        Ok(ratings_from(body))
    }

    /// `/tv/{id}` with its content ratings: what "Fix match" needs to
    /// apply a series picked by id. [`TmdbError::NotFound`] for an unknown id.
    pub async fn series_details(
        &self,
        tmdb_id: u64,
        primary_language: &str,
    ) -> Result<TmdbSeriesDetails, TmdbError> {
        let body: RawSeriesDetails = self
            .get_json(
                &format!("/tv/{tmdb_id}"),
                &[
                    ("language", primary_language),
                    ("append_to_response", "content_ratings,keywords,external_ids"),
                ],
            )
            .await?;
        let ids = body.external_ids.unwrap_or_default();
        let extras = TmdbExtras {
            original_title: blank_to_none(body.original_name),
            tagline: blank_to_none(body.tagline),
            runtime_minutes: body.episode_run_time.iter().copied().find(|r| *r > 0),
            rating: rating_from(body.vote_average, body.vote_count),
            genres: names(body.genres, 10),
            studios: names(body.production_companies, 8),
            keywords: names(body.keywords.map(|k| k.results).unwrap_or_default(), MAX_KEYWORDS),
            imdb_id: imdb_title_id(ids.imdb_id),
            tvdb_id: ids.tvdb_id.filter(|id| *id > 0),
        };
        Ok(TmdbSeriesDetails {
            extras,
            seasons: seasons_from(body.seasons),
            title: body.name,
            year: year_from_release_date(&body.first_air_date),
            overview: body.overview,
            poster_path: body.poster_path,
            backdrop_path: body.backdrop_path,
            ratings: body.content_ratings.map(ratings_from).unwrap_or_default(),
        })
    }

    /// Every movie result for one exact query (no shortening), best first.
    pub async fn search_movie_results(
        &self,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Vec<TmdbMovieMatch>, TmdbError> {
        let year = year.map(|y| y.to_string());
        let mut params = vec![("query", title), ("language", primary_language)];
        if let Some(year) = year.as_deref() {
            params.push(("year", year));
        }
        let body: SearchResponse<RawMovieResult> = self.get_json("/search/movie", &params).await?;
        Ok(body.results.into_iter().map(Into::into).collect())
    }

    /// Every series result for one exact query (no shortening), best first.
    pub async fn search_series_results(
        &self,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Vec<TmdbSeriesMatch>, TmdbError> {
        let year = year.map(|y| y.to_string());
        let mut params = vec![("query", title), ("language", primary_language)];
        if let Some(year) = year.as_deref() {
            params.push(("first_air_date_year", year));
        }
        let body: SearchResponse<RawSeriesResult> = self.get_json("/search/tv", &params).await?;
        Ok(body.results.into_iter().map(Into::into).collect())
    }

    async fn get_json<T: serde::de::DeserializeOwned>(
        &self,
        path: &str,
        params: &[(&str, &str)],
    ) -> Result<T, TmdbError> {
        let mut query = vec![("api_key", self.api_key.as_str())];
        query.extend_from_slice(params);
        let response = self
            .http
            .get(format!("{}{path}", self.base_url))
            .query(&query)
            .send()
            .await?;
        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TmdbError::Unauthorized);
        }
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(TmdbError::NotFound);
        }
        response
            .error_for_status()?
            .json()
            .await
            .map_err(|err| TmdbError::UnexpectedResponse(err.to_string()))
    }

    async fn fetch_images(
        &self,
        path: &str,
        primary_language: &str,
    ) -> Result<TmdbImages, TmdbError> {
        let response = self
            .http
            .get(format!("{}{path}", self.base_url))
            .query(&[("api_key", self.api_key.as_str())])
            .send()
            .await?;

        if response.status() == reqwest::StatusCode::UNAUTHORIZED {
            return Err(TmdbError::Unauthorized);
        }
        let response = response.error_for_status()?;

        let body: ImagesResponse = response
            .json()
            .await
            .map_err(|err| TmdbError::UnexpectedResponse(err.to_string()))?;

        // Only PNG logos are considered: TMDB's `logos` array can also
        // contain SVG entries, and this app serves whatever bytes it
        // downloads back out with a fixed `image/png` content-type (same
        // fixed-type approach `media.rs` already uses for posters/
        // backdrops as `image/jpeg`) rather than sniffing the real format
        // — accepting an SVG here would mean serving vector markup
        // mislabeled as a raster image. Skipping a non-PNG logo is treated
        // exactly like "TMDB has no logo for this title", not an error.
        let png_logos: Vec<RawImage> = body
            .logos
            .into_iter()
            .filter(|image| image.file_path.to_lowercase().ends_with(".png"))
            .collect();

        Ok(TmdbImages {
            poster_path: pick_best_image(&body.posters, primary_language)
                .map(|image| image.file_path.clone()),
            backdrop_path: pick_best_image(&body.backdrops, primary_language)
                .map(|image| image.file_path.clone()),
            logo_path: pick_best_image(&png_logos, primary_language)
                .map(|image| image.file_path.clone()),
        })
    }
}

/// Fetches a TMDB match's language-prioritized images and falls back to the
/// search result's own (non-prioritized) `poster_path`/`backdrop_path` when
/// the extra `/images` request itself fails — a title match is still worth
/// keeping even if this one follow-up call didn't succeed, and TMDB's
/// search-result image is still a real (if not language-optimal) picture
/// rather than nothing at all.
fn images_or_fallback(
    result: Result<TmdbImages, TmdbError>,
    title: &str,
    fallback_poster: Option<&str>,
    fallback_backdrop: Option<&str>,
) -> (Option<String>, Option<String>) {
    let images = match result {
        Ok(images) => images,
        Err(err) => {
            tracing::warn!(
                provider = "tmdb",
                %err,
                title,
                "language-prioritized image lookup failed — using the search result's default image instead"
            );
            TmdbImages::default()
        }
    };
    (
        images
            .poster_path
            .or_else(|| fallback_poster.map(str::to_string)),
        images
            .backdrop_path
            .or_else(|| fallback_backdrop.map(str::to_string)),
    )
}

/// `fields.collection` (or `null`) and the age rating from a movie's
/// details.
fn add_details(fields: &mut serde_json::Value, details: &TmdbMovieDetails) {
    fields["collection"] = details
        .collection
        .as_ref()
        .map(|c| {
            serde_json::json!({
                "id": c.id.to_string(),
                "name": c.name,
                "poster_url": c.poster_path.as_deref().map(|p| image_url(p, "w500")),
                "backdrop_url": c.backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            })
        })
        .unwrap_or(serde_json::Value::Null);
    add_age_rating(fields, &details.certifications);
    add_extras(fields, &details.extras);
}

/// A provider id as TMDB uses it (a number); anything else is no match.
fn parse_tmdb_id(provider_id: &str) -> Option<u64> {
    provider_id.trim().parse().ok()
}

/// Search results show a small poster.
fn thumb(path: Option<&str>) -> Option<String> {
    path.map(|p| image_url(p, "w185"))
}

/// The [`lisek_plugin_api::MetadataProvider`] implementation `lisek-
/// server`'s provider registry constructs one of per scan, from the API
/// key `provider_config` has stored for `"tmdb"` — see that crate's own
/// doc comment for the compile-time-extension architecture this fits into.
pub struct TmdbProvider {
    api_key: String,
    base_url: String,
}

impl TmdbProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Tests point the provider at a mock server.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
        }
    }

    fn client(&self) -> TmdbClient {
        TmdbClient::with_base_url(self.api_key.clone(), self.base_url.clone())
    }
}

#[async_trait]
impl MetadataProvider for TmdbProvider {
    fn provider_key(&self) -> &'static str {
        "tmdb"
    }

    async fn find_movie(
        &self,
        query: &MovieQuery<'_>,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let client = self.client();
        let Some(m) = client
            .search_movie(query.title, query.year, primary_language)
            .await?
        else {
            return Ok(None);
        };
        let images_result = client.movie_images(m.tmdb_id, primary_language).await;
        let logo_path = images_result
            .as_ref()
            .ok()
            .and_then(|images| images.logo_path.clone());
        let (poster_path, backdrop_path) = images_or_fallback(
            images_result,
            query.title,
            m.poster_path.as_deref(),
            m.backdrop_path.as_deref(),
        );
        let mut fields = serde_json::json!({
            "tmdb_id": m.tmdb_id,
            "overview": m.overview,
        });
        // Collection and certification. A failed lookup leaves both keys
        // out (the server then keeps the film's current collection); a
        // successful one without a collection says so with `null`.
        let people = match client.movie_details(m.tmdb_id, primary_language).await {
            Ok(details) => {
                add_details(&mut fields, &details);
                details.people
            }
            Err(err) => {
                tracing::warn!(
                    provider = "tmdb",
                    %err,
                    title = query.title,
                    "movie details lookup failed — no collection, age rating or cast this time"
                );
                None
            }
        };
        Ok(Some(ProviderMatch {
            poster_url: poster_path.as_deref().map(|p| image_url(p, "w500")),
            backdrop_url: backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            logo_url: logo_path.as_deref().map(|p| image_url(p, "w500")),
            banner_url: None,
            people,
            seasons: Vec::new(),
            fields,
        }))
    }

    async fn find_series(
        &self,
        query: &SeriesQuery<'_>,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let client = self.client();
        let Some(m) = client.search_series(query.title, primary_language).await? else {
            return Ok(None);
        };
        let images_result = client.series_images(m.tmdb_id, primary_language).await;
        let logo_path = images_result
            .as_ref()
            .ok()
            .and_then(|images| images.logo_path.clone());
        let (poster_path, backdrop_path) = images_or_fallback(
            images_result,
            query.title,
            m.poster_path.as_deref(),
            m.backdrop_path.as_deref(),
        );
        let mut fields = serde_json::json!({
            "tmdb_id": m.tmdb_id,
            "overview": m.overview,
        });
        let seasons = match client.series_details(m.tmdb_id, primary_language).await {
            Ok(details) => {
                add_age_rating(&mut fields, &details.ratings);
                add_extras(&mut fields, &details.extras);
                details.seasons
            }
            Err(err) => {
                tracing::warn!(
                    provider = "tmdb",
                    %err,
                    title = query.title,
                    "series details lookup failed — no age rating, genres, tags or seasons this time"
                );
                Vec::new()
            }
        };
        let people = series_people_or_none(&client, m.tmdb_id, query.title).await;
        Ok(Some(ProviderMatch {
            poster_url: poster_path.as_deref().map(|p| image_url(p, "w500")),
            backdrop_url: backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            logo_url: logo_path.as_deref().map(|p| image_url(p, "w500")),
            banner_url: None,
            people,
            seasons,
            fields,
        }))
    }

    async fn search(
        &self,
        kind: MediaKind,
        title: &str,
        year: Option<i32>,
        primary_language: &str,
    ) -> Result<Vec<ProviderCandidate>, ProviderError> {
        const LIMIT: usize = 10;
        let client = self.client();
        Ok(match kind {
            MediaKind::Movie => client
                .search_movie_results(title, year, primary_language)
                .await?
                .into_iter()
                .take(LIMIT)
                .map(|m| ProviderCandidate {
                    provider_id: m.tmdb_id.to_string(),
                    poster_url: thumb(m.poster_path.as_deref()),
                    title: m.title,
                    year: m.year,
                    overview: m.overview,
                })
                .collect(),
            MediaKind::Series => client
                .search_series_results(title, year, primary_language)
                .await?
                .into_iter()
                .take(LIMIT)
                .map(|m| ProviderCandidate {
                    provider_id: m.tmdb_id.to_string(),
                    poster_url: thumb(m.poster_path.as_deref()),
                    title: m.title,
                    year: m.year,
                    overview: m.overview,
                })
                .collect(),
        })
    }

    async fn lookup(
        &self,
        kind: MediaKind,
        provider_id: &str,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let Some(tmdb_id) = parse_tmdb_id(provider_id) else {
            return Ok(None);
        };
        let client = self.client();
        let (fields, title, images, poster, backdrop, people, seasons) = match kind {
            MediaKind::Movie => {
                let details = match client.movie_details(tmdb_id, primary_language).await {
                    Ok(details) => details,
                    Err(TmdbError::NotFound) => return Ok(None),
                    Err(err) => return Err(err.into()),
                };
                let mut fields = serde_json::json!({
                    "tmdb_id": tmdb_id,
                    "overview": details.overview,
                    "title": details.title,
                    "year": details.year,
                });
                add_details(&mut fields, &details);
                let images = client.movie_images(tmdb_id, primary_language).await;
                let people = details.people.clone();
                (fields, details.title, images, details.poster_path, details.backdrop_path, people, Vec::new())
            }
            MediaKind::Series => {
                let details = match client.series_details(tmdb_id, primary_language).await {
                    Ok(details) => details,
                    Err(TmdbError::NotFound) => return Ok(None),
                    Err(err) => return Err(err.into()),
                };
                let mut fields = serde_json::json!({
                    "tmdb_id": tmdb_id,
                    "overview": details.overview,
                    "title": details.title,
                    "year": details.year,
                });
                add_age_rating(&mut fields, &details.ratings);
                add_extras(&mut fields, &details.extras);
                let images = client.series_images(tmdb_id, primary_language).await;
                let people = series_people_or_none(&client, tmdb_id, &details.title).await;
                let seasons = details.seasons.clone();
                (fields, details.title, images, details.poster_path, details.backdrop_path, people, seasons)
            }
        };
        let logo_path = images.as_ref().ok().and_then(|i| i.logo_path.clone());
        let (poster_path, backdrop_path) =
            images_or_fallback(images, &title, poster.as_deref(), backdrop.as_deref());
        Ok(Some(ProviderMatch {
            poster_url: poster_path.as_deref().map(|p| image_url(p, "w500")),
            backdrop_url: backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            logo_url: logo_path.as_deref().map(|p| image_url(p, "w500")),
            banner_url: None,
            people,
            seasons,
            fields,
        }))
    }

    async fn season_episodes(
        &self,
        series_id: &str,
        season: i32,
        primary_language: &str,
    ) -> Result<Vec<ProviderEpisode>, ProviderError> {
        let Some(tmdb_id) = parse_tmdb_id(series_id) else {
            return Ok(Vec::new());
        };
        match self.client().season_episodes(tmdb_id, season, primary_language).await {
            Ok(episodes) => Ok(episodes),
            Err(TmdbError::NotFound) => Ok(Vec::new()),
            Err(err) => Err(err.into()),
        }
    }

    async fn person(
        &self,
        provider_id: &str,
        primary_language: &str,
    ) -> Result<Option<ProviderPersonDetails>, ProviderError> {
        let Some(tmdb_id) = parse_tmdb_id(provider_id) else {
            return Ok(None);
        };
        match self.client().person(tmdb_id, primary_language).await {
            Ok(person) => Ok(Some(person)),
            Err(TmdbError::NotFound) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }
}

/// A series' cast, or `None` (keep what the item has) when the lookup failed.
async fn series_people_or_none(
    client: &TmdbClient,
    tmdb_id: u64,
    title: &str,
) -> Option<Vec<ProviderPerson>> {
    match client.series_people(tmdb_id).await {
        Ok(people) => Some(people),
        Err(err) => {
            tracing::warn!(provider = "tmdb", %err, title, "series credits lookup failed — no cast this time");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[test]
    fn image_url_combines_size_and_path() {
        assert_eq!(
            image_url("/abc123.jpg", "w500"),
            "https://image.tmdb.org/t/p/w500/abc123.jpg"
        );
    }

    #[test]
    fn year_from_release_date_parses_a_full_date() {
        assert_eq!(year_from_release_date("2010-07-16"), Some(2010));
    }

    #[test]
    fn year_from_release_date_handles_an_empty_or_malformed_date() {
        assert_eq!(year_from_release_date(""), None);
        assert_eq!(year_from_release_date("2010"), None);
    }

    #[tokio::test]
    async fn search_movie_sends_the_expected_request_and_parses_the_first_result() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("api_key", "test-key"))
            // Normalized (lowercased) by `query_candidates` before this
            // client ever sees it — see the `normalize_query_title` tests
            // in `lisek-plugin-api` for why that's always safe to apply.
            .and(query_param("query", "inception"))
            .and(query_param("year", "2010"))
            .and(query_param("language", "en"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": 27205,
                    "title": "Inception",
                    "release_date": "2010-07-16",
                    "overview": "A thief who steals corporate secrets...",
                    "poster_path": "/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg",
                    "backdrop_path": "/s3TBrRGB1iav7gFOCNx3H31MoES.jpg"
                }]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client
            .search_movie("Inception", Some(2010), "en")
            .await
            .unwrap();

        assert_eq!(
            result,
            Some(TmdbMovieMatch {
                tmdb_id: 27205,
                title: "Inception".to_string(),
                year: Some(2010),
                overview: "A thief who steals corporate secrets...".to_string(),
                poster_path: Some("/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg".to_string()),
                backdrop_path: Some("/s3TBrRGB1iav7gFOCNx3H31MoES.jpg".to_string()),
            })
        );
    }

    #[tokio::test]
    async fn search_movie_omits_the_year_param_when_none_is_given() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "some movie"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client.search_movie("Some Movie", None, "en").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn search_movie_returns_none_when_tmdb_finds_nothing() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client
            .search_movie("Nonexistent Film Xyz", None, "en")
            .await
            .unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn search_movie_reports_unauthorized_for_a_rejected_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "status_message": "Invalid API key"
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("bad-key", server.uri());
        let result = client.search_movie("Inception", None, "en").await;
        assert!(matches!(result, Err(TmdbError::Unauthorized)));
    }

    #[tokio::test]
    async fn search_series_sends_the_expected_request_and_parses_the_first_result() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/tv"))
            .and(query_param("query", "breaking bad"))
            .and(query_param("language", "en"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": 1396,
                    "name": "Breaking Bad",
                    "overview": "A high school chemistry teacher...",
                    "poster_path": "/ggFHVNu6YYI5L9pCfOacjizRGt.jpg",
                    "backdrop_path": "/tsRy63Mu5cu8etL1X7ZLyf7UP1M.jpg"
                }]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client.search_series("Breaking Bad", "en").await.unwrap();

        assert_eq!(
            result,
            Some(TmdbSeriesMatch {
                tmdb_id: 1396,
                title: "Breaking Bad".to_string(),
                year: None,
                overview: "A high school chemistry teacher...".to_string(),
                poster_path: Some("/ggFHVNu6YYI5L9pCfOacjizRGt.jpg".to_string()),
                backdrop_path: Some("/tsRy63Mu5cu8etL1X7ZLyf7UP1M.jpg".to_string()),
            })
        );
    }

    #[tokio::test]
    async fn search_movie_retries_without_the_year_when_the_year_scoped_search_finds_nothing() {
        let server = MockServer::start().await;
        // The year-scoped request (what a parsed "2012" release year sends
        // first) finds nothing — a real-world mismatch between a parsed
        // filename year and TMDB's own primary-release-date year.
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "asterix and obelix"))
            .and(query_param("year", "2012"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .mount(&server)
            .await;
        // The retry, with the year dropped, is what actually finds it —
        // still the *first* query candidate (the plain normalized title),
        // never getting to the two-word fallback.
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "asterix and obelix"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": 76758,
                    "title": "Asterix and Obelix: God Save Britannia",
                    "release_date": "2012-10-03",
                    "overview": "Asterix and Obelix travel to Britannia...",
                    "poster_path": "/britannia.jpg",
                    "backdrop_path": "/britannia-bd.jpg"
                }]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client
            .search_movie("Asterix and Obelix", Some(2012), "en")
            .await
            .unwrap();

        assert_eq!(result.map(|m| m.tmdb_id), Some(76758));
    }

    #[tokio::test]
    async fn search_movie_does_not_retry_when_no_year_was_given_in_the_first_place() {
        let server = MockServer::start().await;
        // Only one route is registered at all — if `search_movie` retried
        // here, there would be nothing left to drop from the query, so a
        // second identical request would just hit this same mock again.
        // `.expect(1)` catches that: a retry loop or a duplicate request
        // fails the test instead of silently doubling TMDB traffic. "Some
        // Movie" is only two words, so `query_candidates` produces exactly
        // one candidate here too (its own first-two-words fallback would
        // duplicate the normalized form) — nothing left to cascade into.
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "some movie"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .expect(1)
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client.search_movie("Some Movie", None, "en").await.unwrap();
        assert_eq!(result, None);
    }

    #[tokio::test]
    async fn search_movie_falls_back_to_the_first_two_words_once_the_full_title_finds_nothing() {
        // The exact real-world regression this cascade exists for: TMDB
        // rejects the full, correctly-parsed title (both with and without
        // the year), so the last-resort two-word query is what actually
        // finds the match — see `query_candidates`'s own tests in
        // `lisek-plugin-api` for why "asterix i" is the third and last
        // candidate for this exact title.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param(
                "query",
                "asterix i obelix w służbie jej królewskiej mości",
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param(
                "query",
                "asterix i obelix w sluzbie jej krolewskiej mosci",
            ))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(serde_json::json!({ "results": [] })),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "asterix i"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": 76758,
                    "title": "Asterix and Obelix: God Save Britannia",
                    "release_date": "2012-10-03",
                    "overview": "Asterix and Obelix travel to Britannia...",
                    "poster_path": "/britannia.jpg",
                    "backdrop_path": "/britannia-bd.jpg"
                }]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let result = client
            .search_movie(
                "Asterix i Obelix W Służbie Jej Królewskiej Mości",
                Some(2012),
                "pl",
            )
            .await
            .unwrap();

        assert_eq!(result.map(|m| m.tmdb_id), Some(76758));
    }

    #[test]
    fn pick_best_image_prefers_the_primary_language_over_everything_else() {
        let images = vec![
            RawImage {
                file_path: "/en.jpg".to_string(),
                iso_639_1: Some("en".to_string()),
                vote_average: 9.0,
            },
            RawImage {
                file_path: "/pl.jpg".to_string(),
                iso_639_1: Some("pl".to_string()),
                vote_average: 1.0,
            },
        ];

        let best = pick_best_image(&images, "pl").unwrap();
        assert_eq!(best.file_path, "/pl.jpg");
    }

    #[test]
    fn pick_best_image_falls_back_to_english_when_the_primary_language_has_none() {
        let images = vec![
            RawImage {
                file_path: "/fr.jpg".to_string(),
                iso_639_1: Some("fr".to_string()),
                vote_average: 9.0,
            },
            RawImage {
                file_path: "/en.jpg".to_string(),
                iso_639_1: Some("en".to_string()),
                vote_average: 1.0,
            },
        ];

        let best = pick_best_image(&images, "pl").unwrap();
        assert_eq!(best.file_path, "/en.jpg");
    }

    #[test]
    fn pick_best_image_falls_back_to_an_untagged_image_before_a_different_language() {
        let images = vec![
            RawImage {
                file_path: "/fr.jpg".to_string(),
                iso_639_1: Some("fr".to_string()),
                vote_average: 9.0,
            },
            RawImage {
                file_path: "/untagged.jpg".to_string(),
                iso_639_1: None,
                vote_average: 1.0,
            },
        ];

        let best = pick_best_image(&images, "pl").unwrap();
        assert_eq!(best.file_path, "/untagged.jpg");
    }

    #[test]
    fn pick_best_image_falls_back_to_any_available_image_as_a_last_resort() {
        let images = vec![RawImage {
            file_path: "/fr.jpg".to_string(),
            iso_639_1: Some("fr".to_string()),
            vote_average: 9.0,
        }];

        let best = pick_best_image(&images, "pl").unwrap();
        assert_eq!(best.file_path, "/fr.jpg");
    }

    #[test]
    fn pick_best_image_returns_none_for_an_empty_list() {
        assert!(pick_best_image(&[], "pl").is_none());
    }

    #[test]
    fn pick_best_image_picks_the_highest_voted_image_within_the_winning_language() {
        let images = vec![
            RawImage {
                file_path: "/pl-low.jpg".to_string(),
                iso_639_1: Some("pl".to_string()),
                vote_average: 2.0,
            },
            RawImage {
                file_path: "/pl-high.jpg".to_string(),
                iso_639_1: Some("pl".to_string()),
                vote_average: 8.0,
            },
        ];

        let best = pick_best_image(&images, "pl").unwrap();
        assert_eq!(best.file_path, "/pl-high.jpg");
    }

    #[tokio::test]
    async fn movie_images_applies_the_language_chain_to_both_posters_and_backdrops() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/27205/images"))
            .and(query_param("api_key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [
                    { "file_path": "/poster-en.jpg", "iso_639_1": "en", "vote_average": 9.0 },
                    { "file_path": "/poster-pl.jpg", "iso_639_1": "pl", "vote_average": 3.0 }
                ],
                "backdrops": [
                    { "file_path": "/backdrop-null.jpg", "iso_639_1": null, "vote_average": 5.0 }
                ]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let images = client.movie_images(27205, "pl").await.unwrap();

        assert_eq!(images.poster_path, Some("/poster-pl.jpg".to_string()));
        assert_eq!(images.backdrop_path, Some("/backdrop-null.jpg".to_string()));
    }

    #[tokio::test]
    async fn movie_images_picks_a_png_logo_and_applies_the_same_language_chain() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/27205/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [],
                "backdrops": [],
                "logos": [
                    { "file_path": "/logo-en.png", "iso_639_1": "en", "vote_average": 9.0 },
                    { "file_path": "/logo-pl.png", "iso_639_1": "pl", "vote_average": 3.0 }
                ]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let images = client.movie_images(27205, "pl").await.unwrap();

        assert_eq!(images.logo_path, Some("/logo-pl.png".to_string()));
    }

    #[tokio::test]
    async fn movie_images_skips_a_non_png_logo_rather_than_serving_it_mislabeled() {
        // TMDB's `logos` array can carry SVG entries — this app serves
        // whatever it downloads back out with a fixed `image/png`
        // content-type, so a non-PNG logo is treated the same as "no
        // logo" rather than risking mislabeled/broken content. Only the
        // higher-voted PNG entry should win here, not the SVG despite its
        // higher vote_average.
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/27205/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [],
                "backdrops": [],
                "logos": [
                    { "file_path": "/logo-vector.svg", "iso_639_1": "en", "vote_average": 9.0 },
                    { "file_path": "/logo-raster.png", "iso_639_1": "en", "vote_average": 1.0 }
                ]
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let images = client.movie_images(27205, "en").await.unwrap();

        assert_eq!(images.logo_path, Some("/logo-raster.png".to_string()));
    }

    #[tokio::test]
    async fn movie_images_reports_unauthorized_for_a_rejected_api_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/27205/images"))
            .respond_with(ResponseTemplate::new(401).set_body_json(serde_json::json!({
                "status_message": "Invalid API key"
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("bad-key", server.uri());
        let result = client.movie_images(27205, "pl").await;
        assert!(matches!(result, Err(TmdbError::Unauthorized)));
    }

    #[tokio::test]
    async fn series_images_hits_the_tv_images_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tv/1396/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [
                    { "file_path": "/series-poster.jpg", "iso_639_1": "en", "vote_average": 5.0 }
                ],
                "backdrops": []
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let images = client.series_images(1396, "en").await.unwrap();

        assert_eq!(images.poster_path, Some("/series-poster.jpg".to_string()));
        assert_eq!(images.backdrop_path, None);
    }

    #[test]
    fn provider_key_is_tmdb() {
        assert_eq!(TmdbProvider::new("k").provider_key(), "tmdb");
    }

    #[test]
    fn certifications_map_to_minimum_ages() {
        assert_eq!(age_from_certification("US", "PG-13"), Some(13));
        assert_eq!(age_from_certification("US", "R"), Some(17));
        assert_eq!(age_from_certification("US", "TV-Y7"), Some(7));
        assert_eq!(age_from_certification("US", "NR"), None);
        assert_eq!(age_from_certification("GB", "12A"), Some(12));
        assert_eq!(age_from_certification("GB", "U"), Some(0));
        assert_eq!(age_from_certification("DE", "FSK 16"), Some(16));
        assert_eq!(age_from_certification("DE", "6"), Some(6));
        assert_eq!(age_from_certification("PL", "12+"), Some(12));
        assert_eq!(age_from_certification("FR", "TP"), Some(0));
        assert_eq!(age_from_certification("FR", "-16"), Some(16));
        assert_eq!(age_from_certification("BR", "L"), Some(0));
        assert_eq!(age_from_certification("XX", "Unrated"), None);
        assert_eq!(age_from_certification("XX", "99"), None);
    }

    #[test]
    fn age_rating_prefers_the_fixed_country_order_then_any_country() {
        let certs = vec![
            ("PL".to_string(), "15".to_string()),
            ("US".to_string(), "NR".to_string()),
            ("DE".to_string(), "12".to_string()),
        ];
        // US has no usable rating; DE comes before PL in the fixed order.
        assert_eq!(
            pick_age_rating(&certs),
            Some((12, "12".to_string(), "DE".to_string()))
        );
        let others = vec![
            ("SE".to_string(), "11".to_string()),
            ("BR".to_string(), "14".to_string()),
        ];
        assert_eq!(
            pick_age_rating(&others),
            Some((14, "14".to_string(), "BR".to_string()))
        );
        assert_eq!(pick_age_rating(&[]), None);
    }

    #[tokio::test]
    async fn movie_details_reads_the_collection_and_the_theatrical_certification() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/11"))
            .and(query_param("api_key", "test-key"))
            .and(query_param("language", "pl"))
            .and(query_param("append_to_response", "release_dates,credits,keywords"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": 11,
                "belongs_to_collection": {
                    "id": 10, "name": "Gwiezdne wojny - kolekcja",
                    "poster_path": "/sw.jpg", "backdrop_path": "/sw-bg.jpg"
                },
                "release_dates": { "results": [
                    { "iso_3166_1": "US", "release_dates": [
                        { "certification": "", "type": 1 },
                        { "certification": "PG", "type": 3 },
                        { "certification": "G", "type": 5 }
                    ]},
                    { "iso_3166_1": "DE", "release_dates": [
                        { "certification": "", "type": 3 }
                    ]}
                ]}
            })))
            .mount(&server)
            .await;

        let client = TmdbClient::with_base_url("test-key", server.uri());
        let details = client.movie_details(11, "pl").await.unwrap();

        assert_eq!(
            details.collection,
            Some(TmdbCollectionRef {
                id: 10,
                name: "Gwiezdne wojny - kolekcja".to_string(),
                poster_path: Some("/sw.jpg".to_string()),
                backdrop_path: Some("/sw-bg.jpg".to_string()),
            })
        );
        assert_eq!(details.certifications, vec![("US".to_string(), "PG".to_string())]);
    }

    #[test]
    fn movie_credits_keep_billing_order_and_main_crew() {
        let raw: RawCredits = serde_json::from_value(serde_json::json!({
            "cast": [
                { "id": 3, "name": "Harrison Ford", "character": "Han Solo", "order": 1, "profile_path": "/hf.jpg" },
                { "id": 2, "name": "Mark Hamill", "character": "Luke Skywalker", "order": 0, "profile_path": null },
                { "id": 9, "name": " ", "character": "Nobody", "order": 2 }
            ],
            "crew": [
                { "id": 1, "name": "George Lucas", "job": "Director", "profile_path": "/gl.jpg" },
                { "id": 1, "name": "George Lucas", "job": "Writer" },
                { "id": 1, "name": "George Lucas", "job": "Screenplay" },
                { "id": 5, "name": "John Williams", "job": "Original Music Composer" }
            ]
        }))
        .unwrap();
        let people = people_from_credits(raw);
        let summary: Vec<_> = people
            .iter()
            .map(|p| (p.provider_id.as_str(), p.role, p.character.as_deref(), p.order))
            .collect();
        assert_eq!(
            summary,
            vec![
                ("2", PersonRole::Actor, Some("Luke Skywalker"), 0),
                ("3", PersonRole::Actor, Some("Han Solo"), 1),
                ("1", PersonRole::Director, None, 0),
                ("1", PersonRole::Writer, None, 0),
            ]
        );
        assert_eq!(people[1].profile_url.as_deref(), Some("https://image.tmdb.org/t/p/w185/hf.jpg"));
        assert_eq!(people[0].profile_url, None);
    }

    #[test]
    fn movie_credits_cap_the_cast() {
        let cast: Vec<_> = (0..40)
            .map(|i| serde_json::json!({ "id": i, "name": format!("Actor {i}"), "order": i }))
            .collect();
        let raw: RawCredits = serde_json::from_value(serde_json::json!({ "cast": cast })).unwrap();
        assert_eq!(people_from_credits(raw).len(), MAX_ACTORS);
    }

    #[test]
    fn series_credits_use_the_main_character_and_rank_crew_by_episodes() {
        let raw: RawAggregateCredits = serde_json::from_value(serde_json::json!({
            "cast": [{ "id": 7, "name": "Bryan Cranston", "order": 0, "profile_path": "/bc.jpg",
                       "roles": [{ "character": "Narrator", "episode_count": 1 },
                                 { "character": "Walter White", "episode_count": 62 }] }],
            "crew": [
                { "id": 8, "name": "Rare Director", "jobs": [{ "job": "Director", "episode_count": 1 }] },
                { "id": 9, "name": "Main Director", "jobs": [{ "job": "Director", "episode_count": 5 }] },
                { "id": 10, "name": "Grip", "jobs": [{ "job": "Key Grip", "episode_count": 60 }] }
            ]
        }))
        .unwrap();
        let people = people_from_aggregate(raw);
        assert_eq!(people[0].character.as_deref(), Some("Walter White"));
        let directors: Vec<_> = people
            .iter()
            .filter(|p| p.role == PersonRole::Director)
            .map(|p| p.name.as_str())
            .collect();
        assert_eq!(directors, ["Main Director", "Rare Director"]);
        assert_eq!(people.len(), 3);
    }

    #[test]
    fn ratings_need_votes_and_imdb_ids_their_prefix() {
        assert_eq!(rating_from(Some(6.534), Some(3)), Some(6.5));
        assert_eq!(rating_from(Some(7.0), Some(0)), None);
        assert_eq!(rating_from(Some(0.0), Some(10)), None);
        assert_eq!(imdb_title_id(Some("tt0322259".into())).as_deref(), Some("tt0322259"));
        assert_eq!(imdb_title_id(Some("nm0000148".into())), None);
        assert_eq!(imdb_title_id(Some("".into())), None);
    }

    #[tokio::test]
    async fn season_episodes_read_titles_stills_and_scores() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tv/1396/season/1"))
            .and(query_param("language", "pl"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "episodes": [
                    { "episode_number": 1, "season_number": 1, "name": "Najwyższy czas", "overview": "Walter...",
                      "runtime": 58, "vote_average": 8.3, "vote_count": 120, "still_path": "/e1.jpg", "air_date": "2008-01-20" },
                    { "episode_number": 2, "name": " ", "runtime": 0, "vote_average": 9.0, "vote_count": 0 },
                    { "name": "no number" }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/tv/1396/season/9"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;
        let provider = TmdbProvider::with_base_url("test-key", server.uri());
        let episodes = provider.season_episodes("1396", 1, "pl").await.unwrap();
        assert_eq!(
            episodes,
            vec![
                ProviderEpisode {
                    season: 1,
                    episode: 1,
                    title: Some("Najwyższy czas".into()),
                    overview: Some("Walter...".into()),
                    runtime_minutes: Some(58),
                    rating: Some(8.3),
                    still_url: Some("https://image.tmdb.org/t/p/w780/e1.jpg".into()),
                    air_date: Some("2008-01-20".into()),
                },
                ProviderEpisode { season: 1, episode: 2, ..ProviderEpisode::default() },
            ]
        );
        assert!(provider.season_episodes("1396", 9, "pl").await.unwrap().is_empty());
        assert!(provider.season_episodes("x", 1, "pl").await.unwrap().is_empty());
    }

    #[test]
    fn tmdb_dates_must_look_like_dates() {
        assert_eq!(tmdb_date(Some("1942-07-13".into())).as_deref(), Some("1942-07-13"));
        assert_eq!(tmdb_date(Some("".into())), None);
        assert_eq!(tmdb_date(Some("1942".into())), None);
        assert_eq!(tmdb_date(None), None);
    }

    #[tokio::test]
    async fn person_reads_the_profile_and_falls_back_to_an_english_biography() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/person/3"))
            .and(query_param("language", "pl"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": 3, "name": "Harrison Ford", "biography": "",
                "birthday": "1942-07-13", "deathday": null,
                "place_of_birth": "Chicago, Illinois, USA", "imdb_id": "nm0000148",
                "profile_path": "/hf.jpg"
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/person/3"))
            .and(query_param("language", "en-US"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "id": 3, "name": "Harrison Ford", "biography": " An American actor. "
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/person/404"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let provider = TmdbProvider::with_base_url("test-key", server.uri());
        let person = provider.person("3", "pl").await.unwrap().unwrap();
        assert_eq!(
            person,
            ProviderPersonDetails {
                name: "Harrison Ford".to_string(),
                biography: "An American actor.".to_string(),
                birth_date: Some("1942-07-13".to_string()),
                death_date: None,
                birth_place: Some("Chicago, Illinois, USA".to_string()),
                imdb_id: Some("nm0000148".to_string()),
                profile_url: Some("https://image.tmdb.org/t/p/h632/hf.jpg".to_string()),
            }
        );
        assert_eq!(provider.person("404", "pl").await.unwrap(), None);
        assert_eq!(provider.person("not-a-number", "pl").await.unwrap(), None);
    }

    async fn mount_movie_search(server: &MockServer) {
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{
                    "id": 11, "title": "Star Wars", "release_date": "1977-05-25",
                    "overview": "A long time ago...", "poster_path": "/p.jpg", "backdrop_path": "/b.jpg"
                }]
            })))
            .mount(server)
            .await;
        Mock::given(method("GET"))
            .and(path("/movie/11/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [], "backdrops": []
            })))
            .mount(server)
            .await;
    }

    #[tokio::test]
    async fn find_movie_adds_collection_and_age_rating_fields() {
        let server = MockServer::start().await;
        mount_movie_search(&server).await;
        Mock::given(method("GET"))
            .and(path("/movie/11"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "belongs_to_collection": {
                    "id": 10, "name": "Star Wars Collection",
                    "poster_path": "/sw.jpg", "backdrop_path": null
                },
                "release_dates": { "results": [
                    { "iso_3166_1": "US", "release_dates": [{ "certification": "PG", "type": 3 }] }
                ]},
                "original_title": "2 Fast 2 Furious",
                "tagline": "Jak szybko lubisz?",
                "runtime": 108,
                "vote_average": 6.53, "vote_count": 7000,
                "genres": [{ "name": "Akcja" }, { "name": "Kryminał" }],
                "production_companies": [{ "name": "Universal Pictures" }],
                "imdb_id": "tt0322259",
                "keywords": { "keywords": [{ "name": "car race" }, { "name": " " }] },
                "credits": {
                    "cast": [{ "id": 2, "name": "Mark Hamill", "character": "Luke Skywalker", "order": 0 }],
                    "crew": [{ "id": 1, "name": "George Lucas", "job": "Director" }]
                }
            })))
            .mount(&server)
            .await;

        let provider = TmdbProvider::with_base_url("test-key", server.uri());
        let found = provider
            .find_movie(&MovieQuery { title: "Star Wars", year: Some(1977) }, "en")
            .await
            .unwrap()
            .unwrap();

        assert_eq!(found.fields["tmdb_id"], 11);
        assert_eq!(
            found.fields["collection"],
            serde_json::json!({
                "id": "10",
                "name": "Star Wars Collection",
                "poster_url": "https://image.tmdb.org/t/p/w500/sw.jpg",
                "backdrop_url": null
            })
        );
        assert_eq!(found.fields["age_rating"], 10);
        assert_eq!(found.fields["certification"], "PG");
        assert_eq!(found.fields["certification_country"], "US");
        assert_eq!(found.fields["original_title"], "2 Fast 2 Furious");
        assert_eq!(found.fields["tagline"], "Jak szybko lubisz?");
        assert_eq!(found.fields["runtime_minutes"], 108);
        assert_eq!(found.fields["rating"], 6.5);
        assert_eq!(found.fields["genres"], serde_json::json!(["Akcja", "Kryminał"]));
        assert_eq!(found.fields["studios"], serde_json::json!(["Universal Pictures"]));
        assert_eq!(found.fields["tags"], serde_json::json!(["car race"]));
        assert_eq!(found.fields["imdb_id"], "tt0322259");
        assert!(found.fields.get("tvdb_id").is_none());
        let people = found.people.expect("credits were appended");
        assert_eq!(
            people.iter().map(|p| (p.name.as_str(), p.role)).collect::<Vec<_>>(),
            [("Mark Hamill", PersonRole::Actor), ("George Lucas", PersonRole::Director)]
        );
        // The search result's own images are the fallback.
        assert_eq!(
            found.poster_url.as_deref(),
            Some("https://image.tmdb.org/t/p/w500/p.jpg")
        );
    }

    #[tokio::test]
    async fn find_movie_without_a_collection_says_null_and_a_failed_lookup_says_nothing() {
        let server = MockServer::start().await;
        mount_movie_search(&server).await;
        Mock::given(method("GET"))
            .and(path("/movie/11"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "belongs_to_collection": null,
                "release_dates": { "results": [] }
            })))
            .mount(&server)
            .await;
        let provider = TmdbProvider::with_base_url("test-key", server.uri());
        let query = MovieQuery { title: "Star Wars", year: Some(1977) };
        let found = provider.find_movie(&query, "en").await.unwrap().unwrap();
        assert!(found.fields["collection"].is_null());
        assert!(found.fields.get("collection").is_some());
        assert!(found.fields.get("age_rating").is_none());
        assert_eq!(found.people, None, "no credits in the answer: keep the cast");

        let failing = MockServer::start().await;
        mount_movie_search(&failing).await;
        Mock::given(method("GET"))
            .and(path("/movie/11"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&failing)
            .await;
        let provider = TmdbProvider::with_base_url("test-key", failing.uri());
        let found = provider.find_movie(&query, "en").await.unwrap().unwrap();
        assert!(found.fields.get("collection").is_none());
        assert_eq!(found.people, None);
        assert_eq!(found.fields["overview"], "A long time ago...");
    }

    #[tokio::test]
    async fn search_lists_candidates_for_the_exact_query() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/movie"))
            .and(query_param("query", "Star Wars"))
            .and(query_param("year", "1977"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [
                    { "id": 11, "title": "Star Wars", "release_date": "1977-05-25",
                      "overview": "A long time ago...", "poster_path": "/sw.jpg" },
                    { "id": 12, "title": "Star Wars: Special", "release_date": "" }
                ]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/search/tv"))
            .and(query_param("first_air_date_year", "2008"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{ "id": 34, "name": "Inazuma Eleven", "first_air_date": "2008-10-05" }]
            })))
            .mount(&server)
            .await;

        let provider = TmdbProvider::with_base_url("k", server.uri());
        let movies = provider.search(MediaKind::Movie, "Star Wars", Some(1977), "en").await.unwrap();
        assert_eq!(movies.len(), 2);
        assert_eq!(
            movies[0],
            ProviderCandidate {
                provider_id: "11".to_string(),
                title: "Star Wars".to_string(),
                year: Some(1977),
                overview: "A long time ago...".to_string(),
                poster_url: Some("https://image.tmdb.org/t/p/w185/sw.jpg".to_string()),
            }
        );
        assert_eq!(movies[1].year, None);
        let series = provider.search(MediaKind::Series, "Inazuma", Some(2008), "pl").await.unwrap();
        assert_eq!((series[0].provider_id.as_str(), series[0].year), ("34", Some(2008)));
    }

    #[tokio::test]
    async fn lookup_returns_the_full_match_for_an_id_and_none_for_an_unknown_one() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/movie/11"))
            .and(query_param("append_to_response", "release_dates,credits,keywords"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "title": "Gwiezdne wojny", "release_date": "1977-05-25", "overview": "Dawno temu...",
                "poster_path": "/p.jpg", "backdrop_path": "/b.jpg",
                "belongs_to_collection": { "id": 10, "name": "Kolekcja", "poster_path": null, "backdrop_path": null },
                "release_dates": { "results": [
                    { "iso_3166_1": "US", "release_dates": [{ "certification": "PG", "type": 3 }] }
                ]}
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/movie/11/images"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/tv/34"))
            .and(query_param("append_to_response", "content_ratings,keywords,external_ids"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "name": "Inazuma Eleven", "first_air_date": "2008-10-05", "overview": "Piłka",
                "poster_path": "/ie.jpg",
                "content_ratings": { "results": [{ "iso_3166_1": "DE", "rating": "6" }] }
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/tv/34/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "posters": [{ "file_path": "/ie-pl.jpg", "iso_639_1": "pl", "vote_average": 1.0 }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/movie/999"))
            .respond_with(ResponseTemplate::new(404))
            .mount(&server)
            .await;

        let provider = TmdbProvider::with_base_url("k", server.uri());
        let movie = provider.lookup(MediaKind::Movie, "11", "pl").await.unwrap().unwrap();
        assert_eq!(movie.fields["title"], "Gwiezdne wojny");
        assert_eq!(movie.fields["year"], 1977);
        assert_eq!(movie.fields["overview"], "Dawno temu...");
        assert_eq!(movie.fields["collection"]["id"], "10");
        assert_eq!(movie.fields["age_rating"], 10);
        // The images request failed: the details' own images stand in.
        assert_eq!(movie.poster_url.as_deref(), Some("https://image.tmdb.org/t/p/w500/p.jpg"));
        assert_eq!(movie.backdrop_url.as_deref(), Some("https://image.tmdb.org/t/p/w1280/b.jpg"));

        let series = provider.lookup(MediaKind::Series, "34", "pl").await.unwrap().unwrap();
        assert_eq!((series.fields["title"].as_str(), series.fields["year"].as_i64()), (Some("Inazuma Eleven"), Some(2008)));
        assert_eq!(series.fields["age_rating"], 6);
        assert_eq!(series.poster_url.as_deref(), Some("https://image.tmdb.org/t/p/w500/ie-pl.jpg"));

        assert!(provider.lookup(MediaKind::Movie, "999", "pl").await.unwrap().is_none());
        assert!(provider.lookup(MediaKind::Movie, "not-a-number", "pl").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn find_series_adds_the_content_rating() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/search/tv"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "results": [{ "id": 1396, "name": "Breaking Bad", "overview": "x" }]
            })))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/tv/1396/images"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/tv/1396"))
            .and(query_param("append_to_response", "content_ratings,keywords,external_ids"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "name": "Breaking Bad",
                "original_name": "Breaking Bad",
                "tagline": "Remember my name",
                "episode_run_time": [0, 47],
                "vote_average": 8.912, "vote_count": 15000,
                "genres": [{ "name": "Dramat" }, { "name": "Kryminał" }],
                "production_companies": [{ "name": "Sony Pictures Television" }],
                "keywords": { "results": [{ "name": "drug dealer" }, { "name": "drug dealer" }, { "name": "new mexico" }] },
                "external_ids": { "imdb_id": "tt0903747", "tvdb_id": 81189 },
                "seasons": [
                    { "season_number": 1, "name": "Sezon 1", "overview": "", "poster_path": "/s1.jpg", "episode_count": 7, "air_date": "2008-01-20" },
                    { "season_number": 0, "name": "Odcinki specjalne", "poster_path": null, "episode_count": 0 }
                ],
                "content_ratings": { "results": [
                    { "iso_3166_1": "DE", "rating": "16" },
                    { "iso_3166_1": "US", "rating": "TV-MA" }
                ]}
            })))
            .mount(&server)
            .await;

        let provider = TmdbProvider::with_base_url("test-key", server.uri());
        let found = provider
            .find_series(&SeriesQuery { title: "Breaking Bad" }, "pl")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(found.fields["age_rating"], 17);
        assert_eq!(found.fields["certification"], "TV-MA");
        assert!(found.fields.get("collection").is_none());
        assert_eq!(found.fields["tagline"], "Remember my name");
        assert_eq!(found.fields["runtime_minutes"], 47);
        assert_eq!(found.fields["rating"], 8.9);
        assert_eq!(found.fields["genres"], serde_json::json!(["Dramat", "Kryminał"]));
        assert_eq!(found.fields["tags"], serde_json::json!(["drug dealer", "new mexico"]));
        assert_eq!(found.fields["studios"], serde_json::json!(["Sony Pictures Television"]));
        assert_eq!(found.fields["imdb_id"], "tt0903747");
        assert_eq!(found.fields["tvdb_id"], 81189);
        assert_eq!(
            found.seasons,
            vec![
                ProviderSeason {
                    number: 0,
                    name: Some("Odcinki specjalne".into()),
                    overview: None,
                    poster_url: None,
                    episode_count: None,
                    air_year: None,
                },
                ProviderSeason {
                    number: 1,
                    name: Some("Sezon 1".into()),
                    overview: None,
                    poster_url: Some("https://image.tmdb.org/t/p/w500/s1.jpg".into()),
                    episode_count: Some(7),
                    air_year: Some(2008),
                },
            ]
        );
    }
}
