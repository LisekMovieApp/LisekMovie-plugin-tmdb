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
use lisek_plugin_api::{MetadataProvider, MovieQuery, ProviderError, ProviderMatch, SeriesQuery};
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
    overview: String,
    poster_path: Option<String>,
    backdrop_path: Option<String>,
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

/// The [`lisek_plugin_api::MetadataProvider`] implementation `lisek-
/// server`'s provider registry constructs one of per scan, from the API
/// key `provider_config` has stored for `"tmdb"` — see that crate's own
/// doc comment for the compile-time-extension architecture this fits into.
pub struct TmdbProvider {
    api_key: String,
}

impl TmdbProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
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
        let client = TmdbClient::new(self.api_key.clone());
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
        Ok(Some(ProviderMatch {
            poster_url: poster_path.as_deref().map(|p| image_url(p, "w500")),
            backdrop_url: backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            logo_url: logo_path.as_deref().map(|p| image_url(p, "w500")),
            banner_url: None,
            fields: serde_json::json!({
                "tmdb_id": m.tmdb_id,
                "overview": m.overview,
            }),
        }))
    }

    async fn find_series(
        &self,
        query: &SeriesQuery<'_>,
        primary_language: &str,
    ) -> Result<Option<ProviderMatch>, ProviderError> {
        let client = TmdbClient::new(self.api_key.clone());
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
        Ok(Some(ProviderMatch {
            poster_url: poster_path.as_deref().map(|p| image_url(p, "w500")),
            backdrop_url: backdrop_path.as_deref().map(|p| image_url(p, "w1280")),
            logo_url: logo_path.as_deref().map(|p| image_url(p, "w500")),
            banner_url: None,
            fields: serde_json::json!({
                "tmdb_id": m.tmdb_id,
                "overview": m.overview,
            }),
        }))
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
}
