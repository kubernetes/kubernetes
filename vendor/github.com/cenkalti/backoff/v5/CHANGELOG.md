# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2024-12-19

### Added

- RetryAfterError can be returned from an operation to indicate how long to wait before the next retry.

### Changed

- Retry function now accepts additional options for specifying max number of tries and max elapsed time.
- Retry function now accepts a context.Context.
- Operation function signature changed to return result (any type) and error.

### Removed

- RetryNotify* and RetryWithData functions. Only single Retry function remains.
- Optional arguments from ExponentialBackoff constructor.
- Clock and Timer interfaces.

### Fixed

- The original error is returned from Retry if there's a PermanentError. (#144)
- The Retry function respects the wrapped PermanentError. (#140)
