# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [1.2.1]
### Changed
- Drop golang/x/lint dependency.

[1.2.1]: https://github.com/uber-go/goleak/compare/v1.2.0...v1.2.1

## [1.2.0]
### Added
- Add Cleanup option that can be used for registering cleanup callbacks. (#78)

### Changed
- Mark VerifyNone as a test helper. (#75)

Thanks to @tallclair for their contribution to this release.

[1.2.0]: https://github.com/uber-go/goleak/compare/v1.1.12...v1.2.0

## [1.1.12]
### Fixed
- Fixed logic for ignoring trace related goroutines on Go versions 1.16 and above.

[1.1.12]: https://github.com/uber-go/goleak/compare/v1.1.11...v1.1.12

## [1.1.11]
### Fixed
- Documentation fix on how to test.
- Update dependency on stretchr/testify to v1.7.0. (#59)
- Update dependency on golang.org/x/tools to address CVE-2020-14040. (#62)

[1.1.11]: https://github.com/uber-go/goleak/compare/v1.1.10...v1.1.11

## [1.1.10]
### Added
- [#49]: Add option to ignore current goroutines, which checks for any additional leaks and allows for incremental adoption of goleak in larger projects.

Thanks to @denis-tingajkin for their contributions to this release.

[#49]: https://github.com/uber-go/goleak/pull/49
[1.1.10]: https://github.com/uber-go/goleak/compare/v1.0.0...v1.1.10

## [1.0.0]
### Changed
- Migrate to Go modules.

### Fixed
- Ignore trace related goroutines that cause false positives with -trace.

[1.0.0]: https://github.com/uber-go/goleak/compare/v0.10.0...v1.0.0

## [0.10.0]
- Initial release.

[0.10.0]: https://github.com/uber-go/goleak/compare/v0.10.0...HEAD