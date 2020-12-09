# Changelog

## [1.2.0] - 2019-08-03

### Added

- Add `Must` helper to raise an error as panic. It can be used with `Load` and `OverLoad`.
- Add more tests to be 100% coverage.
- Add CHANGELOG
- Add more OS for the test: OSX and Windows

### Changed

- Reduce complexity and improve source code for having `A+` score in [goreportcard](https://goreportcard.com/report/github.com/subosito/gotenv).
- Updated README with mentions to all available functions

### Removed

- Remove `ErrFormat`
- Remove `MustLoad` and `MustOverload`, replaced with `Must` helper.

## [1.1.1] - 2018-06-05

### Changed

- Replace `os.Getenv` with `os.LookupEnv` to ensure that the environment variable is not set, by [radding](https://github.com/radding)

## [1.1.0] - 2017-03-20

### Added

- Supports carriage return in env
- Handle files with UTF-8 BOM 

### Changed

- Whitespace handling

### Fixed

- Incorrect variable expansion
- Handling escaped '$' characters

## [1.0.0] - 2014-10-05

First stable release.

