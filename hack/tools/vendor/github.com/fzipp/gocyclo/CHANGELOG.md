# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2021-12-19
### Added
- Support method receivers with type parameters introduced in Go 1.18

### Changed
- Use more efficient filepath.WalkDir instead of filepath.Walk

## [0.3.1] - 2020-10-20
### Added
- Test coverage

### Fixed
- Fix cyclomatic complexity for function literals (base complexity of 1 was missing)

## [0.3.0] - 2020-10-17
### Added
- New `-avg-short` and `-total-short` options for printing average and total cyclomatic complexities without label
- Export the `AnalyzeASTFile` function in package API
- Doc comments for exported functions and types

### Fixed
- Ignore `default` cases

## [0.2.0] - 2020-10-17
### Added
- Support for gocyclo as a package
- Support for ignoring of individual functions via a new `gocyclo:ignore` directive
- New `-total` option to compute total cyclomatic complexity
- New `-ignore` option to ignore files matching a regular expression
- Analysis of function literals at declaration level

### Changed
- Breaking: installation changed to `go get github.com/fzipp/gocyclo/cmd/gocyclo`

## [0.1.0] - 2020-10-17

### Added
- `go.mod` file; beginning of versioning

