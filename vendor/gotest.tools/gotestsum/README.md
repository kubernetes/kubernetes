# gotestsum

`gotestsum` runs tests, prints friendly test output and a summary of the test run.  Requires Go 1.10+.

## Install

Download a binary from [releases](https://github.com/gotestyourself/gotestsum/releases), or get the
source with `go get gotest.tools/gotestsum` (you may need to run `dep ensure`).

## Demo

![Demo](https://raw.githubusercontent.com/gotestyourself/gotestsum/master/docs/demo.gif)

## Docs

[![GoDoc](https://godoc.org/gotest.tools/gotestsum?status.svg)](https://godoc.org/gotest.tools/gotestsum)
[![CircleCI](https://circleci.com/gh/gotestyourself/gotestsum/tree/master.svg?style=shield)](https://circleci.com/gh/gotestyourself/gotestsum/tree/master)
[![Go Reportcard](https://goreportcard.com/badge/gotest.tools/gotestsum)](https://goreportcard.com/report/gotest.tools/gotestsum)

`gotestsum` works by running `go test --json ./...` and reading the JSON
output.

### TOC

- [Format](#format)
- [Summary](#summary)
- [JUnit XML](#junit-xml)
- [JSON file](#json-file-output)
- [Setting go test flags and using custom commands](#custom-go-test-command)

### Format

Set a format with the `--format` flag or the `GOTESTSUM_FORMAT` environment
variable.
```
gotestsum --format short-verbose
```

Supported formats:
 * `dots` - output one character per test.
 * `short` (default) - output a line for each test package.
 * `standard-quiet` - the default `go test` format.
 * `short-verbose` - output a line for each test and package.
 * `standard-verbose` - the standard `go test -v` format.

Have a suggestion for some other format? Please open an issue!

### Summary

A summary of the test run is printed after the test output.

```
DONE 101 tests[, 3 skipped][, 2 failures][, 1 error] in 0.103s
```

The summary includes:
 * A count of: tests run, tests skipped, tests failed, and package build errors.
 * Elapsed time including time to build.
 * Test output of all failed and skipped tests, and any package build errors.

To disable parts of the summary use `--no-summary section`.

Example: hide skipped tests in the summary
```
gotestsum --no-summary=skipped
```

Example: hide failed and skipped
```
gotestsum --no-summary=skipped,failed
```

Example: hide output in the summary, only print names of failed and skipped tests
and errors
```
gotestsum --no-summary=output
```

### JUnit XML

When the `--junitfile` flag or `GOTESTSUM_JUNITFILE` environment variable are set
to a file path `gotestsum` will write a test report, in JUnit XML format, to the file.
This file can be used to integrate with CI systems.

```
gotestsum --junitfile unit-tests.xml
```

### JSON file output

When the `--jsonfile` flag or `GOTESTSUM_JSONFILE` environment variable are set
to a file path `gotestsum` will write a line-delimited JSON file with all the
[test2json](https://golang.org/cmd/test2json/#hdr-Output_Format)
output that was written by `go test --json`. This file can be used to compare test
runs, or find flaky tests.

```
gotestsum --jsonfile test-output.log
```

### Custom `go test` command

By default `gotestsum` runs tests using the command `go test --json ./...`. You
can change the command with positional arguments after a `--`. You can change just the
test directory value (which defaults to `./...`) by setting the `TEST_DIRECTORY`
environment variable.

You can use `--debug` to echo the command before it is run.

Example: set build tags
```
gotestsum -- -tags=integration ./...
```

Example: run tests in a single package
```
gotestsum -- ./io/http
```

Example: enable coverage
```
gotestsum -- -coverprofile=cover.out ./...
```

Example: run a script instead of `go test`
```
gotestsum --raw-command -- ./scripts/run_tests.sh
```

Note: when using `--raw-command` you must ensure that the stdout produced by
the script only contains the `test2json` output. Any stderr produced by the script
will be considered an error (this behaviour is necessary because package build errors
are only reported by writting to stderr, not the `test2json` stdout). Any stderr
produced by tests is not considered an error (it will be in the `test2json` stdout).

Example: using `TEST_DIRECTORY`
```
TEST_DIRECTORY=./io/http gotestsum
```

### Run tests when a file is modified

[filewatcher](https://github.com/dnephin/filewatcher) will automatically set the
`TEST_DIRECTORY` environment variable which makes it easy to integrate
`gotestsum`.

Example: run tests for a package when any file in that package is saved
```
filewatcher gotestsum
```

## Thanks

This package is heavily influenced by the [pytest](https://docs.pytest.org) test runner for `python`.
