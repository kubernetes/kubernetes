# go-junit-report

Converts `go test` output to an xml report, suitable for applications that
expect junit xml reports (e.g. [Jenkins](http://jenkins-ci.org)).

[![Build Status][travis-badge]][travis-link]
[![Report Card][report-badge]][report-link]

## Installation

Go version 1.1 or higher is required. Install or update using the `go get`
command:

```bash
go get -u github.com/jstemmer/go-junit-report
```

## Contribution

Create an Issue and discuss the fix or feature, then fork the package.
Clone to github.com/jstemmer/go-junit-report.  This is necessary because go import uses this path.
Fix or implement feature. Test and then commit change.
Specify #Issue and describe change in the commit message.
Create Pull Request. It can be merged by owner or administrator then.

## Run Tests
go test

## Usage

go-junit-report reads the `go test` verbose output from standard in and writes
junit compatible XML to standard out.

```bash
go test -v 2>&1 | go-junit-report > report.xml
```

Note that it also can parse benchmark output with `-bench` flag:
```bash
go test -v -bench . -count 5 2>&1 | go-junit-report > report.xml
 ```

[travis-badge]: https://travis-ci.org/jstemmer/go-junit-report.svg
[travis-link]: https://travis-ci.org/jstemmer/go-junit-report
[report-badge]: https://goreportcard.com/badge/github.com/jstemmer/go-junit-report
[report-link]: https://goreportcard.com/report/github.com/jstemmer/go-junit-report
