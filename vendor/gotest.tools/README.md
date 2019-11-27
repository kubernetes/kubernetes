# gotest.tools

A collection of packages to augment `testing` and support common patterns.

[![GoDoc](https://godoc.org/gotest.tools?status.svg)](https://godoc.org/gotest.tools)
[![CircleCI](https://circleci.com/gh/gotestyourself/gotest.tools/tree/master.svg?style=shield)](https://circleci.com/gh/gotestyourself/gotest.tools/tree/master)
[![Go Reportcard](https://goreportcard.com/badge/gotest.tools)](https://goreportcard.com/report/gotest.tools)


## Packages

* [assert](http://godoc.org/gotest.tools/assert) -
  compare values and fail the test when a comparison fails
* [env](http://godoc.org/gotest.tools/env) -
  test code which uses environment variables
* [fs](http://godoc.org/gotest.tools/fs) -
  create temporary files and compare a filesystem tree to an expected value
* [golden](http://godoc.org/gotest.tools/golden) -
  compare large multi-line strings against values frozen in golden files
* [icmd](http://godoc.org/gotest.tools/icmd) -
  execute binaries and test the output
* [poll](http://godoc.org/gotest.tools/poll) -
  test asynchronous code by polling until a desired state is reached
* [skip](http://godoc.org/gotest.tools/skip) -
  skip a test and print the source code of the condition used to skip the test

## Related

* [gotest.tools/gotestsum](https://github.com/gotestyourself/gotestsum) - go test runner with custom output
* [maxbrunsfeld/counterfeiter](https://github.com/maxbrunsfeld/counterfeiter) - generate fakes for interfaces
* [jonboulle/clockwork](https://github.com/jonboulle/clockwork) - a fake clock for testing code that uses `time`
