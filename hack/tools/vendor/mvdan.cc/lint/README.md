# lint

[![GoDoc](https://godoc.org/mvdan.cc/lint?status.svg)](https://godoc.org/mvdan.cc/lint)
[![Build Status](https://travis-ci.org/mvdan/lint.svg?branch=master)](https://travis-ci.org/mvdan/lint)

Work in progress. Its API might change before the 1.0 release.

This package intends to define simple interfaces that Go code checkers
can implement. This would simplify calling them from Go code, as well as
running multiple linters while sharing initial loading work.

### metalint

	go get -u mvdan.cc/lint/cmd/metalint

The start of a linter that runs many linters leveraging the common
interface. Not stable yet.

Linters included:

* [unparam](https://mvdan.cc/unparam)
* [interfacer](https://github.com/mvdan/interfacer)

### Related projects

* [golinters](https://github.com/thomasheller/golinters) - Report on
  linter support
