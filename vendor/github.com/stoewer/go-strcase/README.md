[![CircleCI](https://circleci.com/gh/stoewer/go-strcase/tree/master.svg?style=svg)](https://circleci.com/gh/stoewer/go-strcase/tree/master)
[![codecov](https://codecov.io/gh/stoewer/go-strcase/branch/master/graph/badge.svg)](https://codecov.io/gh/stoewer/go-strcase)
[![GoDoc](https://godoc.org/github.com/stoewer/go-strcase?status.svg)](https://pkg.go.dev/github.com/stoewer/go-strcase)
---

Go strcase
==========

The package `strcase` converts between different kinds of naming formats such as camel case 
(`CamelCase`), snake case (`snake_case`) or kebab case (`kebab-case`).
The package is designed to work only with strings consisting of standard ASCII letters. 
Unicode is currently not supported.

Versioning and stability
------------------------

Although the master branch is supposed to remain always backward compatible, the repository
contains version tags in order to support vendoring tools.
The tag names follow semantic versioning conventions and have the following format `v1.0.0`.
This package supports Go modules introduced with version 1.11.

Example
-------

```go
import "github.com/stoewer/go-strcase"

var snake = strcase.SnakeCase("CamelCase")
```

Dependencies
------------

### Build dependencies

* none

### Test dependencies

* `github.com/stretchr/testify`

Run linters and unit tests
-------------------------- 

To run the static code analysis, linters and tests use the following commands:

```
golangci-lint run --config .golangci.yml ./...
go test ./...
```
