# Swag [![Build Status](https://travis-ci.org/go-openapi/swag.svg?branch=master)](https://travis-ci.org/go-openapi/swag) [![codecov](https://codecov.io/gh/go-openapi/swag/branch/master/graph/badge.svg)](https://codecov.io/gh/go-openapi/swag) [![Slack Status](https://slackin.goswagger.io/badge.svg)](https://slackin.goswagger.io)

[![license](http://img.shields.io/badge/license-Apache%20v2-orange.svg)](https://raw.githubusercontent.com/go-openapi/swag/master/LICENSE)
[![GoDoc](https://godoc.org/github.com/go-openapi/swag?status.svg)](http://godoc.org/github.com/go-openapi/swag)
[![GolangCI](https://golangci.com/badges/github.com/go-openapi/swag.svg)](https://golangci.com)
[![Go Report Card](https://goreportcard.com/badge/github.com/go-openapi/swag)](https://goreportcard.com/report/github.com/go-openapi/swag)

Contains a bunch of helper functions for go-openapi and go-swagger projects.

You may also use it standalone for your projects.

* convert between value and pointers for builtin types
* convert from string to builtin types (wraps strconv)
* fast json concatenation
* search in path
* load from file or http
* name mangling


This repo has only few dependencies outside of the standard library:

* JSON utilities depend on github.com/mailru/easyjson
* YAML utilities depend on gopkg.in/yaml.v2
