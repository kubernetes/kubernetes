# Swag [![Build Status](https://github.com/go-openapi/swag/actions/workflows/go-test.yml/badge.svg)](https://github.com/go-openapi/swag/actions?query=workflow%3A"go+test") [![codecov](https://codecov.io/gh/go-openapi/swag/branch/master/graph/badge.svg)](https://codecov.io/gh/go-openapi/swag)

[![Slack Status](https://slackin.goswagger.io/badge.svg)](https://slackin.goswagger.io)
[![license](http://img.shields.io/badge/license-Apache%20v2-orange.svg)](https://raw.githubusercontent.com/go-openapi/swag/master/LICENSE)
[![Go Reference](https://pkg.go.dev/badge/github.com/go-openapi/swag.svg)](https://pkg.go.dev/github.com/go-openapi/swag)
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

* YAML utilities depend on `gopkg.in/yaml.v3`
* `github.com/mailru/easyjson v0.7.7`
