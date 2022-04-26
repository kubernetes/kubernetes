# errchkjson

[![Test Status](https://github.com/breml/errchkjson/actions/workflows/ci.yml/badge.svg)](https://github.com/breml/errchkjson/actions/workflows/ci.yml) [![Go Report Card](https://goreportcard.com/badge/github.com/breml/errchkjson)](https://goreportcard.com/report/github.com/breml/errchkjson) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Checks types passed to the json encoding functions. Reports unsupported types and reports occations, where the check for the returned error can be omited.

Consider this [http.Handler](https://pkg.go.dev/net/http#Handler):

```Go
func JSONHelloWorld(w http.ResponseWriter, r *http.Request) {
	response := struct {
		Message string
		Code    int
	}{
		Message: "Hello World",
		Code:    200,
	}

	body, err := json.Marshal(response)
	if err != nil {
		panic(err) // unreachable, because json encoding of a struct with just a string and an int will never return an error.
	}

	w.Write(body)
}
```

Because the `panic` is not possible to happen, one might refactor the code like this:

```Go
func JSONHelloWorld(w http.ResponseWriter, r *http.Request) {
	response := struct {
		Message string
		Code    int
	}{
		Message: "Hello World",
		Code:    200,
	}

	body, _ := json.Marshal(response)

	w.Write(body)
}
```

This is ok, as long as the struct is not altered in such a way, that could potentially lead
to `json.Marshal` returning an error.

`errchkjson` allows you to lint your code such that the above error returned from `json.Marshal`
can be omitted while still staying safe, because as soon as an unsafe type is added to the
response type, the linter will warn you.

## Installation

Download `errchkjson` from the [releases](https://github.com/breml/errchkjson/releases) or get the latest version from source with:

```shell
go get github.com/breml/errchkjson/cmd/errchkjson
```

## Usage

### Shell

Check everything:

```shell
errchkjson ./...
```

`errchkjson` also recognizes the following command-line options:

The `-omit-safe` flag disables checking for safe returns of errors from json.Marshal

## Types

### Safe

The following types are safe to use with [json encoding functions](https://pkg.go.dev/encoding/json), that is, the encoding to JSON can not fail:

Safe basic types:

* `bool`
* `int`, `int8`, `int16`, `int32`, `int64`, `uint`, `uint8`, `uint16`, `uint32`, `uint64`, `uintptr`
* `string`
* Pointer type of the above listed basic types

Composed types (struct, map, slice, array) are safe, if the type of the value is
safe. For structs, only exported fields are relevant. For maps, the key needs to be either an integer type or a string.

### Unsafe

The following types are unsafe to use with [json encoding functions](https://pkg.go.dev/encoding/json), that is, the encoding to JSON can fail (return an error):

Unsafe basic types:

* `float32`, `float64`
* `interface{}`
* Pointer type of the above listed basic types

Any composed types (struct, map, slice, array) containing an unsafe basic type.

If a type implements the `json.Marshaler` or `encoding.TextMarshaler` interface (e.g. `json.Number`).

### Forbidden

Forbidden basic types:

* `complex64`, `complex128`
* `chan`
* `func`
* `unsafe.Pointer`

Any composed types (struct, map, slice, array) containing a forbidden basic type. Any map
using a key with a forbidden type (`bool`, `float32`, `float64`, `struct`).

## Bugs found during development

During the development of `errcheckjson`, the following issues in package `encoding/json` of the Go standard library have been found and PR have been merged:

* [Issue #34154: encoding/json: string option (struct tag) on string field with SetEscapeHTML(false) escapes anyway](https://github.com/golang/go/issues/34154)
* [PR #34127: encoding/json: fix and optimize marshal for quoted string](https://github.com/golang/go/pull/34127)
* [Issue #34268: encoding/json: wrong encoding for json.Number field with string option (struct tag)](https://github.com/golang/go/issues/34268)
* [PR #34269: encoding/json: make Number with the ,string option marshal with quotes](https://github.com/golang/go/pull/34269)
* [PR #34272: encoding/json: validate strings when decoding into Number](https://github.com/golang/go/pull/34272)
