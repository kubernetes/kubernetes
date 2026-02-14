# jsonpointer

<!-- Badges: status  -->
[![Tests][test-badge]][test-url] [![Coverage][cov-badge]][cov-url] [![CI vuln scan][vuln-scan-badge]][vuln-scan-url] [![CodeQL][codeql-badge]][codeql-url]
<!-- Badges: release & docker images  -->
<!-- Badges: code quality  -->
<!-- Badges: license & compliance -->
[![Release][release-badge]][release-url] [![Go Report Card][gocard-badge]][gocard-url] [![CodeFactor Grade][codefactor-badge]][codefactor-url] [![License][license-badge]][license-url]
<!-- Badges: documentation & support -->
<!-- Badges: others & stats -->
[![GoDoc][godoc-badge]][godoc-url] [![Slack Channel][slack-logo]![slack-badge]][slack-url] [![go version][goversion-badge]][goversion-url] ![Top language][top-badge] ![Commits since latest release][commits-badge]

---

An implementation of JSON Pointer for golang, which supports go `struct`.

## Status

API is stable.

## Import this library in your project

```cmd
go get github.com/go-openapi/jsonpointer
```

## Basic usage

See also some [examples](./examples_test.go)

### Retrieving a value

```go
  import (
    "github.com/go-openapi/jsonpointer"
  )


  var doc any

  ...

	pointer, err := jsonpointer.New("/foo/1")
	if err != nil {
		... // error: e.g. invalid JSON pointer specification
	}

	value, kind, err := pointer.Get(doc)
	if err != nil {
		... // error: e.g. key not found, index out of bounds, etc.
	}

  ...
```

### Setting a value

```go
  ...
  var doc any
  ...
	pointer, err := jsonpointer.New("/foo/1")
	if err != nil {
		... // error: e.g. invalid JSON pointer specification
  }

	doc, err = p.Set(doc, "value")
	if err != nil {
		... // error: e.g. key not found, index out of bounds, etc.
	}
```

## Change log

See <https://github.com/go-openapi/jsonpointer/releases>

## References

<https://tools.ietf.org/html/draft-ietf-appsawg-json-pointer-07>

also known as [RFC6901](https://www.rfc-editor.org/rfc/rfc6901)

## Licensing

This library ships under the [SPDX-License-Identifier: Apache-2.0](./LICENSE).

See the license [NOTICE](./NOTICE), which recalls the licensing terms of all the pieces of software
on top of which it has been built.

## Limitations

The 4.Evaluation part of the previous reference, starting with 'If the currently referenced value is a JSON array,
the reference token MUST contain either...' is not implemented.

That is because our implementation of the JSON pointer only supports explicit references to array elements:
the provision in the spec to resolve non-existent members as "the last element in the array",
using the special trailing character "-" is not implemented.

## Other documentation

* [All-time contributors](./CONTRIBUTORS.md)
* [Contributing guidelines](.github/CONTRIBUTING.md)
* [Maintainers documentation](docs/MAINTAINERS.md)
* [Code style](docs/STYLE.md)

## Cutting a new release

Maintainers can cut a new release by either:

* running [this workflow](https://github.com/go-openapi/jsonpointer/actions/workflows/bump-release.yml)
* or pushing a semver tag
  * signed tags are preferred
  * The tag message is prepended to release notes

<!-- Badges: status  -->
[test-badge]: https://github.com/go-openapi/jsonpointer/actions/workflows/go-test.yml/badge.svg
[test-url]: https://github.com/go-openapi/jsonpointer/actions/workflows/go-test.yml
[cov-badge]: https://codecov.io/gh/go-openapi/jsonpointer/branch/master/graph/badge.svg
[cov-url]: https://codecov.io/gh/go-openapi/jsonpointer
[vuln-scan-badge]: https://github.com/go-openapi/jsonpointer/actions/workflows/scanner.yml/badge.svg
[vuln-scan-url]: https://github.com/go-openapi/jsonpointer/actions/workflows/scanner.yml
[codeql-badge]: https://github.com/go-openapi/jsonpointer/actions/workflows/codeql.yml/badge.svg
[codeql-url]: https://github.com/go-openapi/jsonpointer/actions/workflows/codeql.yml
<!-- Badges: release & docker images  -->
[release-badge]: https://badge.fury.io/gh/go-openapi%2Fjsonpointer.svg
[release-url]: https://badge.fury.io/gh/go-openapi%2Fjsonpointer
[gomod-badge]: https://badge.fury.io/go/github.com%2Fgo-openapi%2Fjsonpointer.svg
[gomod-url]: https://badge.fury.io/go/github.com%2Fgo-openapi%2Fjsonpointer
<!-- Badges: code quality  -->
[gocard-badge]: https://goreportcard.com/badge/github.com/go-openapi/jsonpointer
[gocard-url]: https://goreportcard.com/report/github.com/go-openapi/jsonpointer
[codefactor-badge]: https://img.shields.io/codefactor/grade/github/go-openapi/jsonpointer
[codefactor-url]: https://www.codefactor.io/repository/github/go-openapi/jsonpointer
<!-- Badges: documentation & support -->
[doc-badge]: https://img.shields.io/badge/doc-site-blue?link=https%3A%2F%2Fgoswagger.io%2Fgo-openapi%2F
[doc-url]: https://goswagger.io/go-openapi
[godoc-badge]: https://pkg.go.dev/badge/github.com/go-openapi/jsonpointer
[godoc-url]: http://pkg.go.dev/github.com/go-openapi/jsonpointer
[slack-logo]: https://a.slack-edge.com/e6a93c1/img/icons/favicon-32.png
[slack-badge]: https://img.shields.io/badge/slack-blue?link=https%3A%2F%2Fgoswagger.slack.com%2Farchives%2FC04R30YM
[slack-url]: https://goswagger.slack.com/archives/C04R30YMU
<!-- Badges: license & compliance -->
[license-badge]: http://img.shields.io/badge/license-Apache%20v2-orange.svg
[license-url]: https://github.com/go-openapi/jsonpointer/?tab=Apache-2.0-1-ov-file#readme
<!-- Badges: others & stats -->
[goversion-badge]: https://img.shields.io/github/go-mod/go-version/go-openapi/jsonpointer
[goversion-url]: https://github.com/go-openapi/jsonpointer/blob/master/go.mod
[top-badge]: https://img.shields.io/github/languages/top/go-openapi/jsonpointer
[commits-badge]: https://img.shields.io/github/commits-since/go-openapi/jsonpointer/latest
