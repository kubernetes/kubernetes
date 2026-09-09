# jsonpointer

<!-- Badges: status  -->
[![Tests][test-badge]][test-url] [![Coverage][cov-badge]][cov-url] [![CI vuln scan][vuln-scan-badge]][vuln-scan-url] [![CodeQL][codeql-badge]][codeql-url]
<!-- Badges: release & docker images  -->
<!-- Badges: code quality  -->
<!-- Badges: license & compliance -->
[![Release][release-badge]][release-url] [![Go Report Card][gocard-badge]][gocard-url] [![CodeFactor Grade][codefactor-badge]][codefactor-url] [![License][license-badge]][license-url]
<!-- Badges: documentation & support -->
<!-- Badges: others & stats -->
[![GoDoc][godoc-badge]][godoc-url] [![Discord Channel][discord-badge]][discord-url] [![go version][goversion-badge]][goversion-url] ![Top language][top-badge] ![Commits since latest release][commits-badge]

---

An implementation of JSON Pointer for golang, which supports go `struct`.

## Announcements

* **2026-07-07** : landing v1.0.0
  * stable API pledge

* **2026-06-29** : reinsourced external dependency to swag (v0.24.0)
  * module `github.com/go-openapi/swag/jsonname` is source directly here, so we no longer have any external dependency
  * `jsonname` was never really used by any other package, so it makes sense to deprecate it away from the `swag` family
    and retrofit its functionality here. `jsonpointer` no longer get external dependencies, besides test dependencies.

* **2026-04-15** : added support for trailing "-" for arrays (v0.23.0)
  * this brings full support of [RFC6901][RFC6901]
  * this is supported for types relying on the reflection-based implemented
  * API semantics remain essentially unaltered. Exception: `Pointer.Set(document any,value any) (document any, err error)` 
    can only perform a best-effort to mutate the input document in place. In the case of adding elements to an array with a
    trailing "-", either pass a mutable array (`*[]T`) as the input document, or use the returned updated document instead.
  * types that implement the `JSONSetable` interface may not implement the mutation implied by the trailing "-"

* **2026-04-15** : added support for optional alternate JSON name providers
  * for struct support the defaults might not suit all situations: there are known limitations
    when it comes to handle untagged fields or embedded types.
  * the default name provider in use is not fully aligned with go JSON stdlib
  * exposed an option (or global setting) to change the provider that resolves a struct into json keys
  * the default behavior is not altered

## Status

API is stable and feature-complete.

The project continues to receive regular updates, bug fixes and hygiene maintenance (CI, linting, etc).

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

also known as [RFC6901][RFC6901].

## Licensing

This library ships under the [SPDX-License-Identifier: Apache-2.0](./LICENSE).

See the license [NOTICE](./NOTICE), which recalls the licensing terms of all the pieces of software
on top of which it has been built.

## Limitations

* [RFC6901][RFC6901] is now fully supported, including trailing "-" semantics for arrays (for `Set` operations).
* Default behavior: JSON name detection in go `struct`s
   - Unlike go standard marshaling, untagged fields do not default to the go field name and are ignored.
   - anonymous fields are not traversed if untagged
   - the above limitations may be overcome by calling `UseGoNameProvider()` at initialization time.
   - alternatively, users may inject the desired custom behavior for naming fields as an option.

## Other documentation

* [All-time contributors](./CONTRIBUTORS.md)
* [Contributing guidelines][contributing-doc-site]
* [Maintainers documentation][maintainers-doc-site]
* [Code style][style-doc-site]

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
<!-- Badges: code quality  -->
[gocard-badge]: https://goreportcard.com/badge/github.com/go-openapi/jsonpointer
[gocard-url]: https://goreportcard.com/report/github.com/go-openapi/jsonpointer
[codefactor-badge]: https://img.shields.io/codefactor/grade/github/go-openapi/jsonpointer
[codefactor-url]: https://www.codefactor.io/repository/github/go-openapi/jsonpointer
<!-- Badges: documentation & support -->
[godoc-badge]: https://pkg.go.dev/badge/github.com/go-openapi/jsonpointer
[godoc-url]: http://pkg.go.dev/github.com/go-openapi/jsonpointer
[discord-badge]: https://img.shields.io/discord/1446918742398341256?logo=discord&label=discord&color=blue
[discord-url]: https://discord.gg/FfnFYaC3k5

<!-- Badges: license & compliance -->
[license-badge]: http://img.shields.io/badge/license-Apache%20v2-orange.svg
[license-url]: https://github.com/go-openapi/jsonpointer/?tab=Apache-2.0-1-ov-file#readme
<!-- Badges: others & stats -->
[goversion-badge]: https://img.shields.io/github/go-mod/go-version/go-openapi/jsonpointer
[goversion-url]: https://github.com/go-openapi/jsonpointer/blob/master/go.mod
[top-badge]: https://img.shields.io/github/languages/top/go-openapi/jsonpointer
[commits-badge]: https://img.shields.io/github/commits-since/go-openapi/jsonpointer/latest
[RFC6901]: https://www.rfc-editor.org/rfc/rfc6901
<!-- Organization docs -->
[contributing-doc-site]: https://go-openapi.github.io/doc-site/contributing/contributing/index.html
[maintainers-doc-site]: https://go-openapi.github.io/doc-site/maintainers/index.html
[style-doc-site]: https://go-openapi.github.io/doc-site/contributing/style/index.html
