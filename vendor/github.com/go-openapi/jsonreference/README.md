# jsonreference

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

An implementation of JSON Reference for golang.

## Status

API is stable.

## Import this library in your project

```cmd
go get github.com/go-openapi/jsonreference
```

## Dependencies

* https://github.com/go-openapi/jsonpointer

## Basic usage

## Change log

See <https://github.com/go-openapi/jsonreference/releases>

## References

* http://tools.ietf.org/html/draft-ietf-appsawg-json-pointer-07
* http://tools.ietf.org/html/draft-pbryan-zyp-json-ref-03

## Licensing

This library ships under the [SPDX-License-Identifier: Apache-2.0](./LICENSE).

See the license [NOTICE](./NOTICE), which recalls the licensing terms of all the pieces of software
on top of which it has been built.

## Other documentation

* [All-time contributors](./CONTRIBUTORS.md)
* [Contributing guidelines](.github/CONTRIBUTING.md)
* [Maintainers documentation](docs/MAINTAINERS.md)
* [Code style](docs/STYLE.md)

## Cutting a new release

Maintainers can cut a new release by either:

* running [this workflow](https://github.com/go-openapi/jsonreference/actions/workflows/bump-release.yml)
* or pushing a semver tag
  * signed tags are preferred
  * The tag message is prepended to release notes

<!-- Badges: status  -->
[test-badge]: https://github.com/go-openapi/jsonreference/actions/workflows/go-test.yml/badge.svg
[test-url]: https://github.com/go-openapi/jsonreference/actions/workflows/go-test.yml
[cov-badge]: https://codecov.io/gh/go-openapi/jsonreference/branch/master/graph/badge.svg
[cov-url]: https://codecov.io/gh/go-openapi/jsonreference
[vuln-scan-badge]: https://github.com/go-openapi/jsonreference/actions/workflows/scanner.yml/badge.svg
[vuln-scan-url]: https://github.com/go-openapi/jsonreference/actions/workflows/scanner.yml
[codeql-badge]: https://github.com/go-openapi/jsonreference/actions/workflows/codeql.yml/badge.svg
[codeql-url]: https://github.com/go-openapi/jsonreference/actions/workflows/codeql.yml
<!-- Badges: release & docker images  -->
[release-badge]: https://badge.fury.io/gh/go-openapi%2Fjsonreference.svg
[release-url]: https://badge.fury.io/gh/go-openapi%2Fjsonreference
[gomod-badge]: https://badge.fury.io/go/github.com%2Fgo-openapi%2Fjsonreference.svg
[gomod-url]: https://badge.fury.io/go/github.com%2Fgo-openapi%2Fjsonreference
<!-- Badges: code quality  -->
[gocard-badge]: https://goreportcard.com/badge/github.com/go-openapi/jsonreference
[gocard-url]: https://goreportcard.com/report/github.com/go-openapi/jsonreference
[codefactor-badge]: https://img.shields.io/codefactor/grade/github/go-openapi/jsonreference
[codefactor-url]: https://www.codefactor.io/repository/github/go-openapi/jsonreference
<!-- Badges: documentation & support -->
[doc-badge]: https://img.shields.io/badge/doc-site-blue?link=https%3A%2F%2Fgoswagger.io%2Fgo-openapi%2F
[doc-url]: https://goswagger.io/go-openapi
[godoc-badge]: https://pkg.go.dev/badge/github.com/go-openapi/jsonreference
[godoc-url]: http://pkg.go.dev/github.com/go-openapi/jsonreference
[slack-logo]: https://a.slack-edge.com/e6a93c1/img/icons/favicon-32.png
[slack-badge]: https://img.shields.io/badge/slack-blue?link=https%3A%2F%2Fgoswagger.slack.com%2Farchives%2FC04R30YM
[slack-url]: https://goswagger.slack.com/archives/C04R30YMU
<!-- Badges: license & compliance -->
[license-badge]: http://img.shields.io/badge/license-Apache%20v2-orange.svg
[license-url]: https://github.com/go-openapi/jsonreference/?tab=Apache-2.0-1-ov-file#readme
<!-- Badges: others & stats -->
[goversion-badge]: https://img.shields.io/github/go-mod/go-version/go-openapi/jsonreference
[goversion-url]: https://github.com/go-openapi/jsonreference/blob/master/go.mod
[top-badge]: https://img.shields.io/github/languages/top/go-openapi/jsonreference
[commits-badge]: https://img.shields.io/github/commits-since/go-openapi/jsonreference/latest
