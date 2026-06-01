# jwt-go

[![build](https://github.com/golang-jwt/jwt/actions/workflows/build.yml/badge.svg)](https://github.com/golang-jwt/jwt/actions/workflows/build.yml)
[![Go
Reference](https://pkg.go.dev/badge/github.com/golang-jwt/jwt/v5.svg)](https://pkg.go.dev/github.com/golang-jwt/jwt/v5)
[![Coverage Status](https://coveralls.io/repos/github/golang-jwt/jwt/badge.svg?branch=main)](https://coveralls.io/github/golang-jwt/jwt?branch=main)

A [go](http://www.golang.org) (or 'golang' for search engine friendliness)
implementation of [JSON Web
Tokens](https://datatracker.ietf.org/doc/html/rfc7519).

Starting with [v4.0.0](https://github.com/golang-jwt/jwt/releases/tag/v4.0.0)
this project adds Go module support, but maintains backward compatibility with
older `v3.x.y` tags and upstream `github.com/dgrijalva/jwt-go`. See the
[`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) for more information. Version
v5.0.0 introduces major improvements to the validation of tokens, but is not
entirely backward compatible. 

> After the original author of the library suggested migrating the maintenance
> of `jwt-go`, a dedicated team of open source maintainers decided to clone the
> existing library into this repository. See
> [dgrijalva/jwt-go#462](https://github.com/dgrijalva/jwt-go/issues/462) for a
> detailed discussion on this topic.


**SECURITY NOTICE:** Some older versions of Go have a security issue in the
crypto/elliptic. The recommendation is to upgrade to at least 1.15 See issue
[dgrijalva/jwt-go#216](https://github.com/dgrijalva/jwt-go/issues/216) for more
detail.

**SECURITY NOTICE:** It's important that you [validate the `alg` presented is
what you
expect](https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/).
This library attempts to make it easy to do the right thing by requiring key
types to match the expected alg, but you should take the extra step to verify it in
your usage.  See the examples provided.

### Supported Go versions

Our support of Go versions is aligned with Go's [version release
policy](https://golang.org/doc/devel/release#policy). So we will support a major
version of Go until there are two newer major releases. We no longer support
building jwt-go with unsupported Go versions, as these contain security
vulnerabilities that will not be fixed.

## What the heck is a JWT?

JWT.io has [a great introduction](https://jwt.io/introduction) to JSON Web
Tokens.

In short, it's a signed JSON object that does something useful (for example,
authentication).  It's commonly used for `Bearer` tokens in Oauth 2.  A token is
made of three parts, separated by `.`'s.  The first two parts are JSON objects,
that have been [base64url](https://datatracker.ietf.org/doc/html/rfc4648)
encoded.  The last part is the signature, encoded the same way.

The first part is called the header.  It contains the necessary information for
verifying the last part, the signature.  For example, which encryption method
was used for signing and what key was used.

The part in the middle is the interesting bit.  It's called the Claims and
contains the actual stuff you care about.  Refer to [RFC
7519](https://datatracker.ietf.org/doc/html/rfc7519) for information about
reserved keys and the proper way to add your own.

## What's in the box?

This library supports the parsing and verification as well as the generation and
signing of JWTs.  Current supported signing algorithms are HMAC SHA, RSA,
RSA-PSS, and ECDSA, though hooks are present for adding your own.

## Installation Guidelines

1. To install the jwt package, you first need to have
   [Go](https://go.dev/doc/install) installed, then you can use the command
   below to add `jwt-go` as a dependency in your Go program.

```sh
go get -u github.com/golang-jwt/jwt/v5
```

2. Import it in your code:

```go
import "github.com/golang-jwt/jwt/v5"
```

## Usage

A detailed usage guide, including how to sign and verify tokens can be found on
our [documentation website](https://golang-jwt.github.io/jwt/usage/create/).

## Examples

See [the project documentation](https://pkg.go.dev/github.com/golang-jwt/jwt/v5)
for examples of usage:

* [Simple example of parsing and validating a
  token](https://pkg.go.dev/github.com/golang-jwt/jwt/v5#example-Parse-Hmac)
* [Simple example of building and signing a
  token](https://pkg.go.dev/github.com/golang-jwt/jwt/v5#example-New-Hmac)
* [Directory of
  Examples](https://pkg.go.dev/github.com/golang-jwt/jwt/v5#pkg-examples)

## Compliance

This library was last reviewed to comply with [RFC
7519](https://datatracker.ietf.org/doc/html/rfc7519) dated May 2015 with a few
notable differences:

* In order to protect against accidental use of [Unsecured
  JWTs](https://datatracker.ietf.org/doc/html/rfc7519#section-6), tokens using
  `alg=none` will only be accepted if the constant
  `jwt.UnsafeAllowNoneSignatureType` is provided as the key.

## Project Status & Versioning

This library is considered production ready.  Feedback and feature requests are
appreciated.  The API should be considered stable.  There should be very few
backward-incompatible changes outside of major version updates (and only with
good reason).

This project uses [Semantic Versioning 2.0.0](http://semver.org).  Accepted pull
requests will land on `main`.  Periodically, versions will be tagged from
`main`.  You can find all the releases on [the project releases
page](https://github.com/golang-jwt/jwt/releases).

**BREAKING CHANGES:** A full list of breaking changes is available in
`VERSION_HISTORY.md`.  See [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) for more information on updating
your code.

## Extensions

This library publishes all the necessary components for adding your own signing
methods or key functions.  Simply implement the `SigningMethod` interface and
register a factory method using `RegisterSigningMethod` or provide a
`jwt.Keyfunc`.

A common use case would be integrating with different 3rd party signature
providers, like key management services from various cloud providers or Hardware
Security Modules (HSMs) or to implement additional standards.

| Extension | Purpose                                                                                                  | Repo                                              |
| --------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| GCP       | Integrates with multiple Google Cloud Platform signing tools (AppEngine, IAM API, Cloud KMS)             | https://github.com/someone1/gcp-jwt-go            |
| AWS       | Integrates with AWS Key Management Service, KMS                                                          | https://github.com/matelang/jwt-go-aws-kms        |
| JWKS      | Provides support for JWKS ([RFC 7517](https://datatracker.ietf.org/doc/html/rfc7517)) as a `jwt.Keyfunc` | https://github.com/MicahParks/keyfunc             |
| TPM       | Integrates with Trusted Platform Module (TPM)                                                            | https://github.com/salrashid123/golang-jwt-tpm    |

*Disclaimer*: Unless otherwise specified, these integrations are maintained by
third parties and should not be considered as a primary offer by any of the
mentioned cloud providers

## More

Go package documentation can be found [on
pkg.go.dev](https://pkg.go.dev/github.com/golang-jwt/jwt/v5). Additional
documentation can be found on [our project
page](https://golang-jwt.github.io/jwt/).

The command line utility included in this project (cmd/jwt) provides a
straightforward example of token creation and parsing as well as a useful tool
for debugging your own integration. You'll also find several implementation
examples in the documentation.

[golang-jwt](https://github.com/orgs/golang-jwt) incorporates a modified version
of the JWT logo, which is distributed under the terms of the [MIT
License](https://github.com/jsonwebtoken/jsonwebtoken.github.io/blob/master/LICENSE.txt).
