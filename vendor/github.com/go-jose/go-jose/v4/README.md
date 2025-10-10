# Go JOSE

[![godoc](https://pkg.go.dev/badge/github.com/go-jose/go-jose/v4.svg)](https://pkg.go.dev/github.com/go-jose/go-jose/v4)
[![godoc](https://pkg.go.dev/badge/github.com/go-jose/go-jose/v4/jwt.svg)](https://pkg.go.dev/github.com/go-jose/go-jose/v4/jwt)
[![license](https://img.shields.io/badge/license-apache_2.0-blue.svg?style=flat)](https://raw.githubusercontent.com/go-jose/go-jose/master/LICENSE)

Package jose aims to provide an implementation of the Javascript Object Signing
and Encryption set of standards. This includes support for JSON Web Encryption,
JSON Web Signature, and JSON Web Token standards.

## Overview

The implementation follows the
[JSON Web Encryption](https://dx.doi.org/10.17487/RFC7516) (RFC 7516),
[JSON Web Signature](https://dx.doi.org/10.17487/RFC7515) (RFC 7515), and
[JSON Web Token](https://dx.doi.org/10.17487/RFC7519) (RFC 7519) specifications.
Tables of supported algorithms are shown below. The library supports both
the compact and JWS/JWE JSON Serialization formats, and has optional support for
multiple recipients. It also comes with a small command-line utility
([`jose-util`](https://pkg.go.dev/github.com/go-jose/go-jose/jose-util))
for dealing with JOSE messages in a shell.

**Note**: We use a forked version of the `encoding/json` package from the Go
standard library which uses case-sensitive matching for member names (instead
of [case-insensitive matching](https://www.ietf.org/mail-archive/web/json/current/msg03763.html)).
This is to avoid differences in interpretation of messages between go-jose and
libraries in other languages.

### Versions

The forthcoming Version 5 will be released with several breaking API changes,
and will require Golang's `encoding/json/v2`, which is currently requires 
Go 1.25 built with GOEXPERIMENT=jsonv2.

Version 4 is the current stable version:

    import "github.com/go-jose/go-jose/v4"

It supports at least the current and previous Golang release. Currently it
requires Golang 1.23.

Version 3 is only receiving critical security updates. Migration to Version 4 is recommended.

Versions 1 and 2 are obsolete, but can be found in the old repository, [square/go-jose](https://github.com/square/go-jose).

### Supported algorithms

See below for a table of supported algorithms. Algorithm identifiers match
the names in the [JSON Web Algorithms](https://dx.doi.org/10.17487/RFC7518)
standard where possible. The Godoc reference has a list of constants.

| Key encryption         | Algorithm identifier(s)                        |
|:-----------------------|:-----------------------------------------------|
| RSA-PKCS#1v1.5         | RSA1_5                                         |
| RSA-OAEP               | RSA-OAEP, RSA-OAEP-256                         |
| AES key wrap           | A128KW, A192KW, A256KW                         |
| AES-GCM key wrap       | A128GCMKW, A192GCMKW, A256GCMKW                |
| ECDH-ES + AES key wrap | ECDH-ES+A128KW, ECDH-ES+A192KW, ECDH-ES+A256KW |
| ECDH-ES (direct)       | ECDH-ES<sup>1</sup>                            |
| Direct encryption      | dir<sup>1</sup>                                |

<sup>1. Not supported in multi-recipient mode</sup>

| Signing / MAC     | Algorithm identifier(s) |
|:------------------|:------------------------|
| RSASSA-PKCS#1v1.5 | RS256, RS384, RS512     |
| RSASSA-PSS        | PS256, PS384, PS512     |
| HMAC              | HS256, HS384, HS512     |
| ECDSA             | ES256, ES384, ES512     |
| Ed25519           | EdDSA<sup>2</sup>       |

<sup>2. Only available in version 2 of the package</sup>

| Content encryption | Algorithm identifier(s)                     |
|:-------------------|:--------------------------------------------|
| AES-CBC+HMAC       | A128CBC-HS256, A192CBC-HS384, A256CBC-HS512 |
| AES-GCM            | A128GCM, A192GCM, A256GCM                   |

| Compression        | Algorithm identifiers(s) |
|:-------------------|--------------------------|
| DEFLATE (RFC 1951) | DEF                      |

### Supported key types

See below for a table of supported key types. These are understood by the
library, and can be passed to corresponding functions such as `NewEncrypter` or
`NewSigner`. Each of these keys can also be wrapped in a JWK if desired, which
allows attaching a key id.

| Algorithm(s)      | Corresponding types                                                                                                                  |
|:------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| RSA               | *[rsa.PublicKey](https://pkg.go.dev/crypto/rsa/#PublicKey), *[rsa.PrivateKey](https://pkg.go.dev/crypto/rsa/#PrivateKey)             |
| ECDH, ECDSA       | *[ecdsa.PublicKey](https://pkg.go.dev/crypto/ecdsa/#PublicKey), *[ecdsa.PrivateKey](https://pkg.go.dev/crypto/ecdsa/#PrivateKey)     |
| EdDSA<sup>1</sup> | [ed25519.PublicKey](https://pkg.go.dev/crypto/ed25519#PublicKey), [ed25519.PrivateKey](https://pkg.go.dev/crypto/ed25519#PrivateKey) |
| AES, HMAC         | []byte                                                                                                                               |

<sup>1. Only available in version 2 or later of the package</sup>

## Examples

[![godoc](https://pkg.go.dev/badge/github.com/go-jose/go-jose/v4.svg)](https://pkg.go.dev/github.com/go-jose/go-jose/v4)
[![godoc](https://pkg.go.dev/badge/github.com/go-jose/go-jose/v4/jwt.svg)](https://pkg.go.dev/github.com/go-jose/go-jose/v4/jwt)

Examples can be found in the Godoc
reference for this package. The
[`jose-util`](https://github.com/go-jose/go-jose/tree/main/jose-util)
subdirectory also contains a small command-line utility which might be useful
as an example as well.
