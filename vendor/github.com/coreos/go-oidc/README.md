# go-oidc

[![GoDoc](https://godoc.org/github.com/coreos/go-oidc?status.svg)](https://godoc.org/github.com/coreos/go-oidc)
[![Build Status](https://travis-ci.org/coreos/go-oidc.png?branch=master)](https://travis-ci.org/coreos/go-oidc)

go-oidc provides a comprehensive collection of golang libraries for other projects to implement [OpenID Connect (OIDC)][oidc] server and client components.

[oidc]: http://openid.net/connect

## package documentation

- [github.com/coreos/go-oidc/oidc](http://godoc.org/github.com/coreos/go-oidc/oidc) - OIDC client- and server-related components
- [github.com/coreos/go-oidc/oauth2](http://godoc.org/github.com/coreos/go-oidc/oauth2) - OAuth2-specific code needed by the OIDC components
- [github.com/coreos/go-oidc/jose](http://godoc.org/github.com/coreos/go-oidc/jose) - Javascript Object Signing and Encryption (JOSE) object ([JWS](https://tools.ietf.org/html/draft-ietf-jose-json-web-signature-41), [JWK](https://tools.ietf.org/html/draft-ietf-jose-json-web-key-41)) generation, validation and serialization
- [github.com/coreos/go-oidc/key](http://godoc.org/github.com/coreos/go-oidc/key) - RSA key management for OIDC components
