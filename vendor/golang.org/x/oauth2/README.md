# OAuth2 for Go

[![Build Status](https://travis-ci.org/golang/oauth2.svg?branch=master)](https://travis-ci.org/golang/oauth2)
[![GoDoc](https://godoc.org/golang.org/x/oauth2?status.svg)](https://godoc.org/golang.org/x/oauth2)

oauth2 package contains a client implementation for OAuth 2.0 spec.

## Installation

~~~~
go get golang.org/x/oauth2
~~~~

Or you can manually git clone the repository to
`$(go env GOPATH)/src/golang.org/x/oauth2`.

See godoc for further documentation and examples.

* [godoc.org/golang.org/x/oauth2](https://godoc.org/golang.org/x/oauth2)
* [godoc.org/golang.org/x/oauth2/google](https://godoc.org/golang.org/x/oauth2/google)

## Policy for new packages

We no longer accept new provider-specific packages in this repo if all
they do is add a single endpoint variable. If you just want to add a
single endpoint, add it to the
[godoc.org/golang.org/x/oauth2/endpoints](https://godoc.org/golang.org/x/oauth2/endpoints)
package.

## Report Issues / Send Patches

This repository uses Gerrit for code changes. To learn how to submit changes to
this repository, see https://golang.org/doc/contribute.html.

The main issue tracker for the oauth2 repository is located at
https://github.com/golang/oauth2/issues.
