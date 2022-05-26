# OAuth2 for Go

[![Go Reference](https://pkg.go.dev/badge/golang.org/x/oauth2.svg)](https://pkg.go.dev/golang.org/x/oauth2)
[![Build Status](https://travis-ci.org/golang/oauth2.svg?branch=master)](https://travis-ci.org/golang/oauth2)

oauth2 package contains a client implementation for OAuth 2.0 spec.

## Installation

~~~~
go get golang.org/x/oauth2
~~~~

Or you can manually git clone the repository to
`$(go env GOPATH)/src/golang.org/x/oauth2`.

See pkg.go.dev for further documentation and examples.

* [pkg.go.dev/golang.org/x/oauth2](https://pkg.go.dev/golang.org/x/oauth2)
* [pkg.go.dev/golang.org/x/oauth2/google](https://pkg.go.dev/golang.org/x/oauth2/google)

## Policy for new packages

We no longer accept new provider-specific packages in this repo if all
they do is add a single endpoint variable. If you just want to add a
single endpoint, add it to the
[pkg.go.dev/golang.org/x/oauth2/endpoints](https://pkg.go.dev/golang.org/x/oauth2/endpoints)
package.

## Report Issues / Send Patches

This repository uses Gerrit for code changes. To learn how to submit changes to
this repository, see https://golang.org/doc/contribute.html.

The main issue tracker for the oauth2 repository is located at
https://github.com/golang/oauth2/issues.
