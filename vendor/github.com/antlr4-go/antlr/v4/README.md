[![Go Report Card](https://goreportcard.com/badge/github.com/antlr4-go/antlr?style=flat-square)](https://goreportcard.com/report/github.com/antlr4-go/antlr)
[![PkgGoDev](https://pkg.go.dev/badge/github.com/github.com/antlr4-go/antlr)](https://pkg.go.dev/github.com/antlr4-go/antlr)
[![Release](https://img.shields.io/github/v/release/antlr4-go/antlr?sort=semver&style=flat-square)](https://github.com/antlr4-go/antlr/releases/latest)
[![Release](https://img.shields.io/github/go-mod/go-version/antlr4-go/antlr?style=flat-square)](https://github.com/antlr4-go/antlr/releases/latest)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=flat-square)](https://github.com/antlr4-go/antlr/commit-activity)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![GitHub stars](https://img.shields.io/github/stars/antlr4-go/antlr?style=flat-square&label=Star&maxAge=2592000)](https://GitHub.com/Naereen/StrapDown.js/stargazers/)
# ANTLR4 Go Runtime Module Repo

IMPORTANT: Please submit PRs via a clone of the https://github.com/antlr/antlr4 repo, and not here.

  - Do not submit PRs or any change requests to this repo
  - This repo is read only and is updated by the ANTLR team to create a new release of the Go Runtime for ANTLR
  - This repo contains the Go runtime that your generated projects should import

## Introduction

This repo contains the official modules for the Go Runtime for ANTLR. It is a copy of the runtime maintained
at: https://github.com/antlr/antlr4/tree/master/runtime/Go/antlr and is automatically updated by the ANTLR team to create
the official Go runtime release only. No development work is carried out in this repo and PRs are not accepted here.

The dev branch of this repo is kept in sync with the dev branch of the main ANTLR repo and is updated periodically.

### Why?

The `go get` command is unable to retrieve the Go runtime when it is embedded so
deeply in the main repo. A `go get` against the `antlr/antlr4` repo, while retrieving the correct source code for the runtime,
does not correctly resolve tags and will create a reference in your `go.mod` file that is unclear, will not upgrade smoothly and
causes confusion.

For instance, the current Go runtime release, which is tagged with v4.13.0 in `antlr/antlr4` is retrieved by go get as:

```sh
require (
	github.com/antlr/antlr4/runtime/Go/antlr/v4 v4.0.0-20230219212500-1f9a474cc2dc
)
```

Where you would expect to see:

```sh
require (
    github.com/antlr/antlr4/runtime/Go/antlr/v4 v4.13.0
)
```

The decision was taken to create a separate org in a separate repo to hold the official Go runtime for ANTLR and
from whence users can expect `go get` to behave as expected.


# Documentation
Please read the official documentation at: https://github.com/antlr/antlr4/blob/master/doc/index.md for tips on
migrating existing projects to use the new module location and for information on how to use the Go runtime in
general.
