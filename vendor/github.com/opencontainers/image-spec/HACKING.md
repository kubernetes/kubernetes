# Hacking Guide

## Overview

This guide contains instructions for building artifacts contained in this repository.

### Go

This spec includes several Go packages, and a command line tool considered to be a reference implementation of the OCI image specification.

Prerequisites:
* Go - current release only, earlier releases are not supported
* make

The following make targets are relevant for any work involving the Go packages.

### Linting

The included Go source code is being examined for any linting violations not included in the standard Go compiler. Linting is done using [gometalinter](https://github.com/alecthomas/gometalinter).

Invocation:
```
$ make lint
```

### Tests

This target executes all Go based tests.

Invocation:
```
$ make test
$ make validate-examples
```

### Virtual schema http/FileSystem

The `schema` validator uses a virtual [http/FileSystem](https://golang.org/pkg/net/http/#FileSystem) to load the JSON schema files for validating OCI images and/or manifests.
The virtual filesystem is generated using the `esc` tool and compiled into consumers of the `schema` package so the JSON schema files don't have to be distributed along with and consumer binaries.

Whenever changes are being done in any of the `schema/*.json` files, one must refresh the generated virtual filesystem.
Otherwise schema changes will not be visible inside `schema` consumers.

Prerequisites:
* [esc](https://github.com/mjibson/esc)

Invocation:
```
$ make schema-fs
```

### JSON schema formatting

This target auto-formats all JSON files in the `schema` directory using the `jq` tool.

Prerequisites:
* [jq](https://stedolan.github.io/jq/) >=1.5

Invocation:
```
$ make fmt
```

### OCI image specification PDF/HTML documentation files

This target generates a PDF/HTML version of the OCI image specification.

Prerequisites:
* [Docker](https://www.docker.com/)

Invocation:
```
$ make docs
```

### License header check

This target checks if the source code includes necessary headers.

Invocation:
```
$ make check-license
```

### Clean build artifacts

This target cleans all generated/compiled artifacts.

Invocation:
```
$ make clean
```

### Create PNG images from dot files

This target generates PNG image files from DOT source files in the `img` directory.

Prerequisites:
* [graphviz](http://www.graphviz.org/)

Invocation:
```
$ make img/media-types.png
```
