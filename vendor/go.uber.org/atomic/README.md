# atomic [![GoDoc][doc-img]][doc] [![Build Status][ci-img]][ci] [![Coverage Status][cov-img]][cov] [![Go Report Card][reportcard-img]][reportcard]

Simple wrappers for primitive types to enforce atomic access.

## Installation
`go get -u go.uber.org/atomic`

## Usage
The standard library's `sync/atomic` is powerful, but it's easy to forget which
variables must be accessed atomically. `go.uber.org/atomic` preserves all the
functionality of the standard library, but wraps the primitive types to
provide a safer, more convenient API.

```go
var atom atomic.Uint32
atom.Store(42)
atom.Sub(2)
atom.CAS(40, 11)
```

See the [documentation][doc] for a complete API specification.

## Development Status
Stable.

___
Released under the [MIT License](LICENSE.txt).

[doc-img]: https://godoc.org/github.com/uber-go/atomic?status.svg
[doc]: https://godoc.org/go.uber.org/atomic
[ci-img]: https://travis-ci.com/uber-go/atomic.svg?branch=master
[ci]: https://travis-ci.com/uber-go/atomic
[cov-img]: https://codecov.io/gh/uber-go/atomic/branch/master/graph/badge.svg
[cov]: https://codecov.io/gh/uber-go/atomic
[reportcard-img]: https://goreportcard.com/badge/go.uber.org/atomic
[reportcard]: https://goreportcard.com/report/go.uber.org/atomic
