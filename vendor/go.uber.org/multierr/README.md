# multierr [![GoDoc][doc-img]][doc] [![Build Status][ci-img]][ci] [![Coverage Status][cov-img]][cov]

`multierr` allows combining one or more Go `error`s together.

## Features

- **Idiomatic**:
  multierr follows best practices in Go, and keeps your code idiomatic.
    - It keeps the underlying error type hidden,
      allowing you to deal in `error` values exclusively.
    - It provides APIs to safely append into an error from a `defer` statement.
- **Performant**:
  multierr is optimized for performance:
    - It avoids allocations where possible.
    - It utilizes slice resizing semantics to optimize common cases
      like appending into the same error object from a loop.
- **Interoperable**:
  multierr interoperates with the Go standard library's error APIs seamlessly:
    - The `errors.Is` and `errors.As` functions *just work*.
- **Lightweight**:
  multierr comes with virtually no dependencies.

## Installation

```bash
go get -u go.uber.org/multierr@latest
```

## Status

Stable: No breaking changes will be made before 2.0.

-------------------------------------------------------------------------------

Released under the [MIT License].

[MIT License]: LICENSE.txt
[doc-img]: https://pkg.go.dev/badge/go.uber.org/multierr
[doc]: https://pkg.go.dev/go.uber.org/multierr
[ci-img]: https://github.com/uber-go/multierr/actions/workflows/go.yml/badge.svg
[cov-img]: https://codecov.io/gh/uber-go/multierr/branch/master/graph/badge.svg
[ci]: https://github.com/uber-go/multierr/actions/workflows/go.yml
[cov]: https://codecov.io/gh/uber-go/multierr
