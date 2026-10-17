go-errors/errors
================

[![Build Status](https://travis-ci.org/go-errors/errors.svg?branch=master)](https://travis-ci.org/go-errors/errors)

Package errors adds stacktrace support to errors in go.

This is particularly useful when you want to understand the state of execution
when an error was returned unexpectedly.

It provides the type \*Error which implements the standard golang error
interface, so you can use this library interchangeably with code that is
expecting a normal error return.

Usage
-----

Full documentation is available on
[godoc](https://godoc.org/github.com/go-errors/errors), but here's a simple
example:

```go
package crashy

import "github.com/go-errors/errors"

var Crashed = errors.Errorf("oh dear")

func Crash() error {
    return errors.New(Crashed)
}
```

This can be called as follows:

```go
package main

import (
    "crashy"
    "fmt"
    "github.com/go-errors/errors"
)

func main() {
    err := crashy.Crash()
    if err != nil {
        if errors.Is(err, crashy.Crashed) {
            fmt.Println(err.(*errors.Error).ErrorStack())
        } else {
            panic(err)
        }
    }
}
```

Meta-fu
-------

This package was original written to allow reporting to
[Bugsnag](https://bugsnag.com/) from
[bugsnag-go](https://github.com/bugsnag/bugsnag-go), but after I found similar
packages by Facebook and Dropbox, it was moved to one canonical location so
everyone can benefit.

This package is licensed under the MIT license, see LICENSE.MIT for details.


## Changelog
* v1.1.0 updated to use go1.13's standard-library errors.Is method instead of == in errors.Is
* v1.2.0 added `errors.As` from the standard library.
* v1.3.0 *BREAKING* updated error methods to return `error` instead of `*Error`.
>  Code that needs access to the underlying `*Error` can use the new errors.AsError(e)
> ```
>   // before
>   errors.New(err).ErrorStack()
>   // after
>.  errors.AsError(errors.Wrap(err)).ErrorStack()
> ```
* v1.4.0 *BREAKING* v1.4.0 reverted all changes from v1.3.0 and is identical to v1.2.0
* v1.4.1 no code change, but now without an unnecessary cover.out file.
* v1.4.2 performance improvement to ErrorStack() to avoid unnecessary work https://github.com/go-errors/errors/pull/40
* v1.5.0 add errors.Join() and errors.Unwrap() copying the stdlib https://github.com/go-errors/errors/pull/40
* v1.5.1 fix build on go1.13..go1.19 (broken by adding Join and Unwrap with wrong build constraints)
