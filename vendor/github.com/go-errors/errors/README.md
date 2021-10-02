go-errors/errors
================

[![Build Status](https://travis-ci.org/go-errors/errors.svg?branch=master)](https://travis-ci.org/go-errors/errors)

Package errors adds stacktrace support to errors in go.

This is particularly useful when you want to understand the state of execution
when an error was returned unexpectedly.

It provides the type \*Error which implements the standard golang error
interface, so you can use this library interchangably with code that is
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
