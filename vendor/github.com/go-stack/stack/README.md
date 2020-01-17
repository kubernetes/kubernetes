[![GoDoc](https://godoc.org/github.com/go-stack/stack?status.svg)](https://godoc.org/github.com/go-stack/stack)
[![Go Report Card](https://goreportcard.com/badge/go-stack/stack)](https://goreportcard.com/report/go-stack/stack)
[![TravisCI](https://travis-ci.org/go-stack/stack.svg?branch=master)](https://travis-ci.org/go-stack/stack)
[![Coverage Status](https://coveralls.io/repos/github/go-stack/stack/badge.svg?branch=master)](https://coveralls.io/github/go-stack/stack?branch=master)

# stack

Package stack implements utilities to capture, manipulate, and format call
stacks. It provides a simpler API than package runtime.

The implementation takes care of the minutia and special cases of interpreting
the program counter (pc) values returned by runtime.Callers.

## Versioning

Package stack publishes releases via [semver](http://semver.org/) compatible Git
tags prefixed with a single 'v'. The master branch always contains the latest
release. The develop branch contains unreleased commits.

## Formatting

Package stack's types implement fmt.Formatter, which provides a simple and
flexible way to declaratively configure formatting when used with logging or
error tracking packages.

```go
func DoTheThing() {
    c := stack.Caller(0)
    log.Print(c)          // "source.go:10"
    log.Printf("%+v", c)  // "pkg/path/source.go:10"
    log.Printf("%n", c)   // "DoTheThing"

    s := stack.Trace().TrimRuntime()
    log.Print(s)          // "[source.go:15 caller.go:42 main.go:14]"
}
```

See the docs for all of the supported formatting options.
