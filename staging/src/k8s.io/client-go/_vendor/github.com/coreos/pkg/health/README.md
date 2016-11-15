health
====

A simple framework for implementing an HTTP health check endpoint on servers.

Users implement their `health.Checkable` types, and create a `health.Checker`, from which they can get an `http.HandlerFunc` using `health.Checker.MakeHealthHandlerFunc`.

### Documentation

For more details, visit the docs on [gopkgdoc](http://godoc.org/github.com/coreos/pkg/health)

