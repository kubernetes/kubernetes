go-retryablehttp
================

[![Build Status](http://img.shields.io/travis/hashicorp/go-retryablehttp.svg?style=flat-square)][travis]
[![Go Documentation](http://img.shields.io/badge/go-documentation-blue.svg?style=flat-square)][godocs]

[travis]: http://travis-ci.org/hashicorp/go-retryablehttp
[godocs]: http://godoc.org/github.com/hashicorp/go-retryablehttp

The `retryablehttp` package provides a familiar HTTP client interface with
automatic retries and exponential backoff. It is a thin wrapper over the
standard `net/http` client library and exposes nearly the same public API. This
makes `retryablehttp` very easy to drop into existing programs.

`retryablehttp` performs automatic retries under certain conditions. Mainly, if
an error is returned by the client (connection errors, etc.), or if a 500-range
response code is received (except 501), then a retry is invoked after a wait
period.  Otherwise, the response is returned and left to the caller to
interpret.

The main difference from `net/http` is that requests which take a request body
(POST/PUT et. al) can have the body provided in a number of ways (some more or
less efficient) that allow "rewinding" the request body if the initial request
fails so that the full request can be attempted again. See the
[godoc](http://godoc.org/github.com/hashicorp/go-retryablehttp) for more
details.

Version 0.6.0 and before are compatible with Go prior to 1.12. From 0.6.1 onward, Go 1.12+ is required.

Example Use
===========

Using this library should look almost identical to what you would do with
`net/http`. The most simple example of a GET request is shown below:

```go
resp, err := retryablehttp.Get("/foo")
if err != nil {
    panic(err)
}
```

The returned response object is an `*http.Response`, the same thing you would
usually get from `net/http`. Had the request failed one or more times, the above
call would block and retry with exponential backoff.

For more usage and examples see the
[godoc](http://godoc.org/github.com/hashicorp/go-retryablehttp).
