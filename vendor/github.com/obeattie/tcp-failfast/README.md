# `tcp-failfast`

[![Build Status Widget]][Build Status] [![GoDoc Widget]][GoDoc]

`tcp-failfast` is a Go library which allows control over the TCP "user timeout"
behavior. ⏱

This timeout is specified in [RFC 793] but is not implemented on all platforms.
Currently Linux and Darwin are supported.

[Build Status]: https://travis-ci.org/obeattie/tcp-failfast
[Build Status Widget]: https://travis-ci.org/obeattie/tcp-failfast.svg?branch=master
[GoDoc]: https://godoc.org/github.com/obeattie/tcp-failfast
[GoDoc Widget]: https://godoc.org/github.com/obeattie/tcp-failfast?status.svg
[RFC 793]: https://tools.ietf.org/html/rfc793