This package (golang.org/x/net/http2) is the original source of truth
of the Go HTTP/2 implementation.

As of Go 1.27, the source of truth has moved to the standard library
package net/http/internal/http2.
All new feature development should happen in that package.
Only critical bug fixes and security fixes will be backported to x/net.

The x/net package contains two implementations of the HTTP/2 transport and server:

The original implementation (no longer the source of truth).

A reimplementation of the x/net/http2 APIs in terms of net/http.
This is called "the wrapping implementation", since it wraps net/http.

The original implementation is used when the Go version is less than 1.27.

The wrapping implementation is used when the Go version is at least 1.27.
The build tag "http2legacy" may be set to use the original implementation.
