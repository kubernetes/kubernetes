// Package cleanhttp offers convenience utilities for acquiring "clean"
// http.Transport and http.Client structs.
//
// Values set on http.DefaultClient and http.DefaultTransport affect all
// callers. This can have detrimental effects, esepcially in TLS contexts,
// where client or root certificates set to talk to multiple endpoints can end
// up displacing each other, leading to hard-to-debug issues. This package
// provides non-shared http.Client and http.Transport structs to ensure that
// the configuration will not be overwritten by other parts of the application
// or dependencies.
//
// The DefaultClient and DefaultTransport functions disable idle connections
// and keepalives. Without ensuring that idle connections are closed before
// garbage collection, short-term clients/transports can leak file descriptors,
// eventually leading to "too many open files" errors. If you will be
// connecting to the same hosts repeatedly from the same client, you can use
// DefaultPooledClient to receive a client that has connection pooling
// semantics similar to http.DefaultClient.
//
package cleanhttp
