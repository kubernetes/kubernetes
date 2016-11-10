// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.7

package http2

import (
	"net"
	"net/http"
)

type contextContext interface{}

type fakeContext struct{}

func (fakeContext) Done() <-chan struct{} { return nil }
func (fakeContext) Err() error            { panic("should not be called") }

func reqContext(r *http.Request) fakeContext {
	return fakeContext{}
}

func setResponseUncompressed(res *http.Response) {
	// Nothing.
}

type clientTrace struct{}

func requestTrace(*http.Request) *clientTrace { return nil }
func traceGotConn(*http.Request, *ClientConn) {}
func traceFirstResponseByte(*clientTrace)     {}
func traceWroteHeaders(*clientTrace)          {}
func traceWroteRequest(*clientTrace, error)   {}
func traceGot100Continue(trace *clientTrace)  {}
func traceWait100Continue(trace *clientTrace) {}

func nop() {}

func serverConnBaseContext(c net.Conn, opts *ServeConnOpts) (ctx contextContext, cancel func()) {
	return nil, nop
}

func contextWithCancel(ctx contextContext) (_ contextContext, cancel func()) {
	return ctx, nop
}

func requestWithContext(req *http.Request, ctx contextContext) *http.Request {
	return req
}
