// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.7

package http2

import (
	"context"
	"net"
	"net/http"
	"net/http/httptrace"
	"time"
)

type contextContext interface {
	context.Context
}

func serverConnBaseContext(c net.Conn, opts *ServeConnOpts) (ctx contextContext, cancel func()) {
	ctx, cancel = context.WithCancel(context.Background())
	ctx = context.WithValue(ctx, http.LocalAddrContextKey, c.LocalAddr())
	if hs := opts.baseConfig(); hs != nil {
		ctx = context.WithValue(ctx, http.ServerContextKey, hs)
	}
	return
}

func contextWithCancel(ctx contextContext) (_ contextContext, cancel func()) {
	return context.WithCancel(ctx)
}

func requestWithContext(req *http.Request, ctx contextContext) *http.Request {
	return req.WithContext(ctx)
}

type clientTrace httptrace.ClientTrace

func reqContext(r *http.Request) context.Context { return r.Context() }

func setResponseUncompressed(res *http.Response) { res.Uncompressed = true }

func traceGotConn(req *http.Request, cc *ClientConn) {
	trace := httptrace.ContextClientTrace(req.Context())
	if trace == nil || trace.GotConn == nil {
		return
	}
	ci := httptrace.GotConnInfo{Conn: cc.tconn}
	cc.mu.Lock()
	ci.Reused = cc.nextStreamID > 1
	ci.WasIdle = len(cc.streams) == 0 && ci.Reused
	if ci.WasIdle && !cc.lastActive.IsZero() {
		ci.IdleTime = time.Now().Sub(cc.lastActive)
	}
	cc.mu.Unlock()

	trace.GotConn(ci)
}

func traceWroteHeaders(trace *clientTrace) {
	if trace != nil && trace.WroteHeaders != nil {
		trace.WroteHeaders()
	}
}

func traceGot100Continue(trace *clientTrace) {
	if trace != nil && trace.Got100Continue != nil {
		trace.Got100Continue()
	}
}

func traceWait100Continue(trace *clientTrace) {
	if trace != nil && trace.Wait100Continue != nil {
		trace.Wait100Continue()
	}
}

func traceWroteRequest(trace *clientTrace, err error) {
	if trace != nil && trace.WroteRequest != nil {
		trace.WroteRequest(httptrace.WroteRequestInfo{Err: err})
	}
}

func traceFirstResponseByte(trace *clientTrace) {
	if trace != nil && trace.GotFirstResponseByte != nil {
		trace.GotFirstResponseByte()
	}
}

func requestTrace(req *http.Request) *clientTrace {
	trace := httptrace.ContextClientTrace(req.Context())
	return (*clientTrace)(trace)
}
