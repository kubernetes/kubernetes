// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.7

package http2

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"
)

type contextContext interface {
	Done() <-chan struct{}
	Err() error
}

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

// temporary copy of Go 1.6's private tls.Config.clone:
func cloneTLSConfig(c *tls.Config) *tls.Config {
	return &tls.Config{
		Rand:                     c.Rand,
		Time:                     c.Time,
		Certificates:             c.Certificates,
		NameToCertificate:        c.NameToCertificate,
		GetCertificate:           c.GetCertificate,
		RootCAs:                  c.RootCAs,
		NextProtos:               c.NextProtos,
		ServerName:               c.ServerName,
		ClientAuth:               c.ClientAuth,
		ClientCAs:                c.ClientCAs,
		InsecureSkipVerify:       c.InsecureSkipVerify,
		CipherSuites:             c.CipherSuites,
		PreferServerCipherSuites: c.PreferServerCipherSuites,
		SessionTicketsDisabled:   c.SessionTicketsDisabled,
		SessionTicketKey:         c.SessionTicketKey,
		ClientSessionCache:       c.ClientSessionCache,
		MinVersion:               c.MinVersion,
		MaxVersion:               c.MaxVersion,
		CurvePreferences:         c.CurvePreferences,
	}
}

func (cc *ClientConn) Ping(ctx contextContext) error {
	return cc.ping(ctx)
}

func (t *Transport) idleConnTimeout() time.Duration { return 0 }
