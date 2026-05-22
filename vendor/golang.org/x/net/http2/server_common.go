// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"context"
	"net"
	"net/http"
	"time"
)

// ConfigureServer adds HTTP/2 support to a net/http Server.
//
// The configuration conf may be nil.
//
// ConfigureServer must be called before s begins serving.
func ConfigureServer(s *http.Server, conf *Server) error {
	return configureServer(s, conf)
}

// Server is an HTTP/2 server.
type Server struct {
	// MaxHandlers limits the number of http.Handler ServeHTTP goroutines
	// which may run at a time over all connections.
	// Negative or zero no limit.
	// TODO: implement
	MaxHandlers int

	// MaxConcurrentStreams optionally specifies the number of
	// concurrent streams that each client may have open at a
	// time. This is unrelated to the number of http.Handler goroutines
	// which may be active globally, which is MaxHandlers.
	// If zero, MaxConcurrentStreams defaults to at least 100, per
	// the HTTP/2 spec's recommendations.
	MaxConcurrentStreams uint32

	// MaxDecoderHeaderTableSize optionally specifies the http2
	// SETTINGS_HEADER_TABLE_SIZE to send in the initial settings frame. It
	// informs the remote endpoint of the maximum size of the header compression
	// table used to decode header blocks, in octets. If zero, the default value
	// of 4096 is used.
	MaxDecoderHeaderTableSize uint32

	// MaxEncoderHeaderTableSize optionally specifies an upper limit for the
	// header compression table used for encoding request headers. Received
	// SETTINGS_HEADER_TABLE_SIZE settings are capped at this limit. If zero,
	// the default value of 4096 is used.
	MaxEncoderHeaderTableSize uint32

	// MaxReadFrameSize optionally specifies the largest frame
	// this server is willing to read. A valid value is between
	// 16k and 16M, inclusive. If zero or otherwise invalid, a
	// default value is used.
	MaxReadFrameSize uint32

	// PermitProhibitedCipherSuites, if true, permits the use of
	// cipher suites prohibited by the HTTP/2 spec.
	PermitProhibitedCipherSuites bool

	// IdleTimeout specifies how long until idle clients should be
	// closed with a GOAWAY frame. PING frames are not considered
	// activity for the purposes of IdleTimeout.
	// If zero or negative, there is no timeout.
	IdleTimeout time.Duration

	// ReadIdleTimeout is the timeout after which a health check using a ping
	// frame will be carried out if no frame is received on the connection.
	// If zero, no health check is performed.
	ReadIdleTimeout time.Duration

	// PingTimeout is the timeout after which the connection will be closed
	// if a response to a ping is not received.
	// If zero, a default of 15 seconds is used.
	PingTimeout time.Duration

	// WriteByteTimeout is the timeout after which a connection will be
	// closed if no data can be written to it. The timeout begins when data is
	// available to write, and is extended whenever any bytes are written.
	// If zero or negative, there is no timeout.
	WriteByteTimeout time.Duration

	// MaxUploadBufferPerConnection is the size of the initial flow
	// control window for each connections. The HTTP/2 spec does not
	// allow this to be smaller than 65535 or larger than 2^32-1.
	// If the value is outside this range, a default value will be
	// used instead.
	MaxUploadBufferPerConnection int32

	// MaxUploadBufferPerStream is the size of the initial flow control
	// window for each stream. The HTTP/2 spec does not allow this to
	// be larger than 2^32-1. If the value is zero or larger than the
	// maximum, a default value will be used instead.
	MaxUploadBufferPerStream int32

	// NewWriteScheduler constructs a write scheduler for a connection.
	// If nil, a default scheduler is chosen.
	//
	// Deprecated: User-provided write schedulers are deprecated.
	NewWriteScheduler func() WriteScheduler

	// CountError, if non-nil, is called on HTTP/2 server errors.
	// It's intended to increment a metric for monitoring, such
	// as an expvar or Prometheus metric.
	// The errType consists of only ASCII word characters.
	CountError func(errType string)

	// Internal state. This is a pointer (rather than embedded directly)
	// so that we don't embed a Mutex in this struct, which will make the
	// struct non-copyable, which might break some callers.
	state *serverInternalState
}

// ServeConnOpts are options for the Server.ServeConn method.
type ServeConnOpts struct {
	// Context is the base context to use.
	// If nil, context.Background is used.
	Context context.Context

	// BaseConfig optionally sets the base configuration
	// for values. If nil, defaults are used.
	BaseConfig *http.Server

	// Handler specifies which handler to use for processing
	// requests. If nil, BaseConfig.Handler is used. If BaseConfig
	// or BaseConfig.Handler is nil, http.DefaultServeMux is used.
	Handler http.Handler

	// UpgradeRequest is an initial request received on a connection
	// undergoing an h2c upgrade. The request body must have been
	// completely read from the connection before calling ServeConn,
	// and the 101 Switching Protocols response written.
	UpgradeRequest *http.Request

	// Settings is the decoded contents of the HTTP2-Settings header
	// in an h2c upgrade request.
	Settings []byte

	// SawClientPreface is set if the HTTP/2 connection preface
	// has already been read from the connection.
	SawClientPreface bool
}

// ServeConn serves HTTP/2 requests on the provided connection and
// blocks until the connection is no longer readable.
//
// ServeConn starts speaking HTTP/2 assuming that c has not had any
// reads or writes. It writes its initial settings frame and expects
// to be able to read the preface and settings frame from the
// client. If c has a ConnectionState method like a *tls.Conn, the
// ConnectionState is used to verify the TLS ciphersuite and to set
// the Request.TLS field in Handlers.
//
// ServeConn does not support h2c by itself. Any h2c support must be
// implemented in terms of providing a suitably-behaving net.Conn.
//
// The opts parameter is optional. If nil, default values are used.
func (s *Server) ServeConn(c net.Conn, opts *ServeConnOpts) {
	if opts == nil {
		opts = &ServeConnOpts{}
	}
	s.serveConn(c, opts, nil)
}

func (o *ServeConnOpts) context() context.Context {
	if o != nil && o.Context != nil {
		return o.Context
	}
	return context.Background()
}

func (o *ServeConnOpts) baseConfig() *http.Server {
	if o != nil && o.BaseConfig != nil {
		return o.BaseConfig
	}
	return new(http.Server)
}

func (o *ServeConnOpts) handler() http.Handler {
	if o != nil {
		if o.Handler != nil {
			return o.Handler
		}
		if o.BaseConfig != nil && o.BaseConfig.Handler != nil {
			return o.BaseConfig.Handler
		}
	}
	return http.DefaultServeMux
}

func serverConnBaseContext(c net.Conn, opts *ServeConnOpts) (ctx context.Context, cancel func()) {
	ctx, cancel = context.WithCancel(opts.context())
	ctx = context.WithValue(ctx, http.LocalAddrContextKey, c.LocalAddr())
	if hs := opts.baseConfig(); hs != nil {
		ctx = context.WithValue(ctx, http.ServerContextKey, hs)
	}
	return
}
