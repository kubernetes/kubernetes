// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.27 && !http2legacy

// Server wrapping a net/http.Server.

package http2

import (
	"context"
	"errors"
	"net"
	"net/http"
	"sync"
	"time"
)

type serverInternalState struct {
	s1            *http.Server
	initOnce      sync.Once
	serveConnFunc func(context.Context, net.Conn, http.Handler, bool, *http.Request, []byte)
}

func configureServer(s *http.Server, conf *Server) error {
	if s == nil {
		panic("nil *http.Server")
	}
	if conf == nil {
		conf = new(Server)
	}
	if conf.state != nil {
		// This isn't a panic in the pre-wrapping implementation,
		// but calling ConfigureServer twice with the same http2.Server
		// overwrites internal state on the server.
		// Make the error explicit and early here.
		panic("ConfigureServer may be called only once per Server")
	}
	if h1, h2 := s, conf; h2.IdleTimeout == 0 {
		if h1.IdleTimeout != 0 {
			h2.IdleTimeout = h1.IdleTimeout
		} else {
			h2.IdleTimeout = h1.ReadTimeout
		}
	}
	conf.state = &serverInternalState{
		s1: s,
	}
	sconfig := &serverConfig{s: conf}
	if err := s.Serve(sconfig); err != nil || sconfig.serveConnFunc == nil {
		panic("http2: net/http does not support this version of x/net/http2")
	}
	conf.state.serveConnFunc = sconfig.serveConnFunc
	return nil
}

type serverConfig struct {
	s             *Server
	serveConnFunc func(context.Context, net.Conn, http.Handler, bool, *http.Request, []byte)
}

func (*serverConfig) Accept() (net.Conn, error) {
	return nil, errors.New("unexpected call to Accept")
}
func (*serverConfig) Close() error {
	return nil
}
func (*serverConfig) Addr() net.Addr {
	return nil
}

func (s *serverConfig) ServeConnFunc(f func(context.Context, net.Conn, http.Handler, bool, *http.Request, []byte)) {
	s.serveConnFunc = f
}

func (s *serverConfig) HTTP2Config() http.HTTP2Config {
	return http.HTTP2Config{
		MaxConcurrentStreams:          int(s.s.MaxConcurrentStreams),
		MaxDecoderHeaderTableSize:     int(s.s.MaxDecoderHeaderTableSize),
		MaxEncoderHeaderTableSize:     int(s.s.MaxEncoderHeaderTableSize),
		MaxReadFrameSize:              int(s.s.MaxReadFrameSize),
		PermitProhibitedCipherSuites:  s.s.PermitProhibitedCipherSuites,
		MaxReceiveBufferPerConnection: int(s.s.MaxUploadBufferPerConnection),
		MaxReceiveBufferPerStream:     int(s.s.MaxUploadBufferPerStream),
		SendPingTimeout:               s.s.ReadIdleTimeout,
		PingTimeout:                   s.s.PingTimeout,
		WriteByteTimeout:              s.s.WriteByteTimeout,
		CountError:                    s.s.CountError,
	}
}

func (s *serverConfig) IdleTimeout() time.Duration {
	return s.s.IdleTimeout
}

type serverConn struct{}

func (s *Server) serveConn(c net.Conn, opts *ServeConnOpts, _ func(*serverConn)) {
	var serveConnFunc func(context.Context, net.Conn, http.Handler, bool, *http.Request, []byte)
	switch {
	case opts.BaseConfig != nil:
		// The user has provided us with an http.Server to take configuration from.
		//
		// We can't send our request to opts.BaseConfig, because an http.Server can
		// only be associated with a single http2.Server and the user might
		// use this one with several http.Servers.
		//
		// We can't send our request to s.state.s1, because it doesn't contain
		// the right configuration.
		//
		// So create a one-off copy of opts.BaseConfig and use it.
		h1 := &http.Server{
			TLSConfig:         opts.BaseConfig.TLSConfig,
			ReadTimeout:       opts.BaseConfig.ReadTimeout,
			ReadHeaderTimeout: opts.BaseConfig.ReadHeaderTimeout,
			WriteTimeout:      opts.BaseConfig.WriteTimeout,
			IdleTimeout:       opts.BaseConfig.IdleTimeout,
			MaxHeaderBytes:    opts.BaseConfig.MaxHeaderBytes,
			ConnState:         opts.BaseConfig.ConnState,
			ErrorLog:          opts.BaseConfig.ErrorLog,
			BaseContext:       opts.BaseConfig.BaseContext,
			ConnContext:       opts.BaseConfig.ConnContext,
			HTTP2:             opts.BaseConfig.HTTP2,
		}
		sconfig := &serverConfig{s: s}
		if err := h1.Serve(sconfig); err != nil || sconfig.serveConnFunc == nil {
			panic("http2: net/http does not support this version of x/net/http2")
		}
		serveConnFunc = sconfig.serveConnFunc
	case s.state != nil:
		serveConnFunc = s.state.serveConnFunc
	default:
		// Strange-but-true: Server has no concurrency-safe way to initialize
		// its internal state, so historically ServeConn just doesn't use any
		// persistent state if you don't call ConfigureServer first.
		//
		// If ConfigureServer hasn't been called, create a one-off http.Server
		// for the connection, since we don't have any way to keep one around for reuse.
		h1 := &http.Server{}
		sconfig := &serverConfig{s: s}
		if err := h1.Serve(sconfig); err != nil || sconfig.serveConnFunc == nil {
			panic("http2: net/http does not support this version of x/net/http2")
		}
		serveConnFunc = sconfig.serveConnFunc
	}

	ctx, cancel := serverConnBaseContext(c, opts)
	defer cancel()
	serveConnFunc(ctx, c, opts.handler(), opts.SawClientPreface, opts.UpgradeRequest, opts.Settings)

}

// FrameWriteRequest is a request to write a frame.
//
// Deprecated: User-provided write schedulers are deprecated.
type FrameWriteRequest struct {
	// Ideally we'd define this in writesched_common.go,
	// to avoid duplicating an exported symbol across two files,
	// but the changes required to make this work are fairly large.
}
