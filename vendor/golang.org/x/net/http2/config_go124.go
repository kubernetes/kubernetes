// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.24

package http2

import "net/http"

// fillNetHTTPServerConfig sets fields in conf from srv.HTTP2.
func fillNetHTTPServerConfig(conf *http2Config, srv *http.Server) {
	fillNetHTTPConfig(conf, srv.HTTP2)
}

// fillNetHTTPServerConfig sets fields in conf from tr.HTTP2.
func fillNetHTTPTransportConfig(conf *http2Config, tr *http.Transport) {
	fillNetHTTPConfig(conf, tr.HTTP2)
}

func fillNetHTTPConfig(conf *http2Config, h2 *http.HTTP2Config) {
	if h2 == nil {
		return
	}
	if h2.MaxConcurrentStreams != 0 {
		conf.MaxConcurrentStreams = uint32(h2.MaxConcurrentStreams)
	}
	if h2.MaxEncoderHeaderTableSize != 0 {
		conf.MaxEncoderHeaderTableSize = uint32(h2.MaxEncoderHeaderTableSize)
	}
	if h2.MaxDecoderHeaderTableSize != 0 {
		conf.MaxDecoderHeaderTableSize = uint32(h2.MaxDecoderHeaderTableSize)
	}
	if h2.MaxConcurrentStreams != 0 {
		conf.MaxConcurrentStreams = uint32(h2.MaxConcurrentStreams)
	}
	if h2.MaxReadFrameSize != 0 {
		conf.MaxReadFrameSize = uint32(h2.MaxReadFrameSize)
	}
	if h2.MaxReceiveBufferPerConnection != 0 {
		conf.MaxUploadBufferPerConnection = int32(h2.MaxReceiveBufferPerConnection)
	}
	if h2.MaxReceiveBufferPerStream != 0 {
		conf.MaxUploadBufferPerStream = int32(h2.MaxReceiveBufferPerStream)
	}
	if h2.SendPingTimeout != 0 {
		conf.SendPingTimeout = h2.SendPingTimeout
	}
	if h2.PingTimeout != 0 {
		conf.PingTimeout = h2.PingTimeout
	}
	if h2.WriteByteTimeout != 0 {
		conf.WriteByteTimeout = h2.WriteByteTimeout
	}
	if h2.PermitProhibitedCipherSuites {
		conf.PermitProhibitedCipherSuites = true
	}
	if h2.CountError != nil {
		conf.CountError = h2.CountError
	}
}
