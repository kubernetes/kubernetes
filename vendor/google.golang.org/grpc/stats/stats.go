/*
 *
 * Copyright 2016, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Package stats is for collecting and reporting various network and RPC stats.
// This package is for monitoring purpose only. All fields are read-only.
// All APIs are experimental.
package stats // import "google.golang.org/grpc/stats"

import (
	"net"
	"time"
)

// RPCStats contains stats information about RPCs.
type RPCStats interface {
	isRPCStats()
	// IsClient returns true if this RPCStats is from client side.
	IsClient() bool
}

// Begin contains stats when an RPC begins.
// FailFast are only valid if Client is true.
type Begin struct {
	// Client is true if this Begin is from client side.
	Client bool
	// BeginTime is the time when the RPC begins.
	BeginTime time.Time
	// FailFast indicates if this RPC is failfast.
	FailFast bool
}

// IsClient indicates if this is from client side.
func (s *Begin) IsClient() bool { return s.Client }

func (s *Begin) isRPCStats() {}

// InPayload contains the information for an incoming payload.
type InPayload struct {
	// Client is true if this InPayload is from client side.
	Client bool
	// Payload is the payload with original type.
	Payload interface{}
	// Data is the serialized message payload.
	Data []byte
	// Length is the length of uncompressed data.
	Length int
	// WireLength is the length of data on wire (compressed, signed, encrypted).
	WireLength int
	// RecvTime is the time when the payload is received.
	RecvTime time.Time
}

// IsClient indicates if this is from client side.
func (s *InPayload) IsClient() bool { return s.Client }

func (s *InPayload) isRPCStats() {}

// InHeader contains stats when a header is received.
// FullMethod, addresses and Compression are only valid if Client is false.
type InHeader struct {
	// Client is true if this InHeader is from client side.
	Client bool
	// WireLength is the wire length of header.
	WireLength int

	// FullMethod is the full RPC method string, i.e., /package.service/method.
	FullMethod string
	// RemoteAddr is the remote address of the corresponding connection.
	RemoteAddr net.Addr
	// LocalAddr is the local address of the corresponding connection.
	LocalAddr net.Addr
	// Compression is the compression algorithm used for the RPC.
	Compression string
}

// IsClient indicates if this is from client side.
func (s *InHeader) IsClient() bool { return s.Client }

func (s *InHeader) isRPCStats() {}

// InTrailer contains stats when a trailer is received.
type InTrailer struct {
	// Client is true if this InTrailer is from client side.
	Client bool
	// WireLength is the wire length of trailer.
	WireLength int
}

// IsClient indicates if this is from client side.
func (s *InTrailer) IsClient() bool { return s.Client }

func (s *InTrailer) isRPCStats() {}

// OutPayload contains the information for an outgoing payload.
type OutPayload struct {
	// Client is true if this OutPayload is from client side.
	Client bool
	// Payload is the payload with original type.
	Payload interface{}
	// Data is the serialized message payload.
	Data []byte
	// Length is the length of uncompressed data.
	Length int
	// WireLength is the length of data on wire (compressed, signed, encrypted).
	WireLength int
	// SentTime is the time when the payload is sent.
	SentTime time.Time
}

// IsClient indicates if this is from client side.
func (s *OutPayload) IsClient() bool { return s.Client }

func (s *OutPayload) isRPCStats() {}

// OutHeader contains stats when a header is sent.
// FullMethod, addresses and Compression are only valid if Client is true.
type OutHeader struct {
	// Client is true if this OutHeader is from client side.
	Client bool
	// WireLength is the wire length of header.
	WireLength int

	// FullMethod is the full RPC method string, i.e., /package.service/method.
	FullMethod string
	// RemoteAddr is the remote address of the corresponding connection.
	RemoteAddr net.Addr
	// LocalAddr is the local address of the corresponding connection.
	LocalAddr net.Addr
	// Compression is the compression algorithm used for the RPC.
	Compression string
}

// IsClient indicates if this is from client side.
func (s *OutHeader) IsClient() bool { return s.Client }

func (s *OutHeader) isRPCStats() {}

// OutTrailer contains stats when a trailer is sent.
type OutTrailer struct {
	// Client is true if this OutTrailer is from client side.
	Client bool
	// WireLength is the wire length of trailer.
	WireLength int
}

// IsClient indicates if this is from client side.
func (s *OutTrailer) IsClient() bool { return s.Client }

func (s *OutTrailer) isRPCStats() {}

// End contains stats when an RPC ends.
type End struct {
	// Client is true if this End is from client side.
	Client bool
	// EndTime is the time when the RPC ends.
	EndTime time.Time
	// Error is the error just happened.  It implements status.Status if non-nil.
	Error error
}

// IsClient indicates if this is from client side.
func (s *End) IsClient() bool { return s.Client }

func (s *End) isRPCStats() {}

// ConnStats contains stats information about connections.
type ConnStats interface {
	isConnStats()
	// IsClient returns true if this ConnStats is from client side.
	IsClient() bool
}

// ConnBegin contains the stats of a connection when it is established.
type ConnBegin struct {
	// Client is true if this ConnBegin is from client side.
	Client bool
}

// IsClient indicates if this is from client side.
func (s *ConnBegin) IsClient() bool { return s.Client }

func (s *ConnBegin) isConnStats() {}

// ConnEnd contains the stats of a connection when it ends.
type ConnEnd struct {
	// Client is true if this ConnEnd is from client side.
	Client bool
}

// IsClient indicates if this is from client side.
func (s *ConnEnd) IsClient() bool { return s.Client }

func (s *ConnEnd) isConnStats() {}
