/*
 *
 * Copyright 2024 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package transport

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"

	"google.golang.org/grpc/mem"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// ServerStream implements streaming functionality for a gRPC server.
type ServerStream struct {
	*Stream // Embed for common stream functionality.

	st      internalServerTransport
	ctxDone <-chan struct{} // closed at the end of stream.  Cache of ctx.Done() (for performance)
	// cancel is invoked at the end of stream to cancel ctx. It also stops the
	// timer for monitoring the rpc deadline if configured.
	cancel func()

	// Holds compressor names passed in grpc-accept-encoding metadata from the
	// client.
	clientAdvertisedCompressors string
	headerWireLength            int

	// hdrMu protects outgoing header and trailer metadata.
	hdrMu      sync.Mutex
	header     metadata.MD // the outgoing header metadata.  Updated by WriteHeader.
	headerSent atomic.Bool // atomically set when the headers are sent out.
}

// Read reads an n byte message from the input stream.
func (s *ServerStream) Read(n int) (mem.BufferSlice, error) {
	b, err := s.Stream.read(n)
	if err == nil {
		s.st.incrMsgRecv()
	}
	return b, err
}

// SendHeader sends the header metadata for the given stream.
func (s *ServerStream) SendHeader(md metadata.MD) error {
	return s.st.writeHeader(s, md)
}

// Write writes the hdr and data bytes to the output stream.
func (s *ServerStream) Write(hdr []byte, data mem.BufferSlice, opts *WriteOptions) error {
	return s.st.write(s, hdr, data, opts)
}

// WriteStatus sends the status of a stream to the client.  WriteStatus is
// the final call made on a stream and always occurs.
func (s *ServerStream) WriteStatus(st *status.Status) error {
	return s.st.writeStatus(s, st)
}

// isHeaderSent indicates whether headers have been sent.
func (s *ServerStream) isHeaderSent() bool {
	return s.headerSent.Load()
}

// updateHeaderSent updates headerSent and returns true
// if it was already set.
func (s *ServerStream) updateHeaderSent() bool {
	return s.headerSent.Swap(true)
}

// RecvCompress returns the compression algorithm applied to the inbound
// message. It is empty string if there is no compression applied.
func (s *ServerStream) RecvCompress() string {
	return s.recvCompress
}

// SendCompress returns the send compressor name.
func (s *ServerStream) SendCompress() string {
	return s.sendCompress
}

// ContentSubtype returns the content-subtype for a request. For example, a
// content-subtype of "proto" will result in a content-type of
// "application/grpc+proto". This will always be lowercase.  See
// https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md#requests for
// more details.
func (s *ServerStream) ContentSubtype() string {
	return s.contentSubtype
}

// SetSendCompress sets the compression algorithm to the stream.
func (s *ServerStream) SetSendCompress(name string) error {
	if s.isHeaderSent() || s.getState() == streamDone {
		return errors.New("transport: set send compressor called after headers sent or stream done")
	}

	s.sendCompress = name
	return nil
}

// SetContext sets the context of the stream. This will be deleted once the
// stats handler callouts all move to gRPC layer.
func (s *ServerStream) SetContext(ctx context.Context) {
	s.ctx = ctx
}

// ClientAdvertisedCompressors returns the compressor names advertised by the
// client via grpc-accept-encoding header.
func (s *ServerStream) ClientAdvertisedCompressors() []string {
	values := strings.Split(s.clientAdvertisedCompressors, ",")
	for i, v := range values {
		values[i] = strings.TrimSpace(v)
	}
	return values
}

// Header returns the header metadata of the stream.  It returns the out header
// after t.WriteHeader is called.  It does not block and must not be called
// until after WriteHeader.
func (s *ServerStream) Header() (metadata.MD, error) {
	// Return the header in stream. It will be the out
	// header after t.WriteHeader is called.
	return s.header.Copy(), nil
}

// HeaderWireLength returns the size of the headers of the stream as received
// from the wire.
func (s *ServerStream) HeaderWireLength() int {
	return s.headerWireLength
}

// SetHeader sets the header metadata. This can be called multiple times.
// This should not be called in parallel to other data writes.
func (s *ServerStream) SetHeader(md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	if s.isHeaderSent() || s.getState() == streamDone {
		return ErrIllegalHeaderWrite
	}
	s.hdrMu.Lock()
	s.header = metadata.Join(s.header, md)
	s.hdrMu.Unlock()
	return nil
}

// SetTrailer sets the trailer metadata which will be sent with the RPC status
// by the server. This can be called multiple times.
// This should not be called parallel to other data writes.
func (s *ServerStream) SetTrailer(md metadata.MD) error {
	if md.Len() == 0 {
		return nil
	}
	if s.getState() == streamDone {
		return ErrIllegalHeaderWrite
	}
	s.hdrMu.Lock()
	s.trailer = metadata.Join(s.trailer, md)
	s.hdrMu.Unlock()
	return nil
}
