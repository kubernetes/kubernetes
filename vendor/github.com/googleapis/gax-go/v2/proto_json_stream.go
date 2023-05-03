// Copyright 2022, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package gax

import (
	"encoding/json"
	"errors"
	"io"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

var (
	arrayOpen     = json.Delim('[')
	arrayClose    = json.Delim(']')
	errBadOpening = errors.New("unexpected opening token, expected '['")
)

// ProtoJSONStream represents a wrapper for consuming a stream of protobuf
// messages encoded using protobuf-JSON format. More information on this format
// can be found at https://developers.google.com/protocol-buffers/docs/proto3#json.
// The stream must appear as a comma-delimited, JSON array of obbjects with
// opening and closing square braces.
//
// This is for internal use only.
type ProtoJSONStream struct {
	first, closed bool
	reader        io.ReadCloser
	stream        *json.Decoder
	typ           protoreflect.MessageType
}

// NewProtoJSONStreamReader accepts a stream of bytes via an io.ReadCloser that are
// protobuf-JSON encoded protobuf messages of the given type. The ProtoJSONStream
// must be closed when done.
//
// This is for internal use only.
func NewProtoJSONStreamReader(rc io.ReadCloser, typ protoreflect.MessageType) *ProtoJSONStream {
	return &ProtoJSONStream{
		first:  true,
		reader: rc,
		stream: json.NewDecoder(rc),
		typ:    typ,
	}
}

// Recv decodes the next protobuf message in the stream or returns io.EOF if
// the stream is done. It is not safe to call Recv on the same stream from
// different goroutines, just like it is not safe to do so with a single gRPC
// stream. Type-cast the protobuf message returned to the type provided at
// ProtoJSONStream creation.
// Calls to Recv after calling Close will produce io.EOF.
func (s *ProtoJSONStream) Recv() (proto.Message, error) {
	if s.closed {
		return nil, io.EOF
	}
	if s.first {
		s.first = false

		// Consume the opening '[' so Decode gets one object at a time.
		if t, err := s.stream.Token(); err != nil {
			return nil, err
		} else if t != arrayOpen {
			return nil, errBadOpening
		}
	}

	// Capture the next block of data for the item (a JSON object) in the stream.
	var raw json.RawMessage
	if err := s.stream.Decode(&raw); err != nil {
		e := err
		// To avoid checking the first token of each stream, just attempt to
		// Decode the next blob and if that fails, double check if it is just
		// the closing token ']'. If it is the closing, return io.EOF. If it
		// isn't, return the original error.
		if t, _ := s.stream.Token(); t == arrayClose {
			e = io.EOF
		}
		return nil, e
	}

	// Initialize a new instance of the protobuf message to unmarshal the
	// raw data into.
	m := s.typ.New().Interface()
	err := protojson.Unmarshal(raw, m)

	return m, err
}

// Close closes the stream so that resources are cleaned up.
func (s *ProtoJSONStream) Close() error {
	// Dereference the *json.Decoder so that the memory is gc'd.
	s.stream = nil
	s.closed = true

	return s.reader.Close()
}
