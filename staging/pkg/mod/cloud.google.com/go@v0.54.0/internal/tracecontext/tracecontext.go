// Copyright 2017 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package tracecontext provides encoders and decoders for Stackdriver Trace contexts.
package tracecontext

import "encoding/binary"

const (
	versionID    = 0
	traceIDField = 0
	spanIDField  = 1
	optsField    = 2

	traceIDLen = 16
	spanIDLen  = 8
	optsLen    = 1

	// Len represents the length of trace context.
	Len = 1 + 1 + traceIDLen + 1 + spanIDLen + 1 + optsLen
)

// Encode encodes trace ID, span ID and options into dst. The number of bytes
// written will be returned. If len(dst) isn't big enough to fit the trace context,
// a negative number is returned.
func Encode(dst []byte, traceID []byte, spanID uint64, opts byte) (n int) {
	if len(dst) < Len {
		return -1
	}
	var offset = 0
	putByte := func(b byte) { dst[offset] = b; offset++ }
	putUint64 := func(u uint64) { binary.LittleEndian.PutUint64(dst[offset:], u); offset += 8 }

	putByte(versionID)
	putByte(traceIDField)
	for _, b := range traceID {
		putByte(b)
	}
	putByte(spanIDField)
	putUint64(spanID)
	putByte(optsField)
	putByte(opts)

	return offset
}

// Decode decodes the src into a trace ID, span ID and options. If src doesn't
// contain a valid trace context, ok = false is returned.
func Decode(src []byte) (traceID []byte, spanID uint64, opts byte, ok bool) {
	if len(src) < Len {
		return traceID, spanID, 0, false
	}
	var offset = 0
	readByte := func() byte { b := src[offset]; offset++; return b }
	readUint64 := func() uint64 { v := binary.LittleEndian.Uint64(src[offset:]); offset += 8; return v }

	if readByte() != versionID {
		return traceID, spanID, 0, false
	}
	for offset < len(src) {
		switch readByte() {
		case traceIDField:
			traceID = src[offset : offset+traceIDLen]
			offset += traceIDLen
		case spanIDField:
			spanID = readUint64()
		case optsField:
			opts = readByte()
		}
	}
	return traceID, spanID, opts, true
}
