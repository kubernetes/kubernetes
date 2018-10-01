// Copyright 2017, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package propagation implements the binary trace context format.
package propagation

// TODO: link to external spec document.

// BinaryFormat format:
//
// Binary value: <version_id><version_format>
// version_id: 1 byte representing the version id.
//
// For version_id = 0:
//
// version_format: <field><field>
// field_format: <field_id><field_format>
//
// Fields:
//
// TraceId: (field_id = 0, len = 16, default = "0000000000000000") - 16-byte array representing the trace_id.
// SpanId: (field_id = 1, len = 8, default = "00000000") - 8-byte array representing the span_id.
// TraceOptions: (field_id = 2, len = 1, default = "0") - 1-byte array representing the trace_options.
//
// Fields MUST be encoded using the field id order (smaller to higher).
//
// Valid value example:
//
// {0, 0, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 97,
// 98, 99, 100, 101, 102, 103, 104, 2, 1}
//
// version_id = 0;
// trace_id = {64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79}
// span_id = {97, 98, 99, 100, 101, 102, 103, 104};
// trace_options = {1};

import (
	"net/http"

	"go.opencensus.io/trace"
)

// Binary returns the binary format representation of a SpanContext.
//
// If sc is the zero value, Binary returns nil.
func Binary(sc trace.SpanContext) []byte {
	if sc == (trace.SpanContext{}) {
		return nil
	}
	var b [29]byte
	copy(b[2:18], sc.TraceID[:])
	b[18] = 1
	copy(b[19:27], sc.SpanID[:])
	b[27] = 2
	b[28] = uint8(sc.TraceOptions)
	return b[:]
}

// FromBinary returns the SpanContext represented by b.
//
// If b has an unsupported version ID or contains no TraceID, FromBinary
// returns with ok==false.
func FromBinary(b []byte) (sc trace.SpanContext, ok bool) {
	if len(b) == 0 || b[0] != 0 {
		return trace.SpanContext{}, false
	}
	b = b[1:]
	if len(b) >= 17 && b[0] == 0 {
		copy(sc.TraceID[:], b[1:17])
		b = b[17:]
	} else {
		return trace.SpanContext{}, false
	}
	if len(b) >= 9 && b[0] == 1 {
		copy(sc.SpanID[:], b[1:9])
		b = b[9:]
	}
	if len(b) >= 2 && b[0] == 2 {
		sc.TraceOptions = trace.TraceOptions(b[1])
	}
	return sc, true
}

// HTTPFormat implementations propagate span contexts
// in HTTP requests.
//
// SpanContextFromRequest extracts a span context from incoming
// requests.
//
// SpanContextToRequest modifies the given request to include the given
// span context.
type HTTPFormat interface {
	SpanContextFromRequest(req *http.Request) (sc trace.SpanContext, ok bool)
	SpanContextToRequest(sc trace.SpanContext, req *http.Request)
}

// TODO(jbd): Find a more representative but short name for HTTPFormat.
