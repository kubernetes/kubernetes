// Copyright 2018, OpenCensus Authors
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

// Package b3 contains a propagation.HTTPFormat implementation
// for B3 propagation. See https://github.com/openzipkin/b3-propagation
// for more details.
package b3 // import "go.opencensus.io/plugin/ochttp/propagation/b3"

import (
	"encoding/hex"
	"net/http"

	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

// B3 headers that OpenCensus understands.
const (
	TraceIDHeader = "X-B3-TraceId"
	SpanIDHeader  = "X-B3-SpanId"
	SampledHeader = "X-B3-Sampled"
)

// HTTPFormat implements propagation.HTTPFormat to propagate
// traces in HTTP headers in B3 propagation format.
// HTTPFormat skips the X-B3-ParentId and X-B3-Flags headers
// because there are additional fields not represented in the
// OpenCensus span context. Spans created from the incoming
// header will be the direct children of the client-side span.
// Similarly, receiver of the outgoing spans should use client-side
// span created by OpenCensus as the parent.
type HTTPFormat struct{}

var _ propagation.HTTPFormat = (*HTTPFormat)(nil)

// SpanContextFromRequest extracts a B3 span context from incoming requests.
func (f *HTTPFormat) SpanContextFromRequest(req *http.Request) (sc trace.SpanContext, ok bool) {
	tid, ok := ParseTraceID(req.Header.Get(TraceIDHeader))
	if !ok {
		return trace.SpanContext{}, false
	}
	sid, ok := ParseSpanID(req.Header.Get(SpanIDHeader))
	if !ok {
		return trace.SpanContext{}, false
	}
	sampled, _ := ParseSampled(req.Header.Get(SampledHeader))
	return trace.SpanContext{
		TraceID:      tid,
		SpanID:       sid,
		TraceOptions: sampled,
	}, true
}

// ParseTraceID parses the value of the X-B3-TraceId header.
func ParseTraceID(tid string) (trace.TraceID, bool) {
	if tid == "" {
		return trace.TraceID{}, false
	}
	b, err := hex.DecodeString(tid)
	if err != nil {
		return trace.TraceID{}, false
	}
	var traceID trace.TraceID
	if len(b) <= 8 {
		// The lower 64-bits.
		start := 8 + (8 - len(b))
		copy(traceID[start:], b)
	} else {
		start := 16 - len(b)
		copy(traceID[start:], b)
	}

	return traceID, true
}

// ParseSpanID parses the value of the X-B3-SpanId or X-B3-ParentSpanId headers.
func ParseSpanID(sid string) (spanID trace.SpanID, ok bool) {
	if sid == "" {
		return trace.SpanID{}, false
	}
	b, err := hex.DecodeString(sid)
	if err != nil {
		return trace.SpanID{}, false
	}
	start := 8 - len(b)
	copy(spanID[start:], b)
	return spanID, true
}

// ParseSampled parses the value of the X-B3-Sampled header.
func ParseSampled(sampled string) (trace.TraceOptions, bool) {
	switch sampled {
	case "true", "1":
		return trace.TraceOptions(1), true
	default:
		return trace.TraceOptions(0), false
	}
}

// SpanContextToRequest modifies the given request to include B3 headers.
func (f *HTTPFormat) SpanContextToRequest(sc trace.SpanContext, req *http.Request) {
	req.Header.Set(TraceIDHeader, hex.EncodeToString(sc.TraceID[:]))
	req.Header.Set(SpanIDHeader, hex.EncodeToString(sc.SpanID[:]))

	var sampled string
	if sc.IsSampled() {
		sampled = "1"
	} else {
		sampled = "0"
	}
	req.Header.Set(SampledHeader, sampled)
}
