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

// Package propagation implement X-Cloud-Trace-Context header propagation used
// by Google Cloud products.
package propagation // import "go.opencensus.io/exporter/stackdriver/propagation"

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

const (
	httpHeaderMaxSize = 200
	httpHeader        = `X-Cloud-Trace-Context`
)

var _ propagation.HTTPFormat = (*HTTPFormat)(nil)

// HTTPFormat implements propagation.HTTPFormat to propagate
// traces in HTTP headers for Google Cloud Platform and Stackdriver Trace.
type HTTPFormat struct{}

// SpanContextFromRequest extracts a Stackdriver Trace span context from incoming requests.
func (f *HTTPFormat) SpanContextFromRequest(req *http.Request) (sc trace.SpanContext, ok bool) {
	h := req.Header.Get(httpHeader)
	// See https://cloud.google.com/trace/docs/faq for the header HTTPFormat.
	// Return if the header is empty or missing, or if the header is unreasonably
	// large, to avoid making unnecessary copies of a large string.
	if h == "" || len(h) > httpHeaderMaxSize {
		return trace.SpanContext{}, false
	}

	// Parse the trace id field.
	slash := strings.Index(h, `/`)
	if slash == -1 {
		return trace.SpanContext{}, false
	}
	tid, h := h[:slash], h[slash+1:]

	buf, err := hex.DecodeString(tid)
	if err != nil {
		return trace.SpanContext{}, false
	}
	copy(sc.TraceID[:], buf)

	// Parse the span id field.
	spanstr := h
	semicolon := strings.Index(h, `;`)
	if semicolon != -1 {
		spanstr, h = h[:semicolon], h[semicolon+1:]
	}
	sid, err := strconv.ParseUint(spanstr, 10, 64)
	if err != nil {
		return trace.SpanContext{}, false
	}
	binary.BigEndian.PutUint64(sc.SpanID[:], sid)

	// Parse the options field, options field is optional.
	if !strings.HasPrefix(h, "o=") {
		return sc, true
	}
	o, err := strconv.ParseUint(h[2:], 10, 64)
	if err != nil {
		return trace.SpanContext{}, false
	}
	sc.TraceOptions = trace.TraceOptions(o)
	return sc, true
}

// SpanContextToRequest modifies the given request to include a Stackdriver Trace header.
func (f *HTTPFormat) SpanContextToRequest(sc trace.SpanContext, req *http.Request) {
	sid := binary.BigEndian.Uint64(sc.SpanID[:])
	header := fmt.Sprintf("%s/%d;o=%d", hex.EncodeToString(sc.TraceID[:]), sid, int64(sc.TraceOptions))
	req.Header.Set(httpHeader, header)
}
