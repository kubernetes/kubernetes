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

// Package tracecontext contains HTTP propagator for TraceContext standard.
// See https://github.com/w3c/distributed-tracing for more information.
package tracecontext // import "go.opencensus.io/plugin/ochttp/propagation/tracecontext"

import (
	"encoding/hex"
	"fmt"
	"net/http"
	"net/textproto"
	"regexp"
	"strings"

	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
	"go.opencensus.io/trace/tracestate"
)

const (
	supportedVersion  = 0
	maxVersion        = 254
	maxTracestateLen  = 512
	traceparentHeader = "traceparent"
	tracestateHeader  = "tracestate"
	trimOWSRegexFmt   = `^[\x09\x20]*(.*[^\x20\x09])[\x09\x20]*$`
)

var trimOWSRegExp = regexp.MustCompile(trimOWSRegexFmt)

var _ propagation.HTTPFormat = (*HTTPFormat)(nil)

// HTTPFormat implements the TraceContext trace propagation format.
type HTTPFormat struct{}

// SpanContextFromRequest extracts a span context from incoming requests.
func (f *HTTPFormat) SpanContextFromRequest(req *http.Request) (sc trace.SpanContext, ok bool) {
	h, ok := getRequestHeader(req, traceparentHeader, false)
	if !ok {
		return trace.SpanContext{}, false
	}
	sections := strings.Split(h, "-")
	if len(sections) < 4 {
		return trace.SpanContext{}, false
	}

	if len(sections[0]) != 2 {
		return trace.SpanContext{}, false
	}
	ver, err := hex.DecodeString(sections[0])
	if err != nil {
		return trace.SpanContext{}, false
	}
	version := int(ver[0])
	if version > maxVersion {
		return trace.SpanContext{}, false
	}

	if version == 0 && len(sections) != 4 {
		return trace.SpanContext{}, false
	}

	if len(sections[1]) != 32 {
		return trace.SpanContext{}, false
	}
	tid, err := hex.DecodeString(sections[1])
	if err != nil {
		return trace.SpanContext{}, false
	}
	copy(sc.TraceID[:], tid)

	if len(sections[2]) != 16 {
		return trace.SpanContext{}, false
	}
	sid, err := hex.DecodeString(sections[2])
	if err != nil {
		return trace.SpanContext{}, false
	}
	copy(sc.SpanID[:], sid)

	opts, err := hex.DecodeString(sections[3])
	if err != nil || len(opts) < 1 {
		return trace.SpanContext{}, false
	}
	sc.TraceOptions = trace.TraceOptions(opts[0])

	// Don't allow all zero trace or span ID.
	if sc.TraceID == [16]byte{} || sc.SpanID == [8]byte{} {
		return trace.SpanContext{}, false
	}

	sc.Tracestate = tracestateFromRequest(req)
	return sc, true
}

// getRequestHeader returns a combined header field according to RFC7230 section 3.2.2.
// If commaSeparated is true, multiple header fields with the same field name using be
// combined using ",".
// If no header was found using the given name, "ok" would be false.
// If more than one headers was found using the given name, while commaSeparated is false,
// "ok" would be false.
func getRequestHeader(req *http.Request, name string, commaSeparated bool) (hdr string, ok bool) {
	v := req.Header[textproto.CanonicalMIMEHeaderKey(name)]
	switch len(v) {
	case 0:
		return "", false
	case 1:
		return v[0], true
	default:
		return strings.Join(v, ","), commaSeparated
	}
}

// TODO(rghetia): return an empty Tracestate when parsing tracestate header encounters an error.
// Revisit to return additional boolean value to indicate parsing error when following issues
// are resolved.
// https://github.com/w3c/distributed-tracing/issues/172
// https://github.com/w3c/distributed-tracing/issues/175
func tracestateFromRequest(req *http.Request) *tracestate.Tracestate {
	h, _ := getRequestHeader(req, tracestateHeader, true)
	if h == "" {
		return nil
	}

	var entries []tracestate.Entry
	pairs := strings.Split(h, ",")
	hdrLenWithoutOWS := len(pairs) - 1 // Number of commas
	for _, pair := range pairs {
		matches := trimOWSRegExp.FindStringSubmatch(pair)
		if matches == nil {
			return nil
		}
		pair = matches[1]
		hdrLenWithoutOWS += len(pair)
		if hdrLenWithoutOWS > maxTracestateLen {
			return nil
		}
		kv := strings.Split(pair, "=")
		if len(kv) != 2 {
			return nil
		}
		entries = append(entries, tracestate.Entry{Key: kv[0], Value: kv[1]})
	}
	ts, err := tracestate.New(nil, entries...)
	if err != nil {
		return nil
	}

	return ts
}

func tracestateToRequest(sc trace.SpanContext, req *http.Request) {
	var pairs = make([]string, 0, len(sc.Tracestate.Entries()))
	if sc.Tracestate != nil {
		for _, entry := range sc.Tracestate.Entries() {
			pairs = append(pairs, strings.Join([]string{entry.Key, entry.Value}, "="))
		}
		h := strings.Join(pairs, ",")

		if h != "" && len(h) <= maxTracestateLen {
			req.Header.Set(tracestateHeader, h)
		}
	}
}

// SpanContextToRequest modifies the given request to include traceparent and tracestate headers.
func (f *HTTPFormat) SpanContextToRequest(sc trace.SpanContext, req *http.Request) {
	h := fmt.Sprintf("%x-%x-%x-%x",
		[]byte{supportedVersion},
		sc.TraceID[:],
		sc.SpanID[:],
		[]byte{byte(sc.TraceOptions)})
	req.Header.Set(traceparentHeader, h)
	tracestateToRequest(sc, req)
}
