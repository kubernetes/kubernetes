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

package b3

import (
	"net/http"
	"reflect"
	"testing"

	"go.opencensus.io/trace"
)

func TestHTTPFormat_FromRequest(t *testing.T) {
	tests := []struct {
		name    string
		makeReq func() *http.Request
		wantSc  trace.SpanContext
		wantOk  bool
	}{
		{
			name: "128-bit trace ID + 64-bit span ID; sampled=1",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "463ac35c9f6413ad48485a3953bb6124")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				req.Header.Set(SampledHeader, "1")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(1),
			},
			wantOk: true,
		},
		{
			name: "short trace ID + short span ID; sampled=1",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "000102")
				req.Header.Set(SpanIDHeader, "000102")
				req.Header.Set(SampledHeader, "1")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2},
				SpanID:       trace.SpanID{0, 0, 0, 0, 0, 0, 1, 2},
				TraceOptions: trace.TraceOptions(1),
			},
			wantOk: true,
		},
		{
			name: "64-bit trace ID + 64-bit span ID; sampled=0",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "0020000000000001")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				req.Header.Set(SampledHeader, "0")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 0, 0, 0, 0, 1},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(0),
			},
			wantOk: true,
		},
		{
			name: "128-bit trace ID + 64-bit span ID; no sampling header",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "463ac35c9f6413ad48485a3953bb6124")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(0),
			},
			wantOk: true,
		},
		{
			name: "invalid trace ID + 64-bit span ID; no sampling header",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				return req
			},
			wantSc: trace.SpanContext{},
			wantOk: false,
		},
		{
			name: "128-bit trace ID; invalid span ID; no sampling header",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "463ac35c9f6413ad48485a3953bb6124")
				req.Header.Set(SpanIDHeader, "")
				return req
			},
			wantSc: trace.SpanContext{},
			wantOk: false,
		},
		{
			name: "128-bit trace ID + 64-bit span ID; sampled=true",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "463ac35c9f6413ad48485a3953bb6124")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				req.Header.Set(SampledHeader, "true")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(1),
			},
			wantOk: true,
		},
		{
			name: "128-bit trace ID + 64-bit span ID; sampled=false",
			makeReq: func() *http.Request {
				req, _ := http.NewRequest("GET", "http://example.com", nil)
				req.Header.Set(TraceIDHeader, "463ac35c9f6413ad48485a3953bb6124")
				req.Header.Set(SpanIDHeader, "0020000000000001")
				req.Header.Set(SampledHeader, "false")
				return req
			},
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(0),
			},
			wantOk: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &HTTPFormat{}
			sc, ok := f.SpanContextFromRequest(tt.makeReq())
			if ok != tt.wantOk {
				t.Errorf("HTTPFormat.SpanContextFromRequest() got ok = %v, want %v", ok, tt.wantOk)
			}
			if !reflect.DeepEqual(sc, tt.wantSc) {
				t.Errorf("HTTPFormat.SpanContextFromRequest() got span context = %v, want %v", sc, tt.wantSc)
			}
		})
	}
}

func TestHTTPFormat_ToRequest(t *testing.T) {
	tests := []struct {
		name        string
		sc          trace.SpanContext
		wantHeaders map[string]string
	}{
		{
			name: "valid traceID, header ID, sampled=1",
			sc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(1),
			},
			wantHeaders: map[string]string{
				"X-B3-TraceId": "463ac35c9f6413ad48485a3953bb6124",
				"X-B3-SpanId":  "0020000000000001",
				"X-B3-Sampled": "1",
			},
		},
		{
			name: "valid traceID, header ID, sampled=0",
			sc: trace.SpanContext{
				TraceID:      trace.TraceID{70, 58, 195, 92, 159, 100, 19, 173, 72, 72, 90, 57, 83, 187, 97, 36},
				SpanID:       trace.SpanID{0, 32, 0, 0, 0, 0, 0, 1},
				TraceOptions: trace.TraceOptions(0),
			},
			wantHeaders: map[string]string{
				"X-B3-TraceId": "463ac35c9f6413ad48485a3953bb6124",
				"X-B3-SpanId":  "0020000000000001",
				"X-B3-Sampled": "0",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &HTTPFormat{}
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			f.SpanContextToRequest(tt.sc, req)

			for k, v := range tt.wantHeaders {
				if got, want := req.Header.Get(k), v; got != want {
					t.Errorf("req.Header.Get(%q) = %q; want %q", k, got, want)
				}
			}
		})
	}
}
