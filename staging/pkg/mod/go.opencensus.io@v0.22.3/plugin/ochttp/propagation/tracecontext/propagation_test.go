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

package tracecontext

import (
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"go.opencensus.io/trace"
	"go.opencensus.io/trace/tracestate"
)

var (
	tpHeader        = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	traceID         = trace.TraceID{75, 249, 47, 53, 119, 179, 77, 166, 163, 206, 146, 157, 14, 14, 71, 54}
	spanID          = trace.SpanID{0, 240, 103, 170, 11, 169, 2, 183}
	traceOpt        = trace.TraceOptions(1)
	oversizeValue   = strings.Repeat("a", maxTracestateLen/2)
	oversizeEntry1  = tracestate.Entry{Key: "foo", Value: oversizeValue}
	oversizeEntry2  = tracestate.Entry{Key: "hello", Value: oversizeValue}
	entry1          = tracestate.Entry{Key: "foo", Value: "bar"}
	entry2          = tracestate.Entry{Key: "hello", Value: "world   example"}
	oversizeTs, _   = tracestate.New(nil, oversizeEntry1, oversizeEntry2)
	defaultTs, _    = tracestate.New(nil, nil...)
	nonDefaultTs, _ = tracestate.New(nil, entry1, entry2)
)

func TestHTTPFormat_FromRequest(t *testing.T) {
	tests := []struct {
		name   string
		header string
		wantSc trace.SpanContext
		wantOk bool
	}{
		{
			name:   "future version",
			header: "02-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{75, 249, 47, 53, 119, 179, 77, 166, 163, 206, 146, 157, 14, 14, 71, 54},
				SpanID:       trace.SpanID{0, 240, 103, 170, 11, 169, 2, 183},
				TraceOptions: trace.TraceOptions(1),
			},
			wantOk: true,
		},
		{
			name:   "zero trace ID and span ID",
			header: "00-00000000000000000000000000000000-0000000000000000-01",
			wantSc: trace.SpanContext{},
			wantOk: false,
		},
		{
			name:   "valid header",
			header: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
			wantSc: trace.SpanContext{
				TraceID:      trace.TraceID{75, 249, 47, 53, 119, 179, 77, 166, 163, 206, 146, 157, 14, 14, 71, 54},
				SpanID:       trace.SpanID{0, 240, 103, 170, 11, 169, 2, 183},
				TraceOptions: trace.TraceOptions(1),
			},
			wantOk: true,
		},
		{
			name:   "missing options",
			header: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
			wantSc: trace.SpanContext{},
			wantOk: false,
		},
		{
			name:   "empty options",
			header: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-",
			wantSc: trace.SpanContext{},
			wantOk: false,
		},
	}

	f := &HTTPFormat{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			req.Header.Set("traceparent", tt.header)

			gotSc, gotOk := f.SpanContextFromRequest(req)
			if !reflect.DeepEqual(gotSc, tt.wantSc) {
				t.Errorf("HTTPFormat.FromRequest() gotSc = %v, want %v", gotSc, tt.wantSc)
			}
			if gotOk != tt.wantOk {
				t.Errorf("HTTPFormat.FromRequest() gotOk = %v, want %v", gotOk, tt.wantOk)
			}
		})
	}
}

func TestHTTPFormat_ToRequest(t *testing.T) {
	tests := []struct {
		sc         trace.SpanContext
		wantHeader string
	}{
		{
			sc: trace.SpanContext{
				TraceID:      trace.TraceID{75, 249, 47, 53, 119, 179, 77, 166, 163, 206, 146, 157, 14, 14, 71, 54},
				SpanID:       trace.SpanID{0, 240, 103, 170, 11, 169, 2, 183},
				TraceOptions: trace.TraceOptions(1),
			},
			wantHeader: "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
		},
	}
	for _, tt := range tests {
		t.Run(tt.wantHeader, func(t *testing.T) {
			f := &HTTPFormat{}
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			f.SpanContextToRequest(tt.sc, req)

			h := req.Header.Get("traceparent")
			if got, want := h, tt.wantHeader; got != want {
				t.Errorf("HTTPFormat.ToRequest() header = %v, want %v", got, want)
			}
		})
	}
}

func TestHTTPFormatTracestate_FromRequest(t *testing.T) {
	scWithNonDefaultTracestate := trace.SpanContext{
		TraceID:      traceID,
		SpanID:       spanID,
		TraceOptions: traceOpt,
		Tracestate:   nonDefaultTs,
	}

	scWithDefaultTracestate := trace.SpanContext{
		TraceID:      traceID,
		SpanID:       spanID,
		TraceOptions: traceOpt,
		Tracestate:   defaultTs,
	}

	tests := []struct {
		name     string
		tpHeader string
		tsHeader string
		wantSc   trace.SpanContext
		wantOk   bool
	}{
		{
			name:     "tracestate invalid entries delimiter",
			tpHeader: tpHeader,
			tsHeader: "foo=bar;hello=world",
			wantSc:   scWithDefaultTracestate,
			wantOk:   true,
		},
		{
			name:     "tracestate invalid key-value delimiter",
			tpHeader: tpHeader,
			tsHeader: "foo=bar,hello-world",
			wantSc:   scWithDefaultTracestate,
			wantOk:   true,
		},
		{
			name:     "tracestate invalid value character",
			tpHeader: tpHeader,
			tsHeader: "foo=bar,hello=world   example   \u00a0  ",
			wantSc:   scWithDefaultTracestate,
			wantOk:   true,
		},
		{
			name:     "tracestate blank key-value",
			tpHeader: tpHeader,
			tsHeader: "foo=bar,    ",
			wantSc:   scWithDefaultTracestate,
			wantOk:   true,
		},
		{
			name:     "tracestate oversize header",
			tpHeader: tpHeader,
			tsHeader: fmt.Sprintf("foo=%s,hello=%s", oversizeValue, oversizeValue),
			wantSc:   scWithDefaultTracestate,
			wantOk:   true,
		},
		{
			name:     "tracestate valid",
			tpHeader: tpHeader,
			tsHeader: "foo=bar   ,   hello=world   example",
			wantSc:   scWithNonDefaultTracestate,
			wantOk:   true,
		},
	}

	f := &HTTPFormat{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			req.Header.Set("traceparent", tt.tpHeader)
			req.Header.Set("tracestate", tt.tsHeader)

			gotSc, gotOk := f.SpanContextFromRequest(req)
			if !reflect.DeepEqual(gotSc, tt.wantSc) {
				t.Errorf("HTTPFormat.FromRequest() gotTs = %v, want %v", gotSc.Tracestate, tt.wantSc.Tracestate)
			}
			if gotOk != tt.wantOk {
				t.Errorf("HTTPFormat.FromRequest() gotOk = %v, want %v", gotOk, tt.wantOk)
			}
		})
	}
}

func TestHTTPFormatTracestate_ToRequest(t *testing.T) {
	tests := []struct {
		name       string
		sc         trace.SpanContext
		wantHeader string
	}{
		{
			name: "valid span context with default tracestate",
			sc: trace.SpanContext{
				TraceID:      traceID,
				SpanID:       spanID,
				TraceOptions: traceOpt,
			},
			wantHeader: "",
		},
		{
			name: "valid span context with non default tracestate",
			sc: trace.SpanContext{
				TraceID:      traceID,
				SpanID:       spanID,
				TraceOptions: traceOpt,
				Tracestate:   nonDefaultTs,
			},
			wantHeader: "foo=bar,hello=world   example",
		},
		{
			name: "valid span context with oversize tracestate",
			sc: trace.SpanContext{
				TraceID:      traceID,
				SpanID:       spanID,
				TraceOptions: traceOpt,
				Tracestate:   oversizeTs,
			},
			wantHeader: "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f := &HTTPFormat{}
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			f.SpanContextToRequest(tt.sc, req)

			h := req.Header.Get("tracestate")
			if got, want := h, tt.wantHeader; got != want {
				t.Errorf("HTTPFormat.ToRequest() tracestate header = %v, want %v", got, want)
			}
		})
	}
}
