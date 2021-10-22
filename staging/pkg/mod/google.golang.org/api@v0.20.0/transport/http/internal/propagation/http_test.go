// Copyright 2018 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.8

package propagation

import (
	"net/http"
	"reflect"
	"testing"

	"go.opencensus.io/trace"
)

func TestHTTPFormat(t *testing.T) {
	format := &HTTPFormat{}
	traceID := [16]byte{16, 84, 69, 170, 120, 67, 188, 139, 242, 6, 177, 32, 0, 16, 0, 0}
	spanID1 := [8]byte{255, 0, 0, 0, 0, 0, 0, 123}
	spanID2 := [8]byte{0, 0, 0, 0, 0, 0, 0, 123}
	tests := []struct {
		incoming        string
		wantSpanContext trace.SpanContext
	}{
		{
			incoming: "105445aa7843bc8bf206b12000100000/18374686479671623803;o=1",
			wantSpanContext: trace.SpanContext{
				TraceID:      traceID,
				SpanID:       spanID1,
				TraceOptions: 1,
			},
		},
		{
			incoming: "105445aa7843bc8bf206b12000100000/123;o=0",
			wantSpanContext: trace.SpanContext{
				TraceID:      traceID,
				SpanID:       spanID2,
				TraceOptions: 0,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.incoming, func(t *testing.T) {
			req, _ := http.NewRequest("GET", "http://example.com", nil)
			req.Header.Add(httpHeader, tt.incoming)
			sc, ok := format.SpanContextFromRequest(req)
			if !ok {
				t.Errorf("exporter.SpanContextFromRequest() = false; want true")
			}
			if got, want := sc, tt.wantSpanContext; !reflect.DeepEqual(got, want) {
				t.Errorf("exporter.SpanContextFromRequest() returned span context %v; want %v", got, want)
			}

			req, _ = http.NewRequest("GET", "http://example.com", nil)
			format.SpanContextToRequest(sc, req)
			if got, want := req.Header.Get(httpHeader), tt.incoming; got != want {
				t.Errorf("exporter.SpanContextToRequest() returned header %q; want %q", got, want)
			}
		})
	}
}
