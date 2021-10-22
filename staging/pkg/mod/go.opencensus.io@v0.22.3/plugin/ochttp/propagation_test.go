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

package ochttp

import (
	"context"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"

	"go.opencensus.io/plugin/ochttp/propagation/b3"
	"go.opencensus.io/plugin/ochttp/propagation/tracecontext"
	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

func TestRoundTripAllFormats(t *testing.T) {
	// TODO: test combinations of different formats for chains of calls
	formats := []propagation.HTTPFormat{
		&b3.HTTPFormat{},
		&tracecontext.HTTPFormat{},
	}

	ctx := context.Background()
	ctx, span := trace.StartSpan(ctx, "test", trace.WithSampler(trace.AlwaysSample()))
	sc := span.SpanContext()
	wantStr := fmt.Sprintf("trace_id=%x, span_id=%x, options=%d", sc.TraceID, sc.SpanID, sc.TraceOptions)
	defer span.End()

	for _, format := range formats {
		srv := httptest.NewServer(http.HandlerFunc(func(resp http.ResponseWriter, req *http.Request) {
			sc, ok := format.SpanContextFromRequest(req)
			if !ok {
				resp.WriteHeader(http.StatusBadRequest)
			}
			fmt.Fprintf(resp, "trace_id=%x, span_id=%x, options=%d", sc.TraceID, sc.SpanID, sc.TraceOptions)
		}))
		req, err := http.NewRequest("GET", srv.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		format.SpanContextToRequest(span.SpanContext(), req)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		if resp.StatusCode != 200 {
			t.Fatal(resp.Status)
		}
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if got, want := string(body), wantStr; got != want {
			t.Errorf("%s; want %s", got, want)
		}
		srv.Close()
	}
}
