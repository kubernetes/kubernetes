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

package ochttp_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"go.opencensus.io/plugin/ochttp"
	"go.opencensus.io/trace"
)

func TestSpanAnnotatingClientTrace(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(resp http.ResponseWriter, req *http.Request) {
		resp.Write([]byte("Hello, world!"))
	}))
	defer server.Close()

	recorder := &testExporter{}

	trace.RegisterExporter(recorder)

	tr := ochttp.Transport{
		NewClientTrace: ochttp.NewSpanAnnotatingClientTrace,
		StartOptions: trace.StartOptions{
			Sampler: trace.AlwaysSample(),
		},
	}

	req, err := http.NewRequest("POST", server.URL, strings.NewReader("req-body"))
	if err != nil {
		t.Errorf("error creating request: %v", err)
	}

	resp, err := tr.RoundTrip(req)
	if err != nil {
		t.Errorf("response error: %v", err)
	}
	if err := resp.Body.Close(); err != nil {
		t.Errorf("error closing response body: %v", err)
	}
	if got, want := resp.StatusCode, 200; got != want {
		t.Errorf("resp.StatusCode=%d; want=%d", got, want)
	}

	if got, want := len(recorder.spans), 1; got != want {
		t.Fatalf("span count=%d; want=%d", got, want)
	}

	var annotations []string
	for _, annotation := range recorder.spans[0].Annotations {
		annotations = append(annotations, annotation.Message)
	}

	required := []string{
		"GetConn", "GotConn", "GotFirstResponseByte", "ConnectStart",
		"ConnectDone", "WroteHeaders", "WroteRequest",
	}

	if errs := requiredAnnotations(required, annotations); len(errs) > 0 {
		for _, err := range errs {
			t.Error(err)
		}
	}

}

type testExporter struct {
	mu    sync.Mutex
	spans []*trace.SpanData
}

func (t *testExporter) ExportSpan(s *trace.SpanData) {
	t.mu.Lock()
	t.spans = append(t.spans, s)
	t.mu.Unlock()
}

func requiredAnnotations(required []string, list []string) []error {
	var errs []error
	for _, item := range required {
		var found bool
		for _, v := range list {
			if v == item {
				found = true
			}
		}
		if !found {
			errs = append(errs, errors.New("missing expected annotation: "+item))
		}
	}
	return errs
}
