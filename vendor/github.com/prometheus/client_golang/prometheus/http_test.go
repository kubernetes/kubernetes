// Copyright 2014 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package prometheus

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
)

type respBody string

func (b respBody) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusTeapot)
	w.Write([]byte(b))
}

func TestInstrumentHandler(t *testing.T) {
	defer func(n nower) {
		now = n.(nower)
	}(now)

	instant := time.Now()
	end := instant.Add(30 * time.Second)
	now = nowSeries(instant, end)
	respBody := respBody("Howdy there!")

	hndlr := InstrumentHandler("test-handler", respBody)

	opts := SummaryOpts{
		Subsystem:   "http",
		ConstLabels: Labels{"handler": "test-handler"},
		Objectives:  map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
	}

	reqCnt := NewCounterVec(
		CounterOpts{
			Namespace:   opts.Namespace,
			Subsystem:   opts.Subsystem,
			Name:        "requests_total",
			Help:        "Total number of HTTP requests made.",
			ConstLabels: opts.ConstLabels,
		},
		instLabels,
	)
	err := Register(reqCnt)
	if err == nil {
		t.Fatal("expected reqCnt to be registered already")
	}
	if are, ok := err.(AlreadyRegisteredError); ok {
		reqCnt = are.ExistingCollector.(*CounterVec)
	} else {
		t.Fatal("unexpected registration error:", err)
	}

	opts.Name = "request_duration_microseconds"
	opts.Help = "The HTTP request latencies in microseconds."
	reqDur := NewSummary(opts)
	err = Register(reqDur)
	if err == nil {
		t.Fatal("expected reqDur to be registered already")
	}
	if are, ok := err.(AlreadyRegisteredError); ok {
		reqDur = are.ExistingCollector.(Summary)
	} else {
		t.Fatal("unexpected registration error:", err)
	}

	opts.Name = "request_size_bytes"
	opts.Help = "The HTTP request sizes in bytes."
	reqSz := NewSummary(opts)
	err = Register(reqSz)
	if err == nil {
		t.Fatal("expected reqSz to be registered already")
	}
	if _, ok := err.(AlreadyRegisteredError); !ok {
		t.Fatal("unexpected registration error:", err)
	}

	opts.Name = "response_size_bytes"
	opts.Help = "The HTTP response sizes in bytes."
	resSz := NewSummary(opts)
	err = Register(resSz)
	if err == nil {
		t.Fatal("expected resSz to be registered already")
	}
	if _, ok := err.(AlreadyRegisteredError); !ok {
		t.Fatal("unexpected registration error:", err)
	}

	reqCnt.Reset()

	resp := httptest.NewRecorder()
	req := &http.Request{
		Method: "GET",
	}

	hndlr.ServeHTTP(resp, req)

	if resp.Code != http.StatusTeapot {
		t.Fatalf("expected status %d, got %d", http.StatusTeapot, resp.Code)
	}
	if string(resp.Body.Bytes()) != "Howdy there!" {
		t.Fatalf("expected body %s, got %s", "Howdy there!", string(resp.Body.Bytes()))
	}

	out := &dto.Metric{}
	reqDur.Write(out)
	if want, got := "test-handler", out.Label[0].GetValue(); want != got {
		t.Errorf("want label value %q in reqDur, got %q", want, got)
	}
	if want, got := uint64(1), out.Summary.GetSampleCount(); want != got {
		t.Errorf("want sample count %d in reqDur, got %d", want, got)
	}

	out.Reset()
	if want, got := 1, len(reqCnt.children); want != got {
		t.Errorf("want %d children in reqCnt, got %d", want, got)
	}
	cnt, err := reqCnt.GetMetricWithLabelValues("get", "418")
	if err != nil {
		t.Fatal(err)
	}
	cnt.Write(out)
	if want, got := "418", out.Label[0].GetValue(); want != got {
		t.Errorf("want label value %q in reqCnt, got %q", want, got)
	}
	if want, got := "test-handler", out.Label[1].GetValue(); want != got {
		t.Errorf("want label value %q in reqCnt, got %q", want, got)
	}
	if want, got := "get", out.Label[2].GetValue(); want != got {
		t.Errorf("want label value %q in reqCnt, got %q", want, got)
	}
	if out.Counter == nil {
		t.Fatal("expected non-nil counter in reqCnt")
	}
	if want, got := 1., out.Counter.GetValue(); want != got {
		t.Errorf("want reqCnt of %f, got %f", want, got)
	}
}
