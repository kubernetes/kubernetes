/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package loadmetrics

import (
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestWithEndpointLoadMetrics_HeaderNegotiation(t *testing.T) {
	testCases := []struct {
		name               string
		requestHeaderValue string
		setRequestHeader   bool
		wantResponseReport bool
	}{
		{
			name:               "omits report when request header is absent",
			setRequestHeader:   false,
			wantResponseReport: false,
		},
		{
			name:               "omits report when request header specifies unsupported format",
			setRequestHeader:   true,
			requestHeaderValue: "JSON",
			wantResponseReport: false,
		},
		{
			name:               "omits report when request header has lowercase text",
			setRequestHeader:   true,
			requestHeaderValue: "text",
			wantResponseReport: false,
		},
		{
			name:               "attaches ORCA TEXT report when opted in",
			setRequestHeader:   true,
			requestHeaderValue: FormatText,
			wantResponseReport: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t0 := time.Unix(1700000000, 0)
			fakeSampler, reporter := newTestReporter(t, loadSample{timestamp: t0})
			fakeSampler.emitSample(t0.Add(100*time.Millisecond), 0)
			fakeSampler.emitSample(t0.Add(200*time.Millisecond), 0)

			report := reporter.LoadReport()
			if report == nil {
				t.Fatal("expected load report to be available after full sample window")
			}
			expectedHeader := formatORCATextReport(*report)

			var leakedRequestHeader string
			innerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				leakedRequestHeader = r.Header.Get(RequestHeader)
				_, _ = w.Write([]byte("ok"))
			})

			handler := WithEndpointLoadMetrics(innerHandler, reporter)
			req := httptest.NewRequest(http.MethodGet, "/api/v1/namespaces", nil)
			if tc.setRequestHeader {
				req.Header.Set(RequestHeader, tc.requestHeaderValue)
			}
			recorder := httptest.NewRecorder()

			handler.ServeHTTP(recorder, req)

			if leakedRequestHeader != "" {
				t.Errorf("expected %s to be stripped before inner handler, got %q", RequestHeader, leakedRequestHeader)
			}

			gotHeader := recorder.Header().Get(ResponseHeader)
			if tc.wantResponseReport && gotHeader != expectedHeader {
				t.Errorf("got response header %q, want %q", gotHeader, expectedHeader)
			}
			if !tc.wantResponseReport && gotHeader != "" {
				t.Errorf("expected no %s response header, got %q", ResponseHeader, gotHeader)
			}
		})
	}
}

func TestReporter_SlidingWindowUtilizationAndRPS(t *testing.T) {
	t0 := time.Unix(1700000000, 0)
	fakeSampler, reporter := newTestReporter(t, loadSample{timestamp: t0, cumulativeCPUSeconds: 10.0})

	handler := WithEndpointLoadMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), reporter)

	// Interval 1 [t0, t0+100ms]: execute 10 requests while consuming 0.24 CPU-seconds.
	// With only 2 of 3 samples collected, the window is not full yet so no report is published.
	for i := 0; i < 10; i++ {
		req := httptest.NewRequest(http.MethodGet, "/version", nil)
		handler.ServeHTTP(httptest.NewRecorder(), req)
	}
	fakeSampler.emitSample(t0.Add(100*time.Millisecond), 10.24)
	if got := reporter.LoadReport(); got != nil {
		t.Fatalf("expected no report before window is full, got %+v", got)
	}

	// Interval 2 [t0+100ms, t0+200ms] is idle; the full 200ms window [t0, t0+200ms]
	// now has 3 samples: 10 requests / 0.2s = 50.0 RPS, 0.24 CPU-seconds / (0.2s * 4 cores) = 0.3 CPU utilization.
	fakeSampler.emitSample(t0.Add(200*time.Millisecond), 10.24)

	gotBusy := reporter.LoadReport()
	if gotBusy == nil {
		t.Fatal("expected report to be present once window is full")
	}
	if want := "TEXT cpu_utilization=0.3000, rps_fractional=50.0000"; formatORCATextReport(*gotBusy) != want {
		t.Fatalf("got report %q, want %q", formatORCATextReport(*gotBusy), want)
	}

	// Interval 3 [t0+200ms, t0+300ms] is also idle; the t0 sample has slid out of
	// the window [t0+100ms, t0+300ms], returning utilization and RPS to 0.
	fakeSampler.emitSample(t0.Add(300*time.Millisecond), 10.24)

	gotIdle := reporter.LoadReport()
	if gotIdle == nil {
		t.Fatal("expected idle report to be present")
	}
	if want := "TEXT cpu_utilization=0.0000, rps_fractional=0.0000"; formatORCATextReport(*gotIdle) != want {
		t.Fatalf("got idle report %q, want %q", formatORCATextReport(*gotIdle), want)
	}

	// Interval 4 [t0+300ms, t0+400ms] exceeds 100% CPU across the [t0+200ms, t0+400ms]
	// window; utilization is clamped to 1.0.
	fakeSampler.emitSample(t0.Add(400*time.Millisecond), 11.44)

	gotSaturated := reporter.LoadReport()
	if gotSaturated == nil {
		t.Fatal("expected saturated report to be present")
	}
	if want := "TEXT cpu_utilization=1.0000, rps_fractional=0.0000"; formatORCATextReport(*gotSaturated) != want {
		t.Fatalf("got saturated report %q, want %q", formatORCATextReport(*gotSaturated), want)
	}

	// Sampling error clears both the sample window and the latest report until a full window of 3 valid samples succeeds.
	fakeSampler.emitError(http.ErrServerClosed)
	if got := reporter.LoadReport(); got != nil {
		t.Fatalf("expected report to be cleared after sample error, got %+v", got)
	}
	fakeSampler.emitSample(t0.Add(500*time.Millisecond), 12.00)
	if got := reporter.LoadReport(); got != nil {
		t.Fatalf("expected report to remain nil after first recovery sample, got %+v", got)
	}
	fakeSampler.emitSample(t0.Add(600*time.Millisecond), 12.04)
	if got := reporter.LoadReport(); got != nil {
		t.Fatalf("expected report to remain nil after second recovery sample, got %+v", got)
	}
	fakeSampler.emitSample(t0.Add(700*time.Millisecond), 12.08)
	gotRecovered := reporter.LoadReport()
	if gotRecovered == nil {
		t.Fatal("expected report to recover after full window of valid samples")
	}
	if want := "TEXT cpu_utilization=0.1000, rps_fractional=0.0000"; formatORCATextReport(*gotRecovered) != want {
		t.Fatalf("got recovered report %q, want %q", formatORCATextReport(*gotRecovered), want)
	}
}

func TestNewLoadReporter_LiveRuntimeIntegration(t *testing.T) {
	reporter, err := NewLoadReporter()
	if err != nil {
		t.Fatalf("NewLoadReporter() failed: %v", err)
	}
	reporter.sampleInterval = 5 * time.Millisecond
	reporter.samples = make([]*loadSample, 0, 3)

	stopCh := make(chan struct{})
	defer close(stopCh)
	go reporter.Run(stopCh)

	handler := WithEndpointLoadMetrics(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}), reporter)

	deadline := time.Now().Add(2 * time.Second)
	var reportHeader string
	for time.Now().Before(deadline) {
		req := httptest.NewRequest(http.MethodGet, "/livez", nil)
		req.Header.Set(RequestHeader, FormatText)
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)

		if got := rec.Header().Get(ResponseHeader); got != "" {
			reportHeader = got
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if reportHeader == "" {
		t.Fatal("timed out waiting for live Reporter to publish an ORCA report")
	}
	assertValidORCATextReport(t, reportHeader)
}

func newTestReporter(t *testing.T, initial loadSample) (*fakeLoadSampler, *Reporter) {
	t.Helper()
	fakeSampler := &fakeLoadSampler{
		readyCh:  make(chan struct{}),
		sampleCh: make(chan sampleResult),
		stopCh:   make(chan struct{}),
	}
	reporter, err := newSampler(4, fakeSampler)
	if err != nil {
		t.Fatalf("newSampler failed: %v", err)
	}
	reporter.sampleInterval = 100 * time.Microsecond
	reporter.samples = make([]*loadSample, 0, 3)

	go reporter.Run(fakeSampler.stopCh)
	t.Cleanup(func() {
		close(fakeSampler.stopCh)
	})

	<-fakeSampler.readyCh
	fakeSampler.emitSample(initial.timestamp, initial.cumulativeCPUSeconds)
	return fakeSampler, reporter
}

type fakeLoadSampler struct {
	requestCounter atomic.Uint64
	readyCh        chan struct{}
	sampleCh       chan sampleResult
	stopCh         chan struct{}
}

type sampleResult struct {
	timestamp            time.Time
	cumulativeCPUSeconds float64
	err                  error
}

func (f *fakeLoadSampler) RecordRequest() {
	f.requestCounter.Add(1)
}

func (f *fakeLoadSampler) Sample() (*loadSample, error) {
	select {
	case f.readyCh <- struct{}{}:
	case <-f.stopCh:
		return &loadSample{timestamp: time.Now()}, nil
	}

	select {
	case res := <-f.sampleCh:
		if res.err != nil {
			return nil, res.err
		}
		return &loadSample{
			timestamp:              res.timestamp,
			cumulativeCPUSeconds:   res.cumulativeCPUSeconds,
			cumulativeRequestCount: f.requestCounter.Load(),
		}, nil
	case <-f.stopCh:
		return &loadSample{timestamp: time.Now()}, nil
	}
}

func (f *fakeLoadSampler) emitSample(timestamp time.Time, cumulativeCPUSeconds float64) {
	f.sampleCh <- sampleResult{timestamp: timestamp, cumulativeCPUSeconds: cumulativeCPUSeconds}
	<-f.readyCh
}

func (f *fakeLoadSampler) emitError(err error) {
	f.sampleCh <- sampleResult{err: err}
	<-f.readyCh
}

func assertValidORCATextReport(t *testing.T, report string) {
	t.Helper()
	if !strings.HasPrefix(report, "TEXT ") {
		t.Fatalf("expected ORCA report to start with %q, got %q", "TEXT ", report)
	}

	metricsPayload := strings.TrimPrefix(report, "TEXT ")
	parsed := make(map[string]float64)
	for _, entry := range strings.Split(metricsPayload, ",") {
		parts := strings.SplitN(strings.TrimSpace(entry), "=", 2)
		if len(parts) != 2 {
			t.Fatalf("malformed key=value pair %q in report %q", entry, report)
		}
		val, err := strconv.ParseFloat(parts[1], 64)
		if err != nil || math.IsNaN(val) || math.IsInf(val, 0) || val < 0 {
			t.Fatalf("invalid metric value for %q: %q (err=%v)", parts[0], parts[1], err)
		}
		parsed[parts[0]] = val
	}

	for _, requiredKey := range []string{"cpu_utilization", "rps_fractional"} {
		if _, ok := parsed[requiredKey]; !ok {
			t.Errorf("missing required ORCA key %q in report %q", requiredKey, report)
		}
	}
	if parsed["cpu_utilization"] > 1.0 {
		t.Errorf("cpu_utilization %f exceeds 1.0", parsed["cpu_utilization"])
	}
}
