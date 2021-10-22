// Copyright 2017 The Prometheus Authors
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

package promhttp

import (
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestLabelCheck(t *testing.T) {
	scenarios := map[string]struct {
		varLabels     []string
		constLabels   []string
		curriedLabels []string
		ok            bool
	}{
		"empty": {
			varLabels:     []string{},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            true,
		},
		"code as single var label": {
			varLabels:     []string{"code"},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            true,
		},
		"method as single var label": {
			varLabels:     []string{"method"},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            true,
		},
		"cade and method as var labels": {
			varLabels:     []string{"method", "code"},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            true,
		},
		"valid case with all labels used": {
			varLabels:     []string{"code", "method"},
			constLabels:   []string{"foo", "bar"},
			curriedLabels: []string{"dings", "bums"},
			ok:            true,
		},
		"unsupported var label": {
			varLabels:     []string{"foo"},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            false,
		},
		"mixed var labels": {
			varLabels:     []string{"method", "foo", "code"},
			constLabels:   []string{},
			curriedLabels: []string{},
			ok:            false,
		},
		"unsupported var label but curried": {
			varLabels:     []string{},
			constLabels:   []string{},
			curriedLabels: []string{"foo"},
			ok:            true,
		},
		"mixed var labels but unsupported curried": {
			varLabels:     []string{"code", "method"},
			constLabels:   []string{},
			curriedLabels: []string{"foo"},
			ok:            true,
		},
		"supported label as const and curry": {
			varLabels:     []string{},
			constLabels:   []string{"code"},
			curriedLabels: []string{"method"},
			ok:            true,
		},
		"supported label as const and curry with unsupported as var": {
			varLabels:     []string{"foo"},
			constLabels:   []string{"code"},
			curriedLabels: []string{"method"},
			ok:            false,
		},
	}

	for name, sc := range scenarios {
		t.Run(name, func(t *testing.T) {
			constLabels := prometheus.Labels{}
			for _, l := range sc.constLabels {
				constLabels[l] = "dummy"
			}
			c := prometheus.NewCounterVec(
				prometheus.CounterOpts{
					Name:        "c",
					Help:        "c help",
					ConstLabels: constLabels,
				},
				append(sc.varLabels, sc.curriedLabels...),
			)
			o := prometheus.ObserverVec(prometheus.NewHistogramVec(
				prometheus.HistogramOpts{
					Name:        "c",
					Help:        "c help",
					ConstLabels: constLabels,
				},
				append(sc.varLabels, sc.curriedLabels...),
			))
			for _, l := range sc.curriedLabels {
				c = c.MustCurryWith(prometheus.Labels{l: "dummy"})
				o = o.MustCurryWith(prometheus.Labels{l: "dummy"})
			}

			func() {
				defer func() {
					if err := recover(); err != nil {
						if sc.ok {
							t.Error("unexpected panic:", err)
						}
					} else if !sc.ok {
						t.Error("expected panic")
					}
				}()
				InstrumentHandlerCounter(c, nil)
			}()
			func() {
				defer func() {
					if err := recover(); err != nil {
						if sc.ok {
							t.Error("unexpected panic:", err)
						}
					} else if !sc.ok {
						t.Error("expected panic")
					}
				}()
				InstrumentHandlerDuration(o, nil)
			}()
			if sc.ok {
				// Test if wantCode and wantMethod were detected correctly.
				var wantCode, wantMethod bool
				for _, l := range sc.varLabels {
					if l == "code" {
						wantCode = true
					}
					if l == "method" {
						wantMethod = true
					}
				}
				gotCode, gotMethod := checkLabels(c)
				if gotCode != wantCode {
					t.Errorf("wanted code=%t for counter, got code=%t", wantCode, gotCode)
				}
				if gotMethod != wantMethod {
					t.Errorf("wanted method=%t for counter, got method=%t", wantMethod, gotMethod)
				}
				gotCode, gotMethod = checkLabels(o)
				if gotCode != wantCode {
					t.Errorf("wanted code=%t for observer, got code=%t", wantCode, gotCode)
				}
				if gotMethod != wantMethod {
					t.Errorf("wanted method=%t for observer, got method=%t", wantMethod, gotMethod)
				}
			}
		})
	}
}

func TestMiddlewareAPI(t *testing.T) {
	reg := prometheus.NewRegistry()

	inFlightGauge := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "in_flight_requests",
		Help: "A gauge of requests currently being served by the wrapped handler.",
	})

	counter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_requests_total",
			Help: "A counter for requests to the wrapped handler.",
		},
		[]string{"code", "method"},
	)

	histVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:        "response_duration_seconds",
			Help:        "A histogram of request latencies.",
			Buckets:     prometheus.DefBuckets,
			ConstLabels: prometheus.Labels{"handler": "api"},
		},
		[]string{"method"},
	)

	writeHeaderVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:        "write_header_duration_seconds",
			Help:        "A histogram of time to first write latencies.",
			Buckets:     prometheus.DefBuckets,
			ConstLabels: prometheus.Labels{"handler": "api"},
		},
		[]string{},
	)

	responseSize := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "push_request_size_bytes",
			Help:    "A histogram of request sizes for requests.",
			Buckets: []float64{200, 500, 900, 1500},
		},
		[]string{},
	)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("OK"))
	})

	reg.MustRegister(inFlightGauge, counter, histVec, responseSize, writeHeaderVec)

	chain := InstrumentHandlerInFlight(inFlightGauge,
		InstrumentHandlerCounter(counter,
			InstrumentHandlerDuration(histVec,
				InstrumentHandlerTimeToWriteHeader(writeHeaderVec,
					InstrumentHandlerResponseSize(responseSize, handler),
				),
			),
		),
	)

	r, _ := http.NewRequest("GET", "www.example.com", nil)
	w := httptest.NewRecorder()
	chain.ServeHTTP(w, r)
}

func TestInstrumentTimeToFirstWrite(t *testing.T) {
	var i int
	dobs := &responseWriterDelegator{
		ResponseWriter: httptest.NewRecorder(),
		observeWriteHeader: func(status int) {
			i = status
		},
	}
	d := newDelegator(dobs, nil)

	d.WriteHeader(http.StatusOK)

	if i != http.StatusOK {
		t.Fatalf("failed to execute observeWriteHeader")
	}
}

// testResponseWriter is an http.ResponseWriter that also implements
// http.CloseNotifier, http.Flusher, and io.ReaderFrom.
type testResponseWriter struct {
	closeNotifyCalled, flushCalled, readFromCalled bool
}

func (t *testResponseWriter) Header() http.Header       { return nil }
func (t *testResponseWriter) Write([]byte) (int, error) { return 0, nil }
func (t *testResponseWriter) WriteHeader(int)           {}
func (t *testResponseWriter) CloseNotify() <-chan bool {
	t.closeNotifyCalled = true
	return nil
}
func (t *testResponseWriter) Flush() { t.flushCalled = true }
func (t *testResponseWriter) ReadFrom(io.Reader) (int64, error) {
	t.readFromCalled = true
	return 0, nil
}

// testFlusher is an http.ResponseWriter that also implements http.Flusher.
type testFlusher struct {
	flushCalled bool
}

func (t *testFlusher) Header() http.Header       { return nil }
func (t *testFlusher) Write([]byte) (int, error) { return 0, nil }
func (t *testFlusher) WriteHeader(int)           {}
func (t *testFlusher) Flush()                    { t.flushCalled = true }

func TestInterfaceUpgrade(t *testing.T) {
	w := &testResponseWriter{}
	d := newDelegator(w, nil)
	//lint:ignore SA1019 http.CloseNotifier is deprecated but we don't want to
	//remove support from client_golang yet.
	d.(http.CloseNotifier).CloseNotify()
	if !w.closeNotifyCalled {
		t.Error("CloseNotify not called")
	}
	d.(http.Flusher).Flush()
	if !w.flushCalled {
		t.Error("Flush not called")
	}
	d.(io.ReaderFrom).ReadFrom(nil)
	if !w.readFromCalled {
		t.Error("ReadFrom not called")
	}
	if _, ok := d.(http.Hijacker); ok {
		t.Error("delegator unexpectedly implements http.Hijacker")
	}

	f := &testFlusher{}
	d = newDelegator(f, nil)
	//lint:ignore SA1019 http.CloseNotifier is deprecated but we don't want to
	//remove support from client_golang yet.
	if _, ok := d.(http.CloseNotifier); ok {
		t.Error("delegator unexpectedly implements http.CloseNotifier")
	}
	d.(http.Flusher).Flush()
	if !w.flushCalled {
		t.Error("Flush not called")
	}
	if _, ok := d.(io.ReaderFrom); ok {
		t.Error("delegator unexpectedly implements io.ReaderFrom")
	}
	if _, ok := d.(http.Hijacker); ok {
		t.Error("delegator unexpectedly implements http.Hijacker")
	}
}

func ExampleInstrumentHandlerDuration() {
	inFlightGauge := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "in_flight_requests",
		Help: "A gauge of requests currently being served by the wrapped handler.",
	})

	counter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "api_requests_total",
			Help: "A counter for requests to the wrapped handler.",
		},
		[]string{"code", "method"},
	)

	// duration is partitioned by the HTTP method and handler. It uses custom
	// buckets based on the expected request duration.
	duration := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "request_duration_seconds",
			Help:    "A histogram of latencies for requests.",
			Buckets: []float64{.25, .5, 1, 2.5, 5, 10},
		},
		[]string{"handler", "method"},
	)

	// responseSize has no labels, making it a zero-dimensional
	// ObserverVec.
	responseSize := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "response_size_bytes",
			Help:    "A histogram of response sizes for requests.",
			Buckets: []float64{200, 500, 900, 1500},
		},
		[]string{},
	)

	// Create the handlers that will be wrapped by the middleware.
	pushHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Push"))
	})
	pullHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Pull"))
	})

	// Register all of the metrics in the standard registry.
	prometheus.MustRegister(inFlightGauge, counter, duration, responseSize)

	// Instrument the handlers with all the metrics, injecting the "handler"
	// label by currying.
	pushChain := InstrumentHandlerInFlight(inFlightGauge,
		InstrumentHandlerDuration(duration.MustCurryWith(prometheus.Labels{"handler": "push"}),
			InstrumentHandlerCounter(counter,
				InstrumentHandlerResponseSize(responseSize, pushHandler),
			),
		),
	)
	pullChain := InstrumentHandlerInFlight(inFlightGauge,
		InstrumentHandlerDuration(duration.MustCurryWith(prometheus.Labels{"handler": "pull"}),
			InstrumentHandlerCounter(counter,
				InstrumentHandlerResponseSize(responseSize, pullHandler),
			),
		),
	)

	http.Handle("/metrics", Handler())
	http.Handle("/push", pushChain)
	http.Handle("/pull", pullChain)

	if err := http.ListenAndServe(":3000", nil); err != nil {
		log.Fatal(err)
	}
}
