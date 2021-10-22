// Copyright 2016 The Prometheus Authors
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
	"bytes"
	"errors"
	"log"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

type errorCollector struct{}

func (e errorCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- prometheus.NewDesc("invalid_metric", "not helpful", nil, nil)
}

func (e errorCollector) Collect(ch chan<- prometheus.Metric) {
	ch <- prometheus.NewInvalidMetric(
		prometheus.NewDesc("invalid_metric", "not helpful", nil, nil),
		errors.New("collect error"),
	)
}

type blockingCollector struct {
	CollectStarted, Block chan struct{}
}

func (b blockingCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- prometheus.NewDesc("dummy_desc", "not helpful", nil, nil)
}

func (b blockingCollector) Collect(ch chan<- prometheus.Metric) {
	select {
	case b.CollectStarted <- struct{}{}:
	default:
	}
	// Collects nothing, just waits for a channel receive.
	<-b.Block
}

func TestHandlerErrorHandling(t *testing.T) {

	// Create a registry that collects a MetricFamily with two elements,
	// another with one, and reports an error. Further down, we'll use the
	// same registry in the HandlerOpts.
	reg := prometheus.NewRegistry()

	cnt := prometheus.NewCounter(prometheus.CounterOpts{
		Name: "the_count",
		Help: "Ah-ah-ah! Thunder and lightning!",
	})
	reg.MustRegister(cnt)

	cntVec := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name:        "name",
			Help:        "docstring",
			ConstLabels: prometheus.Labels{"constname": "constvalue"},
		},
		[]string{"labelname"},
	)
	cntVec.WithLabelValues("val1").Inc()
	cntVec.WithLabelValues("val2").Inc()
	reg.MustRegister(cntVec)

	reg.MustRegister(errorCollector{})

	logBuf := &bytes.Buffer{}
	logger := log.New(logBuf, "", 0)

	writer := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/", nil)
	request.Header.Add("Accept", "test/plain")

	errorHandler := HandlerFor(reg, HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: HTTPErrorOnError,
		Registry:      reg,
	})
	continueHandler := HandlerFor(reg, HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: ContinueOnError,
		Registry:      reg,
	})
	panicHandler := HandlerFor(reg, HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: PanicOnError,
		Registry:      reg,
	})
	wantMsg := `error gathering metrics: error collecting metric Desc{fqName: "invalid_metric", help: "not helpful", constLabels: {}, variableLabels: []}: collect error
`
	wantErrorBody := `An error has occurred while serving metrics:

error collecting metric Desc{fqName: "invalid_metric", help: "not helpful", constLabels: {}, variableLabels: []}: collect error
`
	wantOKBody1 := `# HELP name docstring
# TYPE name counter
name{constname="constvalue",labelname="val1"} 1
name{constname="constvalue",labelname="val2"} 1
# HELP promhttp_metric_handler_errors_total Total number of internal errors encountered by the promhttp metric handler.
# TYPE promhttp_metric_handler_errors_total counter
promhttp_metric_handler_errors_total{cause="encoding"} 0
promhttp_metric_handler_errors_total{cause="gathering"} 1
# HELP the_count Ah-ah-ah! Thunder and lightning!
# TYPE the_count counter
the_count 0
`
	// It might happen that counting the gathering error makes it to the
	// promhttp_metric_handler_errors_total counter before it is gathered
	// itself. Thus, we have to bodies that are acceptable for the test.
	wantOKBody2 := `# HELP name docstring
# TYPE name counter
name{constname="constvalue",labelname="val1"} 1
name{constname="constvalue",labelname="val2"} 1
# HELP promhttp_metric_handler_errors_total Total number of internal errors encountered by the promhttp metric handler.
# TYPE promhttp_metric_handler_errors_total counter
promhttp_metric_handler_errors_total{cause="encoding"} 0
promhttp_metric_handler_errors_total{cause="gathering"} 2
# HELP the_count Ah-ah-ah! Thunder and lightning!
# TYPE the_count counter
the_count 0
`

	errorHandler.ServeHTTP(writer, request)
	if got, want := writer.Code, http.StatusInternalServerError; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
	if got := logBuf.String(); got != wantMsg {
		t.Errorf("got log message:\n%s\nwant log message:\n%s\n", got, wantMsg)
	}
	if got := writer.Body.String(); got != wantErrorBody {
		t.Errorf("got body:\n%s\nwant body:\n%s\n", got, wantErrorBody)
	}
	logBuf.Reset()
	writer.Body.Reset()
	writer.Code = http.StatusOK

	continueHandler.ServeHTTP(writer, request)
	if got, want := writer.Code, http.StatusOK; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
	if got := logBuf.String(); got != wantMsg {
		t.Errorf("got log message %q, want %q", got, wantMsg)
	}
	if got := writer.Body.String(); got != wantOKBody1 && got != wantOKBody2 {
		t.Errorf("got body %q, want either %q or %q", got, wantOKBody1, wantOKBody2)
	}

	defer func() {
		if err := recover(); err == nil {
			t.Error("expected panic from panicHandler")
		}
	}()
	panicHandler.ServeHTTP(writer, request)
}

func TestInstrumentMetricHandler(t *testing.T) {
	reg := prometheus.NewRegistry()
	handler := InstrumentMetricHandler(reg, HandlerFor(reg, HandlerOpts{}))
	// Do it again to test idempotency.
	InstrumentMetricHandler(reg, HandlerFor(reg, HandlerOpts{}))
	writer := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/", nil)
	request.Header.Add("Accept", "test/plain")

	handler.ServeHTTP(writer, request)
	if got, want := writer.Code, http.StatusOK; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}

	want := "promhttp_metric_handler_requests_in_flight 1\n"
	if got := writer.Body.String(); !strings.Contains(got, want) {
		t.Errorf("got body %q, does not contain %q", got, want)
	}
	want = "promhttp_metric_handler_requests_total{code=\"200\"} 0\n"
	if got := writer.Body.String(); !strings.Contains(got, want) {
		t.Errorf("got body %q, does not contain %q", got, want)
	}

	writer.Body.Reset()
	handler.ServeHTTP(writer, request)
	if got, want := writer.Code, http.StatusOK; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}

	want = "promhttp_metric_handler_requests_in_flight 1\n"
	if got := writer.Body.String(); !strings.Contains(got, want) {
		t.Errorf("got body %q, does not contain %q", got, want)
	}
	want = "promhttp_metric_handler_requests_total{code=\"200\"} 1\n"
	if got := writer.Body.String(); !strings.Contains(got, want) {
		t.Errorf("got body %q, does not contain %q", got, want)
	}
}

func TestHandlerMaxRequestsInFlight(t *testing.T) {
	reg := prometheus.NewRegistry()
	handler := HandlerFor(reg, HandlerOpts{MaxRequestsInFlight: 1})
	w1 := httptest.NewRecorder()
	w2 := httptest.NewRecorder()
	w3 := httptest.NewRecorder()
	request, _ := http.NewRequest("GET", "/", nil)
	request.Header.Add("Accept", "test/plain")

	c := blockingCollector{Block: make(chan struct{}), CollectStarted: make(chan struct{}, 1)}
	reg.MustRegister(c)

	rq1Done := make(chan struct{})
	go func() {
		handler.ServeHTTP(w1, request)
		close(rq1Done)
	}()
	<-c.CollectStarted

	handler.ServeHTTP(w2, request)

	if got, want := w2.Code, http.StatusServiceUnavailable; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
	if got, want := w2.Body.String(), "Limit of concurrent requests reached (1), try again later.\n"; got != want {
		t.Errorf("got body %q, want %q", got, want)
	}

	close(c.Block)
	<-rq1Done

	handler.ServeHTTP(w3, request)

	if got, want := w3.Code, http.StatusOK; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
}

func TestHandlerTimeout(t *testing.T) {
	reg := prometheus.NewRegistry()
	handler := HandlerFor(reg, HandlerOpts{Timeout: time.Millisecond})
	w := httptest.NewRecorder()

	request, _ := http.NewRequest("GET", "/", nil)
	request.Header.Add("Accept", "test/plain")

	c := blockingCollector{Block: make(chan struct{}), CollectStarted: make(chan struct{}, 1)}
	reg.MustRegister(c)

	handler.ServeHTTP(w, request)

	if got, want := w.Code, http.StatusServiceUnavailable; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
	if got, want := w.Body.String(), "Exceeded configured timeout of 1ms.\n"; got != want {
		t.Errorf("got body %q, want %q", got, want)
	}

	close(c.Block) // To not leak a goroutine.
}
