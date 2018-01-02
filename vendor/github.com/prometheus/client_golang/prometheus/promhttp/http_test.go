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
	"testing"

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

func TestHandlerErrorHandling(t *testing.T) {

	// Create a registry that collects a MetricFamily with two elements,
	// another with one, and reports an error.
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
	})
	continueHandler := HandlerFor(reg, HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: ContinueOnError,
	})
	panicHandler := HandlerFor(reg, HandlerOpts{
		ErrorLog:      logger,
		ErrorHandling: PanicOnError,
	})
	wantMsg := `error gathering metrics: error collecting metric Desc{fqName: "invalid_metric", help: "not helpful", constLabels: {}, variableLabels: []}: collect error
`
	wantErrorBody := `An error has occurred during metrics gathering:

error collecting metric Desc{fqName: "invalid_metric", help: "not helpful", constLabels: {}, variableLabels: []}: collect error
`
	wantOKBody := `# HELP name docstring
# TYPE name counter
name{constname="constvalue",labelname="val1"} 1
name{constname="constvalue",labelname="val2"} 1
# HELP the_count Ah-ah-ah! Thunder and lightning!
# TYPE the_count counter
the_count 0
`

	errorHandler.ServeHTTP(writer, request)
	if got, want := writer.Code, http.StatusInternalServerError; got != want {
		t.Errorf("got HTTP status code %d, want %d", got, want)
	}
	if got := logBuf.String(); got != wantMsg {
		t.Errorf("got log message:\n%s\nwant log mesage:\n%s\n", got, wantMsg)
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
	if got := writer.Body.String(); got != wantOKBody {
		t.Errorf("got body %q, want %q", got, wantOKBody)
	}

	defer func() {
		if err := recover(); err == nil {
			t.Error("expected panic from panicHandler")
		}
	}()
	panicHandler.ServeHTTP(writer, request)
}
