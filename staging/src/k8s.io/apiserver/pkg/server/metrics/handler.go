/*
Copyright 2015 The Kubernetes Authors.

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

package metrics

import (
	"net/http"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

var metricSubsystem = "http"

// SetSubsystem sets a new prefix for all metric.
// An alternative name needs to be set if you are running the deprecated
// handler along with this Prometheus handler
func SetSubsystem(name string) {
	metricSubsystem = name
}

// NewPrometheusHandler creates a http handler using the preferred prometheus
// methods that minic what was provided by the deprecated Default handler
func NewPrometheusHandler() http.Handler {
	instLabels := []string{"method", "code"}
	constLabels := prometheus.Labels{"handler": "prometheus"}

	reqCnt := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem:   metricSubsystem,
			Name:        "requests_total",
			Help:        "Total number of HTTP requests made.",
			ConstLabels: constLabels,
		},
		instLabels,
	)
	if err := prometheus.Register(reqCnt); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			reqCnt = are.ExistingCollector.(*prometheus.CounterVec)
		} else {
			panic(err)
		}
	}

	opts := prometheus.SummaryOpts{
		Subsystem:   metricSubsystem,
		ConstLabels: constLabels,
		Objectives:  map[float64]float64{0.5: 0.05, 0.9: 0.01, 0.99: 0.001},
	}

	opts.Name = "request_duration_seconds"
	opts.Help = "The HTTP request latencies in seconds."
	reqDur := prometheus.NewSummaryVec(opts, nil)
	if err := prometheus.Register(reqDur); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			reqDur = are.ExistingCollector.(*prometheus.SummaryVec)
		} else {
			panic(err)
		}
	}

	opts.Name = "request_size_bytes"
	opts.Help = "The HTTP request sizes in bytes."
	reqSz := prometheus.NewSummaryVec(opts, nil)
	if err := prometheus.Register(reqSz); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			reqSz = are.ExistingCollector.(*prometheus.SummaryVec)
		} else {
			panic(err)
		}
	}

	opts.Name = "response_size_bytes"
	opts.Help = "The HTTP response sizes in bytes."
	resSz := prometheus.NewSummaryVec(opts, nil)
	if err := prometheus.Register(resSz); err != nil {
		if are, ok := err.(prometheus.AlreadyRegisteredError); ok {
			resSz = are.ExistingCollector.(*prometheus.SummaryVec)
		} else {
			panic(err)
		}
	}

	return promhttp.InstrumentHandlerCounter(reqCnt,
		promhttp.InstrumentHandlerDuration(reqDur,
			promhttp.InstrumentHandlerRequestSize(reqSz,
				promhttp.InstrumentHandlerResponseSize(resSz, promhttp.Handler()))))
}
