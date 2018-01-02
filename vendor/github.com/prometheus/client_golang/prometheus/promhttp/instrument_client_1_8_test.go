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

// +build go1.8

package promhttp

import (
	"log"
	"net/http"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

func TestClientMiddlewareAPI(t *testing.T) {
	client := http.DefaultClient
	client.Timeout = 1 * time.Second

	reg := prometheus.NewRegistry()

	inFlightGauge := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "client_in_flight_requests",
		Help: "A gauge of in-flight requests for the wrapped client.",
	})

	counter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "client_api_requests_total",
			Help: "A counter for requests from the wrapped client.",
		},
		[]string{"code", "method"},
	)

	dnsLatencyVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dns_duration_seconds",
			Help:    "Trace dns latency histogram.",
			Buckets: []float64{.005, .01, .025, .05},
		},
		[]string{"event"},
	)

	tlsLatencyVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "tls_duration_seconds",
			Help:    "Trace tls latency histogram.",
			Buckets: []float64{.05, .1, .25, .5},
		},
		[]string{"event"},
	)

	histVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "request_duration_seconds",
			Help:    "A histogram of request latencies.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{"method"},
	)

	reg.MustRegister(counter, tlsLatencyVec, dnsLatencyVec, histVec, inFlightGauge)

	trace := &InstrumentTrace{
		DNSStart: func(t float64) {
			dnsLatencyVec.WithLabelValues("dns_start")
		},
		DNSDone: func(t float64) {
			dnsLatencyVec.WithLabelValues("dns_done")
		},
		TLSHandshakeStart: func(t float64) {
			tlsLatencyVec.WithLabelValues("tls_handshake_start")
		},
		TLSHandshakeDone: func(t float64) {
			tlsLatencyVec.WithLabelValues("tls_handshake_done")
		},
	}

	client.Transport = InstrumentRoundTripperInFlight(inFlightGauge,
		InstrumentRoundTripperCounter(counter,
			InstrumentRoundTripperTrace(trace,
				InstrumentRoundTripperDuration(histVec, http.DefaultTransport),
			),
		),
	)

	resp, err := client.Get("http://google.com")
	if err != nil {
		t.Fatalf("%v", err)
	}
	defer resp.Body.Close()
}

func ExampleInstrumentRoundTripperDuration() {
	client := http.DefaultClient
	client.Timeout = 1 * time.Second

	inFlightGauge := prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "client_in_flight_requests",
		Help: "A gauge of in-flight requests for the wrapped client.",
	})

	counter := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "client_api_requests_total",
			Help: "A counter for requests from the wrapped client.",
		},
		[]string{"code", "method"},
	)

	// dnsLatencyVec uses custom buckets based on expected dns durations.
	// It has an instance label "event", which is set in the
	// DNSStart and DNSDonehook functions defined in the
	// InstrumentTrace struct below.
	dnsLatencyVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dns_duration_seconds",
			Help:    "Trace dns latency histogram.",
			Buckets: []float64{.005, .01, .025, .05},
		},
		[]string{"event"},
	)

	// tlsLatencyVec uses custom buckets based on expected tls durations.
	// It has an instance label "event", which is set in the
	// TLSHandshakeStart and TLSHandshakeDone hook functions defined in the
	// InstrumentTrace struct below.
	tlsLatencyVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "tls_duration_seconds",
			Help:    "Trace tls latency histogram.",
			Buckets: []float64{.05, .1, .25, .5},
		},
		[]string{"event"},
	)

	// histVec has no labels, making it a zero-dimensional ObserverVec.
	histVec := prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "request_duration_seconds",
			Help:    "A histogram of request latencies.",
			Buckets: prometheus.DefBuckets,
		},
		[]string{},
	)

	// Register all of the metrics in the standard registry.
	prometheus.MustRegister(counter, tlsLatencyVec, dnsLatencyVec, histVec, inFlightGauge)

	// Define functions for the available httptrace.ClientTrace hook
	// functions that we want to instrument.
	trace := &InstrumentTrace{
		DNSStart: func(t float64) {
			dnsLatencyVec.WithLabelValues("dns_start")
		},
		DNSDone: func(t float64) {
			dnsLatencyVec.WithLabelValues("dns_done")
		},
		TLSHandshakeStart: func(t float64) {
			tlsLatencyVec.WithLabelValues("tls_handshake_start")
		},
		TLSHandshakeDone: func(t float64) {
			tlsLatencyVec.WithLabelValues("tls_handshake_done")
		},
	}

	// Wrap the default RoundTripper with middleware.
	roundTripper := InstrumentRoundTripperInFlight(inFlightGauge,
		InstrumentRoundTripperCounter(counter,
			InstrumentRoundTripperTrace(trace,
				InstrumentRoundTripperDuration(histVec, http.DefaultTransport),
			),
		),
	)

	// Set the RoundTripper on our client.
	client.Transport = roundTripper

	resp, err := client.Get("http://google.com")
	if err != nil {
		log.Printf("error: %v", err)
	}
	defer resp.Body.Close()
}
