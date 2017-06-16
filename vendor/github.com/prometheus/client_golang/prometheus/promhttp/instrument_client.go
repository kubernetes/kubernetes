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
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// The RoundTripperFunc type is an adapter to allow the use of ordinary
// functions as RoundTrippers. If f is a function with the appropriate
// signature, RountTripperFunc(f) is a RoundTripper that calls f.
type RoundTripperFunc func(req *http.Request) (*http.Response, error)

// RoundTrip implements the RoundTripper interface.
func (rt RoundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return rt(r)
}

// InstrumentRoundTripperInFlight is a middleware that wraps the provided
// http.RoundTripper. It sets the provided prometheus.Gauge to the number of
// requests currently handled by the wrapped http.RoundTripper.
//
// See the example for ExampleInstrumentRoundTripperDuration for example usage.
func InstrumentRoundTripperInFlight(gauge prometheus.Gauge, next http.RoundTripper) RoundTripperFunc {
	return RoundTripperFunc(func(r *http.Request) (*http.Response, error) {
		gauge.Inc()
		defer gauge.Dec()
		return next.RoundTrip(r)
	})
}

// InstrumentRoundTripperCounter is a middleware that wraps the provided
// http.RoundTripper to observe the request result with the provided CounterVec.
// The CounterVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. Partitioning of the CounterVec happens by HTTP status
// code and/or HTTP method if the respective instance label names are present
// in the CounterVec. For unpartitioned counting, use a CounterVec with
// zero labels.
//
// If the wrapped RoundTripper panics or returns a non-nil error, the Counter
// is not incremented.
//
// See the example for ExampleInstrumentRoundTripperDuration for example usage.
func InstrumentRoundTripperCounter(counter *prometheus.CounterVec, next http.RoundTripper) RoundTripperFunc {
	code, method := checkLabels(counter)

	return RoundTripperFunc(func(r *http.Request) (*http.Response, error) {
		resp, err := next.RoundTrip(r)
		if err == nil {
			counter.With(labels(code, method, r.Method, resp.StatusCode)).Inc()
		}
		return resp, err
	})
}

// InstrumentRoundTripperDuration is a middleware that wraps the provided
// http.RoundTripper to observe the request duration with the provided ObserverVec.
// The ObserverVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. The Observe method of the Observer in the ObserverVec
// is called with the request duration in seconds. Partitioning happens by HTTP
// status code and/or HTTP method if the respective instance label names are
// present in the ObserverVec. For unpartitioned observations, use an
// ObserverVec with zero labels. Note that partitioning of Histograms is
// expensive and should be used judiciously.
//
// If the wrapped RoundTripper panics or returns a non-nil error, no values are
// reported.
func InstrumentRoundTripperDuration(obs prometheus.ObserverVec, next http.RoundTripper) RoundTripperFunc {
	code, method := checkLabels(obs)

	return RoundTripperFunc(func(r *http.Request) (*http.Response, error) {
		start := time.Now()
		resp, err := next.RoundTrip(r)
		if err == nil {
			obs.With(labels(code, method, r.Method, resp.StatusCode)).Observe(time.Since(start).Seconds())
		}
		return resp, err
	})
}
