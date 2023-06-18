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
	"crypto/tls"
	"net/http"
	"net/http/httptrace"
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
	return func(r *http.Request) (*http.Response, error) {
		gauge.Inc()
		defer gauge.Dec()
		return next.RoundTrip(r)
	}
}

// InstrumentRoundTripperCounter is a middleware that wraps the provided
// http.RoundTripper to observe the request result with the provided CounterVec.
// The CounterVec must have zero, one, or two non-const non-curried labels. For
// those, the only allowed label names are "code" and "method". The function
// panics otherwise. For the "method" label a predefined default label value set
// is used to filter given values. Values besides predefined values will count
// as `unknown` method.`WithExtraMethods` can be used to add more
// methods to the set. Partitioning of the CounterVec happens by HTTP status code
// and/or HTTP method if the respective instance label names are present in the
// CounterVec. For unpartitioned counting, use a CounterVec with zero labels.
//
// If the wrapped RoundTripper panics or returns a non-nil error, the Counter
// is not incremented.
//
// Use with WithExemplarFromContext to instrument the exemplars on the counter of requests.
//
// See the example for ExampleInstrumentRoundTripperDuration for example usage.
func InstrumentRoundTripperCounter(counter *prometheus.CounterVec, next http.RoundTripper, opts ...Option) RoundTripperFunc {
	rtOpts := defaultOptions()
	for _, o := range opts {
		o.apply(rtOpts)
	}

	// Curry the counter with dynamic labels before checking the remaining labels.
	code, method := checkLabels(counter.MustCurryWith(rtOpts.emptyDynamicLabels()))

	return func(r *http.Request) (*http.Response, error) {
		resp, err := next.RoundTrip(r)
		if err == nil {
			l := labels(code, method, r.Method, resp.StatusCode, rtOpts.extraMethods...)
			for label, resolve := range rtOpts.extraLabelsFromCtx {
				l[label] = resolve(resp.Request.Context())
			}
			addWithExemplar(counter.With(l), 1, rtOpts.getExemplarFn(r.Context()))
		}
		return resp, err
	}
}

// InstrumentRoundTripperDuration is a middleware that wraps the provided
// http.RoundTripper to observe the request duration with the provided
// ObserverVec.  The ObserverVec must have zero, one, or two non-const
// non-curried labels. For those, the only allowed label names are "code" and
// "method". The function panics otherwise. For the "method" label a predefined
// default label value set is used to filter given values. Values besides
// predefined values will count as `unknown` method. `WithExtraMethods`
// can be used to add more methods to the set. The Observe method of the Observer
// in the ObserverVec is called with the request duration in
// seconds. Partitioning happens by HTTP status code and/or HTTP method if the
// respective instance label names are present in the ObserverVec. For
// unpartitioned observations, use an ObserverVec with zero labels. Note that
// partitioning of Histograms is expensive and should be used judiciously.
//
// If the wrapped RoundTripper panics or returns a non-nil error, no values are
// reported.
//
// Use with WithExemplarFromContext to instrument the exemplars on the duration histograms.
//
// Note that this method is only guaranteed to never observe negative durations
// if used with Go1.9+.
func InstrumentRoundTripperDuration(obs prometheus.ObserverVec, next http.RoundTripper, opts ...Option) RoundTripperFunc {
	rtOpts := defaultOptions()
	for _, o := range opts {
		o.apply(rtOpts)
	}

	// Curry the observer with dynamic labels before checking the remaining labels.
	code, method := checkLabels(obs.MustCurryWith(rtOpts.emptyDynamicLabels()))

	return func(r *http.Request) (*http.Response, error) {
		start := time.Now()
		resp, err := next.RoundTrip(r)
		if err == nil {
			l := labels(code, method, r.Method, resp.StatusCode, rtOpts.extraMethods...)
			for label, resolve := range rtOpts.extraLabelsFromCtx {
				l[label] = resolve(resp.Request.Context())
			}
			observeWithExemplar(obs.With(l), time.Since(start).Seconds(), rtOpts.getExemplarFn(r.Context()))
		}
		return resp, err
	}
}

// InstrumentTrace is used to offer flexibility in instrumenting the available
// httptrace.ClientTrace hook functions. Each function is passed a float64
// representing the time in seconds since the start of the http request. A user
// may choose to use separately buckets Histograms, or implement custom
// instance labels on a per function basis.
type InstrumentTrace struct {
	GotConn              func(float64)
	PutIdleConn          func(float64)
	GotFirstResponseByte func(float64)
	Got100Continue       func(float64)
	DNSStart             func(float64)
	DNSDone              func(float64)
	ConnectStart         func(float64)
	ConnectDone          func(float64)
	TLSHandshakeStart    func(float64)
	TLSHandshakeDone     func(float64)
	WroteHeaders         func(float64)
	Wait100Continue      func(float64)
	WroteRequest         func(float64)
}

// InstrumentRoundTripperTrace is a middleware that wraps the provided
// RoundTripper and reports times to hook functions provided in the
// InstrumentTrace struct. Hook functions that are not present in the provided
// InstrumentTrace struct are ignored. Times reported to the hook functions are
// time since the start of the request. Only with Go1.9+, those times are
// guaranteed to never be negative. (Earlier Go versions are not using a
// monotonic clock.) Note that partitioning of Histograms is expensive and
// should be used judiciously.
//
// For hook functions that receive an error as an argument, no observations are
// made in the event of a non-nil error value.
//
// See the example for ExampleInstrumentRoundTripperDuration for example usage.
func InstrumentRoundTripperTrace(it *InstrumentTrace, next http.RoundTripper) RoundTripperFunc {
	return func(r *http.Request) (*http.Response, error) {
		start := time.Now()

		trace := &httptrace.ClientTrace{
			GotConn: func(_ httptrace.GotConnInfo) {
				if it.GotConn != nil {
					it.GotConn(time.Since(start).Seconds())
				}
			},
			PutIdleConn: func(err error) {
				if err != nil {
					return
				}
				if it.PutIdleConn != nil {
					it.PutIdleConn(time.Since(start).Seconds())
				}
			},
			DNSStart: func(_ httptrace.DNSStartInfo) {
				if it.DNSStart != nil {
					it.DNSStart(time.Since(start).Seconds())
				}
			},
			DNSDone: func(_ httptrace.DNSDoneInfo) {
				if it.DNSDone != nil {
					it.DNSDone(time.Since(start).Seconds())
				}
			},
			ConnectStart: func(_, _ string) {
				if it.ConnectStart != nil {
					it.ConnectStart(time.Since(start).Seconds())
				}
			},
			ConnectDone: func(_, _ string, err error) {
				if err != nil {
					return
				}
				if it.ConnectDone != nil {
					it.ConnectDone(time.Since(start).Seconds())
				}
			},
			GotFirstResponseByte: func() {
				if it.GotFirstResponseByte != nil {
					it.GotFirstResponseByte(time.Since(start).Seconds())
				}
			},
			Got100Continue: func() {
				if it.Got100Continue != nil {
					it.Got100Continue(time.Since(start).Seconds())
				}
			},
			TLSHandshakeStart: func() {
				if it.TLSHandshakeStart != nil {
					it.TLSHandshakeStart(time.Since(start).Seconds())
				}
			},
			TLSHandshakeDone: func(_ tls.ConnectionState, err error) {
				if err != nil {
					return
				}
				if it.TLSHandshakeDone != nil {
					it.TLSHandshakeDone(time.Since(start).Seconds())
				}
			},
			WroteHeaders: func() {
				if it.WroteHeaders != nil {
					it.WroteHeaders(time.Since(start).Seconds())
				}
			},
			Wait100Continue: func() {
				if it.Wait100Continue != nil {
					it.Wait100Continue(time.Since(start).Seconds())
				}
			},
			WroteRequest: func(_ httptrace.WroteRequestInfo) {
				if it.WroteRequest != nil {
					it.WroteRequest(time.Since(start).Seconds())
				}
			},
		}
		r = r.WithContext(httptrace.WithClientTrace(r.Context(), trace))

		return next.RoundTrip(r)
	}
}
