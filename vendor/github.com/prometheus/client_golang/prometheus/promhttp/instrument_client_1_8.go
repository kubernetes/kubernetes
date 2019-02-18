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
	"context"
	"crypto/tls"
	"net/http"
	"net/http/httptrace"
	"time"
)

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
	return RoundTripperFunc(func(r *http.Request) (*http.Response, error) {
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
		r = r.WithContext(httptrace.WithClientTrace(context.Background(), trace))

		return next.RoundTrip(r)
	})
}
