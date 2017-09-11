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
	"bufio"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	dto "github.com/prometheus/client_model/go"

	"github.com/prometheus/client_golang/prometheus"
)

// magicString is used for the hacky label test in checkLabels. Remove once fixed.
const magicString = "zZgWfBxLqvG8kc8IMv3POi2Bb0tZI3vAnBx+gBaFi9FyPzB/CzKUer1yufDa"

// InstrumentHandlerInFlight is a middleware that wraps the provided
// http.Handler. It sets the provided prometheus.Gauge to the number of
// requests currently handled by the wrapped http.Handler.
//
// See the example for InstrumentHandlerDuration for example usage.
func InstrumentHandlerInFlight(g prometheus.Gauge, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		g.Inc()
		defer g.Dec()
		next.ServeHTTP(w, r)
	})
}

// InstrumentHandlerDuration is a middleware that wraps the provided
// http.Handler to observe the request duration with the provided ObserverVec.
// The ObserverVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. The Observe method of the Observer in the ObserverVec
// is called with the request duration in seconds. Partitioning happens by HTTP
// status code and/or HTTP method if the respective instance label names are
// present in the ObserverVec. For unpartitioned observations, use an
// ObserverVec with zero labels. Note that partitioning of Histograms is
// expensive and should be used judiciously.
//
// If the wrapped Handler does not set a status code, a status code of 200 is assumed.
//
// If the wrapped Handler panics, no values are reported.
func InstrumentHandlerDuration(obs prometheus.ObserverVec, next http.Handler) http.HandlerFunc {
	code, method := checkLabels(obs)

	if code {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			now := time.Now()
			d := newDelegator(w, nil)
			next.ServeHTTP(d, r)

			obs.With(labels(code, method, r.Method, d.Status())).Observe(time.Since(now).Seconds())
		})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := time.Now()
		next.ServeHTTP(w, r)
		obs.With(labels(code, method, r.Method, 0)).Observe(time.Since(now).Seconds())
	})
}

// InstrumentHandlerCounter is a middleware that wraps the provided
// http.Handler to observe the request result with the provided CounterVec.
// The CounterVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. Partitioning of the CounterVec happens by HTTP status
// code and/or HTTP method if the respective instance label names are present
// in the CounterVec. For unpartitioned counting, use a CounterVec with
// zero labels.
//
// If the wrapped Handler does not set a status code, a status code of 200 is assumed.
//
// If the wrapped Handler panics, the Counter is not incremented.
//
// See the example for InstrumentHandlerDuration for example usage.
func InstrumentHandlerCounter(counter *prometheus.CounterVec, next http.Handler) http.HandlerFunc {
	code, method := checkLabels(counter)

	if code {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			d := newDelegator(w, nil)
			next.ServeHTTP(d, r)
			counter.With(labels(code, method, r.Method, d.Status())).Inc()
		})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)
		counter.With(labels(code, method, r.Method, 0)).Inc()
	})
}

// InstrumentHandlerTimeToWriteHeader is a middleware that wraps the provided
// http.Handler to observe with the provided ObserverVec the request duration
// until the response headers are written. The ObserverVec must have zero, one,
// or two labels. The only allowed label names are "code" and "method". The
// function panics if any other instance labels are provided. The Observe
// method of the Observer in the ObserverVec is called with the request
// duration in seconds. Partitioning happens by HTTP status code and/or HTTP
// method if the respective instance label names are present in the
// ObserverVec. For unpartitioned observations, use an ObserverVec with zero
// labels. Note that partitioning of Histograms is expensive and should be used
// judiciously.
//
// If the wrapped Handler panics before calling WriteHeader, no value is
// reported.
//
// See the example for InstrumentHandlerDuration for example usage.
func InstrumentHandlerTimeToWriteHeader(obs prometheus.ObserverVec, next http.Handler) http.HandlerFunc {
	code, method := checkLabels(obs)

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		now := time.Now()
		d := newDelegator(w, func(status int) {
			obs.With(labels(code, method, r.Method, status)).Observe(time.Since(now).Seconds())
		})
		next.ServeHTTP(d, r)
	})
}

// InstrumentHandlerRequestSize is a middleware that wraps the provided
// http.Handler to observe the request size with the provided ObserverVec.
// The ObserverVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. The Observe method of the Observer in the ObserverVec
// is called with the request size in bytes. Partitioning happens by HTTP
// status code and/or HTTP method if the respective instance label names are
// present in the ObserverVec. For unpartitioned observations, use an
// ObserverVec with zero labels. Note that partitioning of Histograms is
// expensive and should be used judiciously.
//
// If the wrapped Handler does not set a status code, a status code of 200 is assumed.
//
// If the wrapped Handler panics, no values are reported.
//
// See the example for InstrumentHandlerDuration for example usage.
func InstrumentHandlerRequestSize(obs prometheus.ObserverVec, next http.Handler) http.HandlerFunc {
	code, method := checkLabels(obs)

	if code {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			d := newDelegator(w, nil)
			next.ServeHTTP(d, r)
			size := computeApproximateRequestSize(r)
			obs.With(labels(code, method, r.Method, d.Status())).Observe(float64(size))
		})
	}

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		next.ServeHTTP(w, r)
		size := computeApproximateRequestSize(r)
		obs.With(labels(code, method, r.Method, 0)).Observe(float64(size))
	})
}

// InstrumentHandlerResponseSize is a middleware that wraps the provided
// http.Handler to observe the response size with the provided ObserverVec.
// The ObserverVec must have zero, one, or two labels. The only allowed label
// names are "code" and "method". The function panics if any other instance
// labels are provided. The Observe method of the Observer in the ObserverVec
// is called with the response size in bytes. Partitioning happens by HTTP
// status code and/or HTTP method if the respective instance label names are
// present in the ObserverVec. For unpartitioned observations, use an
// ObserverVec with zero labels. Note that partitioning of Histograms is
// expensive and should be used judiciously.
//
// If the wrapped Handler does not set a status code, a status code of 200 is assumed.
//
// If the wrapped Handler panics, no values are reported.
//
// See the example for InstrumentHandlerDuration for example usage.
func InstrumentHandlerResponseSize(obs prometheus.ObserverVec, next http.Handler) http.Handler {
	code, method := checkLabels(obs)
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		d := newDelegator(w, nil)
		next.ServeHTTP(d, r)
		obs.With(labels(code, method, r.Method, d.Status())).Observe(float64(d.Written()))
	})
}

func checkLabels(c prometheus.Collector) (code bool, method bool) {
	// TODO(beorn7): Remove this hacky way to check for instance labels
	// once Descriptors can have their dimensionality queried.
	var (
		desc *prometheus.Desc
		pm   dto.Metric
	)

	descc := make(chan *prometheus.Desc, 1)
	c.Describe(descc)

	select {
	case desc = <-descc:
	default:
		panic("no description provided by collector")
	}
	select {
	case <-descc:
		panic("more than one description provided by collector")
	default:
	}

	close(descc)

	if _, err := prometheus.NewConstMetric(desc, prometheus.UntypedValue, 0); err == nil {
		return
	}
	if m, err := prometheus.NewConstMetric(desc, prometheus.UntypedValue, 0, magicString); err == nil {
		if err := m.Write(&pm); err != nil {
			panic("error checking metric for labels")
		}
		for _, label := range pm.Label {
			name, value := label.GetName(), label.GetValue()
			if value != magicString {
				continue
			}
			switch name {
			case "code":
				code = true
			case "method":
				method = true
			default:
				panic("metric partitioned with non-supported labels")
			}
			return
		}
		panic("previously set label not found – this must never happen")
	}
	if m, err := prometheus.NewConstMetric(desc, prometheus.UntypedValue, 0, magicString, magicString); err == nil {
		if err := m.Write(&pm); err != nil {
			panic("error checking metric for labels")
		}
		for _, label := range pm.Label {
			name, value := label.GetName(), label.GetValue()
			if value != magicString {
				continue
			}
			if name == "code" || name == "method" {
				continue
			}
			panic("metric partitioned with non-supported labels")
		}
		code = true
		method = true
		return
	}
	panic("metric partitioned with non-supported labels")
}

// emptyLabels is a one-time allocation for non-partitioned metrics to avoid
// unnecessary allocations on each request.
var emptyLabels = prometheus.Labels{}

func labels(code, method bool, reqMethod string, status int) prometheus.Labels {
	if !(code || method) {
		return emptyLabels
	}
	labels := prometheus.Labels{}

	if code {
		labels["code"] = sanitizeCode(status)
	}
	if method {
		labels["method"] = sanitizeMethod(reqMethod)
	}

	return labels
}

func computeApproximateRequestSize(r *http.Request) int {
	s := 0
	if r.URL != nil {
		s += len(r.URL.String())
	}

	s += len(r.Method)
	s += len(r.Proto)
	for name, values := range r.Header {
		s += len(name)
		for _, value := range values {
			s += len(value)
		}
	}
	s += len(r.Host)

	// N.B. r.Form and r.MultipartForm are assumed to be included in r.URL.

	if r.ContentLength != -1 {
		s += int(r.ContentLength)
	}
	return s
}

func sanitizeMethod(m string) string {
	switch m {
	case "GET", "get":
		return "get"
	case "PUT", "put":
		return "put"
	case "HEAD", "head":
		return "head"
	case "POST", "post":
		return "post"
	case "DELETE", "delete":
		return "delete"
	case "CONNECT", "connect":
		return "connect"
	case "OPTIONS", "options":
		return "options"
	case "NOTIFY", "notify":
		return "notify"
	default:
		return strings.ToLower(m)
	}
}

// If the wrapped http.Handler has not set a status code, i.e. the value is
// currently 0, santizeCode will return 200, for consistency with behavior in
// the stdlib.
func sanitizeCode(s int) string {
	switch s {
	case 100:
		return "100"
	case 101:
		return "101"

	case 200, 0:
		return "200"
	case 201:
		return "201"
	case 202:
		return "202"
	case 203:
		return "203"
	case 204:
		return "204"
	case 205:
		return "205"
	case 206:
		return "206"

	case 300:
		return "300"
	case 301:
		return "301"
	case 302:
		return "302"
	case 304:
		return "304"
	case 305:
		return "305"
	case 307:
		return "307"

	case 400:
		return "400"
	case 401:
		return "401"
	case 402:
		return "402"
	case 403:
		return "403"
	case 404:
		return "404"
	case 405:
		return "405"
	case 406:
		return "406"
	case 407:
		return "407"
	case 408:
		return "408"
	case 409:
		return "409"
	case 410:
		return "410"
	case 411:
		return "411"
	case 412:
		return "412"
	case 413:
		return "413"
	case 414:
		return "414"
	case 415:
		return "415"
	case 416:
		return "416"
	case 417:
		return "417"
	case 418:
		return "418"

	case 500:
		return "500"
	case 501:
		return "501"
	case 502:
		return "502"
	case 503:
		return "503"
	case 504:
		return "504"
	case 505:
		return "505"

	case 428:
		return "428"
	case 429:
		return "429"
	case 431:
		return "431"
	case 511:
		return "511"

	default:
		return strconv.Itoa(s)
	}
}

type delegator interface {
	Status() int
	Written() int64

	http.ResponseWriter
}

type responseWriterDelegator struct {
	http.ResponseWriter

	handler, method    string
	status             int
	written            int64
	wroteHeader        bool
	observeWriteHeader func(int)
}

func (r *responseWriterDelegator) Status() int {
	return r.status
}

func (r *responseWriterDelegator) Written() int64 {
	return r.written
}

func (r *responseWriterDelegator) WriteHeader(code int) {
	r.status = code
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(code)
	if r.observeWriteHeader != nil {
		r.observeWriteHeader(code)
	}
}

func (r *responseWriterDelegator) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

type fancyDelegator struct {
	*responseWriterDelegator
}

func (r *fancyDelegator) CloseNotify() <-chan bool {
	return r.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

func (r *fancyDelegator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return r.ResponseWriter.(http.Hijacker).Hijack()
}

func (r *fancyDelegator) Flush() {
	r.ResponseWriter.(http.Flusher).Flush()
}

func (r *fancyDelegator) ReadFrom(re io.Reader) (int64, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.(io.ReaderFrom).ReadFrom(re)
	r.written += n
	return n, err
}
