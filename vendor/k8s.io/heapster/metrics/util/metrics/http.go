// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package metrics

import (
	"bufio"
	"io"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/prometheus/client_golang/prometheus"
)

// Code heavily influenced by Prometheus' `InstrumentHandlerFunc`

var instLabels = []string{"method", "code"}

// InstrumentRouteFunc works like Prometheus' InstrumentHandlerFunc but wraps
// the go-restful RouteFunction instead of a HandlerFunc
func InstrumentRouteFunc(handlerName string, routeFunc restful.RouteFunction) restful.RouteFunction {
	opts := prometheus.SummaryOpts{
		Subsystem:   "http",
		ConstLabels: prometheus.Labels{"handler": handlerName},
	}

	reqCnt := prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Subsystem:   opts.Subsystem,
			Name:        "requests_total",
			Help:        "Total number of HTTP requests made.",
			ConstLabels: opts.ConstLabels,
		},
		instLabels,
	)

	opts.Name = "request_duration_microseconds"
	opts.Help = "The HTTP request latencies in microseconds."
	reqDur := prometheus.NewSummary(opts)

	opts.Name = "request_size_bytes"
	opts.Help = "The HTTP request sizes in bytes."
	reqSz := prometheus.NewSummary(opts)

	opts.Name = "response_size_bytes"
	opts.Help = "The HTTP response sizes in bytes."
	resSz := prometheus.NewSummary(opts)

	regReqCnt := prometheus.MustRegisterOrGet(reqCnt).(*prometheus.CounterVec)
	regReqDur := prometheus.MustRegisterOrGet(reqDur).(prometheus.Summary)
	regReqSz := prometheus.MustRegisterOrGet(reqSz).(prometheus.Summary)
	regResSz := prometheus.MustRegisterOrGet(resSz).(prometheus.Summary)

	return restful.RouteFunction(func(request *restful.Request, response *restful.Response) {
		now := time.Now()

		delegate := &responseWriterDelegator{ResponseWriter: response.ResponseWriter}
		out := make(chan int)
		urlLen := 0
		if request.Request.URL != nil {
			urlLen = len(request.Request.URL.String())
		}
		go computeApproximateRequestSize(request.Request, out, urlLen)

		_, cn := response.ResponseWriter.(http.CloseNotifier)
		_, fl := response.ResponseWriter.(http.Flusher)
		_, hj := response.ResponseWriter.(http.Hijacker)
		_, rf := response.ResponseWriter.(io.ReaderFrom)
		var rw http.ResponseWriter
		if cn && fl && hj && rf {
			rw = &fancyResponseWriterDelegator{delegate}
		} else {
			rw = delegate
		}
		response.ResponseWriter = rw

		routeFunc(request, response)

		elapsed := float64(time.Since(now)) / float64(time.Microsecond)

		method := strings.ToLower(request.Request.Method)
		code := strconv.Itoa(delegate.status)
		regReqCnt.WithLabelValues(method, code).Inc()
		regReqDur.Observe(elapsed)
		regResSz.Observe(float64(delegate.written))
		regReqSz.Observe(float64(<-out))
	})
}

func computeApproximateRequestSize(r *http.Request, out chan int, s int) {
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
	out <- s
}

type responseWriterDelegator struct {
	http.ResponseWriter

	handler, method string
	status          int
	written         int64
	wroteHeader     bool
}

func (r *responseWriterDelegator) WriteHeader(code int) {
	r.status = code
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(code)
}

func (r *responseWriterDelegator) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

type fancyResponseWriterDelegator struct {
	*responseWriterDelegator
}

func (f *fancyResponseWriterDelegator) CloseNotify() <-chan bool {
	return f.ResponseWriter.(http.CloseNotifier).CloseNotify()
}

func (f *fancyResponseWriterDelegator) Flush() {
	f.ResponseWriter.(http.Flusher).Flush()
}

func (f *fancyResponseWriterDelegator) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	return f.ResponseWriter.(http.Hijacker).Hijack()
}

func (f *fancyResponseWriterDelegator) ReadFrom(r io.Reader) (int64, error) {
	if !f.wroteHeader {
		f.WriteHeader(http.StatusOK)
	}
	n, err := f.ResponseWriter.(io.ReaderFrom).ReadFrom(r)
	f.written += n
	return n, err
}
