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
	"bufio"
	"fmt"
	"net"
	"net/http"
	"time"

	utilnet "k8s.io/kubernetes/pkg/util/net"

	"github.com/emicklei/go-restful"
	"github.com/prometheus/client_golang/prometheus"
)

var (
	// TODO(a-robinson): Add unit tests for the handling of these metrics once
	// the upstream library supports it.
	requestCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "apiserver_request_count",
			Help: "Counter of apiserver requests broken out for each verb, API resource, client, and HTTP response contentType and code.",
		},
		[]string{"verb", "resource", "client", "contentType", "code"},
	)
	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "apiserver_request_latencies",
			Help: "Response latency distribution in microseconds for each verb, resource and client.",
			// Use buckets ranging from 125 ms to 8 seconds.
			Buckets: prometheus.ExponentialBuckets(125000, 2.0, 7),
		},
		[]string{"verb", "resource"},
	)
	requestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "apiserver_request_latencies_summary",
			Help: "Response latency summary in microseconds for each verb and resource.",
			// Make the sliding window of 1h.
			MaxAge: time.Hour,
		},
		[]string{"verb", "resource"},
	)
)

// Register all metrics.
func Register() {
	prometheus.MustRegister(requestCounter)
	prometheus.MustRegister(requestLatencies)
	prometheus.MustRegister(requestLatenciesSummary)
}

func Monitor(verb, resource *string, client, contentType string, httpCode int, reqStart time.Time) {
	elapsed := float64((time.Since(reqStart)) / time.Microsecond)
	httpCodeStr := fmt.Sprintf("%d", httpCode)
	requestCounter.WithLabelValues(*verb, *resource, client, contentType, httpCodeStr).Inc()
	requestLatencies.WithLabelValues(*verb, *resource).Observe(elapsed)
	requestLatenciesSummary.WithLabelValues(*verb, *resource).Observe(elapsed)
}

func Reset() {
	requestCounter.Reset()
	requestLatencies.Reset()
	requestLatenciesSummary.Reset()
}

// InstrumentRouteFunc works like Prometheus' InstrumentHandlerFunc but wraps
// the go-restful RouteFunction instead of a HandlerFunc
func InstrumentRouteFunc(verb, resource string, routeFunc restful.RouteFunction) restful.RouteFunction {
	return restful.RouteFunction(func(request *restful.Request, response *restful.Response) {
		now := time.Now()

		delegate := &responseWriterDelegator{ResponseWriter: response.ResponseWriter}

		_, cn := response.ResponseWriter.(http.CloseNotifier)
		_, fl := response.ResponseWriter.(http.Flusher)
		_, hj := response.ResponseWriter.(http.Hijacker)
		var rw http.ResponseWriter
		if cn && fl && hj {
			rw = &fancyResponseWriterDelegator{delegate}
		} else {
			rw = delegate
		}
		response.ResponseWriter = rw

		routeFunc(request, response)
		Monitor(&verb, &resource, utilnet.GetHTTPClient(request.Request), rw.Header().Get("Content-Type"), delegate.status, now)
	})
}

type responseWriterDelegator struct {
	http.ResponseWriter

	status      int
	written     int64
	wroteHeader bool
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
