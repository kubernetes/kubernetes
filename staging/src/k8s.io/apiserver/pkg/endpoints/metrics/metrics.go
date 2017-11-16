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
	//"fmt"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/endpoints/request"

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
		[]string{"verb", "resource", "subresource", "scope", "client", "contentType", "code"},
	)
	longRunningRequestGauge = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "apiserver_longrunning_gauge",
			Help: "Gauge of all active long-running apiserver requests broken out by verb, API resource, and scope. Not all requests are tracked this way.",
		},
		[]string{"verb", "resource", "subresource", "scope"},
	)
	requestLatencies = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "apiserver_request_latencies",
			Help: "Response latency distribution in microseconds for each verb, resource and subresource.",
			// Use buckets ranging from 125 ms to 8 seconds.
			Buckets: prometheus.ExponentialBuckets(125000, 2.0, 7),
		},
		[]string{"verb", "resource", "subresource", "scope"},
	)
	requestLatenciesSummary = prometheus.NewSummaryVec(
		prometheus.SummaryOpts{
			Name: "apiserver_request_latencies_summary",
			Help: "Response latency summary in microseconds for each verb, resource and subresource.",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on our SLI definition (medium term).
			MaxAge: 5 * time.Hour,
		},
		[]string{"verb", "resource", "subresource", "scope"},
	)
	responseSizes = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "apiserver_response_sizes",
			Help: "Response size distribution in bytes for each verb, resource, subresource and scope (namespace/cluster).",
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets: prometheus.ExponentialBuckets(1000, 10.0, 7),
		},
		[]string{"verb", "resource", "subresource", "scope"},
	)
	kubectlExeRegexp = regexp.MustCompile(`^.*((?i:kubectl\.exe))`)
)

// Register all metrics.
func Register() {
	prometheus.MustRegister(requestCounter)
	prometheus.MustRegister(longRunningRequestGauge)
	prometheus.MustRegister(requestLatencies)
	prometheus.MustRegister(requestLatenciesSummary)
	prometheus.MustRegister(responseSizes)
}

// Record records a single request to the standard metrics endpoints. For use by handlers that perform their own
// processing. All API paths should use InstrumentRouteFunc implicitly. Use this instead of MonitorRequest if
// you already have a RequestInfo object.
func Record(req *http.Request, requestInfo *request.RequestInfo, contentType string, code int, responseSizeInBytes int, elapsed time.Duration) {
	if requestInfo == nil {
		requestInfo = &request.RequestInfo{Verb: req.Method, Path: req.URL.Path}
	}
	scope := cleanScope(requestInfo)
	if requestInfo.IsResourceRequest {
		MonitorRequest(req, strings.ToUpper(requestInfo.Verb), requestInfo.Resource, requestInfo.Subresource, contentType, scope, code, responseSizeInBytes, elapsed)
	} else {
		MonitorRequest(req, strings.ToUpper(requestInfo.Verb), "", requestInfo.Path, contentType, scope, code, responseSizeInBytes, elapsed)
	}
}

// RecordLongRunning tracks the execution of a long running request against the API server. It provides an accurate count
// of the total number of open long running requests. requestInfo may be nil if the caller is not in the normal request flow.
func RecordLongRunning(req *http.Request, requestInfo *request.RequestInfo, fn func()) {
	if requestInfo == nil {
		requestInfo = &request.RequestInfo{Verb: req.Method, Path: req.URL.Path}
	}
	var g prometheus.Gauge
	scope := cleanScope(requestInfo)
	reportedVerb := cleanVerb(strings.ToUpper(requestInfo.Verb), req)
	if requestInfo.IsResourceRequest {
		g = longRunningRequestGauge.WithLabelValues(reportedVerb, requestInfo.Resource, requestInfo.Subresource, scope)
	} else {
		g = longRunningRequestGauge.WithLabelValues(reportedVerb, "", requestInfo.Path, scope)
	}
	g.Inc()
	defer g.Dec()
	fn()
}

// MonitorRequest handles standard transformations for client and the reported verb and then invokes Monitor to record
// a request. verb must be uppercase to be backwards compatible with existing monitoring tooling.
func MonitorRequest(req *http.Request, verb, resource, subresource, scope, contentType string, httpCode, respSize int, elapsed time.Duration) {
	reportedVerb := cleanVerb(verb, req)
	client := cleanUserAgent(utilnet.GetHTTPClient(req))
	elapsedMicroseconds := float64(elapsed / time.Microsecond)
	requestCounter.WithLabelValues(reportedVerb, resource, subresource, scope, client, contentType, codeToString(httpCode)).Inc()
	requestLatencies.WithLabelValues(reportedVerb, resource, subresource, scope).Observe(elapsedMicroseconds)
	requestLatenciesSummary.WithLabelValues(reportedVerb, resource, subresource, scope).Observe(elapsedMicroseconds)
	// We are only interested in response sizes of read requests.
	if verb == "GET" || verb == "LIST" {
		responseSizes.WithLabelValues(reportedVerb, resource, subresource, scope).Observe(float64(respSize))
	}
}

func Reset() {
	requestCounter.Reset()
	requestLatencies.Reset()
	requestLatenciesSummary.Reset()
	responseSizes.Reset()
}

// InstrumentRouteFunc works like Prometheus' InstrumentHandlerFunc but wraps
// the go-restful RouteFunction instead of a HandlerFunc
func InstrumentRouteFunc(verb, resource, subresource, scope string, routeFunc restful.RouteFunction) restful.RouteFunction {
	return restful.RouteFunction(func(request *restful.Request, response *restful.Response) {
		now := time.Now()

		delegate := &ResponseWriterDelegator{ResponseWriter: response.ResponseWriter}

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

		MonitorRequest(request.Request, verb, resource, subresource, scope, delegate.Header().Get("Content-Type"), delegate.Status(), delegate.ContentLength(), time.Now().Sub(now))
	})
}

func cleanScope(requestInfo *request.RequestInfo) string {
	if requestInfo.Namespace != "" {
		return "namespace"
	}
	if requestInfo.Name != "" {
		return "resource"
	}
	if requestInfo.IsResourceRequest {
		return "cluster"
	}
	// this is the empty scope
	return ""
}

func cleanVerb(verb string, request *http.Request) string {
	reportedVerb := verb
	if verb == "LIST" {
		// see apimachinery/pkg/runtime/conversion.go Convert_Slice_string_To_bool
		if values := request.URL.Query()["watch"]; len(values) > 0 {
			if value := strings.ToLower(values[0]); value != "0" && value != "false" {
				reportedVerb = "WATCH"
			}
		}
	}
	// normalize the legacy WATCHLIST to WATCH to ensure users aren't surprised by metrics
	if verb == "WATCHLIST" {
		reportedVerb = "WATCH"
	}
	return reportedVerb
}

func cleanUserAgent(ua string) string {
	// We collapse all "web browser"-type user agents into one "browser" to reduce metric cardinality.
	if strings.HasPrefix(ua, "Mozilla/") {
		return "Browser"
	}
	// If an old "kubectl.exe" has passed us its full path, we discard the path portion.
	ua = kubectlExeRegexp.ReplaceAllString(ua, "$1")
	return ua
}

// ResponseWriterDelegator interface wraps http.ResponseWriter to additionally record content-length, status-code, etc.
type ResponseWriterDelegator struct {
	http.ResponseWriter

	status      int
	written     int64
	wroteHeader bool
}

func (r *ResponseWriterDelegator) WriteHeader(code int) {
	r.status = code
	r.wroteHeader = true
	r.ResponseWriter.WriteHeader(code)
}

func (r *ResponseWriterDelegator) Write(b []byte) (int, error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

func (r *ResponseWriterDelegator) Status() int {
	return r.status
}

func (r *ResponseWriterDelegator) ContentLength() int {
	return int(r.written)
}

type fancyResponseWriterDelegator struct {
	*ResponseWriterDelegator
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

// Small optimization over Itoa
func codeToString(s int) string {
	switch s {
	case 100:
		return "100"
	case 101:
		return "101"

	case 200:
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
