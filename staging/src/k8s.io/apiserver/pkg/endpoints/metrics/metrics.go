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
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	restful "github.com/emicklei/go-restful"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/types"
	utilsets "k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

// resettableCollector is the interface implemented by prometheus.MetricVec
// that can be used by Prometheus to collect metrics and reset their values.
type resettableCollector interface {
	compbasemetrics.Registerable
	Reset()
}

const (
	APIServerComponent string = "apiserver"
	OtherContentType   string = "other"
	OtherRequestMethod string = "other"
)

/*
 * By default, all the following metrics are defined as falling under
 * ALPHA stability level https://github.com/kubernetes/enhancements/blob/master/keps/sig-instrumentation/20190404-kubernetes-control-plane-metrics-stability.md#stability-classes)
 *
 * Promoting the stability level of the metric is a responsibility of the component owner, since it
 * involves explicitly acknowledging support for the metric across multiple releases, in accordance with
 * the metric stability policy.
 */
var (
	// TODO(a-robinson): Add unit tests for the handling of these metrics once
	// the upstream library supports it.
	requestCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_request_total",
			Help:           "Counter of apiserver requests broken out for each verb, dry run value, group, version, resource, scope, component, client, and HTTP response contentType and code.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		// The label_name contentType doesn't follow the label_name convention defined here:
		// https://github.com/kubernetes/community/blob/master/contributors/devel/sig-instrumentation/instrumentation.md
		// But changing it would break backwards compatibility. Future label_names
		// should be all lowercase and separated by underscores.
		[]string{"verb", "dry_run", "group", "version", "resource", "subresource", "scope", "component", "client", "contentType", "code"},
	)
	deprecatedRequestCounter = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:              "apiserver_request_count",
			Help:              "Counter of apiserver requests broken out for each verb, group, version, resource, scope, component, client, and HTTP response contentType and code.",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.14.0",
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component", "client", "contentType", "code"},
	)
	longRunningRequestGauge = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_longrunning_gauge",
			Help:           "Gauge of all active long-running apiserver requests broken out by verb, group, version, resource, scope and component. Not all requests are tracked this way.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component"},
	)
	requestLatencies = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name: "apiserver_request_duration_seconds",
			Help: "Response latency distribution in seconds for each verb, dry run value, group, version, resource, subresource, scope and component.",
			// This metric is used for verifying api call latencies SLO,
			// as well as tracking regressions in this aspects.
			// Thus we customize buckets significantly, to empower both usecases.
			Buckets: []float64{0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
				1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 50, 60},
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"verb", "dry_run", "group", "version", "resource", "subresource", "scope", "component"},
	)
	deprecatedRequestLatencies = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name: "apiserver_request_latencies",
			Help: "Response latency distribution in microseconds for each verb, group, version, resource, subresource, scope and component.",
			// Use buckets ranging from 125 ms to 8 seconds.
			Buckets:           compbasemetrics.ExponentialBuckets(125000, 2.0, 7),
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.14.0",
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component"},
	)
	deprecatedRequestLatenciesSummary = compbasemetrics.NewSummaryVec(
		&compbasemetrics.SummaryOpts{
			Name: "apiserver_request_latencies_summary",
			Help: "Response latency summary in microseconds for each verb, group, version, resource, subresource, scope and component.",
			// Make the sliding window of 5h.
			// TODO: The value for this should be based on our SLI definition (medium term).
			MaxAge:            5 * time.Hour,
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.14.0",
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component"},
	)
	responseSizes = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name: "apiserver_response_sizes",
			Help: "Response size distribution in bytes for each group, version, verb, resource, subresource, scope and component.",
			// Use buckets ranging from 1000 bytes (1KB) to 10^9 bytes (1GB).
			Buckets:        compbasemetrics.ExponentialBuckets(1000, 10.0, 7),
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component"},
	)
	// DroppedRequests is a number of requests dropped with 'Try again later' response"
	DroppedRequests = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_dropped_requests_total",
			Help:           "Number of requests dropped with 'Try again later' response",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"requestKind"},
	)
	DeprecatedDroppedRequests = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:              "apiserver_dropped_requests",
			Help:              "Number of requests dropped with 'Try again later' response",
			StabilityLevel:    compbasemetrics.ALPHA,
			DeprecatedVersion: "1.14.0",
		},
		[]string{"requestKind"},
	)
	// RegisteredWatchers is a number of currently registered watchers splitted by resource.
	RegisteredWatchers = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_registered_watchers",
			Help:           "Number of currently registered watchers for a given resources",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "version", "kind"},
	)
	WatchEvents = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_watch_events_total",
			Help:           "Number of events sent in watch clients",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "version", "kind"},
	)
	WatchEventsSizes = compbasemetrics.NewHistogramVec(
		&compbasemetrics.HistogramOpts{
			Name:           "apiserver_watch_events_sizes",
			Help:           "Watch event size distribution in bytes",
			Buckets:        compbasemetrics.ExponentialBuckets(1024, 2.0, 8), // 1K, 2K, 4K, 8K, ..., 128K.
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"group", "version", "kind"},
	)
	// Because of volatility of the base metric this is pre-aggregated one. Instead of reporting current usage all the time
	// it reports maximal usage during the last second.
	currentInflightRequests = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "apiserver_current_inflight_requests",
			Help:           "Maximal number of currently used inflight request limit of this apiserver per request kind in last second.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"requestKind"},
	)

	requestTerminationsTotal = compbasemetrics.NewCounterVec(
		&compbasemetrics.CounterOpts{
			Name:           "apiserver_request_terminations_total",
			Help:           "Number of requests which apiserver terminated in self-defense.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"verb", "group", "version", "resource", "subresource", "scope", "component", "code"},
	)
	kubectlExeRegexp = regexp.MustCompile(`^.*((?i:kubectl\.exe))`)

	metrics = []resettableCollector{
		requestCounter,
		deprecatedRequestCounter,
		longRunningRequestGauge,
		requestLatencies,
		deprecatedRequestLatencies,
		deprecatedRequestLatenciesSummary,
		responseSizes,
		DroppedRequests,
		DeprecatedDroppedRequests,
		RegisteredWatchers,
		WatchEvents,
		WatchEventsSizes,
		currentInflightRequests,
		requestTerminationsTotal,
	}

	// these are the known (e.g. whitelisted/known) content types which we will report for
	// request metrics. Any other RFC compliant content types will be aggregated under 'unknown'
	knownMetricContentTypes = utilsets.NewString(
		"application/apply-patch+yaml",
		"application/json",
		"application/json-patch+json",
		"application/merge-patch+json",
		"application/strategic-merge-patch+json",
		"application/vnd.kubernetes.protobuf",
		"application/vnd.kubernetes.protobuf;stream=watch",
		"application/yaml",
		"text/plain",
		"text/plain;charset=utf-8")
	// these are the valid request methods which we report in our metrics. Any other request methods
	// will be aggregated under 'unknown'
	validRequestMethods = utilsets.NewString(
		"APPLY",
		"CONNECT",
		"CREATE",
		"DELETE",
		"DELETECOLLECTION",
		"GET",
		"LIST",
		"PATCH",
		"POST",
		"PROXY",
		"PUT",
		"UPDATE",
		"WATCH",
		"WATCHLIST")
)

const (
	// ReadOnlyKind is a string identifying read only request kind
	ReadOnlyKind = "readOnly"
	// MutatingKind is a string identifying mutating request kind
	MutatingKind = "mutating"
)

var registerMetrics sync.Once

// Register all metrics.
func Register() {
	registerMetrics.Do(func() {
		for _, metric := range metrics {
			legacyregistry.MustRegister(metric)
		}
	})
}

// Reset all metrics.
func Reset() {
	for _, metric := range metrics {
		metric.Reset()
	}
}

func UpdateInflightRequestMetrics(nonmutating, mutating int) {
	currentInflightRequests.WithLabelValues(ReadOnlyKind).Set(float64(nonmutating))
	currentInflightRequests.WithLabelValues(MutatingKind).Set(float64(mutating))
}

// RecordRequestTermination records that the request was terminated early as part of a resource
// preservation or apiserver self-defense mechanism (e.g. timeouts, maxinflight throttling,
// proxyHandler errors). RecordRequestTermination should only be called zero or one times
// per request.
func RecordRequestTermination(req *http.Request, requestInfo *request.RequestInfo, component string, code int) {
	if requestInfo == nil {
		requestInfo = &request.RequestInfo{Verb: req.Method, Path: req.URL.Path}
	}
	scope := CleanScope(requestInfo)
	// We don't use verb from <requestInfo>, as for the healthy path
	// MonitorRequest is called from InstrumentRouteFunc which is registered
	// in installer.go with predefined list of verbs (different than those
	// translated to RequestInfo).
	// However, we need to tweak it e.g. to differentiate GET from LIST.
	verb := canonicalVerb(strings.ToUpper(req.Method), scope)
	// set verbs to a bounded set of known and expected verbs
	if !validRequestMethods.Has(verb) {
		verb = OtherRequestMethod
	}
	if requestInfo.IsResourceRequest {
		requestTerminationsTotal.WithLabelValues(cleanVerb(verb, req), requestInfo.APIGroup, requestInfo.APIVersion, requestInfo.Resource, requestInfo.Subresource, scope, component, codeToString(code)).Inc()
	} else {
		requestTerminationsTotal.WithLabelValues(cleanVerb(verb, req), "", "", "", requestInfo.Path, scope, component, codeToString(code)).Inc()
	}
}

// RecordLongRunning tracks the execution of a long running request against the API server. It provides an accurate count
// of the total number of open long running requests. requestInfo may be nil if the caller is not in the normal request flow.
func RecordLongRunning(req *http.Request, requestInfo *request.RequestInfo, component string, fn func()) {
	if requestInfo == nil {
		requestInfo = &request.RequestInfo{Verb: req.Method, Path: req.URL.Path}
	}
	var g compbasemetrics.GaugeMetric
	scope := CleanScope(requestInfo)
	// We don't use verb from <requestInfo>, as for the healthy path
	// MonitorRequest is called from InstrumentRouteFunc which is registered
	// in installer.go with predefined list of verbs (different than those
	// translated to RequestInfo).
	// However, we need to tweak it e.g. to differentiate GET from LIST.
	reportedVerb := cleanVerb(canonicalVerb(strings.ToUpper(req.Method), scope), req)
	if requestInfo.IsResourceRequest {
		g = longRunningRequestGauge.WithLabelValues(reportedVerb, requestInfo.APIGroup, requestInfo.APIVersion, requestInfo.Resource, requestInfo.Subresource, scope, component)
	} else {
		g = longRunningRequestGauge.WithLabelValues(reportedVerb, "", "", "", requestInfo.Path, scope, component)
	}
	g.Inc()
	defer g.Dec()
	fn()
}

// MonitorRequest handles standard transformations for client and the reported verb and then invokes Monitor to record
// a request. verb must be uppercase to be backwards compatible with existing monitoring tooling.
func MonitorRequest(req *http.Request, verb, group, version, resource, subresource, scope, component, contentType string, httpCode, respSize int, elapsed time.Duration) {
	reportedVerb := cleanVerb(verb, req)
	dryRun := cleanDryRun(req.URL)
	// blank out client string here, in order to avoid cardinality issues
	client := ""
	elapsedMicroseconds := float64(elapsed / time.Microsecond)
	elapsedSeconds := elapsed.Seconds()
	cleanedContentType := cleanContentType(contentType)
	requestCounter.WithLabelValues(reportedVerb, dryRun, group, version, resource, subresource, scope, component, client, cleanedContentType, codeToString(httpCode)).Inc()
	deprecatedRequestCounter.WithLabelValues(reportedVerb, group, version, resource, subresource, scope, component, client, cleanedContentType, codeToString(httpCode)).Inc()
	requestLatencies.WithLabelValues(reportedVerb, dryRun, group, version, resource, subresource, scope, component).Observe(elapsedSeconds)
	deprecatedRequestLatencies.WithLabelValues(reportedVerb, group, version, resource, subresource, scope, component).Observe(elapsedMicroseconds)
	deprecatedRequestLatenciesSummary.WithLabelValues(reportedVerb, group, version, resource, subresource, scope, component).Observe(elapsedMicroseconds)
	// We are only interested in response sizes of read requests.
	if verb == "GET" || verb == "LIST" {
		responseSizes.WithLabelValues(reportedVerb, group, version, resource, subresource, scope, component).Observe(float64(respSize))
	}
}

// InstrumentRouteFunc works like Prometheus' InstrumentHandlerFunc but wraps
// the go-restful RouteFunction instead of a HandlerFunc plus some Kubernetes endpoint specific information.
func InstrumentRouteFunc(verb, group, version, resource, subresource, scope, component string, routeFunc restful.RouteFunction) restful.RouteFunction {
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

		MonitorRequest(request.Request, verb, group, version, resource, subresource, scope, component, delegate.Header().Get("Content-Type"), delegate.Status(), delegate.ContentLength(), time.Since(now))
	})
}

// InstrumentHandlerFunc works like Prometheus' InstrumentHandlerFunc but adds some Kubernetes endpoint specific information.
func InstrumentHandlerFunc(verb, group, version, resource, subresource, scope, component string, handler http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		now := time.Now()

		delegate := &ResponseWriterDelegator{ResponseWriter: w}

		_, cn := w.(http.CloseNotifier)
		_, fl := w.(http.Flusher)
		_, hj := w.(http.Hijacker)
		if cn && fl && hj {
			w = &fancyResponseWriterDelegator{delegate}
		} else {
			w = delegate
		}

		handler(w, req)

		MonitorRequest(req, verb, group, version, resource, subresource, scope, component, delegate.Header().Get("Content-Type"), delegate.Status(), delegate.ContentLength(), time.Since(now))
	}
}

// cleanContentType binds the contentType (for metrics related purposes) to a
// bounded set of known/expected content-types.
func cleanContentType(contentType string) string {
	normalizedContentType := strings.ToLower(contentType)
	if strings.HasSuffix(contentType, " stream=watch") || strings.HasSuffix(contentType, " charset=utf-8") {
		normalizedContentType = strings.ReplaceAll(contentType, " ", "")
	}
	if knownMetricContentTypes.Has(normalizedContentType) {
		return normalizedContentType
	}
	return OtherContentType
}

// CleanScope returns the scope of the request.
func CleanScope(requestInfo *request.RequestInfo) string {
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

func canonicalVerb(verb string, scope string) string {
	switch verb {
	case "GET", "HEAD":
		if scope != "resource" {
			return "LIST"
		}
		return "GET"
	default:
		return verb
	}
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
	if verb == "PATCH" && request.Header.Get("Content-Type") == string(types.ApplyPatchType) && utilfeature.DefaultFeatureGate.Enabled(features.ServerSideApply) {
		reportedVerb = "APPLY"
	}
	if validRequestMethods.Has(reportedVerb) {
		return reportedVerb
	}
	return OtherRequestMethod
}

func cleanDryRun(u *url.URL) string {
	// avoid allocating when we don't see dryRun in the query
	if !strings.Contains(u.RawQuery, "dryRun") {
		return ""
	}
	dryRun := u.Query()["dryRun"]
	if errs := validation.ValidateDryRun(nil, dryRun); len(errs) > 0 {
		return "invalid"
	}
	// Since dryRun could be valid with any arbitrarily long length
	// we have to dedup and sort the elements before joining them together
	// TODO: this is a fairly large allocation for what it does, consider
	//   a sort and dedup in a single pass
	return strings.Join(utilsets.NewString(dryRun...).List(), ",")
}

func cleanUserAgent(ua string) string {
	// We collapse all "web browser"-type user agents into one "browser" to reduce metric cardinality.
	if strings.HasPrefix(ua, "Mozilla/") {
		return "Browser"
	}
	// If an old "kubectl.exe" has passed us its full path, we discard the path portion.
	if kubectlExeRegexp.MatchString(ua) {
		// avoid an allocation
		ua = kubectlExeRegexp.ReplaceAllString(ua, "$1")
	}
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
