/*
Copyright The Kubernetes Authors.

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

package loadmetrics

import (
	"fmt"
	"net/http"
)

const (
	RequestHeader  = "endpoint-load-metrics-request"
	ResponseHeader = "endpoint-load-metrics"
	FormatText     = "TEXT"
)

func WithEndpointLoadMetrics(handler http.Handler, sampler *Reporter) http.Handler {
	if sampler == nil {
		return handler
	}
	return &endpointLoadMetricsHandler{
		handler: handler,
		sampler: sampler,
	}
}

type endpointLoadMetricsHandler struct {
	handler http.Handler
	sampler *Reporter
}

// Response headers must be set before the inner handler writes the response
// status or body. Setting the header from the latest pre-computed report
// before ServeHTTP avoids wrapping http.ResponseWriter and prevents concurrent
// map access on w.Header() when downstream filters (such as
// WithTimeoutForNonLongRunningRequests) run the handler in a separate goroutine.
func (h *endpointLoadMetricsHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if consumeTextLoadMetricsRequest(req) {
		if report := h.sampler.LoadReport(); report != nil {
			w.Header().Set(ResponseHeader, formatORCATextReport(*report))
		}
	}
	h.sampler.RecordRequest()
	h.handler.ServeHTTP(w, req)
}

// Removing the request header immediately prevents leaking load-balancer
// negotiation to aggregated API servers, webhooks, or proxied pods/nodes/services.
func consumeTextLoadMetricsRequest(req *http.Request) bool {
	requestedFormat := req.Header.Get(RequestHeader)
	req.Header.Del(RequestHeader)
	return requestedFormat == FormatText
}

func formatORCATextReport(report ReportORCA) string {
	return fmt.Sprintf("TEXT cpu_utilization=%.4f, rps_fractional=%.4f", report.CPUUtilization, report.RequestsPerSecond)
}
