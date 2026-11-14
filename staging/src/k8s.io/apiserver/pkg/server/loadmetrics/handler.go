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

// Package loadmetrics implements Open Request Cost Aggregation (ORCA) load
// reporting for kube-apiserver (KEP-6315).
//
// See:
//   - CNCF xDS ORCA load report schema: https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto
//   - Envoy ORCA HTTP header specification: https://github.com/envoyproxy/envoy/blob/main/source/common/orca/orca_parser.h
//   - KEP-6315: https://git.k8s.io/enhancements/keps/sig-api-machinery/6315-weighted-load-balancing
package loadmetrics

import (
	"fmt"
	"net/http"

	"k8s.io/apiserver/pkg/endpoints/responsewriter"
)

const (
	// RequestHeader is the Kubernetes opt-in HTTP request header defined by KEP-6315
	// (https://git.k8s.io/enhancements/keps/sig-api-machinery/6315-weighted-load-balancing#load-balancer-opt-in).
	// ORCA does not standardize a request negotiation header because backends in
	// trusted environments emit load reports unconditionally; kube-apiserver requires
	// the upstream load balancer to send this header with value FormatText so
	// utilization is not exposed to arbitrary clients by default.
	RequestHeader = "endpoint-load-metrics-request"

	// ResponseHeader is the standard ORCA HTTP response header defined by Envoy
	// (https://github.com/envoyproxy/envoy/blob/main/source/common/orca/orca_parser.h)
	// for carrying text or JSON encoded xds.data.orca.v3.OrcaLoadReport metrics
	// (https://github.com/cncf/xds/blob/main/xds/data/orca/v3/orca_load_report.proto).
	ResponseHeader = "endpoint-load-metrics"

	// ResponseHeaderBin is the gRPC ORCA binary protobuf response header
	// (https://github.com/envoyproxy/envoy/blob/main/source/common/orca/orca_parser.h).
	// Although kube-apiserver does not use gRPC or emit this header, Envoy's HTTP
	// ORCA parser checks it on all HTTP responses and gives it precedence over
	// ResponseHeader. kube-apiserver strips it so proxied HTTP backends (such as
	// aggregated API servers or pods/services/nodes reached via /proxy) cannot
	// inject it to override kube-apiserver's own load report.
	ResponseHeaderBin = "endpoint-load-metrics-bin"

	// FormatText selects the ORCA native HTTP text format ("TEXT key=value, ..."),
	// defined in https://github.com/envoyproxy/envoy/blob/main/source/common/orca/orca_parser.h.
	// Other ORCA encodings (such as "JSON", "BIN", or the "endpoint-load-metrics-bin"
	// binary protobuf header) are not supported.
	FormatText = "TEXT"
)

// WithEndpointLoadMetrics wraps handler to record request counts on sampler,
// strip any ORCA response headers returned by proxied backends, and when the
// request carries "endpoint-load-metrics-request: TEXT", attach the latest
// pre-computed ORCA load report in the "endpoint-load-metrics" response header.
// If sampler is nil, handler is returned unchanged.
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

// ServeHTTP handles an incoming HTTP request by checking for ORCA load report
// negotiation, recording the request in the load sampler, and delegating to the
// underlying handler with a wrapped http.ResponseWriter that sanitizes and
// attaches ORCA response headers when the response header is written.
func (h *endpointLoadMetricsHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	requestedTextFormat := consumeTextLoadMetricsRequest(req)
	h.sampler.RecordRequest()

	wrapped := &loadMetricsResponseWriter{
		ResponseWriter:      w,
		sampler:             h.sampler,
		requestedTextFormat: requestedTextFormat,
	}
	h.handler.ServeHTTP(responsewriter.WrapForHTTP1Or2(wrapped), req)
}

var _ http.ResponseWriter = &loadMetricsResponseWriter{}
var _ responsewriter.UserProvidedDecorator = &loadMetricsResponseWriter{}

// loadMetricsResponseWriter intercepts WriteHeader and Write to attach the apiserver
// load report and strip any ORCA response headers returned by proxied backends (such as aggregated API servers
// or pod/node/service proxy subresources).
type loadMetricsResponseWriter struct {
	http.ResponseWriter
	sampler             *Reporter
	requestedTextFormat bool
	headerWritten       bool
}

func (w *loadMetricsResponseWriter) Unwrap() http.ResponseWriter {
	return w.ResponseWriter
}

func (w *loadMetricsResponseWriter) WriteHeader(statusCode int) {
	w.finalizeHeaders()
	w.ResponseWriter.WriteHeader(statusCode)
}

func (w *loadMetricsResponseWriter) Write(b []byte) (int, error) {
	w.finalizeHeaders()
	return w.ResponseWriter.Write(b)
}

func (w *loadMetricsResponseWriter) finalizeHeaders() {
	if w.headerWritten {
		return
	}
	w.headerWritten = true

	// Strip gRPC ORCA binary protobuf header.
	w.Header().Del(ResponseHeaderBin)
	var headerValue string
	if w.requestedTextFormat {
		if report := w.sampler.LoadReport(); report != nil {
			headerValue = formatORCATextReport(*report)
		}
	}
	if headerValue != "" {
		w.Header().Set(ResponseHeader, headerValue)
	} else {
		w.Header().Del(ResponseHeader)
	}
}

// consumeTextLoadMetricsRequest reports whether req negotiates ORCA TEXT load
// metrics and strips RequestHeader from req. Removing the header immediately
// prevents leaking load-balancer negotiation to aggregated API servers, webhooks,
// or proxied pods, nodes, and services.
func consumeTextLoadMetricsRequest(req *http.Request) bool {
	requestedFormat := req.Header.Get(RequestHeader)
	req.Header.Del(RequestHeader)
	return requestedFormat == FormatText
}

// formatORCATextReport formats report using the Envoy ORCA TEXT header syntax.
func formatORCATextReport(report ReportORCA) string {
	return fmt.Sprintf("TEXT cpu_utilization=%.4f, rps_fractional=%.4f", report.CPUUtilization, report.RequestsPerSecond)
}
