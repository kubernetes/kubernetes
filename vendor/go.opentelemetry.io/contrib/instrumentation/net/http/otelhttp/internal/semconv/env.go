// Code created by gotmpl. DO NOT MODIFY.
// source: internal/shared/semconv/env.go.tmpl

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
)

// OTelSemConvStabilityOptIn is an environment variable.
// That can be set to "old" or "http/dup" to opt into the new HTTP semantic conventions.
const OTelSemConvStabilityOptIn = "OTEL_SEMCONV_STABILITY_OPT_IN"

type ResponseTelemetry struct {
	StatusCode int
	ReadBytes  int64
	ReadError  error
	WriteBytes int64
	WriteError error
}

type HTTPServer struct {
	duplicate bool

	// Old metrics
	requestBytesCounter  metric.Int64Counter
	responseBytesCounter metric.Int64Counter
	serverLatencyMeasure metric.Float64Histogram

	// New metrics
	requestBodySizeHistogram  metric.Int64Histogram
	responseBodySizeHistogram metric.Int64Histogram
	requestDurationHistogram  metric.Float64Histogram
}

// RequestTraceAttrs returns trace attributes for an HTTP request received by a
// server.
//
// The server must be the primary server name if it is known. For example this
// would be the ServerName directive
// (https://httpd.apache.org/docs/2.4/mod/core.html#servername) for an Apache
// server, and the server_name directive
// (http://nginx.org/en/docs/http/ngx_http_core_module.html#server_name) for an
// nginx server. More generically, the primary server name would be the host
// header value that matches the default virtual host of an HTTP server. It
// should include the host identifier and if a port is used to route to the
// server that port identifier should be included as an appropriate port
// suffix.
//
// If the primary server name is not known, server should be an empty string.
// The req Host will be used to determine the server instead.
func (s HTTPServer) RequestTraceAttrs(server string, req *http.Request) []attribute.KeyValue {
	if s.duplicate {
		return append(OldHTTPServer{}.RequestTraceAttrs(server, req), CurrentHTTPServer{}.RequestTraceAttrs(server, req)...)
	}
	return OldHTTPServer{}.RequestTraceAttrs(server, req)
}

// ResponseTraceAttrs returns trace attributes for telemetry from an HTTP response.
//
// If any of the fields in the ResponseTelemetry are not set the attribute will be omitted.
func (s HTTPServer) ResponseTraceAttrs(resp ResponseTelemetry) []attribute.KeyValue {
	if s.duplicate {
		return append(OldHTTPServer{}.ResponseTraceAttrs(resp), CurrentHTTPServer{}.ResponseTraceAttrs(resp)...)
	}
	return OldHTTPServer{}.ResponseTraceAttrs(resp)
}

// Route returns the attribute for the route.
func (s HTTPServer) Route(route string) attribute.KeyValue {
	return OldHTTPServer{}.Route(route)
}

// Status returns a span status code and message for an HTTP status code
// value returned by a server. Status codes in the 400-499 range are not
// returned as errors.
func (s HTTPServer) Status(code int) (codes.Code, string) {
	if code < 100 || code >= 600 {
		return codes.Error, fmt.Sprintf("Invalid HTTP status code %d", code)
	}
	if code >= 500 {
		return codes.Error, ""
	}
	return codes.Unset, ""
}

type ServerMetricData struct {
	ServerName   string
	ResponseSize int64

	MetricData
	MetricAttributes
}

type MetricAttributes struct {
	Req                  *http.Request
	StatusCode           int
	AdditionalAttributes []attribute.KeyValue
}

type MetricData struct {
	RequestSize int64
	ElapsedTime float64
}

var (
	metricAddOptionPool = &sync.Pool{
		New: func() interface{} {
			return &[]metric.AddOption{}
		},
	}

	metricRecordOptionPool = &sync.Pool{
		New: func() interface{} {
			return &[]metric.RecordOption{}
		},
	}
)

func (s HTTPServer) RecordMetrics(ctx context.Context, md ServerMetricData) {
	if s.requestBytesCounter != nil && s.responseBytesCounter != nil && s.serverLatencyMeasure != nil {
		attributes := OldHTTPServer{}.MetricAttributes(md.ServerName, md.Req, md.StatusCode, md.AdditionalAttributes)
		o := metric.WithAttributeSet(attribute.NewSet(attributes...))
		addOpts := metricAddOptionPool.Get().(*[]metric.AddOption)
		*addOpts = append(*addOpts, o)
		s.requestBytesCounter.Add(ctx, md.RequestSize, *addOpts...)
		s.responseBytesCounter.Add(ctx, md.ResponseSize, *addOpts...)
		s.serverLatencyMeasure.Record(ctx, md.ElapsedTime, o)
		*addOpts = (*addOpts)[:0]
		metricAddOptionPool.Put(addOpts)
	}

	if s.duplicate && s.requestDurationHistogram != nil && s.requestBodySizeHistogram != nil && s.responseBodySizeHistogram != nil {
		attributes := CurrentHTTPServer{}.MetricAttributes(md.ServerName, md.Req, md.StatusCode, md.AdditionalAttributes)
		o := metric.WithAttributeSet(attribute.NewSet(attributes...))
		recordOpts := metricRecordOptionPool.Get().(*[]metric.RecordOption)
		*recordOpts = append(*recordOpts, o)
		s.requestBodySizeHistogram.Record(ctx, md.RequestSize, *recordOpts...)
		s.responseBodySizeHistogram.Record(ctx, md.ResponseSize, *recordOpts...)
		s.requestDurationHistogram.Record(ctx, md.ElapsedTime, o)
		*recordOpts = (*recordOpts)[:0]
		metricRecordOptionPool.Put(recordOpts)
	}
}

func NewHTTPServer(meter metric.Meter) HTTPServer {
	env := strings.ToLower(os.Getenv(OTelSemConvStabilityOptIn))
	duplicate := env == "http/dup"
	server := HTTPServer{
		duplicate: duplicate,
	}
	server.requestBytesCounter, server.responseBytesCounter, server.serverLatencyMeasure = OldHTTPServer{}.createMeasures(meter)
	if duplicate {
		server.requestBodySizeHistogram, server.responseBodySizeHistogram, server.requestDurationHistogram = CurrentHTTPServer{}.createMeasures(meter)
	}
	return server
}

type HTTPClient struct {
	duplicate bool

	// old metrics
	requestBytesCounter  metric.Int64Counter
	responseBytesCounter metric.Int64Counter
	latencyMeasure       metric.Float64Histogram

	// new metrics
	requestBodySize metric.Int64Histogram
	requestDuration metric.Float64Histogram
}

func NewHTTPClient(meter metric.Meter) HTTPClient {
	env := strings.ToLower(os.Getenv(OTelSemConvStabilityOptIn))
	duplicate := env == "http/dup"
	client := HTTPClient{
		duplicate: duplicate,
	}
	client.requestBytesCounter, client.responseBytesCounter, client.latencyMeasure = OldHTTPClient{}.createMeasures(meter)
	if duplicate {
		client.requestBodySize, client.requestDuration = CurrentHTTPClient{}.createMeasures(meter)
	}

	return client
}

// RequestTraceAttrs returns attributes for an HTTP request made by a client.
func (c HTTPClient) RequestTraceAttrs(req *http.Request) []attribute.KeyValue {
	if c.duplicate {
		return append(OldHTTPClient{}.RequestTraceAttrs(req), CurrentHTTPClient{}.RequestTraceAttrs(req)...)
	}
	return OldHTTPClient{}.RequestTraceAttrs(req)
}

// ResponseTraceAttrs returns metric attributes for an HTTP request made by a client.
func (c HTTPClient) ResponseTraceAttrs(resp *http.Response) []attribute.KeyValue {
	if c.duplicate {
		return append(OldHTTPClient{}.ResponseTraceAttrs(resp), CurrentHTTPClient{}.ResponseTraceAttrs(resp)...)
	}

	return OldHTTPClient{}.ResponseTraceAttrs(resp)
}

func (c HTTPClient) Status(code int) (codes.Code, string) {
	if code < 100 || code >= 600 {
		return codes.Error, fmt.Sprintf("Invalid HTTP status code %d", code)
	}
	if code >= 400 {
		return codes.Error, ""
	}
	return codes.Unset, ""
}

func (c HTTPClient) ErrorType(err error) attribute.KeyValue {
	if c.duplicate {
		return CurrentHTTPClient{}.ErrorType(err)
	}

	return attribute.KeyValue{}
}

type MetricOpts struct {
	measurement metric.MeasurementOption
	addOptions  metric.AddOption
}

func (o MetricOpts) MeasurementOption() metric.MeasurementOption {
	return o.measurement
}

func (o MetricOpts) AddOptions() metric.AddOption {
	return o.addOptions
}

func (c HTTPClient) MetricOptions(ma MetricAttributes) map[string]MetricOpts {
	opts := map[string]MetricOpts{}

	attributes := OldHTTPClient{}.MetricAttributes(ma.Req, ma.StatusCode, ma.AdditionalAttributes)
	set := metric.WithAttributeSet(attribute.NewSet(attributes...))
	opts["old"] = MetricOpts{
		measurement: set,
		addOptions:  set,
	}

	if c.duplicate {
		attributes := CurrentHTTPClient{}.MetricAttributes(ma.Req, ma.StatusCode, ma.AdditionalAttributes)
		set := metric.WithAttributeSet(attribute.NewSet(attributes...))
		opts["new"] = MetricOpts{
			measurement: set,
			addOptions:  set,
		}
	}

	return opts
}

func (s HTTPClient) RecordMetrics(ctx context.Context, md MetricData, opts map[string]MetricOpts) {
	if s.requestBytesCounter == nil || s.latencyMeasure == nil {
		// This will happen if an HTTPClient{} is used instead of NewHTTPClient().
		return
	}

	s.requestBytesCounter.Add(ctx, md.RequestSize, opts["old"].AddOptions())
	s.latencyMeasure.Record(ctx, md.ElapsedTime, opts["old"].MeasurementOption())

	if s.duplicate {
		s.requestBodySize.Record(ctx, md.RequestSize, opts["new"].MeasurementOption())
		s.requestDuration.Record(ctx, md.ElapsedTime, opts["new"].MeasurementOption())
	}
}

func (s HTTPClient) RecordResponseSize(ctx context.Context, responseData int64, opts map[string]MetricOpts) {
	if s.responseBytesCounter == nil {
		// This will happen if an HTTPClient{} is used instead of NewHTTPClient().
		return
	}

	s.responseBytesCounter.Add(ctx, responseData, opts["old"].AddOptions())
}
