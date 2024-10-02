// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"

import (
	"errors"
	"io"
	"net/http"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconvutil"
	"go.opentelemetry.io/otel/attribute"
	semconv "go.opentelemetry.io/otel/semconv/v1.20.0"
)

type oldHTTPServer struct{}

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
func (o oldHTTPServer) RequestTraceAttrs(server string, req *http.Request) []attribute.KeyValue {
	return semconvutil.HTTPServerRequest(server, req)
}

// ResponseTraceAttrs returns trace attributes for telemetry from an HTTP response.
//
// If any of the fields in the ResponseTelemetry are not set the attribute will be omitted.
func (o oldHTTPServer) ResponseTraceAttrs(resp ResponseTelemetry) []attribute.KeyValue {
	attributes := []attribute.KeyValue{}

	if resp.ReadBytes > 0 {
		attributes = append(attributes, semconv.HTTPRequestContentLength(int(resp.ReadBytes)))
	}
	if resp.ReadError != nil && !errors.Is(resp.ReadError, io.EOF) {
		// This is not in the semantic conventions, but is historically provided
		attributes = append(attributes, attribute.String("http.read_error", resp.ReadError.Error()))
	}
	if resp.WriteBytes > 0 {
		attributes = append(attributes, semconv.HTTPResponseContentLength(int(resp.WriteBytes)))
	}
	if resp.StatusCode > 0 {
		attributes = append(attributes, semconv.HTTPStatusCode(resp.StatusCode))
	}
	if resp.WriteError != nil && !errors.Is(resp.WriteError, io.EOF) {
		// This is not in the semantic conventions, but is historically provided
		attributes = append(attributes, attribute.String("http.write_error", resp.WriteError.Error()))
	}

	return attributes
}

// Route returns the attribute for the route.
func (o oldHTTPServer) Route(route string) attribute.KeyValue {
	return semconv.HTTPRoute(route)
}

// HTTPStatusCode returns the attribute for the HTTP status code.
// This is a temporary function needed by metrics.  This will be removed when MetricsRequest is added.
func HTTPStatusCode(status int) attribute.KeyValue {
	return semconv.HTTPStatusCode(status)
}
