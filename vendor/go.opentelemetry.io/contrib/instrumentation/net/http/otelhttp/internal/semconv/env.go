// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"

import (
	"fmt"
	"net/http"
	"os"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

type ResponseTelemetry struct {
	StatusCode int
	ReadBytes  int64
	ReadError  error
	WriteBytes int64
	WriteError error
}

type HTTPServer struct {
	duplicate bool
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
		return append(oldHTTPServer{}.RequestTraceAttrs(server, req), newHTTPServer{}.RequestTraceAttrs(server, req)...)
	}
	return oldHTTPServer{}.RequestTraceAttrs(server, req)
}

// ResponseTraceAttrs returns trace attributes for telemetry from an HTTP response.
//
// If any of the fields in the ResponseTelemetry are not set the attribute will be omitted.
func (s HTTPServer) ResponseTraceAttrs(resp ResponseTelemetry) []attribute.KeyValue {
	if s.duplicate {
		return append(oldHTTPServer{}.ResponseTraceAttrs(resp), newHTTPServer{}.ResponseTraceAttrs(resp)...)
	}
	return oldHTTPServer{}.ResponseTraceAttrs(resp)
}

// Route returns the attribute for the route.
func (s HTTPServer) Route(route string) attribute.KeyValue {
	return oldHTTPServer{}.Route(route)
}

func NewHTTPServer() HTTPServer {
	env := strings.ToLower(os.Getenv("OTEL_HTTP_CLIENT_COMPATIBILITY_MODE"))
	return HTTPServer{duplicate: env == "http/dup"}
}

// ServerStatus returns a span status code and message for an HTTP status code
// value returned by a server. Status codes in the 400-499 range are not
// returned as errors.
func ServerStatus(code int) (codes.Code, string) {
	if code < 100 || code >= 600 {
		return codes.Error, fmt.Sprintf("Invalid HTTP status code %d", code)
	}
	if code >= 500 {
		return codes.Error, ""
	}
	return codes.Unset, ""
}
