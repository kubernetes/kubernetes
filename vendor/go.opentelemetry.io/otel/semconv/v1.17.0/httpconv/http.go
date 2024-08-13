// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package httpconv provides OpenTelemetry HTTP semantic conventions for
// tracing telemetry.
package httpconv // import "go.opentelemetry.io/otel/semconv/v1.17.0/httpconv"

import (
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/semconv/internal/v2"
	semconv "go.opentelemetry.io/otel/semconv/v1.17.0"
)

var (
	nc = &internal.NetConv{
		NetHostNameKey:     semconv.NetHostNameKey,
		NetHostPortKey:     semconv.NetHostPortKey,
		NetPeerNameKey:     semconv.NetPeerNameKey,
		NetPeerPortKey:     semconv.NetPeerPortKey,
		NetSockPeerAddrKey: semconv.NetSockPeerAddrKey,
		NetSockPeerPortKey: semconv.NetSockPeerPortKey,
		NetTransportOther:  semconv.NetTransportOther,
		NetTransportTCP:    semconv.NetTransportTCP,
		NetTransportUDP:    semconv.NetTransportUDP,
		NetTransportInProc: semconv.NetTransportInProc,
	}

	hc = &internal.HTTPConv{
		NetConv: nc,

		EnduserIDKey:                 semconv.EnduserIDKey,
		HTTPClientIPKey:              semconv.HTTPClientIPKey,
		HTTPFlavorKey:                semconv.HTTPFlavorKey,
		HTTPMethodKey:                semconv.HTTPMethodKey,
		HTTPRequestContentLengthKey:  semconv.HTTPRequestContentLengthKey,
		HTTPResponseContentLengthKey: semconv.HTTPResponseContentLengthKey,
		HTTPRouteKey:                 semconv.HTTPRouteKey,
		HTTPSchemeHTTP:               semconv.HTTPSchemeHTTP,
		HTTPSchemeHTTPS:              semconv.HTTPSchemeHTTPS,
		HTTPStatusCodeKey:            semconv.HTTPStatusCodeKey,
		HTTPTargetKey:                semconv.HTTPTargetKey,
		HTTPURLKey:                   semconv.HTTPURLKey,
		HTTPUserAgentKey:             semconv.HTTPUserAgentKey,
	}
)

// ClientResponse returns trace attributes for an HTTP response received by a
// client from a server. It will return the following attributes if the related
// values are defined in resp: "http.status.code",
// "http.response_content_length".
//
// This does not add all OpenTelemetry required attributes for an HTTP event,
// it assumes ClientRequest was used to create the span with a complete set of
// attributes. If a complete set of attributes can be generated using the
// request contained in resp. For example:
//
//	append(ClientResponse(resp), ClientRequest(resp.Request)...)
func ClientResponse(resp *http.Response) []attribute.KeyValue {
	return hc.ClientResponse(resp)
}

// ClientRequest returns trace attributes for an HTTP request made by a client.
// The following attributes are always returned: "http.url", "http.flavor",
// "http.method", "net.peer.name". The following attributes are returned if the
// related values are defined in req: "net.peer.port", "http.user_agent",
// "http.request_content_length", "enduser.id".
func ClientRequest(req *http.Request) []attribute.KeyValue {
	return hc.ClientRequest(req)
}

// ClientStatus returns a span status code and message for an HTTP status code
// value received by a client.
func ClientStatus(code int) (codes.Code, string) {
	return hc.ClientStatus(code)
}

// ServerRequest returns trace attributes for an HTTP request received by a
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
//
// The following attributes are always returned: "http.method", "http.scheme",
// "http.flavor", "http.target", "net.host.name". The following attributes are
// returned if they related values are defined in req: "net.host.port",
// "net.sock.peer.addr", "net.sock.peer.port", "http.user_agent", "enduser.id",
// "http.client_ip".
func ServerRequest(server string, req *http.Request) []attribute.KeyValue {
	return hc.ServerRequest(server, req)
}

// ServerStatus returns a span status code and message for an HTTP status code
// value returned by a server. Status codes in the 400-499 range are not
// returned as errors.
func ServerStatus(code int) (codes.Code, string) {
	return hc.ServerStatus(code)
}

// RequestHeader returns the contents of h as attributes.
//
// Instrumentation should require an explicit configuration of which headers to
// captured and then prune what they pass here. Including all headers can be a
// security risk - explicit configuration helps avoid leaking sensitive
// information.
//
// The User-Agent header is already captured in the http.user_agent attribute
// from ClientRequest and ServerRequest. Instrumentation may provide an option
// to capture that header here even though it is not recommended. Otherwise,
// instrumentation should filter that out of what is passed.
func RequestHeader(h http.Header) []attribute.KeyValue {
	return hc.RequestHeader(h)
}

// ResponseHeader returns the contents of h as attributes.
//
// Instrumentation should require an explicit configuration of which headers to
// captured and then prune what they pass here. Including all headers can be a
// security risk - explicit configuration helps avoid leaking sensitive
// information.
//
// The User-Agent header is already captured in the http.user_agent attribute
// from ClientRequest and ServerRequest. Instrumentation may provide an option
// to capture that header here even though it is not recommended. Otherwise,
// instrumentation should filter that out of what is passed.
func ResponseHeader(h http.Header) []attribute.KeyValue {
	return hc.ResponseHeader(h)
}
