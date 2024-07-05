// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package semconv // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"

import (
	"net/http"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	semconvNew "go.opentelemetry.io/otel/semconv/v1.24.0"
)

type newHTTPServer struct{}

// TraceRequest returns trace attributes for an HTTP request received by a
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
func (n newHTTPServer) RequestTraceAttrs(server string, req *http.Request) []attribute.KeyValue {
	count := 3 // ServerAddress, Method, Scheme

	var host string
	var p int
	if server == "" {
		host, p = splitHostPort(req.Host)
	} else {
		// Prioritize the primary server name.
		host, p = splitHostPort(server)
		if p < 0 {
			_, p = splitHostPort(req.Host)
		}
	}

	hostPort := requiredHTTPPort(req.TLS != nil, p)
	if hostPort > 0 {
		count++
	}

	method, methodOriginal := n.method(req.Method)
	if methodOriginal != (attribute.KeyValue{}) {
		count++
	}

	scheme := n.scheme(req.TLS != nil)

	if peer, peerPort := splitHostPort(req.RemoteAddr); peer != "" {
		// The Go HTTP server sets RemoteAddr to "IP:port", this will not be a
		// file-path that would be interpreted with a sock family.
		count++
		if peerPort > 0 {
			count++
		}
	}

	useragent := req.UserAgent()
	if useragent != "" {
		count++
	}

	clientIP := serverClientIP(req.Header.Get("X-Forwarded-For"))
	if clientIP != "" {
		count++
	}

	if req.URL != nil && req.URL.Path != "" {
		count++
	}

	protoName, protoVersion := netProtocol(req.Proto)
	if protoName != "" && protoName != "http" {
		count++
	}
	if protoVersion != "" {
		count++
	}

	attrs := make([]attribute.KeyValue, 0, count)
	attrs = append(attrs,
		semconvNew.ServerAddress(host),
		method,
		scheme,
	)

	if hostPort > 0 {
		attrs = append(attrs, semconvNew.ServerPort(hostPort))
	}
	if methodOriginal != (attribute.KeyValue{}) {
		attrs = append(attrs, methodOriginal)
	}

	if peer, peerPort := splitHostPort(req.RemoteAddr); peer != "" {
		// The Go HTTP server sets RemoteAddr to "IP:port", this will not be a
		// file-path that would be interpreted with a sock family.
		attrs = append(attrs, semconvNew.NetworkPeerAddress(peer))
		if peerPort > 0 {
			attrs = append(attrs, semconvNew.NetworkPeerPort(peerPort))
		}
	}

	if useragent := req.UserAgent(); useragent != "" {
		attrs = append(attrs, semconvNew.UserAgentOriginal(useragent))
	}

	if clientIP != "" {
		attrs = append(attrs, semconvNew.ClientAddress(clientIP))
	}

	if req.URL != nil && req.URL.Path != "" {
		attrs = append(attrs, semconvNew.URLPath(req.URL.Path))
	}

	if protoName != "" && protoName != "http" {
		attrs = append(attrs, semconvNew.NetworkProtocolName(protoName))
	}
	if protoVersion != "" {
		attrs = append(attrs, semconvNew.NetworkProtocolVersion(protoVersion))
	}

	return attrs
}

func (n newHTTPServer) method(method string) (attribute.KeyValue, attribute.KeyValue) {
	if method == "" {
		return semconvNew.HTTPRequestMethodGet, attribute.KeyValue{}
	}
	if attr, ok := methodLookup[method]; ok {
		return attr, attribute.KeyValue{}
	}

	orig := semconvNew.HTTPRequestMethodOriginal(method)
	if attr, ok := methodLookup[strings.ToUpper(method)]; ok {
		return attr, orig
	}
	return semconvNew.HTTPRequestMethodGet, orig
}

func (n newHTTPServer) scheme(https bool) attribute.KeyValue { // nolint:revive
	if https {
		return semconvNew.URLScheme("https")
	}
	return semconvNew.URLScheme("http")
}

// TraceResponse returns trace attributes for telemetry from an HTTP response.
//
// If any of the fields in the ResponseTelemetry are not set the attribute will be omitted.
func (n newHTTPServer) ResponseTraceAttrs(resp ResponseTelemetry) []attribute.KeyValue {
	var count int

	if resp.ReadBytes > 0 {
		count++
	}
	if resp.WriteBytes > 0 {
		count++
	}
	if resp.StatusCode > 0 {
		count++
	}

	attributes := make([]attribute.KeyValue, 0, count)

	if resp.ReadBytes > 0 {
		attributes = append(attributes,
			semconvNew.HTTPRequestBodySize(int(resp.ReadBytes)),
		)
	}
	if resp.WriteBytes > 0 {
		attributes = append(attributes,
			semconvNew.HTTPResponseBodySize(int(resp.WriteBytes)),
		)
	}
	if resp.StatusCode > 0 {
		attributes = append(attributes,
			semconvNew.HTTPResponseStatusCode(resp.StatusCode),
		)
	}

	return attributes
}

// Route returns the attribute for the route.
func (n newHTTPServer) Route(route string) attribute.KeyValue {
	return semconvNew.HTTPRoute(route)
}
