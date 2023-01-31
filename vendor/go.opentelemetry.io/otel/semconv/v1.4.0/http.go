// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.4.0"

import (
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/semconv/internal"
	"go.opentelemetry.io/otel/trace"
)

// HTTP scheme attributes.
var (
	HTTPSchemeHTTP  = HTTPSchemeKey.String("http")
	HTTPSchemeHTTPS = HTTPSchemeKey.String("https")
)

var sc = &internal.SemanticConventions{
	EnduserIDKey:                EnduserIDKey,
	HTTPClientIPKey:             HTTPClientIPKey,
	HTTPFlavorKey:               HTTPFlavorKey,
	HTTPHostKey:                 HTTPHostKey,
	HTTPMethodKey:               HTTPMethodKey,
	HTTPRequestContentLengthKey: HTTPRequestContentLengthKey,
	HTTPRouteKey:                HTTPRouteKey,
	HTTPSchemeHTTP:              HTTPSchemeHTTP,
	HTTPSchemeHTTPS:             HTTPSchemeHTTPS,
	HTTPServerNameKey:           HTTPServerNameKey,
	HTTPStatusCodeKey:           HTTPStatusCodeKey,
	HTTPTargetKey:               HTTPTargetKey,
	HTTPURLKey:                  HTTPURLKey,
	HTTPUserAgentKey:            HTTPUserAgentKey,
	NetHostIPKey:                NetHostIPKey,
	NetHostNameKey:              NetHostNameKey,
	NetHostPortKey:              NetHostPortKey,
	NetPeerIPKey:                NetPeerIPKey,
	NetPeerNameKey:              NetPeerNameKey,
	NetPeerPortKey:              NetPeerPortKey,
	NetTransportIP:              NetTransportIP,
	NetTransportOther:           NetTransportOther,
	NetTransportTCP:             NetTransportTCP,
	NetTransportUDP:             NetTransportUDP,
	NetTransportUnix:            NetTransportUnix,
}

// NetAttributesFromHTTPRequest generates attributes of the net
// namespace as specified by the OpenTelemetry specification for a
// span.  The network parameter is a string that net.Dial function
// from standard library can understand.
func NetAttributesFromHTTPRequest(network string, request *http.Request) []attribute.KeyValue {
	return sc.NetAttributesFromHTTPRequest(network, request)
}

// EndUserAttributesFromHTTPRequest generates attributes of the
// enduser namespace as specified by the OpenTelemetry specification
// for a span.
func EndUserAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	return sc.EndUserAttributesFromHTTPRequest(request)
}

// HTTPClientAttributesFromHTTPRequest generates attributes of the
// http namespace as specified by the OpenTelemetry specification for
// a span on the client side.
func HTTPClientAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	return sc.HTTPClientAttributesFromHTTPRequest(request)
}

// HTTPServerMetricAttributesFromHTTPRequest generates low-cardinality attributes
// to be used with server-side HTTP metrics.
func HTTPServerMetricAttributesFromHTTPRequest(serverName string, request *http.Request) []attribute.KeyValue {
	return sc.HTTPServerMetricAttributesFromHTTPRequest(serverName, request)
}

// HTTPServerAttributesFromHTTPRequest generates attributes of the
// http namespace as specified by the OpenTelemetry specification for
// a span on the server side. Currently, only basic authentication is
// supported.
func HTTPServerAttributesFromHTTPRequest(serverName, route string, request *http.Request) []attribute.KeyValue {
	return sc.HTTPServerAttributesFromHTTPRequest(serverName, route, request)
}

// HTTPAttributesFromHTTPStatusCode generates attributes of the http
// namespace as specified by the OpenTelemetry specification for a
// span.
func HTTPAttributesFromHTTPStatusCode(code int) []attribute.KeyValue {
	return sc.HTTPAttributesFromHTTPStatusCode(code)
}

// SpanStatusFromHTTPStatusCode generates a status code and a message
// as specified by the OpenTelemetry specification for a span.
func SpanStatusFromHTTPStatusCode(code int) (codes.Code, string) {
	return internal.SpanStatusFromHTTPStatusCode(code)
}

// SpanStatusFromHTTPStatusCodeAndSpanKind generates a status code and a message
// as specified by the OpenTelemetry specification for a span.
// Exclude 4xx for SERVER to set the appropriate status.
func SpanStatusFromHTTPStatusCodeAndSpanKind(code int, spanKind trace.SpanKind) (codes.Code, string) {
	return internal.SpanStatusFromHTTPStatusCodeAndSpanKind(code, spanKind)
}
