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

package semconv // import "go.opentelemetry.io/otel/semconv"

import (
	"fmt"
	"net"
	"net/http"
	"strconv"
	"strings"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
)

// NetAttributesFromHTTPRequest generates attributes of the net
// namespace as specified by the OpenTelemetry specification for a
// span.  The network parameter is a string that net.Dial function
// from standard library can understand.
func NetAttributesFromHTTPRequest(network string, request *http.Request) []attribute.KeyValue {
	attrs := []attribute.KeyValue{}

	switch network {
	case "tcp", "tcp4", "tcp6":
		attrs = append(attrs, NetTransportTCP)
	case "udp", "udp4", "udp6":
		attrs = append(attrs, NetTransportUDP)
	case "ip", "ip4", "ip6":
		attrs = append(attrs, NetTransportIP)
	case "unix", "unixgram", "unixpacket":
		attrs = append(attrs, NetTransportUnix)
	default:
		attrs = append(attrs, NetTransportOther)
	}

	peerName, peerIP, peerPort := "", "", 0
	{
		hostPart := request.RemoteAddr
		portPart := ""
		if idx := strings.LastIndex(hostPart, ":"); idx >= 0 {
			hostPart = request.RemoteAddr[:idx]
			portPart = request.RemoteAddr[idx+1:]
		}
		if hostPart != "" {
			if ip := net.ParseIP(hostPart); ip != nil {
				peerIP = ip.String()
			} else {
				peerName = hostPart
			}

			if portPart != "" {
				numPort, err := strconv.ParseUint(portPart, 10, 16)
				if err == nil {
					peerPort = (int)(numPort)
				} else {
					peerName, peerIP = "", ""
				}
			}
		}
	}
	if peerName != "" {
		attrs = append(attrs, NetPeerNameKey.String(peerName))
	}
	if peerIP != "" {
		attrs = append(attrs, NetPeerIPKey.String(peerIP))
	}
	if peerPort != 0 {
		attrs = append(attrs, NetPeerPortKey.Int(peerPort))
	}

	hostIP, hostName, hostPort := "", "", 0
	for _, someHost := range []string{request.Host, request.Header.Get("Host"), request.URL.Host} {
		hostPart := ""
		if idx := strings.LastIndex(someHost, ":"); idx >= 0 {
			strPort := someHost[idx+1:]
			numPort, err := strconv.ParseUint(strPort, 10, 16)
			if err == nil {
				hostPort = (int)(numPort)
			}
			hostPart = someHost[:idx]
		} else {
			hostPart = someHost
		}
		if hostPart != "" {
			ip := net.ParseIP(hostPart)
			if ip != nil {
				hostIP = ip.String()
			} else {
				hostName = hostPart
			}
			break
		} else {
			hostPort = 0
		}
	}
	if hostIP != "" {
		attrs = append(attrs, NetHostIPKey.String(hostIP))
	}
	if hostName != "" {
		attrs = append(attrs, NetHostNameKey.String(hostName))
	}
	if hostPort != 0 {
		attrs = append(attrs, NetHostPortKey.Int(hostPort))
	}

	return attrs
}

// EndUserAttributesFromHTTPRequest generates attributes of the
// enduser namespace as specified by the OpenTelemetry specification
// for a span.
func EndUserAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	if username, _, ok := request.BasicAuth(); ok {
		return []attribute.KeyValue{EnduserIDKey.String(username)}
	}
	return nil
}

// HTTPClientAttributesFromHTTPRequest generates attributes of the
// http namespace as specified by the OpenTelemetry specification for
// a span on the client side.
func HTTPClientAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	attrs := []attribute.KeyValue{}

	if request.Method != "" {
		attrs = append(attrs, HTTPMethodKey.String(request.Method))
	} else {
		attrs = append(attrs, HTTPMethodKey.String(http.MethodGet))
	}

	attrs = append(attrs, HTTPURLKey.String(request.URL.String()))

	return append(attrs, httpCommonAttributesFromHTTPRequest(request)...)
}

func httpCommonAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	attrs := []attribute.KeyValue{}
	if ua := request.UserAgent(); ua != "" {
		attrs = append(attrs, HTTPUserAgentKey.String(ua))
	}
	if request.ContentLength > 0 {
		attrs = append(attrs, HTTPRequestContentLengthKey.Int64(request.ContentLength))
	}

	return append(attrs, httpBasicAttributesFromHTTPRequest(request)...)
}

func httpBasicAttributesFromHTTPRequest(request *http.Request) []attribute.KeyValue {
	// as these attributes are used by HTTPServerMetricAttributesFromHTTPRequest, they should be low-cardinality
	attrs := []attribute.KeyValue{}

	if request.TLS != nil {
		attrs = append(attrs, HTTPSchemeHTTPS)
	} else {
		attrs = append(attrs, HTTPSchemeHTTP)
	}

	if request.Host != "" {
		attrs = append(attrs, HTTPHostKey.String(request.Host))
	}

	flavor := ""
	if request.ProtoMajor == 1 {
		flavor = fmt.Sprintf("1.%d", request.ProtoMinor)
	} else if request.ProtoMajor == 2 {
		flavor = "2"
	}
	if flavor != "" {
		attrs = append(attrs, HTTPFlavorKey.String(flavor))
	}

	return attrs
}

// HTTPServerMetricAttributesFromHTTPRequest generates low-cardinality attributes
// to be used with server-side HTTP metrics.
func HTTPServerMetricAttributesFromHTTPRequest(serverName string, request *http.Request) []attribute.KeyValue {
	attrs := []attribute.KeyValue{}
	if serverName != "" {
		attrs = append(attrs, HTTPServerNameKey.String(serverName))
	}
	return append(attrs, httpBasicAttributesFromHTTPRequest(request)...)
}

// HTTPServerAttributesFromHTTPRequest generates attributes of the
// http namespace as specified by the OpenTelemetry specification for
// a span on the server side. Currently, only basic authentication is
// supported.
func HTTPServerAttributesFromHTTPRequest(serverName, route string, request *http.Request) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		HTTPMethodKey.String(request.Method),
		HTTPTargetKey.String(request.RequestURI),
	}

	if serverName != "" {
		attrs = append(attrs, HTTPServerNameKey.String(serverName))
	}
	if route != "" {
		attrs = append(attrs, HTTPRouteKey.String(route))
	}
	if values, ok := request.Header["X-Forwarded-For"]; ok && len(values) > 0 {
		attrs = append(attrs, HTTPClientIPKey.String(values[0]))
	}

	return append(attrs, httpCommonAttributesFromHTTPRequest(request)...)
}

// HTTPAttributesFromHTTPStatusCode generates attributes of the http
// namespace as specified by the OpenTelemetry specification for a
// span.
func HTTPAttributesFromHTTPStatusCode(code int) []attribute.KeyValue {
	attrs := []attribute.KeyValue{
		HTTPStatusCodeKey.Int(code),
	}
	return attrs
}

type codeRange struct {
	fromInclusive int
	toInclusive   int
}

func (r codeRange) contains(code int) bool {
	return r.fromInclusive <= code && code <= r.toInclusive
}

var validRangesPerCategory = map[int][]codeRange{
	1: {
		{http.StatusContinue, http.StatusEarlyHints},
	},
	2: {
		{http.StatusOK, http.StatusAlreadyReported},
		{http.StatusIMUsed, http.StatusIMUsed},
	},
	3: {
		{http.StatusMultipleChoices, http.StatusUseProxy},
		{http.StatusTemporaryRedirect, http.StatusPermanentRedirect},
	},
	4: {
		{http.StatusBadRequest, http.StatusTeapot}, // yes, teapot is so useful…
		{http.StatusMisdirectedRequest, http.StatusUpgradeRequired},
		{http.StatusPreconditionRequired, http.StatusTooManyRequests},
		{http.StatusRequestHeaderFieldsTooLarge, http.StatusRequestHeaderFieldsTooLarge},
		{http.StatusUnavailableForLegalReasons, http.StatusUnavailableForLegalReasons},
	},
	5: {
		{http.StatusInternalServerError, http.StatusLoopDetected},
		{http.StatusNotExtended, http.StatusNetworkAuthenticationRequired},
	},
}

// SpanStatusFromHTTPStatusCode generates a status code and a message
// as specified by the OpenTelemetry specification for a span.
func SpanStatusFromHTTPStatusCode(code int) (codes.Code, string) {
	spanCode, valid := validateHTTPStatusCode(code)
	if !valid {
		return spanCode, fmt.Sprintf("Invalid HTTP status code %d", code)
	}
	return spanCode, ""
}

// Validates the HTTP status code and returns corresponding span status code.
// If the `code` is not a valid HTTP status code, returns span status Error
// and false.
func validateHTTPStatusCode(code int) (codes.Code, bool) {
	category := code / 100
	ranges, ok := validRangesPerCategory[category]
	if !ok {
		return codes.Error, false
	}
	ok = false
	for _, crange := range ranges {
		ok = crange.contains(code)
		if ok {
			break
		}
	}
	if !ok {
		return codes.Error, false
	}
	if category > 0 && category < 4 {
		return codes.Unset, true
	}
	return codes.Error, true
}
