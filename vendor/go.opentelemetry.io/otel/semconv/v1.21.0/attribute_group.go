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

// Code generated from semantic convention specification. DO NOT EDIT.

package semconv // import "go.opentelemetry.io/otel/semconv/v1.21.0"

import "go.opentelemetry.io/otel/attribute"

// These attributes may be used to describe the client in a connection-based
// network interaction where there is one side that initiates the connection
// (the client is the side that initiates the connection). This covers all TCP
// network interactions since TCP is connection-based and one side initiates
// the connection (an exception is made for peer-to-peer communication over TCP
// where the "user-facing" surface of the protocol / API does not expose a
// clear notion of client and server). This also covers UDP network
// interactions where one side initiates the interaction, e.g. QUIC (HTTP/3)
// and DNS.
const (
	// ClientAddressKey is the attribute Key conforming to the "client.address"
	// semantic conventions. It represents the client address - unix domain
	// socket name, IPv4 or IPv6 address.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/tmp/my.sock', '10.1.2.80'
	// Note: When observed from the server side, and when communicating through
	// an intermediary, `client.address` SHOULD represent client address behind
	// any intermediaries (e.g. proxies) if it's available.
	ClientAddressKey = attribute.Key("client.address")

	// ClientPortKey is the attribute Key conforming to the "client.port"
	// semantic conventions. It represents the client port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 65123
	// Note: When observed from the server side, and when communicating through
	// an intermediary, `client.port` SHOULD represent client port behind any
	// intermediaries (e.g. proxies) if it's available.
	ClientPortKey = attribute.Key("client.port")

	// ClientSocketAddressKey is the attribute Key conforming to the
	// "client.socket.address" semantic conventions. It represents the
	// immediate client peer address - unix domain socket name, IPv4 or IPv6
	// address.
	//
	// Type: string
	// RequirementLevel: Recommended (If different than `client.address`.)
	// Stability: stable
	// Examples: '/tmp/my.sock', '127.0.0.1'
	ClientSocketAddressKey = attribute.Key("client.socket.address")

	// ClientSocketPortKey is the attribute Key conforming to the
	// "client.socket.port" semantic conventions. It represents the immediate
	// client peer port number
	//
	// Type: int
	// RequirementLevel: Recommended (If different than `client.port`.)
	// Stability: stable
	// Examples: 35555
	ClientSocketPortKey = attribute.Key("client.socket.port")
)

// ClientAddress returns an attribute KeyValue conforming to the
// "client.address" semantic conventions. It represents the client address -
// unix domain socket name, IPv4 or IPv6 address.
func ClientAddress(val string) attribute.KeyValue {
	return ClientAddressKey.String(val)
}

// ClientPort returns an attribute KeyValue conforming to the "client.port"
// semantic conventions. It represents the client port number
func ClientPort(val int) attribute.KeyValue {
	return ClientPortKey.Int(val)
}

// ClientSocketAddress returns an attribute KeyValue conforming to the
// "client.socket.address" semantic conventions. It represents the immediate
// client peer address - unix domain socket name, IPv4 or IPv6 address.
func ClientSocketAddress(val string) attribute.KeyValue {
	return ClientSocketAddressKey.String(val)
}

// ClientSocketPort returns an attribute KeyValue conforming to the
// "client.socket.port" semantic conventions. It represents the immediate
// client peer port number
func ClientSocketPort(val int) attribute.KeyValue {
	return ClientSocketPortKey.Int(val)
}

// Describes deprecated HTTP attributes.
const (
	// HTTPMethodKey is the attribute Key conforming to the "http.method"
	// semantic conventions. It represents the deprecated, use
	// `http.request.method` instead.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'GET', 'POST', 'HEAD'
	HTTPMethodKey = attribute.Key("http.method")

	// HTTPStatusCodeKey is the attribute Key conforming to the
	// "http.status_code" semantic conventions. It represents the deprecated,
	// use `http.response.status_code` instead.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 200
	HTTPStatusCodeKey = attribute.Key("http.status_code")

	// HTTPSchemeKey is the attribute Key conforming to the "http.scheme"
	// semantic conventions. It represents the deprecated, use `url.scheme`
	// instead.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'http', 'https'
	HTTPSchemeKey = attribute.Key("http.scheme")

	// HTTPURLKey is the attribute Key conforming to the "http.url" semantic
	// conventions. It represents the deprecated, use `url.full` instead.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'https://www.foo.bar/search?q=OpenTelemetry#SemConv'
	HTTPURLKey = attribute.Key("http.url")

	// HTTPTargetKey is the attribute Key conforming to the "http.target"
	// semantic conventions. It represents the deprecated, use `url.path` and
	// `url.query` instead.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/search?q=OpenTelemetry#SemConv'
	HTTPTargetKey = attribute.Key("http.target")

	// HTTPRequestContentLengthKey is the attribute Key conforming to the
	// "http.request_content_length" semantic conventions. It represents the
	// deprecated, use `http.request.body.size` instead.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 3495
	HTTPRequestContentLengthKey = attribute.Key("http.request_content_length")

	// HTTPResponseContentLengthKey is the attribute Key conforming to the
	// "http.response_content_length" semantic conventions. It represents the
	// deprecated, use `http.response.body.size` instead.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 3495
	HTTPResponseContentLengthKey = attribute.Key("http.response_content_length")
)

// HTTPMethod returns an attribute KeyValue conforming to the "http.method"
// semantic conventions. It represents the deprecated, use
// `http.request.method` instead.
func HTTPMethod(val string) attribute.KeyValue {
	return HTTPMethodKey.String(val)
}

// HTTPStatusCode returns an attribute KeyValue conforming to the
// "http.status_code" semantic conventions. It represents the deprecated, use
// `http.response.status_code` instead.
func HTTPStatusCode(val int) attribute.KeyValue {
	return HTTPStatusCodeKey.Int(val)
}

// HTTPScheme returns an attribute KeyValue conforming to the "http.scheme"
// semantic conventions. It represents the deprecated, use `url.scheme`
// instead.
func HTTPScheme(val string) attribute.KeyValue {
	return HTTPSchemeKey.String(val)
}

// HTTPURL returns an attribute KeyValue conforming to the "http.url"
// semantic conventions. It represents the deprecated, use `url.full` instead.
func HTTPURL(val string) attribute.KeyValue {
	return HTTPURLKey.String(val)
}

// HTTPTarget returns an attribute KeyValue conforming to the "http.target"
// semantic conventions. It represents the deprecated, use `url.path` and
// `url.query` instead.
func HTTPTarget(val string) attribute.KeyValue {
	return HTTPTargetKey.String(val)
}

// HTTPRequestContentLength returns an attribute KeyValue conforming to the
// "http.request_content_length" semantic conventions. It represents the
// deprecated, use `http.request.body.size` instead.
func HTTPRequestContentLength(val int) attribute.KeyValue {
	return HTTPRequestContentLengthKey.Int(val)
}

// HTTPResponseContentLength returns an attribute KeyValue conforming to the
// "http.response_content_length" semantic conventions. It represents the
// deprecated, use `http.response.body.size` instead.
func HTTPResponseContentLength(val int) attribute.KeyValue {
	return HTTPResponseContentLengthKey.Int(val)
}

// These attributes may be used for any network related operation.
const (
	// NetSockPeerNameKey is the attribute Key conforming to the
	// "net.sock.peer.name" semantic conventions. It represents the deprecated,
	// use `server.socket.domain` on client spans.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/var/my.sock'
	NetSockPeerNameKey = attribute.Key("net.sock.peer.name")

	// NetSockPeerAddrKey is the attribute Key conforming to the
	// "net.sock.peer.addr" semantic conventions. It represents the deprecated,
	// use `server.socket.address` on client spans and `client.socket.address`
	// on server spans.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '192.168.0.1'
	NetSockPeerAddrKey = attribute.Key("net.sock.peer.addr")

	// NetSockPeerPortKey is the attribute Key conforming to the
	// "net.sock.peer.port" semantic conventions. It represents the deprecated,
	// use `server.socket.port` on client spans and `client.socket.port` on
	// server spans.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 65531
	NetSockPeerPortKey = attribute.Key("net.sock.peer.port")

	// NetPeerNameKey is the attribute Key conforming to the "net.peer.name"
	// semantic conventions. It represents the deprecated, use `server.address`
	// on client spans and `client.address` on server spans.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'example.com'
	NetPeerNameKey = attribute.Key("net.peer.name")

	// NetPeerPortKey is the attribute Key conforming to the "net.peer.port"
	// semantic conventions. It represents the deprecated, use `server.port` on
	// client spans and `client.port` on server spans.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	NetPeerPortKey = attribute.Key("net.peer.port")

	// NetHostNameKey is the attribute Key conforming to the "net.host.name"
	// semantic conventions. It represents the deprecated, use
	// `server.address`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'example.com'
	NetHostNameKey = attribute.Key("net.host.name")

	// NetHostPortKey is the attribute Key conforming to the "net.host.port"
	// semantic conventions. It represents the deprecated, use `server.port`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	NetHostPortKey = attribute.Key("net.host.port")

	// NetSockHostAddrKey is the attribute Key conforming to the
	// "net.sock.host.addr" semantic conventions. It represents the deprecated,
	// use `server.socket.address`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '/var/my.sock'
	NetSockHostAddrKey = attribute.Key("net.sock.host.addr")

	// NetSockHostPortKey is the attribute Key conforming to the
	// "net.sock.host.port" semantic conventions. It represents the deprecated,
	// use `server.socket.port`.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 8080
	NetSockHostPortKey = attribute.Key("net.sock.host.port")

	// NetTransportKey is the attribute Key conforming to the "net.transport"
	// semantic conventions. It represents the deprecated, use
	// `network.transport`.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: deprecated
	NetTransportKey = attribute.Key("net.transport")

	// NetProtocolNameKey is the attribute Key conforming to the
	// "net.protocol.name" semantic conventions. It represents the deprecated,
	// use `network.protocol.name`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: 'amqp', 'http', 'mqtt'
	NetProtocolNameKey = attribute.Key("net.protocol.name")

	// NetProtocolVersionKey is the attribute Key conforming to the
	// "net.protocol.version" semantic conventions. It represents the
	// deprecated, use `network.protocol.version`.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: deprecated
	// Examples: '3.1.1'
	NetProtocolVersionKey = attribute.Key("net.protocol.version")

	// NetSockFamilyKey is the attribute Key conforming to the
	// "net.sock.family" semantic conventions. It represents the deprecated,
	// use `network.transport` and `network.type`.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: deprecated
	NetSockFamilyKey = attribute.Key("net.sock.family")
)

var (
	// ip_tcp
	NetTransportTCP = NetTransportKey.String("ip_tcp")
	// ip_udp
	NetTransportUDP = NetTransportKey.String("ip_udp")
	// Named or anonymous pipe
	NetTransportPipe = NetTransportKey.String("pipe")
	// In-process communication
	NetTransportInProc = NetTransportKey.String("inproc")
	// Something else (non IP-based)
	NetTransportOther = NetTransportKey.String("other")
)

var (
	// IPv4 address
	NetSockFamilyInet = NetSockFamilyKey.String("inet")
	// IPv6 address
	NetSockFamilyInet6 = NetSockFamilyKey.String("inet6")
	// Unix domain socket path
	NetSockFamilyUnix = NetSockFamilyKey.String("unix")
)

// NetSockPeerName returns an attribute KeyValue conforming to the
// "net.sock.peer.name" semantic conventions. It represents the deprecated, use
// `server.socket.domain` on client spans.
func NetSockPeerName(val string) attribute.KeyValue {
	return NetSockPeerNameKey.String(val)
}

// NetSockPeerAddr returns an attribute KeyValue conforming to the
// "net.sock.peer.addr" semantic conventions. It represents the deprecated, use
// `server.socket.address` on client spans and `client.socket.address` on
// server spans.
func NetSockPeerAddr(val string) attribute.KeyValue {
	return NetSockPeerAddrKey.String(val)
}

// NetSockPeerPort returns an attribute KeyValue conforming to the
// "net.sock.peer.port" semantic conventions. It represents the deprecated, use
// `server.socket.port` on client spans and `client.socket.port` on server
// spans.
func NetSockPeerPort(val int) attribute.KeyValue {
	return NetSockPeerPortKey.Int(val)
}

// NetPeerName returns an attribute KeyValue conforming to the
// "net.peer.name" semantic conventions. It represents the deprecated, use
// `server.address` on client spans and `client.address` on server spans.
func NetPeerName(val string) attribute.KeyValue {
	return NetPeerNameKey.String(val)
}

// NetPeerPort returns an attribute KeyValue conforming to the
// "net.peer.port" semantic conventions. It represents the deprecated, use
// `server.port` on client spans and `client.port` on server spans.
func NetPeerPort(val int) attribute.KeyValue {
	return NetPeerPortKey.Int(val)
}

// NetHostName returns an attribute KeyValue conforming to the
// "net.host.name" semantic conventions. It represents the deprecated, use
// `server.address`.
func NetHostName(val string) attribute.KeyValue {
	return NetHostNameKey.String(val)
}

// NetHostPort returns an attribute KeyValue conforming to the
// "net.host.port" semantic conventions. It represents the deprecated, use
// `server.port`.
func NetHostPort(val int) attribute.KeyValue {
	return NetHostPortKey.Int(val)
}

// NetSockHostAddr returns an attribute KeyValue conforming to the
// "net.sock.host.addr" semantic conventions. It represents the deprecated, use
// `server.socket.address`.
func NetSockHostAddr(val string) attribute.KeyValue {
	return NetSockHostAddrKey.String(val)
}

// NetSockHostPort returns an attribute KeyValue conforming to the
// "net.sock.host.port" semantic conventions. It represents the deprecated, use
// `server.socket.port`.
func NetSockHostPort(val int) attribute.KeyValue {
	return NetSockHostPortKey.Int(val)
}

// NetProtocolName returns an attribute KeyValue conforming to the
// "net.protocol.name" semantic conventions. It represents the deprecated, use
// `network.protocol.name`.
func NetProtocolName(val string) attribute.KeyValue {
	return NetProtocolNameKey.String(val)
}

// NetProtocolVersion returns an attribute KeyValue conforming to the
// "net.protocol.version" semantic conventions. It represents the deprecated,
// use `network.protocol.version`.
func NetProtocolVersion(val string) attribute.KeyValue {
	return NetProtocolVersionKey.String(val)
}

// These attributes may be used to describe the receiver of a network
// exchange/packet. These should be used when there is no client/server
// relationship between the two sides, or when that relationship is unknown.
// This covers low-level network interactions (e.g. packet tracing) where you
// don't know if there was a connection or which side initiated it. This also
// covers unidirectional UDP flows and peer-to-peer communication where the
// "user-facing" surface of the protocol / API does not expose a clear notion
// of client and server.
const (
	// DestinationDomainKey is the attribute Key conforming to the
	// "destination.domain" semantic conventions. It represents the domain name
	// of the destination system.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'foo.example.com'
	// Note: This value may be a host name, a fully qualified domain name, or
	// another host naming format.
	DestinationDomainKey = attribute.Key("destination.domain")

	// DestinationAddressKey is the attribute Key conforming to the
	// "destination.address" semantic conventions. It represents the peer
	// address, for example IP address or UNIX socket name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '10.5.3.2'
	DestinationAddressKey = attribute.Key("destination.address")

	// DestinationPortKey is the attribute Key conforming to the
	// "destination.port" semantic conventions. It represents the peer port
	// number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3389, 2888
	DestinationPortKey = attribute.Key("destination.port")
)

// DestinationDomain returns an attribute KeyValue conforming to the
// "destination.domain" semantic conventions. It represents the domain name of
// the destination system.
func DestinationDomain(val string) attribute.KeyValue {
	return DestinationDomainKey.String(val)
}

// DestinationAddress returns an attribute KeyValue conforming to the
// "destination.address" semantic conventions. It represents the peer address,
// for example IP address or UNIX socket name.
func DestinationAddress(val string) attribute.KeyValue {
	return DestinationAddressKey.String(val)
}

// DestinationPort returns an attribute KeyValue conforming to the
// "destination.port" semantic conventions. It represents the peer port number
func DestinationPort(val int) attribute.KeyValue {
	return DestinationPortKey.Int(val)
}

// Describes HTTP attributes.
const (
	// HTTPRequestMethodKey is the attribute Key conforming to the
	// "http.request.method" semantic conventions. It represents the hTTP
	// request method.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'GET', 'POST', 'HEAD'
	// Note: HTTP request method value SHOULD be "known" to the
	// instrumentation.
	// By default, this convention defines "known" methods as the ones listed
	// in [RFC9110](https://www.rfc-editor.org/rfc/rfc9110.html#name-methods)
	// and the PATCH method defined in
	// [RFC5789](https://www.rfc-editor.org/rfc/rfc5789.html).
	//
	// If the HTTP request method is not known to instrumentation, it MUST set
	// the `http.request.method` attribute to `_OTHER` and, except if reporting
	// a metric, MUST
	// set the exact method received in the request line as value of the
	// `http.request.method_original` attribute.
	//
	// If the HTTP instrumentation could end up converting valid HTTP request
	// methods to `_OTHER`, then it MUST provide a way to override
	// the list of known HTTP methods. If this override is done via environment
	// variable, then the environment variable MUST be named
	// OTEL_INSTRUMENTATION_HTTP_KNOWN_METHODS and support a comma-separated
	// list of case-sensitive known HTTP methods
	// (this list MUST be a full override of the default known method, it is
	// not a list of known methods in addition to the defaults).
	//
	// HTTP method names are case-sensitive and `http.request.method` attribute
	// value MUST match a known HTTP method name exactly.
	// Instrumentations for specific web frameworks that consider HTTP methods
	// to be case insensitive, SHOULD populate a canonical equivalent.
	// Tracing instrumentations that do so, MUST also set
	// `http.request.method_original` to the original value.
	HTTPRequestMethodKey = attribute.Key("http.request.method")

	// HTTPResponseStatusCodeKey is the attribute Key conforming to the
	// "http.response.status_code" semantic conventions. It represents the
	// [HTTP response status
	// code](https://tools.ietf.org/html/rfc7231#section-6).
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If and only if one was
	// received/sent.)
	// Stability: stable
	// Examples: 200
	HTTPResponseStatusCodeKey = attribute.Key("http.response.status_code")
)

var (
	// CONNECT method
	HTTPRequestMethodConnect = HTTPRequestMethodKey.String("CONNECT")
	// DELETE method
	HTTPRequestMethodDelete = HTTPRequestMethodKey.String("DELETE")
	// GET method
	HTTPRequestMethodGet = HTTPRequestMethodKey.String("GET")
	// HEAD method
	HTTPRequestMethodHead = HTTPRequestMethodKey.String("HEAD")
	// OPTIONS method
	HTTPRequestMethodOptions = HTTPRequestMethodKey.String("OPTIONS")
	// PATCH method
	HTTPRequestMethodPatch = HTTPRequestMethodKey.String("PATCH")
	// POST method
	HTTPRequestMethodPost = HTTPRequestMethodKey.String("POST")
	// PUT method
	HTTPRequestMethodPut = HTTPRequestMethodKey.String("PUT")
	// TRACE method
	HTTPRequestMethodTrace = HTTPRequestMethodKey.String("TRACE")
	// Any HTTP method that the instrumentation has no prior knowledge of
	HTTPRequestMethodOther = HTTPRequestMethodKey.String("_OTHER")
)

// HTTPResponseStatusCode returns an attribute KeyValue conforming to the
// "http.response.status_code" semantic conventions. It represents the [HTTP
// response status code](https://tools.ietf.org/html/rfc7231#section-6).
func HTTPResponseStatusCode(val int) attribute.KeyValue {
	return HTTPResponseStatusCodeKey.Int(val)
}

// HTTP Server attributes
const (
	// HTTPRouteKey is the attribute Key conforming to the "http.route"
	// semantic conventions. It represents the matched route (path template in
	// the format used by the respective server framework). See note below
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If and only if it's available)
	// Stability: stable
	// Examples: '/users/:userID?', '{controller}/{action}/{id?}'
	// Note: MUST NOT be populated when this is not supported by the HTTP
	// server framework as the route attribute should have low-cardinality and
	// the URI path can NOT substitute it.
	// SHOULD include the [application
	// root](/docs/http/http-spans.md#http-server-definitions) if there is one.
	HTTPRouteKey = attribute.Key("http.route")
)

// HTTPRoute returns an attribute KeyValue conforming to the "http.route"
// semantic conventions. It represents the matched route (path template in the
// format used by the respective server framework). See note below
func HTTPRoute(val string) attribute.KeyValue {
	return HTTPRouteKey.String(val)
}

// Attributes for Events represented using Log Records.
const (
	// EventNameKey is the attribute Key conforming to the "event.name"
	// semantic conventions. It represents the name identifies the event.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'click', 'exception'
	EventNameKey = attribute.Key("event.name")

	// EventDomainKey is the attribute Key conforming to the "event.domain"
	// semantic conventions. It represents the domain identifies the business
	// context for the events.
	//
	// Type: Enum
	// RequirementLevel: Required
	// Stability: stable
	// Note: Events across different domains may have same `event.name`, yet be
	// unrelated events.
	EventDomainKey = attribute.Key("event.domain")
)

var (
	// Events from browser apps
	EventDomainBrowser = EventDomainKey.String("browser")
	// Events from mobile apps
	EventDomainDevice = EventDomainKey.String("device")
	// Events from Kubernetes
	EventDomainK8S = EventDomainKey.String("k8s")
)

// EventName returns an attribute KeyValue conforming to the "event.name"
// semantic conventions. It represents the name identifies the event.
func EventName(val string) attribute.KeyValue {
	return EventNameKey.String(val)
}

// The attributes described in this section are rather generic. They may be
// used in any Log Record they apply to.
const (
	// LogRecordUIDKey is the attribute Key conforming to the "log.record.uid"
	// semantic conventions. It represents a unique identifier for the Log
	// Record.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '01ARZ3NDEKTSV4RRFFQ69G5FAV'
	// Note: If an id is provided, other log records with the same id will be
	// considered duplicates and can be removed safely. This means, that two
	// distinguishable log records MUST have different values.
	// The id MAY be an [Universally Unique Lexicographically Sortable
	// Identifier (ULID)](https://github.com/ulid/spec), but other identifiers
	// (e.g. UUID) may be used as needed.
	LogRecordUIDKey = attribute.Key("log.record.uid")
)

// LogRecordUID returns an attribute KeyValue conforming to the
// "log.record.uid" semantic conventions. It represents a unique identifier for
// the Log Record.
func LogRecordUID(val string) attribute.KeyValue {
	return LogRecordUIDKey.String(val)
}

// Describes Log attributes
const (
	// LogIostreamKey is the attribute Key conforming to the "log.iostream"
	// semantic conventions. It represents the stream associated with the log.
	// See below for a list of well-known values.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	LogIostreamKey = attribute.Key("log.iostream")
)

var (
	// Logs from stdout stream
	LogIostreamStdout = LogIostreamKey.String("stdout")
	// Events from stderr stream
	LogIostreamStderr = LogIostreamKey.String("stderr")
)

// A file to which log was emitted.
const (
	// LogFileNameKey is the attribute Key conforming to the "log.file.name"
	// semantic conventions. It represents the basename of the file.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'audit.log'
	LogFileNameKey = attribute.Key("log.file.name")

	// LogFilePathKey is the attribute Key conforming to the "log.file.path"
	// semantic conventions. It represents the full path to the file.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/var/log/mysql/audit.log'
	LogFilePathKey = attribute.Key("log.file.path")

	// LogFileNameResolvedKey is the attribute Key conforming to the
	// "log.file.name_resolved" semantic conventions. It represents the
	// basename of the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'uuid.log'
	LogFileNameResolvedKey = attribute.Key("log.file.name_resolved")

	// LogFilePathResolvedKey is the attribute Key conforming to the
	// "log.file.path_resolved" semantic conventions. It represents the full
	// path to the file, with symlinks resolved.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/var/lib/docker/uuid.log'
	LogFilePathResolvedKey = attribute.Key("log.file.path_resolved")
)

// LogFileName returns an attribute KeyValue conforming to the
// "log.file.name" semantic conventions. It represents the basename of the
// file.
func LogFileName(val string) attribute.KeyValue {
	return LogFileNameKey.String(val)
}

// LogFilePath returns an attribute KeyValue conforming to the
// "log.file.path" semantic conventions. It represents the full path to the
// file.
func LogFilePath(val string) attribute.KeyValue {
	return LogFilePathKey.String(val)
}

// LogFileNameResolved returns an attribute KeyValue conforming to the
// "log.file.name_resolved" semantic conventions. It represents the basename of
// the file, with symlinks resolved.
func LogFileNameResolved(val string) attribute.KeyValue {
	return LogFileNameResolvedKey.String(val)
}

// LogFilePathResolved returns an attribute KeyValue conforming to the
// "log.file.path_resolved" semantic conventions. It represents the full path
// to the file, with symlinks resolved.
func LogFilePathResolved(val string) attribute.KeyValue {
	return LogFilePathResolvedKey.String(val)
}

// Describes JVM memory metric attributes.
const (
	// TypeKey is the attribute Key conforming to the "type" semantic
	// conventions. It represents the type of memory.
	//
	// Type: Enum
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'heap', 'non_heap'
	TypeKey = attribute.Key("type")

	// PoolKey is the attribute Key conforming to the "pool" semantic
	// conventions. It represents the name of the memory pool.
	//
	// Type: string
	// RequirementLevel: Recommended
	// Stability: stable
	// Examples: 'G1 Old Gen', 'G1 Eden space', 'G1 Survivor Space'
	// Note: Pool names are generally obtained via
	// [MemoryPoolMXBean#getName()](https://docs.oracle.com/en/java/javase/11/docs/api/java.management/java/lang/management/MemoryPoolMXBean.html#getName()).
	PoolKey = attribute.Key("pool")
)

var (
	// Heap memory
	TypeHeap = TypeKey.String("heap")
	// Non-heap memory
	TypeNonHeap = TypeKey.String("non_heap")
)

// Pool returns an attribute KeyValue conforming to the "pool" semantic
// conventions. It represents the name of the memory pool.
func Pool(val string) attribute.KeyValue {
	return PoolKey.String(val)
}

// These attributes may be used to describe the server in a connection-based
// network interaction where there is one side that initiates the connection
// (the client is the side that initiates the connection). This covers all TCP
// network interactions since TCP is connection-based and one side initiates
// the connection (an exception is made for peer-to-peer communication over TCP
// where the "user-facing" surface of the protocol / API does not expose a
// clear notion of client and server). This also covers UDP network
// interactions where one side initiates the interaction, e.g. QUIC (HTTP/3)
// and DNS.
const (
	// ServerAddressKey is the attribute Key conforming to the "server.address"
	// semantic conventions. It represents the logical server hostname, matches
	// server FQDN if available, and IP or socket address if FQDN is not known.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'example.com'
	ServerAddressKey = attribute.Key("server.address")

	// ServerPortKey is the attribute Key conforming to the "server.port"
	// semantic conventions. It represents the logical server port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 80, 8080, 443
	ServerPortKey = attribute.Key("server.port")

	// ServerSocketDomainKey is the attribute Key conforming to the
	// "server.socket.domain" semantic conventions. It represents the domain
	// name of an immediate peer.
	//
	// Type: string
	// RequirementLevel: Recommended (If different than `server.address`.)
	// Stability: stable
	// Examples: 'proxy.example.com'
	// Note: Typically observed from the client side, and represents a proxy or
	// other intermediary domain name.
	ServerSocketDomainKey = attribute.Key("server.socket.domain")

	// ServerSocketAddressKey is the attribute Key conforming to the
	// "server.socket.address" semantic conventions. It represents the physical
	// server IP address or Unix socket address. If set from the client, should
	// simply use the socket's peer address, and not attempt to find any actual
	// server IP (i.e., if set from client, this may represent some proxy
	// server instead of the logical server).
	//
	// Type: string
	// RequirementLevel: Recommended (If different than `server.address`.)
	// Stability: stable
	// Examples: '10.5.3.2'
	ServerSocketAddressKey = attribute.Key("server.socket.address")

	// ServerSocketPortKey is the attribute Key conforming to the
	// "server.socket.port" semantic conventions. It represents the physical
	// server port.
	//
	// Type: int
	// RequirementLevel: Recommended (If different than `server.port`.)
	// Stability: stable
	// Examples: 16456
	ServerSocketPortKey = attribute.Key("server.socket.port")
)

// ServerAddress returns an attribute KeyValue conforming to the
// "server.address" semantic conventions. It represents the logical server
// hostname, matches server FQDN if available, and IP or socket address if FQDN
// is not known.
func ServerAddress(val string) attribute.KeyValue {
	return ServerAddressKey.String(val)
}

// ServerPort returns an attribute KeyValue conforming to the "server.port"
// semantic conventions. It represents the logical server port number
func ServerPort(val int) attribute.KeyValue {
	return ServerPortKey.Int(val)
}

// ServerSocketDomain returns an attribute KeyValue conforming to the
// "server.socket.domain" semantic conventions. It represents the domain name
// of an immediate peer.
func ServerSocketDomain(val string) attribute.KeyValue {
	return ServerSocketDomainKey.String(val)
}

// ServerSocketAddress returns an attribute KeyValue conforming to the
// "server.socket.address" semantic conventions. It represents the physical
// server IP address or Unix socket address. If set from the client, should
// simply use the socket's peer address, and not attempt to find any actual
// server IP (i.e., if set from client, this may represent some proxy server
// instead of the logical server).
func ServerSocketAddress(val string) attribute.KeyValue {
	return ServerSocketAddressKey.String(val)
}

// ServerSocketPort returns an attribute KeyValue conforming to the
// "server.socket.port" semantic conventions. It represents the physical server
// port.
func ServerSocketPort(val int) attribute.KeyValue {
	return ServerSocketPortKey.Int(val)
}

// These attributes may be used to describe the sender of a network
// exchange/packet. These should be used when there is no client/server
// relationship between the two sides, or when that relationship is unknown.
// This covers low-level network interactions (e.g. packet tracing) where you
// don't know if there was a connection or which side initiated it. This also
// covers unidirectional UDP flows and peer-to-peer communication where the
// "user-facing" surface of the protocol / API does not expose a clear notion
// of client and server.
const (
	// SourceDomainKey is the attribute Key conforming to the "source.domain"
	// semantic conventions. It represents the domain name of the source
	// system.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'foo.example.com'
	// Note: This value may be a host name, a fully qualified domain name, or
	// another host naming format.
	SourceDomainKey = attribute.Key("source.domain")

	// SourceAddressKey is the attribute Key conforming to the "source.address"
	// semantic conventions. It represents the source address, for example IP
	// address or Unix socket name.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '10.5.3.2'
	SourceAddressKey = attribute.Key("source.address")

	// SourcePortKey is the attribute Key conforming to the "source.port"
	// semantic conventions. It represents the source port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3389, 2888
	SourcePortKey = attribute.Key("source.port")
)

// SourceDomain returns an attribute KeyValue conforming to the
// "source.domain" semantic conventions. It represents the domain name of the
// source system.
func SourceDomain(val string) attribute.KeyValue {
	return SourceDomainKey.String(val)
}

// SourceAddress returns an attribute KeyValue conforming to the
// "source.address" semantic conventions. It represents the source address, for
// example IP address or Unix socket name.
func SourceAddress(val string) attribute.KeyValue {
	return SourceAddressKey.String(val)
}

// SourcePort returns an attribute KeyValue conforming to the "source.port"
// semantic conventions. It represents the source port number
func SourcePort(val int) attribute.KeyValue {
	return SourcePortKey.Int(val)
}

// These attributes may be used for any network related operation.
const (
	// NetworkTransportKey is the attribute Key conforming to the
	// "network.transport" semantic conventions. It represents the [OSI
	// Transport Layer](https://osi-model.com/transport-layer/) or
	// [Inter-process Communication
	// method](https://en.wikipedia.org/wiki/Inter-process_communication). The
	// value SHOULD be normalized to lowercase.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'tcp', 'udp'
	NetworkTransportKey = attribute.Key("network.transport")

	// NetworkTypeKey is the attribute Key conforming to the "network.type"
	// semantic conventions. It represents the [OSI Network
	// Layer](https://osi-model.com/network-layer/) or non-OSI equivalent. The
	// value SHOULD be normalized to lowercase.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'ipv4', 'ipv6'
	NetworkTypeKey = attribute.Key("network.type")

	// NetworkProtocolNameKey is the attribute Key conforming to the
	// "network.protocol.name" semantic conventions. It represents the [OSI
	// Application Layer](https://osi-model.com/application-layer/) or non-OSI
	// equivalent. The value SHOULD be normalized to lowercase.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'amqp', 'http', 'mqtt'
	NetworkProtocolNameKey = attribute.Key("network.protocol.name")

	// NetworkProtocolVersionKey is the attribute Key conforming to the
	// "network.protocol.version" semantic conventions. It represents the
	// version of the application layer protocol used. See note below.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '3.1.1'
	// Note: `network.protocol.version` refers to the version of the protocol
	// used and might be different from the protocol client's version. If the
	// HTTP client used has a version of `0.27.2`, but sends HTTP version
	// `1.1`, this attribute should be set to `1.1`.
	NetworkProtocolVersionKey = attribute.Key("network.protocol.version")
)

var (
	// TCP
	NetworkTransportTCP = NetworkTransportKey.String("tcp")
	// UDP
	NetworkTransportUDP = NetworkTransportKey.String("udp")
	// Named or anonymous pipe. See note below
	NetworkTransportPipe = NetworkTransportKey.String("pipe")
	// Unix domain socket
	NetworkTransportUnix = NetworkTransportKey.String("unix")
)

var (
	// IPv4
	NetworkTypeIpv4 = NetworkTypeKey.String("ipv4")
	// IPv6
	NetworkTypeIpv6 = NetworkTypeKey.String("ipv6")
)

// NetworkProtocolName returns an attribute KeyValue conforming to the
// "network.protocol.name" semantic conventions. It represents the [OSI
// Application Layer](https://osi-model.com/application-layer/) or non-OSI
// equivalent. The value SHOULD be normalized to lowercase.
func NetworkProtocolName(val string) attribute.KeyValue {
	return NetworkProtocolNameKey.String(val)
}

// NetworkProtocolVersion returns an attribute KeyValue conforming to the
// "network.protocol.version" semantic conventions. It represents the version
// of the application layer protocol used. See note below.
func NetworkProtocolVersion(val string) attribute.KeyValue {
	return NetworkProtocolVersionKey.String(val)
}

// These attributes may be used for any network related operation.
const (
	// NetworkConnectionTypeKey is the attribute Key conforming to the
	// "network.connection.type" semantic conventions. It represents the
	// internet connection type.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'wifi'
	NetworkConnectionTypeKey = attribute.Key("network.connection.type")

	// NetworkConnectionSubtypeKey is the attribute Key conforming to the
	// "network.connection.subtype" semantic conventions. It represents the
	// this describes more details regarding the connection.type. It may be the
	// type of cell technology connection, but it could be used for describing
	// details about a wifi connection.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'LTE'
	NetworkConnectionSubtypeKey = attribute.Key("network.connection.subtype")

	// NetworkCarrierNameKey is the attribute Key conforming to the
	// "network.carrier.name" semantic conventions. It represents the name of
	// the mobile carrier.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'sprint'
	NetworkCarrierNameKey = attribute.Key("network.carrier.name")

	// NetworkCarrierMccKey is the attribute Key conforming to the
	// "network.carrier.mcc" semantic conventions. It represents the mobile
	// carrier country code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '310'
	NetworkCarrierMccKey = attribute.Key("network.carrier.mcc")

	// NetworkCarrierMncKey is the attribute Key conforming to the
	// "network.carrier.mnc" semantic conventions. It represents the mobile
	// carrier network code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '001'
	NetworkCarrierMncKey = attribute.Key("network.carrier.mnc")

	// NetworkCarrierIccKey is the attribute Key conforming to the
	// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
	// alpha-2 2-character country code associated with the mobile carrier
	// network.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'DE'
	NetworkCarrierIccKey = attribute.Key("network.carrier.icc")
)

var (
	// wifi
	NetworkConnectionTypeWifi = NetworkConnectionTypeKey.String("wifi")
	// wired
	NetworkConnectionTypeWired = NetworkConnectionTypeKey.String("wired")
	// cell
	NetworkConnectionTypeCell = NetworkConnectionTypeKey.String("cell")
	// unavailable
	NetworkConnectionTypeUnavailable = NetworkConnectionTypeKey.String("unavailable")
	// unknown
	NetworkConnectionTypeUnknown = NetworkConnectionTypeKey.String("unknown")
)

var (
	// GPRS
	NetworkConnectionSubtypeGprs = NetworkConnectionSubtypeKey.String("gprs")
	// EDGE
	NetworkConnectionSubtypeEdge = NetworkConnectionSubtypeKey.String("edge")
	// UMTS
	NetworkConnectionSubtypeUmts = NetworkConnectionSubtypeKey.String("umts")
	// CDMA
	NetworkConnectionSubtypeCdma = NetworkConnectionSubtypeKey.String("cdma")
	// EVDO Rel. 0
	NetworkConnectionSubtypeEvdo0 = NetworkConnectionSubtypeKey.String("evdo_0")
	// EVDO Rev. A
	NetworkConnectionSubtypeEvdoA = NetworkConnectionSubtypeKey.String("evdo_a")
	// CDMA2000 1XRTT
	NetworkConnectionSubtypeCdma20001xrtt = NetworkConnectionSubtypeKey.String("cdma2000_1xrtt")
	// HSDPA
	NetworkConnectionSubtypeHsdpa = NetworkConnectionSubtypeKey.String("hsdpa")
	// HSUPA
	NetworkConnectionSubtypeHsupa = NetworkConnectionSubtypeKey.String("hsupa")
	// HSPA
	NetworkConnectionSubtypeHspa = NetworkConnectionSubtypeKey.String("hspa")
	// IDEN
	NetworkConnectionSubtypeIden = NetworkConnectionSubtypeKey.String("iden")
	// EVDO Rev. B
	NetworkConnectionSubtypeEvdoB = NetworkConnectionSubtypeKey.String("evdo_b")
	// LTE
	NetworkConnectionSubtypeLte = NetworkConnectionSubtypeKey.String("lte")
	// EHRPD
	NetworkConnectionSubtypeEhrpd = NetworkConnectionSubtypeKey.String("ehrpd")
	// HSPAP
	NetworkConnectionSubtypeHspap = NetworkConnectionSubtypeKey.String("hspap")
	// GSM
	NetworkConnectionSubtypeGsm = NetworkConnectionSubtypeKey.String("gsm")
	// TD-SCDMA
	NetworkConnectionSubtypeTdScdma = NetworkConnectionSubtypeKey.String("td_scdma")
	// IWLAN
	NetworkConnectionSubtypeIwlan = NetworkConnectionSubtypeKey.String("iwlan")
	// 5G NR (New Radio)
	NetworkConnectionSubtypeNr = NetworkConnectionSubtypeKey.String("nr")
	// 5G NRNSA (New Radio Non-Standalone)
	NetworkConnectionSubtypeNrnsa = NetworkConnectionSubtypeKey.String("nrnsa")
	// LTE CA
	NetworkConnectionSubtypeLteCa = NetworkConnectionSubtypeKey.String("lte_ca")
)

// NetworkCarrierName returns an attribute KeyValue conforming to the
// "network.carrier.name" semantic conventions. It represents the name of the
// mobile carrier.
func NetworkCarrierName(val string) attribute.KeyValue {
	return NetworkCarrierNameKey.String(val)
}

// NetworkCarrierMcc returns an attribute KeyValue conforming to the
// "network.carrier.mcc" semantic conventions. It represents the mobile carrier
// country code.
func NetworkCarrierMcc(val string) attribute.KeyValue {
	return NetworkCarrierMccKey.String(val)
}

// NetworkCarrierMnc returns an attribute KeyValue conforming to the
// "network.carrier.mnc" semantic conventions. It represents the mobile carrier
// network code.
func NetworkCarrierMnc(val string) attribute.KeyValue {
	return NetworkCarrierMncKey.String(val)
}

// NetworkCarrierIcc returns an attribute KeyValue conforming to the
// "network.carrier.icc" semantic conventions. It represents the ISO 3166-1
// alpha-2 2-character country code associated with the mobile carrier network.
func NetworkCarrierIcc(val string) attribute.KeyValue {
	return NetworkCarrierIccKey.String(val)
}

// Semantic conventions for HTTP client and server Spans.
const (
	// HTTPRequestMethodOriginalKey is the attribute Key conforming to the
	// "http.request.method_original" semantic conventions. It represents the
	// original HTTP method sent by the client in the request line.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If and only if it's different
	// than `http.request.method`.)
	// Stability: stable
	// Examples: 'GeT', 'ACL', 'foo'
	HTTPRequestMethodOriginalKey = attribute.Key("http.request.method_original")

	// HTTPRequestBodySizeKey is the attribute Key conforming to the
	// "http.request.body.size" semantic conventions. It represents the size of
	// the request payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as
	// the
	// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
	// header. For requests using transport encoding, this should be the
	// compressed size.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3495
	HTTPRequestBodySizeKey = attribute.Key("http.request.body.size")

	// HTTPResponseBodySizeKey is the attribute Key conforming to the
	// "http.response.body.size" semantic conventions. It represents the size
	// of the response payload body in bytes. This is the number of bytes
	// transferred excluding headers and is often, but not always, present as
	// the
	// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
	// header. For requests using transport encoding, this should be the
	// compressed size.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 3495
	HTTPResponseBodySizeKey = attribute.Key("http.response.body.size")
)

// HTTPRequestMethodOriginal returns an attribute KeyValue conforming to the
// "http.request.method_original" semantic conventions. It represents the
// original HTTP method sent by the client in the request line.
func HTTPRequestMethodOriginal(val string) attribute.KeyValue {
	return HTTPRequestMethodOriginalKey.String(val)
}

// HTTPRequestBodySize returns an attribute KeyValue conforming to the
// "http.request.body.size" semantic conventions. It represents the size of the
// request payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPRequestBodySize(val int) attribute.KeyValue {
	return HTTPRequestBodySizeKey.Int(val)
}

// HTTPResponseBodySize returns an attribute KeyValue conforming to the
// "http.response.body.size" semantic conventions. It represents the size of
// the response payload body in bytes. This is the number of bytes transferred
// excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPResponseBodySize(val int) attribute.KeyValue {
	return HTTPResponseBodySizeKey.Int(val)
}

// Semantic convention describing per-message attributes populated on messaging
// spans or links.
const (
	// MessagingMessageIDKey is the attribute Key conforming to the
	// "messaging.message.id" semantic conventions. It represents a value used
	// by the messaging system as an identifier for the message, represented as
	// a string.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '452a7c7c7c7048c2f887f61572b18fc2'
	MessagingMessageIDKey = attribute.Key("messaging.message.id")

	// MessagingMessageConversationIDKey is the attribute Key conforming to the
	// "messaging.message.conversation_id" semantic conventions. It represents
	// the [conversation ID](#conversations) identifying the conversation to
	// which the message belongs, represented as a string. Sometimes called
	// "Correlation ID".
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'MyConversationID'
	MessagingMessageConversationIDKey = attribute.Key("messaging.message.conversation_id")

	// MessagingMessagePayloadSizeBytesKey is the attribute Key conforming to
	// the "messaging.message.payload_size_bytes" semantic conventions. It
	// represents the (uncompressed) size of the message payload in bytes. Also
	// use this attribute if it is unknown whether the compressed or
	// uncompressed payload size is reported.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 2738
	MessagingMessagePayloadSizeBytesKey = attribute.Key("messaging.message.payload_size_bytes")

	// MessagingMessagePayloadCompressedSizeBytesKey is the attribute Key
	// conforming to the "messaging.message.payload_compressed_size_bytes"
	// semantic conventions. It represents the compressed size of the message
	// payload in bytes.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 2048
	MessagingMessagePayloadCompressedSizeBytesKey = attribute.Key("messaging.message.payload_compressed_size_bytes")
)

// MessagingMessageID returns an attribute KeyValue conforming to the
// "messaging.message.id" semantic conventions. It represents a value used by
// the messaging system as an identifier for the message, represented as a
// string.
func MessagingMessageID(val string) attribute.KeyValue {
	return MessagingMessageIDKey.String(val)
}

// MessagingMessageConversationID returns an attribute KeyValue conforming
// to the "messaging.message.conversation_id" semantic conventions. It
// represents the [conversation ID](#conversations) identifying the
// conversation to which the message belongs, represented as a string.
// Sometimes called "Correlation ID".
func MessagingMessageConversationID(val string) attribute.KeyValue {
	return MessagingMessageConversationIDKey.String(val)
}

// MessagingMessagePayloadSizeBytes returns an attribute KeyValue conforming
// to the "messaging.message.payload_size_bytes" semantic conventions. It
// represents the (uncompressed) size of the message payload in bytes. Also use
// this attribute if it is unknown whether the compressed or uncompressed
// payload size is reported.
func MessagingMessagePayloadSizeBytes(val int) attribute.KeyValue {
	return MessagingMessagePayloadSizeBytesKey.Int(val)
}

// MessagingMessagePayloadCompressedSizeBytes returns an attribute KeyValue
// conforming to the "messaging.message.payload_compressed_size_bytes" semantic
// conventions. It represents the compressed size of the message payload in
// bytes.
func MessagingMessagePayloadCompressedSizeBytes(val int) attribute.KeyValue {
	return MessagingMessagePayloadCompressedSizeBytesKey.Int(val)
}

// Semantic convention for attributes that describe messaging destination on
// broker
const (
	// MessagingDestinationNameKey is the attribute Key conforming to the
	// "messaging.destination.name" semantic conventions. It represents the
	// message destination name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'MyQueue', 'MyTopic'
	// Note: Destination name SHOULD uniquely identify a specific queue, topic
	// or other entity within the broker. If
	// the broker does not have such notion, the destination name SHOULD
	// uniquely identify the broker.
	MessagingDestinationNameKey = attribute.Key("messaging.destination.name")

	// MessagingDestinationTemplateKey is the attribute Key conforming to the
	// "messaging.destination.template" semantic conventions. It represents the
	// low cardinality representation of the messaging destination name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/customers/{customerID}'
	// Note: Destination names could be constructed from templates. An example
	// would be a destination name involving a user name or product id.
	// Although the destination name in this case is of high cardinality, the
	// underlying template is of low cardinality and can be effectively used
	// for grouping and aggregation.
	MessagingDestinationTemplateKey = attribute.Key("messaging.destination.template")

	// MessagingDestinationTemporaryKey is the attribute Key conforming to the
	// "messaging.destination.temporary" semantic conventions. It represents a
	// boolean that is true if the message destination is temporary and might
	// not exist anymore after messages are processed.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	MessagingDestinationTemporaryKey = attribute.Key("messaging.destination.temporary")

	// MessagingDestinationAnonymousKey is the attribute Key conforming to the
	// "messaging.destination.anonymous" semantic conventions. It represents a
	// boolean that is true if the message destination is anonymous (could be
	// unnamed or have auto-generated name).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	MessagingDestinationAnonymousKey = attribute.Key("messaging.destination.anonymous")
)

// MessagingDestinationName returns an attribute KeyValue conforming to the
// "messaging.destination.name" semantic conventions. It represents the message
// destination name
func MessagingDestinationName(val string) attribute.KeyValue {
	return MessagingDestinationNameKey.String(val)
}

// MessagingDestinationTemplate returns an attribute KeyValue conforming to
// the "messaging.destination.template" semantic conventions. It represents the
// low cardinality representation of the messaging destination name
func MessagingDestinationTemplate(val string) attribute.KeyValue {
	return MessagingDestinationTemplateKey.String(val)
}

// MessagingDestinationTemporary returns an attribute KeyValue conforming to
// the "messaging.destination.temporary" semantic conventions. It represents a
// boolean that is true if the message destination is temporary and might not
// exist anymore after messages are processed.
func MessagingDestinationTemporary(val bool) attribute.KeyValue {
	return MessagingDestinationTemporaryKey.Bool(val)
}

// MessagingDestinationAnonymous returns an attribute KeyValue conforming to
// the "messaging.destination.anonymous" semantic conventions. It represents a
// boolean that is true if the message destination is anonymous (could be
// unnamed or have auto-generated name).
func MessagingDestinationAnonymous(val bool) attribute.KeyValue {
	return MessagingDestinationAnonymousKey.Bool(val)
}

// Attributes for RabbitMQ
const (
	// MessagingRabbitmqDestinationRoutingKeyKey is the attribute Key
	// conforming to the "messaging.rabbitmq.destination.routing_key" semantic
	// conventions. It represents the rabbitMQ message routing key.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If not empty.)
	// Stability: stable
	// Examples: 'myKey'
	MessagingRabbitmqDestinationRoutingKeyKey = attribute.Key("messaging.rabbitmq.destination.routing_key")
)

// MessagingRabbitmqDestinationRoutingKey returns an attribute KeyValue
// conforming to the "messaging.rabbitmq.destination.routing_key" semantic
// conventions. It represents the rabbitMQ message routing key.
func MessagingRabbitmqDestinationRoutingKey(val string) attribute.KeyValue {
	return MessagingRabbitmqDestinationRoutingKeyKey.String(val)
}

// Attributes for Apache Kafka
const (
	// MessagingKafkaMessageKeyKey is the attribute Key conforming to the
	// "messaging.kafka.message.key" semantic conventions. It represents the
	// message keys in Kafka are used for grouping alike messages to ensure
	// they're processed on the same partition. They differ from
	// `messaging.message.id` in that they're not unique. If the key is `null`,
	// the attribute MUST NOT be set.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'myKey'
	// Note: If the key type is not string, it's string representation has to
	// be supplied for the attribute. If the key has no unambiguous, canonical
	// string form, don't include its value.
	MessagingKafkaMessageKeyKey = attribute.Key("messaging.kafka.message.key")

	// MessagingKafkaConsumerGroupKey is the attribute Key conforming to the
	// "messaging.kafka.consumer.group" semantic conventions. It represents the
	// name of the Kafka Consumer Group that is handling the message. Only
	// applies to consumers, not producers.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'my-group'
	MessagingKafkaConsumerGroupKey = attribute.Key("messaging.kafka.consumer.group")

	// MessagingKafkaDestinationPartitionKey is the attribute Key conforming to
	// the "messaging.kafka.destination.partition" semantic conventions. It
	// represents the partition the message is sent to.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 2
	MessagingKafkaDestinationPartitionKey = attribute.Key("messaging.kafka.destination.partition")

	// MessagingKafkaMessageOffsetKey is the attribute Key conforming to the
	// "messaging.kafka.message.offset" semantic conventions. It represents the
	// offset of a record in the corresponding Kafka partition.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 42
	MessagingKafkaMessageOffsetKey = attribute.Key("messaging.kafka.message.offset")

	// MessagingKafkaMessageTombstoneKey is the attribute Key conforming to the
	// "messaging.kafka.message.tombstone" semantic conventions. It represents
	// a boolean that is true if the message is a tombstone.
	//
	// Type: boolean
	// RequirementLevel: ConditionallyRequired (If value is `true`. When
	// missing, the value is assumed to be `false`.)
	// Stability: stable
	MessagingKafkaMessageTombstoneKey = attribute.Key("messaging.kafka.message.tombstone")
)

// MessagingKafkaMessageKey returns an attribute KeyValue conforming to the
// "messaging.kafka.message.key" semantic conventions. It represents the
// message keys in Kafka are used for grouping alike messages to ensure they're
// processed on the same partition. They differ from `messaging.message.id` in
// that they're not unique. If the key is `null`, the attribute MUST NOT be
// set.
func MessagingKafkaMessageKey(val string) attribute.KeyValue {
	return MessagingKafkaMessageKeyKey.String(val)
}

// MessagingKafkaConsumerGroup returns an attribute KeyValue conforming to
// the "messaging.kafka.consumer.group" semantic conventions. It represents the
// name of the Kafka Consumer Group that is handling the message. Only applies
// to consumers, not producers.
func MessagingKafkaConsumerGroup(val string) attribute.KeyValue {
	return MessagingKafkaConsumerGroupKey.String(val)
}

// MessagingKafkaDestinationPartition returns an attribute KeyValue
// conforming to the "messaging.kafka.destination.partition" semantic
// conventions. It represents the partition the message is sent to.
func MessagingKafkaDestinationPartition(val int) attribute.KeyValue {
	return MessagingKafkaDestinationPartitionKey.Int(val)
}

// MessagingKafkaMessageOffset returns an attribute KeyValue conforming to
// the "messaging.kafka.message.offset" semantic conventions. It represents the
// offset of a record in the corresponding Kafka partition.
func MessagingKafkaMessageOffset(val int) attribute.KeyValue {
	return MessagingKafkaMessageOffsetKey.Int(val)
}

// MessagingKafkaMessageTombstone returns an attribute KeyValue conforming
// to the "messaging.kafka.message.tombstone" semantic conventions. It
// represents a boolean that is true if the message is a tombstone.
func MessagingKafkaMessageTombstone(val bool) attribute.KeyValue {
	return MessagingKafkaMessageTombstoneKey.Bool(val)
}

// Attributes for Apache RocketMQ
const (
	// MessagingRocketmqNamespaceKey is the attribute Key conforming to the
	// "messaging.rocketmq.namespace" semantic conventions. It represents the
	// namespace of RocketMQ resources, resources in different namespaces are
	// individual.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'myNamespace'
	MessagingRocketmqNamespaceKey = attribute.Key("messaging.rocketmq.namespace")

	// MessagingRocketmqClientGroupKey is the attribute Key conforming to the
	// "messaging.rocketmq.client_group" semantic conventions. It represents
	// the name of the RocketMQ producer/consumer group that is handling the
	// message. The client type is identified by the SpanKind.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'myConsumerGroup'
	MessagingRocketmqClientGroupKey = attribute.Key("messaging.rocketmq.client_group")

	// MessagingRocketmqMessageDeliveryTimestampKey is the attribute Key
	// conforming to the "messaging.rocketmq.message.delivery_timestamp"
	// semantic conventions. It represents the timestamp in milliseconds that
	// the delay message is expected to be delivered to consumer.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If the message type is delay
	// and delay time level is not specified.)
	// Stability: stable
	// Examples: 1665987217045
	MessagingRocketmqMessageDeliveryTimestampKey = attribute.Key("messaging.rocketmq.message.delivery_timestamp")

	// MessagingRocketmqMessageDelayTimeLevelKey is the attribute Key
	// conforming to the "messaging.rocketmq.message.delay_time_level" semantic
	// conventions. It represents the delay time level for delay message, which
	// determines the message delay time.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If the message type is delay
	// and delivery timestamp is not specified.)
	// Stability: stable
	// Examples: 3
	MessagingRocketmqMessageDelayTimeLevelKey = attribute.Key("messaging.rocketmq.message.delay_time_level")

	// MessagingRocketmqMessageGroupKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.group" semantic conventions. It represents
	// the it is essential for FIFO message. Messages that belong to the same
	// message group are always processed one by one within the same consumer
	// group.
	//
	// Type: string
	// RequirementLevel: ConditionallyRequired (If the message type is FIFO.)
	// Stability: stable
	// Examples: 'myMessageGroup'
	MessagingRocketmqMessageGroupKey = attribute.Key("messaging.rocketmq.message.group")

	// MessagingRocketmqMessageTypeKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.type" semantic conventions. It represents
	// the type of message.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	MessagingRocketmqMessageTypeKey = attribute.Key("messaging.rocketmq.message.type")

	// MessagingRocketmqMessageTagKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.tag" semantic conventions. It represents the
	// secondary classifier of message besides topic.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'tagA'
	MessagingRocketmqMessageTagKey = attribute.Key("messaging.rocketmq.message.tag")

	// MessagingRocketmqMessageKeysKey is the attribute Key conforming to the
	// "messaging.rocketmq.message.keys" semantic conventions. It represents
	// the key(s) of message, another way to mark message besides message id.
	//
	// Type: string[]
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'keyA', 'keyB'
	MessagingRocketmqMessageKeysKey = attribute.Key("messaging.rocketmq.message.keys")

	// MessagingRocketmqConsumptionModelKey is the attribute Key conforming to
	// the "messaging.rocketmq.consumption_model" semantic conventions. It
	// represents the model of message consumption. This only applies to
	// consumer spans.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	MessagingRocketmqConsumptionModelKey = attribute.Key("messaging.rocketmq.consumption_model")
)

var (
	// Normal message
	MessagingRocketmqMessageTypeNormal = MessagingRocketmqMessageTypeKey.String("normal")
	// FIFO message
	MessagingRocketmqMessageTypeFifo = MessagingRocketmqMessageTypeKey.String("fifo")
	// Delay message
	MessagingRocketmqMessageTypeDelay = MessagingRocketmqMessageTypeKey.String("delay")
	// Transaction message
	MessagingRocketmqMessageTypeTransaction = MessagingRocketmqMessageTypeKey.String("transaction")
)

var (
	// Clustering consumption model
	MessagingRocketmqConsumptionModelClustering = MessagingRocketmqConsumptionModelKey.String("clustering")
	// Broadcasting consumption model
	MessagingRocketmqConsumptionModelBroadcasting = MessagingRocketmqConsumptionModelKey.String("broadcasting")
)

// MessagingRocketmqNamespace returns an attribute KeyValue conforming to
// the "messaging.rocketmq.namespace" semantic conventions. It represents the
// namespace of RocketMQ resources, resources in different namespaces are
// individual.
func MessagingRocketmqNamespace(val string) attribute.KeyValue {
	return MessagingRocketmqNamespaceKey.String(val)
}

// MessagingRocketmqClientGroup returns an attribute KeyValue conforming to
// the "messaging.rocketmq.client_group" semantic conventions. It represents
// the name of the RocketMQ producer/consumer group that is handling the
// message. The client type is identified by the SpanKind.
func MessagingRocketmqClientGroup(val string) attribute.KeyValue {
	return MessagingRocketmqClientGroupKey.String(val)
}

// MessagingRocketmqMessageDeliveryTimestamp returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delivery_timestamp" semantic
// conventions. It represents the timestamp in milliseconds that the delay
// message is expected to be delivered to consumer.
func MessagingRocketmqMessageDeliveryTimestamp(val int) attribute.KeyValue {
	return MessagingRocketmqMessageDeliveryTimestampKey.Int(val)
}

// MessagingRocketmqMessageDelayTimeLevel returns an attribute KeyValue
// conforming to the "messaging.rocketmq.message.delay_time_level" semantic
// conventions. It represents the delay time level for delay message, which
// determines the message delay time.
func MessagingRocketmqMessageDelayTimeLevel(val int) attribute.KeyValue {
	return MessagingRocketmqMessageDelayTimeLevelKey.Int(val)
}

// MessagingRocketmqMessageGroup returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.group" semantic conventions. It represents
// the it is essential for FIFO message. Messages that belong to the same
// message group are always processed one by one within the same consumer
// group.
func MessagingRocketmqMessageGroup(val string) attribute.KeyValue {
	return MessagingRocketmqMessageGroupKey.String(val)
}

// MessagingRocketmqMessageTag returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.tag" semantic conventions. It represents the
// secondary classifier of message besides topic.
func MessagingRocketmqMessageTag(val string) attribute.KeyValue {
	return MessagingRocketmqMessageTagKey.String(val)
}

// MessagingRocketmqMessageKeys returns an attribute KeyValue conforming to
// the "messaging.rocketmq.message.keys" semantic conventions. It represents
// the key(s) of message, another way to mark message besides message id.
func MessagingRocketmqMessageKeys(val ...string) attribute.KeyValue {
	return MessagingRocketmqMessageKeysKey.StringSlice(val)
}

// Attributes describing URL.
const (
	// URLSchemeKey is the attribute Key conforming to the "url.scheme"
	// semantic conventions. It represents the [URI
	// scheme](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) component
	// identifying the used protocol.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'https', 'ftp', 'telnet'
	URLSchemeKey = attribute.Key("url.scheme")

	// URLFullKey is the attribute Key conforming to the "url.full" semantic
	// conventions. It represents the absolute URL describing a network
	// resource according to [RFC3986](https://www.rfc-editor.org/rfc/rfc3986)
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'https://www.foo.bar/search?q=OpenTelemetry#SemConv',
	// '//localhost'
	// Note: For network calls, URL usually has
	// `scheme://host[:port][path][?query][#fragment]` format, where the
	// fragment is not transmitted over HTTP, but if it is known, it should be
	// included nevertheless.
	// `url.full` MUST NOT contain credentials passed via URL in form of
	// `https://username:password@www.example.com/`. In such case username and
	// password should be redacted and attribute's value should be
	// `https://REDACTED:REDACTED@www.example.com/`.
	// `url.full` SHOULD capture the absolute URL when it is available (or can
	// be reconstructed) and SHOULD NOT be validated or modified except for
	// sanitizing purposes.
	URLFullKey = attribute.Key("url.full")

	// URLPathKey is the attribute Key conforming to the "url.path" semantic
	// conventions. It represents the [URI
	// path](https://www.rfc-editor.org/rfc/rfc3986#section-3.3) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/search'
	// Note: When missing, the value is assumed to be `/`
	URLPathKey = attribute.Key("url.path")

	// URLQueryKey is the attribute Key conforming to the "url.query" semantic
	// conventions. It represents the [URI
	// query](https://www.rfc-editor.org/rfc/rfc3986#section-3.4) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'q=OpenTelemetry'
	// Note: Sensitive content provided in query string SHOULD be scrubbed when
	// instrumentations can identify it.
	URLQueryKey = attribute.Key("url.query")

	// URLFragmentKey is the attribute Key conforming to the "url.fragment"
	// semantic conventions. It represents the [URI
	// fragment](https://www.rfc-editor.org/rfc/rfc3986#section-3.5) component
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'SemConv'
	URLFragmentKey = attribute.Key("url.fragment")
)

// URLScheme returns an attribute KeyValue conforming to the "url.scheme"
// semantic conventions. It represents the [URI
// scheme](https://www.rfc-editor.org/rfc/rfc3986#section-3.1) component
// identifying the used protocol.
func URLScheme(val string) attribute.KeyValue {
	return URLSchemeKey.String(val)
}

// URLFull returns an attribute KeyValue conforming to the "url.full"
// semantic conventions. It represents the absolute URL describing a network
// resource according to [RFC3986](https://www.rfc-editor.org/rfc/rfc3986)
func URLFull(val string) attribute.KeyValue {
	return URLFullKey.String(val)
}

// URLPath returns an attribute KeyValue conforming to the "url.path"
// semantic conventions. It represents the [URI
// path](https://www.rfc-editor.org/rfc/rfc3986#section-3.3) component
func URLPath(val string) attribute.KeyValue {
	return URLPathKey.String(val)
}

// URLQuery returns an attribute KeyValue conforming to the "url.query"
// semantic conventions. It represents the [URI
// query](https://www.rfc-editor.org/rfc/rfc3986#section-3.4) component
func URLQuery(val string) attribute.KeyValue {
	return URLQueryKey.String(val)
}

// URLFragment returns an attribute KeyValue conforming to the
// "url.fragment" semantic conventions. It represents the [URI
// fragment](https://www.rfc-editor.org/rfc/rfc3986#section-3.5) component
func URLFragment(val string) attribute.KeyValue {
	return URLFragmentKey.String(val)
}

// Describes user-agent attributes.
const (
	// UserAgentOriginalKey is the attribute Key conforming to the
	// "user_agent.original" semantic conventions. It represents the value of
	// the [HTTP
	// User-Agent](https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent)
	// header sent by the client.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'CERN-LineMode/2.15 libwww/2.17b3'
	UserAgentOriginalKey = attribute.Key("user_agent.original")
)

// UserAgentOriginal returns an attribute KeyValue conforming to the
// "user_agent.original" semantic conventions. It represents the value of the
// [HTTP
// User-Agent](https://www.rfc-editor.org/rfc/rfc9110.html#field.user-agent)
// header sent by the client.
func UserAgentOriginal(val string) attribute.KeyValue {
	return UserAgentOriginalKey.String(val)
}
