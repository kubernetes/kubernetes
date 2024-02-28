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

package semconv // import "go.opentelemetry.io/otel/semconv/v1.20.0"

import "go.opentelemetry.io/otel/attribute"

// Describes HTTP attributes.
const (
	// HTTPMethodKey is the attribute Key conforming to the "http.method"
	// semantic conventions. It represents the hTTP request method.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'GET', 'POST', 'HEAD'
	HTTPMethodKey = attribute.Key("http.method")

	// HTTPStatusCodeKey is the attribute Key conforming to the
	// "http.status_code" semantic conventions. It represents the [HTTP
	// response status code](https://tools.ietf.org/html/rfc7231#section-6).
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If and only if one was
	// received/sent.)
	// Stability: stable
	// Examples: 200
	HTTPStatusCodeKey = attribute.Key("http.status_code")
)

// HTTPMethod returns an attribute KeyValue conforming to the "http.method"
// semantic conventions. It represents the hTTP request method.
func HTTPMethod(val string) attribute.KeyValue {
	return HTTPMethodKey.String(val)
}

// HTTPStatusCode returns an attribute KeyValue conforming to the
// "http.status_code" semantic conventions. It represents the [HTTP response
// status code](https://tools.ietf.org/html/rfc7231#section-6).
func HTTPStatusCode(val int) attribute.KeyValue {
	return HTTPStatusCodeKey.Int(val)
}

// HTTP Server spans attributes
const (
	// HTTPSchemeKey is the attribute Key conforming to the "http.scheme"
	// semantic conventions. It represents the URI scheme identifying the used
	// protocol.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'http', 'https'
	HTTPSchemeKey = attribute.Key("http.scheme")

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
	// root](/specification/trace/semantic_conventions/http.md#http-server-definitions)
	// if there is one.
	HTTPRouteKey = attribute.Key("http.route")
)

// HTTPScheme returns an attribute KeyValue conforming to the "http.scheme"
// semantic conventions. It represents the URI scheme identifying the used
// protocol.
func HTTPScheme(val string) attribute.KeyValue {
	return HTTPSchemeKey.String(val)
}

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

// These attributes may be used for any network related operation.
const (
	// NetTransportKey is the attribute Key conforming to the "net.transport"
	// semantic conventions. It represents the transport protocol used. See
	// note below.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	NetTransportKey = attribute.Key("net.transport")

	// NetProtocolNameKey is the attribute Key conforming to the
	// "net.protocol.name" semantic conventions. It represents the application
	// layer protocol used. The value SHOULD be normalized to lowercase.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'amqp', 'http', 'mqtt'
	NetProtocolNameKey = attribute.Key("net.protocol.name")

	// NetProtocolVersionKey is the attribute Key conforming to the
	// "net.protocol.version" semantic conventions. It represents the version
	// of the application layer protocol used. See note below.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '3.1.1'
	// Note: `net.protocol.version` refers to the version of the protocol used
	// and might be different from the protocol client's version. If the HTTP
	// client used has a version of `0.27.2`, but sends HTTP version `1.1`,
	// this attribute should be set to `1.1`.
	NetProtocolVersionKey = attribute.Key("net.protocol.version")

	// NetSockPeerNameKey is the attribute Key conforming to the
	// "net.sock.peer.name" semantic conventions. It represents the remote
	// socket peer name.
	//
	// Type: string
	// RequirementLevel: Recommended (If available and different from
	// `net.peer.name` and if `net.sock.peer.addr` is set.)
	// Stability: stable
	// Examples: 'proxy.example.com'
	NetSockPeerNameKey = attribute.Key("net.sock.peer.name")

	// NetSockPeerAddrKey is the attribute Key conforming to the
	// "net.sock.peer.addr" semantic conventions. It represents the remote
	// socket peer address: IPv4 or IPv6 for internet protocols, path for local
	// communication,
	// [etc](https://man7.org/linux/man-pages/man7/address_families.7.html).
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '127.0.0.1', '/tmp/mysql.sock'
	NetSockPeerAddrKey = attribute.Key("net.sock.peer.addr")

	// NetSockPeerPortKey is the attribute Key conforming to the
	// "net.sock.peer.port" semantic conventions. It represents the remote
	// socket peer port.
	//
	// Type: int
	// RequirementLevel: Recommended (If defined for the address family and if
	// different than `net.peer.port` and if `net.sock.peer.addr` is set.)
	// Stability: stable
	// Examples: 16456
	NetSockPeerPortKey = attribute.Key("net.sock.peer.port")

	// NetSockFamilyKey is the attribute Key conforming to the
	// "net.sock.family" semantic conventions. It represents the protocol
	// [address
	// family](https://man7.org/linux/man-pages/man7/address_families.7.html)
	// which is used for communication.
	//
	// Type: Enum
	// RequirementLevel: ConditionallyRequired (If different than `inet` and if
	// any of `net.sock.peer.addr` or `net.sock.host.addr` are set. Consumers
	// of telemetry SHOULD accept both IPv4 and IPv6 formats for the address in
	// `net.sock.peer.addr` if `net.sock.family` is not set. This is to support
	// instrumentations that follow previous versions of this document.)
	// Stability: stable
	// Examples: 'inet6', 'bluetooth'
	NetSockFamilyKey = attribute.Key("net.sock.family")

	// NetPeerNameKey is the attribute Key conforming to the "net.peer.name"
	// semantic conventions. It represents the logical remote hostname, see
	// note below.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'example.com'
	// Note: `net.peer.name` SHOULD NOT be set if capturing it would require an
	// extra DNS lookup.
	NetPeerNameKey = attribute.Key("net.peer.name")

	// NetPeerPortKey is the attribute Key conforming to the "net.peer.port"
	// semantic conventions. It represents the logical remote port number
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 80, 8080, 443
	NetPeerPortKey = attribute.Key("net.peer.port")

	// NetHostNameKey is the attribute Key conforming to the "net.host.name"
	// semantic conventions. It represents the logical local hostname or
	// similar, see note below.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'localhost'
	NetHostNameKey = attribute.Key("net.host.name")

	// NetHostPortKey is the attribute Key conforming to the "net.host.port"
	// semantic conventions. It represents the logical local port number,
	// preferably the one that the peer used to connect
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 8080
	NetHostPortKey = attribute.Key("net.host.port")

	// NetSockHostAddrKey is the attribute Key conforming to the
	// "net.sock.host.addr" semantic conventions. It represents the local
	// socket address. Useful in case of a multi-IP host.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '192.168.0.1'
	NetSockHostAddrKey = attribute.Key("net.sock.host.addr")

	// NetSockHostPortKey is the attribute Key conforming to the
	// "net.sock.host.port" semantic conventions. It represents the local
	// socket port number.
	//
	// Type: int
	// RequirementLevel: ConditionallyRequired (If defined for the address
	// family and if different than `net.host.port` and if `net.sock.host.addr`
	// is set. In other cases, it is still recommended to set this.)
	// Stability: stable
	// Examples: 35555
	NetSockHostPortKey = attribute.Key("net.sock.host.port")
)

var (
	// ip_tcp
	NetTransportTCP = NetTransportKey.String("ip_tcp")
	// ip_udp
	NetTransportUDP = NetTransportKey.String("ip_udp")
	// Named or anonymous pipe. See note below
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

// NetProtocolName returns an attribute KeyValue conforming to the
// "net.protocol.name" semantic conventions. It represents the application
// layer protocol used. The value SHOULD be normalized to lowercase.
func NetProtocolName(val string) attribute.KeyValue {
	return NetProtocolNameKey.String(val)
}

// NetProtocolVersion returns an attribute KeyValue conforming to the
// "net.protocol.version" semantic conventions. It represents the version of
// the application layer protocol used. See note below.
func NetProtocolVersion(val string) attribute.KeyValue {
	return NetProtocolVersionKey.String(val)
}

// NetSockPeerName returns an attribute KeyValue conforming to the
// "net.sock.peer.name" semantic conventions. It represents the remote socket
// peer name.
func NetSockPeerName(val string) attribute.KeyValue {
	return NetSockPeerNameKey.String(val)
}

// NetSockPeerAddr returns an attribute KeyValue conforming to the
// "net.sock.peer.addr" semantic conventions. It represents the remote socket
// peer address: IPv4 or IPv6 for internet protocols, path for local
// communication,
// [etc](https://man7.org/linux/man-pages/man7/address_families.7.html).
func NetSockPeerAddr(val string) attribute.KeyValue {
	return NetSockPeerAddrKey.String(val)
}

// NetSockPeerPort returns an attribute KeyValue conforming to the
// "net.sock.peer.port" semantic conventions. It represents the remote socket
// peer port.
func NetSockPeerPort(val int) attribute.KeyValue {
	return NetSockPeerPortKey.Int(val)
}

// NetPeerName returns an attribute KeyValue conforming to the
// "net.peer.name" semantic conventions. It represents the logical remote
// hostname, see note below.
func NetPeerName(val string) attribute.KeyValue {
	return NetPeerNameKey.String(val)
}

// NetPeerPort returns an attribute KeyValue conforming to the
// "net.peer.port" semantic conventions. It represents the logical remote port
// number
func NetPeerPort(val int) attribute.KeyValue {
	return NetPeerPortKey.Int(val)
}

// NetHostName returns an attribute KeyValue conforming to the
// "net.host.name" semantic conventions. It represents the logical local
// hostname or similar, see note below.
func NetHostName(val string) attribute.KeyValue {
	return NetHostNameKey.String(val)
}

// NetHostPort returns an attribute KeyValue conforming to the
// "net.host.port" semantic conventions. It represents the logical local port
// number, preferably the one that the peer used to connect
func NetHostPort(val int) attribute.KeyValue {
	return NetHostPortKey.Int(val)
}

// NetSockHostAddr returns an attribute KeyValue conforming to the
// "net.sock.host.addr" semantic conventions. It represents the local socket
// address. Useful in case of a multi-IP host.
func NetSockHostAddr(val string) attribute.KeyValue {
	return NetSockHostAddrKey.String(val)
}

// NetSockHostPort returns an attribute KeyValue conforming to the
// "net.sock.host.port" semantic conventions. It represents the local socket
// port number.
func NetSockHostPort(val int) attribute.KeyValue {
	return NetSockHostPortKey.Int(val)
}

// These attributes may be used for any network related operation.
const (
	// NetHostConnectionTypeKey is the attribute Key conforming to the
	// "net.host.connection.type" semantic conventions. It represents the
	// internet connection type currently being used by the host.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'wifi'
	NetHostConnectionTypeKey = attribute.Key("net.host.connection.type")

	// NetHostConnectionSubtypeKey is the attribute Key conforming to the
	// "net.host.connection.subtype" semantic conventions. It represents the
	// this describes more details regarding the connection.type. It may be the
	// type of cell technology connection, but it could be used for describing
	// details about a wifi connection.
	//
	// Type: Enum
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'LTE'
	NetHostConnectionSubtypeKey = attribute.Key("net.host.connection.subtype")

	// NetHostCarrierNameKey is the attribute Key conforming to the
	// "net.host.carrier.name" semantic conventions. It represents the name of
	// the mobile carrier.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'sprint'
	NetHostCarrierNameKey = attribute.Key("net.host.carrier.name")

	// NetHostCarrierMccKey is the attribute Key conforming to the
	// "net.host.carrier.mcc" semantic conventions. It represents the mobile
	// carrier country code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '310'
	NetHostCarrierMccKey = attribute.Key("net.host.carrier.mcc")

	// NetHostCarrierMncKey is the attribute Key conforming to the
	// "net.host.carrier.mnc" semantic conventions. It represents the mobile
	// carrier network code.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '001'
	NetHostCarrierMncKey = attribute.Key("net.host.carrier.mnc")

	// NetHostCarrierIccKey is the attribute Key conforming to the
	// "net.host.carrier.icc" semantic conventions. It represents the ISO
	// 3166-1 alpha-2 2-character country code associated with the mobile
	// carrier network.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'DE'
	NetHostCarrierIccKey = attribute.Key("net.host.carrier.icc")
)

var (
	// wifi
	NetHostConnectionTypeWifi = NetHostConnectionTypeKey.String("wifi")
	// wired
	NetHostConnectionTypeWired = NetHostConnectionTypeKey.String("wired")
	// cell
	NetHostConnectionTypeCell = NetHostConnectionTypeKey.String("cell")
	// unavailable
	NetHostConnectionTypeUnavailable = NetHostConnectionTypeKey.String("unavailable")
	// unknown
	NetHostConnectionTypeUnknown = NetHostConnectionTypeKey.String("unknown")
)

var (
	// GPRS
	NetHostConnectionSubtypeGprs = NetHostConnectionSubtypeKey.String("gprs")
	// EDGE
	NetHostConnectionSubtypeEdge = NetHostConnectionSubtypeKey.String("edge")
	// UMTS
	NetHostConnectionSubtypeUmts = NetHostConnectionSubtypeKey.String("umts")
	// CDMA
	NetHostConnectionSubtypeCdma = NetHostConnectionSubtypeKey.String("cdma")
	// EVDO Rel. 0
	NetHostConnectionSubtypeEvdo0 = NetHostConnectionSubtypeKey.String("evdo_0")
	// EVDO Rev. A
	NetHostConnectionSubtypeEvdoA = NetHostConnectionSubtypeKey.String("evdo_a")
	// CDMA2000 1XRTT
	NetHostConnectionSubtypeCdma20001xrtt = NetHostConnectionSubtypeKey.String("cdma2000_1xrtt")
	// HSDPA
	NetHostConnectionSubtypeHsdpa = NetHostConnectionSubtypeKey.String("hsdpa")
	// HSUPA
	NetHostConnectionSubtypeHsupa = NetHostConnectionSubtypeKey.String("hsupa")
	// HSPA
	NetHostConnectionSubtypeHspa = NetHostConnectionSubtypeKey.String("hspa")
	// IDEN
	NetHostConnectionSubtypeIden = NetHostConnectionSubtypeKey.String("iden")
	// EVDO Rev. B
	NetHostConnectionSubtypeEvdoB = NetHostConnectionSubtypeKey.String("evdo_b")
	// LTE
	NetHostConnectionSubtypeLte = NetHostConnectionSubtypeKey.String("lte")
	// EHRPD
	NetHostConnectionSubtypeEhrpd = NetHostConnectionSubtypeKey.String("ehrpd")
	// HSPAP
	NetHostConnectionSubtypeHspap = NetHostConnectionSubtypeKey.String("hspap")
	// GSM
	NetHostConnectionSubtypeGsm = NetHostConnectionSubtypeKey.String("gsm")
	// TD-SCDMA
	NetHostConnectionSubtypeTdScdma = NetHostConnectionSubtypeKey.String("td_scdma")
	// IWLAN
	NetHostConnectionSubtypeIwlan = NetHostConnectionSubtypeKey.String("iwlan")
	// 5G NR (New Radio)
	NetHostConnectionSubtypeNr = NetHostConnectionSubtypeKey.String("nr")
	// 5G NRNSA (New Radio Non-Standalone)
	NetHostConnectionSubtypeNrnsa = NetHostConnectionSubtypeKey.String("nrnsa")
	// LTE CA
	NetHostConnectionSubtypeLteCa = NetHostConnectionSubtypeKey.String("lte_ca")
)

// NetHostCarrierName returns an attribute KeyValue conforming to the
// "net.host.carrier.name" semantic conventions. It represents the name of the
// mobile carrier.
func NetHostCarrierName(val string) attribute.KeyValue {
	return NetHostCarrierNameKey.String(val)
}

// NetHostCarrierMcc returns an attribute KeyValue conforming to the
// "net.host.carrier.mcc" semantic conventions. It represents the mobile
// carrier country code.
func NetHostCarrierMcc(val string) attribute.KeyValue {
	return NetHostCarrierMccKey.String(val)
}

// NetHostCarrierMnc returns an attribute KeyValue conforming to the
// "net.host.carrier.mnc" semantic conventions. It represents the mobile
// carrier network code.
func NetHostCarrierMnc(val string) attribute.KeyValue {
	return NetHostCarrierMncKey.String(val)
}

// NetHostCarrierIcc returns an attribute KeyValue conforming to the
// "net.host.carrier.icc" semantic conventions. It represents the ISO 3166-1
// alpha-2 2-character country code associated with the mobile carrier network.
func NetHostCarrierIcc(val string) attribute.KeyValue {
	return NetHostCarrierIccKey.String(val)
}

// Semantic conventions for HTTP client and server Spans.
const (
	// HTTPRequestContentLengthKey is the attribute Key conforming to the
	// "http.request_content_length" semantic conventions. It represents the
	// size of the request payload body in bytes. This is the number of bytes
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
	HTTPRequestContentLengthKey = attribute.Key("http.request_content_length")

	// HTTPResponseContentLengthKey is the attribute Key conforming to the
	// "http.response_content_length" semantic conventions. It represents the
	// size of the response payload body in bytes. This is the number of bytes
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
	HTTPResponseContentLengthKey = attribute.Key("http.response_content_length")
)

// HTTPRequestContentLength returns an attribute KeyValue conforming to the
// "http.request_content_length" semantic conventions. It represents the size
// of the request payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPRequestContentLength(val int) attribute.KeyValue {
	return HTTPRequestContentLengthKey.Int(val)
}

// HTTPResponseContentLength returns an attribute KeyValue conforming to the
// "http.response_content_length" semantic conventions. It represents the size
// of the response payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length](https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length)
// header. For requests using transport encoding, this should be the compressed
// size.
func HTTPResponseContentLength(val int) attribute.KeyValue {
	return HTTPResponseContentLengthKey.Int(val)
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

// Semantic convention for attributes that describe messaging source on broker
const (
	// MessagingSourceNameKey is the attribute Key conforming to the
	// "messaging.source.name" semantic conventions. It represents the message
	// source name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'MyQueue', 'MyTopic'
	// Note: Source name SHOULD uniquely identify a specific queue, topic, or
	// other entity within the broker. If
	// the broker does not have such notion, the source name SHOULD uniquely
	// identify the broker.
	MessagingSourceNameKey = attribute.Key("messaging.source.name")

	// MessagingSourceTemplateKey is the attribute Key conforming to the
	// "messaging.source.template" semantic conventions. It represents the low
	// cardinality representation of the messaging source name
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: '/customers/{customerID}'
	// Note: Source names could be constructed from templates. An example would
	// be a source name involving a user name or product id. Although the
	// source name in this case is of high cardinality, the underlying template
	// is of low cardinality and can be effectively used for grouping and
	// aggregation.
	MessagingSourceTemplateKey = attribute.Key("messaging.source.template")

	// MessagingSourceTemporaryKey is the attribute Key conforming to the
	// "messaging.source.temporary" semantic conventions. It represents a
	// boolean that is true if the message source is temporary and might not
	// exist anymore after messages are processed.
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	MessagingSourceTemporaryKey = attribute.Key("messaging.source.temporary")

	// MessagingSourceAnonymousKey is the attribute Key conforming to the
	// "messaging.source.anonymous" semantic conventions. It represents a
	// boolean that is true if the message source is anonymous (could be
	// unnamed or have auto-generated name).
	//
	// Type: boolean
	// RequirementLevel: Optional
	// Stability: stable
	MessagingSourceAnonymousKey = attribute.Key("messaging.source.anonymous")
)

// MessagingSourceName returns an attribute KeyValue conforming to the
// "messaging.source.name" semantic conventions. It represents the message
// source name
func MessagingSourceName(val string) attribute.KeyValue {
	return MessagingSourceNameKey.String(val)
}

// MessagingSourceTemplate returns an attribute KeyValue conforming to the
// "messaging.source.template" semantic conventions. It represents the low
// cardinality representation of the messaging source name
func MessagingSourceTemplate(val string) attribute.KeyValue {
	return MessagingSourceTemplateKey.String(val)
}

// MessagingSourceTemporary returns an attribute KeyValue conforming to the
// "messaging.source.temporary" semantic conventions. It represents a boolean
// that is true if the message source is temporary and might not exist anymore
// after messages are processed.
func MessagingSourceTemporary(val bool) attribute.KeyValue {
	return MessagingSourceTemporaryKey.Bool(val)
}

// MessagingSourceAnonymous returns an attribute KeyValue conforming to the
// "messaging.source.anonymous" semantic conventions. It represents a boolean
// that is true if the message source is anonymous (could be unnamed or have
// auto-generated name).
func MessagingSourceAnonymous(val bool) attribute.KeyValue {
	return MessagingSourceAnonymousKey.Bool(val)
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

	// MessagingKafkaClientIDKey is the attribute Key conforming to the
	// "messaging.kafka.client_id" semantic conventions. It represents the
	// client ID for the Consumer or Producer that is handling the message.
	//
	// Type: string
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 'client-5'
	MessagingKafkaClientIDKey = attribute.Key("messaging.kafka.client_id")

	// MessagingKafkaDestinationPartitionKey is the attribute Key conforming to
	// the "messaging.kafka.destination.partition" semantic conventions. It
	// represents the partition the message is sent to.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 2
	MessagingKafkaDestinationPartitionKey = attribute.Key("messaging.kafka.destination.partition")

	// MessagingKafkaSourcePartitionKey is the attribute Key conforming to the
	// "messaging.kafka.source.partition" semantic conventions. It represents
	// the partition the message is received from.
	//
	// Type: int
	// RequirementLevel: Optional
	// Stability: stable
	// Examples: 2
	MessagingKafkaSourcePartitionKey = attribute.Key("messaging.kafka.source.partition")

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

// MessagingKafkaClientID returns an attribute KeyValue conforming to the
// "messaging.kafka.client_id" semantic conventions. It represents the client
// ID for the Consumer or Producer that is handling the message.
func MessagingKafkaClientID(val string) attribute.KeyValue {
	return MessagingKafkaClientIDKey.String(val)
}

// MessagingKafkaDestinationPartition returns an attribute KeyValue
// conforming to the "messaging.kafka.destination.partition" semantic
// conventions. It represents the partition the message is sent to.
func MessagingKafkaDestinationPartition(val int) attribute.KeyValue {
	return MessagingKafkaDestinationPartitionKey.Int(val)
}

// MessagingKafkaSourcePartition returns an attribute KeyValue conforming to
// the "messaging.kafka.source.partition" semantic conventions. It represents
// the partition the message is received from.
func MessagingKafkaSourcePartition(val int) attribute.KeyValue {
	return MessagingKafkaSourcePartitionKey.Int(val)
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

	// MessagingRocketmqClientIDKey is the attribute Key conforming to the
	// "messaging.rocketmq.client_id" semantic conventions. It represents the
	// unique identifier for each client.
	//
	// Type: string
	// RequirementLevel: Required
	// Stability: stable
	// Examples: 'myhost@8742@s8083jm'
	MessagingRocketmqClientIDKey = attribute.Key("messaging.rocketmq.client_id")

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

// MessagingRocketmqClientID returns an attribute KeyValue conforming to the
// "messaging.rocketmq.client_id" semantic conventions. It represents the
// unique identifier for each client.
func MessagingRocketmqClientID(val string) attribute.KeyValue {
	return MessagingRocketmqClientIDKey.String(val)
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
