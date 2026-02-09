// Code generated from semantic convention specification. DO NOT EDIT.

// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package httpconv provides types and functionality for OpenTelemetry semantic
// conventions in the "http" namespace.
package httpconv

import (
	"context"
	"sync"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
)

var (
	addOptPool = &sync.Pool{New: func() any { return &[]metric.AddOption{} }}
	recOptPool = &sync.Pool{New: func() any { return &[]metric.RecordOption{} }}
)

// ErrorTypeAttr is an attribute conforming to the error.type semantic
// conventions. It represents the describes a class of error the operation ended
// with.
type ErrorTypeAttr string

var (
	// ErrorTypeOther is a fallback error value to be used when the instrumentation
	// doesn't define a custom value.
	ErrorTypeOther ErrorTypeAttr = "_OTHER"
)

// ConnectionStateAttr is an attribute conforming to the http.connection.state
// semantic conventions. It represents the state of the HTTP connection in the
// HTTP connection pool.
type ConnectionStateAttr string

var (
	// ConnectionStateActive is the active state.
	ConnectionStateActive ConnectionStateAttr = "active"
	// ConnectionStateIdle is the idle state.
	ConnectionStateIdle ConnectionStateAttr = "idle"
)

// RequestMethodAttr is an attribute conforming to the http.request.method
// semantic conventions. It represents the HTTP request method.
type RequestMethodAttr string

var (
	// RequestMethodConnect is the CONNECT method.
	RequestMethodConnect RequestMethodAttr = "CONNECT"
	// RequestMethodDelete is the DELETE method.
	RequestMethodDelete RequestMethodAttr = "DELETE"
	// RequestMethodGet is the GET method.
	RequestMethodGet RequestMethodAttr = "GET"
	// RequestMethodHead is the HEAD method.
	RequestMethodHead RequestMethodAttr = "HEAD"
	// RequestMethodOptions is the OPTIONS method.
	RequestMethodOptions RequestMethodAttr = "OPTIONS"
	// RequestMethodPatch is the PATCH method.
	RequestMethodPatch RequestMethodAttr = "PATCH"
	// RequestMethodPost is the POST method.
	RequestMethodPost RequestMethodAttr = "POST"
	// RequestMethodPut is the PUT method.
	RequestMethodPut RequestMethodAttr = "PUT"
	// RequestMethodTrace is the TRACE method.
	RequestMethodTrace RequestMethodAttr = "TRACE"
	// RequestMethodOther is the any HTTP method that the instrumentation has no
	// prior knowledge of.
	RequestMethodOther RequestMethodAttr = "_OTHER"
)

// UserAgentSyntheticTypeAttr is an attribute conforming to the
// user_agent.synthetic.type semantic conventions. It represents the specifies
// the category of synthetic traffic, such as tests or bots.
type UserAgentSyntheticTypeAttr string

var (
	// UserAgentSyntheticTypeBot is the bot source.
	UserAgentSyntheticTypeBot UserAgentSyntheticTypeAttr = "bot"
	// UserAgentSyntheticTypeTest is the synthetic test source.
	UserAgentSyntheticTypeTest UserAgentSyntheticTypeAttr = "test"
)

// ClientActiveRequests is an instrument used to record metric values conforming
// to the "http.client.active_requests" semantic conventions. It represents the
// number of active HTTP requests.
type ClientActiveRequests struct {
	metric.Int64UpDownCounter
}

var newClientActiveRequestsOpts = []metric.Int64UpDownCounterOption{
	metric.WithDescription("Number of active HTTP requests."),
	metric.WithUnit("{request}"),
}

// NewClientActiveRequests returns a new ClientActiveRequests instrument.
func NewClientActiveRequests(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (ClientActiveRequests, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientActiveRequests{noop.Int64UpDownCounter{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientActiveRequestsOpts
	} else {
		opt = append(opt, newClientActiveRequestsOpts...)
	}

	i, err := m.Int64UpDownCounter(
		"http.client.active_requests",
		opt...,
	)
	if err != nil {
		return ClientActiveRequests{noop.Int64UpDownCounter{}}, err
	}
	return ClientActiveRequests{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientActiveRequests) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (ClientActiveRequests) Name() string {
	return "http.client.active_requests"
}

// Unit returns the semantic convention unit of the instrument
func (ClientActiveRequests) Unit() string {
	return "{request}"
}

// Description returns the semantic convention description of the instrument
func (ClientActiveRequests) Description() string {
	return "Number of active HTTP requests."
}

// Add adds incr to the existing count for attrs.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
func (m ClientActiveRequests) Add(
	ctx context.Context,
	incr int64,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
func (m ClientActiveRequests) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrURLTemplate returns an optional attribute for the "url.template" semantic
// convention. It represents the low-cardinality template of an
// [absolute path reference].
//
// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
func (ClientActiveRequests) AttrURLTemplate(val string) attribute.KeyValue {
	return attribute.String("url.template", val)
}

// AttrRequestMethod returns an optional attribute for the "http.request.method"
// semantic convention. It represents the HTTP request method.
func (ClientActiveRequests) AttrRequestMethod(val RequestMethodAttr) attribute.KeyValue {
	return attribute.String("http.request.method", string(val))
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientActiveRequests) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// ClientConnectionDuration is an instrument used to record metric values
// conforming to the "http.client.connection.duration" semantic conventions. It
// represents the duration of the successfully established outbound HTTP
// connections.
type ClientConnectionDuration struct {
	metric.Float64Histogram
}

var newClientConnectionDurationOpts = []metric.Float64HistogramOption{
	metric.WithDescription("The duration of the successfully established outbound HTTP connections."),
	metric.WithUnit("s"),
}

// NewClientConnectionDuration returns a new ClientConnectionDuration instrument.
func NewClientConnectionDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (ClientConnectionDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientConnectionDuration{noop.Float64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientConnectionDurationOpts
	} else {
		opt = append(opt, newClientConnectionDurationOpts...)
	}

	i, err := m.Float64Histogram(
		"http.client.connection.duration",
		opt...,
	)
	if err != nil {
		return ClientConnectionDuration{noop.Float64Histogram{}}, err
	}
	return ClientConnectionDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientConnectionDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ClientConnectionDuration) Name() string {
	return "http.client.connection.duration"
}

// Unit returns the semantic convention unit of the instrument
func (ClientConnectionDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (ClientConnectionDuration) Description() string {
	return "The duration of the successfully established outbound HTTP connections."
}

// Record records val to the current distribution for attrs.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
func (m ClientConnectionDuration) Record(
	ctx context.Context,
	val float64,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
func (m ClientConnectionDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrNetworkPeerAddress returns an optional attribute for the
// "network.peer.address" semantic convention. It represents the peer address of
// the network connection - IP address or Unix domain socket name.
func (ClientConnectionDuration) AttrNetworkPeerAddress(val string) attribute.KeyValue {
	return attribute.String("network.peer.address", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ClientConnectionDuration) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientConnectionDuration) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// ClientOpenConnections is an instrument used to record metric values conforming
// to the "http.client.open_connections" semantic conventions. It represents the
// number of outbound HTTP connections that are currently active or idle on the
// client.
type ClientOpenConnections struct {
	metric.Int64UpDownCounter
}

var newClientOpenConnectionsOpts = []metric.Int64UpDownCounterOption{
	metric.WithDescription("Number of outbound HTTP connections that are currently active or idle on the client."),
	metric.WithUnit("{connection}"),
}

// NewClientOpenConnections returns a new ClientOpenConnections instrument.
func NewClientOpenConnections(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (ClientOpenConnections, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientOpenConnections{noop.Int64UpDownCounter{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientOpenConnectionsOpts
	} else {
		opt = append(opt, newClientOpenConnectionsOpts...)
	}

	i, err := m.Int64UpDownCounter(
		"http.client.open_connections",
		opt...,
	)
	if err != nil {
		return ClientOpenConnections{noop.Int64UpDownCounter{}}, err
	}
	return ClientOpenConnections{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientOpenConnections) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (ClientOpenConnections) Name() string {
	return "http.client.open_connections"
}

// Unit returns the semantic convention unit of the instrument
func (ClientOpenConnections) Unit() string {
	return "{connection}"
}

// Description returns the semantic convention description of the instrument
func (ClientOpenConnections) Description() string {
	return "Number of outbound HTTP connections that are currently active or idle on the client."
}

// Add adds incr to the existing count for attrs.
//
// The connectionState is the state of the HTTP connection in the HTTP connection
// pool.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
func (m ClientOpenConnections) Add(
	ctx context.Context,
	incr int64,
	connectionState ConnectionStateAttr,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.connection.state", string(connectionState)),
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
func (m ClientOpenConnections) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrNetworkPeerAddress returns an optional attribute for the
// "network.peer.address" semantic convention. It represents the peer address of
// the network connection - IP address or Unix domain socket name.
func (ClientOpenConnections) AttrNetworkPeerAddress(val string) attribute.KeyValue {
	return attribute.String("network.peer.address", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ClientOpenConnections) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientOpenConnections) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// ClientRequestBodySize is an instrument used to record metric values conforming
// to the "http.client.request.body.size" semantic conventions. It represents the
// size of HTTP client request bodies.
type ClientRequestBodySize struct {
	metric.Int64Histogram
}

var newClientRequestBodySizeOpts = []metric.Int64HistogramOption{
	metric.WithDescription("Size of HTTP client request bodies."),
	metric.WithUnit("By"),
}

// NewClientRequestBodySize returns a new ClientRequestBodySize instrument.
func NewClientRequestBodySize(
	m metric.Meter,
	opt ...metric.Int64HistogramOption,
) (ClientRequestBodySize, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientRequestBodySize{noop.Int64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientRequestBodySizeOpts
	} else {
		opt = append(opt, newClientRequestBodySizeOpts...)
	}

	i, err := m.Int64Histogram(
		"http.client.request.body.size",
		opt...,
	)
	if err != nil {
		return ClientRequestBodySize{noop.Int64Histogram{}}, err
	}
	return ClientRequestBodySize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientRequestBodySize) Inst() metric.Int64Histogram {
	return m.Int64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ClientRequestBodySize) Name() string {
	return "http.client.request.body.size"
}

// Unit returns the semantic convention unit of the instrument
func (ClientRequestBodySize) Unit() string {
	return "By"
}

// Description returns the semantic convention description of the instrument
func (ClientRequestBodySize) Description() string {
	return "Size of HTTP client request bodies."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
//
// The size of the request payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ClientRequestBodySize) Record(
	ctx context.Context,
	val int64,
	requestMethod RequestMethodAttr,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Int64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// The size of the request payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ClientRequestBodySize) RecordSet(ctx context.Context, val int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ClientRequestBodySize) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ClientRequestBodySize) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ClientRequestBodySize) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrURLTemplate returns an optional attribute for the "url.template" semantic
// convention. It represents the low-cardinality template of an
// [absolute path reference].
//
// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
func (ClientRequestBodySize) AttrURLTemplate(val string) attribute.KeyValue {
	return attribute.String("url.template", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ClientRequestBodySize) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientRequestBodySize) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// ClientRequestDuration is an instrument used to record metric values conforming
// to the "http.client.request.duration" semantic conventions. It represents the
// duration of HTTP client requests.
type ClientRequestDuration struct {
	metric.Float64Histogram
}

var newClientRequestDurationOpts = []metric.Float64HistogramOption{
	metric.WithDescription("Duration of HTTP client requests."),
	metric.WithUnit("s"),
}

// NewClientRequestDuration returns a new ClientRequestDuration instrument.
func NewClientRequestDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (ClientRequestDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientRequestDuration{noop.Float64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientRequestDurationOpts
	} else {
		opt = append(opt, newClientRequestDurationOpts...)
	}

	i, err := m.Float64Histogram(
		"http.client.request.duration",
		opt...,
	)
	if err != nil {
		return ClientRequestDuration{noop.Float64Histogram{}}, err
	}
	return ClientRequestDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientRequestDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ClientRequestDuration) Name() string {
	return "http.client.request.duration"
}

// Unit returns the semantic convention unit of the instrument
func (ClientRequestDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (ClientRequestDuration) Description() string {
	return "Duration of HTTP client requests."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
func (m ClientRequestDuration) Record(
	ctx context.Context,
	val float64,
	requestMethod RequestMethodAttr,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
func (m ClientRequestDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ClientRequestDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ClientRequestDuration) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ClientRequestDuration) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ClientRequestDuration) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientRequestDuration) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// AttrURLTemplate returns an optional attribute for the "url.template" semantic
// convention. It represents the low-cardinality template of an
// [absolute path reference].
//
// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
func (ClientRequestDuration) AttrURLTemplate(val string) attribute.KeyValue {
	return attribute.String("url.template", val)
}

// ClientResponseBodySize is an instrument used to record metric values
// conforming to the "http.client.response.body.size" semantic conventions. It
// represents the size of HTTP client response bodies.
type ClientResponseBodySize struct {
	metric.Int64Histogram
}

var newClientResponseBodySizeOpts = []metric.Int64HistogramOption{
	metric.WithDescription("Size of HTTP client response bodies."),
	metric.WithUnit("By"),
}

// NewClientResponseBodySize returns a new ClientResponseBodySize instrument.
func NewClientResponseBodySize(
	m metric.Meter,
	opt ...metric.Int64HistogramOption,
) (ClientResponseBodySize, error) {
	// Check if the meter is nil.
	if m == nil {
		return ClientResponseBodySize{noop.Int64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newClientResponseBodySizeOpts
	} else {
		opt = append(opt, newClientResponseBodySizeOpts...)
	}

	i, err := m.Int64Histogram(
		"http.client.response.body.size",
		opt...,
	)
	if err != nil {
		return ClientResponseBodySize{noop.Int64Histogram{}}, err
	}
	return ClientResponseBodySize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ClientResponseBodySize) Inst() metric.Int64Histogram {
	return m.Int64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ClientResponseBodySize) Name() string {
	return "http.client.response.body.size"
}

// Unit returns the semantic convention unit of the instrument
func (ClientResponseBodySize) Unit() string {
	return "By"
}

// Description returns the semantic convention description of the instrument
func (ClientResponseBodySize) Description() string {
	return "Size of HTTP client response bodies."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The serverAddress is the server domain name if available without reverse DNS
// lookup; otherwise, IP address or Unix domain socket name.
//
// The serverPort is the server port number.
//
// All additional attrs passed are included in the recorded value.
//
// The size of the response payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ClientResponseBodySize) Record(
	ctx context.Context,
	val int64,
	requestMethod RequestMethodAttr,
	serverAddress string,
	serverPort int,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("server.address", serverAddress),
				attribute.Int("server.port", serverPort),
			)...,
		),
	)

	m.Int64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// The size of the response payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ClientResponseBodySize) RecordSet(ctx context.Context, val int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ClientResponseBodySize) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ClientResponseBodySize) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ClientResponseBodySize) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrURLTemplate returns an optional attribute for the "url.template" semantic
// convention. It represents the low-cardinality template of an
// [absolute path reference].
//
// [absolute path reference]: https://www.rfc-editor.org/rfc/rfc3986#section-4.2
func (ClientResponseBodySize) AttrURLTemplate(val string) attribute.KeyValue {
	return attribute.String("url.template", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ClientResponseBodySize) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrURLScheme returns an optional attribute for the "url.scheme" semantic
// convention. It represents the [URI scheme] component identifying the used
// protocol.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (ClientResponseBodySize) AttrURLScheme(val string) attribute.KeyValue {
	return attribute.String("url.scheme", val)
}

// ServerActiveRequests is an instrument used to record metric values conforming
// to the "http.server.active_requests" semantic conventions. It represents the
// number of active HTTP server requests.
type ServerActiveRequests struct {
	metric.Int64UpDownCounter
}

var newServerActiveRequestsOpts = []metric.Int64UpDownCounterOption{
	metric.WithDescription("Number of active HTTP server requests."),
	metric.WithUnit("{request}"),
}

// NewServerActiveRequests returns a new ServerActiveRequests instrument.
func NewServerActiveRequests(
	m metric.Meter,
	opt ...metric.Int64UpDownCounterOption,
) (ServerActiveRequests, error) {
	// Check if the meter is nil.
	if m == nil {
		return ServerActiveRequests{noop.Int64UpDownCounter{}}, nil
	}

	if len(opt) == 0 {
		opt = newServerActiveRequestsOpts
	} else {
		opt = append(opt, newServerActiveRequestsOpts...)
	}

	i, err := m.Int64UpDownCounter(
		"http.server.active_requests",
		opt...,
	)
	if err != nil {
		return ServerActiveRequests{noop.Int64UpDownCounter{}}, err
	}
	return ServerActiveRequests{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ServerActiveRequests) Inst() metric.Int64UpDownCounter {
	return m.Int64UpDownCounter
}

// Name returns the semantic convention name of the instrument.
func (ServerActiveRequests) Name() string {
	return "http.server.active_requests"
}

// Unit returns the semantic convention unit of the instrument
func (ServerActiveRequests) Unit() string {
	return "{request}"
}

// Description returns the semantic convention description of the instrument
func (ServerActiveRequests) Description() string {
	return "Number of active HTTP server requests."
}

// Add adds incr to the existing count for attrs.
//
// The requestMethod is the HTTP request method.
//
// The urlScheme is the the [URI scheme] component identifying the used protocol.
//
// All additional attrs passed are included in the recorded value.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (m ServerActiveRequests) Add(
	ctx context.Context,
	incr int64,
	requestMethod RequestMethodAttr,
	urlScheme string,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("url.scheme", urlScheme),
			)...,
		),
	)

	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AddSet adds incr to the existing count for set.
func (m ServerActiveRequests) AddSet(ctx context.Context, incr int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64UpDownCounter.Add(ctx, incr)
		return
	}

	o := addOptPool.Get().(*[]metric.AddOption)
	defer func() {
		*o = (*o)[:0]
		addOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64UpDownCounter.Add(ctx, incr, *o...)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the name of the local HTTP server that
// received the request.
func (ServerActiveRequests) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the port of the local HTTP server that received the
// request.
func (ServerActiveRequests) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// ServerRequestBodySize is an instrument used to record metric values conforming
// to the "http.server.request.body.size" semantic conventions. It represents the
// size of HTTP server request bodies.
type ServerRequestBodySize struct {
	metric.Int64Histogram
}

var newServerRequestBodySizeOpts = []metric.Int64HistogramOption{
	metric.WithDescription("Size of HTTP server request bodies."),
	metric.WithUnit("By"),
}

// NewServerRequestBodySize returns a new ServerRequestBodySize instrument.
func NewServerRequestBodySize(
	m metric.Meter,
	opt ...metric.Int64HistogramOption,
) (ServerRequestBodySize, error) {
	// Check if the meter is nil.
	if m == nil {
		return ServerRequestBodySize{noop.Int64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newServerRequestBodySizeOpts
	} else {
		opt = append(opt, newServerRequestBodySizeOpts...)
	}

	i, err := m.Int64Histogram(
		"http.server.request.body.size",
		opt...,
	)
	if err != nil {
		return ServerRequestBodySize{noop.Int64Histogram{}}, err
	}
	return ServerRequestBodySize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ServerRequestBodySize) Inst() metric.Int64Histogram {
	return m.Int64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ServerRequestBodySize) Name() string {
	return "http.server.request.body.size"
}

// Unit returns the semantic convention unit of the instrument
func (ServerRequestBodySize) Unit() string {
	return "By"
}

// Description returns the semantic convention description of the instrument
func (ServerRequestBodySize) Description() string {
	return "Size of HTTP server request bodies."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The urlScheme is the the [URI scheme] component identifying the used protocol.
//
// All additional attrs passed are included in the recorded value.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
//
// The size of the request payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ServerRequestBodySize) Record(
	ctx context.Context,
	val int64,
	requestMethod RequestMethodAttr,
	urlScheme string,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("url.scheme", urlScheme),
			)...,
		),
	)

	m.Int64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// The size of the request payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ServerRequestBodySize) RecordSet(ctx context.Context, val int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ServerRequestBodySize) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ServerRequestBodySize) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrRoute returns an optional attribute for the "http.route" semantic
// convention. It represents the matched route, that is, the path template in the
// format used by the respective server framework.
func (ServerRequestBodySize) AttrRoute(val string) attribute.KeyValue {
	return attribute.String("http.route", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ServerRequestBodySize) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ServerRequestBodySize) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the name of the local HTTP server that
// received the request.
func (ServerRequestBodySize) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the port of the local HTTP server that received the
// request.
func (ServerRequestBodySize) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// AttrUserAgentSyntheticType returns an optional attribute for the
// "user_agent.synthetic.type" semantic convention. It represents the specifies
// the category of synthetic traffic, such as tests or bots.
func (ServerRequestBodySize) AttrUserAgentSyntheticType(val UserAgentSyntheticTypeAttr) attribute.KeyValue {
	return attribute.String("user_agent.synthetic.type", string(val))
}

// ServerRequestDuration is an instrument used to record metric values conforming
// to the "http.server.request.duration" semantic conventions. It represents the
// duration of HTTP server requests.
type ServerRequestDuration struct {
	metric.Float64Histogram
}

var newServerRequestDurationOpts = []metric.Float64HistogramOption{
	metric.WithDescription("Duration of HTTP server requests."),
	metric.WithUnit("s"),
}

// NewServerRequestDuration returns a new ServerRequestDuration instrument.
func NewServerRequestDuration(
	m metric.Meter,
	opt ...metric.Float64HistogramOption,
) (ServerRequestDuration, error) {
	// Check if the meter is nil.
	if m == nil {
		return ServerRequestDuration{noop.Float64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newServerRequestDurationOpts
	} else {
		opt = append(opt, newServerRequestDurationOpts...)
	}

	i, err := m.Float64Histogram(
		"http.server.request.duration",
		opt...,
	)
	if err != nil {
		return ServerRequestDuration{noop.Float64Histogram{}}, err
	}
	return ServerRequestDuration{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ServerRequestDuration) Inst() metric.Float64Histogram {
	return m.Float64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ServerRequestDuration) Name() string {
	return "http.server.request.duration"
}

// Unit returns the semantic convention unit of the instrument
func (ServerRequestDuration) Unit() string {
	return "s"
}

// Description returns the semantic convention description of the instrument
func (ServerRequestDuration) Description() string {
	return "Duration of HTTP server requests."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The urlScheme is the the [URI scheme] component identifying the used protocol.
//
// All additional attrs passed are included in the recorded value.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
func (m ServerRequestDuration) Record(
	ctx context.Context,
	val float64,
	requestMethod RequestMethodAttr,
	urlScheme string,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("url.scheme", urlScheme),
			)...,
		),
	)

	m.Float64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
func (m ServerRequestDuration) RecordSet(ctx context.Context, val float64, set attribute.Set) {
	if set.Len() == 0 {
		m.Float64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Float64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ServerRequestDuration) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ServerRequestDuration) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrRoute returns an optional attribute for the "http.route" semantic
// convention. It represents the matched route, that is, the path template in the
// format used by the respective server framework.
func (ServerRequestDuration) AttrRoute(val string) attribute.KeyValue {
	return attribute.String("http.route", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ServerRequestDuration) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ServerRequestDuration) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the name of the local HTTP server that
// received the request.
func (ServerRequestDuration) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the port of the local HTTP server that received the
// request.
func (ServerRequestDuration) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// AttrUserAgentSyntheticType returns an optional attribute for the
// "user_agent.synthetic.type" semantic convention. It represents the specifies
// the category of synthetic traffic, such as tests or bots.
func (ServerRequestDuration) AttrUserAgentSyntheticType(val UserAgentSyntheticTypeAttr) attribute.KeyValue {
	return attribute.String("user_agent.synthetic.type", string(val))
}

// ServerResponseBodySize is an instrument used to record metric values
// conforming to the "http.server.response.body.size" semantic conventions. It
// represents the size of HTTP server response bodies.
type ServerResponseBodySize struct {
	metric.Int64Histogram
}

var newServerResponseBodySizeOpts = []metric.Int64HistogramOption{
	metric.WithDescription("Size of HTTP server response bodies."),
	metric.WithUnit("By"),
}

// NewServerResponseBodySize returns a new ServerResponseBodySize instrument.
func NewServerResponseBodySize(
	m metric.Meter,
	opt ...metric.Int64HistogramOption,
) (ServerResponseBodySize, error) {
	// Check if the meter is nil.
	if m == nil {
		return ServerResponseBodySize{noop.Int64Histogram{}}, nil
	}

	if len(opt) == 0 {
		opt = newServerResponseBodySizeOpts
	} else {
		opt = append(opt, newServerResponseBodySizeOpts...)
	}

	i, err := m.Int64Histogram(
		"http.server.response.body.size",
		opt...,
	)
	if err != nil {
		return ServerResponseBodySize{noop.Int64Histogram{}}, err
	}
	return ServerResponseBodySize{i}, nil
}

// Inst returns the underlying metric instrument.
func (m ServerResponseBodySize) Inst() metric.Int64Histogram {
	return m.Int64Histogram
}

// Name returns the semantic convention name of the instrument.
func (ServerResponseBodySize) Name() string {
	return "http.server.response.body.size"
}

// Unit returns the semantic convention unit of the instrument
func (ServerResponseBodySize) Unit() string {
	return "By"
}

// Description returns the semantic convention description of the instrument
func (ServerResponseBodySize) Description() string {
	return "Size of HTTP server response bodies."
}

// Record records val to the current distribution for attrs.
//
// The requestMethod is the HTTP request method.
//
// The urlScheme is the the [URI scheme] component identifying the used protocol.
//
// All additional attrs passed are included in the recorded value.
//
// [URI scheme]: https://www.rfc-editor.org/rfc/rfc3986#section-3.1
//
// The size of the response payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ServerResponseBodySize) Record(
	ctx context.Context,
	val int64,
	requestMethod RequestMethodAttr,
	urlScheme string,
	attrs ...attribute.KeyValue,
) {
	if len(attrs) == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(
		*o,
		metric.WithAttributes(
			append(
				attrs,
				attribute.String("http.request.method", string(requestMethod)),
				attribute.String("url.scheme", urlScheme),
			)...,
		),
	)

	m.Int64Histogram.Record(ctx, val, *o...)
}

// RecordSet records val to the current distribution for set.
//
// The size of the response payload body in bytes. This is the number of bytes
// transferred excluding headers and is often, but not always, present as the
// [Content-Length] header. For requests using transport encoding, this should be
// the compressed size.
//
// [Content-Length]: https://www.rfc-editor.org/rfc/rfc9110.html#field.content-length
func (m ServerResponseBodySize) RecordSet(ctx context.Context, val int64, set attribute.Set) {
	if set.Len() == 0 {
		m.Int64Histogram.Record(ctx, val)
		return
	}

	o := recOptPool.Get().(*[]metric.RecordOption)
	defer func() {
		*o = (*o)[:0]
		recOptPool.Put(o)
	}()

	*o = append(*o, metric.WithAttributeSet(set))
	m.Int64Histogram.Record(ctx, val, *o...)
}

// AttrErrorType returns an optional attribute for the "error.type" semantic
// convention. It represents the describes a class of error the operation ended
// with.
func (ServerResponseBodySize) AttrErrorType(val ErrorTypeAttr) attribute.KeyValue {
	return attribute.String("error.type", string(val))
}

// AttrResponseStatusCode returns an optional attribute for the
// "http.response.status_code" semantic convention. It represents the
// [HTTP response status code].
//
// [HTTP response status code]: https://tools.ietf.org/html/rfc7231#section-6
func (ServerResponseBodySize) AttrResponseStatusCode(val int) attribute.KeyValue {
	return attribute.Int("http.response.status_code", val)
}

// AttrRoute returns an optional attribute for the "http.route" semantic
// convention. It represents the matched route, that is, the path template in the
// format used by the respective server framework.
func (ServerResponseBodySize) AttrRoute(val string) attribute.KeyValue {
	return attribute.String("http.route", val)
}

// AttrNetworkProtocolName returns an optional attribute for the
// "network.protocol.name" semantic convention. It represents the
// [OSI application layer] or non-OSI equivalent.
//
// [OSI application layer]: https://wikipedia.org/wiki/Application_layer
func (ServerResponseBodySize) AttrNetworkProtocolName(val string) attribute.KeyValue {
	return attribute.String("network.protocol.name", val)
}

// AttrNetworkProtocolVersion returns an optional attribute for the
// "network.protocol.version" semantic convention. It represents the actual
// version of the protocol used for network communication.
func (ServerResponseBodySize) AttrNetworkProtocolVersion(val string) attribute.KeyValue {
	return attribute.String("network.protocol.version", val)
}

// AttrServerAddress returns an optional attribute for the "server.address"
// semantic convention. It represents the name of the local HTTP server that
// received the request.
func (ServerResponseBodySize) AttrServerAddress(val string) attribute.KeyValue {
	return attribute.String("server.address", val)
}

// AttrServerPort returns an optional attribute for the "server.port" semantic
// convention. It represents the port of the local HTTP server that received the
// request.
func (ServerResponseBodySize) AttrServerPort(val int) attribute.KeyValue {
	return attribute.Int("server.port", val)
}

// AttrUserAgentSyntheticType returns an optional attribute for the
// "user_agent.synthetic.type" semantic convention. It represents the specifies
// the category of synthetic traffic, such as tests or bots.
func (ServerResponseBodySize) AttrUserAgentSyntheticType(val UserAgentSyntheticTypeAttr) attribute.KeyValue {
	return attribute.String("user_agent.synthetic.type", string(val))
}
