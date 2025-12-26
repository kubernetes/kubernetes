// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelgrpc // import "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/trace"
	"google.golang.org/grpc/stats"
)

// ScopeName is the instrumentation scope name.
const ScopeName = "go.opentelemetry.io/contrib/instrumentation/google.golang.org/grpc/otelgrpc"

// InterceptorFilter is a predicate used to determine whether a given request in
// interceptor info should be instrumented. A InterceptorFilter must return true if
// the request should be traced.
//
// Deprecated: Use stats handlers instead.
type InterceptorFilter func(*InterceptorInfo) bool

// Filter is a predicate used to determine whether a given request in
// should be instrumented by the attached RPC tag info.
// A Filter must return true if the request should be instrumented.
type Filter func(*stats.RPCTagInfo) bool

// config is a group of options for this instrumentation.
type config struct {
	Filter            Filter
	InterceptorFilter InterceptorFilter
	Propagators       propagation.TextMapPropagator
	TracerProvider    trace.TracerProvider
	MeterProvider     metric.MeterProvider
	SpanStartOptions  []trace.SpanStartOption
	SpanAttributes    []attribute.KeyValue
	MetricAttributes  []attribute.KeyValue

	PublicEndpoint   bool
	PublicEndpointFn func(ctx context.Context, info *stats.RPCTagInfo) bool

	ReceivedEvent bool
	SentEvent     bool
}

// Option applies an option value for a config.
type Option interface {
	apply(*config)
}

// newConfig returns a config configured with all the passed Options.
func newConfig(opts []Option) *config {
	c := &config{
		Propagators:    otel.GetTextMapPropagator(),
		TracerProvider: otel.GetTracerProvider(),
		MeterProvider:  otel.GetMeterProvider(),
	}
	for _, o := range opts {
		o.apply(c)
	}
	return c
}

type publicEndpointOption struct{ p bool }

func (o publicEndpointOption) apply(c *config) {
	c.PublicEndpoint = o.p
}

// WithPublicEndpoint configures the Handler to link the span with an incoming
// span context. If this option is not provided, then the association is a child
// association instead of a link.
func WithPublicEndpoint() Option {
	return publicEndpointOption{p: true}
}

type publicEndpointFnOption struct {
	fn func(context.Context, *stats.RPCTagInfo) bool
}

func (o publicEndpointFnOption) apply(c *config) {
	if o.fn != nil {
		c.PublicEndpointFn = o.fn
	}
}

// WithPublicEndpointFn runs with every request, and allows conditionally
// configuring the Handler to link the span with an incoming span context. If
// this option is not provided or returns false, then the association is a
// child association instead of a link.
// Note: WithPublicEndpoint takes precedence over WithPublicEndpointFn.
func WithPublicEndpointFn(fn func(context.Context, *stats.RPCTagInfo) bool) Option {
	return publicEndpointFnOption{fn: fn}
}

type propagatorsOption struct{ p propagation.TextMapPropagator }

func (o propagatorsOption) apply(c *config) {
	if o.p != nil {
		c.Propagators = o.p
	}
}

// WithPropagators returns an Option to use the Propagators when extracting
// and injecting trace context from requests.
func WithPropagators(p propagation.TextMapPropagator) Option {
	return propagatorsOption{p: p}
}

type tracerProviderOption struct{ tp trace.TracerProvider }

func (o tracerProviderOption) apply(c *config) {
	if o.tp != nil {
		c.TracerProvider = o.tp
	}
}

// WithInterceptorFilter returns an Option to use the request filter.
//
// Deprecated: Use stats handlers instead.
func WithInterceptorFilter(f InterceptorFilter) Option {
	return interceptorFilterOption{f: f}
}

type interceptorFilterOption struct {
	f InterceptorFilter
}

func (o interceptorFilterOption) apply(c *config) {
	if o.f != nil {
		c.InterceptorFilter = o.f
	}
}

// WithFilter returns an Option to use the request filter.
func WithFilter(f Filter) Option {
	return filterOption{f: f}
}

type filterOption struct {
	f Filter
}

func (o filterOption) apply(c *config) {
	if o.f != nil {
		c.Filter = o.f
	}
}

// WithTracerProvider returns an Option to use the TracerProvider when
// creating a Tracer.
func WithTracerProvider(tp trace.TracerProvider) Option {
	return tracerProviderOption{tp: tp}
}

type meterProviderOption struct{ mp metric.MeterProvider }

func (o meterProviderOption) apply(c *config) {
	if o.mp != nil {
		c.MeterProvider = o.mp
	}
}

// WithMeterProvider returns an Option to use the MeterProvider when
// creating a Meter. If this option is not provide the global MeterProvider will be used.
func WithMeterProvider(mp metric.MeterProvider) Option {
	return meterProviderOption{mp: mp}
}

// Event type that can be recorded, see WithMessageEvents.
type Event int

// Different types of events that can be recorded, see WithMessageEvents.
const (
	ReceivedEvents Event = iota
	SentEvents
)

type messageEventsProviderOption struct {
	events []Event
}

func (m messageEventsProviderOption) apply(c *config) {
	for _, e := range m.events {
		switch e {
		case ReceivedEvents:
			c.ReceivedEvent = true
		case SentEvents:
			c.SentEvent = true
		}
	}
}

// WithMessageEvents configures the Handler to record the specified events
// (span.AddEvent) on spans. By default only summary attributes are added at the
// end of the request.
//
// Valid events are:
//   - ReceivedEvents: Record the number of bytes read after every gRPC read operation.
//   - SentEvents: Record the number of bytes written after every gRPC write operation.
func WithMessageEvents(events ...Event) Option {
	return messageEventsProviderOption{events: events}
}

type spanStartOption struct{ opts []trace.SpanStartOption }

func (o spanStartOption) apply(c *config) {
	c.SpanStartOptions = append(c.SpanStartOptions, o.opts...)
}

// WithSpanOptions configures an additional set of
// trace.SpanOptions, which are applied to each new span.
//
// Deprecated: It is only used by the deprecated interceptor, and is unused by [NewClientHandler] and [NewServerHandler].
func WithSpanOptions(opts ...trace.SpanStartOption) Option {
	return spanStartOption{opts}
}

type spanAttributesOption struct{ a []attribute.KeyValue }

func (o spanAttributesOption) apply(c *config) {
	if o.a != nil {
		c.SpanAttributes = o.a
	}
}

// WithSpanAttributes returns an Option to add custom attributes to the spans.
func WithSpanAttributes(a ...attribute.KeyValue) Option {
	return spanAttributesOption{a: a}
}

type metricAttributesOption struct{ a []attribute.KeyValue }

func (o metricAttributesOption) apply(c *config) {
	if o.a != nil {
		c.MetricAttributes = o.a
	}
}

// WithMetricAttributes returns an Option to add custom attributes to the metrics.
func WithMetricAttributes(a ...attribute.KeyValue) Option {
	return metricAttributesOption{a: a}
}
