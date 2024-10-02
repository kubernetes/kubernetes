// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelhttp // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

import (
	"context"
	"io"
	"net/http"
	"net/http/httptrace"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconvutil"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/propagation"
	semconv "go.opentelemetry.io/otel/semconv/v1.20.0"
	"go.opentelemetry.io/otel/trace"
)

// Transport implements the http.RoundTripper interface and wraps
// outbound HTTP(S) requests with a span and enriches it with metrics.
type Transport struct {
	rt http.RoundTripper

	tracer            trace.Tracer
	meter             metric.Meter
	propagators       propagation.TextMapPropagator
	spanStartOptions  []trace.SpanStartOption
	filters           []Filter
	spanNameFormatter func(string, *http.Request) string
	clientTrace       func(context.Context) *httptrace.ClientTrace

	requestBytesCounter  metric.Int64Counter
	responseBytesCounter metric.Int64Counter
	latencyMeasure       metric.Float64Histogram
}

var _ http.RoundTripper = &Transport{}

// NewTransport wraps the provided http.RoundTripper with one that
// starts a span, injects the span context into the outbound request headers,
// and enriches it with metrics.
//
// If the provided http.RoundTripper is nil, http.DefaultTransport will be used
// as the base http.RoundTripper.
func NewTransport(base http.RoundTripper, opts ...Option) *Transport {
	if base == nil {
		base = http.DefaultTransport
	}

	t := Transport{
		rt: base,
	}

	defaultOpts := []Option{
		WithSpanOptions(trace.WithSpanKind(trace.SpanKindClient)),
		WithSpanNameFormatter(defaultTransportFormatter),
	}

	c := newConfig(append(defaultOpts, opts...)...)
	t.applyConfig(c)
	t.createMeasures()

	return &t
}

func (t *Transport) applyConfig(c *config) {
	t.tracer = c.Tracer
	t.meter = c.Meter
	t.propagators = c.Propagators
	t.spanStartOptions = c.SpanStartOptions
	t.filters = c.Filters
	t.spanNameFormatter = c.SpanNameFormatter
	t.clientTrace = c.ClientTrace
}

func (t *Transport) createMeasures() {
	var err error
	t.requestBytesCounter, err = t.meter.Int64Counter(
		clientRequestSize,
		metric.WithUnit("By"),
		metric.WithDescription("Measures the size of HTTP request messages."),
	)
	handleErr(err)

	t.responseBytesCounter, err = t.meter.Int64Counter(
		clientResponseSize,
		metric.WithUnit("By"),
		metric.WithDescription("Measures the size of HTTP response messages."),
	)
	handleErr(err)

	t.latencyMeasure, err = t.meter.Float64Histogram(
		clientDuration,
		metric.WithUnit("ms"),
		metric.WithDescription("Measures the duration of outbound HTTP requests."),
	)
	handleErr(err)
}

func defaultTransportFormatter(_ string, r *http.Request) string {
	return "HTTP " + r.Method
}

// RoundTrip creates a Span and propagates its context via the provided request's headers
// before handing the request to the configured base RoundTripper. The created span will
// end when the response body is closed or when a read from the body returns io.EOF.
func (t *Transport) RoundTrip(r *http.Request) (*http.Response, error) {
	requestStartTime := time.Now()
	for _, f := range t.filters {
		if !f(r) {
			// Simply pass through to the base RoundTripper if a filter rejects the request
			return t.rt.RoundTrip(r)
		}
	}

	tracer := t.tracer

	if tracer == nil {
		if span := trace.SpanFromContext(r.Context()); span.SpanContext().IsValid() {
			tracer = newTracer(span.TracerProvider())
		} else {
			tracer = newTracer(otel.GetTracerProvider())
		}
	}

	opts := append([]trace.SpanStartOption{}, t.spanStartOptions...) // start with the configured options

	ctx, span := tracer.Start(r.Context(), t.spanNameFormatter("", r), opts...)

	if t.clientTrace != nil {
		ctx = httptrace.WithClientTrace(ctx, t.clientTrace(ctx))
	}

	labeler, found := LabelerFromContext(ctx)
	if !found {
		ctx = ContextWithLabeler(ctx, labeler)
	}

	r = r.Clone(ctx) // According to RoundTripper spec, we shouldn't modify the origin request.

	// use a body wrapper to determine the request size
	var bw bodyWrapper
	// if request body is nil or NoBody, we don't want to mutate the body as it
	// will affect the identity of it in an unforeseeable way because we assert
	// ReadCloser fulfills a certain interface and it is indeed nil or NoBody.
	if r.Body != nil && r.Body != http.NoBody {
		bw.ReadCloser = r.Body
		// noop to prevent nil panic. not using this record fun yet.
		bw.record = func(int64) {}
		r.Body = &bw
	}

	span.SetAttributes(semconvutil.HTTPClientRequest(r)...)
	t.propagators.Inject(ctx, propagation.HeaderCarrier(r.Header))

	res, err := t.rt.RoundTrip(r)
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
		span.End()
		return res, err
	}

	// metrics
	metricAttrs := append(labeler.Get(), semconvutil.HTTPClientRequestMetrics(r)...)
	if res.StatusCode > 0 {
		metricAttrs = append(metricAttrs, semconv.HTTPStatusCode(res.StatusCode))
	}
	o := metric.WithAttributeSet(attribute.NewSet(metricAttrs...))
	addOpts := []metric.AddOption{o} // Allocate vararg slice once.
	t.requestBytesCounter.Add(ctx, bw.read.Load(), addOpts...)
	// For handling response bytes we leverage a callback when the client reads the http response
	readRecordFunc := func(n int64) {
		t.responseBytesCounter.Add(ctx, n, addOpts...)
	}

	// traces
	span.SetAttributes(semconvutil.HTTPClientResponse(res)...)
	span.SetStatus(semconvutil.HTTPClientStatus(res.StatusCode))

	res.Body = newWrappedBody(span, readRecordFunc, res.Body)

	// Use floating point division here for higher precision (instead of Millisecond method).
	elapsedTime := float64(time.Since(requestStartTime)) / float64(time.Millisecond)

	t.latencyMeasure.Record(ctx, elapsedTime, o)

	return res, err
}

// newWrappedBody returns a new and appropriately scoped *wrappedBody as an
// io.ReadCloser. If the passed body implements io.Writer, the returned value
// will implement io.ReadWriteCloser.
func newWrappedBody(span trace.Span, record func(n int64), body io.ReadCloser) io.ReadCloser {
	// The successful protocol switch responses will have a body that
	// implement an io.ReadWriteCloser. Ensure this interface type continues
	// to be satisfied if that is the case.
	if _, ok := body.(io.ReadWriteCloser); ok {
		return &wrappedBody{span: span, record: record, body: body}
	}

	// Remove the implementation of the io.ReadWriteCloser and only implement
	// the io.ReadCloser.
	return struct{ io.ReadCloser }{&wrappedBody{span: span, record: record, body: body}}
}

// wrappedBody is the response body type returned by the transport
// instrumentation to complete a span. Errors encountered when using the
// response body are recorded in span tracking the response.
//
// The span tracking the response is ended when this body is closed.
//
// If the response body implements the io.Writer interface (i.e. for
// successful protocol switches), the wrapped body also will.
type wrappedBody struct {
	span     trace.Span
	recorded atomic.Bool
	record   func(n int64)
	body     io.ReadCloser
	read     atomic.Int64
}

var _ io.ReadWriteCloser = &wrappedBody{}

func (wb *wrappedBody) Write(p []byte) (int, error) {
	// This will not panic given the guard in newWrappedBody.
	n, err := wb.body.(io.Writer).Write(p)
	if err != nil {
		wb.span.RecordError(err)
		wb.span.SetStatus(codes.Error, err.Error())
	}
	return n, err
}

func (wb *wrappedBody) Read(b []byte) (int, error) {
	n, err := wb.body.Read(b)
	// Record the number of bytes read
	wb.read.Add(int64(n))

	switch err {
	case nil:
		// nothing to do here but fall through to the return
	case io.EOF:
		wb.recordBytesRead()
		wb.span.End()
	default:
		wb.span.RecordError(err)
		wb.span.SetStatus(codes.Error, err.Error())
	}
	return n, err
}

// recordBytesRead is a function that ensures the number of bytes read is recorded once and only once.
func (wb *wrappedBody) recordBytesRead() {
	// note: it is more performant (and equally correct) to use atomic.Bool over sync.Once here. In the event that
	// two goroutines are racing to call this method, the number of bytes read will no longer increase. Using
	// CompareAndSwap allows later goroutines to return quickly and not block waiting for the race winner to finish
	// calling wb.record(wb.read.Load()).
	if wb.recorded.CompareAndSwap(false, true) {
		// Record the total number of bytes read
		wb.record(wb.read.Load())
	}
}

func (wb *wrappedBody) Close() error {
	wb.recordBytesRead()
	wb.span.End()
	if wb.body != nil {
		return wb.body.Close()
	}
	return nil
}
