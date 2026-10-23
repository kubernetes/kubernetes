// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelhttp

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptrace"
	"sync"
	"sync/atomic"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/propagation"
	otelsemconv "go.opentelemetry.io/otel/semconv/v1.43.0"
	"go.opentelemetry.io/otel/trace"

	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp/internal/semconv"
)

// Transport implements the http.RoundTripper interface and wraps
// outbound HTTP(S) requests with a span and enriches it with metrics.
type Transport struct {
	rt http.RoundTripper

	tracer             trace.Tracer
	propagators        propagation.TextMapPropagator
	spanStartOptions   []trace.SpanStartOption
	filters            []Filter
	spanNameFormatter  func(string, *http.Request) string
	clientTrace        func(context.Context) *httptrace.ClientTrace
	metricAttributesFn func(*http.Request) []attribute.KeyValue

	semconv semconv.HTTPClient
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

	return &t
}

func (t *Transport) applyConfig(c *config) {
	t.tracer = c.Tracer
	t.propagators = c.Propagators
	t.spanStartOptions = c.SpanStartOptions
	t.filters = c.Filters
	t.spanNameFormatter = c.SpanNameFormatter
	t.clientTrace = c.ClientTrace
	t.semconv = semconv.NewHTTPClient(c.Meter)
	t.metricAttributesFn = c.MetricAttributesFn
}

func defaultTransportFormatter(_ string, r *http.Request) string {
	return "HTTP " + r.Method
}

// requestTracker tracks the request body lifecycle of a single RoundTrip and
// records the client metrics exactly once, after the request body reaches a
// final state.
//
// A single tracker is allocated per request with a body, replacing the
// body-wrapper allocation of the untracked implementation: the first request
// body is wrapped by the embedded first field, and the lifecycle state is
// held in explicit fields with methods instead of escaping closures. Only
// GetBody retries allocate an additional trackedRequestBody.
type requestTracker struct {
	transport *Transport
	first     trackedRequestBody

	originalGetBody func() (io.ReadCloser, error)

	// mu protects body, published, and recorded. Body close callbacks may
	// fire on a different goroutine (e.g. during transport retries or
	// asynchronous closes) while the main goroutine publishes the metric
	// data after RoundTrip returns.
	mu        sync.Mutex
	body      *trackedRequestBody // Last wrapped request body; nil records no data.
	published bool
	recorded  bool

	// Metric data published when RoundTrip returns; immutable afterwards.
	ctx      context.Context
	duration time.Duration
	opts     semconv.MetricOpts
}

func newRequestTracker(t *Transport, r *http.Request) *requestTracker {
	rt := &requestTracker{transport: t}
	rt.first = trackedRequestBody{body: r.Body, tracker: rt}
	rt.body = &rt.first
	r.Body = &rt.first
	if r.GetBody != nil {
		rt.originalGetBody = r.GetBody
		r.GetBody = rt.getBody
	}
	return rt
}

func (rt *requestTracker) getBody() (io.ReadCloser, error) {
	b, err := rt.originalGetBody()
	if err != nil {
		// The underlying transport will fail to make a retry request,
		// hence, record no data.
		rt.setBody(nil)
		return nil, err
	}
	if b == nil || b == http.NoBody {
		return b, nil
	}
	tb := &trackedRequestBody{body: b, tracker: rt}
	rt.setBody(tb)
	return tb, nil
}

func (rt *requestTracker) setBody(b *trackedRequestBody) {
	rt.mu.Lock()
	rt.body = b
	rt.mu.Unlock()
}

// publish stores the metric data gathered when RoundTrip returns and records
// the metrics if the request body already reached a final state.
func (rt *requestTracker) publish(ctx context.Context, d time.Duration, opts semconv.MetricOpts) {
	rt.mu.Lock()
	rt.ctx = ctx
	rt.duration = d
	rt.opts = opts
	rt.published = true
	rt.mu.Unlock()
	rt.maybeRecord()
}

// maybeRecord records the client metrics once the metric data has been
// published and the request body has reached a final state. Close is the
// only authoritative completion signal for a non-nil body: a body may yield
// more bytes than its declared ContentLength, so a read count is never proof
// of completion.
func (rt *requestTracker) maybeRecord() {
	rt.mu.Lock()
	body := rt.body
	record := rt.published && !rt.recorded && (body == nil || body.closed.Load())
	if record {
		rt.recorded = true
	}
	rt.mu.Unlock()
	if !record {
		return
	}

	// The winning caller evaluates the request size here so the recorded
	// value reflects the latest byte count.
	var requestSize int64
	if body != nil {
		requestSize = body.read.Load()
	}
	rt.transport.semconv.RecordMetrics(
		rt.ctx,
		semconv.MetricData{
			RequestSize:     requestSize,
			RequestDuration: rt.duration,
		},
		rt.opts,
	)
}

// trackedRequestBody wraps a request body to count read bytes and signal the
// tracker when the body is closed.
type trackedRequestBody struct {
	body    io.ReadCloser
	tracker *requestTracker

	read   atomic.Int64
	closed atomic.Bool
}

var _ io.ReadCloser = (*trackedRequestBody)(nil)

func (b *trackedRequestBody) Read(p []byte) (int, error) {
	n, err := b.body.Read(p)
	b.read.Add(int64(n))
	return n, err
}

func (b *trackedRequestBody) Close() error {
	// Finalize telemetry before delegating to the underlying Close, which
	// may block (mirroring wrappedBody.Close). Otherwise a blocking close
	// could suppress client metrics indefinitely.
	b.closed.Store(true)
	b.tracker.maybeRecord()
	return b.body.Close()
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

	ctx, span := tracer.Start(r.Context(), t.spanNameFormatter("", r), t.spanStartOptions...)

	if t.clientTrace != nil {
		ctx = httptrace.WithClientTrace(ctx, t.clientTrace(ctx))
	}

	labeler, found := ClientLabelerFromContext(ctx)
	if !found {
		ctx = ContextWithClientLabeler(ctx, labeler)
	}

	r = r.Clone(ctx) // According to RoundTripper spec, we shouldn't modify the origin request.

	// Track the request body lifecycle so metrics report the final read
	// size even when the transport keeps reading the body after RoundTrip
	// returns. Bodyless requests allocate no tracker and record
	// synchronously below.
	var tracker *requestTracker
	if r.Body != nil && r.Body != http.NoBody {
		tracker = newRequestTracker(t, r)
	}

	span.SetAttributes(t.semconv.RequestTraceAttrs(r)...)
	t.propagators.Inject(ctx, propagation.HeaderCarrier(r.Header))

	res, err := t.rt.RoundTrip(r)
	if err == nil {
		res, err = ensureResponseBody(t.rt, r, res)
	}

	requestDuration := time.Since(requestStartTime)
	statusCode := 0
	if err == nil {
		statusCode = res.StatusCode
	}
	metricOpts := t.semconv.MetricOptions(semconv.MetricAttributes{
		Req:                  r,
		Resp:                 res,
		StatusCode:           statusCode,
		Err:                  err,
		AdditionalAttributes: append(labeler.Get(), t.metricAttributesFromRequest(r)...),
	})

	if tracker != nil {
		// Delay metric recording until the request body reaches a final
		// state. The transport can continue reading the body after
		// RoundTrip returns, including on errors, since RoundTrippers may
		// close the request body asynchronously.
		tracker.publish(ctx, requestDuration, metricOpts)
	} else {
		t.semconv.RecordMetrics(
			ctx,
			semconv.MetricData{RequestDuration: requestDuration},
			metricOpts,
		)
	}

	if err != nil {
		span.SetAttributes(otelsemconv.ErrorType(err))
		span.SetStatus(codes.Error, err.Error())
		span.End()

		return res, err
	}

	readRecordFunc := func(int64) {}
	res.Body = newWrappedBody(span, readRecordFunc, res.Body)
	// traces
	span.SetAttributes(t.semconv.ResponseTraceAttrs(res)...)
	span.SetStatus(t.semconv.Status(res.StatusCode))

	return res, nil
}

func ensureResponseBody(rt http.RoundTripper, r *http.Request, res *http.Response) (*http.Response, error) {
	switch {
	case res == nil:
		return nil, fmt.Errorf("http: RoundTripper implementation (%T) returned a nil *Response with a nil error", rt)
	case res.Body != nil:
		return res, nil
	case res.ContentLength > 0 && r.Method != http.MethodHead:
		return nil, fmt.Errorf("http: RoundTripper implementation (%T) returned a *Response with content length %d but a nil Body", rt, res.ContentLength)
	default:
		res.Body = http.NoBody
		return res, nil
	}
}

func (t *Transport) metricAttributesFromRequest(r *http.Request) []attribute.KeyValue {
	var attributeForRequest []attribute.KeyValue
	if t.metricAttributesFn != nil {
		attributeForRequest = t.metricAttributesFn(r)
	}
	return attributeForRequest
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
		wb.span.SetAttributes(otelsemconv.ErrorType(err))
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
		wb.recordMetricsOnce()
	default:
		wb.span.SetAttributes(otelsemconv.ErrorType(err))
		wb.span.SetStatus(codes.Error, err.Error())
	}
	return n, err
}

// recordMetricsOnce ensures the final number of bytes read is recorded once.
func (wb *wrappedBody) recordMetricsOnce() {
	// note: it is more performant (and equally correct) to use atomic.Bool
	// over sync.Once here. In the event that two goroutines are racing to
	// call this method, the number of bytes read will no longer increase.
	// Using CompareAndSwap allows later goroutines to return quickly and not
	// block waiting for the race winner to finish calling
	// wb.record(wb.read.Load()).
	if wb.recorded.CompareAndSwap(false, true) {
		wb.record(wb.read.Load())
		wb.span.End()
	}
}

func (wb *wrappedBody) Close() error {
	// Finalize telemetry before delegating to the underlying Close, which
	// may block (e.g. upgraded streams or custom RoundTripper bodies).
	wb.recordMetricsOnce()
	if wb.body != nil {
		return wb.body.Close()
	}
	return nil
}
