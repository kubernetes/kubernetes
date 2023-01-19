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

package otelhttp // import "go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"

import (
	"io"
	"net/http"
	"time"

	"github.com/felixge/httpsnoop"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/instrument/syncfloat64"
	"go.opentelemetry.io/otel/metric/instrument/syncint64"
	"go.opentelemetry.io/otel/propagation"
	semconv "go.opentelemetry.io/otel/semconv/v1.12.0"
	"go.opentelemetry.io/otel/trace"
)

var _ http.Handler = &Handler{}

// Handler is http middleware that corresponds to the http.Handler interface and
// is designed to wrap a http.Mux (or equivalent), while individual routes on
// the mux are wrapped with WithRouteTag. A Handler will add various attributes
// to the span using the attribute.Keys defined in this package.
type Handler struct {
	operation string
	handler   http.Handler

	tracer            trace.Tracer
	meter             metric.Meter
	propagators       propagation.TextMapPropagator
	spanStartOptions  []trace.SpanStartOption
	readEvent         bool
	writeEvent        bool
	filters           []Filter
	spanNameFormatter func(string, *http.Request) string
	counters          map[string]syncint64.Counter
	valueRecorders    map[string]syncfloat64.Histogram
	publicEndpoint    bool
	publicEndpointFn  func(*http.Request) bool
}

func defaultHandlerFormatter(operation string, _ *http.Request) string {
	return operation
}

// NewHandler wraps the passed handler, functioning like middleware, in a span
// named after the operation and with any provided Options.
func NewHandler(handler http.Handler, operation string, opts ...Option) http.Handler {
	h := Handler{
		handler:   handler,
		operation: operation,
	}

	defaultOpts := []Option{
		WithSpanOptions(trace.WithSpanKind(trace.SpanKindServer)),
		WithSpanNameFormatter(defaultHandlerFormatter),
	}

	c := newConfig(append(defaultOpts, opts...)...)
	h.configure(c)
	h.createMeasures()

	return &h
}

func (h *Handler) configure(c *config) {
	h.tracer = c.Tracer
	h.meter = c.Meter
	h.propagators = c.Propagators
	h.spanStartOptions = c.SpanStartOptions
	h.readEvent = c.ReadEvent
	h.writeEvent = c.WriteEvent
	h.filters = c.Filters
	h.spanNameFormatter = c.SpanNameFormatter
	h.publicEndpoint = c.PublicEndpoint
	h.publicEndpointFn = c.PublicEndpointFn
}

func handleErr(err error) {
	if err != nil {
		otel.Handle(err)
	}
}

func (h *Handler) createMeasures() {
	h.counters = make(map[string]syncint64.Counter)
	h.valueRecorders = make(map[string]syncfloat64.Histogram)

	requestBytesCounter, err := h.meter.SyncInt64().Counter(RequestContentLength)
	handleErr(err)

	responseBytesCounter, err := h.meter.SyncInt64().Counter(ResponseContentLength)
	handleErr(err)

	serverLatencyMeasure, err := h.meter.SyncFloat64().Histogram(ServerLatency)
	handleErr(err)

	h.counters[RequestContentLength] = requestBytesCounter
	h.counters[ResponseContentLength] = responseBytesCounter
	h.valueRecorders[ServerLatency] = serverLatencyMeasure
}

// ServeHTTP serves HTTP requests (http.Handler).
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	requestStartTime := time.Now()
	for _, f := range h.filters {
		if !f(r) {
			// Simply pass through to the handler if a filter rejects the request
			h.handler.ServeHTTP(w, r)
			return
		}
	}

	ctx := h.propagators.Extract(r.Context(), propagation.HeaderCarrier(r.Header))
	opts := h.spanStartOptions
	if h.publicEndpoint || (h.publicEndpointFn != nil && h.publicEndpointFn(r.WithContext(ctx))) {
		opts = append(opts, trace.WithNewRoot())
		// Linking incoming span context if any for public endpoint.
		if s := trace.SpanContextFromContext(ctx); s.IsValid() && s.IsRemote() {
			opts = append(opts, trace.WithLinks(trace.Link{SpanContext: s}))
		}
	}

	opts = append([]trace.SpanStartOption{
		trace.WithAttributes(semconv.NetAttributesFromHTTPRequest("tcp", r)...),
		trace.WithAttributes(semconv.EndUserAttributesFromHTTPRequest(r)...),
		trace.WithAttributes(semconv.HTTPServerAttributesFromHTTPRequest(h.operation, "", r)...),
	}, opts...) // start with the configured options

	tracer := h.tracer

	if tracer == nil {
		if span := trace.SpanFromContext(r.Context()); span.SpanContext().IsValid() {
			tracer = newTracer(span.TracerProvider())
		} else {
			tracer = newTracer(otel.GetTracerProvider())
		}
	}

	ctx, span := tracer.Start(ctx, h.spanNameFormatter(h.operation, r), opts...)
	defer span.End()

	readRecordFunc := func(int64) {}
	if h.readEvent {
		readRecordFunc = func(n int64) {
			span.AddEvent("read", trace.WithAttributes(ReadBytesKey.Int64(n)))
		}
	}

	var bw bodyWrapper
	// if request body is nil or NoBody, we don't want to mutate the body as it
	// will affect the identity of it in an unforeseeable way because we assert
	// ReadCloser fulfills a certain interface and it is indeed nil or NoBody.
	if r.Body != nil && r.Body != http.NoBody {
		bw.ReadCloser = r.Body
		bw.record = readRecordFunc
		r.Body = &bw
	}

	writeRecordFunc := func(int64) {}
	if h.writeEvent {
		writeRecordFunc = func(n int64) {
			span.AddEvent("write", trace.WithAttributes(WroteBytesKey.Int64(n)))
		}
	}

	rww := &respWriterWrapper{
		ResponseWriter: w,
		record:         writeRecordFunc,
		ctx:            ctx,
		props:          h.propagators,
		statusCode:     200, // default status code in case the Handler doesn't write anything
	}

	// Wrap w to use our ResponseWriter methods while also exposing
	// other interfaces that w may implement (http.CloseNotifier,
	// http.Flusher, http.Hijacker, http.Pusher, io.ReaderFrom).

	w = httpsnoop.Wrap(w, httpsnoop.Hooks{
		Header: func(httpsnoop.HeaderFunc) httpsnoop.HeaderFunc {
			return rww.Header
		},
		Write: func(httpsnoop.WriteFunc) httpsnoop.WriteFunc {
			return rww.Write
		},
		WriteHeader: func(httpsnoop.WriteHeaderFunc) httpsnoop.WriteHeaderFunc {
			return rww.WriteHeader
		},
	})

	labeler := &Labeler{}
	ctx = injectLabeler(ctx, labeler)

	h.handler.ServeHTTP(w, r.WithContext(ctx))

	setAfterServeAttributes(span, bw.read, rww.written, rww.statusCode, bw.err, rww.err)

	// Add metrics
	attributes := append(labeler.Get(), semconv.HTTPServerMetricAttributesFromHTTPRequest(h.operation, r)...)
	h.counters[RequestContentLength].Add(ctx, bw.read, attributes...)
	h.counters[ResponseContentLength].Add(ctx, rww.written, attributes...)

	// Use floating point division here for higher precision (instead of Millisecond method).
	elapsedTime := float64(time.Since(requestStartTime)) / float64(time.Millisecond)

	h.valueRecorders[ServerLatency].Record(ctx, elapsedTime, attributes...)
}

func setAfterServeAttributes(span trace.Span, read, wrote int64, statusCode int, rerr, werr error) {
	attributes := []attribute.KeyValue{}

	// TODO: Consider adding an event after each read and write, possibly as an
	// option (defaulting to off), so as to not create needlessly verbose spans.
	if read > 0 {
		attributes = append(attributes, ReadBytesKey.Int64(read))
	}
	if rerr != nil && rerr != io.EOF {
		attributes = append(attributes, ReadErrorKey.String(rerr.Error()))
	}
	if wrote > 0 {
		attributes = append(attributes, WroteBytesKey.Int64(wrote))
	}
	attributes = append(attributes, semconv.HTTPAttributesFromHTTPStatusCode(statusCode)...)
	span.SetStatus(semconv.SpanStatusFromHTTPStatusCodeAndSpanKind(statusCode, trace.SpanKindServer))

	if werr != nil && werr != io.EOF {
		attributes = append(attributes, WriteErrorKey.String(werr.Error()))
	}
	span.SetAttributes(attributes...)
}

// WithRouteTag annotates a span with the provided route name using the
// RouteKey Tag.
func WithRouteTag(route string, h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		span := trace.SpanFromContext(r.Context())
		span.SetAttributes(semconv.HTTPRouteKey.String(route))
		h.ServeHTTP(w, r)
	})
}
