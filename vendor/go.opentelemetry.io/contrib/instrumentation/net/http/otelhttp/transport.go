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

package otelhttp

import (
	"context"
	"io"
	"net/http"

	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/semconv"
	"go.opentelemetry.io/otel/trace"
)

// Transport implements the http.RoundTripper interface and wraps
// outbound HTTP(S) requests with a span.
type Transport struct {
	rt http.RoundTripper

	tracer            trace.Tracer
	propagators       propagation.TextMapPropagator
	spanStartOptions  []trace.SpanOption
	filters           []Filter
	spanNameFormatter func(string, *http.Request) string
}

var _ http.RoundTripper = &Transport{}

// NewTransport wraps the provided http.RoundTripper with one that
// starts a span and injects the span context into the outbound request headers.
//
// If the provided http.RoundTripper is nil, http.DefaultTransport will be used
// as the base http.RoundTripper
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
}

func defaultTransportFormatter(_ string, r *http.Request) string {
	return r.Method
}

// RoundTrip creates a Span and propagates its context via the provided request's headers
// before handing the request to the configured base RoundTripper. The created span will
// end when the response body is closed or when a read from the body returns io.EOF.
func (t *Transport) RoundTrip(r *http.Request) (*http.Response, error) {
	for _, f := range t.filters {
		if !f(r) {
			// Simply pass through to the base RoundTripper if a filter rejects the request
			return t.rt.RoundTrip(r)
		}
	}

	opts := append([]trace.SpanOption{}, t.spanStartOptions...) // start with the configured options

	ctx, span := t.tracer.Start(r.Context(), t.spanNameFormatter("", r), opts...)

	r = r.WithContext(ctx)
	span.SetAttributes(semconv.HTTPClientAttributesFromHTTPRequest(r)...)
	t.propagators.Inject(ctx, propagation.HeaderCarrier(r.Header))

	res, err := t.rt.RoundTrip(r)
	if err != nil {
		span.RecordError(err)
		span.End()
		return res, err
	}

	span.SetAttributes(semconv.HTTPAttributesFromHTTPStatusCode(res.StatusCode)...)
	span.SetStatus(semconv.SpanStatusFromHTTPStatusCode(res.StatusCode))
	res.Body = &wrappedBody{ctx: ctx, span: span, body: res.Body}

	return res, err
}

type wrappedBody struct {
	ctx  context.Context
	span trace.Span
	body io.ReadCloser
}

var _ io.ReadCloser = &wrappedBody{}

func (wb *wrappedBody) Read(b []byte) (int, error) {
	n, err := wb.body.Read(b)

	switch err {
	case nil:
		// nothing to do here but fall through to the return
	case io.EOF:
		wb.span.End()
	default:
		wb.span.RecordError(err)
	}
	return n, err
}

func (wb *wrappedBody) Close() error {
	wb.span.End()
	return wb.body.Close()
}
