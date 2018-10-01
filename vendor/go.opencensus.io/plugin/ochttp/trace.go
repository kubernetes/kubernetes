// Copyright 2018, OpenCensus Authors
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

package ochttp

import (
	"io"
	"net/http"
	"net/http/httptrace"

	"go.opencensus.io/plugin/ochttp/propagation/b3"
	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

// TODO(jbd): Add godoc examples.

var defaultFormat propagation.HTTPFormat = &b3.HTTPFormat{}

// Attributes recorded on the span for the requests.
// Only trace exporters will need them.
const (
	HostAttribute       = "http.host"
	MethodAttribute     = "http.method"
	PathAttribute       = "http.path"
	UserAgentAttribute  = "http.user_agent"
	StatusCodeAttribute = "http.status_code"
)

type traceTransport struct {
	base           http.RoundTripper
	startOptions   trace.StartOptions
	format         propagation.HTTPFormat
	formatSpanName func(*http.Request) string
	newClientTrace func(*http.Request, *trace.Span) *httptrace.ClientTrace
}

// TODO(jbd): Add message events for request and response size.

// RoundTrip creates a trace.Span and inserts it into the outgoing request's headers.
// The created span can follow a parent span, if a parent is presented in
// the request's context.
func (t *traceTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	name := t.formatSpanName(req)
	// TODO(jbd): Discuss whether we want to prefix
	// outgoing requests with Sent.
	ctx, span := trace.StartSpan(req.Context(), name,
		trace.WithSampler(t.startOptions.Sampler),
		trace.WithSpanKind(trace.SpanKindClient))

	if t.newClientTrace != nil {
		req = req.WithContext(httptrace.WithClientTrace(ctx, t.newClientTrace(req, span)))
	} else {
		req = req.WithContext(ctx)
	}

	if t.format != nil {
		// SpanContextToRequest will modify its Request argument, which is
		// contrary to the contract for http.RoundTripper, so we need to
		// pass it a copy of the Request.
		// However, the Request struct itself was already copied by
		// the WithContext calls above and so we just need to copy the header.
		header := make(http.Header)
		for k, v := range req.Header {
			header[k] = v
		}
		req.Header = header
		t.format.SpanContextToRequest(span.SpanContext(), req)
	}

	span.AddAttributes(requestAttrs(req)...)
	resp, err := t.base.RoundTrip(req)
	if err != nil {
		span.SetStatus(trace.Status{Code: trace.StatusCodeUnknown, Message: err.Error()})
		span.End()
		return resp, err
	}

	span.AddAttributes(responseAttrs(resp)...)
	span.SetStatus(TraceStatus(resp.StatusCode, resp.Status))

	// span.End() will be invoked after
	// a read from resp.Body returns io.EOF or when
	// resp.Body.Close() is invoked.
	resp.Body = &bodyTracker{rc: resp.Body, span: span}
	return resp, err
}

// bodyTracker wraps a response.Body and invokes
// trace.EndSpan on encountering io.EOF on reading
// the body of the original response.
type bodyTracker struct {
	rc   io.ReadCloser
	span *trace.Span
}

var _ io.ReadCloser = (*bodyTracker)(nil)

func (bt *bodyTracker) Read(b []byte) (int, error) {
	n, err := bt.rc.Read(b)

	switch err {
	case nil:
		return n, nil
	case io.EOF:
		bt.span.End()
	default:
		// For all other errors, set the span status
		bt.span.SetStatus(trace.Status{
			// Code 2 is the error code for Internal server error.
			Code:    2,
			Message: err.Error(),
		})
	}
	return n, err
}

func (bt *bodyTracker) Close() error {
	// Invoking endSpan on Close will help catch the cases
	// in which a read returned a non-nil error, we set the
	// span status but didn't end the span.
	bt.span.End()
	return bt.rc.Close()
}

// CancelRequest cancels an in-flight request by closing its connection.
func (t *traceTransport) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := t.base.(canceler); ok {
		cr.CancelRequest(req)
	}
}

func spanNameFromURL(req *http.Request) string {
	return req.URL.Path
}

func requestAttrs(r *http.Request) []trace.Attribute {
	return []trace.Attribute{
		trace.StringAttribute(PathAttribute, r.URL.Path),
		trace.StringAttribute(HostAttribute, r.URL.Host),
		trace.StringAttribute(MethodAttribute, r.Method),
		trace.StringAttribute(UserAgentAttribute, r.UserAgent()),
	}
}

func responseAttrs(resp *http.Response) []trace.Attribute {
	return []trace.Attribute{
		trace.Int64Attribute(StatusCodeAttribute, int64(resp.StatusCode)),
	}
}

// TraceStatus is a utility to convert the HTTP status code to a trace.Status that
// represents the outcome as closely as possible.
func TraceStatus(httpStatusCode int, statusLine string) trace.Status {
	var code int32
	if httpStatusCode < 200 || httpStatusCode >= 400 {
		code = trace.StatusCodeUnknown
	}
	switch httpStatusCode {
	case 499:
		code = trace.StatusCodeCancelled
	case http.StatusBadRequest:
		code = trace.StatusCodeInvalidArgument
	case http.StatusGatewayTimeout:
		code = trace.StatusCodeDeadlineExceeded
	case http.StatusNotFound:
		code = trace.StatusCodeNotFound
	case http.StatusForbidden:
		code = trace.StatusCodePermissionDenied
	case http.StatusUnauthorized: // 401 is actually unauthenticated.
		code = trace.StatusCodeUnauthenticated
	case http.StatusTooManyRequests:
		code = trace.StatusCodeResourceExhausted
	case http.StatusNotImplemented:
		code = trace.StatusCodeUnimplemented
	case http.StatusServiceUnavailable:
		code = trace.StatusCodeUnavailable
	case http.StatusOK:
		code = trace.StatusCodeOK
	}
	return trace.Status{Code: code, Message: codeToStr[code]}
}

var codeToStr = map[int32]string{
	trace.StatusCodeOK:                 `OK`,
	trace.StatusCodeCancelled:          `CANCELLED`,
	trace.StatusCodeUnknown:            `UNKNOWN`,
	trace.StatusCodeInvalidArgument:    `INVALID_ARGUMENT`,
	trace.StatusCodeDeadlineExceeded:   `DEADLINE_EXCEEDED`,
	trace.StatusCodeNotFound:           `NOT_FOUND`,
	trace.StatusCodeAlreadyExists:      `ALREADY_EXISTS`,
	trace.StatusCodePermissionDenied:   `PERMISSION_DENIED`,
	trace.StatusCodeResourceExhausted:  `RESOURCE_EXHAUSTED`,
	trace.StatusCodeFailedPrecondition: `FAILED_PRECONDITION`,
	trace.StatusCodeAborted:            `ABORTED`,
	trace.StatusCodeOutOfRange:         `OUT_OF_RANGE`,
	trace.StatusCodeUnimplemented:      `UNIMPLEMENTED`,
	trace.StatusCodeInternal:           `INTERNAL`,
	trace.StatusCodeUnavailable:        `UNAVAILABLE`,
	trace.StatusCodeDataLoss:           `DATA_LOSS`,
	trace.StatusCodeUnauthenticated:    `UNAUTHENTICATED`,
}

func isHealthEndpoint(path string) bool {
	// Health checking is pretty frequent and
	// traces collected for health endpoints
	// can be extremely noisy and expensive.
	// Disable canonical health checking endpoints
	// like /healthz and /_ah/health for now.
	if path == "/healthz" || path == "/_ah/health" {
		return true
	}
	return false
}
