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
	"net/http"
	"net/http/httptrace"

	"go.opencensus.io/trace"
	"go.opencensus.io/trace/propagation"
)

// Transport is an http.RoundTripper that instruments all outgoing requests with
// OpenCensus stats and tracing.
//
// The zero value is intended to be a useful default, but for
// now it's recommended that you explicitly set Propagation, since the default
// for this may change.
type Transport struct {
	// Base may be set to wrap another http.RoundTripper that does the actual
	// requests. By default http.DefaultTransport is used.
	//
	// If base HTTP roundtripper implements CancelRequest,
	// the returned round tripper will be cancelable.
	Base http.RoundTripper

	// Propagation defines how traces are propagated. If unspecified, a default
	// (currently B3 format) will be used.
	Propagation propagation.HTTPFormat

	// StartOptions are applied to the span started by this Transport around each
	// request.
	//
	// StartOptions.SpanKind will always be set to trace.SpanKindClient
	// for spans started by this transport.
	StartOptions trace.StartOptions

	// GetStartOptions allows to set start options per request. If set,
	// StartOptions is going to be ignored.
	GetStartOptions func(*http.Request) trace.StartOptions

	// NameFromRequest holds the function to use for generating the span name
	// from the information found in the outgoing HTTP Request. By default the
	// name equals the URL Path.
	FormatSpanName func(*http.Request) string

	// NewClientTrace may be set to a function allowing the current *trace.Span
	// to be annotated with HTTP request event information emitted by the
	// httptrace package.
	NewClientTrace func(*http.Request, *trace.Span) *httptrace.ClientTrace

	// TODO: Implement tag propagation for HTTP.
}

// RoundTrip implements http.RoundTripper, delegating to Base and recording stats and traces for the request.
func (t *Transport) RoundTrip(req *http.Request) (*http.Response, error) {
	rt := t.base()
	if isHealthEndpoint(req.URL.Path) {
		return rt.RoundTrip(req)
	}
	// TODO: remove excessive nesting of http.RoundTrippers here.
	format := t.Propagation
	if format == nil {
		format = defaultFormat
	}
	spanNameFormatter := t.FormatSpanName
	if spanNameFormatter == nil {
		spanNameFormatter = spanNameFromURL
	}

	startOpts := t.StartOptions
	if t.GetStartOptions != nil {
		startOpts = t.GetStartOptions(req)
	}

	rt = &traceTransport{
		base:   rt,
		format: format,
		startOptions: trace.StartOptions{
			Sampler:  startOpts.Sampler,
			SpanKind: trace.SpanKindClient,
		},
		formatSpanName: spanNameFormatter,
		newClientTrace: t.NewClientTrace,
	}
	rt = statsTransport{base: rt}
	return rt.RoundTrip(req)
}

func (t *Transport) base() http.RoundTripper {
	if t.Base != nil {
		return t.Base
	}
	return http.DefaultTransport
}

// CancelRequest cancels an in-flight request by closing its connection.
func (t *Transport) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := t.base().(canceler); ok {
		cr.CancelRequest(req)
	}
}
