// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package jaeger

import (
	"github.com/opentracing/opentracing-go"
)

// TODO this file should not be needed after TChannel PR.

type formatKey int

// SpanContextFormat is a constant used as OpenTracing Format.
// Requires *SpanContext as carrier.
// This format is intended for interop with TChannel or other Zipkin-like tracers.
const SpanContextFormat formatKey = iota

type jaegerTraceContextPropagator struct {
	tracer *tracer
}

func (p *jaegerTraceContextPropagator) Inject(
	ctx SpanContext,
	abstractCarrier interface{},
) error {
	carrier, ok := abstractCarrier.(*SpanContext)
	if !ok {
		return opentracing.ErrInvalidCarrier
	}

	carrier.CopyFrom(&ctx)
	return nil
}

func (p *jaegerTraceContextPropagator) Extract(abstractCarrier interface{}) (SpanContext, error) {
	carrier, ok := abstractCarrier.(*SpanContext)
	if !ok {
		return emptyContext, opentracing.ErrInvalidCarrier
	}
	ctx := new(SpanContext)
	ctx.CopyFrom(carrier)
	return *ctx, nil
}
