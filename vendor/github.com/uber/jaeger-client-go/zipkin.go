// Copyright (c) 2017 Uber Technologies, Inc.
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

// ZipkinSpanFormat is an OpenTracing carrier format constant
const ZipkinSpanFormat = "zipkin-span-format"

// ExtractableZipkinSpan is a type of Carrier used for integration with Zipkin-aware
// RPC frameworks (like TChannel). It does not support baggage, only trace IDs.
type ExtractableZipkinSpan interface {
	TraceID() uint64
	SpanID() uint64
	ParentID() uint64
	Flags() byte
}

// InjectableZipkinSpan is a type of Carrier used for integration with Zipkin-aware
// RPC frameworks (like TChannel). It does not support baggage, only trace IDs.
type InjectableZipkinSpan interface {
	SetTraceID(traceID uint64)
	SetSpanID(spanID uint64)
	SetParentID(parentID uint64)
	SetFlags(flags byte)
}

type zipkinPropagator struct {
	tracer *tracer
}

func (p *zipkinPropagator) Inject(
	ctx SpanContext,
	abstractCarrier interface{},
) error {
	carrier, ok := abstractCarrier.(InjectableZipkinSpan)
	if !ok {
		return opentracing.ErrInvalidCarrier
	}

	carrier.SetTraceID(ctx.TraceID().Low) // TODO this cannot work with 128bit IDs
	carrier.SetSpanID(uint64(ctx.SpanID()))
	carrier.SetParentID(uint64(ctx.ParentID()))
	carrier.SetFlags(ctx.flags)
	return nil
}

func (p *zipkinPropagator) Extract(abstractCarrier interface{}) (SpanContext, error) {
	carrier, ok := abstractCarrier.(ExtractableZipkinSpan)
	if !ok {
		return emptyContext, opentracing.ErrInvalidCarrier
	}
	if carrier.TraceID() == 0 {
		return emptyContext, opentracing.ErrSpanContextNotFound
	}
	var ctx SpanContext
	ctx.traceID.Low = carrier.TraceID()
	ctx.spanID = SpanID(carrier.SpanID())
	ctx.parentID = SpanID(carrier.ParentID())
	ctx.flags = carrier.Flags()
	return ctx, nil
}
