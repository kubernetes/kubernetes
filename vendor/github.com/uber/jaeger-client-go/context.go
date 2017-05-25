// Copyright (c) 2016 Uber Technologies, Inc.

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
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	flagSampled = byte(1)
	flagDebug   = byte(2)
)

var (
	errEmptyTracerStateString     = errors.New("Cannot convert empty string to tracer state")
	errMalformedTracerStateString = errors.New("String does not match tracer state format")

	emptyContext = SpanContext{}
)

// TraceID represents unique 128bit identifier of a trace
type TraceID struct {
	High, Low uint64
}

// SpanID represents unique 64bit identifier of a span
type SpanID uint64

// SpanContext represents propagated span identity and state
type SpanContext struct {
	// traceID represents globally unique ID of the trace.
	// Usually generated as a random number.
	traceID TraceID

	// spanID represents span ID that must be unique within its trace,
	// but does not have to be globally unique.
	spanID SpanID

	// parentID refers to the ID of the parent span.
	// Should be 0 if the current span is a root span.
	parentID SpanID

	// flags is a bitmap containing such bits as 'sampled' and 'debug'.
	flags byte

	// Distributed Context baggage. The is a snapshot in time.
	baggage map[string]string

	// debugID can be set to some correlation ID when the context is being
	// extracted from a TextMap carrier.
	//
	// See JaegerDebugHeader in constants.go
	debugID string
}

// ForeachBaggageItem implements ForeachBaggageItem() of opentracing.SpanContext
func (c SpanContext) ForeachBaggageItem(handler func(k, v string) bool) {
	for k, v := range c.baggage {
		if !handler(k, v) {
			break
		}
	}
}

// IsSampled returns whether this trace was chosen for permanent storage
// by the sampling mechanism of the tracer.
func (c SpanContext) IsSampled() bool {
	return (c.flags & flagSampled) == flagSampled
}

// IsDebug indicates whether sampling was explicitly requested by the service.
func (c SpanContext) IsDebug() bool {
	return (c.flags & flagDebug) == flagDebug
}

// IsValid indicates whether this context actually represents a valid trace.
func (c SpanContext) IsValid() bool {
	return c.traceID.IsValid() && c.spanID != 0
}

func (c SpanContext) String() string {
	if c.traceID.High == 0 {
		return fmt.Sprintf("%x:%x:%x:%x", c.traceID.Low, uint64(c.spanID), uint64(c.parentID), c.flags)
	}
	return fmt.Sprintf("%x%016x:%x:%x:%x", c.traceID.High, c.traceID.Low, uint64(c.spanID), uint64(c.parentID), c.flags)
}

// ContextFromString reconstructs the Context encoded in a string
func ContextFromString(value string) (SpanContext, error) {
	var context SpanContext
	if value == "" {
		return emptyContext, errEmptyTracerStateString
	}
	parts := strings.Split(value, ":")
	if len(parts) != 4 {
		return emptyContext, errMalformedTracerStateString
	}
	var err error
	if context.traceID, err = TraceIDFromString(parts[0]); err != nil {
		return emptyContext, err
	}
	if context.spanID, err = SpanIDFromString(parts[1]); err != nil {
		return emptyContext, err
	}
	if context.parentID, err = SpanIDFromString(parts[2]); err != nil {
		return emptyContext, err
	}
	flags, err := strconv.ParseUint(parts[3], 10, 8)
	if err != nil {
		return emptyContext, err
	}
	context.flags = byte(flags)
	return context, nil
}

// TraceID returns the trace ID of this span context
func (c SpanContext) TraceID() TraceID {
	return c.traceID
}

// SpanID returns the span ID of this span context
func (c SpanContext) SpanID() SpanID {
	return c.spanID
}

// ParentID returns the parent span ID of this span context
func (c SpanContext) ParentID() SpanID {
	return c.parentID
}

// NewSpanContext creates a new instance of SpanContext
func NewSpanContext(traceID TraceID, spanID, parentID SpanID, sampled bool, baggage map[string]string) SpanContext {
	flags := byte(0)
	if sampled {
		flags = flagSampled
	}
	return SpanContext{
		traceID:  traceID,
		spanID:   spanID,
		parentID: parentID,
		flags:    flags,
		baggage:  baggage}
}

// CopyFrom copies data from ctx into this context, including span identity and baggage.
// TODO This is only used by interop.go. Remove once TChannel Go supports OpenTracing.
func (c *SpanContext) CopyFrom(ctx *SpanContext) {
	c.traceID = ctx.traceID
	c.spanID = ctx.spanID
	c.parentID = ctx.parentID
	c.flags = ctx.flags
	if l := len(ctx.baggage); l > 0 {
		c.baggage = make(map[string]string, l)
		for k, v := range ctx.baggage {
			c.baggage[k] = v
		}
	} else {
		c.baggage = nil
	}
}

// WithBaggageItem creates a new context with an extra baggage item.
func (c SpanContext) WithBaggageItem(key, value string) SpanContext {
	var newBaggage map[string]string
	if c.baggage == nil {
		newBaggage = map[string]string{key: value}
	} else {
		newBaggage = make(map[string]string, len(c.baggage)+1)
		for k, v := range c.baggage {
			newBaggage[k] = v
		}
		newBaggage[key] = value
	}
	// Use positional parameters so the compiler will help catch new fields.
	return SpanContext{c.traceID, c.spanID, c.parentID, c.flags, newBaggage, ""}
}

// isDebugIDContainerOnly returns true when the instance of the context is only
// used to return the debug/correlation ID from extract() method. This happens
// in the situation when "jaeger-debug-id" header is passed in the carrier to
// the extract() method, but the request otherwise has no span context in it.
// Previously this would've returned opentracing.ErrSpanContextNotFound from the
// extract method, but now it returns a dummy context with only debugID filled in.
//
// See JaegerDebugHeader in constants.go
// See textMapPropagator#Extract
func (c *SpanContext) isDebugIDContainerOnly() bool {
	return !c.traceID.IsValid() && c.debugID != ""
}

// ------- TraceID -------

func (t TraceID) String() string {
	if t.High == 0 {
		return fmt.Sprintf("%x", t.Low)
	}
	return fmt.Sprintf("%x%016x", t.High, t.Low)
}

// TraceIDFromString creates a TraceID from a hexadecimal string
func TraceIDFromString(s string) (TraceID, error) {
	var hi, lo uint64
	var err error
	if len(s) > 32 {
		return TraceID{}, fmt.Errorf("TraceID cannot be longer than 32 hex characters: %s", s)
	} else if len(s) > 16 {
		hiLen := len(s) - 16
		if hi, err = strconv.ParseUint(s[0:hiLen], 16, 64); err != nil {
			return TraceID{}, err
		}
		if lo, err = strconv.ParseUint(s[hiLen:], 16, 64); err != nil {
			return TraceID{}, err
		}
	} else {
		if lo, err = strconv.ParseUint(s, 16, 64); err != nil {
			return TraceID{}, err
		}
	}
	return TraceID{High: hi, Low: lo}, nil
}

// IsValid checks if the trace ID is valid, i.e. not zero.
func (t TraceID) IsValid() bool {
	return t.High != 0 || t.Low != 0
}

// ------- SpanID -------

func (s SpanID) String() string {
	return fmt.Sprintf("%x", uint64(s))
}

// SpanIDFromString creates a SpanID from a hexadecimal string
func SpanIDFromString(s string) (SpanID, error) {
	if len(s) > 16 {
		return SpanID(0), fmt.Errorf("SpanID cannot be longer than 16 hex characters: %s", s)
	}
	id, err := strconv.ParseUint(s, 16, 64)
	if err != nil {
		return SpanID(0), err
	}
	return SpanID(id), nil
}
