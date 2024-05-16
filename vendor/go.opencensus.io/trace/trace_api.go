// Copyright 2020, OpenCensus Authors
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

package trace

import (
	"context"
)

// DefaultTracer is the tracer used when package-level exported functions are invoked.
var DefaultTracer Tracer = &tracer{}

// Tracer can start spans and access context functions.
type Tracer interface {

	// StartSpan starts a new child span of the current span in the context. If
	// there is no span in the context, creates a new trace and span.
	//
	// Returned context contains the newly created span. You can use it to
	// propagate the returned span in process.
	StartSpan(ctx context.Context, name string, o ...StartOption) (context.Context, *Span)

	// StartSpanWithRemoteParent starts a new child span of the span from the given parent.
	//
	// If the incoming context contains a parent, it ignores. StartSpanWithRemoteParent is
	// preferred for cases where the parent is propagated via an incoming request.
	//
	// Returned context contains the newly created span. You can use it to
	// propagate the returned span in process.
	StartSpanWithRemoteParent(ctx context.Context, name string, parent SpanContext, o ...StartOption) (context.Context, *Span)

	// FromContext returns the Span stored in a context, or nil if there isn't one.
	FromContext(ctx context.Context) *Span

	// NewContext returns a new context with the given Span attached.
	NewContext(parent context.Context, s *Span) context.Context
}

// StartSpan starts a new child span of the current span in the context. If
// there is no span in the context, creates a new trace and span.
//
// Returned context contains the newly created span. You can use it to
// propagate the returned span in process.
func StartSpan(ctx context.Context, name string, o ...StartOption) (context.Context, *Span) {
	return DefaultTracer.StartSpan(ctx, name, o...)
}

// StartSpanWithRemoteParent starts a new child span of the span from the given parent.
//
// If the incoming context contains a parent, it ignores. StartSpanWithRemoteParent is
// preferred for cases where the parent is propagated via an incoming request.
//
// Returned context contains the newly created span. You can use it to
// propagate the returned span in process.
func StartSpanWithRemoteParent(ctx context.Context, name string, parent SpanContext, o ...StartOption) (context.Context, *Span) {
	return DefaultTracer.StartSpanWithRemoteParent(ctx, name, parent, o...)
}

// FromContext returns the Span stored in a context, or a Span that is not
// recording events if there isn't one.
func FromContext(ctx context.Context) *Span {
	return DefaultTracer.FromContext(ctx)
}

// NewContext returns a new context with the given Span attached.
func NewContext(parent context.Context, s *Span) context.Context {
	return DefaultTracer.NewContext(parent, s)
}

// SpanInterface represents a span of a trace.  It has an associated SpanContext, and
// stores data accumulated while the span is active.
//
// Ideally users should interact with Spans by calling the functions in this
// package that take a Context parameter.
type SpanInterface interface {

	// IsRecordingEvents returns true if events are being recorded for this span.
	// Use this check to avoid computing expensive annotations when they will never
	// be used.
	IsRecordingEvents() bool

	// End ends the span.
	End()

	// SpanContext returns the SpanContext of the span.
	SpanContext() SpanContext

	// SetName sets the name of the span, if it is recording events.
	SetName(name string)

	// SetStatus sets the status of the span, if it is recording events.
	SetStatus(status Status)

	// AddAttributes sets attributes in the span.
	//
	// Existing attributes whose keys appear in the attributes parameter are overwritten.
	AddAttributes(attributes ...Attribute)

	// Annotate adds an annotation with attributes.
	// Attributes can be nil.
	Annotate(attributes []Attribute, str string)

	// Annotatef adds an annotation with attributes.
	Annotatef(attributes []Attribute, format string, a ...interface{})

	// AddMessageSendEvent adds a message send event to the span.
	//
	// messageID is an identifier for the message, which is recommended to be
	// unique in this span and the same between the send event and the receive
	// event (this allows to identify a message between the sender and receiver).
	// For example, this could be a sequence id.
	AddMessageSendEvent(messageID, uncompressedByteSize, compressedByteSize int64)

	// AddMessageReceiveEvent adds a message receive event to the span.
	//
	// messageID is an identifier for the message, which is recommended to be
	// unique in this span and the same between the send event and the receive
	// event (this allows to identify a message between the sender and receiver).
	// For example, this could be a sequence id.
	AddMessageReceiveEvent(messageID, uncompressedByteSize, compressedByteSize int64)

	// AddLink adds a link to the span.
	AddLink(l Link)

	// String prints a string representation of a span.
	String() string
}

// NewSpan is a convenience function for creating a *Span out of a *span
func NewSpan(s SpanInterface) *Span {
	return &Span{internal: s}
}

// Span is a struct wrapper around the SpanInt interface, which allows correctly handling
// nil spans, while also allowing the SpanInterface implementation to be swapped out.
type Span struct {
	internal SpanInterface
}

// Internal returns the underlying implementation of the Span
func (s *Span) Internal() SpanInterface {
	return s.internal
}

// IsRecordingEvents returns true if events are being recorded for this span.
// Use this check to avoid computing expensive annotations when they will never
// be used.
func (s *Span) IsRecordingEvents() bool {
	if s == nil {
		return false
	}
	return s.internal.IsRecordingEvents()
}

// End ends the span.
func (s *Span) End() {
	if s == nil {
		return
	}
	s.internal.End()
}

// SpanContext returns the SpanContext of the span.
func (s *Span) SpanContext() SpanContext {
	if s == nil {
		return SpanContext{}
	}
	return s.internal.SpanContext()
}

// SetName sets the name of the span, if it is recording events.
func (s *Span) SetName(name string) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.SetName(name)
}

// SetStatus sets the status of the span, if it is recording events.
func (s *Span) SetStatus(status Status) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.SetStatus(status)
}

// AddAttributes sets attributes in the span.
//
// Existing attributes whose keys appear in the attributes parameter are overwritten.
func (s *Span) AddAttributes(attributes ...Attribute) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.AddAttributes(attributes...)
}

// Annotate adds an annotation with attributes.
// Attributes can be nil.
func (s *Span) Annotate(attributes []Attribute, str string) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.Annotate(attributes, str)
}

// Annotatef adds an annotation with attributes.
func (s *Span) Annotatef(attributes []Attribute, format string, a ...interface{}) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.Annotatef(attributes, format, a...)
}

// AddMessageSendEvent adds a message send event to the span.
//
// messageID is an identifier for the message, which is recommended to be
// unique in this span and the same between the send event and the receive
// event (this allows to identify a message between the sender and receiver).
// For example, this could be a sequence id.
func (s *Span) AddMessageSendEvent(messageID, uncompressedByteSize, compressedByteSize int64) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.AddMessageSendEvent(messageID, uncompressedByteSize, compressedByteSize)
}

// AddMessageReceiveEvent adds a message receive event to the span.
//
// messageID is an identifier for the message, which is recommended to be
// unique in this span and the same between the send event and the receive
// event (this allows to identify a message between the sender and receiver).
// For example, this could be a sequence id.
func (s *Span) AddMessageReceiveEvent(messageID, uncompressedByteSize, compressedByteSize int64) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.AddMessageReceiveEvent(messageID, uncompressedByteSize, compressedByteSize)
}

// AddLink adds a link to the span.
func (s *Span) AddLink(l Link) {
	if !s.IsRecordingEvents() {
		return
	}
	s.internal.AddLink(l)
}

// String prints a string representation of a span.
func (s *Span) String() string {
	if s == nil {
		return "<nil>"
	}
	return s.internal.String()
}
