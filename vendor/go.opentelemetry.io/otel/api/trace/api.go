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

package trace

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel/api/core"
)

type Provider interface {
	// Tracer creates a named tracer that implements Tracer interface.
	// If the name is an empty string then provider uses default name.
	Tracer(name string) Tracer
}

type Tracer interface {
	// Start a span.
	Start(ctx context.Context, spanName string, opts ...StartOption) (context.Context, Span)

	// WithSpan wraps the execution of the fn function with a span.
	// It starts a new span, sets it as an active span in the context,
	// executes the fn function and closes the span before returning the result of fn.
	WithSpan(
		ctx context.Context,
		spanName string,
		fn func(ctx context.Context) error,
		opts ...StartOption,
	) error
}

// EndConfig provides options to set properties of span at the time of ending
// the span.
type EndConfig struct {
	EndTime time.Time
}

// EndOption applies changes to EndConfig that sets options when the span is ended.
type EndOption func(*EndConfig)

// WithEndTime sets the end time of the span to provided time t, when it is ended.
func WithEndTime(t time.Time) EndOption {
	return func(c *EndConfig) {
		c.EndTime = t
	}
}

// ErrorConfig provides options to set properties of an error event at the time it is recorded.
type ErrorConfig struct {
	Timestamp  time.Time
	StatusCode codes.Code
}

// ErrorOption applies changes to ErrorConfig that sets options when an error event is recorded.
type ErrorOption func(*ErrorConfig)

// WithErrorTime sets the time at which the error event should be recorded.
func WithErrorTime(t time.Time) ErrorOption {
	return func(c *ErrorConfig) {
		c.Timestamp = t
	}
}

// WithErrorStatus indicates the span status that should be set when recording an error event.
func WithErrorStatus(s codes.Code) ErrorOption {
	return func(c *ErrorConfig) {
		c.StatusCode = s
	}
}

type Span interface {
	// Tracer returns tracer used to create this span. Tracer cannot be nil.
	Tracer() Tracer

	// End completes the span. No updates are allowed to span after it
	// ends. The only exception is setting status of the span.
	End(options ...EndOption)

	// AddEvent adds an event to the span.
	AddEvent(ctx context.Context, name string, attrs ...core.KeyValue)
	// AddEventWithTimestamp adds an event with a custom timestamp
	// to the span.
	AddEventWithTimestamp(ctx context.Context, timestamp time.Time, name string, attrs ...core.KeyValue)

	// IsRecording returns true if the span is active and recording events is enabled.
	IsRecording() bool

	// RecordError records an error as a span event.
	RecordError(ctx context.Context, err error, opts ...ErrorOption)

	// SpanContext returns span context of the span. Returned SpanContext is usable
	// even after the span ends.
	SpanContext() core.SpanContext

	// SetStatus sets the status of the span in the form of a code
	// and a message.  SetStatus overrides the value of previous
	// calls to SetStatus on the Span.
	//
	// The default span status is OK, so it is not necessary to
	// explicitly set an OK status on successful Spans unless it
	// is to add an OK message or to override a previous status on the Span.
	SetStatus(codes.Code, string)

	// SetName sets the name of the span.
	SetName(name string)

	// Set span attributes
	SetAttributes(...core.KeyValue)
}

// StartOption applies changes to StartConfig that sets options at span start time.
type StartOption func(*StartConfig)

// StartConfig provides options to set properties of span at the time of starting
// a new span.
type StartConfig struct {
	Attributes []core.KeyValue
	StartTime  time.Time
	Links      []Link
	Record     bool
	NewRoot    bool
	SpanKind   SpanKind
}

// Link is used to establish relationship between two spans within the same Trace or
// across different Traces. Few examples of Link usage.
//   1. Batch Processing: A batch of elements may contain elements associated with one
//      or more traces/spans. Since there can only be one parent SpanContext, Link is
//      used to keep reference to SpanContext of all elements in the batch.
//   2. Public Endpoint: A SpanContext in incoming client request on a public endpoint
//      is untrusted from service provider perspective. In such case it is advisable to
//      start a new trace with appropriate sampling decision.
//      However, it is desirable to associate incoming SpanContext to new trace initiated
//      on service provider side so two traces (from Client and from Service Provider) can
//      be correlated.
type Link struct {
	core.SpanContext
	Attributes []core.KeyValue
}

// SpanKind represents the role of a Span inside a Trace. Often, this defines how a Span
// will be processed and visualized by various backends.
type SpanKind int

const (
	// As a convenience, these match the proto definition, see
	// opentelemetry/proto/trace/v1/trace.proto
	//
	// The unspecified value is not a valid `SpanKind`.  Use
	// `ValidateSpanKind()` to coerce a span kind to a valid
	// value.
	SpanKindUnspecified SpanKind = 0
	SpanKindInternal    SpanKind = 1
	SpanKindServer      SpanKind = 2
	SpanKindClient      SpanKind = 3
	SpanKindProducer    SpanKind = 4
	SpanKindConsumer    SpanKind = 5
)

// ValidateSpanKind returns a valid span kind value.  This will coerce
// invalid values into the default value, SpanKindInternal.
func ValidateSpanKind(spanKind SpanKind) SpanKind {
	switch spanKind {
	case SpanKindInternal,
		SpanKindServer,
		SpanKindClient,
		SpanKindProducer,
		SpanKindConsumer:
		// valid
		return spanKind
	default:
		return SpanKindInternal
	}
}

// String returns the specified name of the SpanKind in lower-case.
func (sk SpanKind) String() string {
	switch sk {
	case SpanKindInternal:
		return "internal"
	case SpanKindServer:
		return "server"
	case SpanKindClient:
		return "client"
	case SpanKindProducer:
		return "producer"
	case SpanKindConsumer:
		return "consumer"
	default:
		return "unspecified"
	}
}

// WithStartTime sets the start time of the span to provided time t, when it is started.
// In absence of this option, wall clock time is used as start time.
// This option is typically used when starting of the span is delayed.
func WithStartTime(t time.Time) StartOption {
	return func(c *StartConfig) {
		c.StartTime = t
	}
}

// WithAttributes sets attributes to span. These attributes provides additional
// data about the span.
// Multiple `WithAttributes` options appends the attributes preserving the order.
func WithAttributes(attrs ...core.KeyValue) StartOption {
	return func(c *StartConfig) {
		c.Attributes = append(c.Attributes, attrs...)
	}
}

// WithRecord specifies that the span should be recorded.
// Note that the implementation may still override this preference,
// e.g., if the span is a child in an unsampled trace.
func WithRecord() StartOption {
	return func(c *StartConfig) {
		c.Record = true
	}
}

// WithNewRoot specifies that the current span or remote span context
// in context passed to `Start` should be ignored when deciding about
// a parent, which effectively means creating a span with new trace
// ID. The current span and the remote span context may be added as
// links to the span by the implementation.
func WithNewRoot() StartOption {
	return func(c *StartConfig) {
		c.NewRoot = true
	}
}

// LinkedTo allows instantiating a Span with initial Links.
func LinkedTo(sc core.SpanContext, attrs ...core.KeyValue) StartOption {
	return func(c *StartConfig) {
		c.Links = append(c.Links, Link{sc, attrs})
	}
}

// WithSpanKind specifies the role a Span on a Trace.
func WithSpanKind(sk SpanKind) StartOption {
	return func(c *StartConfig) {
		c.SpanKind = sk
	}
}
