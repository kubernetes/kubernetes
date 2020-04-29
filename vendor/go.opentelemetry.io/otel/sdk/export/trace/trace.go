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

package trace // import "go.opentelemetry.io/otel/sdk/export/trace"

import (
	"context"
	"time"

	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel/api/core"
	apitrace "go.opentelemetry.io/otel/api/trace"
	"go.opentelemetry.io/otel/sdk/resource"
)

// SpanSyncer is a type for functions that receive a single sampled trace span.
//
// The ExportSpan method is called synchronously. Therefore, it should not take
// forever to process the span.
//
// The SpanData should not be modified.
type SpanSyncer interface {
	ExportSpan(context.Context, *SpanData)
}

// SpanBatcher is a type for functions that receive batched of sampled trace
// spans.
//
// The ExportSpans method is called asynchronously. However its should not take
// forever to process the spans.
//
// The SpanData should not be modified.
type SpanBatcher interface {
	ExportSpans(context.Context, []*SpanData)
}

// SpanData contains all the information collected by a span.
type SpanData struct {
	SpanContext  core.SpanContext
	ParentSpanID core.SpanID
	SpanKind     apitrace.SpanKind
	Name         string
	StartTime    time.Time
	// The wall clock time of EndTime will be adjusted to always be offset
	// from StartTime by the duration of the span.
	EndTime                  time.Time
	Attributes               []core.KeyValue
	MessageEvents            []Event
	Links                    []apitrace.Link
	StatusCode               codes.Code
	StatusMessage            string
	HasRemoteParent          bool
	DroppedAttributeCount    int
	DroppedMessageEventCount int
	DroppedLinkCount         int

	// ChildSpanCount holds the number of child span created for this span.
	ChildSpanCount int

	// Resource contains attributes representing an entity that produced this span.
	Resource *resource.Resource
}

// Event is used to describe an Event with a message string and set of
// Attributes.
type Event struct {
	// Name is the name of this event
	Name string

	// Attributes contains a list of keyvalue pairs.
	Attributes []core.KeyValue

	// Time is the time at which this event was recorded.
	Time time.Time
}
