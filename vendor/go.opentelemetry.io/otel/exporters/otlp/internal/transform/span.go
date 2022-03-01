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

package transform

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	tracepb "go.opentelemetry.io/proto/otlp/trace/v1"

	"go.opentelemetry.io/otel/sdk/instrumentation"
	tracesdk "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

const (
	maxMessageEventsPerSpan = 128
)

// SpanData transforms a slice of SpanSnapshot into a slice of OTLP
// ResourceSpans.
func SpanData(sdl []*tracesdk.SpanSnapshot) []*tracepb.ResourceSpans {
	if len(sdl) == 0 {
		return nil
	}

	rsm := make(map[attribute.Distinct]*tracepb.ResourceSpans)

	type ilsKey struct {
		r  attribute.Distinct
		il instrumentation.Library
	}
	ilsm := make(map[ilsKey]*tracepb.InstrumentationLibrarySpans)

	var resources int
	for _, sd := range sdl {
		if sd == nil {
			continue
		}

		rKey := sd.Resource.Equivalent()
		iKey := ilsKey{
			r:  rKey,
			il: sd.InstrumentationLibrary,
		}
		ils, iOk := ilsm[iKey]
		if !iOk {
			// Either the resource or instrumentation library were unknown.
			ils = &tracepb.InstrumentationLibrarySpans{
				InstrumentationLibrary: instrumentationLibrary(sd.InstrumentationLibrary),
				Spans:                  []*tracepb.Span{},
			}
		}
		ils.Spans = append(ils.Spans, span(sd))
		ilsm[iKey] = ils

		rs, rOk := rsm[rKey]
		if !rOk {
			resources++
			// The resource was unknown.
			rs = &tracepb.ResourceSpans{
				Resource:                    Resource(sd.Resource),
				InstrumentationLibrarySpans: []*tracepb.InstrumentationLibrarySpans{ils},
			}
			rsm[rKey] = rs
			continue
		}

		// The resource has been seen before. Check if the instrumentation
		// library lookup was unknown because if so we need to add it to the
		// ResourceSpans. Otherwise, the instrumentation library has already
		// been seen and the append we did above will be included it in the
		// InstrumentationLibrarySpans reference.
		if !iOk {
			rs.InstrumentationLibrarySpans = append(rs.InstrumentationLibrarySpans, ils)
		}
	}

	// Transform the categorized map into a slice
	rss := make([]*tracepb.ResourceSpans, 0, resources)
	for _, rs := range rsm {
		rss = append(rss, rs)
	}
	return rss
}

// span transforms a Span into an OTLP span.
func span(sd *tracesdk.SpanSnapshot) *tracepb.Span {
	if sd == nil {
		return nil
	}

	tid := sd.SpanContext.TraceID()
	sid := sd.SpanContext.SpanID()

	s := &tracepb.Span{
		TraceId:                tid[:],
		SpanId:                 sid[:],
		TraceState:             sd.SpanContext.TraceState().String(),
		Status:                 status(sd.StatusCode, sd.StatusMessage),
		StartTimeUnixNano:      uint64(sd.StartTime.UnixNano()),
		EndTimeUnixNano:        uint64(sd.EndTime.UnixNano()),
		Links:                  links(sd.Links),
		Kind:                   spanKind(sd.SpanKind),
		Name:                   sd.Name,
		Attributes:             Attributes(sd.Attributes),
		Events:                 spanEvents(sd.MessageEvents),
		DroppedAttributesCount: uint32(sd.DroppedAttributeCount),
		DroppedEventsCount:     uint32(sd.DroppedMessageEventCount),
		DroppedLinksCount:      uint32(sd.DroppedLinkCount),
	}

	if psid := sd.Parent.SpanID(); psid.IsValid() {
		s.ParentSpanId = psid[:]
	}

	return s
}

// status transform a span code and message into an OTLP span status.
func status(status codes.Code, message string) *tracepb.Status {
	var c tracepb.Status_StatusCode
	switch status {
	case codes.Error:
		c = tracepb.Status_STATUS_CODE_ERROR
	default:
		c = tracepb.Status_STATUS_CODE_OK
	}
	return &tracepb.Status{
		Code:    c,
		Message: message,
	}
}

// links transforms span Links to OTLP span links.
func links(links []trace.Link) []*tracepb.Span_Link {
	if len(links) == 0 {
		return nil
	}

	sl := make([]*tracepb.Span_Link, 0, len(links))
	for _, otLink := range links {
		// This redefinition is necessary to prevent otLink.*ID[:] copies
		// being reused -- in short we need a new otLink per iteration.
		otLink := otLink

		tid := otLink.TraceID()
		sid := otLink.SpanID()

		sl = append(sl, &tracepb.Span_Link{
			TraceId:    tid[:],
			SpanId:     sid[:],
			Attributes: Attributes(otLink.Attributes),
		})
	}
	return sl
}

// spanEvents transforms span Events to an OTLP span events.
func spanEvents(es []trace.Event) []*tracepb.Span_Event {
	if len(es) == 0 {
		return nil
	}

	evCount := len(es)
	if evCount > maxMessageEventsPerSpan {
		evCount = maxMessageEventsPerSpan
	}
	events := make([]*tracepb.Span_Event, 0, evCount)
	messageEvents := 0

	// Transform message events
	for _, e := range es {
		if messageEvents >= maxMessageEventsPerSpan {
			break
		}
		messageEvents++
		events = append(events,
			&tracepb.Span_Event{
				Name:         e.Name,
				TimeUnixNano: uint64(e.Time.UnixNano()),
				Attributes:   Attributes(e.Attributes),
				// TODO (rghetia) : Add Drop Counts when supported.
			},
		)
	}

	return events
}

// spanKind transforms a SpanKind to an OTLP span kind.
func spanKind(kind trace.SpanKind) tracepb.Span_SpanKind {
	switch kind {
	case trace.SpanKindInternal:
		return tracepb.Span_SPAN_KIND_INTERNAL
	case trace.SpanKindClient:
		return tracepb.Span_SPAN_KIND_CLIENT
	case trace.SpanKindServer:
		return tracepb.Span_SPAN_KIND_SERVER
	case trace.SpanKindProducer:
		return tracepb.Span_SPAN_KIND_PRODUCER
	case trace.SpanKindConsumer:
		return tracepb.Span_SPAN_KIND_CONSUMER
	default:
		return tracepb.Span_SPAN_KIND_UNSPECIFIED
	}
}
