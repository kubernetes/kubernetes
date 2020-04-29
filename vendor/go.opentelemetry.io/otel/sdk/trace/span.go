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
	"fmt"
	"reflect"
	"sync"
	"time"

	"google.golang.org/grpc/codes"

	"go.opentelemetry.io/otel/api/core"
	apitrace "go.opentelemetry.io/otel/api/trace"
	export "go.opentelemetry.io/otel/sdk/export/trace"
	"go.opentelemetry.io/otel/sdk/internal"
)

const (
	errorTypeKey    = core.Key("error.type")
	errorMessageKey = core.Key("error.message")
	errorEventName  = "error"
)

// span implements apitrace.Span interface.
type span struct {
	// data contains information recorded about the span.
	//
	// It will be non-nil if we are exporting the span or recording events for it.
	// Otherwise, data is nil, and the span is simply a carrier for the
	// SpanContext, so that the trace ID is propagated.
	data        *export.SpanData
	mu          sync.Mutex // protects the contents of *data (but not the pointer value.)
	spanContext core.SpanContext

	// attributes are capped at configured limit. When the capacity is reached an oldest entry
	// is removed to create room for a new entry.
	attributes *attributesMap

	// messageEvents are stored in FIFO queue capped by configured limit.
	messageEvents *evictedQueue

	// links are stored in FIFO queue capped by configured limit.
	links *evictedQueue

	// spanStore is the spanStore this span belongs to, if any, otherwise it is nil.
	//*spanStore
	endOnce sync.Once

	executionTracerTaskEnd func()  // ends the execution tracer span
	tracer                 *tracer // tracer used to create span.
}

var _ apitrace.Span = &span{}

func (s *span) SpanContext() core.SpanContext {
	if s == nil {
		return core.EmptySpanContext()
	}
	return s.spanContext
}

func (s *span) IsRecording() bool {
	if s == nil {
		return false
	}
	return s.data != nil
}

func (s *span) SetStatus(code codes.Code, msg string) {
	if s == nil {
		return
	}
	if !s.IsRecording() {
		return
	}
	s.mu.Lock()
	s.data.StatusCode = code
	s.data.StatusMessage = msg
	s.mu.Unlock()
}

func (s *span) SetAttributes(attributes ...core.KeyValue) {
	if !s.IsRecording() {
		return
	}
	s.copyToCappedAttributes(attributes...)
}

func (s *span) End(options ...apitrace.EndOption) {
	if s == nil {
		return
	}

	if s.executionTracerTaskEnd != nil {
		s.executionTracerTaskEnd()
	}
	if !s.IsRecording() {
		return
	}
	opts := apitrace.EndConfig{}
	for _, opt := range options {
		opt(&opts)
	}
	s.endOnce.Do(func() {
		sps, _ := s.tracer.provider.spanProcessors.Load().(spanProcessorMap)
		mustExportOrProcess := len(sps) > 0
		if mustExportOrProcess {
			sd := s.makeSpanData()
			if opts.EndTime.IsZero() {
				sd.EndTime = internal.MonotonicEndTime(sd.StartTime)
			} else {
				sd.EndTime = opts.EndTime
			}
			for sp := range sps {
				sp.OnEnd(sd)
			}
		}
	})
}

func (s *span) RecordError(ctx context.Context, err error, opts ...apitrace.ErrorOption) {
	if s == nil || err == nil {
		return
	}

	if !s.IsRecording() {
		return
	}

	cfg := apitrace.ErrorConfig{}

	for _, o := range opts {
		o(&cfg)
	}

	if cfg.Timestamp.IsZero() {
		cfg.Timestamp = time.Now()
	}

	if cfg.StatusCode != codes.OK {
		s.SetStatus(cfg.StatusCode, "")
	}

	errType := reflect.TypeOf(err)
	errTypeString := fmt.Sprintf("%s.%s", errType.PkgPath(), errType.Name())
	if errTypeString == "." {
		// PkgPath() and Name() may be empty for builtin Types
		errTypeString = errType.String()
	}

	s.AddEventWithTimestamp(ctx, cfg.Timestamp, errorEventName,
		errorTypeKey.String(errTypeString),
		errorMessageKey.String(err.Error()),
	)
}

func (s *span) Tracer() apitrace.Tracer {
	return s.tracer
}

func (s *span) AddEvent(ctx context.Context, name string, attrs ...core.KeyValue) {
	if !s.IsRecording() {
		return
	}
	s.addEventWithTimestamp(time.Now(), name, attrs...)
}

func (s *span) AddEventWithTimestamp(ctx context.Context, timestamp time.Time, name string, attrs ...core.KeyValue) {
	if !s.IsRecording() {
		return
	}
	s.addEventWithTimestamp(timestamp, name, attrs...)
}

func (s *span) addEventWithTimestamp(timestamp time.Time, name string, attrs ...core.KeyValue) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.messageEvents.add(export.Event{
		Name:       name,
		Attributes: attrs,
		Time:       timestamp,
	})
}

func (s *span) SetName(name string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.data == nil {
		// TODO: now what?
		return
	}
	s.data.Name = name
	// SAMPLING
	noParent := !s.data.ParentSpanID.IsValid()
	var ctx core.SpanContext
	if noParent {
		ctx = core.EmptySpanContext()
	} else {
		// FIXME: Where do we get the parent context from?
		// From SpanStore?
		ctx = s.data.SpanContext
	}
	data := samplingData{
		noParent:     noParent,
		remoteParent: s.data.HasRemoteParent,
		parent:       ctx,
		name:         name,
		cfg:          s.tracer.provider.config.Load().(*Config),
		span:         s,
		attributes:   s.data.Attributes,
		links:        s.data.Links,
		kind:         s.data.SpanKind,
	}
	sampled := makeSamplingDecision(data)

	// Adding attributes directly rather than using s.SetAttributes()
	// as s.mu is already locked and attempting to do so would deadlock.
	for _, a := range sampled.Attributes {
		s.attributes.add(a)
	}
}

func (s *span) addLink(link apitrace.Link) {
	if !s.IsRecording() {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.links.add(link)
}

// makeSpanData produces a SpanData representing the current state of the span.
// It requires that s.data is non-nil.
func (s *span) makeSpanData() *export.SpanData {
	var sd export.SpanData
	s.mu.Lock()
	defer s.mu.Unlock()
	sd = *s.data

	s.attributes.toSpanData(&sd)

	if len(s.messageEvents.queue) > 0 {
		sd.MessageEvents = s.interfaceArrayToMessageEventArray()
		sd.DroppedMessageEventCount = s.messageEvents.droppedCount
	}
	if len(s.links.queue) > 0 {
		sd.Links = s.interfaceArrayToLinksArray()
		sd.DroppedLinkCount = s.links.droppedCount
	}
	return &sd
}

func (s *span) interfaceArrayToLinksArray() []apitrace.Link {
	linkArr := make([]apitrace.Link, 0)
	for _, value := range s.links.queue {
		linkArr = append(linkArr, value.(apitrace.Link))
	}
	return linkArr
}

func (s *span) interfaceArrayToMessageEventArray() []export.Event {
	messageEventArr := make([]export.Event, 0)
	for _, value := range s.messageEvents.queue {
		messageEventArr = append(messageEventArr, value.(export.Event))
	}
	return messageEventArr
}

func (s *span) copyToCappedAttributes(attributes ...core.KeyValue) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, a := range attributes {
		s.attributes.add(a)
	}
}

func (s *span) addChild() {
	if !s.IsRecording() {
		return
	}
	s.mu.Lock()
	s.data.ChildSpanCount++
	s.mu.Unlock()
}

func startSpanInternal(tr *tracer, name string, parent core.SpanContext, remoteParent bool, o apitrace.StartConfig) *span {
	var noParent bool
	span := &span{}
	span.spanContext = parent

	cfg := tr.provider.config.Load().(*Config)

	if parent == core.EmptySpanContext() {
		span.spanContext.TraceID = cfg.IDGenerator.NewTraceID()
		noParent = true
	}
	span.spanContext.SpanID = cfg.IDGenerator.NewSpanID()
	data := samplingData{
		noParent:     noParent,
		remoteParent: remoteParent,
		parent:       parent,
		name:         name,
		cfg:          cfg,
		span:         span,
		attributes:   o.Attributes,
		links:        o.Links,
		kind:         o.SpanKind,
	}
	sampled := makeSamplingDecision(data)

	// TODO: [rghetia] restore when spanstore is added.
	// if !internal.LocalSpanStoreEnabled && !span.spanContext.IsSampled() && !o.Record {
	if !span.spanContext.IsSampled() && !o.Record {
		return span
	}

	startTime := o.StartTime
	if startTime.IsZero() {
		startTime = time.Now()
	}
	span.data = &export.SpanData{
		SpanContext:     span.spanContext,
		StartTime:       startTime,
		SpanKind:        apitrace.ValidateSpanKind(o.SpanKind),
		Name:            name,
		HasRemoteParent: remoteParent,
		Resource:        cfg.Resource,
	}
	span.attributes = newAttributesMap(cfg.MaxAttributesPerSpan)
	span.messageEvents = newEvictedQueue(cfg.MaxEventsPerSpan)
	span.links = newEvictedQueue(cfg.MaxLinksPerSpan)

	span.SetAttributes(sampled.Attributes...)

	if !noParent {
		span.data.ParentSpanID = parent.SpanID
	}
	// TODO: [rghetia] restore when spanstore is added.
	//if internal.LocalSpanStoreEnabled {
	//	ss := spanStoreForNameCreateIfNew(name)
	//	if ss != nil {
	//		span.spanStore = ss
	//		ss.add(span)
	//	}
	//}

	return span
}

type samplingData struct {
	noParent     bool
	remoteParent bool
	parent       core.SpanContext
	name         string
	cfg          *Config
	span         *span
	attributes   []core.KeyValue
	links        []apitrace.Link
	kind         apitrace.SpanKind
}

func makeSamplingDecision(data samplingData) SamplingResult {
	if data.noParent || data.remoteParent {
		// If this span is the child of a local span and no
		// Sampler is set in the options, keep the parent's
		// TraceFlags.
		//
		// Otherwise, consult the Sampler in the options if it
		// is non-nil, otherwise the default sampler.
		sampler := data.cfg.DefaultSampler
		//if o.Sampler != nil {
		//	sampler = o.Sampler
		//}
		spanContext := &data.span.spanContext
		sampled := sampler.ShouldSample(SamplingParameters{
			ParentContext:   data.parent,
			TraceID:         spanContext.TraceID,
			SpanID:          spanContext.SpanID,
			Name:            data.name,
			HasRemoteParent: data.remoteParent,
			Kind:            data.kind,
			Attributes:      data.attributes,
			Links:           data.links,
		})
		if sampled.Decision == RecordAndSampled {
			spanContext.TraceFlags |= core.TraceFlagsSampled
		} else {
			spanContext.TraceFlags &^= core.TraceFlagsSampled
		}
		return sampled
	}
	if data.parent.TraceFlags&core.TraceFlagsSampled != 0 {
		return SamplingResult{Decision: RecordAndSampled}
	}
	return SamplingResult{Decision: NotRecord}
}
