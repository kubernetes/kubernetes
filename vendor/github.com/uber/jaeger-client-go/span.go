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
	"strings"
	"sync"
	"time"

	"github.com/opentracing/opentracing-go"
	"github.com/opentracing/opentracing-go/ext"
	"github.com/opentracing/opentracing-go/log"
)

// Span implements opentracing.Span
type Span struct {
	sync.RWMutex

	tracer *tracer

	context SpanContext

	// The name of the "operation" this span is an instance of.
	// Known as a "span name" in some implementations.
	operationName string

	// firstInProcess, if true, indicates that this span is the root of the (sub)tree
	// of spans in the current process. In other words it's true for the root spans,
	// and the ingress spans when the process joins another trace.
	firstInProcess bool

	// startTime is the timestamp indicating when the span began, with microseconds precision.
	startTime time.Time

	// duration returns duration of the span with microseconds precision.
	// Zero value means duration is unknown.
	duration time.Duration

	// tags attached to this span
	tags []Tag

	// The span's "micro-log"
	logs []opentracing.LogRecord

	// references for this span
	references []Reference

	observer SpanObserver
}

// Tag is a simple key value wrapper
type Tag struct {
	key   string
	value interface{}
}

// SetOperationName sets or changes the operation name.
func (s *Span) SetOperationName(operationName string) opentracing.Span {
	s.Lock()
	defer s.Unlock()
	if s.context.IsSampled() {
		s.operationName = operationName
	}
	s.observer.OnSetOperationName(operationName)
	return s
}

// SetTag implements SetTag() of opentracing.Span
func (s *Span) SetTag(key string, value interface{}) opentracing.Span {
	s.observer.OnSetTag(key, value)
	if key == string(ext.SamplingPriority) && setSamplingPriority(s, value) {
		return s
	}
	s.Lock()
	defer s.Unlock()
	if s.context.IsSampled() {
		s.setTagNoLocking(key, value)
	}
	return s
}

func (s *Span) setTagNoLocking(key string, value interface{}) {
	s.tags = append(s.tags, Tag{key: key, value: value})
}

func (s *Span) setTracerTags(tags []Tag) {
	s.Lock()
	for _, tag := range tags {
		s.tags = append(s.tags, tag)
	}
	s.Unlock()
}

// LogFields implements opentracing.Span API
func (s *Span) LogFields(fields ...log.Field) {
	s.Lock()
	defer s.Unlock()
	if !s.context.IsSampled() {
		return
	}
	s.logFieldsNoLocking(fields...)
}

// this function should only be called while holding a Write lock
func (s *Span) logFieldsNoLocking(fields ...log.Field) {
	lr := opentracing.LogRecord{
		Fields:    fields,
		Timestamp: time.Now(),
	}
	s.appendLog(lr)
}

// LogKV implements opentracing.Span API
func (s *Span) LogKV(alternatingKeyValues ...interface{}) {
	s.RLock()
	sampled := s.context.IsSampled()
	s.RUnlock()
	if !sampled {
		return
	}
	fields, err := log.InterleavedKVToFields(alternatingKeyValues...)
	if err != nil {
		s.LogFields(log.Error(err), log.String("function", "LogKV"))
		return
	}
	s.LogFields(fields...)
}

// LogEvent implements opentracing.Span API
func (s *Span) LogEvent(event string) {
	s.Log(opentracing.LogData{Event: event})
}

// LogEventWithPayload implements opentracing.Span API
func (s *Span) LogEventWithPayload(event string, payload interface{}) {
	s.Log(opentracing.LogData{Event: event, Payload: payload})
}

// Log implements opentracing.Span API
func (s *Span) Log(ld opentracing.LogData) {
	s.Lock()
	defer s.Unlock()
	if s.context.IsSampled() {
		if ld.Timestamp.IsZero() {
			ld.Timestamp = s.tracer.timeNow()
		}
		s.appendLog(ld.ToLogRecord())
	}
}

// this function should only be called while holding a Write lock
func (s *Span) appendLog(lr opentracing.LogRecord) {
	// TODO add logic to limit number of logs per span (issue #46)
	s.logs = append(s.logs, lr)
}

// SetBaggageItem implements SetBaggageItem() of opentracing.SpanContext
func (s *Span) SetBaggageItem(key, value string) opentracing.Span {
	key = normalizeBaggageKey(key)
	s.Lock()
	defer s.Unlock()
	s.context = s.context.WithBaggageItem(key, value)
	if s.context.IsSampled() {
		// If sampled, record the baggage in the span
		s.logFieldsNoLocking(
			log.String("event", "baggage"),
			log.String("key", key),
			log.String("value", value),
		)
	}
	return s
}

// BaggageItem implements BaggageItem() of opentracing.SpanContext
func (s *Span) BaggageItem(key string) string {
	key = normalizeBaggageKey(key)
	s.RLock()
	defer s.RUnlock()
	return s.context.baggage[key]
}

// Finish implements opentracing.Span API
func (s *Span) Finish() {
	s.FinishWithOptions(opentracing.FinishOptions{})
}

// FinishWithOptions implements opentracing.Span API
func (s *Span) FinishWithOptions(options opentracing.FinishOptions) {
	if options.FinishTime.IsZero() {
		options.FinishTime = s.tracer.timeNow()
	}
	s.observer.OnFinish(options)
	s.Lock()
	if s.context.IsSampled() {
		s.duration = options.FinishTime.Sub(s.startTime)
		// Note: bulk logs are not subject to maxLogsPerSpan limit
		if options.LogRecords != nil {
			s.logs = append(s.logs, options.LogRecords...)
		}
		for _, ld := range options.BulkLogData {
			s.logs = append(s.logs, ld.ToLogRecord())
		}
	}
	s.Unlock()
	// call reportSpan even for non-sampled traces, to return span to the pool
	s.tracer.reportSpan(s)
}

// Context implements opentracing.Span API
func (s *Span) Context() opentracing.SpanContext {
	return s.context
}

// Tracer implements opentracing.Span API
func (s *Span) Tracer() opentracing.Tracer {
	return s.tracer
}

func (s *Span) String() string {
	s.RLock()
	defer s.RUnlock()
	return s.context.String()
}

// OperationName allows retrieving current operation name.
func (s *Span) OperationName() string {
	s.RLock()
	defer s.RUnlock()
	return s.operationName
}

func setSamplingPriority(s *Span, value interface{}) bool {
	s.Lock()
	defer s.Unlock()
	if val, ok := value.(uint16); ok {
		if val > 0 {
			s.context.flags = s.context.flags | flagDebug | flagSampled
		} else {
			s.context.flags = s.context.flags & (^flagSampled)
		}
		return true
	}
	return false
}

// Converts end-user baggage key into internal representation.
// Used for both read and write access to baggage items.
func normalizeBaggageKey(key string) string {
	// TODO(yurishkuro) normalizeBaggageKey: cache the results in some bounded LRU cache
	return strings.Replace(strings.ToLower(key), "_", "-", -1)
}
