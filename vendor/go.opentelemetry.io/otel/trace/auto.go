// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package trace // import "go.opentelemetry.io/otel/trace"

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	semconv "go.opentelemetry.io/otel/semconv/v1.37.0"
	"go.opentelemetry.io/otel/trace/embedded"
	"go.opentelemetry.io/otel/trace/internal/telemetry"
)

// newAutoTracerProvider returns an auto-instrumentable [trace.TracerProvider].
// If an [go.opentelemetry.io/auto.Instrumentation] is configured to instrument
// the process using the returned TracerProvider, all of the telemetry it
// produces will be processed and handled by that Instrumentation. By default,
// if no Instrumentation instruments the TracerProvider it will not generate
// any trace telemetry.
func newAutoTracerProvider() TracerProvider { return tracerProviderInstance }

var tracerProviderInstance = new(autoTracerProvider)

type autoTracerProvider struct{ embedded.TracerProvider }

var _ TracerProvider = autoTracerProvider{}

func (autoTracerProvider) Tracer(name string, opts ...TracerOption) Tracer {
	cfg := NewTracerConfig(opts...)
	return autoTracer{
		name:      name,
		version:   cfg.InstrumentationVersion(),
		schemaURL: cfg.SchemaURL(),
	}
}

type autoTracer struct {
	embedded.Tracer

	name, schemaURL, version string
}

var _ Tracer = autoTracer{}

func (t autoTracer) Start(ctx context.Context, name string, opts ...SpanStartOption) (context.Context, Span) {
	var psc, sc SpanContext
	sampled := true
	span := new(autoSpan)

	// Ask eBPF for sampling decision and span context info.
	t.start(ctx, span, &psc, &sampled, &sc)

	span.sampled.Store(sampled)
	span.spanContext = sc

	ctx = ContextWithSpan(ctx, span)

	if sampled {
		// Only build traces if sampled.
		cfg := NewSpanStartConfig(opts...)
		span.traces, span.span = t.traces(name, cfg, span.spanContext, psc)
	}

	return ctx, span
}

// Expected to be implemented in eBPF.
//
//go:noinline
func (*autoTracer) start(
	ctx context.Context,
	spanPtr *autoSpan,
	psc *SpanContext,
	sampled *bool,
	sc *SpanContext,
) {
	start(ctx, spanPtr, psc, sampled, sc)
}

// start is used for testing.
var start = func(context.Context, *autoSpan, *SpanContext, *bool, *SpanContext) {}

func (t autoTracer) traces(name string, cfg SpanConfig, sc, psc SpanContext) (*telemetry.Traces, *telemetry.Span) {
	span := &telemetry.Span{
		TraceID:      telemetry.TraceID(sc.TraceID()),
		SpanID:       telemetry.SpanID(sc.SpanID()),
		Flags:        uint32(sc.TraceFlags()),
		TraceState:   sc.TraceState().String(),
		ParentSpanID: telemetry.SpanID(psc.SpanID()),
		Name:         name,
		Kind:         spanKind(cfg.SpanKind()),
	}

	span.Attrs, span.DroppedAttrs = convCappedAttrs(maxSpan.Attrs, cfg.Attributes())

	links := cfg.Links()
	if limit := maxSpan.Links; limit == 0 {
		n := int64(len(links))
		if n > 0 {
			span.DroppedLinks = uint32(min(n, math.MaxUint32)) // nolint: gosec  // Bounds checked.
		}
	} else {
		if limit > 0 {
			n := int64(max(len(links)-limit, 0))
			span.DroppedLinks = uint32(min(n, math.MaxUint32)) // nolint: gosec  // Bounds checked.
			links = links[n:]
		}
		span.Links = convLinks(links)
	}

	if t := cfg.Timestamp(); !t.IsZero() {
		span.StartTime = cfg.Timestamp()
	} else {
		span.StartTime = time.Now()
	}

	return &telemetry.Traces{
		ResourceSpans: []*telemetry.ResourceSpans{
			{
				ScopeSpans: []*telemetry.ScopeSpans{
					{
						Scope: &telemetry.Scope{
							Name:    t.name,
							Version: t.version,
						},
						Spans:     []*telemetry.Span{span},
						SchemaURL: t.schemaURL,
					},
				},
			},
		},
	}, span
}

func spanKind(kind SpanKind) telemetry.SpanKind {
	switch kind {
	case SpanKindInternal:
		return telemetry.SpanKindInternal
	case SpanKindServer:
		return telemetry.SpanKindServer
	case SpanKindClient:
		return telemetry.SpanKindClient
	case SpanKindProducer:
		return telemetry.SpanKindProducer
	case SpanKindConsumer:
		return telemetry.SpanKindConsumer
	}
	return telemetry.SpanKind(0) // undefined.
}

type autoSpan struct {
	embedded.Span

	spanContext SpanContext
	sampled     atomic.Bool

	mu     sync.Mutex
	traces *telemetry.Traces
	span   *telemetry.Span
}

func (s *autoSpan) SpanContext() SpanContext {
	if s == nil {
		return SpanContext{}
	}
	// s.spanContext is immutable, do not acquire lock s.mu.
	return s.spanContext
}

func (s *autoSpan) IsRecording() bool {
	if s == nil {
		return false
	}

	return s.sampled.Load()
}

func (s *autoSpan) SetStatus(c codes.Code, msg string) {
	if s == nil || !s.sampled.Load() {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.span.Status == nil {
		s.span.Status = new(telemetry.Status)
	}

	s.span.Status.Message = msg

	switch c {
	case codes.Unset:
		s.span.Status.Code = telemetry.StatusCodeUnset
	case codes.Error:
		s.span.Status.Code = telemetry.StatusCodeError
	case codes.Ok:
		s.span.Status.Code = telemetry.StatusCodeOK
	}
}

func (s *autoSpan) SetAttributes(attrs ...attribute.KeyValue) {
	if s == nil || !s.sampled.Load() {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	limit := maxSpan.Attrs
	if limit == 0 {
		// No attributes allowed.
		n := int64(len(attrs))
		if n > 0 {
			s.span.DroppedAttrs += uint32(min(n, math.MaxUint32)) // nolint: gosec  // Bounds checked.
		}
		return
	}

	m := make(map[string]int)
	for i, a := range s.span.Attrs {
		m[a.Key] = i
	}

	for _, a := range attrs {
		val := convAttrValue(a.Value)
		if val.Empty() {
			s.span.DroppedAttrs++
			continue
		}

		if idx, ok := m[string(a.Key)]; ok {
			s.span.Attrs[idx] = telemetry.Attr{
				Key:   string(a.Key),
				Value: val,
			}
		} else if limit < 0 || len(s.span.Attrs) < limit {
			s.span.Attrs = append(s.span.Attrs, telemetry.Attr{
				Key:   string(a.Key),
				Value: val,
			})
			m[string(a.Key)] = len(s.span.Attrs) - 1
		} else {
			s.span.DroppedAttrs++
		}
	}
}

// convCappedAttrs converts up to limit attrs into a []telemetry.Attr. The
// number of dropped attributes is also returned.
func convCappedAttrs(limit int, attrs []attribute.KeyValue) ([]telemetry.Attr, uint32) {
	n := len(attrs)
	if limit == 0 {
		var out uint32
		if n > 0 {
			out = uint32(min(int64(n), math.MaxUint32)) // nolint: gosec  // Bounds checked.
		}
		return nil, out
	}

	if limit < 0 {
		// Unlimited.
		return convAttrs(attrs), 0
	}

	if n < 0 {
		n = 0
	}

	limit = min(n, limit)
	return convAttrs(attrs[:limit]), uint32(n - limit) // nolint: gosec  // Bounds checked.
}

func convAttrs(attrs []attribute.KeyValue) []telemetry.Attr {
	if len(attrs) == 0 {
		// Avoid allocations if not necessary.
		return nil
	}

	out := make([]telemetry.Attr, 0, len(attrs))
	for _, attr := range attrs {
		key := string(attr.Key)
		val := convAttrValue(attr.Value)
		if val.Empty() {
			continue
		}
		out = append(out, telemetry.Attr{Key: key, Value: val})
	}
	return out
}

func convAttrValue(value attribute.Value) telemetry.Value {
	switch value.Type() {
	case attribute.BOOL:
		return telemetry.BoolValue(value.AsBool())
	case attribute.INT64:
		return telemetry.Int64Value(value.AsInt64())
	case attribute.FLOAT64:
		return telemetry.Float64Value(value.AsFloat64())
	case attribute.STRING:
		v := truncate(maxSpan.AttrValueLen, value.AsString())
		return telemetry.StringValue(v)
	case attribute.BOOLSLICE:
		slice := value.AsBoolSlice()
		out := make([]telemetry.Value, 0, len(slice))
		for _, v := range slice {
			out = append(out, telemetry.BoolValue(v))
		}
		return telemetry.SliceValue(out...)
	case attribute.INT64SLICE:
		slice := value.AsInt64Slice()
		out := make([]telemetry.Value, 0, len(slice))
		for _, v := range slice {
			out = append(out, telemetry.Int64Value(v))
		}
		return telemetry.SliceValue(out...)
	case attribute.FLOAT64SLICE:
		slice := value.AsFloat64Slice()
		out := make([]telemetry.Value, 0, len(slice))
		for _, v := range slice {
			out = append(out, telemetry.Float64Value(v))
		}
		return telemetry.SliceValue(out...)
	case attribute.STRINGSLICE:
		slice := value.AsStringSlice()
		out := make([]telemetry.Value, 0, len(slice))
		for _, v := range slice {
			v = truncate(maxSpan.AttrValueLen, v)
			out = append(out, telemetry.StringValue(v))
		}
		return telemetry.SliceValue(out...)
	}
	return telemetry.Value{}
}

// truncate returns a truncated version of s such that it contains less than
// the limit number of characters. Truncation is applied by returning the limit
// number of valid characters contained in s.
//
// If limit is negative, it returns the original string.
//
// UTF-8 is supported. When truncating, all invalid characters are dropped
// before applying truncation.
//
// If s already contains less than the limit number of bytes, it is returned
// unchanged. No invalid characters are removed.
func truncate(limit int, s string) string {
	// This prioritize performance in the following order based on the most
	// common expected use-cases.
	//
	//  - Short values less than the default limit (128).
	//  - Strings with valid encodings that exceed the limit.
	//  - No limit.
	//  - Strings with invalid encodings that exceed the limit.
	if limit < 0 || len(s) <= limit {
		return s
	}

	// Optimistically, assume all valid UTF-8.
	var b strings.Builder
	count := 0
	for i, c := range s {
		if c != utf8.RuneError {
			count++
			if count > limit {
				return s[:i]
			}
			continue
		}

		_, size := utf8.DecodeRuneInString(s[i:])
		if size == 1 {
			// Invalid encoding.
			b.Grow(len(s) - 1)
			_, _ = b.WriteString(s[:i])
			s = s[i:]
			break
		}
	}

	// Fast-path, no invalid input.
	if b.Cap() == 0 {
		return s
	}

	// Truncate while validating UTF-8.
	for i := 0; i < len(s) && count < limit; {
		c := s[i]
		if c < utf8.RuneSelf {
			// Optimization for single byte runes (common case).
			_ = b.WriteByte(c)
			i++
			count++
			continue
		}

		_, size := utf8.DecodeRuneInString(s[i:])
		if size == 1 {
			// We checked for all 1-byte runes above, this is a RuneError.
			i++
			continue
		}

		_, _ = b.WriteString(s[i : i+size])
		i += size
		count++
	}

	return b.String()
}

func (s *autoSpan) End(opts ...SpanEndOption) {
	if s == nil || !s.sampled.Swap(false) {
		return
	}

	// s.end exists so the lock (s.mu) is not held while s.ended is called.
	s.ended(s.end(opts))
}

func (s *autoSpan) end(opts []SpanEndOption) []byte {
	s.mu.Lock()
	defer s.mu.Unlock()

	cfg := NewSpanEndConfig(opts...)
	if t := cfg.Timestamp(); !t.IsZero() {
		s.span.EndTime = cfg.Timestamp()
	} else {
		s.span.EndTime = time.Now()
	}

	b, _ := json.Marshal(s.traces) // TODO: do not ignore this error.
	return b
}

// Expected to be implemented in eBPF.
//
//go:noinline
func (*autoSpan) ended(buf []byte) { ended(buf) }

// ended is used for testing.
var ended = func([]byte) {}

func (s *autoSpan) RecordError(err error, opts ...EventOption) {
	if s == nil || err == nil || !s.sampled.Load() {
		return
	}

	cfg := NewEventConfig(opts...)

	attrs := cfg.Attributes()
	attrs = append(attrs,
		semconv.ExceptionType(typeStr(err)),
		semconv.ExceptionMessage(err.Error()),
	)
	if cfg.StackTrace() {
		buf := make([]byte, 2048)
		n := runtime.Stack(buf, false)
		attrs = append(attrs, semconv.ExceptionStacktrace(string(buf[0:n])))
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.addEvent(semconv.ExceptionEventName, cfg.Timestamp(), attrs)
}

func typeStr(i any) string {
	t := reflect.TypeOf(i)
	if t.PkgPath() == "" && t.Name() == "" {
		// Likely a builtin type.
		return t.String()
	}
	return fmt.Sprintf("%s.%s", t.PkgPath(), t.Name())
}

func (s *autoSpan) AddEvent(name string, opts ...EventOption) {
	if s == nil || !s.sampled.Load() {
		return
	}

	cfg := NewEventConfig(opts...)

	s.mu.Lock()
	defer s.mu.Unlock()

	s.addEvent(name, cfg.Timestamp(), cfg.Attributes())
}

// addEvent adds an event with name and attrs at tStamp to the span. The span
// lock (s.mu) needs to be held by the caller.
func (s *autoSpan) addEvent(name string, tStamp time.Time, attrs []attribute.KeyValue) {
	limit := maxSpan.Events

	if limit == 0 {
		s.span.DroppedEvents++
		return
	}

	if limit > 0 && len(s.span.Events) == limit {
		// Drop head while avoiding allocation of more capacity.
		copy(s.span.Events[:limit-1], s.span.Events[1:])
		s.span.Events = s.span.Events[:limit-1]
		s.span.DroppedEvents++
	}

	e := &telemetry.SpanEvent{Time: tStamp, Name: name}
	e.Attrs, e.DroppedAttrs = convCappedAttrs(maxSpan.EventAttrs, attrs)

	s.span.Events = append(s.span.Events, e)
}

func (s *autoSpan) AddLink(link Link) {
	if s == nil || !s.sampled.Load() {
		return
	}

	l := maxSpan.Links

	s.mu.Lock()
	defer s.mu.Unlock()

	if l == 0 {
		s.span.DroppedLinks++
		return
	}

	if l > 0 && len(s.span.Links) == l {
		// Drop head while avoiding allocation of more capacity.
		copy(s.span.Links[:l-1], s.span.Links[1:])
		s.span.Links = s.span.Links[:l-1]
		s.span.DroppedLinks++
	}

	s.span.Links = append(s.span.Links, convLink(link))
}

func convLinks(links []Link) []*telemetry.SpanLink {
	out := make([]*telemetry.SpanLink, 0, len(links))
	for _, link := range links {
		out = append(out, convLink(link))
	}
	return out
}

func convLink(link Link) *telemetry.SpanLink {
	l := &telemetry.SpanLink{
		TraceID:    telemetry.TraceID(link.SpanContext.TraceID()),
		SpanID:     telemetry.SpanID(link.SpanContext.SpanID()),
		TraceState: link.SpanContext.TraceState().String(),
		Flags:      uint32(link.SpanContext.TraceFlags()),
	}
	l.Attrs, l.DroppedAttrs = convCappedAttrs(maxSpan.LinkAttrs, link.Attributes)

	return l
}

func (s *autoSpan) SetName(name string) {
	if s == nil || !s.sampled.Load() {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	s.span.Name = name
}

func (*autoSpan) TracerProvider() TracerProvider { return newAutoTracerProvider() }

// maxSpan are the span limits resolved during startup.
var maxSpan = newSpanLimits()

type spanLimits struct {
	// Attrs is the number of allowed attributes for a span.
	//
	// This is resolved from the environment variable value for the
	// OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT key if it exists. Otherwise, the
	// environment variable value for OTEL_ATTRIBUTE_COUNT_LIMIT, or 128 if
	// that is not set, is used.
	Attrs int
	// AttrValueLen is the maximum attribute value length allowed for a span.
	//
	// This is resolved from the environment variable value for the
	// OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT key if it exists. Otherwise, the
	// environment variable value for OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT, or -1
	// if that is not set, is used.
	AttrValueLen int
	// Events is the number of allowed events for a span.
	//
	// This is resolved from the environment variable value for the
	// OTEL_SPAN_EVENT_COUNT_LIMIT key, or 128 is used if that is not set.
	Events int
	// EventAttrs is the number of allowed attributes for a span event.
	//
	// The is resolved from the environment variable value for the
	// OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT key, or 128 is used if that is not set.
	EventAttrs int
	// Links is the number of allowed Links for a span.
	//
	// This is resolved from the environment variable value for the
	// OTEL_SPAN_LINK_COUNT_LIMIT, or 128 is used if that is not set.
	Links int
	// LinkAttrs is the number of allowed attributes for a span link.
	//
	// This is resolved from the environment variable value for the
	// OTEL_LINK_ATTRIBUTE_COUNT_LIMIT, or 128 is used if that is not set.
	LinkAttrs int
}

func newSpanLimits() spanLimits {
	return spanLimits{
		Attrs: firstEnv(
			128,
			"OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT",
			"OTEL_ATTRIBUTE_COUNT_LIMIT",
		),
		AttrValueLen: firstEnv(
			-1, // Unlimited.
			"OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT",
			"OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
		),
		Events:     firstEnv(128, "OTEL_SPAN_EVENT_COUNT_LIMIT"),
		EventAttrs: firstEnv(128, "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT"),
		Links:      firstEnv(128, "OTEL_SPAN_LINK_COUNT_LIMIT"),
		LinkAttrs:  firstEnv(128, "OTEL_LINK_ATTRIBUTE_COUNT_LIMIT"),
	}
}

// firstEnv returns the parsed integer value of the first matching environment
// variable from keys. The defaultVal is returned if the value is not an
// integer or no match is found.
func firstEnv(defaultVal int, keys ...string) int {
	for _, key := range keys {
		strV := os.Getenv(key)
		if strV == "" {
			continue
		}

		v, err := strconv.Atoi(strV)
		if err == nil {
			return v
		}
		// Ignore invalid environment variable.
	}

	return defaultVal
}
