package responsewriters

import (
	"io"
	"net/http"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"k8s.io/component-base/tracing"
	"k8s.io/utils/clock"
)

// layer tracks metrics for a specific tracing layer (e.g., "Serialize", "Compress").
type layer struct {
	name          string
	duration      time.Duration
	childDuration time.Duration
	bytes         int64
	count         int64
	child         *layer
}

// TraceWriterManager manages layers of traced writers and reports their metrics
// to both OpenTelemetry and legacy traces via tracing.Span.AddEvent.
type TraceWriterManager struct {
	span   *tracing.Span
	layers []*layer
	clock  clock.PassiveClock
}

// NewTraceWriterManager creates a new manager associated with the given span.
func NewTraceWriterManager(span *tracing.Span) *TraceWriterManager {
	return &TraceWriterManager{
		span:  span,
		clock: clock.RealClock{},
	}
}

// SetClock allows injecting a custom clock for testing.
func (m *TraceWriterManager) SetClock(c clock.PassiveClock) {
	m.clock = c
}

// WrapWriter wraps the given writer to track writes as a named layer.
func (m *TraceWriterManager) WrapWriter(w io.Writer, name string) WriterFlusher {
	l := &layer{name: name}
	tw := &tracedWriter{
		next:  w,
		layer: l,
		m:     m,
	}

	// Find child layer if the writer is a tracedWriter or wraps one.
	if unwrapper, ok := w.(interface{ Unwrap() io.Writer }); ok {
		if childTw, ok := w.(*tracedWriter); ok {
			l.child = childTw.layer
		} else if childTw, ok := unwrapper.Unwrap().(*tracedWriter); ok {
			l.child = childTw.layer
		}
	} else if childTw, ok := w.(*tracedWriter); ok {
		l.child = childTw.layer
	}

	m.layers = append(m.layers, l)
	return tw
}

// WithWriter creates a new WriterSpan for tracking a logical step in writing.
// It wraps the provided writer to track IO time separately from processing time.
func (m *TraceWriterManager) WithWriter(name string, w io.Writer) (io.Writer, *WriterSpan) {
	var childLayer *layer
	for current := w; current != nil; {
		if tw, ok := current.(*tracedWriter); ok {
			childLayer = tw.layer
			break
		}
		if unwrapper, ok := current.(interface{ Unwrap() io.Writer }); ok {
			current = unwrapper.Unwrap()
		} else {
			break
		}
	}

	if childLayer != nil {
		l := &layer{name: name, child: childLayer}
		m.layers = append(m.layers, l)
		return w, &WriterSpan{
			m:     m,
			layer: l,
			start: m.clock.Now(),
		}
	}

	ioLayer := &layer{name: "Writer"}
	m.layers = append(m.layers, ioLayer)

	tw := &tracedWriter{
		next:  w,
		layer: ioLayer,
		m:     m,
	}

	l := &layer{name: name, child: ioLayer}
	m.layers = append(m.layers, l)

	return tw, &WriterSpan{
		m:     m,
		layer: l,
		start: m.clock.Now(),
	}
}

// ReportLayers calculates exclusive durations and reports metrics for all layers.
func (m *TraceWriterManager) ReportLayers() {
	var layers []string
	for _, l := range m.layers {
		layers = append(layers, l.name)
		
		var exclusive time.Duration
		if l.childDuration > 0 {
			exclusive = l.duration - l.childDuration
		} else if l.child != nil {
			exclusive = l.duration - l.child.duration
		} else {
			exclusive = l.duration
		}
		
		if exclusive < 0 {
			exclusive = 0
		}

		reportBytes := l.bytes
		reportCount := l.count

		var attrs []attribute.KeyValue
		if reportBytes > 0 || reportCount > 0 {
			attrs = append(attrs, attribute.Int64("size", reportBytes), attribute.Int64("count", reportCount))
		}

		// Report duration and parallel flag as fields via AddEvent
		attrs = append(attrs, attribute.String("duration", exclusive.String()), attribute.Bool("parallel", true))
		
		m.span.AddEvent(l.name, attrs...)
	}
	if len(layers) > 0 {
		m.span.SetAttributes(attribute.StringSlice("writer.layers", layers))
	}
}

// WriterFlusher combines io.Writer and http.Flusher.
type WriterFlusher interface {
	io.Writer
	http.Flusher
}

// tracedWriter wraps an io.Writer to accumulate duration and byte counts.
type tracedWriter struct {
	next  io.Writer
	layer *layer
	m     *TraceWriterManager
}

var _ io.Writer = (*tracedWriter)(nil)
var _ http.Flusher = (*tracedWriter)(nil)

func (w *tracedWriter) Write(p []byte) (n int, err error) {
	start := w.m.clock.Now()
	n, err = w.next.Write(p)
	duration := w.m.clock.Since(start)
	w.layer.duration += duration
	w.layer.bytes += int64(n)
	w.layer.count++
	return n, err
}

func (w *tracedWriter) Flush() {
	flusher, ok := w.next.(http.Flusher)
	if !ok {
		return
	}
	start := w.m.clock.Now()
	flusher.Flush()
	duration := w.m.clock.Since(start)
	w.layer.duration += duration
	w.layer.count++
}

func (w *tracedWriter) Unwrap() io.Writer {
	return w.next
}

// WriterSpan represents a timed logical step in writing.
type WriterSpan struct {
	m     *TraceWriterManager
	layer *layer
	start time.Time
}

// End completes the span and records its duration.
func (s *WriterSpan) End() {
	duration := s.m.clock.Since(s.start)
	s.layer.duration = duration
	if s.layer.child != nil {
		s.layer.childDuration = s.layer.child.duration
	}
}

// Done is a convenience alias for End.
func (s *WriterSpan) Done() {
	s.End()
}
