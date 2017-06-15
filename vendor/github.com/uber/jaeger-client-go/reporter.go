// Copyright (c) 2016 Uber Technologies, Inc.
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
	"sync"
	"sync/atomic"
	"time"

	"github.com/opentracing/opentracing-go"

	"github.com/uber/jaeger-client-go/log"
	"github.com/uber/jaeger-client-go/thrift-gen/zipkincore"
	"github.com/uber/jaeger-client-go/transport"
)

// Reporter is called by the tracer when a span is completed to report the span to the tracing collector.
type Reporter interface {
	// Report submits a new span to collectors, possibly asynchronously and/or with buffering.
	Report(span *Span)

	// Close does a clean shutdown of the reporter, flushing any traces that may be buffered in memory.
	Close()
}

// ------------------------------

type nullReporter struct{}

// NewNullReporter creates a no-op reporter that ignores all reported spans.
func NewNullReporter() Reporter {
	return &nullReporter{}
}

// Report implements Report() method of Reporter by doing nothing.
func (r *nullReporter) Report(span *Span) {
	// no-op
}

// Close implements Close() method of Reporter by doing nothing.
func (r *nullReporter) Close() {
	// no-op
}

// ------------------------------

type loggingReporter struct {
	logger Logger
}

// NewLoggingReporter creates a reporter that logs all reported spans to provided logger.
func NewLoggingReporter(logger Logger) Reporter {
	return &loggingReporter{logger}
}

// Report implements Report() method of Reporter by logging the span to the logger.
func (r *loggingReporter) Report(span *Span) {
	r.logger.Infof("Reporting span %+v", span)
}

// Close implements Close() method of Reporter by doing nothing.
func (r *loggingReporter) Close() {
	// no-op
}

// ------------------------------

// InMemoryReporter is used for testing, and simply collects spans in memory.
type InMemoryReporter struct {
	spans []opentracing.Span
	lock  sync.Mutex
}

// NewInMemoryReporter creates a reporter that stores spans in memory.
// NOTE: the Tracer should be created with options.PoolSpans = false.
func NewInMemoryReporter() *InMemoryReporter {
	return &InMemoryReporter{
		spans: make([]opentracing.Span, 0, 10),
	}
}

// Report implements Report() method of Reporter by storing the span in the buffer.
func (r *InMemoryReporter) Report(span *Span) {
	r.lock.Lock()
	r.spans = append(r.spans, span)
	r.lock.Unlock()
}

// Close implements Close() method of Reporter by doing nothing.
func (r *InMemoryReporter) Close() {
	// no-op
}

// SpansSubmitted returns the number of spans accumulated in the buffer.
func (r *InMemoryReporter) SpansSubmitted() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return len(r.spans)
}

// GetSpans returns accumulated spans as a copy of the buffer.
func (r *InMemoryReporter) GetSpans() []opentracing.Span {
	r.lock.Lock()
	defer r.lock.Unlock()
	copied := make([]opentracing.Span, len(r.spans))
	copy(copied, r.spans)
	return copied
}

// Reset clears all accumulated spans.
func (r *InMemoryReporter) Reset() {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.spans = nil
}

// ------------------------------

type compositeReporter struct {
	reporters []Reporter
}

// NewCompositeReporter creates a reporter that ignores all reported spans.
func NewCompositeReporter(reporters ...Reporter) Reporter {
	return &compositeReporter{reporters: reporters}
}

// Report implements Report() method of Reporter by delegating to each underlying reporter.
func (r *compositeReporter) Report(span *Span) {
	for _, reporter := range r.reporters {
		reporter.Report(span)
	}
}

// Close implements Close() method of Reporter by closing each underlying reporter.
func (r *compositeReporter) Close() {
	for _, reporter := range r.reporters {
		reporter.Close()
	}
}

// ------------------------------

const (
	defaultQueueSize           = 100
	defaultBufferFlushInterval = 10 * time.Second
)

type remoteReporter struct {
	reporterOptions
	sender       transport.Transport
	queue        chan *zipkincore.Span
	queueLength  int64 // signed because metric's gauge is signed
	queueDrained sync.WaitGroup
	flushSignal  chan *sync.WaitGroup
}

// NewRemoteReporter creates a new reporter that sends spans out of process by means of Sender
func NewRemoteReporter(sender transport.Transport, opts ...ReporterOption) Reporter {
	options := reporterOptions{}
	for _, option := range opts {
		option(&options)
	}
	if options.bufferFlushInterval <= 0 {
		options.bufferFlushInterval = defaultBufferFlushInterval
	}
	if options.logger == nil {
		options.logger = log.NullLogger
	}
	if options.metrics == nil {
		options.metrics = NewNullMetrics()
	}
	if options.queueSize <= 0 {
		options.queueSize = defaultQueueSize
	}
	reporter := &remoteReporter{
		reporterOptions: options,
		sender:          sender,
		flushSignal:     make(chan *sync.WaitGroup),
		queue:           make(chan *zipkincore.Span, options.queueSize),
	}
	go reporter.processQueue()
	return reporter
}

// Report implements Report() method of Reporter.
// It passes the span to a background go-routine for submission to Jaeger.
func (r *remoteReporter) Report(span *Span) {
	thriftSpan := buildThriftSpan(span)
	select {
	case r.queue <- thriftSpan:
		atomic.AddInt64(&r.queueLength, 1)
	default:
		r.metrics.ReporterDropped.Inc(1)
	}
}

// Close implements Close() method of Reporter by waiting for the queue to be drained.
func (r *remoteReporter) Close() {
	r.queueDrained.Add(1)
	close(r.queue)
	r.queueDrained.Wait()
	r.sender.Close()
}

// processQueue reads spans from the queue, converts them to Thrift, and stores them in an internal buffer.
// When the buffer length reaches batchSize, it is flushed by submitting the accumulated spans to Jaeger.
// Buffer also gets flushed automatically every batchFlushInterval seconds, just in case the tracer stopped
// reporting new spans.
func (r *remoteReporter) processQueue() {
	timer := time.NewTicker(r.bufferFlushInterval)
	for {
		select {
		case span, ok := <-r.queue:
			if ok {
				atomic.AddInt64(&r.queueLength, -1)
				if flushed, err := r.sender.Append(span); err != nil {
					r.metrics.ReporterFailure.Inc(int64(flushed))
					r.logger.Error(err.Error())
				} else if flushed > 0 {
					r.metrics.ReporterSuccess.Inc(int64(flushed))
					// to reduce the number of gauge stats, we only emit queue length on flush
					r.metrics.ReporterQueueLength.Update(atomic.LoadInt64(&r.queueLength))
				}
			} else {
				// queue closed
				timer.Stop()
				r.flush()
				r.queueDrained.Done()
				return
			}
		case <-timer.C:
			r.flush()
		case wg := <-r.flushSignal: // for testing
			r.flush()
			wg.Done()
		}
	}
}

// flush causes the Sender to flush its accumulated spans and clear the buffer
func (r *remoteReporter) flush() {
	if flushed, err := r.sender.Flush(); err != nil {
		r.metrics.ReporterFailure.Inc(int64(flushed))
		r.logger.Error(err.Error())
	} else if flushed > 0 {
		r.metrics.ReporterSuccess.Inc(int64(flushed))
	}
}
