// Copyright (c) 2017 Uber Technologies, Inc.
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
	"time"
)

// ReporterOption is a function that sets some option on the reporter.
type ReporterOption func(c *reporterOptions)

// ReporterOptions is a factory for all available ReporterOption's
var ReporterOptions reporterOptions

// reporterOptions control behavior of the reporter.
type reporterOptions struct {
	// queueSize is the size of internal queue where reported spans are stored before they are processed in the background
	queueSize int
	// bufferFlushInterval is how often the buffer is force-flushed, even if it's not full
	bufferFlushInterval time.Duration
	// logger is used to log errors of span submissions
	logger Logger
	// metrics is used to record runtime stats
	metrics *Metrics
}

// QueueSize creates a ReporterOption that sets the size of the internal queue where
// spans are stored before they are processed.
func (reporterOptions) QueueSize(queueSize int) ReporterOption {
	return func(r *reporterOptions) {
		r.queueSize = queueSize
	}
}

// Metrics creates a ReporterOption that initializes Metrics in the reporter,
// which is used to record runtime statistics.
func (reporterOptions) Metrics(metrics *Metrics) ReporterOption {
	return func(r *reporterOptions) {
		r.metrics = metrics
	}
}

// BufferFlushInterval creates a ReporterOption that sets how often the queue
// is force-flushed.
func (reporterOptions) BufferFlushInterval(bufferFlushInterval time.Duration) ReporterOption {
	return func(r *reporterOptions) {
		r.bufferFlushInterval = bufferFlushInterval
	}
}

// Logger creates a ReporterOption that initializes the logger used to log
// errors of span submissions.
func (reporterOptions) Logger(logger Logger) ReporterOption {
	return func(r *reporterOptions) {
		r.logger = logger
	}
}
