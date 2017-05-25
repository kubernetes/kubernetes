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
	"github.com/uber/jaeger-lib/metrics"
)

// Metrics is a container of all stats emitted by Jaeger tracer.
type Metrics struct {
	// Number of traces started by this tracer as sampled
	TracesStartedSampled metrics.Counter `metric:"traces" tags:"state=started,sampled=y"`

	// Number of traces started by this tracer as not sampled
	TracesStartedNotSampled metrics.Counter `metric:"traces" tags:"state=started,sampled=n"`

	// Number of externally started sampled traces this tracer joined
	TracesJoinedSampled metrics.Counter `metric:"traces" tags:"state=joined,sampled=y"`

	// Number of externally started not-sampled traces this tracer joined
	TracesJoinedNotSampled metrics.Counter `metric:"traces" tags:"state=joined,sampled=n"`

	// Number of sampled spans started by this tracer
	SpansStarted metrics.Counter `metric:"spans" tags:"group=lifecycle,state=started"`

	// Number of sampled spans finished by this tracer
	SpansFinished metrics.Counter `metric:"spans" tags:"group=lifecycle,state=finished"`

	// Number of sampled spans started by this tracer
	SpansSampled metrics.Counter `metric:"spans" tags:"group=sampling,sampled=y"`

	// Number of not-sampled spans started by this tracer
	SpansNotSampled metrics.Counter `metric:"spans" tags:"group=sampling,sampled=n"`

	// Number of errors decoding tracing context
	DecodingErrors metrics.Counter `metric:"decoding-errors"`

	// Number of spans successfully reported
	ReporterSuccess metrics.Counter `metric:"reporter-spans" tags:"state=success"`

	// Number of spans in failed attempts to report
	ReporterFailure metrics.Counter `metric:"reporter-spans" tags:"state=failure"`

	// Number of spans dropped due to internal queue overflow
	ReporterDropped metrics.Counter `metric:"reporter-spans" tags:"state=dropped"`

	// Current number of spans in the reporter queue
	ReporterQueueLength metrics.Gauge `metric:"reporter-queue"`

	// Number of times the Sampler succeeded to retrieve sampling strategy
	SamplerRetrieved metrics.Counter `metric:"sampler" tags:"state=retrieved"`

	// Number of times the Sampler succeeded to retrieve and update sampling strategy
	SamplerUpdated metrics.Counter `metric:"sampler" tags:"state=updated"`

	// Number of times the Sampler failed to update sampling strategy
	SamplerUpdateFailure metrics.Counter `metric:"sampler" tags:"state=failure,phase=updating"`

	// Number of times the Sampler failed to retrieve sampling strategy
	SamplerQueryFailure metrics.Counter `metric:"sampler" tags:"state=failure,phase=query"`
}

// NewMetrics creates a new Metrics struct and initializes it.
func NewMetrics(factory metrics.Factory, globalTags map[string]string) *Metrics {
	m := &Metrics{}
	metrics.Init(m, factory.Namespace("jaeger", nil), globalTags)
	return m
}

// NewNullMetrics creates a new Metrics struct that won't report any metrics.
func NewNullMetrics() *Metrics {
	return NewMetrics(metrics.NullFactory, nil)
}
