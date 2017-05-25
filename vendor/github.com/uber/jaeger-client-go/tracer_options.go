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
	"time"
)

// TracerOption is a function that sets some option on the tracer
type TracerOption func(tracer *tracer)

// TracerOptions is a factory for all available TracerOption's
var TracerOptions tracerOptions

type tracerOptions struct{}

// Metrics creates a TracerOption that initializes Metrics on the tracer,
// which is used to emit statistics.
func (tracerOptions) Metrics(m *Metrics) TracerOption {
	return func(tracer *tracer) {
		tracer.metrics = *m
	}
}

// Logger creates a TracerOption that gives the tracer a Logger.
func (tracerOptions) Logger(logger Logger) TracerOption {
	return func(tracer *tracer) {
		tracer.logger = logger
	}
}

// TimeNow creates a TracerOption that gives the tracer a function
// used to generate timestamps for spans.
func (tracerOptions) TimeNow(timeNow func() time.Time) TracerOption {
	return func(tracer *tracer) {
		tracer.timeNow = timeNow
	}
}

// RandomNumber creates a TracerOption that gives the tracer
// a thread-safe random number generator function for generating trace IDs.
func (tracerOptions) RandomNumber(randomNumber func() uint64) TracerOption {
	return func(tracer *tracer) {
		tracer.randomNumber = randomNumber
	}
}

// PoolSpans creates a TracerOption that tells the tracer whether it should use
// an object pool to minimize span allocations.
// This should be used with care, only if the service is not running any async tasks
// that can access parent spans after those spans have been finished.
func (tracerOptions) PoolSpans(poolSpans bool) TracerOption {
	return func(tracer *tracer) {
		tracer.options.poolSpans = poolSpans
	}
}

// HostIPv4 creates a TracerOption that identifies the current service/process.
// If not set, the factory method will obtain the current IP address.
func (tracerOptions) HostIPv4(hostIPv4 uint32) TracerOption {
	return func(tracer *tracer) {
		tracer.hostIPv4 = hostIPv4
	}
}

func (tracerOptions) Injector(format interface{}, injector Injector) TracerOption {
	return func(tracer *tracer) {
		tracer.injectors[format] = injector
	}
}

func (tracerOptions) Extractor(format interface{}, extractor Extractor) TracerOption {
	return func(tracer *tracer) {
		tracer.extractors[format] = extractor
	}
}

func (tracerOptions) Observer(observer Observer) TracerOption {
	return func(tracer *tracer) {
		tracer.observer.append(observer)
	}
}
