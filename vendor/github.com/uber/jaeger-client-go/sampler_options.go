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

// SamplerOption is a function that sets some option on the sampler
type SamplerOption func(options *samplerOptions)

// SamplerOptions is a factory for all available SamplerOption's
var SamplerOptions samplerOptions

type samplerOptions struct {
	metrics                 *Metrics
	maxOperations           int
	sampler                 Sampler
	logger                  Logger
	samplingServerURL       string
	samplingRefreshInterval time.Duration
}

// Metrics creates a SamplerOption that initializes Metrics on the sampler,
// which is used to emit statistics.
func (samplerOptions) Metrics(m *Metrics) SamplerOption {
	return func(o *samplerOptions) {
		o.metrics = m
	}
}

// MaxOperations creates a SamplerOption that sets the maximum number of
// operations the sampler will keep track of.
func (samplerOptions) MaxOperations(maxOperations int) SamplerOption {
	return func(o *samplerOptions) {
		o.maxOperations = maxOperations
	}
}

// InitialSampler creates a SamplerOption that sets the initial sampler
// to use before a remote sampler is created and used.
func (samplerOptions) InitialSampler(sampler Sampler) SamplerOption {
	return func(o *samplerOptions) {
		o.sampler = sampler
	}
}

// Logger creates a SamplerOption that sets the logger used by the sampler.
func (samplerOptions) Logger(logger Logger) SamplerOption {
	return func(o *samplerOptions) {
		o.logger = logger
	}
}

// SamplingServerURL creates a SamplerOption that sets the sampling server url
// of the local agent that contains the sampling strategies.
func (samplerOptions) SamplingServerURL(samplingServerURL string) SamplerOption {
	return func(o *samplerOptions) {
		o.samplingServerURL = samplingServerURL
	}
}

// SamplingRefreshInterval creates a SamplerOption that sets how often the
// sampler will poll local agent for the appropriate sampling strategy.
func (samplerOptions) SamplingRefreshInterval(samplingRefreshInterval time.Duration) SamplerOption {
	return func(o *samplerOptions) {
		o.samplingRefreshInterval = samplingRefreshInterval
	}
}
