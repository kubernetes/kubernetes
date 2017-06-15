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

const (
	// JaegerClientVersion is the version of the client library reported as Span tag.
	JaegerClientVersion = "Go-2.6.1dev"

	// JaegerClientVersionTagKey is the name of the tag used to report client version.
	JaegerClientVersionTagKey = "jaeger.version"

	// JaegerDebugHeader is the name of HTTP header or a TextMap carrier key which,
	// if found in the carrier, forces the trace to be sampled as "debug" trace.
	// The value of the header is recorded as the tag on the root span, so that the
	// trace can be found in the UI using this value as a correlation ID.
	JaegerDebugHeader = "jaeger-debug-id"

	// JaegerBaggageHeader is the name of the HTTP header that is used to submit baggage.
	// It differs from TraceBaggageHeaderPrefix in that it can be used only in cases where
	// a root span does not exist.
	JaegerBaggageHeader = "jaeger-baggage"

	// TracerHostnameTagKey used to report host name of the process.
	TracerHostnameTagKey = "jaeger.hostname"

	// SamplerTypeTagKey reports which sampler was used on the root span.
	SamplerTypeTagKey = "sampler.type"

	// SamplerParamTagKey reports the parameter of the sampler, like sampling probability.
	SamplerParamTagKey = "sampler.param"

	// TracerStateHeaderName is the http header name used to propagate tracing context.
	// This must be in lower-case to avoid mismatches when decoding incoming headers.
	TracerStateHeaderName = "uber-trace-id"

	// TraceBaggageHeaderPrefix is the prefix for http headers used to propagate baggage.
	// This must be in lower-case to avoid mismatches when decoding incoming headers.
	TraceBaggageHeaderPrefix = "uberctx-"

	// SamplerTypeConst is the type of sampler that always makes the same decision.
	SamplerTypeConst = "const"

	// SamplerTypeRemote is the type of sampler that polls Jaeger agent for sampling strategy.
	SamplerTypeRemote = "remote"

	// SamplerTypeProbabilistic is the type of sampler that samples traces
	// with a certain fixed probability.
	SamplerTypeProbabilistic = "probabilistic"

	// SamplerTypeRateLimiting is the type of sampler that samples
	// only up to a fixed number of traces per second.
	SamplerTypeRateLimiting = "ratelimiting"

	// SamplerTypeLowerBound is the type of sampler that samples
	// only up to a fixed number of traces per second.
	SamplerTypeLowerBound = "lowerbound"
)
