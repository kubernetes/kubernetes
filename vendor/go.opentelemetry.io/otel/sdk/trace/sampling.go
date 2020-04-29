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
	"encoding/binary"
	"fmt"

	"go.opentelemetry.io/otel/api/core"
	api "go.opentelemetry.io/otel/api/trace"
)

// Sampler decides whether a trace should be sampled and exported.
type Sampler interface {
	ShouldSample(SamplingParameters) SamplingResult
	Description() string
}

// SamplingParameters contains the values passed to a Sampler.
type SamplingParameters struct {
	ParentContext   core.SpanContext
	TraceID         core.TraceID
	SpanID          core.SpanID
	Name            string
	HasRemoteParent bool
	Kind            api.SpanKind
	Attributes      []core.KeyValue
	Links           []api.Link
}

// SamplingDecision indicates whether a span is recorded and sampled.
type SamplingDecision uint8

// Valid sampling decisions
const (
	NotRecord SamplingDecision = iota
	Record
	RecordAndSampled
)

// SamplingResult conveys a SamplingDecision and a set of Attributes.
type SamplingResult struct {
	Decision   SamplingDecision
	Attributes []core.KeyValue
}

type probabilitySampler struct {
	traceIDUpperBound uint64
	description       string
}

func (ps probabilitySampler) ShouldSample(p SamplingParameters) SamplingResult {
	if p.ParentContext.IsSampled() {
		return SamplingResult{Decision: RecordAndSampled}
	}

	x := binary.BigEndian.Uint64(p.TraceID[0:8]) >> 1
	if x < ps.traceIDUpperBound {
		return SamplingResult{Decision: RecordAndSampled}
	}
	return SamplingResult{Decision: NotRecord}
}

func (ps probabilitySampler) Description() string {
	return ps.description
}

// ProbabilitySampler samples a given fraction of traces. Fractions >= 1 will
// always sample. If the parent span is sampled, then it's child spans will
// automatically be sampled. Fractions < 0 are treated as zero, but spans may
// still be sampled if their parent is.
func ProbabilitySampler(fraction float64) Sampler {
	if fraction >= 1 {
		return AlwaysSample()
	}

	if fraction <= 0 {
		fraction = 0
	}

	return &probabilitySampler{
		traceIDUpperBound: uint64(fraction * (1 << 63)),
		description:       fmt.Sprintf("ProbabilitySampler{%g}", fraction),
	}
}

type alwaysOnSampler struct{}

func (as alwaysOnSampler) ShouldSample(p SamplingParameters) SamplingResult {
	return SamplingResult{Decision: RecordAndSampled}
}

func (as alwaysOnSampler) Description() string {
	return "AlwaysOnSampler"
}

// AlwaysSample returns a Sampler that samples every trace.
// Be careful about using this sampler in a production application with
// significant traffic: a new trace will be started and exported for every
// request.
func AlwaysSample() Sampler {
	return alwaysOnSampler{}
}

type alwaysOffSampler struct{}

func (as alwaysOffSampler) ShouldSample(p SamplingParameters) SamplingResult {
	return SamplingResult{Decision: NotRecord}
}

func (as alwaysOffSampler) Description() string {
	return "AlwaysOffSampler"
}

// NeverSample returns a Sampler that samples no traces.
func NeverSample() Sampler {
	return alwaysOffSampler{}
}

// AlwaysParentSample returns a Sampler that samples a trace only
// if the parent span is sampled.
// This Sampler is a passthrough to the ProbabilitySampler with
// a fraction of value 0.
func AlwaysParentSample() Sampler {
	return &probabilitySampler{
		traceIDUpperBound: 0,
		description:       "AlwaysParentSampler",
	}
}
