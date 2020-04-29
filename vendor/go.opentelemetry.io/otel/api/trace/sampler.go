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

import "go.opentelemetry.io/otel/api/core"

type Sampler interface {
	// ShouldSample returns a Decision that contains a decision whether to sample
	// or not sample the span to be created. Decision is based on a Sampler specific
	// algorithm that takes into account one or more input parameters.
	ShouldSample(
		sc core.SpanContext,
		remote bool,
		traceID core.TraceID,
		spanID core.SpanID,
		spanName string,
		spanKind SpanKind,
		attributes []core.KeyValue,
		links []Link,
	) Decision

	// Description returns of the sampler. It contains its name or short description
	// and its configured properties.
	// For example 'ProbabilitySampler:{0.00001}'
	Description() string
}

type Decision struct {
	// Sampled is set true if the span should be sampled.
	Sampled bool

	// Attributes provides insight into Sample	r's decision process.
	// It could be empty slice or nil if no attributes are recorded by the sampler.
	Attributes []core.KeyValue
}
