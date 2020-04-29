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
	"go.opentelemetry.io/otel/api/core"
)

const (
	alwaysOnSamplerDescription = "AlwaysOnSampler"
)

var alwaysOnSamplerDecision = Decision{Sampled: true}

type alwaysOnSampler struct{}

// ShouldSample implements Sampler interface.
// It always returns a Decision with Sampled value set to true
// and with Attributes set to an empty slice.
func (as alwaysOnSampler) ShouldSample(
	_ core.SpanContext,
	_ bool,
	_ core.TraceID,
	_ core.SpanID,
	_ string,
	_ SpanKind,
	_ []core.KeyValue,
	_ []Link,
) Decision {
	return alwaysOnSamplerDecision
}

// Description implements Sampler interface.
// It returns the description of this sampler.
func (as alwaysOnSampler) Description() string {
	return alwaysOnSamplerDescription
}

var _ Sampler = alwaysOnSampler{}

func AlwaysOnSampler() Sampler {
	return alwaysOnSampler{}
}
