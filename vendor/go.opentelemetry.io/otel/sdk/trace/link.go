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

package trace // import "go.opentelemetry.io/otel/sdk/trace"

import (
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Link is the relationship between two Spans. The relationship can be within
// the same Trace or across different Traces.
type Link struct {
	// SpanContext of the linked Span.
	SpanContext trace.SpanContext

	// Attributes describe the aspects of the link.
	Attributes []attribute.KeyValue

	// DroppedAttributeCount is the number of attributes that were not
	// recorded due to configured limits being reached.
	DroppedAttributeCount int
}
