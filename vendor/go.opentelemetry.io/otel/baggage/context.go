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

package baggage // import "go.opentelemetry.io/otel/baggage"

import (
	"context"

	"go.opentelemetry.io/otel/internal/baggage"
)

// ContextWithBaggage returns a copy of parent with baggage.
func ContextWithBaggage(parent context.Context, b Baggage) context.Context {
	// Delegate so any hooks for the OpenTracing bridge are handled.
	return baggage.ContextWithList(parent, b.list)
}

// ContextWithoutBaggage returns a copy of parent with no baggage.
func ContextWithoutBaggage(parent context.Context) context.Context {
	// Delegate so any hooks for the OpenTracing bridge are handled.
	return baggage.ContextWithList(parent, nil)
}

// FromContext returns the baggage contained in ctx.
func FromContext(ctx context.Context) Baggage {
	// Delegate so any hooks for the OpenTracing bridge are handled.
	return Baggage{list: baggage.ListFromContext(ctx)}
}
