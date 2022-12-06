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

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"

	"go.opentelemetry.io/otel/baggage"
)

const baggageHeader = "baggage"

// Baggage is a propagator that supports the W3C Baggage format.
//
// This propagates user-defined baggage associated with a trace. The complete
// specification is defined at https://www.w3.org/TR/baggage/.
type Baggage struct{}

var _ TextMapPropagator = Baggage{}

// Inject sets baggage key-values from ctx into the carrier.
func (b Baggage) Inject(ctx context.Context, carrier TextMapCarrier) {
	bStr := baggage.FromContext(ctx).String()
	if bStr != "" {
		carrier.Set(baggageHeader, bStr)
	}
}

// Extract returns a copy of parent with the baggage from the carrier added.
func (b Baggage) Extract(parent context.Context, carrier TextMapCarrier) context.Context {
	bStr := carrier.Get(baggageHeader)
	if bStr == "" {
		return parent
	}

	bag, err := baggage.Parse(bStr)
	if err != nil {
		return parent
	}
	return baggage.ContextWithBaggage(parent, bag)
}

// Fields returns the keys who's values are set with Inject.
func (b Baggage) Fields() []string {
	return []string{baggageHeader}
}
