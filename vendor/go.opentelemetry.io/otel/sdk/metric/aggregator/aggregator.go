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

package aggregator // import "go.opentelemetry.io/otel/sdk/metric/aggregator"

import (
	"fmt"
	"math"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/number"
	export "go.opentelemetry.io/otel/sdk/export/metric"
	"go.opentelemetry.io/otel/sdk/export/metric/aggregation"
)

// NewInconsistentAggregatorError formats an error describing an attempt to
// Checkpoint or Merge different-type aggregators.  The result can be unwrapped as
// an ErrInconsistentType.
func NewInconsistentAggregatorError(a1, a2 export.Aggregator) error {
	return fmt.Errorf("%w: %T and %T", aggregation.ErrInconsistentType, a1, a2)
}

// RangeTest is a common routine for testing for valid input values.
// This rejects NaN values.  This rejects negative values when the
// metric instrument does not support negative values, including
// monotonic counter metrics and absolute ValueRecorder metrics.
func RangeTest(num number.Number, descriptor *metric.Descriptor) error {
	numberKind := descriptor.NumberKind()

	if numberKind == number.Float64Kind && math.IsNaN(num.AsFloat64()) {
		return aggregation.ErrNaNInput
	}

	switch descriptor.InstrumentKind() {
	case metric.CounterInstrumentKind, metric.SumObserverInstrumentKind:
		if num.IsNegative(numberKind) {
			return aggregation.ErrNegativeInput
		}
	}
	return nil
}
