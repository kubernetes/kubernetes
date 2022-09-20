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

package asyncint64 // import "go.opentelemetry.io/otel/metric/instrument/asyncint64"

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric/instrument"
)

// InstrumentProvider provides access to individual instruments.
type InstrumentProvider interface {
	// Counter creates an instrument for recording increasing values.
	Counter(name string, opts ...instrument.Option) (Counter, error)

	// UpDownCounter creates an instrument for recording changes of a value.
	UpDownCounter(name string, opts ...instrument.Option) (UpDownCounter, error)

	// Gauge creates an instrument for recording the current value.
	Gauge(name string, opts ...instrument.Option) (Gauge, error)
}

// Counter is an instrument that records increasing values.
type Counter interface {
	// Observe records the state of the instrument.
	//
	// It is only valid to call this within a callback. If called outside of the
	// registered callback it should have no effect on the instrument, and an
	// error will be reported via the error handler.
	Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue)

	instrument.Asynchronous
}

// UpDownCounter is an instrument that records increasing or decreasing values.
type UpDownCounter interface {
	// Observe records the state of the instrument.
	//
	// It is only valid to call this within a callback. If called outside of the
	// registered callback it should have no effect on the instrument, and an
	// error will be reported via the error handler.
	Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue)

	instrument.Asynchronous
}

// Gauge is an instrument that records independent readings.
type Gauge interface {
	// Observe records the state of the instrument.
	//
	// It is only valid to call this within a callback. If called outside of the
	// registered callback it should have no effect on the instrument, and an
	// error will be reported via the error handler.
	Observe(ctx context.Context, x int64, attrs ...attribute.KeyValue)

	instrument.Asynchronous
}
