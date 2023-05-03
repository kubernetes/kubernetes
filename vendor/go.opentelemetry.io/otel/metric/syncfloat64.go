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

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"context"

	"go.opentelemetry.io/otel/metric/embedded"
)

// Float64Counter is an instrument that records increasing float64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Float64Counter interface {
	embedded.Float64Counter

	// Add records a change to the counter.
	Add(ctx context.Context, incr float64, opts ...AddOption)
}

// Float64CounterConfig contains options for synchronous counter instruments that
// record int64 values.
type Float64CounterConfig struct {
	description string
	unit        string
}

// NewFloat64CounterConfig returns a new [Float64CounterConfig] with all opts
// applied.
func NewFloat64CounterConfig(opts ...Float64CounterOption) Float64CounterConfig {
	var config Float64CounterConfig
	for _, o := range opts {
		config = o.applyFloat64Counter(config)
	}
	return config
}

// Description returns the configured description.
func (c Float64CounterConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Float64CounterConfig) Unit() string {
	return c.unit
}

// Float64CounterOption applies options to a [Float64CounterConfig]. See
// [Option] for other options that can be used as a Float64CounterOption.
type Float64CounterOption interface {
	applyFloat64Counter(Float64CounterConfig) Float64CounterConfig
}

// Float64UpDownCounter is an instrument that records increasing or decreasing
// float64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Float64UpDownCounter interface {
	embedded.Float64UpDownCounter

	// Add records a change to the counter.
	Add(ctx context.Context, incr float64, opts ...AddOption)
}

// Float64UpDownCounterConfig contains options for synchronous counter
// instruments that record int64 values.
type Float64UpDownCounterConfig struct {
	description string
	unit        string
}

// NewFloat64UpDownCounterConfig returns a new [Float64UpDownCounterConfig]
// with all opts applied.
func NewFloat64UpDownCounterConfig(opts ...Float64UpDownCounterOption) Float64UpDownCounterConfig {
	var config Float64UpDownCounterConfig
	for _, o := range opts {
		config = o.applyFloat64UpDownCounter(config)
	}
	return config
}

// Description returns the configured description.
func (c Float64UpDownCounterConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Float64UpDownCounterConfig) Unit() string {
	return c.unit
}

// Float64UpDownCounterOption applies options to a
// [Float64UpDownCounterConfig]. See [Option] for other options that can be
// used as a Float64UpDownCounterOption.
type Float64UpDownCounterOption interface {
	applyFloat64UpDownCounter(Float64UpDownCounterConfig) Float64UpDownCounterConfig
}

// Float64Histogram is an instrument that records a distribution of float64
// values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Float64Histogram interface {
	embedded.Float64Histogram

	// Record adds an additional value to the distribution.
	Record(ctx context.Context, incr float64, opts ...RecordOption)
}

// Float64HistogramConfig contains options for synchronous counter instruments
// that record int64 values.
type Float64HistogramConfig struct {
	description string
	unit        string
}

// NewFloat64HistogramConfig returns a new [Float64HistogramConfig] with all
// opts applied.
func NewFloat64HistogramConfig(opts ...Float64HistogramOption) Float64HistogramConfig {
	var config Float64HistogramConfig
	for _, o := range opts {
		config = o.applyFloat64Histogram(config)
	}
	return config
}

// Description returns the configured description.
func (c Float64HistogramConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Float64HistogramConfig) Unit() string {
	return c.unit
}

// Float64HistogramOption applies options to a [Float64HistogramConfig]. See
// [Option] for other options that can be used as a Float64HistogramOption.
type Float64HistogramOption interface {
	applyFloat64Histogram(Float64HistogramConfig) Float64HistogramConfig
}
