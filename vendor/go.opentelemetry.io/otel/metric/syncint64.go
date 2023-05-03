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

// Int64Counter is an instrument that records increasing int64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Int64Counter interface {
	embedded.Int64Counter

	// Add records a change to the counter.
	Add(ctx context.Context, incr int64, opts ...AddOption)
}

// Int64CounterConfig contains options for synchronous counter instruments that
// record int64 values.
type Int64CounterConfig struct {
	description string
	unit        string
}

// NewInt64CounterConfig returns a new [Int64CounterConfig] with all opts
// applied.
func NewInt64CounterConfig(opts ...Int64CounterOption) Int64CounterConfig {
	var config Int64CounterConfig
	for _, o := range opts {
		config = o.applyInt64Counter(config)
	}
	return config
}

// Description returns the configured description.
func (c Int64CounterConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Int64CounterConfig) Unit() string {
	return c.unit
}

// Int64CounterOption applies options to a [Int64CounterConfig]. See [Option]
// for other options that can be used as an Int64CounterOption.
type Int64CounterOption interface {
	applyInt64Counter(Int64CounterConfig) Int64CounterConfig
}

// Int64UpDownCounter is an instrument that records increasing or decreasing
// int64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Int64UpDownCounter interface {
	embedded.Int64UpDownCounter

	// Add records a change to the counter.
	Add(ctx context.Context, incr int64, opts ...AddOption)
}

// Int64UpDownCounterConfig contains options for synchronous counter
// instruments that record int64 values.
type Int64UpDownCounterConfig struct {
	description string
	unit        string
}

// NewInt64UpDownCounterConfig returns a new [Int64UpDownCounterConfig] with
// all opts applied.
func NewInt64UpDownCounterConfig(opts ...Int64UpDownCounterOption) Int64UpDownCounterConfig {
	var config Int64UpDownCounterConfig
	for _, o := range opts {
		config = o.applyInt64UpDownCounter(config)
	}
	return config
}

// Description returns the configured description.
func (c Int64UpDownCounterConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Int64UpDownCounterConfig) Unit() string {
	return c.unit
}

// Int64UpDownCounterOption applies options to a [Int64UpDownCounterConfig].
// See [Option] for other options that can be used as an
// Int64UpDownCounterOption.
type Int64UpDownCounterOption interface {
	applyInt64UpDownCounter(Int64UpDownCounterConfig) Int64UpDownCounterConfig
}

// Int64Histogram is an instrument that records a distribution of int64
// values.
//
// Warning: Methods may be added to this interface in minor releases. See
// [go.opentelemetry.io/otel/metric] package documentation on API
// implementation for information on how to set default behavior for
// unimplemented methods.
type Int64Histogram interface {
	embedded.Int64Histogram

	// Record adds an additional value to the distribution.
	Record(ctx context.Context, incr int64, opts ...RecordOption)
}

// Int64HistogramConfig contains options for synchronous counter instruments
// that record int64 values.
type Int64HistogramConfig struct {
	description string
	unit        string
}

// NewInt64HistogramConfig returns a new [Int64HistogramConfig] with all opts
// applied.
func NewInt64HistogramConfig(opts ...Int64HistogramOption) Int64HistogramConfig {
	var config Int64HistogramConfig
	for _, o := range opts {
		config = o.applyInt64Histogram(config)
	}
	return config
}

// Description returns the configured description.
func (c Int64HistogramConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Int64HistogramConfig) Unit() string {
	return c.unit
}

// Int64HistogramOption applies options to a [Int64HistogramConfig]. See
// [Option] for other options that can be used as an Int64HistogramOption.
type Int64HistogramOption interface {
	applyInt64Histogram(Int64HistogramConfig) Int64HistogramConfig
}
