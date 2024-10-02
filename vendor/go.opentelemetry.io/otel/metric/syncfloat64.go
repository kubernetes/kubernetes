// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"context"

	"go.opentelemetry.io/otel/metric/embedded"
)

// Float64Counter is an instrument that records increasing float64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Float64Counter interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Float64Counter

	// Add records a change to the counter.
	//
	// Use the WithAttributeSet (or, if performance is not a concern,
	// the WithAttributes) option to include measurement attributes.
	Add(ctx context.Context, incr float64, options ...AddOption)
}

// Float64CounterConfig contains options for synchronous counter instruments that
// record float64 values.
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
// [InstrumentOption] for other options that can be used as a
// Float64CounterOption.
type Float64CounterOption interface {
	applyFloat64Counter(Float64CounterConfig) Float64CounterConfig
}

// Float64UpDownCounter is an instrument that records increasing or decreasing
// float64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Float64UpDownCounter interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Float64UpDownCounter

	// Add records a change to the counter.
	//
	// Use the WithAttributeSet (or, if performance is not a concern,
	// the WithAttributes) option to include measurement attributes.
	Add(ctx context.Context, incr float64, options ...AddOption)
}

// Float64UpDownCounterConfig contains options for synchronous counter
// instruments that record float64 values.
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
// [Float64UpDownCounterConfig]. See [InstrumentOption] for other options that
// can be used as a Float64UpDownCounterOption.
type Float64UpDownCounterOption interface {
	applyFloat64UpDownCounter(Float64UpDownCounterConfig) Float64UpDownCounterConfig
}

// Float64Histogram is an instrument that records a distribution of float64
// values.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Float64Histogram interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Float64Histogram

	// Record adds an additional value to the distribution.
	//
	// Use the WithAttributeSet (or, if performance is not a concern,
	// the WithAttributes) option to include measurement attributes.
	Record(ctx context.Context, incr float64, options ...RecordOption)
}

// Float64HistogramConfig contains options for synchronous histogram
// instruments that record float64 values.
type Float64HistogramConfig struct {
	description              string
	unit                     string
	explicitBucketBoundaries []float64
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

// ExplicitBucketBoundaries returns the configured explicit bucket boundaries.
func (c Float64HistogramConfig) ExplicitBucketBoundaries() []float64 {
	return c.explicitBucketBoundaries
}

// Float64HistogramOption applies options to a [Float64HistogramConfig]. See
// [InstrumentOption] for other options that can be used as a
// Float64HistogramOption.
type Float64HistogramOption interface {
	applyFloat64Histogram(Float64HistogramConfig) Float64HistogramConfig
}

// Float64Gauge is an instrument that records instantaneous float64 values.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// default behavior for unimplemented methods.
type Float64Gauge interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Float64Gauge

	// Record records the instantaneous value.
	//
	// Use the WithAttributeSet (or, if performance is not a concern,
	// the WithAttributes) option to include measurement attributes.
	Record(ctx context.Context, value float64, options ...RecordOption)
}

// Float64GaugeConfig contains options for synchronous gauge instruments that
// record float64 values.
type Float64GaugeConfig struct {
	description string
	unit        string
}

// NewFloat64GaugeConfig returns a new [Float64GaugeConfig] with all opts
// applied.
func NewFloat64GaugeConfig(opts ...Float64GaugeOption) Float64GaugeConfig {
	var config Float64GaugeConfig
	for _, o := range opts {
		config = o.applyFloat64Gauge(config)
	}
	return config
}

// Description returns the configured description.
func (c Float64GaugeConfig) Description() string {
	return c.description
}

// Unit returns the configured unit.
func (c Float64GaugeConfig) Unit() string {
	return c.unit
}

// Float64GaugeOption applies options to a [Float64GaugeConfig]. See
// [InstrumentOption] for other options that can be used as a
// Float64GaugeOption.
type Float64GaugeOption interface {
	applyFloat64Gauge(Float64GaugeConfig) Float64GaugeConfig
}
