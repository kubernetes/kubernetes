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

import "go.opentelemetry.io/otel/attribute"

// Observable is used as a grouping mechanism for all instruments that are
// updated within a Callback.
type Observable interface {
	observable()
}

// InstrumentOption applies options to all instruments.
type InstrumentOption interface {
	Int64CounterOption
	Int64UpDownCounterOption
	Int64HistogramOption
	Int64ObservableCounterOption
	Int64ObservableUpDownCounterOption
	Int64ObservableGaugeOption

	Float64CounterOption
	Float64UpDownCounterOption
	Float64HistogramOption
	Float64ObservableCounterOption
	Float64ObservableUpDownCounterOption
	Float64ObservableGaugeOption
}

type descOpt string

func (o descOpt) applyFloat64Counter(c Float64CounterConfig) Float64CounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64UpDownCounter(c Float64UpDownCounterConfig) Float64UpDownCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64Histogram(c Float64HistogramConfig) Float64HistogramConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64ObservableCounter(c Float64ObservableCounterConfig) Float64ObservableCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64ObservableUpDownCounter(c Float64ObservableUpDownCounterConfig) Float64ObservableUpDownCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyFloat64ObservableGauge(c Float64ObservableGaugeConfig) Float64ObservableGaugeConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64Counter(c Int64CounterConfig) Int64CounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64UpDownCounter(c Int64UpDownCounterConfig) Int64UpDownCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64Histogram(c Int64HistogramConfig) Int64HistogramConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64ObservableCounter(c Int64ObservableCounterConfig) Int64ObservableCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64ObservableUpDownCounter(c Int64ObservableUpDownCounterConfig) Int64ObservableUpDownCounterConfig {
	c.description = string(o)
	return c
}

func (o descOpt) applyInt64ObservableGauge(c Int64ObservableGaugeConfig) Int64ObservableGaugeConfig {
	c.description = string(o)
	return c
}

// WithDescription sets the instrument description.
func WithDescription(desc string) InstrumentOption { return descOpt(desc) }

type unitOpt string

func (o unitOpt) applyFloat64Counter(c Float64CounterConfig) Float64CounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64UpDownCounter(c Float64UpDownCounterConfig) Float64UpDownCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64Histogram(c Float64HistogramConfig) Float64HistogramConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64ObservableCounter(c Float64ObservableCounterConfig) Float64ObservableCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64ObservableUpDownCounter(c Float64ObservableUpDownCounterConfig) Float64ObservableUpDownCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyFloat64ObservableGauge(c Float64ObservableGaugeConfig) Float64ObservableGaugeConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64Counter(c Int64CounterConfig) Int64CounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64UpDownCounter(c Int64UpDownCounterConfig) Int64UpDownCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64Histogram(c Int64HistogramConfig) Int64HistogramConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64ObservableCounter(c Int64ObservableCounterConfig) Int64ObservableCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64ObservableUpDownCounter(c Int64ObservableUpDownCounterConfig) Int64ObservableUpDownCounterConfig {
	c.unit = string(o)
	return c
}

func (o unitOpt) applyInt64ObservableGauge(c Int64ObservableGaugeConfig) Int64ObservableGaugeConfig {
	c.unit = string(o)
	return c
}

// WithUnit sets the instrument unit.
func WithUnit(u string) InstrumentOption { return unitOpt(u) }

// AddOption applies options to an addition measurement. See
// [MeasurementOption] for other options that can be used as an AddOption.
type AddOption interface {
	applyAdd(AddConfig) AddConfig
}

// AddConfig contains options for an addition measurement.
type AddConfig struct {
	attrs attribute.Set
}

// NewAddConfig returns a new [AddConfig] with all opts applied.
func NewAddConfig(opts []AddOption) AddConfig {
	config := AddConfig{attrs: *attribute.EmptySet()}
	for _, o := range opts {
		config = o.applyAdd(config)
	}
	return config
}

// Attributes returns the configured attribute set.
func (c AddConfig) Attributes() attribute.Set {
	return c.attrs
}

// RecordOption applies options to an addition measurement. See
// [MeasurementOption] for other options that can be used as a RecordOption.
type RecordOption interface {
	applyRecord(RecordConfig) RecordConfig
}

// RecordConfig contains options for a recorded measurement.
type RecordConfig struct {
	attrs attribute.Set
}

// NewRecordConfig returns a new [RecordConfig] with all opts applied.
func NewRecordConfig(opts []RecordOption) RecordConfig {
	config := RecordConfig{attrs: *attribute.EmptySet()}
	for _, o := range opts {
		config = o.applyRecord(config)
	}
	return config
}

// Attributes returns the configured attribute set.
func (c RecordConfig) Attributes() attribute.Set {
	return c.attrs
}

// ObserveOption applies options to an addition measurement. See
// [MeasurementOption] for other options that can be used as a ObserveOption.
type ObserveOption interface {
	applyObserve(ObserveConfig) ObserveConfig
}

// ObserveConfig contains options for an observed measurement.
type ObserveConfig struct {
	attrs attribute.Set
}

// NewObserveConfig returns a new [ObserveConfig] with all opts applied.
func NewObserveConfig(opts []ObserveOption) ObserveConfig {
	config := ObserveConfig{attrs: *attribute.EmptySet()}
	for _, o := range opts {
		config = o.applyObserve(config)
	}
	return config
}

// Attributes returns the configured attribute set.
func (c ObserveConfig) Attributes() attribute.Set {
	return c.attrs
}

// MeasurementOption applies options to all instrument measurement.
type MeasurementOption interface {
	AddOption
	RecordOption
	ObserveOption
}

type attrOpt struct {
	set attribute.Set
}

// mergeSets returns the union of keys between a and b. Any duplicate keys will
// use the value associated with b.
func mergeSets(a, b attribute.Set) attribute.Set {
	// NewMergeIterator uses the first value for any duplicates.
	iter := attribute.NewMergeIterator(&b, &a)
	merged := make([]attribute.KeyValue, 0, a.Len()+b.Len())
	for iter.Next() {
		merged = append(merged, iter.Attribute())
	}
	return attribute.NewSet(merged...)
}

func (o attrOpt) applyAdd(c AddConfig) AddConfig {
	switch {
	case o.set.Len() == 0:
	case c.attrs.Len() == 0:
		c.attrs = o.set
	default:
		c.attrs = mergeSets(c.attrs, o.set)
	}
	return c
}

func (o attrOpt) applyRecord(c RecordConfig) RecordConfig {
	switch {
	case o.set.Len() == 0:
	case c.attrs.Len() == 0:
		c.attrs = o.set
	default:
		c.attrs = mergeSets(c.attrs, o.set)
	}
	return c
}

func (o attrOpt) applyObserve(c ObserveConfig) ObserveConfig {
	switch {
	case o.set.Len() == 0:
	case c.attrs.Len() == 0:
		c.attrs = o.set
	default:
		c.attrs = mergeSets(c.attrs, o.set)
	}
	return c
}

// WithAttributeSet sets the attribute Set associated with a measurement is
// made with.
//
// If multiple WithAttributeSet or WithAttributes options are passed the
// attributes will be merged together in the order they are passed. Attributes
// with duplicate keys will use the last value passed.
func WithAttributeSet(attributes attribute.Set) MeasurementOption {
	return attrOpt{set: attributes}
}

// WithAttributes converts attributes into an attribute Set and sets the Set to
// be associated with a measurement. This is shorthand for:
//
//	cp := make([]attribute.KeyValue, len(attributes))
//	copy(cp, attributes)
//	WithAttributes(attribute.NewSet(cp...))
//
// [attribute.NewSet] may modify the passed attributes so this will make a copy
// of attributes before creating a set in order to ensure this function is
// concurrent safe. This makes this option function less optimized in
// comparison to [WithAttributeSet]. Therefore, [WithAttributeSet] should be
// preferred for performance sensitive code.
//
// See [WithAttributeSet] for information about how multiple WithAttributes are
// merged.
func WithAttributes(attributes ...attribute.KeyValue) MeasurementOption {
	cp := make([]attribute.KeyValue, len(attributes))
	copy(cp, attributes)
	return attrOpt{set: attribute.NewSet(cp...)}
}
