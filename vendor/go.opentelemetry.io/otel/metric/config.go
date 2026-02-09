// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package metric // import "go.opentelemetry.io/otel/metric"

import (
	"slices"

	"go.opentelemetry.io/otel/attribute"
)

// MeterConfig contains options for Meters.
type MeterConfig struct {
	instrumentationVersion string
	schemaURL              string
	attrs                  attribute.Set

	// Ensure forward compatibility by explicitly making this not comparable.
	noCmp [0]func() //nolint: unused  // This is indeed used.
}

// InstrumentationVersion returns the version of the library providing
// instrumentation.
func (cfg MeterConfig) InstrumentationVersion() string {
	return cfg.instrumentationVersion
}

// InstrumentationAttributes returns the attributes associated with the library
// providing instrumentation.
func (cfg MeterConfig) InstrumentationAttributes() attribute.Set {
	return cfg.attrs
}

// SchemaURL is the schema_url of the library providing instrumentation.
func (cfg MeterConfig) SchemaURL() string {
	return cfg.schemaURL
}

// MeterOption is an interface for applying Meter options.
type MeterOption interface {
	// applyMeter is used to set a MeterOption value of a MeterConfig.
	applyMeter(MeterConfig) MeterConfig
}

// NewMeterConfig creates a new MeterConfig and applies
// all the given options.
func NewMeterConfig(opts ...MeterOption) MeterConfig {
	var config MeterConfig
	for _, o := range opts {
		config = o.applyMeter(config)
	}
	return config
}

type meterOptionFunc func(MeterConfig) MeterConfig

func (fn meterOptionFunc) applyMeter(cfg MeterConfig) MeterConfig {
	return fn(cfg)
}

// WithInstrumentationVersion sets the instrumentation version.
func WithInstrumentationVersion(version string) MeterOption {
	return meterOptionFunc(func(config MeterConfig) MeterConfig {
		config.instrumentationVersion = version
		return config
	})
}

// WithInstrumentationAttributes adds the instrumentation attributes.
//
// This is equivalent to calling [WithInstrumentationAttributeSet] with an
// [attribute.Set] created from a clone of the passed attributes.
// [WithInstrumentationAttributeSet] is recommended for more control.
//
// If multiple [WithInstrumentationAttributes] or [WithInstrumentationAttributeSet]
// options are passed, the attributes will be merged together in the order
// they are passed. Attributes with duplicate keys will use the last value passed.
func WithInstrumentationAttributes(attr ...attribute.KeyValue) MeterOption {
	set := attribute.NewSet(slices.Clone(attr)...)
	return WithInstrumentationAttributeSet(set)
}

// WithInstrumentationAttributeSet adds the instrumentation attributes.
//
// If multiple [WithInstrumentationAttributes] or [WithInstrumentationAttributeSet]
// options are passed, the attributes will be merged together in the order
// they are passed. Attributes with duplicate keys will use the last value passed.
func WithInstrumentationAttributeSet(set attribute.Set) MeterOption {
	if set.Len() == 0 {
		return meterOptionFunc(func(config MeterConfig) MeterConfig {
			return config
		})
	}

	return meterOptionFunc(func(config MeterConfig) MeterConfig {
		if config.attrs.Len() == 0 {
			config.attrs = set
		} else {
			config.attrs = mergeSets(config.attrs, set)
		}
		return config
	})
}

// WithSchemaURL sets the schema URL.
func WithSchemaURL(schemaURL string) MeterOption {
	return meterOptionFunc(func(config MeterConfig) MeterConfig {
		config.schemaURL = schemaURL
		return config
	})
}
