// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package metric // import "go.opentelemetry.io/otel/metric"

import "go.opentelemetry.io/otel/attribute"

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

// WithInstrumentationAttributes sets the instrumentation attributes.
//
// The passed attributes will be de-duplicated.
func WithInstrumentationAttributes(attr ...attribute.KeyValue) MeterOption {
	return meterOptionFunc(func(config MeterConfig) MeterConfig {
		config.attrs = attribute.NewSet(attr...)
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
