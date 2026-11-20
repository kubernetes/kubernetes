// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package log

import (
	"context"
	"slices"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/log/embedded"
)

// Logger emits log records.
//
// Warning: Methods may be added to this interface in minor releases. See
// package documentation on API implementation for information on how to set
// the default behavior for unimplemented methods.
type Logger interface {
	// Users of the interface can ignore this. This embedded type is only used
	// by implementations of this interface. See the "API Implementations"
	// section of the package documentation for more information.
	embedded.Logger

	// Emit emits a log record.
	//
	// The record may be held by the implementation. Callers should not mutate
	// the record after it is passed.
	//
	// Implementations of this method need to be safe for a user to call
	// concurrently.
	Emit(ctx context.Context, record Record)

	// Enabled reports whether the Logger emits for the given context and
	// param.
	//
	// Calling Enabled is optional. It is not required before calling
	// [Logger.Emit].
	//
	// Enabled is useful when constructing a [Record] is expensive. A caller can
	// call Enabled first and skip record construction when it returns false.
	// When constructing the Record is inexpensive, a caller can emit it
	// directly.
	//
	// The returned value is not static and may change over time. A cached value
	// can become stale.
	//
	// The passed param is likely to contain only partial information about a
	// record (e.g., a param with only the Severity set).
	// If a Logger needs more information than is provided, it
	// is said to be in an indeterminate state (see below).
	//
	// The returned value is true if the Logger emits for the provided context
	// and param, and false if the Logger does not emit. The returned value may be
	// true or false in an indeterminate state.
	// An implementation should default to returning true for an indeterminate
	// state, but may return false if there are valid reasons to do so in
	// particular circumstances (e.g., performance or correctness).
	//
	// The param should not be held by the implementation. A copy should be
	// made if the param needs to be held after the call returns.
	//
	// Implementations of this method need to be safe for a user to call
	// concurrently.
	Enabled(ctx context.Context, param EnabledParameters) bool
}

// LoggerOption applies configuration options to a [Logger].
type LoggerOption interface {
	// applyLogger applies a LoggerOption to a LoggerConfig.
	applyLogger(LoggerConfig) LoggerConfig
}

// LoggerConfig contains options for a [Logger].
type LoggerConfig struct {
	// Ensure forward compatibility by explicitly making this not comparable.
	noCmp [0]func() //nolint: unused  // This is indeed used.

	version   string
	schemaURL string
	attrs     attribute.Set
}

type experimentalOption interface {
	Experimental()
}

// NewLoggerConfig returns a new [LoggerConfig] with all the options applied.
func NewLoggerConfig(options ...LoggerOption) LoggerConfig {
	var c LoggerConfig
	for _, opt := range options {
		if _, ok := opt.(experimentalOption); ok {
			continue
		}
		c = opt.applyLogger(c)
	}
	return c
}

// InstrumentationVersion returns the version of the library providing
// instrumentation.
func (cfg LoggerConfig) InstrumentationVersion() string {
	return cfg.version
}

// InstrumentationAttributes returns the attributes associated with the library
// providing instrumentation.
func (cfg LoggerConfig) InstrumentationAttributes() attribute.Set {
	return cfg.attrs
}

// SchemaURL returns the schema URL of the library providing instrumentation.
func (cfg LoggerConfig) SchemaURL() string {
	return cfg.schemaURL
}

type loggerOptionFunc func(LoggerConfig) LoggerConfig

func (fn loggerOptionFunc) applyLogger(cfg LoggerConfig) LoggerConfig {
	return fn(cfg)
}

// WithInstrumentationVersion returns a [LoggerOption] that sets the
// instrumentation version of a [Logger].
func WithInstrumentationVersion(version string) LoggerOption {
	return loggerOptionFunc(func(config LoggerConfig) LoggerConfig {
		config.version = version
		return config
	})
}

// mergeSets returns the union of the keys in a and b. For duplicate keys, the
// value associated with b is used.
func mergeSets(a, b attribute.Set) attribute.Set {
	// NewMergeIterator uses the first value for any duplicates.
	iter := attribute.NewMergeIterator(&b, &a)
	merged := make([]attribute.KeyValue, 0, a.Len()+b.Len())
	for iter.Next() {
		merged = append(merged, iter.Attribute())
	}
	return attribute.NewSet(merged...)
}

// WithInstrumentationAttributes returns a [LoggerOption] that sets the
// instrumentation attributes of a [Logger].
//
// This is equivalent to calling WithInstrumentationAttributeSet with an
// [attribute.Set] created from a clone of the passed attributes.
// [WithInstrumentationAttributeSet] is recommended for more control.
//
// If multiple [WithInstrumentationAttributes] or [WithInstrumentationAttributeSet]
// options are passed, the attributes will be merged together in the order
// they are passed. For attributes with duplicate keys, the last value passed
// will be used.
func WithInstrumentationAttributes(attr ...attribute.KeyValue) LoggerOption {
	set := attribute.NewSet(slices.Clone(attr)...)
	return WithInstrumentationAttributeSet(set)
}

// WithInstrumentationAttributeSet returns a [LoggerOption] that adds
// instrumentation attributes to a [Logger].
//
// If multiple [WithInstrumentationAttributes] or [WithInstrumentationAttributeSet]
// options are passed, the attributes will be merged together in the order
// they are passed. For attributes with duplicate keys, the last value passed
// will be used.
func WithInstrumentationAttributeSet(set attribute.Set) LoggerOption {
	if set.Len() == 0 {
		return loggerOptionFunc(func(config LoggerConfig) LoggerConfig {
			return config
		})
	}

	return loggerOptionFunc(func(config LoggerConfig) LoggerConfig {
		if config.attrs.Len() == 0 {
			config.attrs = set
		} else {
			config.attrs = mergeSets(config.attrs, set)
		}
		return config
	})
}

// WithSchemaURL returns a [LoggerOption] that sets the schema URL for a
// [Logger].
func WithSchemaURL(schemaURL string) LoggerOption {
	return loggerOptionFunc(func(config LoggerConfig) LoggerConfig {
		config.schemaURL = schemaURL
		return config
	})
}

// EnabledParameters represents the payload for [Logger.Enabled].
type EnabledParameters struct {
	Severity  Severity
	EventName string
}
