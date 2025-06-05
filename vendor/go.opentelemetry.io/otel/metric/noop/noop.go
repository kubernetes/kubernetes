// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

// Package noop provides an implementation of the OpenTelemetry metric API that
// produces no telemetry and minimizes used computation resources.
//
// Using this package to implement the OpenTelemetry metric API will
// effectively disable OpenTelemetry.
//
// This implementation can be embedded in other implementations of the
// OpenTelemetry metric API. Doing so will mean the implementation defaults to
// no operation for methods it does not implement.
package noop // import "go.opentelemetry.io/otel/metric/noop"

import (
	"context"

	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/embedded"
)

var (
	// Compile-time check this implements the OpenTelemetry API.

	_ metric.MeterProvider                  = MeterProvider{}
	_ metric.Meter                          = Meter{}
	_ metric.Observer                       = Observer{}
	_ metric.Registration                   = Registration{}
	_ metric.Int64Counter                   = Int64Counter{}
	_ metric.Float64Counter                 = Float64Counter{}
	_ metric.Int64UpDownCounter             = Int64UpDownCounter{}
	_ metric.Float64UpDownCounter           = Float64UpDownCounter{}
	_ metric.Int64Histogram                 = Int64Histogram{}
	_ metric.Float64Histogram               = Float64Histogram{}
	_ metric.Int64Gauge                     = Int64Gauge{}
	_ metric.Float64Gauge                   = Float64Gauge{}
	_ metric.Int64ObservableCounter         = Int64ObservableCounter{}
	_ metric.Float64ObservableCounter       = Float64ObservableCounter{}
	_ metric.Int64ObservableGauge           = Int64ObservableGauge{}
	_ metric.Float64ObservableGauge         = Float64ObservableGauge{}
	_ metric.Int64ObservableUpDownCounter   = Int64ObservableUpDownCounter{}
	_ metric.Float64ObservableUpDownCounter = Float64ObservableUpDownCounter{}
	_ metric.Int64Observer                  = Int64Observer{}
	_ metric.Float64Observer                = Float64Observer{}
)

// MeterProvider is an OpenTelemetry No-Op MeterProvider.
type MeterProvider struct{ embedded.MeterProvider }

// NewMeterProvider returns a MeterProvider that does not record any telemetry.
func NewMeterProvider() MeterProvider {
	return MeterProvider{}
}

// Meter returns an OpenTelemetry Meter that does not record any telemetry.
func (MeterProvider) Meter(string, ...metric.MeterOption) metric.Meter {
	return Meter{}
}

// Meter is an OpenTelemetry No-Op Meter.
type Meter struct{ embedded.Meter }

// Int64Counter returns a Counter used to record int64 measurements that
// produces no telemetry.
func (Meter) Int64Counter(string, ...metric.Int64CounterOption) (metric.Int64Counter, error) {
	return Int64Counter{}, nil
}

// Int64UpDownCounter returns an UpDownCounter used to record int64
// measurements that produces no telemetry.
func (Meter) Int64UpDownCounter(string, ...metric.Int64UpDownCounterOption) (metric.Int64UpDownCounter, error) {
	return Int64UpDownCounter{}, nil
}

// Int64Histogram returns a Histogram used to record int64 measurements that
// produces no telemetry.
func (Meter) Int64Histogram(string, ...metric.Int64HistogramOption) (metric.Int64Histogram, error) {
	return Int64Histogram{}, nil
}

// Int64Gauge returns a Gauge used to record int64 measurements that
// produces no telemetry.
func (Meter) Int64Gauge(string, ...metric.Int64GaugeOption) (metric.Int64Gauge, error) {
	return Int64Gauge{}, nil
}

// Int64ObservableCounter returns an ObservableCounter used to record int64
// measurements that produces no telemetry.
func (Meter) Int64ObservableCounter(
	string,
	...metric.Int64ObservableCounterOption,
) (metric.Int64ObservableCounter, error) {
	return Int64ObservableCounter{}, nil
}

// Int64ObservableUpDownCounter returns an ObservableUpDownCounter used to
// record int64 measurements that produces no telemetry.
func (Meter) Int64ObservableUpDownCounter(
	string,
	...metric.Int64ObservableUpDownCounterOption,
) (metric.Int64ObservableUpDownCounter, error) {
	return Int64ObservableUpDownCounter{}, nil
}

// Int64ObservableGauge returns an ObservableGauge used to record int64
// measurements that produces no telemetry.
func (Meter) Int64ObservableGauge(string, ...metric.Int64ObservableGaugeOption) (metric.Int64ObservableGauge, error) {
	return Int64ObservableGauge{}, nil
}

// Float64Counter returns a Counter used to record int64 measurements that
// produces no telemetry.
func (Meter) Float64Counter(string, ...metric.Float64CounterOption) (metric.Float64Counter, error) {
	return Float64Counter{}, nil
}

// Float64UpDownCounter returns an UpDownCounter used to record int64
// measurements that produces no telemetry.
func (Meter) Float64UpDownCounter(string, ...metric.Float64UpDownCounterOption) (metric.Float64UpDownCounter, error) {
	return Float64UpDownCounter{}, nil
}

// Float64Histogram returns a Histogram used to record int64 measurements that
// produces no telemetry.
func (Meter) Float64Histogram(string, ...metric.Float64HistogramOption) (metric.Float64Histogram, error) {
	return Float64Histogram{}, nil
}

// Float64Gauge returns a Gauge used to record float64 measurements that
// produces no telemetry.
func (Meter) Float64Gauge(string, ...metric.Float64GaugeOption) (metric.Float64Gauge, error) {
	return Float64Gauge{}, nil
}

// Float64ObservableCounter returns an ObservableCounter used to record int64
// measurements that produces no telemetry.
func (Meter) Float64ObservableCounter(
	string,
	...metric.Float64ObservableCounterOption,
) (metric.Float64ObservableCounter, error) {
	return Float64ObservableCounter{}, nil
}

// Float64ObservableUpDownCounter returns an ObservableUpDownCounter used to
// record int64 measurements that produces no telemetry.
func (Meter) Float64ObservableUpDownCounter(
	string,
	...metric.Float64ObservableUpDownCounterOption,
) (metric.Float64ObservableUpDownCounter, error) {
	return Float64ObservableUpDownCounter{}, nil
}

// Float64ObservableGauge returns an ObservableGauge used to record int64
// measurements that produces no telemetry.
func (Meter) Float64ObservableGauge(
	string,
	...metric.Float64ObservableGaugeOption,
) (metric.Float64ObservableGauge, error) {
	return Float64ObservableGauge{}, nil
}

// RegisterCallback performs no operation.
func (Meter) RegisterCallback(metric.Callback, ...metric.Observable) (metric.Registration, error) {
	return Registration{}, nil
}

// Observer acts as a recorder of measurements for multiple instruments in a
// Callback, it performing no operation.
type Observer struct{ embedded.Observer }

// ObserveFloat64 performs no operation.
func (Observer) ObserveFloat64(metric.Float64Observable, float64, ...metric.ObserveOption) {
}

// ObserveInt64 performs no operation.
func (Observer) ObserveInt64(metric.Int64Observable, int64, ...metric.ObserveOption) {
}

// Registration is the registration of a Callback with a No-Op Meter.
type Registration struct{ embedded.Registration }

// Unregister unregisters the Callback the Registration represents with the
// No-Op Meter. This will always return nil because the No-Op Meter performs no
// operation, including hold any record of registrations.
func (Registration) Unregister() error { return nil }

// Int64Counter is an OpenTelemetry Counter used to record int64 measurements.
// It produces no telemetry.
type Int64Counter struct{ embedded.Int64Counter }

// Add performs no operation.
func (Int64Counter) Add(context.Context, int64, ...metric.AddOption) {}

// Float64Counter is an OpenTelemetry Counter used to record float64
// measurements. It produces no telemetry.
type Float64Counter struct{ embedded.Float64Counter }

// Add performs no operation.
func (Float64Counter) Add(context.Context, float64, ...metric.AddOption) {}

// Int64UpDownCounter is an OpenTelemetry UpDownCounter used to record int64
// measurements. It produces no telemetry.
type Int64UpDownCounter struct{ embedded.Int64UpDownCounter }

// Add performs no operation.
func (Int64UpDownCounter) Add(context.Context, int64, ...metric.AddOption) {}

// Float64UpDownCounter is an OpenTelemetry UpDownCounter used to record
// float64 measurements. It produces no telemetry.
type Float64UpDownCounter struct{ embedded.Float64UpDownCounter }

// Add performs no operation.
func (Float64UpDownCounter) Add(context.Context, float64, ...metric.AddOption) {}

// Int64Histogram is an OpenTelemetry Histogram used to record int64
// measurements. It produces no telemetry.
type Int64Histogram struct{ embedded.Int64Histogram }

// Record performs no operation.
func (Int64Histogram) Record(context.Context, int64, ...metric.RecordOption) {}

// Float64Histogram is an OpenTelemetry Histogram used to record float64
// measurements. It produces no telemetry.
type Float64Histogram struct{ embedded.Float64Histogram }

// Record performs no operation.
func (Float64Histogram) Record(context.Context, float64, ...metric.RecordOption) {}

// Int64Gauge is an OpenTelemetry Gauge used to record instantaneous int64
// measurements. It produces no telemetry.
type Int64Gauge struct{ embedded.Int64Gauge }

// Record performs no operation.
func (Int64Gauge) Record(context.Context, int64, ...metric.RecordOption) {}

// Float64Gauge is an OpenTelemetry Gauge used to record instantaneous float64
// measurements. It produces no telemetry.
type Float64Gauge struct{ embedded.Float64Gauge }

// Record performs no operation.
func (Float64Gauge) Record(context.Context, float64, ...metric.RecordOption) {}

// Int64ObservableCounter is an OpenTelemetry ObservableCounter used to record
// int64 measurements. It produces no telemetry.
type Int64ObservableCounter struct {
	metric.Int64Observable
	embedded.Int64ObservableCounter
}

// Float64ObservableCounter is an OpenTelemetry ObservableCounter used to record
// float64 measurements. It produces no telemetry.
type Float64ObservableCounter struct {
	metric.Float64Observable
	embedded.Float64ObservableCounter
}

// Int64ObservableGauge is an OpenTelemetry ObservableGauge used to record
// int64 measurements. It produces no telemetry.
type Int64ObservableGauge struct {
	metric.Int64Observable
	embedded.Int64ObservableGauge
}

// Float64ObservableGauge is an OpenTelemetry ObservableGauge used to record
// float64 measurements. It produces no telemetry.
type Float64ObservableGauge struct {
	metric.Float64Observable
	embedded.Float64ObservableGauge
}

// Int64ObservableUpDownCounter is an OpenTelemetry ObservableUpDownCounter
// used to record int64 measurements. It produces no telemetry.
type Int64ObservableUpDownCounter struct {
	metric.Int64Observable
	embedded.Int64ObservableUpDownCounter
}

// Float64ObservableUpDownCounter is an OpenTelemetry ObservableUpDownCounter
// used to record float64 measurements. It produces no telemetry.
type Float64ObservableUpDownCounter struct {
	metric.Float64Observable
	embedded.Float64ObservableUpDownCounter
}

// Int64Observer is a recorder of int64 measurements that performs no operation.
type Int64Observer struct{ embedded.Int64Observer }

// Observe performs no operation.
func (Int64Observer) Observe(int64, ...metric.ObserveOption) {}

// Float64Observer is a recorder of float64 measurements that performs no
// operation.
type Float64Observer struct{ embedded.Float64Observer }

// Observe performs no operation.
func (Float64Observer) Observe(float64, ...metric.ObserveOption) {}
