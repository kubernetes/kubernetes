// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel // import "go.opentelemetry.io/otel"

import (
	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/metric"
)

// Meter returns a Meter from the global MeterProvider. The name must be the
// name of the library providing instrumentation. This name may be the same as
// the instrumented code only if that code provides built-in instrumentation.
// If the name is empty, then an implementation defined default name will be
// used instead.
//
// If this is called before a global MeterProvider is registered the returned
// Meter will be a No-op implementation of a Meter. When a global MeterProvider
// is registered for the first time, the returned Meter, and all the
// instruments it has created or will create, are recreated automatically from
// the new MeterProvider.
//
// This is short for GetMeterProvider().Meter(name).
func Meter(name string, opts ...metric.MeterOption) metric.Meter {
	return GetMeterProvider().Meter(name, opts...)
}

// GetMeterProvider returns the registered global meter provider.
//
// If no global GetMeterProvider has been registered, a No-op GetMeterProvider
// implementation is returned. When a global GetMeterProvider is registered for
// the first time, the returned GetMeterProvider, and all the Meters it has
// created or will create, are recreated automatically from the new
// GetMeterProvider.
func GetMeterProvider() metric.MeterProvider {
	return global.MeterProvider()
}

// SetMeterProvider registers mp as the global MeterProvider.
func SetMeterProvider(mp metric.MeterProvider) {
	global.SetMeterProvider(mp)
}
