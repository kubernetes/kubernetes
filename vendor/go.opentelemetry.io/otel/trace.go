// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otel // import "go.opentelemetry.io/otel"

import (
	"go.opentelemetry.io/otel/internal/global"
	"go.opentelemetry.io/otel/trace"
)

// Tracer creates a named tracer that implements Tracer interface.
// If the name is an empty string then provider uses default name.
//
// This is short for GetTracerProvider().Tracer(name, opts...)
func Tracer(name string, opts ...trace.TracerOption) trace.Tracer {
	return GetTracerProvider().Tracer(name, opts...)
}

// GetTracerProvider returns the registered global trace provider.
// If none is registered then an instance of NoopTracerProvider is returned.
//
// Use the trace provider to create a named tracer. E.g.
//
//	tracer := otel.GetTracerProvider().Tracer("example.com/foo")
//
// or
//
//	tracer := otel.Tracer("example.com/foo")
func GetTracerProvider() trace.TracerProvider {
	return global.TracerProvider()
}

// SetTracerProvider registers `tp` as the global trace provider.
func SetTracerProvider(tp trace.TracerProvider) {
	global.SetTracerProvider(tp)
}
