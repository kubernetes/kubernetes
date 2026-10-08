// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package otelhttp

import (
	"net/http"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

// Legacy HTTP attribute keys retained for compatibility.
const (
	// ReadBytesKey is the attribute key for the number of bytes returned by an
	// individual read from an HTTP request body.
	//
	// Deprecated: there is no direct semantic-convention replacement.
	ReadBytesKey = attribute.Key("http.read_bytes")

	// ReadErrorKey is the attribute key for the string form of a non-EOF error
	// returned while reading an HTTP request body.
	//
	// Deprecated: there is no direct semantic-convention replacement.
	ReadErrorKey = attribute.Key("http.read_error")

	// WroteBytesKey is the attribute key for the number of bytes returned by an
	// individual write to an HTTP response body.
	//
	// Deprecated: there is no direct semantic-convention replacement.
	WroteBytesKey = attribute.Key("http.wrote_bytes")

	// WriteErrorKey is the attribute key for the string form of a non-EOF error
	// returned while writing an HTTP response body.
	//
	// Deprecated: there is no direct semantic-convention replacement.
	WriteErrorKey = attribute.Key("http.write_error")
)

// Filter is a predicate used to determine whether a given http.request should
// be traced. A Filter must return true if the request should be traced.
type Filter func(*http.Request) bool

func newTracer(tp trace.TracerProvider) trace.Tracer {
	return tp.Tracer(ScopeName, trace.WithInstrumentationVersion(Version))
}
