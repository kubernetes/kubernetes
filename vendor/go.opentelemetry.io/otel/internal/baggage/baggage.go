// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

/*
Package baggage provides base types and functionality to store and retrieve
baggage in Go context. This package exists because the OpenTracing bridge to
OpenTelemetry needs to synchronize state whenever baggage for a context is
modified and that context contains an OpenTracing span. If it were not for
this need this package would not need to exist and the
`go.opentelemetry.io/otel/baggage` package would be the singular place where
W3C baggage is handled.
*/
package baggage // import "go.opentelemetry.io/otel/internal/baggage"

// List is the collection of baggage members. The W3C allows for duplicates,
// but OpenTelemetry does not, therefore, this is represented as a map.
type List map[string]Item

// Item is the value and metadata properties part of a list-member.
type Item struct {
	Value      string
	Properties []Property
}

// Property is a metadata entry for a list-member.
type Property struct {
	Key, Value string

	// HasValue indicates if a zero-value value means the property does not
	// have a value or if it was the zero-value.
	HasValue bool
}
