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
package baggage

// List is the collection of baggage members. The W3C allows for duplicates,
// but OpenTelemetry does not, therefore, this is represented as a map.
type List map[string]Item

// Item is the value and metadata or properties part of a list-member.
// Metadata and properties are mutually exclusive representations.
type Item struct {
	value      string
	metadata   string
	properties []Property
}

// NewItemWithMetadata returns an Item with opaque W3C metadata.
func NewItemWithMetadata(value, metadata string) Item {
	return Item{value: value, metadata: metadata}
}

// NewItemWithProperties returns an Item with materialized properties.
func NewItemWithProperties(value string, properties []Property) Item {
	return Item{value: value, properties: properties}
}

// Value returns the value of i.
func (i Item) Value() string { return i.value }

// Metadata returns the opaque W3C metadata of i. It is empty for an Item
// created with properties.
func (i Item) Metadata() string { return i.metadata }

// Properties returns the materialized properties of i. It is nil for an Item
// created with metadata.
func (i Item) Properties() []Property { return i.properties }

// Property is a metadata entry for a list-member.
type Property struct {
	Key, Value string

	// HasValue indicates if a zero-value value means the property does not
	// have a value or if it was the zero-value.
	HasValue bool
}
