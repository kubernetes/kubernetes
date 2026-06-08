// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"
	"net/http"
)

// TextMapCarrier is the storage medium used by a TextMapPropagator.
// See ValuesGetter for how a TextMapCarrier can get multiple values for a key.
type TextMapCarrier interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Get returns the value associated with the passed key.
	Get(key string) string
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Set stores the key-value pair.
	Set(key, value string)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Keys lists the keys stored in this carrier.
	Keys() []string
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

// ValuesGetter can return multiple values for a single key,
// with contrast to TextMapCarrier.Get which returns a single value.
type ValuesGetter interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Values returns all values associated with the passed key.
	Values(key string) []string
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

// MapCarrier is a TextMapCarrier that uses a map held in memory as a storage
// medium for propagated key-value pairs.
type MapCarrier map[string]string

// Compile time check that MapCarrier implements the TextMapCarrier.
var _ TextMapCarrier = MapCarrier{}

// Get returns the value associated with the passed key.
func (c MapCarrier) Get(key string) string {
	return c[key]
}

// Set stores the key-value pair.
func (c MapCarrier) Set(key, value string) {
	c[key] = value
}

// Keys lists the keys stored in this carrier.
func (c MapCarrier) Keys() []string {
	keys := make([]string, 0, len(c))
	for k := range c {
		keys = append(keys, k)
	}
	return keys
}

// HeaderCarrier adapts http.Header to satisfy the TextMapCarrier and ValuesGetter interfaces.
type HeaderCarrier http.Header

// Compile time check that HeaderCarrier implements ValuesGetter.
var _ TextMapCarrier = HeaderCarrier{}

// Compile time check that HeaderCarrier implements TextMapCarrier.
var _ ValuesGetter = HeaderCarrier{}

// Get returns the first value associated with the passed key.
func (hc HeaderCarrier) Get(key string) string {
	return http.Header(hc).Get(key)
}

// Values returns all values associated with the passed key.
func (hc HeaderCarrier) Values(key string) []string {
	return http.Header(hc).Values(key)
}

// Set stores the key-value pair.
func (hc HeaderCarrier) Set(key, value string) {
	http.Header(hc).Set(key, value)
}

// Keys lists the keys stored in this carrier.
func (hc HeaderCarrier) Keys() []string {
	keys := make([]string, 0, len(hc))
	for k := range hc {
		keys = append(keys, k)
	}
	return keys
}

// TextMapPropagator propagates cross-cutting concerns as key-value text
// pairs within a carrier that travels in-band across process boundaries.
type TextMapPropagator interface {
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Inject set cross-cutting concerns from the Context into the carrier.
	Inject(ctx context.Context, carrier TextMapCarrier)
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Extract reads cross-cutting concerns from the carrier into a Context.
	// Implementations may check if the carrier implements ValuesGetter,
	// to support extraction of multiple values per key.
	Extract(ctx context.Context, carrier TextMapCarrier) context.Context
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.

	// Fields returns the keys whose values are set with Inject.
	Fields() []string
	// DO NOT CHANGE: any modification will not be backwards compatible and
	// must never be done outside of a new major release.
}

type compositeTextMapPropagator []TextMapPropagator

func (p compositeTextMapPropagator) Inject(ctx context.Context, carrier TextMapCarrier) {
	for _, i := range p {
		i.Inject(ctx, carrier)
	}
}

func (p compositeTextMapPropagator) Extract(ctx context.Context, carrier TextMapCarrier) context.Context {
	for _, i := range p {
		ctx = i.Extract(ctx, carrier)
	}
	return ctx
}

func (p compositeTextMapPropagator) Fields() []string {
	unique := make(map[string]struct{})
	for _, i := range p {
		for _, k := range i.Fields() {
			unique[k] = struct{}{}
		}
	}

	fields := make([]string, 0, len(unique))
	for k := range unique {
		fields = append(fields, k)
	}
	return fields
}

// NewCompositeTextMapPropagator returns a unified TextMapPropagator from the
// group of passed TextMapPropagator. This allows different cross-cutting
// concerns to be propagates in a unified manner.
//
// The returned TextMapPropagator will inject and extract cross-cutting
// concerns in the order the TextMapPropagators were provided. Additionally,
// the Fields method will return a de-duplicated slice of the keys that are
// set with the Inject method.
func NewCompositeTextMapPropagator(p ...TextMapPropagator) TextMapPropagator {
	return compositeTextMapPropagator(p)
}
