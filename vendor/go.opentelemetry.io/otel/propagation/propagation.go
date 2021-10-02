// Copyright The OpenTelemetry Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package propagation // import "go.opentelemetry.io/otel/propagation"

import (
	"context"
	"net/http"
)

// TextMapCarrier is the storage medium used by a TextMapPropagator.
type TextMapCarrier interface {
	// Get returns the value associated with the passed key.
	Get(key string) string
	// Set stores the key-value pair.
	Set(key string, value string)
	// Keys lists the keys stored in this carrier.
	Keys() []string
}

// HeaderCarrier adapts http.Header to satisfy the TextMapCarrier interface.
type HeaderCarrier http.Header

// Get returns the value associated with the passed key.
func (hc HeaderCarrier) Get(key string) string {
	return http.Header(hc).Get(key)
}

// Set stores the key-value pair.
func (hc HeaderCarrier) Set(key string, value string) {
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
	// Inject set cross-cutting concerns from the Context into the carrier.
	Inject(ctx context.Context, carrier TextMapCarrier)
	// Extract reads cross-cutting concerns from the carrier into a Context.
	Extract(ctx context.Context, carrier TextMapCarrier) context.Context
	// Fields returns the keys who's values are set with Inject.
	Fields() []string
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
