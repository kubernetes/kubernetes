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

package resource // import "go.opentelemetry.io/otel/sdk/resource"

import (
	"context"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
)

// Resource describes an entity about which identifying information
// and metadata is exposed.  Resource is an immutable object,
// equivalent to a map from key to unique value.
//
// Resources should be passed and stored as pointers
// (`*resource.Resource`).  The `nil` value is equivalent to an empty
// Resource.
type Resource struct {
	attrs attribute.Set
}

var (
	emptyResource Resource

	defaultResource *Resource = func(r *Resource, err error) *Resource {
		if err != nil {
			otel.Handle(err)
		}
		return r
	}(Detect(context.Background(), defaultServiceNameDetector{}, FromEnv{}, TelemetrySDK{}))
)

// NewWithAttributes creates a resource from attrs. If attrs contains
// duplicate keys, the last value will be used. If attrs contains any invalid
// items those items will be dropped.
func NewWithAttributes(attrs ...attribute.KeyValue) *Resource {
	if len(attrs) == 0 {
		return &emptyResource
	}

	// Ensure attributes comply with the specification:
	// https://github.com/open-telemetry/opentelemetry-specification/blob/v1.0.1/specification/common/common.md#attributes
	s, _ := attribute.NewSetWithFiltered(attrs, func(kv attribute.KeyValue) bool {
		return kv.Valid()
	})

	// If attrs only contains invalid entries do not allocate a new resource.
	if s.Len() == 0 {
		return &emptyResource
	}

	return &Resource{s} //nolint
}

// String implements the Stringer interface and provides a
// human-readable form of the resource.
//
// Avoid using this representation as the key in a map of resources,
// use Equivalent() as the key instead.
func (r *Resource) String() string {
	if r == nil {
		return ""
	}
	return r.attrs.Encoded(attribute.DefaultEncoder())
}

// Attributes returns a copy of attributes from the resource in a sorted order.
// To avoid allocating a new slice, use an iterator.
func (r *Resource) Attributes() []attribute.KeyValue {
	if r == nil {
		r = Empty()
	}
	return r.attrs.ToSlice()
}

// Iter returns an interator of the Resource attributes.
// This is ideal to use if you do not want a copy of the attributes.
func (r *Resource) Iter() attribute.Iterator {
	if r == nil {
		r = Empty()
	}
	return r.attrs.Iter()
}

// Equal returns true when a Resource is equivalent to this Resource.
func (r *Resource) Equal(eq *Resource) bool {
	if r == nil {
		r = Empty()
	}
	if eq == nil {
		eq = Empty()
	}
	return r.Equivalent() == eq.Equivalent()
}

// Merge creates a new resource by combining resource a and b.
//
// If there are common keys between resource a and b, then the value
// from resource b will overwrite the value from resource a, even
// if resource b's value is empty.
func Merge(a, b *Resource) *Resource {
	if a == nil && b == nil {
		return Empty()
	}
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}

	// Note: 'b' attributes will overwrite 'a' with last-value-wins in attribute.Key()
	// Meaning this is equivalent to: append(a.Attributes(), b.Attributes()...)
	mi := attribute.NewMergeIterator(b.Set(), a.Set())
	combine := make([]attribute.KeyValue, 0, a.Len()+b.Len())
	for mi.Next() {
		combine = append(combine, mi.Label())
	}
	return NewWithAttributes(combine...)
}

// Empty returns an instance of Resource with no attributes.  It is
// equivalent to a `nil` Resource.
func Empty() *Resource {
	return &emptyResource
}

// Default returns an instance of Resource with a default
// "service.name" and OpenTelemetrySDK attributes
func Default() *Resource {
	return defaultResource
}

// Environment returns an instance of Resource with attributes
// extracted from the OTEL_RESOURCE_ATTRIBUTES environment variable.
func Environment() *Resource {
	detector := &FromEnv{}
	resource, err := detector.Detect(context.Background())
	if err == nil {
		otel.Handle(err)
	}
	return resource
}

// Equivalent returns an object that can be compared for equality
// between two resources.  This value is suitable for use as a key in
// a map.
func (r *Resource) Equivalent() attribute.Distinct {
	return r.Set().Equivalent()
}

// Set returns the equivalent *attribute.Set of this resources attributes.
func (r *Resource) Set() *attribute.Set {
	if r == nil {
		r = Empty()
	}
	return &r.attrs
}

// MarshalJSON encodes the resource attributes as a JSON list of { "Key":
// "...", "Value": ... } pairs in order sorted by key.
func (r *Resource) MarshalJSON() ([]byte, error) {
	if r == nil {
		r = Empty()
	}
	return r.attrs.MarshalJSON()
}

// Len returns the number of unique key-values in this Resource.
func (r *Resource) Len() int {
	if r == nil {
		return 0
	}
	return r.attrs.Len()
}

// Encoded returns an encoded representation of the resource.
func (r *Resource) Encoded(enc attribute.Encoder) string {
	if r == nil {
		return ""
	}
	return r.attrs.Encoded(enc)
}
