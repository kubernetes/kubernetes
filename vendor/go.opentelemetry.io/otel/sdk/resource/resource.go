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

// Package resource provides functionality for resource, which capture
// identifying information about the entities for which signals are exported.
package resource

import (
	"go.opentelemetry.io/otel/api/core"
	"go.opentelemetry.io/otel/api/label"
)

// Resource describes an entity about which identifying information
// and metadata is exposed.  Resource is an immutable object,
// equivalent to a map from key to unique value.
//
// Resources should be passed and stored as pointers
// (`*resource.Resource`).  The `nil` value is equivalent to an empty
// Resource.
type Resource struct {
	labels label.Set
}

var emptyResource Resource

// New creates a resource from a set of attributes.  If there are
// duplicate keys present in the list of attributes, then the last
// value found for the key is preserved.
func New(kvs ...core.KeyValue) *Resource {
	return &Resource{
		labels: label.NewSet(kvs...),
	}
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
	return r.labels.Encoded(label.DefaultEncoder())
}

// Attributes returns a copy of attributes from the resource in a sorted order.
// To avoid allocating a new slice, use an iterator.
func (r *Resource) Attributes() []core.KeyValue {
	if r == nil {
		r = Empty()
	}
	return r.labels.ToSlice()
}

// Iter returns an interator of the Resource attributes.
// This is ideal to use if you do not want a copy of the attributes.
func (r *Resource) Iter() label.Iterator {
	if r == nil {
		r = Empty()
	}
	return r.labels.Iter()
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
// from resource a is preserved.
func Merge(a, b *Resource) *Resource {
	if a == nil {
		a = Empty()
	}
	if b == nil {
		b = Empty()
	}
	// Note: 'b' is listed first so that 'a' will overwrite with
	// last-value-wins in label.New()
	combine := append(b.Attributes(), a.Attributes()...)
	return New(combine...)
}

// Empty returns an instance of Resource with no attributes.  It is
// equivalent to a `nil` Resource.
func Empty() *Resource {
	return &emptyResource
}

// Equivalent returns an object that can be compared for equality
// between two resources.  This value is suitable for use as a key in
// a map.
func (r *Resource) Equivalent() label.Distinct {
	if r == nil {
		r = Empty()
	}
	return r.labels.Equivalent()
}

// MarshalJSON encodes labels as a JSON list of { "Key": "...", "Value": ... }
// pairs in order sorted by key.
func (r *Resource) MarshalJSON() ([]byte, error) {
	if r == nil {
		r = Empty()
	}
	return r.labels.MarshalJSON()
}

// Len returns the number of unique key-values in this Resource.
func (r *Resource) Len() int {
	if r == nil {
		return 0
	}
	return r.labels.Len()
}

// Encoded returns an encoded representation of the resource by
// applying a label encoder.  The result is cached by the underlying
// label set.
func (r *Resource) Encoded(enc label.Encoder) string {
	if r == nil {
		return ""
	}
	return r.labels.Encoded(enc)
}
