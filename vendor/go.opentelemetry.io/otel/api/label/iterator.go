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

package label

import (
	"go.opentelemetry.io/otel/api/core"
)

// Iterator allows iterating over the set of labels in order,
// sorted by key.
type Iterator struct {
	storage *Set
	idx     int
}

// Next moves the iterator to the next position. Returns false if there
// are no more labels.
func (i *Iterator) Next() bool {
	i.idx++
	return i.idx < i.Len()
}

// Label returns current core.KeyValue. Must be called only after Next returns
// true.
func (i *Iterator) Label() core.KeyValue {
	kv, _ := i.storage.Get(i.idx)
	return kv
}

// Attribute is a synonym for Label().
func (i *Iterator) Attribute() core.KeyValue {
	return i.Label()
}

// IndexedLabel returns current index and label. Must be called only
// after Next returns true.
func (i *Iterator) IndexedLabel() (int, core.KeyValue) {
	return i.idx, i.Label()
}

// IndexedAttribute is a synonym for IndexedLabel().
func (i *Iterator) IndexedAttribute() (int, core.KeyValue) {
	return i.IndexedLabel()
}

// Len returns a number of labels in the iterator's `*Set`.
func (i *Iterator) Len() int {
	return i.storage.Len()
}

// ToSlice is a convenience function that creates a slice of labels
// from the passed iterator. The iterator is set up to start from the
// beginning before creating the slice.
func (i *Iterator) ToSlice() []core.KeyValue {
	l := i.Len()
	if l == 0 {
		return nil
	}
	i.idx = -1
	slice := make([]core.KeyValue, 0, l)
	for i.Next() {
		slice = append(slice, i.Label())
	}
	return slice
}
