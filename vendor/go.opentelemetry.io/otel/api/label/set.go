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

package label // import "go.opentelemetry.io/otel/api/label"

import (
	"encoding/json"
	"reflect"
	"sort"
	"sync"

	"go.opentelemetry.io/otel/api/core"
)

type (
	// Set is the representation for a distinct label set.  It
	// manages an immutable set of labels, with an internal cache
	// for storing label encodings.
	//
	// This type supports the `Equivalent` method of comparison
	// using values of type `Distinct`.
	//
	// This type is used to implement:
	// 1. Metric labels
	// 2. Resource sets
	// 3. Correlation map (TODO)
	Set struct {
		equivalent Distinct

		lock     sync.Mutex
		encoders [maxConcurrentEncoders]EncoderID
		encoded  [maxConcurrentEncoders]string
	}

	// Distinct wraps a variable-size array of `core.KeyValue`,
	// constructed with keys in sorted order.  This can be used as
	// a map key or for equality checking between Sets.
	Distinct struct {
		iface interface{}
	}

	// Sortable implements `sort.Interface`, used for sorting
	// `core.KeyValue`.  This is an exported type to support a
	// memory optimization.  A pointer to one of these is needed
	// for the call to `sort.Stable()`, which the caller may
	// provide in order to avoid an allocation.  See
	// `NewSetWithSortable()`.
	Sortable []core.KeyValue
)

var (
	// keyValueType is used in `computeDistinctReflect`.
	keyValueType = reflect.TypeOf(core.KeyValue{})

	// emptySet is returned for empty label sets.
	emptySet = &Set{
		equivalent: Distinct{
			iface: [0]core.KeyValue{},
		},
	}
)

const maxConcurrentEncoders = 3

func EmptySet() *Set {
	return emptySet
}

// reflect abbreviates `reflect.ValueOf`.
func (d Distinct) reflect() reflect.Value {
	return reflect.ValueOf(d.iface)
}

// Valid returns true if this value refers to a valid `*Set`.
func (d Distinct) Valid() bool {
	return d.iface != nil
}

// Len returns the number of labels in this set.
func (l *Set) Len() int {
	if l == nil || !l.equivalent.Valid() {
		return 0
	}
	return l.equivalent.reflect().Len()
}

// Get returns the KeyValue at ordered position `idx` in this set.
func (l *Set) Get(idx int) (core.KeyValue, bool) {
	if l == nil {
		return core.KeyValue{}, false
	}
	value := l.equivalent.reflect()

	if idx >= 0 && idx < value.Len() {
		// Note: The Go compiler successfully avoids an allocation for
		// the interface{} conversion here:
		return value.Index(idx).Interface().(core.KeyValue), true
	}

	return core.KeyValue{}, false
}

// Value returns the value of a specified key in this set.
func (l *Set) Value(k core.Key) (core.Value, bool) {
	if l == nil {
		return core.Value{}, false
	}
	value := l.equivalent.reflect()
	vlen := value.Len()

	idx := sort.Search(vlen, func(idx int) bool {
		return value.Index(idx).Interface().(core.KeyValue).Key >= k
	})
	if idx >= vlen {
		return core.Value{}, false
	}
	kv := value.Index(idx).Interface().(core.KeyValue)
	if k == kv.Key {
		return kv.Value, true
	}
	return core.Value{}, false
}

// HasValue tests whether a key is defined in this set.
func (l *Set) HasValue(k core.Key) bool {
	if l == nil {
		return false
	}
	_, ok := l.Value(k)
	return ok
}

// Iter returns an iterator for visiting the labels in this set.
func (l *Set) Iter() Iterator {
	return Iterator{
		storage: l,
		idx:     -1,
	}
}

// ToSlice returns the set of labels belonging to this set, sorted,
// where keys appear no more than once.
func (l *Set) ToSlice() []core.KeyValue {
	iter := l.Iter()
	return iter.ToSlice()
}

// Equivalent returns a value that may be used as a map key.  The
// Distinct type guarantees that the result will equal the equivalent
// Distinct value of any label set with the same elements as this,
// where sets are made unique by choosing the last value in the input
// for any given key.
func (l *Set) Equivalent() Distinct {
	if l == nil || !l.equivalent.Valid() {
		return emptySet.equivalent
	}
	return l.equivalent
}

// Equals returns true if the argument set is equivalent to this set.
func (l *Set) Equals(o *Set) bool {
	return l.Equivalent() == o.Equivalent()
}

// Encoded returns the encoded form of this set, according to
// `encoder`.  The result will be cached in this `*Set`.
func (l *Set) Encoded(encoder Encoder) string {
	if l == nil || encoder == nil {
		return ""
	}

	id := encoder.ID()
	if !id.Valid() {
		// Invalid IDs are not cached.
		return encoder.Encode(l.Iter())
	}

	var lookup *string
	l.lock.Lock()
	for idx := 0; idx < maxConcurrentEncoders; idx++ {
		if l.encoders[idx] == id {
			lookup = &l.encoded[idx]
			break
		}
	}
	l.lock.Unlock()

	if lookup != nil {
		return *lookup
	}

	r := encoder.Encode(l.Iter())

	l.lock.Lock()
	defer l.lock.Unlock()

	for idx := 0; idx < maxConcurrentEncoders; idx++ {
		if l.encoders[idx] == id {
			return l.encoded[idx]
		}
		if !l.encoders[idx].Valid() {
			l.encoders[idx] = id
			l.encoded[idx] = r
			return r
		}
	}

	// TODO: This is a performance cliff.  Find a way for this to
	// generate a warning.
	return r
}

// NewSet returns a new `*Set`.  See the documentation for
// `NewSetWithSortable` for more details.
//
// Except for empty sets, this method adds an additional allocation
// compared with a call to `NewSetWithSortable`.
func NewSet(kvs ...core.KeyValue) Set {
	// Check for empty set.
	if len(kvs) == 0 {
		return Set{
			equivalent: emptySet.equivalent,
		}
	}

	return NewSetWithSortable(kvs, new(Sortable))
}

// NewSetWithSortable returns a new `*Set`.
//
// Duplicate keys are eliminated by taking the last value.  This
// re-orders the input slice so that unique last-values are contiguous
// at the end of the slice.
//
// This ensures the following:
//
// - Last-value-wins semantics
// - Caller sees the reordering, but doesn't lose values
// - Repeated call preserve last-value wins.
//
// Note that methods are defined `*Set`, although no allocation for
// `Set` is required.  Callers can avoid memory allocations by:
//
// - allocating a `Sortable` for use as a temporary in this method
// - allocating a `Set` for storing the return value of this
//   constructor.
//
// The result maintains a cache of encoded labels, by label.EncoderID.
// This value should not be copied after its first use.
func NewSetWithSortable(kvs []core.KeyValue, tmp *Sortable) Set {
	// Check for empty set.
	if len(kvs) == 0 {
		return Set{
			equivalent: emptySet.equivalent,
		}
	}

	*tmp = kvs

	// Stable sort so the following de-duplication can implement
	// last-value-wins semantics.
	sort.Stable(tmp)

	*tmp = nil

	position := len(kvs) - 1
	offset := position - 1

	// The requirements stated above require that the stable
	// result be placed in the end of the input slice, while
	// overwritten values are swapped to the beginning.
	//
	// De-duplicate with last-value-wins semantics.  Preserve
	// duplicate values at the beginning of the input slice.
	for ; offset >= 0; offset-- {
		if kvs[offset].Key == kvs[position].Key {
			continue
		}
		kvs[offset], kvs[position-1] = kvs[position-1], kvs[offset]
		position--
	}

	return Set{
		equivalent: computeDistinct(kvs[position:]),
	}
}

// computeDistinct returns a `Distinct` using either the fixed- or
// reflect-oriented code path, depending on the size of the input.
// The input slice is assumed to already be sorted and de-duplicated.
func computeDistinct(kvs []core.KeyValue) Distinct {
	iface := computeDistinctFixed(kvs)
	if iface == nil {
		iface = computeDistinctReflect(kvs)
	}
	return Distinct{
		iface: iface,
	}
}

// computeDistinctFixed computes a `Distinct` for small slices.  It
// returns nil if the input is too large for this code path.
func computeDistinctFixed(kvs []core.KeyValue) interface{} {
	switch len(kvs) {
	case 1:
		ptr := new([1]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 2:
		ptr := new([2]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 3:
		ptr := new([3]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 4:
		ptr := new([4]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 5:
		ptr := new([5]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 6:
		ptr := new([6]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 7:
		ptr := new([7]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 8:
		ptr := new([8]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 9:
		ptr := new([9]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 10:
		ptr := new([10]core.KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	default:
		return nil
	}
}

// computeDistinctReflect computes a `Distinct` using reflection,
// works for any size input.
func computeDistinctReflect(kvs []core.KeyValue) interface{} {
	at := reflect.New(reflect.ArrayOf(len(kvs), keyValueType)).Elem()
	for i, kv := range kvs {
		*(at.Index(i).Addr().Interface().(*core.KeyValue)) = kv
	}
	return at.Interface()
}

// MarshalJSON returns the JSON encoding of the `*Set`.
func (l *Set) MarshalJSON() ([]byte, error) {
	return json.Marshal(l.equivalent.iface)
}

// Len implements `sort.Interface`.
func (l *Sortable) Len() int {
	return len(*l)
}

// Swap implements `sort.Interface`.
func (l *Sortable) Swap(i, j int) {
	(*l)[i], (*l)[j] = (*l)[j], (*l)[i]
}

// Less implements `sort.Interface`.
func (l *Sortable) Less(i, j int) bool {
	return (*l)[i].Key < (*l)[j].Key
}
