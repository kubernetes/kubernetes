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

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"encoding/json"
	"reflect"
	"sort"
	"sync"
)

type (
	// Set is the representation for a distinct attribute set. It manages an
	// immutable set of attributes, with an internal cache for storing
	// attribute encodings.
	//
	// This type supports the Equivalent method of comparison using values of
	// type Distinct.
	Set struct {
		equivalent Distinct
	}

	// Distinct wraps a variable-size array of KeyValue, constructed with keys
	// in sorted order. This can be used as a map key or for equality checking
	// between Sets.
	Distinct struct {
		iface interface{}
	}

	// Sortable implements sort.Interface, used for sorting KeyValue. This is
	// an exported type to support a memory optimization. A pointer to one of
	// these is needed for the call to sort.Stable(), which the caller may
	// provide in order to avoid an allocation. See NewSetWithSortable().
	Sortable []KeyValue
)

var (
	// keyValueType is used in computeDistinctReflect.
	keyValueType = reflect.TypeOf(KeyValue{})

	// emptySet is returned for empty attribute sets.
	emptySet = &Set{
		equivalent: Distinct{
			iface: [0]KeyValue{},
		},
	}

	// sortables is a pool of Sortables used to create Sets with a user does
	// not provide one.
	sortables = sync.Pool{
		New: func() interface{} { return new(Sortable) },
	}
)

// EmptySet returns a reference to a Set with no elements.
//
// This is a convenience provided for optimized calling utility.
func EmptySet() *Set {
	return emptySet
}

// reflectValue abbreviates reflect.ValueOf(d).
func (d Distinct) reflectValue() reflect.Value {
	return reflect.ValueOf(d.iface)
}

// Valid returns true if this value refers to a valid Set.
func (d Distinct) Valid() bool {
	return d.iface != nil
}

// Len returns the number of attributes in this set.
func (l *Set) Len() int {
	if l == nil || !l.equivalent.Valid() {
		return 0
	}
	return l.equivalent.reflectValue().Len()
}

// Get returns the KeyValue at ordered position idx in this set.
func (l *Set) Get(idx int) (KeyValue, bool) {
	if l == nil || !l.equivalent.Valid() {
		return KeyValue{}, false
	}
	value := l.equivalent.reflectValue()

	if idx >= 0 && idx < value.Len() {
		// Note: The Go compiler successfully avoids an allocation for
		// the interface{} conversion here:
		return value.Index(idx).Interface().(KeyValue), true
	}

	return KeyValue{}, false
}

// Value returns the value of a specified key in this set.
func (l *Set) Value(k Key) (Value, bool) {
	if l == nil || !l.equivalent.Valid() {
		return Value{}, false
	}
	rValue := l.equivalent.reflectValue()
	vlen := rValue.Len()

	idx := sort.Search(vlen, func(idx int) bool {
		return rValue.Index(idx).Interface().(KeyValue).Key >= k
	})
	if idx >= vlen {
		return Value{}, false
	}
	keyValue := rValue.Index(idx).Interface().(KeyValue)
	if k == keyValue.Key {
		return keyValue.Value, true
	}
	return Value{}, false
}

// HasValue tests whether a key is defined in this set.
func (l *Set) HasValue(k Key) bool {
	if l == nil {
		return false
	}
	_, ok := l.Value(k)
	return ok
}

// Iter returns an iterator for visiting the attributes in this set.
func (l *Set) Iter() Iterator {
	return Iterator{
		storage: l,
		idx:     -1,
	}
}

// ToSlice returns the set of attributes belonging to this set, sorted, where
// keys appear no more than once.
func (l *Set) ToSlice() []KeyValue {
	iter := l.Iter()
	return iter.ToSlice()
}

// Equivalent returns a value that may be used as a map key. The Distinct type
// guarantees that the result will equal the equivalent. Distinct value of any
// attribute set with the same elements as this, where sets are made unique by
// choosing the last value in the input for any given key.
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

// Encoded returns the encoded form of this set, according to encoder.
func (l *Set) Encoded(encoder Encoder) string {
	if l == nil || encoder == nil {
		return ""
	}

	return encoder.Encode(l.Iter())
}

func empty() Set {
	return Set{
		equivalent: emptySet.equivalent,
	}
}

// NewSet returns a new Set. See the documentation for
// NewSetWithSortableFiltered for more details.
//
// Except for empty sets, this method adds an additional allocation compared
// with calls that include a Sortable.
func NewSet(kvs ...KeyValue) Set {
	// Check for empty set.
	if len(kvs) == 0 {
		return empty()
	}
	srt := sortables.Get().(*Sortable)
	s, _ := NewSetWithSortableFiltered(kvs, srt, nil)
	sortables.Put(srt)
	return s
}

// NewSetWithSortable returns a new Set. See the documentation for
// NewSetWithSortableFiltered for more details.
//
// This call includes a Sortable option as a memory optimization.
func NewSetWithSortable(kvs []KeyValue, tmp *Sortable) Set {
	// Check for empty set.
	if len(kvs) == 0 {
		return empty()
	}
	s, _ := NewSetWithSortableFiltered(kvs, tmp, nil)
	return s
}

// NewSetWithFiltered returns a new Set. See the documentation for
// NewSetWithSortableFiltered for more details.
//
// This call includes a Filter to include/exclude attribute keys from the
// return value. Excluded keys are returned as a slice of attribute values.
func NewSetWithFiltered(kvs []KeyValue, filter Filter) (Set, []KeyValue) {
	// Check for empty set.
	if len(kvs) == 0 {
		return empty(), nil
	}
	srt := sortables.Get().(*Sortable)
	s, filtered := NewSetWithSortableFiltered(kvs, srt, filter)
	sortables.Put(srt)
	return s, filtered
}

// NewSetWithSortableFiltered returns a new Set.
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
// Note that methods are defined on Set, although this returns Set. Callers
// can avoid memory allocations by:
//
// - allocating a Sortable for use as a temporary in this method
// - allocating a Set for storing the return value of this constructor.
//
// The result maintains a cache of encoded attributes, by attribute.EncoderID.
// This value should not be copied after its first use.
//
// The second []KeyValue return value is a list of attributes that were
// excluded by the Filter (if non-nil).
func NewSetWithSortableFiltered(kvs []KeyValue, tmp *Sortable, filter Filter) (Set, []KeyValue) {
	// Check for empty set.
	if len(kvs) == 0 {
		return empty(), nil
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
		position--
		kvs[offset], kvs[position] = kvs[position], kvs[offset]
	}
	if filter != nil {
		return filterSet(kvs[position:], filter)
	}
	return Set{
		equivalent: computeDistinct(kvs[position:]),
	}, nil
}

// filterSet reorders kvs so that included keys are contiguous at the end of
// the slice, while excluded keys precede the included keys.
func filterSet(kvs []KeyValue, filter Filter) (Set, []KeyValue) {
	var excluded []KeyValue

	// Move attributes that do not match the filter so they're adjacent before
	// calling computeDistinct().
	distinctPosition := len(kvs)

	// Swap indistinct keys forward and distinct keys toward the
	// end of the slice.
	offset := len(kvs) - 1
	for ; offset >= 0; offset-- {
		if filter(kvs[offset]) {
			distinctPosition--
			kvs[offset], kvs[distinctPosition] = kvs[distinctPosition], kvs[offset]
			continue
		}
	}
	excluded = kvs[:distinctPosition]

	return Set{
		equivalent: computeDistinct(kvs[distinctPosition:]),
	}, excluded
}

// Filter returns a filtered copy of this Set. See the documentation for
// NewSetWithSortableFiltered for more details.
func (l *Set) Filter(re Filter) (Set, []KeyValue) {
	if re == nil {
		return Set{
			equivalent: l.equivalent,
		}, nil
	}

	// Note: This could be refactored to avoid the temporary slice
	// allocation, if it proves to be expensive.
	return filterSet(l.ToSlice(), re)
}

// computeDistinct returns a Distinct using either the fixed- or
// reflect-oriented code path, depending on the size of the input. The input
// slice is assumed to already be sorted and de-duplicated.
func computeDistinct(kvs []KeyValue) Distinct {
	iface := computeDistinctFixed(kvs)
	if iface == nil {
		iface = computeDistinctReflect(kvs)
	}
	return Distinct{
		iface: iface,
	}
}

// computeDistinctFixed computes a Distinct for small slices. It returns nil
// if the input is too large for this code path.
func computeDistinctFixed(kvs []KeyValue) interface{} {
	switch len(kvs) {
	case 1:
		ptr := new([1]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 2:
		ptr := new([2]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 3:
		ptr := new([3]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 4:
		ptr := new([4]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 5:
		ptr := new([5]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 6:
		ptr := new([6]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 7:
		ptr := new([7]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 8:
		ptr := new([8]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 9:
		ptr := new([9]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	case 10:
		ptr := new([10]KeyValue)
		copy((*ptr)[:], kvs)
		return *ptr
	default:
		return nil
	}
}

// computeDistinctReflect computes a Distinct using reflection, works for any
// size input.
func computeDistinctReflect(kvs []KeyValue) interface{} {
	at := reflect.New(reflect.ArrayOf(len(kvs), keyValueType)).Elem()
	for i, keyValue := range kvs {
		*(at.Index(i).Addr().Interface().(*KeyValue)) = keyValue
	}
	return at.Interface()
}

// MarshalJSON returns the JSON encoding of the Set.
func (l *Set) MarshalJSON() ([]byte, error) {
	return json.Marshal(l.equivalent.iface)
}

// MarshalLog is the marshaling function used by the logging system to represent this exporter.
func (l Set) MarshalLog() interface{} {
	kvs := make(map[string]string)
	for _, kv := range l.ToSlice() {
		kvs[string(kv.Key)] = kv.Value.Emit()
	}
	return kvs
}

// Len implements sort.Interface.
func (l *Sortable) Len() int {
	return len(*l)
}

// Swap implements sort.Interface.
func (l *Sortable) Swap(i, j int) {
	(*l)[i], (*l)[j] = (*l)[j], (*l)[i]
}

// Less implements sort.Interface.
func (l *Sortable) Less(i, j int) bool {
	return (*l)[i].Key < (*l)[j].Key
}
