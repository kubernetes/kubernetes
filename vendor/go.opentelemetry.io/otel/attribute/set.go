// Copyright The OpenTelemetry Authors
// SPDX-License-Identifier: Apache-2.0

package attribute // import "go.opentelemetry.io/otel/attribute"

import (
	"cmp"
	"encoding/json"
	"reflect"
	"slices"
	"sort"

	"go.opentelemetry.io/otel/attribute/internal/xxhash"
)

type (
	// Set is the representation for a distinct attribute set. It manages an
	// immutable set of attributes, with an internal cache for storing
	// attribute encodings.
	//
	// This type will remain comparable for backwards compatibility. The
	// equivalence of Sets across versions is not guaranteed to be stable.
	// Prior versions may find two Sets to be equal or not when compared
	// directly (i.e. ==), but subsequent versions may not. Users should use
	// the Equals method to ensure stable equivalence checking.
	//
	// Users should also use the Distinct returned from Equivalent as a map key
	// instead of a Set directly. Set has relatively poor performance when used
	// as a map key compared to Distinct.
	Set struct {
		hash uint64
		data any
	}

	// Distinct is an identifier of a Set which is very likely to be unique.
	//
	// Distinct should be used as a map key instead of a Set for to provide better
	// performance for map operations.
	Distinct struct {
		hash uint64
	}

	// Sortable implements sort.Interface, used for sorting KeyValue.
	//
	// Deprecated: This type is no longer used. It was added as a performance
	// optimization for Go < 1.21 that is no longer needed (Go < 1.21 is no
	// longer supported by the module).
	Sortable []KeyValue
)

// Compile time check these types remain comparable.
var (
	_ = isComparable(Set{})
	_ = isComparable(Distinct{})
)

func isComparable[T comparable](t T) T { return t }

var (
	// keyValueType is used in computeDistinctReflect.
	keyValueType = reflect.TypeOf(KeyValue{})

	// emptyHash is the hash of an empty set.
	emptyHash = xxhash.New().Sum64()

	// userDefinedEmptySet is an empty set. It was mistakenly exposed to users
	// as something they can assign to, so it must remain addressable and
	// mutable.
	//
	// This is kept for backwards compatibility, but should not be used in new code.
	userDefinedEmptySet = &Set{
		hash: emptyHash,
		data: [0]KeyValue{},
	}

	emptySet = Set{
		hash: emptyHash,
		data: [0]KeyValue{},
	}
)

// EmptySet returns a reference to a Set with no elements.
//
// This is a convenience provided for optimized calling utility.
func EmptySet() *Set {
	// Continue to return the pointer to the user-defined empty set for
	// backwards-compatibility.
	//
	// New code should not use this, instead use emptySet.
	return userDefinedEmptySet
}

// Valid reports whether this value refers to a valid Set.
func (d Distinct) Valid() bool { return d.hash != 0 }

// reflectValue abbreviates reflect.ValueOf(d).
func (l Set) reflectValue() reflect.Value {
	return reflect.ValueOf(l.data)
}

// Len returns the number of attributes in this set.
func (l *Set) Len() int {
	if l == nil || l.hash == 0 {
		return 0
	}
	return l.reflectValue().Len()
}

// Get returns the KeyValue at ordered position idx in this set.
func (l *Set) Get(idx int) (KeyValue, bool) {
	if l == nil || l.hash == 0 {
		return KeyValue{}, false
	}
	value := l.reflectValue()

	if idx >= 0 && idx < value.Len() {
		// Note: The Go compiler successfully avoids an allocation for
		// the interface{} conversion here:
		return value.Index(idx).Interface().(KeyValue), true
	}

	return KeyValue{}, false
}

// Value returns the value of a specified key in this set.
func (l *Set) Value(k Key) (Value, bool) {
	if l == nil || l.hash == 0 {
		return Value{}, false
	}
	rValue := l.reflectValue()
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

// HasValue reports whether a key is defined in this set.
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

// Equivalent returns a value that may be used as a map key. Equal Distinct
// values are very likely to be equivalent attribute Sets. Distinct value of any
// attribute set with the same elements as this, where sets are made unique by
// choosing the last value in the input for any given key.
func (l *Set) Equivalent() Distinct {
	if l == nil || l.hash == 0 {
		return Distinct{hash: emptySet.hash}
	}
	return Distinct{hash: l.hash}
}

// Equals reports whether the argument set is equivalent to this set.
func (l *Set) Equals(o *Set) bool {
	if l.Equivalent() != o.Equivalent() {
		return false
	}
	if l == nil || l.hash == 0 {
		l = &emptySet
	}
	if o == nil || o.hash == 0 {
		o = &emptySet
	}
	return l.data == o.data
}

// Encoded returns the encoded form of this set, according to encoder.
func (l *Set) Encoded(encoder Encoder) string {
	if l == nil || encoder == nil {
		return ""
	}

	return encoder.Encode(l.Iter())
}

// NewSet returns a new Set. See the documentation for
// NewSetWithSortableFiltered for more details.
//
// Except for empty sets, this method adds an additional allocation compared
// with calls that include a Sortable.
func NewSet(kvs ...KeyValue) Set {
	s, _ := NewSetWithFiltered(kvs, nil)
	return s
}

// NewSetWithSortable returns a new Set. See the documentation for
// NewSetWithSortableFiltered for more details.
//
// This call includes a Sortable option as a memory optimization.
//
// Deprecated: Use [NewSet] instead.
func NewSetWithSortable(kvs []KeyValue, _ *Sortable) Set {
	s, _ := NewSetWithFiltered(kvs, nil)
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
		return emptySet, nil
	}

	// Stable sort so the following de-duplication can implement
	// last-value-wins semantics.
	slices.SortStableFunc(kvs, func(a, b KeyValue) int {
		return cmp.Compare(a.Key, b.Key)
	})

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
	kvs = kvs[position:]

	if filter != nil {
		if div := filteredToFront(kvs, filter); div != 0 {
			return newSet(kvs[div:]), kvs[:div]
		}
	}
	return newSet(kvs), nil
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
//
// Deprecated: Use [NewSetWithFiltered] instead.
func NewSetWithSortableFiltered(kvs []KeyValue, _ *Sortable, filter Filter) (Set, []KeyValue) {
	return NewSetWithFiltered(kvs, filter)
}

// filteredToFront filters slice in-place using keep function. All KeyValues that need to
// be removed are moved to the front. All KeyValues that need to be kept are
// moved (in-order) to the back. The index for the first KeyValue to be kept is
// returned.
func filteredToFront(slice []KeyValue, keep Filter) int {
	n := len(slice)
	j := n
	for i := n - 1; i >= 0; i-- {
		if keep(slice[i]) {
			j--
			slice[i], slice[j] = slice[j], slice[i]
		}
	}
	return j
}

// Filter returns a filtered copy of this Set. See the documentation for
// NewSetWithSortableFiltered for more details.
func (l *Set) Filter(re Filter) (Set, []KeyValue) {
	if re == nil {
		return *l, nil
	}

	// Iterate in reverse to the first attribute that will be filtered out.
	n := l.Len()
	first := n - 1
	for ; first >= 0; first-- {
		kv, _ := l.Get(first)
		if !re(kv) {
			break
		}
	}

	// No attributes will be dropped, return the immutable Set l and nil.
	if first < 0 {
		return *l, nil
	}

	// Copy now that we know we need to return a modified set.
	//
	// Do not do this in-place on the underlying storage of *Set l. Sets are
	// immutable and filtering should not change this.
	slice := l.ToSlice()

	// Don't re-iterate the slice if only slice[0] is filtered.
	if first == 0 {
		// It is safe to assume len(slice) >= 1 given we found at least one
		// attribute above that needs to be filtered out.
		return newSet(slice[1:]), slice[:1]
	}

	// Move the filtered slice[first] to the front (preserving order).
	kv := slice[first]
	copy(slice[1:first+1], slice[:first])
	slice[0] = kv

	// Do not re-evaluate re(slice[first+1:]).
	div := filteredToFront(slice[1:first+1], re) + 1
	return newSet(slice[div:]), slice[:div]
}

// newSet returns a new set based on the sorted and uniqued kvs.
func newSet(kvs []KeyValue) Set {
	s := Set{
		hash: hashKVs(kvs),
		data: computeDataFixed(kvs),
	}
	if s.data == nil {
		s.data = computeDataReflect(kvs)
	}
	return s
}

// computeDataFixed computes a Set data for small slices. It returns nil if the
// input is too large for this code path.
func computeDataFixed(kvs []KeyValue) any {
	switch len(kvs) {
	case 1:
		return [1]KeyValue(kvs)
	case 2:
		return [2]KeyValue(kvs)
	case 3:
		return [3]KeyValue(kvs)
	case 4:
		return [4]KeyValue(kvs)
	case 5:
		return [5]KeyValue(kvs)
	case 6:
		return [6]KeyValue(kvs)
	case 7:
		return [7]KeyValue(kvs)
	case 8:
		return [8]KeyValue(kvs)
	case 9:
		return [9]KeyValue(kvs)
	case 10:
		return [10]KeyValue(kvs)
	default:
		return nil
	}
}

// computeDataReflect computes a Set data using reflection, works for any size
// input.
func computeDataReflect(kvs []KeyValue) any {
	at := reflect.New(reflect.ArrayOf(len(kvs), keyValueType)).Elem()
	for i, keyValue := range kvs {
		*(at.Index(i).Addr().Interface().(*KeyValue)) = keyValue
	}
	return at.Interface()
}

// MarshalJSON returns the JSON encoding of the Set.
func (l *Set) MarshalJSON() ([]byte, error) {
	return json.Marshal(l.data)
}

// MarshalLog is the marshaling function used by the logging system to represent this Set.
func (l Set) MarshalLog() any {
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
