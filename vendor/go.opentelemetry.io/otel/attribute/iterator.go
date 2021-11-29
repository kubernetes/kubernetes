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

// Iterator allows iterating over the set of labels in order,
// sorted by key.
type Iterator struct {
	storage *Set
	idx     int
}

// MergeIterator supports iterating over two sets of labels while
// eliminating duplicate values from the combined set.  The first
// iterator value takes precedence.
type MergeIterator struct {
	one     oneIterator
	two     oneIterator
	current KeyValue
}

type oneIterator struct {
	iter  Iterator
	done  bool
	label KeyValue
}

// Next moves the iterator to the next position. Returns false if there
// are no more labels.
func (i *Iterator) Next() bool {
	i.idx++
	return i.idx < i.Len()
}

// Label returns current KeyValue. Must be called only after Next returns
// true.
func (i *Iterator) Label() KeyValue {
	kv, _ := i.storage.Get(i.idx)
	return kv
}

// Attribute is a synonym for Label().
func (i *Iterator) Attribute() KeyValue {
	return i.Label()
}

// IndexedLabel returns current index and attribute. Must be called only
// after Next returns true.
func (i *Iterator) IndexedLabel() (int, KeyValue) {
	return i.idx, i.Label()
}

// Len returns a number of labels in the iterator's `*Set`.
func (i *Iterator) Len() int {
	return i.storage.Len()
}

// ToSlice is a convenience function that creates a slice of labels
// from the passed iterator. The iterator is set up to start from the
// beginning before creating the slice.
func (i *Iterator) ToSlice() []KeyValue {
	l := i.Len()
	if l == 0 {
		return nil
	}
	i.idx = -1
	slice := make([]KeyValue, 0, l)
	for i.Next() {
		slice = append(slice, i.Label())
	}
	return slice
}

// NewMergeIterator returns a MergeIterator for merging two label sets
// Duplicates are resolved by taking the value from the first set.
func NewMergeIterator(s1, s2 *Set) MergeIterator {
	mi := MergeIterator{
		one: makeOne(s1.Iter()),
		two: makeOne(s2.Iter()),
	}
	return mi
}

func makeOne(iter Iterator) oneIterator {
	oi := oneIterator{
		iter: iter,
	}
	oi.advance()
	return oi
}

func (oi *oneIterator) advance() {
	if oi.done = !oi.iter.Next(); !oi.done {
		oi.label = oi.iter.Label()
	}
}

// Next returns true if there is another label available.
func (m *MergeIterator) Next() bool {
	if m.one.done && m.two.done {
		return false
	}
	if m.one.done {
		m.current = m.two.label
		m.two.advance()
		return true
	}
	if m.two.done {
		m.current = m.one.label
		m.one.advance()
		return true
	}
	if m.one.label.Key == m.two.label.Key {
		m.current = m.one.label // first iterator label value wins
		m.one.advance()
		m.two.advance()
		return true
	}
	if m.one.label.Key < m.two.label.Key {
		m.current = m.one.label
		m.one.advance()
		return true
	}
	m.current = m.two.label
	m.two.advance()
	return true
}

// Label returns the current value after Next() returns true.
func (m *MergeIterator) Label() KeyValue {
	return m.current
}
