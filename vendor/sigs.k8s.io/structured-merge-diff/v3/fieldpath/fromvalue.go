/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package fieldpath

import (
	"sigs.k8s.io/structured-merge-diff/v3/value"
)

// SetFromValue creates a set containing every leaf field mentioned in v.
func SetFromValue(v value.Value) *Set {
	s := NewSet()

	w := objectWalker{
		path:      Path{},
		value:     v,
		allocator: value.NewFreelistAllocator(),
		do:        func(p Path) { s.Insert(p) },
	}

	w.walk()
	return s
}

type objectWalker struct {
	path      Path
	value     value.Value
	allocator value.Allocator

	do func(Path)
}

func (w *objectWalker) walk() {
	switch {
	case w.value.IsNull():
	case w.value.IsFloat():
	case w.value.IsInt():
	case w.value.IsString():
	case w.value.IsBool():
		// All leaf fields handled the same way (after the switch
		// statement).

	// Descend
	case w.value.IsList():
		// If the list were atomic, we'd break here, but we don't have
		// a schema, so we can't tell.
		l := w.value.AsListUsing(w.allocator)
		defer w.allocator.Free(l)
		iter := l.RangeUsing(w.allocator)
		defer w.allocator.Free(iter)
		for iter.Next() {
			i, value := iter.Item()
			w2 := *w
			w2.path = append(w.path, w.GuessBestListPathElement(i, value))
			w2.value = value
			w2.walk()
		}
		return
	case w.value.IsMap():
		// If the map/struct were atomic, we'd break here, but we don't
		// have a schema, so we can't tell.

		m := w.value.AsMapUsing(w.allocator)
		defer w.allocator.Free(m)
		m.IterateUsing(w.allocator, func(k string, val value.Value) bool {
			w2 := *w
			w2.path = append(w.path, PathElement{FieldName: &k})
			w2.value = val
			w2.walk()
			return true
		})
		return
	}

	// Leaf fields get added to the set.
	if len(w.path) > 0 {
		w.do(w.path)
	}
}

// AssociativeListCandidateFieldNames lists the field names which are
// considered keys if found in a list element.
var AssociativeListCandidateFieldNames = []string{
	"key",
	"id",
	"name",
}

// GuessBestListPathElement guesses whether item is an associative list
// element, which should be referenced by key(s), or if it is not and therefore
// referencing by index is acceptable. Currently this is done by checking
// whether item has any of the fields listed in
// AssociativeListCandidateFieldNames which have scalar values.
func (w *objectWalker) GuessBestListPathElement(index int, item value.Value) PathElement {
	if !item.IsMap() {
		// Non map items could be parts of sets or regular "atomic"
		// lists. We won't try to guess whether something should be a
		// set or not.
		return PathElement{Index: &index}
	}

	m := item.AsMapUsing(w.allocator)
	defer w.allocator.Free(m)
	var keys value.FieldList
	for _, name := range AssociativeListCandidateFieldNames {
		f, ok := m.Get(name)
		if !ok {
			continue
		}
		// only accept primitive/scalar types as keys.
		if f.IsNull() || f.IsMap() || f.IsList() {
			continue
		}
		keys = append(keys, value.Field{Name: name, Value: f})
	}
	if len(keys) > 0 {
		keys.Sort()
		return PathElement{Key: &keys}
	}
	return PathElement{Index: &index}
}
