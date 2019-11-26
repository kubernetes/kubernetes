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
	"sigs.k8s.io/structured-merge-diff/value"
)

// SetFromValue creates a set containing every leaf field mentioned in v.
func SetFromValue(v value.Value) *Set {
	s := NewSet()

	w := objectWalker{
		path:  Path{},
		value: v,
		do:    func(p Path) { s.Insert(p) },
	}

	w.walk()
	return s
}

type objectWalker struct {
	path  Path
	value value.Value

	do func(Path)
}

func (w *objectWalker) walk() {
	switch {
	case w.value.Null:
	case w.value.FloatValue != nil:
	case w.value.IntValue != nil:
	case w.value.StringValue != nil:
	case w.value.BooleanValue != nil:
		// All leaf fields handled the same way (after the switch
		// statement).

	// Descend
	case w.value.ListValue != nil:
		// If the list were atomic, we'd break here, but we don't have
		// a schema, so we can't tell.

		for i, child := range w.value.ListValue.Items {
			w2 := *w
			w2.path = append(w.path, GuessBestListPathElement(i, child))
			w2.value = child
			w2.walk()
		}
		return
	case w.value.MapValue != nil:
		// If the map/struct were atomic, we'd break here, but we don't
		// have a schema, so we can't tell.

		for i := range w.value.MapValue.Items {
			child := w.value.MapValue.Items[i]
			w2 := *w
			w2.path = append(w.path, PathElement{FieldName: &child.Name})
			w2.value = child.Value
			w2.walk()
		}
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
func GuessBestListPathElement(index int, item value.Value) PathElement {
	if item.MapValue == nil {
		// Non map items could be parts of sets or regular "atomic"
		// lists. We won't try to guess whether something should be a
		// set or not.
		return PathElement{Index: &index}
	}

	var keys value.FieldList
	for _, name := range AssociativeListCandidateFieldNames {
		f, ok := item.MapValue.Get(name)
		if !ok {
			continue
		}
		// only accept primitive/scalar types as keys.
		if f.Value.Null || f.Value.MapValue != nil || f.Value.ListValue != nil {
			continue
		}
		keys = append(keys, *f)
	}
	if len(keys) > 0 {
		keys.Sort()
		return PathElement{Key: &keys}
	}
	return PathElement{Index: &index}
}
