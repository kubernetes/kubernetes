/*
Copyright 2022 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"strings"

	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// MapList provides a "lookup by key" operation for lists (arrays) with x-kubernetes-list-type=map.
type MapList interface {
	// Get returns the first element having given key, for all
	// x-kubernetes-list-map-keys, to the provided object. If the provided object isn't itself a valid MapList element,
	// get returns nil.
	Get(interface{}) interface{}
}

type keyStrategy interface {
	// CompositeKeyFor returns a composite key for the provided object, if possible, and a
	// boolean that indicates whether or not a key could be generated for the provided object.
	CompositeKeyFor(value.Map) (interface{}, bool)
}

// singleKeyStrategy is a cheaper strategy for associative lists that have exactly one key.
type singleKeyStrategy struct {
	key string
}

// CompositeKeyFor directly returns the value of the single key  to
// use as a composite key.
func (ks *singleKeyStrategy) CompositeKeyFor(obj value.Map) (interface{}, bool) {
	v, ok := obj.Get(ks.key)
	if !ok {
		return nil, false
	}

	switch {
	case v.IsBool():
		fallthrough
	case v.IsFloat():
		fallthrough
	case v.IsInt():
		fallthrough
	case v.IsString():
		return v.Unstructured(), true
	default:
		return nil, false // non-scalar
	}
}

// multiKeyStrategy computes a composite key of all key values.
type multiKeyStrategy struct {
	sts Schema
}

// CompositeKeyFor returns a composite key computed from the values of all
// keys.
func (ks *multiKeyStrategy) CompositeKeyFor(obj value.Map) (interface{}, bool) {
	const keyDelimiter = "\x00" // 0 byte should never appear in the composite key except as delimiter

	var delimited strings.Builder
	for _, key := range ks.sts.XListMapKeys() {
		v, ok := obj.Get(key)
		if !ok {
			return nil, false
		}

		switch {
		case v.IsBool():
			fmt.Fprintf(&delimited, keyDelimiter+"%t", v.AsBool())
		case v.IsFloat():
			fmt.Fprintf(&delimited, keyDelimiter+"%f", v.AsFloat())
		case v.IsInt():
			fmt.Fprintf(&delimited, keyDelimiter+"%d", v.AsInt())
		case v.IsString():
			fmt.Fprintf(&delimited, keyDelimiter+"%q", v.AsString())
		default:
			return nil, false // values must be scalars
		}
	}
	return delimited.String(), true
}

// emptyMapList is a MapList containing no elements.
type emptyMapList struct{}

func (emptyMapList) Get(interface{}) interface{} {
	return nil
}

type mapListImpl struct {
	sts Schema
	ks  keyStrategy
	// keyedItems contains all lazily keyed map items
	keyedItems map[interface{}]interface{}
	// unkeyedItems contains all map items that have not yet been keyed
	unkeyedItems value.List
	unkeyedIndex int
}

func (a *mapListImpl) Get(obj interface{}) interface{} {
	var val value.Value
	if asVal, ok := obj.(value.Value); ok {
		val = asVal
	} else {
		val = value.NewValueInterface(obj)
	}

	if !val.IsMap() {
		return nil
	}

	mobj := val.AsMap()
	key, ok := a.ks.CompositeKeyFor(mobj)
	if !ok {
		return nil
	}
	if match, ok := a.keyedItems[key]; ok {
		return match
	}
	// keep keying items until we either find a match or run out of unkeyed items
	for a.unkeyedItems.Length() > a.unkeyedIndex {
		// dequeue an unkeyed item
		item := a.unkeyedItems.At(a.unkeyedIndex)
		a.unkeyedIndex += 1

		// key the item
		if !item.IsMap() {
			continue
		}
		mitem := item.AsMap()
		itemKey, ok := a.ks.CompositeKeyFor(mitem)
		if !ok {
			continue
		}
		if _, exists := a.keyedItems[itemKey]; !exists {
			a.keyedItems[itemKey] = mitem
		}

		// if it matches, short-circuit
		if itemKey == key {
			return item
		}
	}

	return nil
}

func makeKeyStrategy(sts Schema) keyStrategy {
	listMapKeys := sts.XListMapKeys()
	if len(listMapKeys) == 1 {
		key := listMapKeys[0]
		return &singleKeyStrategy{
			key: key,
		}
	}

	return &multiKeyStrategy{
		sts: sts,
	}
}

// MakeMapList returns a queryable interface over the provided x-kubernetes-list-type=map
// keyedItems. If the provided schema is _not_ an array with x-kubernetes-list-type=map, returns an
// empty mapList.
func MakeMapList(sts Schema, items []interface{}) (rv MapList) {
	if sts.Type() != "array" || sts.XListType() != "map" || len(sts.XListMapKeys()) == 0 || len(items) == 0 {
		return emptyMapList{}
	}
	ks := makeKeyStrategy(sts)
	return &mapListImpl{
		sts:          sts,
		ks:           ks,
		keyedItems:   map[interface{}]interface{}{},
		unkeyedItems: value.NewValueInterface(items).AsList(),
	}
}
