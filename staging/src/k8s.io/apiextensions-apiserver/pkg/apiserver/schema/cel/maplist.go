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

package cel

import (
	"fmt"
	"strings"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// mapList provides a "lookup by key" operation for lists (arrays) with x-kubernetes-list-type=map.
type mapList interface {
	// get returns the first element having given key, for all
	// x-kubernetes-list-map-keys, to the provided object. If the provided object isn't itself a valid mapList element,
	// get returns nil.
	get(interface{}) interface{}
}

type keyStrategy interface {
	// CompositeKeyFor returns a composite key for the provided object, if possible, and a
	// boolean that indicates whether or not a key could be generated for the provided object.
	CompositeKeyFor(map[string]interface{}) (interface{}, bool)
}

// singleKeyStrategy is a cheaper strategy for associative lists that have exactly one key.
type singleKeyStrategy struct {
	key string
}

// CompositeKeyFor directly returns the value of the single key  to
// use as a composite key.
func (ks *singleKeyStrategy) CompositeKeyFor(obj map[string]interface{}) (interface{}, bool) {
	v, ok := obj[ks.key]
	if !ok {
		return nil, false
	}

	switch v.(type) {
	case bool, float64, int64, string:
		return v, true
	default:
		return nil, false // non-scalar
	}
}

// multiKeyStrategy computes a composite key of all key values.
type multiKeyStrategy struct {
	sts *schema.Structural
}

// CompositeKeyFor returns a composite key computed from the values of all
// keys.
func (ks *multiKeyStrategy) CompositeKeyFor(obj map[string]interface{}) (interface{}, bool) {
	const keyDelimiter = "\x00" // 0 byte should never appear in the composite key except as delimiter

	var delimited strings.Builder
	for _, key := range ks.sts.XListMapKeys {
		v, ok := obj[key]
		if !ok {
			return nil, false
		}

		switch v.(type) {
		case bool:
			fmt.Fprintf(&delimited, keyDelimiter+"%t", v)
		case float64:
			fmt.Fprintf(&delimited, keyDelimiter+"%f", v)
		case int64:
			fmt.Fprintf(&delimited, keyDelimiter+"%d", v)
		case string:
			fmt.Fprintf(&delimited, keyDelimiter+"%q", v)
		default:
			return nil, false // values must be scalars
		}
	}
	return delimited.String(), true
}

// emptyMapList is a mapList containing no elements.
type emptyMapList struct{}

func (emptyMapList) get(interface{}) interface{} {
	return nil
}

type mapListImpl struct {
	sts *schema.Structural
	ks  keyStrategy
	// keyedItems contains all lazily keyed map items
	keyedItems map[interface{}]interface{}
	// unkeyedItems contains all map items that have not yet been keyed
	unkeyedItems []interface{}
}

func (a *mapListImpl) get(obj interface{}) interface{} {
	mobj, ok := obj.(map[string]interface{})
	if !ok {
		return nil
	}

	key, ok := a.ks.CompositeKeyFor(mobj)
	if !ok {
		return nil
	}
	if match, ok := a.keyedItems[key]; ok {
		return match
	}
	// keep keying items until we either find a match or run out of unkeyed items
	for len(a.unkeyedItems) > 0 {
		// dequeue an unkeyed item
		item := a.unkeyedItems[0]
		a.unkeyedItems = a.unkeyedItems[1:]

		// key the item
		mitem, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		itemKey, ok := a.ks.CompositeKeyFor(mitem)
		if !ok {
			continue
		}
		if _, exists := a.keyedItems[itemKey]; !exists {
			a.keyedItems[itemKey] = mitem
		}

		// if it matches, short-circuit
		if itemKey == key {
			return mitem
		}
	}

	return nil
}

func makeKeyStrategy(sts *schema.Structural) keyStrategy {
	if len(sts.XListMapKeys) == 1 {
		key := sts.XListMapKeys[0]
		return &singleKeyStrategy{
			key: key,
		}
	}

	return &multiKeyStrategy{
		sts: sts,
	}
}

// makeMapList returns a queryable interface over the provided x-kubernetes-list-type=map
// keyedItems. If the provided schema is _not_ an array with x-kubernetes-list-type=map, returns an
// empty mapList.
func makeMapList(sts *schema.Structural, items []interface{}) (rv mapList) {
	if sts.Type != "array" || sts.XListType == nil || *sts.XListType != "map" || len(sts.XListMapKeys) == 0 || len(items) == 0 {
		return emptyMapList{}
	}
	ks := makeKeyStrategy(sts)
	return &mapListImpl{
		sts:          sts,
		ks:           ks,
		keyedItems:   map[interface{}]interface{}{},
		unkeyedItems: items,
	}
}
