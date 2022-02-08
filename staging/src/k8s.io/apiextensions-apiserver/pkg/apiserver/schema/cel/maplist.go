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
	"hash/maphash"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// mapList provides a "lookup by key" operation for lists (arrays) with x-kubernetes-list-type=map.
type mapList interface {
	// get returns the unique element having identical values, for all
	// x-kubernetes-list-map-keys, to the provided object. If no such unique element exists, or
	// if the provided object isn't itself a valid mapList element, get returns nil.
	get(interface{}) interface{}
}

type keyStrategy interface {
	// CompositeKeyFor returns a composite key for the provided object, if possible, and a
	// boolean that indicates whether or not a key could be generated for the provided object.
	CompositeKeyFor(map[string]interface{}) (interface{}, bool)
}

// singleKeyStrategy is a cheaper strategy for associative lists that have exactly one key.
type singleKeyStrategy struct {
	key     string
	defawlt interface{} // default is a keyword
}

// CompositeKeyFor directly returns the value of the single key (or its default value, if absent) to
// use as a composite key.
func (ks *singleKeyStrategy) CompositeKeyFor(obj map[string]interface{}) (interface{}, bool) {
	v, ok := obj[ks.key]
	if !ok {
		v = ks.defawlt // substitute default value
	}

	switch v.(type) {
	case bool, float64, int64, string:
		return v, true
	default:
		return nil, false // non-scalar
	}
}

// hashKeyStrategy computes a hash of all key values.
type hashKeyStrategy struct {
	sts    *schema.Structural
	hasher maphash.Hash
}

// CompositeKeyFor returns a hash computed from the values (or default values, if absent) of all
// keys.
func (ks *hashKeyStrategy) CompositeKeyFor(obj map[string]interface{}) (interface{}, bool) {
	const keyDelimiter = "\x00" // 0 byte should never appear in the hash input except as delimiter

	ks.hasher.Reset()
	for _, key := range ks.sts.XListMapKeys {
		v, ok := obj[key]
		if !ok {
			v = ks.sts.Properties[key].Default.Object
		}

		switch v.(type) {
		case bool:
			fmt.Fprintf(&ks.hasher, keyDelimiter+"%t", v)
		case float64:
			fmt.Fprintf(&ks.hasher, keyDelimiter+"%f", v)
		case int64:
			fmt.Fprintf(&ks.hasher, keyDelimiter+"%d", v)
		case string:
			fmt.Fprintf(&ks.hasher, keyDelimiter+"%q", v)
		default:
			return nil, false // values must be scalars
		}
	}
	return ks.hasher.Sum64(), true
}

// emptyMapList is a mapList containing no elements.
type emptyMapList struct{}

func (emptyMapList) get(interface{}) interface{} {
	return nil
}

type mapListImpl struct {
	sts      *schema.Structural
	ks       keyStrategy
	elements map[interface{}][]interface{} // composite key -> bucket
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

	// Scan bucket to handle key collisions and duplicate key sets:
	var match interface{}
	for _, element := range a.elements[key] {
		all := true
		for _, key := range a.sts.XListMapKeys {
			va, ok := element.(map[string]interface{})[key]
			if !ok {
				va = a.sts.Properties[key].Default.Object
			}

			vb, ok := mobj[key]
			if !ok {
				vb = a.sts.Properties[key].Default.Object
			}

			all = all && (va == vb)
		}

		if !all {
			continue
		}

		if match != nil {
			// Duplicate key set / more than one element matches. This condition should
			// have generated a validation error elsewhere.
			return nil
		}
		match = element
	}
	return match // can be nil
}

func makeKeyStrategy(sts *schema.Structural) keyStrategy {
	if len(sts.XListMapKeys) == 1 {
		key := sts.XListMapKeys[0]
		return &singleKeyStrategy{
			key:     key,
			defawlt: sts.Properties[key].Default.Object,
		}
	}

	return &hashKeyStrategy{
		sts: sts,
	}
}

// makeMapList returns a queryable interface over the provided x-kubernetes-list-type=map
// elements. If the provided schema is _not_ an array with x-kubernetes-list-type=map, returns an
// empty mapList.
func makeMapList(sts *schema.Structural, ks keyStrategy, items []interface{}) (rv mapList) {
	if sts.Type != "array" || sts.XListType == nil || *sts.XListType != "map" || len(sts.XListMapKeys) == 0 || len(items) == 0 {
		return emptyMapList{}
	}

	elements := make(map[interface{}][]interface{}, len(items))

	for _, item := range items {
		mitem, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if key, ok := ks.CompositeKeyFor(mitem); ok {
			elements[key] = append(elements[key], mitem)
		}
	}

	return &mapListImpl{
		sts:      sts,
		ks:       ks,
		elements: elements,
	}
}
