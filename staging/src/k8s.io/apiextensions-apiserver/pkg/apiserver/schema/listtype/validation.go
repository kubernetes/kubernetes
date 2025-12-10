/*
Copyright 2019 The Kubernetes Authors.

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

package listtype

import (
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"

	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// ValidateListSetsAndMaps validates that arrays with x-kubernetes-list-type "map" and "set" fulfill the uniqueness
// invariants for the keys (maps) and whole elements (sets).
func ValidateListSetsAndMaps(fldPath *field.Path, s *schema.Structural, obj map[string]interface{}) field.ErrorList {
	if s == nil || obj == nil {
		return nil
	}

	var errs field.ErrorList

	if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
		for k, v := range obj {
			errs = append(errs, validationListSetAndMaps(fldPath.Key(k), s.AdditionalProperties.Structural, v)...)
		}
	}
	if s.Properties != nil {
		for k, v := range obj {
			if sub, ok := s.Properties[k]; ok {
				errs = append(errs, validationListSetAndMaps(fldPath.Child(k), &sub, v)...)
			}
		}
	}

	return errs
}

func validationListSetAndMaps(fldPath *field.Path, s *schema.Structural, obj interface{}) field.ErrorList {
	switch obj := obj.(type) {
	case []interface{}:
		return validateListSetsAndMapsArray(fldPath, s, obj)
	case map[string]interface{}:
		return ValidateListSetsAndMaps(fldPath, s, obj)
	}
	return nil
}

func validateListSetsAndMapsArray(fldPath *field.Path, s *schema.Structural, obj []interface{}) field.ErrorList {
	var errs field.ErrorList

	if s.XListType != nil {
		switch *s.XListType {
		case "set":
			nonUnique, err := validateListSet(fldPath, obj)
			if err != nil {
				errs = append(errs, err)
			} else {
				for _, i := range nonUnique {
					errs = append(errs, field.Duplicate(fldPath.Index(i), obj[i]))
				}
			}
		case "map":
			errs = append(errs, validateListMap(fldPath, s, obj)...)
		}
		// if a case is ever added here then one should also be added to pkg/apiserver/schema/cel/values.go
	}

	if s.Items != nil {
		for i := range obj {
			errs = append(errs, validationListSetAndMaps(fldPath.Index(i), s.Items, obj[i])...)
		}
	}

	return errs
}

// validateListSet validated uniqueness of unstructured objects (scalar and compound) and
// returns the first non-unique appearance of items.
//
// As a special case to distinguish undefined key and null values, we allow unspecifiedKeyValue and nullObjectValue
// which are both handled like scalars with correct comparison by Golang.
func validateListSet(fldPath *field.Path, obj []interface{}) ([]int, *field.Error) {
	if len(obj) <= 1 {
		return nil, nil
	}

	seenScalars := make(map[interface{}]int, len(obj))
	seenCompounds := make(map[string]int, len(obj))
	var nonUniqueIndices []int
	for i, x := range obj {
		switch x.(type) {
		case map[string]interface{}, []interface{}:
			bs, err := json.Marshal(x)
			if err != nil {
				return nil, field.Invalid(fldPath.Index(i), x, "internal error")
			}
			s := string(bs)
			if times, seen := seenCompounds[s]; !seen {
				seenCompounds[s] = 1
			} else {
				seenCompounds[s]++
				if times == 1 {
					nonUniqueIndices = append(nonUniqueIndices, i)
				}
			}
		default:
			if times, seen := seenScalars[x]; !seen {
				seenScalars[x] = 1
			} else {
				seenScalars[x]++
				if times == 1 {
					nonUniqueIndices = append(nonUniqueIndices, i)
				}
			}
		}
	}

	return nonUniqueIndices, nil
}

func validateListMap(fldPath *field.Path, s *schema.Structural, obj []interface{}) field.ErrorList {
	// only allow nil and objects
	for i, x := range obj {
		if _, ok := x.(map[string]interface{}); x != nil && !ok {
			return field.ErrorList{field.Invalid(fldPath.Index(i), x, "must be an object for an array of list-type map")}
		}
	}

	if len(obj) <= 1 {
		return nil
	}

	// optimize simple case of one key
	if len(s.XListMapKeys) == 1 {
		type unspecifiedKeyValue struct{}

		keyField := s.XListMapKeys[0]
		keys := make([]interface{}, 0, len(obj))
		for _, x := range obj {
			if x == nil {
				keys = append(keys, unspecifiedKeyValue{}) // nil object means unspecified key
				continue
			}

			x := x.(map[string]interface{})

			// undefined key?
			key, ok := x[keyField]
			if !ok {
				keys = append(keys, unspecifiedKeyValue{})
				continue
			}

			keys = append(keys, key)
		}

		nonUnique, err := validateListSet(fldPath, keys)
		if err != nil {
			return field.ErrorList{err}
		}

		var errs field.ErrorList
		for _, i := range nonUnique {
			switch keys[i] {
			case unspecifiedKeyValue{}:
				errs = append(errs, field.Duplicate(fldPath.Index(i), map[string]interface{}{}))
			default:
				errs = append(errs, field.Duplicate(fldPath.Index(i), map[string]interface{}{keyField: keys[i]}))
			}
		}

		return errs
	}

	// multiple key fields
	keys := make([]interface{}, 0, len(obj))
	for _, x := range obj {
		key := map[string]interface{}{}
		if x == nil {
			keys = append(keys, key)
			continue
		}

		x := x.(map[string]interface{})

		for _, keyField := range s.XListMapKeys {
			if k, ok := x[keyField]; ok {
				key[keyField] = k
			}
		}

		keys = append(keys, key)
	}

	nonUnique, err := validateListSet(fldPath, keys)
	if err != nil {
		return field.ErrorList{err}
	}

	var errs field.ErrorList
	for _, i := range nonUnique {
		errs = append(errs, field.Duplicate(fldPath.Index(i), keys[i]))
	}

	return errs
}
