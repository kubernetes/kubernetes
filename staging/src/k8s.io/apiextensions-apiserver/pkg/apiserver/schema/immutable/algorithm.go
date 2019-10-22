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

package immutable

import (
	"bytes"
	"encoding/json"
	"reflect"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// Immutable validates immutable values are respected or error otherwise
func Immutable(old interface{}, new interface{}, s *structuralschema.Structural, pth *field.Path) field.ErrorList {
	if s == nil {
		return nil
	}

	var allErrs field.ErrorList
	switch old := old.(type) {
	case map[string]interface{}:
		newObj, ok := new.(map[string]interface{})
		for k, prop := range s.Properties {
			if !ok && prop.XImmutability != nil {
				// e.g the CRD definition has changed
				allErrs = append(allErrs, field.Invalid(pth.Child(k), old[k], "is immutable"))
				continue
			}

			if prop.XImmutability != nil {
				if !isImmutable(*prop.XImmutability, old, newObj, k) {
					allErrs = append(allErrs, field.Invalid(pth.Child(k), newObj[k], "is immutable"))
				}
				continue
			}

			newMap, _ := newObj[k].(map[string]interface{})
			oldMap, _ := old[k].(map[string]interface{})
			if isMap(prop) && strOrEmpty(prop.AdditionalProperties.Structural.XImmutability) == structuralschema.XImmutabilityImmutable {
				if !isMapImmutableValues(oldMap, newMap) {
					allErrs = append(allErrs, field.Invalid(pth.Child(k), newObj[k], "is immutable"))
				}
				continue
			}

			if isMap(prop) && prop.XKeyImmutability != nil {
				if !isMapImmutableKeys(*prop.XKeyImmutability, oldMap, newMap) {
					allErrs = append(allErrs, field.Invalid(pth.Child(k), newObj[k], "is immutable"))
				}
				continue
			}

			allErrs = append(allErrs, Immutable(old[k], newObj[k], &prop, pth.Child(k))...)
		}
	case []interface{}:
		newArray, _ := new.([]interface{})
		if isListAtomic(*s) && strOrEmpty(s.Items.XImmutability) == structuralschema.XImmutabilityImmutable {
			if !isListAtomicImmutableItems(old, newArray) {
				return append(allErrs, field.Invalid(pth.Child(s.Title), old, "is immutable"))
			}
		}

		if isListMap(*s) && strOrEmpty(s.Items.XImmutability) == structuralschema.XImmutabilityImmutable {
			if is, err := isListMapImmutableItems(s.XListMapKeys, old, newArray); !is || err != nil {
				return append(allErrs, field.Invalid(pth.Child(s.Title), old, "is immutable"))
			}
		}

		if isListAtomic(*s) && s.XKeyImmutability != nil {
			if !isListAtomicImmutableKeys(*s.XKeyImmutability, old, newArray) {
				return append(allErrs, field.Invalid(pth.Child(s.Title), old, "is immutable"))
			}
		}

		if isListMap(*s) && s.XKeyImmutability != nil {
			if is, err := isListMapImmutableKeys(*s.XKeyImmutability, s.XListMapKeys, old, newArray); !is || err != nil {
				return append(allErrs, field.Invalid(pth.Child(s.Title), old, "is immutable"))
			}
		}

		if isListSet(*s) && s.XKeyImmutability != nil {
			if is, err := isListSetImmutableKeys(*s.XKeyImmutability, old, newArray); !is || err != nil {
				return append(allErrs, field.Invalid(pth.Child(s.Title), old, "is immutable"))
			}
		}
	default:
		// scalars, do nothing
	}
	return allErrs
}

func isListMap(s structuralschema.Structural) bool {
	return s.XListType != nil && *s.XListType == structuralschema.XListTypeMap &&
		s.Type == structuralschema.GenericTypeArray
}

func isListSet(s structuralschema.Structural) bool {
	return s.Type == structuralschema.GenericTypeArray &&
		s.XListType != nil && *s.XListType == structuralschema.XListTypeSet
}

func isListAtomic(s structuralschema.Structural) bool {
	return s.Type == structuralschema.GenericTypeArray &&
		(strOrEmpty(s.XListType) == "" || strOrEmpty(s.XListType) == structuralschema.XListTypeAtomic)
}

func isMap(s structuralschema.Structural) bool {
	return s.Type == structuralschema.GenericTypeObject &&
		s.AdditionalProperties != nil
}

func strOrEmpty(pointer *string) string {
	if pointer == nil {
		return ""
	}
	return *pointer
}

// isMutableMapImmutableValues
// Mutable map, immutable values, map type undefined / granular
// Equivalently to the list type map, removal and addition of key-value pairs are allowed,
// while direct value change is not.
func isMapImmutableValues(oldMap, newMap map[string]interface{}) bool {
	for key := range oldMap {
		if _, ok := newMap[key]; ok {
			if !reflect.DeepEqual(oldMap[key], newMap[key]) {
				return false
			}
		}
	}
	return true
}

// Mutable array, immutable items, list type undefined / atomic
// The items with their respective index in the array are immutable.
// Hence, appending and removal at the end of the array are allowed, the change of an item or the change of the order are not.
func isListAtomicImmutableItems(old, new []interface{}) bool {
	shortestArray := old
	if len(new) < len(old) {
		shortestArray = new
	}
	for key := range shortestArray {
		if !reflect.DeepEqual(old[key], new[key]) {
			return false
		}
	}
	return true
}

// Mutable array, immutable items, list type map
// The key-value pairs in the array are immutable, the set of keys is not.
// Hence, addition and removal of key-values pairs is allowed.
func isListMapImmutableItems(keys []string, old, new []interface{}) (bool, error) {
	indexedList := make(map[string]interface{}, len(old))
	for i := range old {
		var indexBuffer bytes.Buffer
		oldItem, ok := old[i].(map[string]interface{})
		if !ok {
			return false, nil
		}

		for _, key := range keys {
			key, err := json.Marshal(oldItem[key])
			if err != nil {
				return false, err
			}
			indexBuffer.WriteString(string(key))
		}
		indexedList[indexBuffer.String()] = oldItem
	}

	for i := range new {
		var indexBuffer bytes.Buffer
		newItem, ok := new[i].(map[string]interface{})
		if !ok {
			return false, nil
		}

		for _, key := range keys {
			key, _ := json.Marshal(newItem[key])
			indexBuffer.WriteString(string(key))
		}

		if oldItem, ok := indexedList[indexBuffer.String()]; ok {
			if !reflect.DeepEqual(oldItem, newItem) {
				return false, nil
			}
		}
	}

	return true, nil
}

// Immutable array keys, mutable values, list type map
// The set of keys is immutable/add-only/remove-only.
// Hence, non-key values can be changed.
// New key-values cannot be added (Immutable, RemoveOnly) and old key-values cannot be removed (Immutable, AddOnly).
func isListMapImmutableKeys(immutability string, keys []string, old, new []interface{}) (bool, error) {
	if immutability == structuralschema.XImmutabilityImmutable {
		if len(old) != len(new) {
			return false, nil
		}
	}

	if immutability == structuralschema.XImmutabilityAddOnly {
		if len(old) > len(new) {
			return false, nil
		}
	}

	if immutability == structuralschema.XImmutabilityRemoveOnly {
		if len(old) < len(new) {
			return false, nil
		}
	}

	shortestArray := old
	if len(new) < len(old) {
		shortestArray = new
	}

	indexedList := make(map[string]interface{}, len(shortestArray))
	for i := range shortestArray {
		var indexBuffer bytes.Buffer
		oldItem, ok := old[i].(map[string]interface{})
		if !ok {
			return false, nil
		}

		for _, key := range keys {
			key, err := json.Marshal(oldItem[key])
			if err != nil {
				return false, err
			}
			indexBuffer.WriteString(string(key))
		}
		indexedList[indexBuffer.String()] = oldItem
	}

	for i := range shortestArray {
		var indexBuffer bytes.Buffer
		newItem, ok := new[i].(map[string]interface{})
		if !ok {
			return false, nil
		}

		for _, key := range keys {
			key, err := json.Marshal(newItem[key])
			if err != nil {
				return false, err
			}
			indexBuffer.WriteString(string(key))
		}

		if _, ok := indexedList[indexBuffer.String()]; !ok {
			return false, nil
		}
	}

	return true, nil
}

// Immutable array keys, mutable value, list type set
// The whole items are the keys. Hence,
// new items cannot be added (Immutable, RemoveOnly), old items cannot be removed (Immutable, AddOnly), but the order can.
// Note: this is different from a set with x-kubernetes-mutability: Immutable. The latter does not allow order changes.
func isListSetImmutableKeys(immutability string, old, new []interface{}) (bool, error) {
	oldKeyCount := make(map[interface{}]int, len(old))
	for _, v := range old {
		index, err := json.Marshal(v)
		if err != nil {
			return false, err
		}
		oldKeyCount[string(index)]++
	}

	newKeyCount := make(map[interface{}]int, len(new))
	for _, v := range new {
		index, err := json.Marshal(v)
		if err != nil {
			return false, err
		}
		newKeyCount[string(index)]++
	}

	if immutability == structuralschema.XImmutabilityImmutable {
		if len(old) != len(new) {
			return false, nil
		}
		for k := range oldKeyCount {
			if oldKeyCount[k] != newKeyCount[k] {
				return false, nil
			}
		}
	}

	if immutability == structuralschema.XImmutabilityAddOnly {
		if len(old) > len(new) {
			return false, nil
		}
		for k := range oldKeyCount {
			if _, ok := newKeyCount[k]; !ok {
				return false, nil

			}
		}
	}

	if immutability == structuralschema.XImmutabilityRemoveOnly {
		if len(old) < len(new) {
			return false, nil

		}
		for k := range newKeyCount {
			if _, ok := oldKeyCount[k]; !ok {
				return false, nil
			}
		}
	}

	return true, nil
}

// Immutable map keys, mutable values, map type undefined / granular / atomic
// Equivalently to the list type map,
// the set of keys is immutable/addOnly/removeOnly. Hence, values can be changed.
// New key-values cannot be added (Immutable, RemoveOnly) and old key-values cannot be removed (Immutable, AddOnly).
func isMapImmutableKeys(immutability string, old, new map[string]interface{}) bool {
	if immutability == structuralschema.XImmutabilityImmutable {
		if len(old) != len(new) {
			return false
		}
		for k := range old {
			if _, ok := new[k]; !ok {
				return false
			}
		}
	}

	if immutability == structuralschema.XImmutabilityAddOnly {
		if len(old) > len(new) {
			return false
		}
		for k := range old {
			if _, ok := new[k]; !ok {
				return false
			}
		}
	}

	if immutability == structuralschema.XImmutabilityRemoveOnly {
		if len(old) < len(new) {
			return false
		}
		for k := range new {
			if _, ok := old[k]; !ok {
				return false
			}
		}
	}

	return true
}

// isImmutable Validate immutable/addOnly/removeOnly against the current property regardless of the type.
// XImmutability/XKeyImmutability below this level is disallowed.
func isImmutable(immutability string, old, new map[string]interface{}, k string) bool {
	_, foundOld := old[k]
	_, foundNew := new[k]

	switch immutability {
	case structuralschema.XImmutabilityImmutable:
		if (foundOld != foundNew) || !(reflect.DeepEqual(old[k], new[k])) {
			return false
		}
	case structuralschema.XImmutabilityAddOnly:
		if (foundOld && !foundNew) ||
			foundOld && !(reflect.DeepEqual(old[k], new[k])) {
			return false
		}
	case structuralschema.XImmutabilityRemoveOnly:
		if foundNew && !(reflect.DeepEqual(old[k], new[k])) {
			return false
		}
	}
	return true
}

// Immutable array keys, mutable values, list type undefined / atomic
// The set of indices is immutable/add-only/remove-only.
// Hence, appending (Immutable, RemoveOnly) or shrinking (Immutable, AddOnly) is disallowed,
// but changes that do not change the length are allowed.
func isListAtomicImmutableKeys(immutability string, old, new []interface{}) bool {
	if immutability == structuralschema.XImmutabilityImmutable {
		if len(old) != len(new) {
			return false
		}
		return true
	}

	if immutability == structuralschema.XImmutabilityAddOnly {
		if len(old) > len(new) {
			return false
		}
		return true
	}

	if immutability == structuralschema.XImmutabilityRemoveOnly {
		if len(old) < len(new) {
			return false
		}
		return true
	}
	return true
}
