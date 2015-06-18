/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package strategicpatch

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"

	forkedjson "github.com/GoogleCloudPlatform/kubernetes/third_party/forked/json"
)

// An alternate implementation of JSON Merge Patch
// (https://tools.ietf.org/html/rfc7386) which supports the ability to annotate
// certain fields with metadata that indicates whether the elements of JSON
// lists should be merged or replaced.
//
// For more information, see the PATCH section of docs/api-conventions.md.
func StrategicMergePatchData(original, patch []byte, dataStruct interface{}) ([]byte, error) {
	var o map[string]interface{}
	err := json.Unmarshal(original, &o)
	if err != nil {
		return nil, err
	}

	var p map[string]interface{}
	err = json.Unmarshal(patch, &p)
	if err != nil {
		return nil, err
	}

	t := reflect.TypeOf(dataStruct)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf("strategic merge patch needs a struct, %s received instead", t.Kind().String())
	}

	result, err := mergeMap(o, p, t)
	if err != nil {
		return nil, err
	}

	return json.Marshal(result)
}

const specialKey = "$patch"

// Merge fields from a patch map into the original map. Note: This may modify
// both the original map and the patch because getting a deep copy of a map in
// golang is highly non-trivial.
func mergeMap(original, patch map[string]interface{}, t reflect.Type) (map[string]interface{}, error) {
	// If the map contains "$patch: replace", don't merge it, just use the
	// patch map directly. Later on, can add a non-recursive replace that only
	// affects the map that the $patch is in.
	if v, ok := patch[specialKey]; ok {
		if v == "replace" {
			delete(patch, specialKey)
			return patch, nil
		}
		return nil, fmt.Errorf("unknown patch type found: %s", v)
	}

	// nil is an accepted value for original to simplify logic in other places.
	// If original is nil, create a map so if patch requires us to modify the
	// map, it'll work.
	if original == nil {
		original = map[string]interface{}{}
	}

	// Start merging the patch into the original.
	for k, patchV := range patch {
		// If the value of this key is null, delete the key if it exists in the
		// original. Otherwise, skip it.
		if patchV == nil {
			if _, ok := original[k]; ok {
				delete(original, k)
			}
			continue
		}
		_, ok := original[k]
		if !ok {
			// If it's not in the original document, just take the patch value.
			original[k] = patchV
			continue
		}
		// If the data type is a pointer, resolve the element.
		if t.Kind() == reflect.Ptr {
			t = t.Elem()
		}

		// If they're both maps or lists, recurse into the value.
		// First find the fieldPatchStrategy and fieldPatchMergeKey.
		fieldType, fieldPatchStrategy, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, k)
		if err != nil {
			return nil, err
		}

		originalType := reflect.TypeOf(original[k])
		patchType := reflect.TypeOf(patchV)
		if originalType == patchType {
			if originalType.Kind() == reflect.Map && fieldPatchStrategy != "replace" {
				typedOriginal := original[k].(map[string]interface{})
				typedPatch := patchV.(map[string]interface{})
				var err error
				original[k], err = mergeMap(typedOriginal, typedPatch, fieldType)
				if err != nil {
					return nil, err
				}
				continue
			}
			if originalType.Kind() == reflect.Slice && fieldPatchStrategy == "merge" {
				elemType := fieldType.Elem()
				typedOriginal := original[k].([]interface{})
				typedPatch := patchV.([]interface{})
				var err error
				original[k], err = mergeSlice(typedOriginal, typedPatch, elemType, fieldPatchMergeKey)
				if err != nil {
					return nil, err
				}
				continue
			}
		}

		// If originalType and patchType are different OR the types are both
		// maps or slices but we're just supposed to replace them, just take
		// the value from patch.
		original[k] = patchV
	}

	return original, nil
}

// Merge two slices together. Note: This may modify both the original slice and
// the patch because getting a deep copy of a slice in golang is highly
// non-trivial.
func mergeSlice(original, patch []interface{}, elemType reflect.Type, mergeKey string) ([]interface{}, error) {
	if len(original) == 0 && len(patch) == 0 {
		return original, nil
	}
	// All the values must be of the same type, but not a list.
	t, err := sliceElementType(original, patch)
	if err != nil {
		return nil, fmt.Errorf("types of list elements need to be the same, type: %s: %v",
			elemType.Kind().String(), err)
	}
	if t.Kind() == reflect.Slice {
		return nil, fmt.Errorf("not supporting merging lists of lists yet")
	}
	// If the elements are not maps, merge the slices of scalars.
	if t.Kind() != reflect.Map {
		// Maybe in the future add a "concat" mode that doesn't
		// uniqify.
		both := append(original, patch...)
		return uniqifyScalars(both), nil
	}

	if mergeKey == "" {
		return nil, fmt.Errorf("cannot merge lists without merge key for type %s", elemType.Kind().String())
	}

	// First look for any special $patch elements.
	patchWithoutSpecialElements := []interface{}{}
	replace := false
	for _, v := range patch {
		typedV := v.(map[string]interface{})
		patchType, ok := typedV[specialKey]
		if ok {
			if patchType == "delete" {
				mergeValue, ok := typedV[mergeKey]
				if ok {
					_, originalKey, found := findMapInSliceBasedOnKeyValue(original, mergeKey, mergeValue)
					if found {
						// Delete the element at originalKey.
						original = append(original[:originalKey], original[originalKey+1:]...)
					}
				} else {
					return nil, fmt.Errorf("delete patch type with no merge key defined")
				}
			} else if patchType == "replace" {
				replace = true
				// Continue iterating through the array to prune any other $patch elements.
			} else if patchType == "merge" {
				return nil, fmt.Errorf("merging lists cannot yet be specified in the patch")
			} else {
				return nil, fmt.Errorf("unknown patch type found: %s", patchType)
			}
		} else {
			patchWithoutSpecialElements = append(patchWithoutSpecialElements, v)
		}
	}

	if replace {
		return patchWithoutSpecialElements, nil
	}

	patch = patchWithoutSpecialElements

	// Merge patch into original.
	for _, v := range patch {
		// Because earlier we confirmed that all the elements are maps.
		typedV := v.(map[string]interface{})
		mergeValue, ok := typedV[mergeKey]
		if !ok {
			return nil, fmt.Errorf("all list elements need the merge key %s", mergeKey)
		}
		// If we find a value with this merge key value in original, merge the
		// maps. Otherwise append onto original.
		originalMap, originalKey, found := findMapInSliceBasedOnKeyValue(original, mergeKey, mergeValue)
		if found {
			var mergedMaps interface{}
			var err error
			// Merge into original.
			mergedMaps, err = mergeMap(originalMap, typedV, elemType)
			if err != nil {
				return nil, err
			}
			original[originalKey] = mergedMaps
		} else {
			original = append(original, v)
		}
	}

	return original, nil
}

// This panics if any element of the slice is not a map.
func findMapInSliceBasedOnKeyValue(m []interface{}, key string, value interface{}) (map[string]interface{}, int, bool) {
	for k, v := range m {
		typedV := v.(map[string]interface{})
		valueToMatch, ok := typedV[key]
		if ok && valueToMatch == value {
			return typedV, k, true
		}
	}
	return nil, 0, false
}

// This function takes a JSON map and sorts all the lists that should be merged
// by key. This is needed by tests because in JSON, list order is significant,
// but in Strategic Merge Patch, merge lists do not have significant order.
// Sorting the lists allows for order-insensitive comparison of patched maps.
func sortMergeListsByName(mapJSON []byte, dataStruct interface{}) ([]byte, error) {
	var m map[string]interface{}
	err := json.Unmarshal(mapJSON, &m)
	if err != nil {
		return nil, err
	}

	newM, err := sortMergeListsByNameMap(m, reflect.TypeOf(dataStruct))
	if err != nil {
		return nil, err
	}

	return json.Marshal(newM)
}

func sortMergeListsByNameMap(s map[string]interface{}, t reflect.Type) (map[string]interface{}, error) {
	newS := map[string]interface{}{}
	for k, v := range s {
		fieldType, fieldPatchStrategy, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, k)
		if err != nil {
			return nil, err
		}
		// If v is a map or a merge slice, recurse.
		if typedV, ok := v.(map[string]interface{}); ok {
			var err error
			v, err = sortMergeListsByNameMap(typedV, fieldType)
			if err != nil {
				return nil, err
			}
		} else if typedV, ok := v.([]interface{}); ok {
			if fieldPatchStrategy == "merge" {
				var err error
				v, err = sortMergeListsByNameArray(typedV, fieldType.Elem(), fieldPatchMergeKey)
				if err != nil {
					return nil, err
				}
			}
		}
		newS[k] = v
	}
	return newS, nil
}

func sortMergeListsByNameArray(s []interface{}, elemType reflect.Type, mergeKey string) ([]interface{}, error) {
	if len(s) == 0 {
		return s, nil
	}
	// We don't support lists of lists yet.
	t, err := sliceElementType(s)
	if err != nil {
		return nil, err
	}
	if t.Kind() == reflect.Slice {
		return nil, fmt.Errorf("not supporting lists of lists yet")
	}

	// If the elements are not maps...
	if t.Kind() != reflect.Map {
		// Sort the elements, because they may have been merged out of order.
		return uniqifyAndSortScalars(s), nil
	}

	// Elements are maps - if one of the keys of the map is a map or a
	// list, we need to recurse into it.
	newS := []interface{}{}
	for _, elem := range s {
		typedElem := elem.(map[string]interface{})
		newElem, err := sortMergeListsByNameMap(typedElem, elemType)
		if err != nil {
			return nil, err
		}
		newS = append(newS, newElem)
	}

	// Sort the maps.
	newS = sortMapsBasedOnField(newS, mergeKey)
	return newS, nil
}

func sortMapsBasedOnField(m []interface{}, fieldName string) []interface{} {
	mapM := []map[string]interface{}{}
	for _, v := range m {
		mapM = append(mapM, v.(map[string]interface{}))
	}
	ss := SortableSliceOfMaps{mapM, fieldName}
	sort.Sort(ss)
	newM := []interface{}{}
	for _, v := range ss.s {
		newM = append(newM, v)
	}
	return newM
}

type SortableSliceOfMaps struct {
	s []map[string]interface{}
	k string // key to sort on
}

func (ss SortableSliceOfMaps) Len() int {
	return len(ss.s)
}

func (ss SortableSliceOfMaps) Less(i, j int) bool {
	iStr := fmt.Sprintf("%v", ss.s[i][ss.k])
	jStr := fmt.Sprintf("%v", ss.s[j][ss.k])
	return sort.StringsAreSorted([]string{iStr, jStr})
}

func (ss SortableSliceOfMaps) Swap(i, j int) {
	tmp := ss.s[i]
	ss.s[i] = ss.s[j]
	ss.s[j] = tmp
}

func uniqifyAndSortScalars(s []interface{}) []interface{} {
	s = uniqifyScalars(s)

	ss := SortableSliceOfScalars{s}
	sort.Sort(ss)
	return ss.s
}

func uniqifyScalars(s []interface{}) []interface{} {
	// Clever algorithm to uniqify.
	length := len(s) - 1
	for i := 0; i < length; i++ {
		for j := i + 1; j <= length; j++ {
			if s[i] == s[j] {
				s[j] = s[length]
				s = s[0:length]
				length--
				j--
			}
		}
	}
	return s
}

type SortableSliceOfScalars struct {
	s []interface{}
}

func (ss SortableSliceOfScalars) Len() int {
	return len(ss.s)
}

func (ss SortableSliceOfScalars) Less(i, j int) bool {
	iStr := fmt.Sprintf("%v", ss.s[i])
	jStr := fmt.Sprintf("%v", ss.s[j])
	return sort.StringsAreSorted([]string{iStr, jStr})
}

func (ss SortableSliceOfScalars) Swap(i, j int) {
	tmp := ss.s[i]
	ss.s[i] = ss.s[j]
	ss.s[j] = tmp
}

// Returns the type of the elements of N slice(s). If the type is different,
// returns an error.
func sliceElementType(slices ...[]interface{}) (reflect.Type, error) {
	var prevType reflect.Type
	for _, s := range slices {
		// Go through elements of all given slices and make sure they are all the same type.
		for _, v := range s {
			currentType := reflect.TypeOf(v)
			if prevType == nil {
				prevType = currentType
			} else {
				if prevType != currentType {
					return nil, fmt.Errorf("at least two types found: %s and %s", prevType, currentType)
				}
				prevType = currentType
			}
		}
	}
	if prevType == nil {
		return nil, fmt.Errorf("no elements in any given slices")
	}
	return prevType, nil
}
