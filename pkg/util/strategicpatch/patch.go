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

	forkedjson "k8s.io/kubernetes/third_party/forked/json"
)

// An alternate implementation of JSON Merge Patch
// (https://tools.ietf.org/html/rfc7386) which supports the ability to annotate
// certain fields with metadata that indicates whether the elements of JSON
// lists should be merged or replaced.
//
// For more information, see the PATCH section of docs/api-conventions.md.
//
// Some of the content of this package was borrowed with minor adaptations from
// evanphx/json-patch and openshift/origin.

const specialKey = "$patch"
const specialValue = "delete"

var errBadJSONDoc = fmt.Errorf("Invalid JSON document")
var errNoListOfLists = fmt.Errorf("Lists of lists are not supported")

// CreateStrategicMergePatch creates a patch that can be passed to StrategicMergePatch.
// The original and modified documents must be passed to the method as json encoded content.
// It will return a mergeable json document with differences from original to modified, or an error
// if either of the two documents is invalid.
func CreateStrategicMergePatch(original, modified []byte, dataStruct interface{}) ([]byte, error) {
	originalMap := map[string]interface{}{}
	err := json.Unmarshal(original, &originalMap)
	if err != nil {
		return nil, errBadJSONDoc
	}

	modifiedMap := map[string]interface{}{}
	err = json.Unmarshal(modified, &modifiedMap)
	if err != nil {
		return nil, errBadJSONDoc
	}

	t, err := getTagStructType(dataStruct)
	if err != nil {
		return nil, err
	}

	patchMap, err := diffMaps(originalMap, modifiedMap, t, false, false)
	if err != nil {
		return nil, err
	}

	return json.Marshal(patchMap)
}

// Returns a (recursive) strategic merge patch that yields modified when applied to original.
func diffMaps(original, modified map[string]interface{}, t reflect.Type, ignoreAdditions, ignoreChangesAndDeletions bool) (map[string]interface{}, error) {
	patch := map[string]interface{}{}
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	for key, modifiedValue := range modified {
		originalValue, ok := original[key]
		// value was added
		if !ok {
			if !ignoreAdditions {
				patch[key] = modifiedValue
			}

			continue
		}

		if key == specialKey {
			originalString, ok := originalValue.(string)
			if !ok {
				return nil, fmt.Errorf("invalid value for special key: %s", specialKey)
			}

			modifiedString, ok := modifiedValue.(string)
			if !ok {
				return nil, fmt.Errorf("invalid value for special key: %s", specialKey)
			}

			if modifiedString != originalString {
				patch[key] = modifiedValue
			}

			continue
		}

		if !ignoreChangesAndDeletions {
			// If types have changed, replace completely
			if reflect.TypeOf(originalValue) != reflect.TypeOf(modifiedValue) {
				patch[key] = modifiedValue
				continue
			}
		}

		// Types are the same, compare values
		switch originalValueTyped := originalValue.(type) {
		case map[string]interface{}:
			modifiedValueTyped := modifiedValue.(map[string]interface{})
			fieldType, _, _, err := forkedjson.LookupPatchMetadata(t, key)
			if err != nil {
				return nil, err
			}

			patchValue, err := diffMaps(originalValueTyped, modifiedValueTyped, fieldType, ignoreAdditions, ignoreChangesAndDeletions)
			if err != nil {
				return nil, err
			}

			if len(patchValue) > 0 {
				patch[key] = patchValue
			}

			continue
		case []interface{}:
			modifiedValueTyped := modifiedValue.([]interface{})
			fieldType, fieldPatchStrategy, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, key)
			if err != nil {
				return nil, err
			}

			if fieldPatchStrategy == "merge" {
				patchValue, err := diffLists(originalValueTyped, modifiedValueTyped, fieldType.Elem(), fieldPatchMergeKey, ignoreAdditions, ignoreChangesAndDeletions)
				if err != nil {
					return nil, err
				}

				if len(patchValue) > 0 {
					patch[key] = patchValue
				}

				continue
			}
		}

		if !ignoreChangesAndDeletions {
			if !reflect.DeepEqual(originalValue, modifiedValue) {
				patch[key] = modifiedValue
			}
		}
	}

	if !ignoreChangesAndDeletions {
		// Now add all deleted values as nil
		for key := range original {
			_, found := modified[key]
			if !found {
				patch[key] = nil
			}
		}
	}

	return patch, nil
}

// Returns a (recursive) strategic merge patch that yields modified when applied to original,
// for a pair of lists with merge semantics.
func diffLists(original, modified []interface{}, t reflect.Type, mergeKey string, ignoreAdditions, ignoreChangesAndDeletions bool) ([]interface{}, error) {
	if len(original) == 0 {
		if len(modified) == 0 || ignoreAdditions {
			return nil, nil
		}

		return modified, nil
	}

	elementType, err := sliceElementType(original, modified)
	if err != nil {
		return nil, err
	}

	var patch []interface{}

	// If the elements are not maps...
	if elementType.Kind() == reflect.Map {
		patch, err = diffListsOfMaps(original, modified, t, mergeKey, ignoreAdditions, ignoreChangesAndDeletions)
	} else {
		patch, err = diffListsOfScalars(original, modified, ignoreAdditions)
	}

	if err != nil {
		return nil, err
	}

	return patch, nil
}

// Returns a (recursive) strategic merge patch that yields modified when applied to original,
// for a pair of lists of scalars with merge semantics.
func diffListsOfScalars(original, modified []interface{}, ignoreAdditions bool) ([]interface{}, error) {
	if len(modified) == 0 {
		// There is no need to check the length of original because there is no way to create
		// a patch that deletes a scalar from a list of scalars with merge semantics.
		return nil, nil
	}

	patch := []interface{}{}

	originalScalars := uniqifyAndSortScalars(original)
	modifiedScalars := uniqifyAndSortScalars(modified)
	originalIndex, modifiedIndex := 0, 0

loopB:
	for ; modifiedIndex < len(modifiedScalars); modifiedIndex++ {
		for ; originalIndex < len(originalScalars); originalIndex++ {
			originalString := fmt.Sprintf("%v", original[originalIndex])
			modifiedString := fmt.Sprintf("%v", modified[modifiedIndex])
			if originalString >= modifiedString {
				if originalString != modifiedString {
					if !ignoreAdditions {
						patch = append(patch, modified[modifiedIndex])
					}
				}

				continue loopB
			}
			// There is no else clause because there is no way to create a patch that deletes
			// a scalar from a list of scalars with merge semantics.
		}

		break
	}

	if !ignoreAdditions {
		// Add any remaining items found only in modified
		for ; modifiedIndex < len(modifiedScalars); modifiedIndex++ {
			patch = append(patch, modified[modifiedIndex])
		}
	}

	return patch, nil
}

var errNoMergeKeyFmt = "map: %v does not contain declared merge key: %s"
var errBadArgTypeFmt = "expected a %s, but received a %t"

// Returns a (recursive) strategic merge patch that yields modified when applied to original,
// for a pair of lists of maps with merge semantics.
func diffListsOfMaps(original, modified []interface{}, t reflect.Type, mergeKey string, ignoreAdditions, ignoreChangesAndDeletions bool) ([]interface{}, error) {
	patch := make([]interface{}, 0)

	originalSorted, err := sortMergeListsByNameArray(original, t, mergeKey, false)
	if err != nil {
		return nil, err
	}

	modifiedSorted, err := sortMergeListsByNameArray(modified, t, mergeKey, false)
	if err != nil {
		return nil, err
	}

	originalIndex, modifiedIndex := 0, 0

loopB:
	for ; modifiedIndex < len(modifiedSorted); modifiedIndex++ {
		modifiedMap, ok := modifiedSorted[modifiedIndex].(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf(errBadArgTypeFmt, "map[string]interface{}", modifiedSorted[modifiedIndex])
		}

		modifiedValue, ok := modifiedMap[mergeKey]
		if !ok {
			return nil, fmt.Errorf(errNoMergeKeyFmt, modifiedMap, mergeKey)
		}

		for ; originalIndex < len(originalSorted); originalIndex++ {
			originalMap, ok := originalSorted[originalIndex].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf(errBadArgTypeFmt, "map[string]interface{}", originalSorted[originalIndex])
			}

			originalValue, ok := originalMap[mergeKey]
			if !ok {
				return nil, fmt.Errorf(errNoMergeKeyFmt, originalMap, mergeKey)
			}

			// Assume that the merge key values are comparable strings
			originalString := fmt.Sprintf("%v", originalValue)
			modifiedString := fmt.Sprintf("%v", modifiedValue)
			if originalString >= modifiedString {
				if originalString == modifiedString {
					patchValue, err := diffMaps(originalMap, modifiedMap, t, ignoreAdditions, ignoreChangesAndDeletions)
					if err != nil {
						return nil, err
					}

					originalIndex++
					if len(patchValue) > 0 {
						patchValue[mergeKey] = modifiedValue
						patch = append(patch, patchValue)
					}
				} else if !ignoreAdditions {
					patch = append(patch, modifiedMap)
				}

				continue loopB
			}

			if !ignoreChangesAndDeletions {
				patch = append(patch, map[string]interface{}{mergeKey: originalValue, specialKey: specialValue})
			}
		}

		break
	}

	if !ignoreChangesAndDeletions {
		// Delete any remaining items found only in original
		for ; originalIndex < len(originalSorted); originalIndex++ {
			originalMap, ok := originalSorted[originalIndex].(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf(errBadArgTypeFmt, "map[string]interface{}", originalSorted[originalIndex])
			}

			originalValue, ok := originalMap[mergeKey]
			if !ok {
				return nil, fmt.Errorf(errNoMergeKeyFmt, originalMap, mergeKey)
			}

			patch = append(patch, map[string]interface{}{mergeKey: originalValue, specialKey: specialValue})
		}
	}

	if !ignoreAdditions {
		// Add any remaining items found only in modified
		for ; modifiedIndex < len(modifiedSorted); modifiedIndex++ {
			patch = append(patch, modified[modifiedIndex])
		}
	}

	return patch, nil
}

// StrategicMergePatchData applies a patch using strategic merge patch semantics.
// Deprecated: StrategicMergePatchData is deprecated. Use the synonym StrategicMergePatch,
// instead, which follows the naming convention of evanphx/json-patch.
func StrategicMergePatchData(original, patch []byte, dataStruct interface{}) ([]byte, error) {
	return StrategicMergePatch(original, patch, dataStruct)
}

// StrategicMergePatch applies a strategic merge patch. The patch and the original document
// must be json encoded content. A patch can be created from an original and a modified document
// by calling CreateStrategicMergePatch.
func StrategicMergePatch(original, patch []byte, dataStruct interface{}) ([]byte, error) {
	originalMap := map[string]interface{}{}
	err := json.Unmarshal(original, &originalMap)
	if err != nil {
		return nil, errBadJSONDoc
	}

	patchMap := map[string]interface{}{}
	err = json.Unmarshal(patch, &patchMap)
	if err != nil {
		return nil, errBadJSONDoc
	}

	t, err := getTagStructType(dataStruct)
	if err != nil {
		return nil, err
	}

	result, err := mergeMap(originalMap, patchMap, t)
	if err != nil {
		return nil, err
	}

	return json.Marshal(result)
}

func getTagStructType(dataStruct interface{}) (reflect.Type, error) {
	t := reflect.TypeOf(dataStruct)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return nil, fmt.Errorf(errBadArgTypeFmt, "struct", t.Kind().String())
	}

	return t, nil
}

var errBadPatchTypeFmt = "unknown patch type: %s in map: %v"

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

		return nil, fmt.Errorf(errBadPatchTypeFmt, v, patch)
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
		originalType := reflect.TypeOf(original[k])
		patchType := reflect.TypeOf(patchV)
		if originalType == patchType {
			// First find the fieldPatchStrategy and fieldPatchMergeKey.
			fieldType, fieldPatchStrategy, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, k)
			if err != nil {
				return nil, err
			}

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
		return nil, err
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
			if patchType == specialValue {
				mergeValue, ok := typedV[mergeKey]
				if ok {
					_, originalKey, found, err := findMapInSliceBasedOnKeyValue(original, mergeKey, mergeValue)
					if err != nil {
						return nil, err
					}

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
				return nil, fmt.Errorf(errBadPatchTypeFmt, patchType, typedV)
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
			return nil, fmt.Errorf(errNoMergeKeyFmt, typedV, mergeKey)
		}

		// If we find a value with this merge key value in original, merge the
		// maps. Otherwise append onto original.
		originalMap, originalKey, found, err := findMapInSliceBasedOnKeyValue(original, mergeKey, mergeValue)
		if err != nil {
			return nil, err
		}

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

// This method no longer panics if any element of the slice is not a map.
func findMapInSliceBasedOnKeyValue(m []interface{}, key string, value interface{}) (map[string]interface{}, int, bool, error) {
	for k, v := range m {
		typedV, ok := v.(map[string]interface{})
		if !ok {
			return nil, 0, false, fmt.Errorf("value for key %v is not a map.", k)
		}

		valueToMatch, ok := typedV[key]
		if ok && valueToMatch == value {
			return typedV, k, true, nil
		}
	}

	return nil, 0, false, nil
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
		if k != specialKey {
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
					v, err = sortMergeListsByNameArray(typedV, fieldType.Elem(), fieldPatchMergeKey, true)
					if err != nil {
						return nil, err
					}
				}
			}
		}

		newS[k] = v
	}

	return newS, nil
}

func sortMergeListsByNameArray(s []interface{}, elemType reflect.Type, mergeKey string, recurse bool) ([]interface{}, error) {
	if len(s) == 0 {
		return s, nil
	}

	// We don't support lists of lists yet.
	t, err := sliceElementType(s)
	if err != nil {
		return nil, err
	}

	// If the elements are not maps...
	if t.Kind() != reflect.Map {
		// Sort the elements, because they may have been merged out of order.
		return uniqifyAndSortScalars(s), nil
	}

	// Elements are maps - if one of the keys of the map is a map or a
	// list, we may need to recurse into it.
	newS := []interface{}{}
	for _, elem := range s {
		if recurse {
			typedElem := elem.(map[string]interface{})
			newElem, err := sortMergeListsByNameMap(typedElem, elemType)
			if err != nil {
				return nil, err
			}

			newS = append(newS, newElem)
		} else {
			newS = append(newS, elem)
		}
	}

	// Sort the maps.
	newS = sortMapsBasedOnField(newS, mergeKey)
	return newS, nil
}

func sortMapsBasedOnField(m []interface{}, fieldName string) []interface{} {
	mapM := mapSliceFromSlice(m)
	ss := SortableSliceOfMaps{mapM, fieldName}
	sort.Sort(ss)
	newS := sliceFromMapSlice(ss.s)
	return newS
}

func mapSliceFromSlice(m []interface{}) []map[string]interface{} {
	newM := []map[string]interface{}{}
	for _, v := range m {
		vt := v.(map[string]interface{})
		newM = append(newM, vt)
	}

	return newM
}

func sliceFromMapSlice(s []map[string]interface{}) []interface{} {
	newS := []interface{}{}
	for _, v := range s {
		newS = append(newS, v)
	}

	return newS
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
// another slice or undefined, returns an error.
func sliceElementType(slices ...[]interface{}) (reflect.Type, error) {
	var prevType reflect.Type
	for _, s := range slices {
		// Go through elements of all given slices and make sure they are all the same type.
		for _, v := range s {
			currentType := reflect.TypeOf(v)
			if prevType == nil {
				prevType = currentType
				// We don't support lists of lists yet.
				if prevType.Kind() == reflect.Slice {
					return nil, errNoListOfLists
				}
			} else {
				if prevType != currentType {
					return nil, fmt.Errorf("list element types are not identical: %v", fmt.Sprint(slices))
				}
				prevType = currentType
			}
		}
	}

	if prevType == nil {
		return nil, fmt.Errorf("no elements in any of the given slices")
	}

	return prevType, nil
}
