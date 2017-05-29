/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	forkedjson "k8s.io/apimachinery/third_party/forked/golang/json"
)

// An alternate implementation of JSON Merge Patch
// (https://tools.ietf.org/html/rfc7386) which supports the ability to annotate
// certain fields with metadata that indicates whether the elements of JSON
// lists should be merged or replaced.
//
// For more information, see the PATCH section of docs/devel/api-conventions.md.
//
// Some of the content of this package was borrowed with minor adaptations from
// evanphx/json-patch and openshift/origin.

const (
	directiveMarker  = "$patch"
	deleteDirective  = "delete"
	replaceDirective = "replace"
	mergeDirective   = "merge"

	retainKeysStrategy = "retainKeys"

	deleteFromPrimitiveListDirectivePrefix = "$deleteFromPrimitiveList"
	retainKeysDirective                    = "$" + retainKeysStrategy
)

// JSONMap is a representations of JSON object encoded as map[string]interface{}
// where the children can be either map[string]interface{}, []interface{} or
// primitive type).
// Operating on JSONMap representation is much faster as it doesn't require any
// json marshaling and/or unmarshaling operations.
type JSONMap map[string]interface{}

type DiffOptions struct {
	// IgnoreChangesAndAdditions indicates if we keep the changes and additions in the patch.
	IgnoreChangesAndAdditions bool
	// IgnoreDeletions indicates if we keep the deletions in the patch.
	IgnoreDeletions bool
	// We introduce a new value retainKeys for patchStrategy.
	// It indicates that all fields needing to be preserved must be
	// present in the `retainKeys` list.
	// And the fields that are present will be merged with live object.
	// All the missing fields will be cleared when patching.
	BuildRetainKeysDirective bool
}

type MergeOptions struct {
	// MergeParallelList indicates if we are merging the parallel list.
	// We don't merge parallel list when calling mergeMap() in CreateThreeWayMergePatch()
	// which is called client-side.
	// We merge parallel list iff when calling mergeMap() in StrategicMergeMapPatch()
	// which is called server-side
	MergeParallelList bool
	// IgnoreUnmatchedNulls indicates if we should process the unmatched nulls.
	IgnoreUnmatchedNulls bool
}

// The following code is adapted from github.com/openshift/origin/pkg/util/jsonmerge.
// Instead of defining a Delta that holds an original, a patch and a set of preconditions,
// the reconcile method accepts a set of preconditions as an argument.

// CreateTwoWayMergePatch creates a patch that can be passed to StrategicMergePatch from an original
// document and a modified document, which are passed to the method as json encoded content. It will
// return a patch that yields the modified document when applied to the original document, or an error
// if either of the two documents is invalid.
func CreateTwoWayMergePatch(original, modified []byte, dataStruct interface{}, fns ...mergepatch.PreconditionFunc) ([]byte, error) {
	originalMap := map[string]interface{}{}
	if len(original) > 0 {
		if err := json.Unmarshal(original, &originalMap); err != nil {
			return nil, mergepatch.ErrBadJSONDoc
		}
	}

	modifiedMap := map[string]interface{}{}
	if len(modified) > 0 {
		if err := json.Unmarshal(modified, &modifiedMap); err != nil {
			return nil, mergepatch.ErrBadJSONDoc
		}
	}

	patchMap, err := CreateTwoWayMergeMapPatch(originalMap, modifiedMap, dataStruct, fns...)
	if err != nil {
		return nil, err
	}

	return json.Marshal(patchMap)
}

// CreateTwoWayMergeMapPatch creates a patch from an original and modified JSON objects,
// encoded JSONMap.
// The serialized version of the map can then be passed to StrategicMergeMapPatch.
func CreateTwoWayMergeMapPatch(original, modified JSONMap, dataStruct interface{}, fns ...mergepatch.PreconditionFunc) (JSONMap, error) {
	t, err := getTagStructType(dataStruct)
	if err != nil {
		return nil, err
	}

	diffOptions := DiffOptions{}
	patchMap, err := diffMaps(original, modified, t, diffOptions)
	if err != nil {
		return nil, err
	}

	// Apply the preconditions to the patch, and return an error if any of them fail.
	for _, fn := range fns {
		if !fn(patchMap) {
			return nil, mergepatch.NewErrPreconditionFailed(patchMap)
		}
	}

	return patchMap, nil
}

// Returns a (recursive) strategic merge patch that yields modified when applied to original.
// Including:
// - Adding fields to the patch present in modified, missing from original
// - Setting fields to the patch present in modified and original with different values
// - Delete fields present in original, missing from modified through
// - IFF map field - set to nil in patch
// - IFF list of maps && merge strategy - use deleteDirective for the elements
// - IFF list of primitives && merge strategy - use parallel deletion list
// - IFF list of maps or primitives with replace strategy (default) - set patch value to the value in modified
// - Build $retainKeys directive for fields with retainKeys patch strategy
func diffMaps(original, modified map[string]interface{}, t reflect.Type, diffOptions DiffOptions) (map[string]interface{}, error) {
	patch := map[string]interface{}{}
	// Get the underlying type for pointers
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	// This will be used to build the $retainKeys directive sent in the patch
	retainKeysList := make([]interface{}, 0, len(modified))

	// Compare each value in the modified map against the value in the original map
	for key, modifiedValue := range modified {
		// Get the underlying type for pointers
		if diffOptions.BuildRetainKeysDirective && modifiedValue != nil {
			retainKeysList = append(retainKeysList, key)
		}

		originalValue, ok := original[key]
		if !ok {
			// Key was added, so add to patch
			if !diffOptions.IgnoreChangesAndAdditions {
				patch[key] = modifiedValue
			}
			continue
		}

		// The patch may have a patch directive
		// TODO: figure out if we need this. This shouldn't be needed by apply. When would the original map have patch directives in it?
		foundDirectiveMarker, err := handleDirectiveMarker(key, originalValue, modifiedValue, patch)
		if err != nil {
			return nil, err
		}
		if foundDirectiveMarker {
			continue
		}

		if reflect.TypeOf(originalValue) != reflect.TypeOf(modifiedValue) {
			// Types have changed, so add to patch
			if !diffOptions.IgnoreChangesAndAdditions {
				patch[key] = modifiedValue
			}
			continue
		}

		// Types are the same, so compare values
		switch originalValueTyped := originalValue.(type) {
		case map[string]interface{}:
			modifiedValueTyped := modifiedValue.(map[string]interface{})
			err = handleMapDiff(key, originalValueTyped, modifiedValueTyped, patch, t, diffOptions)
		case []interface{}:
			modifiedValueTyped := modifiedValue.([]interface{})
			err = handleSliceDiff(key, originalValueTyped, modifiedValueTyped, patch, t, diffOptions)
		default:
			replacePatchFieldIfNotEqual(key, originalValue, modifiedValue, patch, diffOptions)
		}
		if err != nil {
			return nil, err
		}
	}

	updatePatchIfMissing(original, modified, patch, diffOptions)
	// Insert the retainKeysList iff there are values present in the retainKeysList and
	// either of the following is true:
	// - the patch is not empty
	// - there are additional field in original that need to be cleared
	if len(retainKeysList) > 0 &&
		(len(patch) > 0 || hasAdditionalNewField(original, modified)) {
		patch[retainKeysDirective] = sortScalars(retainKeysList)
	}
	return patch, nil
}

// handleDirectiveMarker handles how to diff directive marker between 2 objects
func handleDirectiveMarker(key string, originalValue, modifiedValue interface{}, patch map[string]interface{}) (bool, error) {
	if key == directiveMarker {
		originalString, ok := originalValue.(string)
		if !ok {
			return false, fmt.Errorf("invalid value for special key: %s", directiveMarker)
		}
		modifiedString, ok := modifiedValue.(string)
		if !ok {
			return false, fmt.Errorf("invalid value for special key: %s", directiveMarker)
		}
		if modifiedString != originalString {
			patch[directiveMarker] = modifiedValue
		}
		return true, nil
	}
	return false, nil
}

// handleMapDiff diff between 2 maps `originalValueTyped` and `modifiedValue`,
// puts the diff in the `patch` associated with `key`
// key is the key associated with originalValue and modifiedValue.
// originalValue, modifiedValue are the old and new value respectively.They are both maps
// patch is the patch map that contains key and the updated value, and it is the parent of originalValue, modifiedValue
// diffOptions contains multiple options to control how we do the diff.
func handleMapDiff(key string, originalValue, modifiedValue, patch map[string]interface{},
	t reflect.Type, diffOptions DiffOptions) error {
	fieldType, fieldPatchStrategies, _, err := forkedjson.LookupPatchMetadata(t, key)
	if err != nil {
		// We couldn't look up metadata for the field
		// If the values are identical, this doesn't matter, no patch is needed
		if reflect.DeepEqual(originalValue, modifiedValue) {
			return nil
		}
		// Otherwise, return the error
		return err
	}
	retainKeys, patchStrategy, err := extractRetainKeysPatchStrategy(fieldPatchStrategies)
	if err != nil {
		return err
	}
	diffOptions.BuildRetainKeysDirective = retainKeys
	switch patchStrategy {
	// The patch strategic from metadata tells us to replace the entire object instead of diffing it
	case replaceDirective:
		if !diffOptions.IgnoreChangesAndAdditions {
			patch[key] = modifiedValue
		}
	default:
		patchValue, err := diffMaps(originalValue, modifiedValue, fieldType, diffOptions)
		if err != nil {
			return err
		}
		// Maps were not identical, use provided patch value
		if len(patchValue) > 0 {
			patch[key] = patchValue
		}
	}
	return nil
}

// handleSliceDiff diff between 2 slices `originalValueTyped` and `modifiedValue`,
// puts the diff in the `patch` associated with `key`
// key is the key associated with originalValue and modifiedValue.
// originalValue, modifiedValue are the old and new value respectively.They are both slices
// patch is the patch map that contains key and the updated value, and it is the parent of originalValue, modifiedValue
// diffOptions contains multiple options to control how we do the diff.
func handleSliceDiff(key string, originalValue, modifiedValue []interface{}, patch map[string]interface{},
	t reflect.Type, diffOptions DiffOptions) error {
	fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, key)
	if err != nil {
		// We couldn't look up metadata for the field
		// If the values are identical, this doesn't matter, no patch is needed
		if reflect.DeepEqual(originalValue, modifiedValue) {
			return nil
		}
		// Otherwise, return the error
		return err
	}
	retainKeys, patchStrategy, err := extractRetainKeysPatchStrategy(fieldPatchStrategies)
	if err != nil {
		return err
	}
	switch patchStrategy {
	// Merge the 2 slices using mergePatchKey
	case mergeDirective:
		diffOptions.BuildRetainKeysDirective = retainKeys
		addList, deletionList, err := diffLists(originalValue, modifiedValue, fieldType.Elem(), fieldPatchMergeKey, diffOptions)
		if err != nil {
			return err
		}
		if len(addList) > 0 {
			patch[key] = addList
		}
		// generate a parallel list for deletion
		if len(deletionList) > 0 {
			parallelDeletionListKey := fmt.Sprintf("%s/%s", deleteFromPrimitiveListDirectivePrefix, key)
			patch[parallelDeletionListKey] = deletionList
		}
	default:
		replacePatchFieldIfNotEqual(key, originalValue, modifiedValue, patch, diffOptions)
	}
	return nil
}

// replacePatchFieldIfNotEqual updates the patch if original and modified are not deep equal
// if diffOptions.IgnoreChangesAndAdditions is false.
// original is the old value, maybe either the live cluster object or the last applied configuration
// modified is the new value, is always the users new config
func replacePatchFieldIfNotEqual(key string, original, modified interface{},
	patch map[string]interface{}, diffOptions DiffOptions) {
	if diffOptions.IgnoreChangesAndAdditions {
		// Ignoring changes - do nothing
		return
	}
	if reflect.DeepEqual(original, modified) {
		// Contents are identical - do nothing
		return
	}
	// Create a patch to replace the old value with the new one
	patch[key] = modified
}

// updatePatchIfMissing iterates over `original` when ignoreDeletions is false.
// Clear the field whose key is not present in `modified`.
// original is the old value, maybe either the live cluster object or the last applied configuration
// modified is the new value, is always the users new config
func updatePatchIfMissing(original, modified, patch map[string]interface{}, diffOptions DiffOptions) {
	if diffOptions.IgnoreDeletions {
		// Ignoring deletion - do nothing
		return
	}
	// Add nils for deleted values
	for key := range original {
		if _, found := modified[key]; !found {
			patch[key] = nil
		}
	}
}

// Returns a (recursive) strategic merge patch and a parallel deletion list if necessary.
// Only list of primitives with merge strategy will generate a parallel deletion list.
// These two lists should yield modified when applied to original, for lists with merge semantics.
func diffLists(original, modified []interface{}, t reflect.Type, mergeKey string, diffOptions DiffOptions) ([]interface{}, []interface{}, error) {
	if len(original) == 0 {
		// Both slices are empty - do nothing
		if len(modified) == 0 || diffOptions.IgnoreChangesAndAdditions {
			return nil, nil, nil
		}

		// Old slice was empty - add all elements from the new slice
		return modified, nil, nil
	}

	elementType, err := sliceElementType(original, modified)
	if err != nil {
		return nil, nil, err
	}

	switch elementType.Kind() {
	case reflect.Map:
		patchList, err := diffListsOfMaps(original, modified, t, mergeKey, diffOptions)
		return patchList, nil, err
	case reflect.Slice:
		// Lists of Lists are not permitted by the api
		return nil, nil, mergepatch.ErrNoListOfLists
	default:
		return diffListsOfScalars(original, modified, diffOptions)
	}
}

// diffListsOfScalars returns 2 lists, the first one is addList and the second one is deletionList.
// Argument diffOptions.IgnoreChangesAndAdditions controls if calculate addList. true means not calculate.
// Argument diffOptions.IgnoreDeletions controls if calculate deletionList. true means not calculate.
func diffListsOfScalars(original, modified []interface{}, diffOptions DiffOptions) ([]interface{}, []interface{}, error) {
	// Sort the scalars for easier calculating the diff
	originalScalars := sortScalars(original)
	modifiedScalars := sortScalars(modified)

	originalIndex, modifiedIndex := 0, 0
	addList := []interface{}{}
	deletionList := []interface{}{}

	for {
		originalInBounds := originalIndex < len(originalScalars)
		modifiedInBounds := modifiedIndex < len(modifiedScalars)
		if !originalInBounds && !modifiedInBounds {
			break
		}
		// we need to compare the string representation of the scalar,
		// because the scalar is an interface which doesn't support either < or >
		// And that's how func sortScalars compare scalars.
		var originalString, modifiedString string
		var originalValue, modifiedValue interface{}
		if originalInBounds {
			originalValue = originalScalars[originalIndex]
			originalString = fmt.Sprintf("%v", originalValue)
		}
		if modifiedInBounds {
			modifiedValue = modifiedScalars[modifiedIndex]
			modifiedString = fmt.Sprintf("%v", modifiedValue)
		}

		originalV, modifiedV := compareListValuesAtIndex(originalInBounds, modifiedInBounds, originalString, modifiedString)
		switch {
		case originalV == nil && modifiedV == nil:
			originalIndex++
			modifiedIndex++
		case originalV != nil && modifiedV == nil:
			if !diffOptions.IgnoreDeletions {
				deletionList = append(deletionList, originalValue)
			}
			originalIndex++
		case originalV == nil && modifiedV != nil:
			if !diffOptions.IgnoreChangesAndAdditions {
				addList = append(addList, modifiedValue)
			}
			modifiedIndex++
		default:
			return nil, nil, fmt.Errorf("Unexpected returned value from compareListValuesAtIndex: %v and %v", originalV, modifiedV)
		}
	}

	return addList, deletionList, nil
}

// If first return value is non-nil, list1 contains an element not present in list2
// If second return value is non-nil, list2 contains an element not present in list1
func compareListValuesAtIndex(list1Inbounds, list2Inbounds bool, list1Value, list2Value string) (interface{}, interface{}) {
	bothInBounds := list1Inbounds && list2Inbounds
	switch {
	// scalars are identical
	case bothInBounds && list1Value == list2Value:
		return nil, nil
	// only list2 is in bound
	case !list1Inbounds:
		fallthrough
	// list2 has additional scalar
	case bothInBounds && list1Value > list2Value:
		return nil, list2Value
	// only original is in bound
	case !list2Inbounds:
		fallthrough
	// original has additional scalar
	case bothInBounds && list1Value < list2Value:
		return list1Value, nil
	default:
		return nil, nil
	}
}

// Returns a (recursive) strategic merge patch that yields modified when applied to original,
// for a pair of lists of maps with merge semantics.
func diffListsOfMaps(original, modified []interface{}, t reflect.Type, mergeKey string, diffOptions DiffOptions) ([]interface{}, error) {
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
	for {
		originalInBounds := originalIndex < len(originalSorted)
		modifiedInBounds := modifiedIndex < len(modifiedSorted)
		bothInBounds := originalInBounds && modifiedInBounds
		if !originalInBounds && !modifiedInBounds {
			break
		}

		var originalElementMergeKeyValueString, modifiedElementMergeKeyValueString string
		var originalElementMergeKeyValue, modifiedElementMergeKeyValue interface{}
		var originalElement, modifiedElement map[string]interface{}
		if originalInBounds {
			originalElement, originalElementMergeKeyValue, err = getMapAndMergeKeyValueByIndex(originalIndex, mergeKey, originalSorted)
			if err != nil {
				return nil, err
			}
			originalElementMergeKeyValueString = fmt.Sprintf("%v", originalElementMergeKeyValue)
		}
		if modifiedInBounds {
			modifiedElement, modifiedElementMergeKeyValue, err = getMapAndMergeKeyValueByIndex(modifiedIndex, mergeKey, modifiedSorted)
			if err != nil {
				return nil, err
			}
			modifiedElementMergeKeyValueString = fmt.Sprintf("%v", modifiedElementMergeKeyValue)
		}

		switch {
		case bothInBounds && ItemMatchesOriginalAndModifiedSlice(originalElementMergeKeyValueString, modifiedElementMergeKeyValueString):
			// Merge key values are equal, so recurse
			patchValue, err := diffMaps(originalElement, modifiedElement, t, diffOptions)
			if err != nil {
				return nil, err
			}
			if len(patchValue) > 0 {
				patchValue[mergeKey] = modifiedElementMergeKeyValue
				patch = append(patch, patchValue)
			}
			originalIndex++
			modifiedIndex++
		// only modified is in bound
		case !originalInBounds:
			fallthrough
		// modified has additional map
		case bothInBounds && ItemAddedToModifiedSlice(originalElementMergeKeyValueString, modifiedElementMergeKeyValueString):
			if !diffOptions.IgnoreChangesAndAdditions {
				patch = append(patch, modifiedElement)
			}
			modifiedIndex++
		// only original is in bound
		case !modifiedInBounds:
			fallthrough
		// original has additional map
		case bothInBounds && ItemRemovedFromModifiedSlice(originalElementMergeKeyValueString, modifiedElementMergeKeyValueString):
			if !diffOptions.IgnoreDeletions {
				// Item was deleted, so add delete directive
				patch = append(patch, CreateDeleteDirective(mergeKey, originalElementMergeKeyValue))
			}
			originalIndex++
		}
	}

	return patch, nil
}

// getMapAndMergeKeyValueByIndex return a map in the list and its merge key value given the index of the map.
func getMapAndMergeKeyValueByIndex(index int, mergeKey string, listOfMaps []interface{}) (map[string]interface{}, interface{}, error) {
	m, ok := listOfMaps[index].(map[string]interface{})
	if !ok {
		return nil, nil, mergepatch.ErrBadArgType(m, listOfMaps[index])
	}

	val, ok := m[mergeKey]
	if !ok {
		return nil, nil, mergepatch.ErrNoMergeKey(m, mergeKey)
	}
	return m, val, nil
}

// StrategicMergePatch applies a strategic merge patch. The patch and the original document
// must be json encoded content. A patch can be created from an original and a modified document
// by calling CreateStrategicMergePatch.
func StrategicMergePatch(original, patch []byte, dataStruct interface{}) ([]byte, error) {
	originalMap, err := handleUnmarshal(original)
	if err != nil {
		return nil, err
	}
	patchMap, err := handleUnmarshal(patch)
	if err != nil {
		return nil, err
	}

	result, err := StrategicMergeMapPatch(originalMap, patchMap, dataStruct)
	if err != nil {
		return nil, err
	}

	return json.Marshal(result)
}

func handleUnmarshal(j []byte) (map[string]interface{}, error) {
	if j == nil {
		j = []byte("{}")
	}

	m := map[string]interface{}{}
	err := json.Unmarshal(j, &m)
	if err != nil {
		return nil, mergepatch.ErrBadJSONDoc
	}
	return m, nil
}

// StrategicMergePatch applies a strategic merge patch. The original and patch documents
// must be JSONMap. A patch can be created from an original and modified document by
// calling CreateTwoWayMergeMapPatch.
// Warning: the original and patch JSONMap objects are mutated by this function and should not be reused.
func StrategicMergeMapPatch(original, patch JSONMap, dataStruct interface{}) (JSONMap, error) {
	t, err := getTagStructType(dataStruct)
	if err != nil {
		return nil, err
	}
	mergeOptions := MergeOptions{
		MergeParallelList:    true,
		IgnoreUnmatchedNulls: true,
	}
	return mergeMap(original, patch, t, mergeOptions)
}

func getTagStructType(dataStruct interface{}) (reflect.Type, error) {
	if dataStruct == nil {
		return nil, mergepatch.ErrBadArgKind(struct{}{}, nil)
	}

	t := reflect.TypeOf(dataStruct)
	// Get the underlying type for pointers
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	if t.Kind() != reflect.Struct {
		return nil, mergepatch.ErrBadArgKind(struct{}{}, dataStruct)
	}

	return t, nil
}

// handleDirectiveInMergeMap handles the patch directive when merging 2 maps.
func handleDirectiveInMergeMap(directive interface{}, patch map[string]interface{}) (map[string]interface{}, error) {
	if directive == replaceDirective {
		// If the patch contains "$patch: replace", don't merge it, just use the
		// patch directly. Later on, we can add a single level replace that only
		// affects the map that the $patch is in.
		delete(patch, directiveMarker)
		return patch, nil
	}

	if directive == deleteDirective {
		// If the patch contains "$patch: delete", don't merge it, just return
		//  an empty map.
		return map[string]interface{}{}, nil
	}

	return nil, mergepatch.ErrBadPatchType(directive, patch)
}

// preprocessDeletionListForMerging preprocesses the deletion list.
// it returns shouldContinue, isDeletionList, noPrefixKey
func preprocessDeletionListForMerging(key string, original map[string]interface{},
	patchVal interface{}, mergeDeletionList bool) (bool, bool, string, error) {
	// If found a parallel list for deletion and we are going to merge the list,
	// overwrite the key to the original key and set flag isDeleteList
	foundParallelListPrefix := strings.HasPrefix(key, deleteFromPrimitiveListDirectivePrefix)
	if foundParallelListPrefix {
		if !mergeDeletionList {
			original[key] = patchVal
			return true, false, "", nil
		}
		substrings := strings.SplitN(key, "/", 2)
		if len(substrings) <= 1 {
			return false, false, "", mergepatch.ErrBadPatchFormatForPrimitiveList
		}
		return false, true, substrings[1], nil
	}
	return false, false, "", nil
}

// applyRetainKeysDirective looks for a retainKeys directive and applies to original
// - if no directive exists do nothing
// - if directive is found, clear keys in original missing from the directive list
// - validate that all keys present in the patch are present in the retainKeys directive
// note: original may be another patch request, e.g. applying the add+modified patch to the deletions patch. In this case it may have directives
func applyRetainKeysDirective(original, patch map[string]interface{}, options MergeOptions) error {
	retainKeysInPatch, foundInPatch := patch[retainKeysDirective]
	if !foundInPatch {
		return nil
	}
	// cleanup the directive
	delete(patch, retainKeysDirective)

	if !options.MergeParallelList {
		// If original is actually a patch, make sure the retainKeys directives are the same in both patches if present in both.
		// If not present in the original patch, copy from the modified patch.
		retainKeysInOriginal, foundInOriginal := original[retainKeysDirective]
		if foundInOriginal {
			if !reflect.DeepEqual(retainKeysInOriginal, retainKeysInPatch) {
				// This error actually should never happen.
				return fmt.Errorf("%v and %v are not deep equal: this may happen when calculating the 3-way diff patch", retainKeysInOriginal, retainKeysInPatch)
			}
		} else {
			original[retainKeysDirective] = retainKeysInPatch
		}
		return nil
	}

	retainKeysList, ok := retainKeysInPatch.([]interface{})
	if !ok {
		return mergepatch.ErrBadPatchFormatForRetainKeys
	}

	// validate patch to make sure all fields in the patch are present in the retainKeysList.
	// The map is used only as a set, the value is never referenced
	m := map[interface{}]struct{}{}
	for _, v := range retainKeysList {
		m[v] = struct{}{}
	}
	for k, v := range patch {
		if v == nil || strings.HasPrefix(k, deleteFromPrimitiveListDirectivePrefix) {
			continue
		}
		// If there is an item present in the patch but not in the retainKeys list,
		// the patch is invalid.
		if _, found := m[k]; !found {
			return mergepatch.ErrBadPatchFormatForRetainKeys
		}
	}

	// clear not present fields
	for k := range original {
		if _, found := m[k]; !found {
			delete(original, k)
		}
	}
	return nil
}

// Merge fields from a patch map into the original map. Note: This may modify
// both the original map and the patch because getting a deep copy of a map in
// golang is highly non-trivial.
// flag mergeOptions.MergeParallelList controls if using the parallel list to delete or keeping the list.
// If patch contains any null field (e.g. field_1: null) that is not
// present in original, then to propagate it to the end result use
// mergeOptions.IgnoreUnmatchedNulls == false.
func mergeMap(original, patch map[string]interface{}, t reflect.Type, mergeOptions MergeOptions) (map[string]interface{}, error) {
	if v, ok := patch[directiveMarker]; ok {
		return handleDirectiveInMergeMap(v, patch)
	}

	// nil is an accepted value for original to simplify logic in other places.
	// If original is nil, replace it with an empty map and then apply the patch.
	if original == nil {
		original = map[string]interface{}{}
	}

	err := applyRetainKeysDirective(original, patch, mergeOptions)
	if err != nil {
		return nil, err
	}

	// Start merging the patch into the original.
	for k, patchV := range patch {
		skipProcessing, isDeleteList, noPrefixKey, err := preprocessDeletionListForMerging(k, original, patchV, mergeOptions.MergeParallelList)
		if err != nil {
			return nil, err
		}
		if skipProcessing {
			continue
		}
		if len(noPrefixKey) > 0 {
			k = noPrefixKey
		}

		// If the value of this key is null, delete the key if it exists in the
		// original. Otherwise, check if we want to preserve it or skip it.
		// Preserving the null value is useful when we want to send an explicit
		// delete to the API server.
		if patchV == nil {
			if _, ok := original[k]; ok {
				delete(original, k)
			}
			if mergeOptions.IgnoreUnmatchedNulls {
				continue
			}
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

		originalType := reflect.TypeOf(original[k])
		patchType := reflect.TypeOf(patchV)
		if originalType != patchType {
			original[k] = patchV
			continue
		}
		// If they're both maps or lists, recurse into the value.
		// First find the fieldPatchStrategy and fieldPatchMergeKey.
		fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, k)
		if err != nil {
			return nil, err
		}
		_, patchStrategy, err := extractRetainKeysPatchStrategy(fieldPatchStrategies)
		if err != nil {
			return nil, err
		}

		switch originalType.Kind() {
		case reflect.Map:

			original[k], err = mergeMapHandler(original[k], patchV, fieldType, patchStrategy, mergeOptions)
		case reflect.Slice:
			original[k], err = mergeSliceHandler(original[k], patchV, fieldType, patchStrategy, fieldPatchMergeKey, isDeleteList, mergeOptions)
		default:
			original[k] = patchV
		}
		if err != nil {
			return nil, err
		}
	}
	return original, nil
}

// mergeMapHandler handles how to merge `patchV` whose key is `key` with `original` respecting
// fieldPatchStrategy and mergeOptions.
func mergeMapHandler(original, patch interface{}, fieldType reflect.Type,
	fieldPatchStrategy string, mergeOptions MergeOptions) (map[string]interface{}, error) {
	typedOriginal, typedPatch, err := mapTypeAssertion(original, patch)
	if err != nil {
		return nil, err
	}

	if fieldPatchStrategy != replaceDirective {
		return mergeMap(typedOriginal, typedPatch, fieldType, mergeOptions)
	} else {
		return typedPatch, nil
	}
}

// mergeSliceHandler handles how to merge `patchV` whose key is `key` with `original` respecting
// fieldPatchStrategy, fieldPatchMergeKey, isDeleteList and mergeOptions.
func mergeSliceHandler(original, patch interface{}, fieldType reflect.Type,
	fieldPatchStrategy, fieldPatchMergeKey string, isDeleteList bool, mergeOptions MergeOptions) ([]interface{}, error) {
	typedOriginal, typedPatch, err := sliceTypeAssertion(original, patch)
	if err != nil {
		return nil, err
	}

	if fieldPatchStrategy == mergeDirective {
		elemType := fieldType.Elem()
		return mergeSlice(typedOriginal, typedPatch, elemType, fieldPatchMergeKey, mergeOptions, isDeleteList)
	} else {
		return typedPatch, nil
	}
}

// Merge two slices together. Note: This may modify both the original slice and
// the patch because getting a deep copy of a slice in golang is highly
// non-trivial.
func mergeSlice(original, patch []interface{}, elemType reflect.Type, mergeKey string, mergeOptions MergeOptions, isDeleteList bool) ([]interface{}, error) {
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
		if mergeOptions.MergeParallelList && isDeleteList {
			return deleteFromSlice(original, patch), nil
		}
		// Maybe in the future add a "concat" mode that doesn't
		// uniqify.
		both := append(original, patch...)
		return uniqifyScalars(both), nil
	}

	if mergeKey == "" {
		return nil, fmt.Errorf("cannot merge lists without merge key for type %s", elemType.Kind().String())
	}

	original, patch, err = mergeSliceWithSpecialElements(original, patch, mergeKey)
	if err != nil {
		return nil, err
	}

	return mergeSliceWithoutSpecialElements(original, patch, mergeKey, elemType, mergeOptions)
}

// mergeSliceWithSpecialElements handles special elements with directiveMarker
// before merging the slices. It returns a updated `original` and a patch without special elements.
// original and patch must be slices of maps, they should be checked before calling this function.
func mergeSliceWithSpecialElements(original, patch []interface{}, mergeKey string) ([]interface{}, []interface{}, error) {
	patchWithoutSpecialElements := []interface{}{}
	replace := false
	for _, v := range patch {
		typedV := v.(map[string]interface{})
		patchType, ok := typedV[directiveMarker]
		if !ok {
			patchWithoutSpecialElements = append(patchWithoutSpecialElements, v)
		} else {
			switch patchType {
			case deleteDirective:
				mergeValue, ok := typedV[mergeKey]
				if ok {
					var err error
					original, err = deleteMatchingEntries(original, mergeKey, mergeValue)
					if err != nil {
						return nil, nil, err
					}
				} else {
					return nil, nil, fmt.Errorf("delete patch type with no merge key defined")
				}
			case replaceDirective:
				replace = true
				// Continue iterating through the array to prune any other $patch elements.
			case mergeDirective:
				return nil, nil, fmt.Errorf("merging lists cannot yet be specified in the patch")
			default:
				return nil, nil, mergepatch.ErrBadPatchType(patchType, typedV)
			}
		}
	}
	if replace {
		return patchWithoutSpecialElements, nil, nil
	}
	return original, patchWithoutSpecialElements, nil
}

// delete all matching entries (based on merge key) from a merging list
func deleteMatchingEntries(original []interface{}, mergeKey string, mergeValue interface{}) ([]interface{}, error) {
	for {
		_, originalKey, found, err := findMapInSliceBasedOnKeyValue(original, mergeKey, mergeValue)
		if err != nil {
			return nil, err
		}

		if !found {
			break
		}
		// Delete the element at originalKey.
		original = append(original[:originalKey], original[originalKey+1:]...)
	}
	return original, nil
}

// mergeSliceWithoutSpecialElements merges slices with non-special elements.
// original and patch must be slices of maps, they should be checked before calling this function.
func mergeSliceWithoutSpecialElements(original, patch []interface{}, mergeKey string, elemType reflect.Type, mergeOptions MergeOptions) ([]interface{}, error) {
	for _, v := range patch {
		typedV := v.(map[string]interface{})
		mergeValue, ok := typedV[mergeKey]
		if !ok {
			return nil, mergepatch.ErrNoMergeKey(typedV, mergeKey)
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
			mergedMaps, err = mergeMap(originalMap, typedV, elemType, mergeOptions)
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

// deleteFromSlice uses the parallel list to delete the items in a list of scalars
func deleteFromSlice(current, toDelete []interface{}) []interface{} {
	currentScalars := uniqifyAndSortScalars(current)
	toDeleteScalars := uniqifyAndSortScalars(toDelete)

	currentIndex, toDeleteIndex := 0, 0
	mergedList := []interface{}{}

	for currentIndex < len(currentScalars) && toDeleteIndex < len(toDeleteScalars) {
		originalString := fmt.Sprintf("%v", currentScalars[currentIndex])
		modifiedString := fmt.Sprintf("%v", toDeleteScalars[toDeleteIndex])

		switch {
		// found an item to delete
		case originalString == modifiedString:
			currentIndex++
		// Request to delete an item that was not found in the current list
		case originalString > modifiedString:
			toDeleteIndex++
		// Found an item that was not part of the deletion list, keep it
		case originalString < modifiedString:
			mergedList = append(mergedList, currentScalars[currentIndex])
			currentIndex++
		}
	}
	return append(mergedList, currentScalars[currentIndex:]...)
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

// Function sortMergeListsByNameMap recursively sorts the merge lists by its mergeKey in a map.
func sortMergeListsByNameMap(s map[string]interface{}, t reflect.Type) (map[string]interface{}, error) {
	newS := map[string]interface{}{}
	for k, v := range s {
		if k == retainKeysDirective {
			typedV, ok := v.([]interface{})
			if !ok {
				return nil, mergepatch.ErrBadPatchFormatForRetainKeys
			}
			v = sortScalars(typedV)
		} else if strings.HasPrefix(k, deleteFromPrimitiveListDirectivePrefix) {
			typedV, ok := v.([]interface{})
			if !ok {
				return nil, mergepatch.ErrBadPatchFormatForPrimitiveList
			}
			v = uniqifyAndSortScalars(typedV)
		} else if k != directiveMarker {
			fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(t, k)
			if err != nil {
				return nil, err
			}
			_, patchStrategy, err := extractRetainKeysPatchStrategy(fieldPatchStrategies)
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
				if patchStrategy == mergeDirective {
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

// Function sortMergeListsByNameMap recursively sorts the merge lists by its mergeKey in an array.
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
	return sortScalars(s)
}

func sortScalars(s []interface{}) []interface{} {
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
					return nil, mergepatch.ErrNoListOfLists
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

// MergingMapsHaveConflicts returns true if the left and right JSON interface
// objects overlap with different values in any key. All keys are required to be
// strings. Since patches of the same Type have congruent keys, this is valid
// for multiple patch types. This method supports strategic merge patch semantics.
func MergingMapsHaveConflicts(left, right map[string]interface{}, dataStruct interface{}) (bool, error) {
	t, err := getTagStructType(dataStruct)
	if err != nil {
		return true, err
	}

	return mergingMapFieldsHaveConflicts(left, right, t, "", "")
}

func mergingMapFieldsHaveConflicts(
	left, right interface{},
	fieldType reflect.Type,
	fieldPatchStrategy, fieldPatchMergeKey string,
) (bool, error) {
	switch leftType := left.(type) {
	case map[string]interface{}:
		rightType, ok := right.(map[string]interface{})
		if !ok {
			return true, nil
		}
		leftMarker, okLeft := leftType[directiveMarker]
		rightMarker, okRight := rightType[directiveMarker]
		// if one or the other has a directive marker,
		// then we need to consider that before looking at the individual keys,
		// since a directive operates on the whole map.
		if okLeft || okRight {
			// if one has a directive marker and the other doesn't,
			// then we have a conflict, since one is deleting or replacing the whole map,
			// and the other is doing things to individual keys.
			if okLeft != okRight {
				return true, nil
			}
			// if they both have markers, but they are not the same directive,
			// then we have a conflict because they're doing different things to the map.
			if leftMarker != rightMarker {
				return true, nil
			}
		}
		if fieldPatchStrategy == replaceDirective {
			return false, nil
		}
		// Check the individual keys.
		return mapsHaveConflicts(leftType, rightType, fieldType)

	case []interface{}:
		rightType, ok := right.([]interface{})
		if !ok {
			return true, nil
		}
		return slicesHaveConflicts(leftType, rightType, fieldType, fieldPatchStrategy, fieldPatchMergeKey)

	case string, float64, bool, int, int64, nil:
		return !reflect.DeepEqual(left, right), nil
	default:
		return true, fmt.Errorf("unknown type: %v", reflect.TypeOf(left))
	}
}

func mapsHaveConflicts(typedLeft, typedRight map[string]interface{}, structType reflect.Type) (bool, error) {
	for key, leftValue := range typedLeft {
		if key != directiveMarker && key != retainKeysDirective {
			if rightValue, ok := typedRight[key]; ok {
				fieldType, fieldPatchStrategies, fieldPatchMergeKey, err := forkedjson.LookupPatchMetadata(structType, key)
				if err != nil {
					return true, err
				}
				_, patchStrategy, err := extractRetainKeysPatchStrategy(fieldPatchStrategies)
				if err != nil {
					return true, err
				}

				if hasConflicts, err := mergingMapFieldsHaveConflicts(leftValue, rightValue,
					fieldType, patchStrategy, fieldPatchMergeKey); hasConflicts {
					return true, err
				}
			}
		}
	}

	return false, nil
}

func slicesHaveConflicts(
	typedLeft, typedRight []interface{},
	fieldType reflect.Type,
	fieldPatchStrategy, fieldPatchMergeKey string,
) (bool, error) {
	elementType, err := sliceElementType(typedLeft, typedRight)
	if err != nil {
		return true, err
	}

	valueType := fieldType.Elem()
	if fieldPatchStrategy == mergeDirective {
		// Merging lists of scalars have no conflicts by definition
		// So we only need to check further if the elements are maps
		if elementType.Kind() != reflect.Map {
			return false, nil
		}

		// Build a map for each slice and then compare the two maps
		leftMap, err := sliceOfMapsToMapOfMaps(typedLeft, fieldPatchMergeKey)
		if err != nil {
			return true, err
		}

		rightMap, err := sliceOfMapsToMapOfMaps(typedRight, fieldPatchMergeKey)
		if err != nil {
			return true, err
		}

		return mapsOfMapsHaveConflicts(leftMap, rightMap, valueType)
	}

	// Either we don't have type information, or these are non-merging lists
	if len(typedLeft) != len(typedRight) {
		return true, nil
	}

	// Sort scalar slices to prevent ordering issues
	// We have no way to sort non-merging lists of maps
	if elementType.Kind() != reflect.Map {
		typedLeft = uniqifyAndSortScalars(typedLeft)
		typedRight = uniqifyAndSortScalars(typedRight)
	}

	// Compare the slices element by element in order
	// This test will fail if the slices are not sorted
	for i := range typedLeft {
		if hasConflicts, err := mergingMapFieldsHaveConflicts(typedLeft[i], typedRight[i], valueType, "", ""); hasConflicts {
			return true, err
		}
	}

	return false, nil
}

func sliceOfMapsToMapOfMaps(slice []interface{}, mergeKey string) (map[string]interface{}, error) {
	result := make(map[string]interface{}, len(slice))
	for _, value := range slice {
		typedValue, ok := value.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("invalid element type in merging list:%v", slice)
		}

		mergeValue, ok := typedValue[mergeKey]
		if !ok {
			return nil, fmt.Errorf("cannot find merge key `%s` in merging list element:%v", mergeKey, typedValue)
		}

		result[fmt.Sprintf("%s", mergeValue)] = typedValue
	}

	return result, nil
}

func mapsOfMapsHaveConflicts(typedLeft, typedRight map[string]interface{}, structType reflect.Type) (bool, error) {
	for key, leftValue := range typedLeft {
		if rightValue, ok := typedRight[key]; ok {
			if hasConflicts, err := mergingMapFieldsHaveConflicts(leftValue, rightValue, structType, "", ""); hasConflicts {
				return true, err
			}
		}
	}

	return false, nil
}

// CreateThreeWayMergePatch reconciles a modified configuration with an original configuration,
// while preserving any changes or deletions made to the original configuration in the interim,
// and not overridden by the current configuration. All three documents must be passed to the
// method as json encoded content. It will return a strategic merge patch, or an error if any
// of the documents is invalid, or if there are any preconditions that fail against the modified
// configuration, or, if overwrite is false and there are conflicts between the modified and current
// configurations. Conflicts are defined as keys changed differently from original to modified
// than from original to current. In other words, a conflict occurs if modified changes any key
// in a way that is different from how it is changed in current (e.g., deleting it, changing its
// value). We also propagate values fields that do not exist in original but are explicitly
// defined in modified.
func CreateThreeWayMergePatch(original, modified, current []byte, dataStruct interface{}, overwrite bool, fns ...mergepatch.PreconditionFunc) ([]byte, error) {
	originalMap := map[string]interface{}{}
	if len(original) > 0 {
		if err := json.Unmarshal(original, &originalMap); err != nil {
			return nil, mergepatch.ErrBadJSONDoc
		}
	}

	modifiedMap := map[string]interface{}{}
	if len(modified) > 0 {
		if err := json.Unmarshal(modified, &modifiedMap); err != nil {
			return nil, mergepatch.ErrBadJSONDoc
		}
	}

	currentMap := map[string]interface{}{}
	if len(current) > 0 {
		if err := json.Unmarshal(current, &currentMap); err != nil {
			return nil, mergepatch.ErrBadJSONDoc
		}
	}

	t, err := getTagStructType(dataStruct)
	if err != nil {
		return nil, err
	}

	// The patch is the difference from current to modified without deletions, plus deletions
	// from original to modified. To find it, we compute deletions, which are the deletions from
	// original to modified, and delta, which is the difference from current to modified without
	// deletions, and then apply delta to deletions as a patch, which should be strictly additive.
	deltaMapDiffOptions := DiffOptions{
		IgnoreDeletions: true,
	}
	deltaMap, err := diffMaps(currentMap, modifiedMap, t, deltaMapDiffOptions)
	if err != nil {
		return nil, err
	}
	deletionsMapDiffOptions := DiffOptions{
		IgnoreChangesAndAdditions: true,
	}
	deletionsMap, err := diffMaps(originalMap, modifiedMap, t, deletionsMapDiffOptions)
	if err != nil {
		return nil, err
	}

	mergeOptions := MergeOptions{}
	patchMap, err := mergeMap(deletionsMap, deltaMap, t, mergeOptions)
	if err != nil {
		return nil, err
	}

	// Apply the preconditions to the patch, and return an error if any of them fail.
	for _, fn := range fns {
		if !fn(patchMap) {
			return nil, mergepatch.NewErrPreconditionFailed(patchMap)
		}
	}

	// If overwrite is false, and the patch contains any keys that were changed differently,
	// then return a conflict error.
	if !overwrite {
		changeMapDiffOptions := DiffOptions{}
		changedMap, err := diffMaps(originalMap, currentMap, t, changeMapDiffOptions)
		if err != nil {
			return nil, err
		}

		hasConflicts, err := MergingMapsHaveConflicts(patchMap, changedMap, dataStruct)
		if err != nil {
			return nil, err
		}

		if hasConflicts {
			return nil, mergepatch.NewErrConflict(mergepatch.ToYAMLOrError(patchMap), mergepatch.ToYAMLOrError(changedMap))
		}
	}

	return json.Marshal(patchMap)
}

func ItemAddedToModifiedSlice(original, modified string) bool { return original > modified }

func ItemRemovedFromModifiedSlice(original, modified string) bool { return original < modified }

func ItemMatchesOriginalAndModifiedSlice(original, modified string) bool { return original == modified }

func CreateDeleteDirective(mergeKey string, mergeKeyValue interface{}) map[string]interface{} {
	return map[string]interface{}{mergeKey: mergeKeyValue, directiveMarker: deleteDirective}
}

func mapTypeAssertion(original, patch interface{}) (map[string]interface{}, map[string]interface{}, error) {
	typedOriginal, ok := original.(map[string]interface{})
	if !ok {
		return nil, nil, mergepatch.ErrBadArgType(typedOriginal, original)
	}
	typedPatch, ok := patch.(map[string]interface{})
	if !ok {
		return nil, nil, mergepatch.ErrBadArgType(typedPatch, patch)
	}
	return typedOriginal, typedPatch, nil
}

func sliceTypeAssertion(original, patch interface{}) ([]interface{}, []interface{}, error) {
	typedOriginal, ok := original.([]interface{})
	if !ok {
		return nil, nil, mergepatch.ErrBadArgType(typedOriginal, original)
	}
	typedPatch, ok := patch.([]interface{})
	if !ok {
		return nil, nil, mergepatch.ErrBadArgType(typedPatch, patch)
	}
	return typedOriginal, typedPatch, nil
}

// extractRetainKeysPatchStrategy process patch strategy, which is a string may contains multiple
// patch strategies seperated by ",". It returns a boolean var indicating if it has
// retainKeys strategies and a string for the other strategy.
func extractRetainKeysPatchStrategy(strategies []string) (bool, string, error) {
	switch len(strategies) {
	case 0:
		return false, "", nil
	case 1:
		singleStrategy := strategies[0]
		switch singleStrategy {
		case retainKeysStrategy:
			return true, "", nil
		default:
			return false, singleStrategy, nil
		}
	case 2:
		switch {
		case strategies[0] == retainKeysStrategy:
			return true, strategies[1], nil
		case strategies[1] == retainKeysStrategy:
			return true, strategies[0], nil
		default:
			return false, "", fmt.Errorf("unexpected patch strategy: %v", strategies)
		}
	default:
		return false, "", fmt.Errorf("unexpected patch strategy: %v", strategies)
	}
}

// hasAdditionalNewField returns if original map has additional key with non-nil value than modified.
func hasAdditionalNewField(original, modified map[string]interface{}) bool {
	for k, v := range original {
		if v == nil {
			continue
		}
		if _, found := modified[k]; !found {
			return true
		}
	}
	return false
}
