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
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/ghodss/yaml"
)

type StrategicMergePatchTestCases struct {
	TestCases []StrategicMergePatchTestCase
}

type SortMergeListTestCases struct {
	TestCases []SortMergeListTestCase
}

type StrategicMergePatchTestCaseData struct {
	Original map[string]interface{}
	Patch    map[string]interface{}
	Modified map[string]interface{}
}

type StrategicMergePatchTestCase struct {
	Description string
	StrategicMergePatchTestCaseData
}

type SortMergeListTestCase struct {
	Description string
	Original    map[string]interface{}
	Sorted      map[string]interface{}
}

type MergeItem struct {
	Name              string
	Value             string
	Other             string
	MergingList       []MergeItem `patchStrategy:"merge" patchMergeKey:"name"`
	NonMergingList    []MergeItem
	MergingIntList    []int `patchStrategy:"merge"`
	NonMergingIntList []int
	MergeItemPtr      *MergeItem `patchStrategy:"merge" patchMergeKey:"name"`
	SimpleMap         map[string]string
}

// These are test cases for SortMergeList, used to assert that it (recursively)
// sorts both merging and non merging lists correctly.
var sortMergeListTestCaseData = []byte(`
testCases:
  - description: sort one list of maps
    original:
      mergingList:
        - name: 1
        - name: 3
        - name: 2
    sorted:
      mergingList:
        - name: 1
        - name: 2
        - name: 3
  - description: sort lists of maps but not nested lists of maps
    original:
      mergingList:
        - name: 2
          nonMergingList:
            - name: 1
            - name: 3
            - name: 2
        - name: 1
          nonMergingList:
            - name: 2
            - name: 1
    sorted:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 2
            - name: 1
        - name: 2
          nonMergingList:
            - name: 1
            - name: 3
            - name: 2
  - description: sort lists of maps and nested lists of maps
    fieldTypes:
    original:
      mergingList:
        - name: 2
          mergingList:
            - name: 1
            - name: 3
            - name: 2
        - name: 1
          mergingList:
            - name: 2
            - name: 1
    sorted:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
            - name: 2
        - name: 2
          mergingList:
            - name: 1
            - name: 2
            - name: 3
  - description: merging list should NOT sort when nested in non merging list
    original:
      nonMergingList:
        - name: 2
          mergingList:
            - name: 1
            - name: 3
            - name: 2
        - name: 1
          mergingList:
            - name: 2
            - name: 1
    sorted:
      nonMergingList:
        - name: 2
          mergingList:
            - name: 1
            - name: 3
            - name: 2
        - name: 1
          mergingList:
            - name: 2
            - name: 1
  - description: sort very nested list of maps
    fieldTypes:
    original:
      mergingList:
        - mergingList:
            - mergingList:
                - name: 2
                - name: 1
    sorted:
      mergingList:
        - mergingList:
            - mergingList:
                - name: 1
                - name: 2
  - description: sort nested lists of ints
    original:
      mergingList:
        - name: 2
          mergingIntList:
            - 1
            - 3
            - 2
        - name: 1
          mergingIntList:
            - 2
            - 1
    sorted:
      mergingList:
        - name: 1
          mergingIntList:
            - 1
            - 2
        - name: 2
          mergingIntList:
            - 1
            - 2
            - 3
  - description: sort nested pointers of ints
    original:
      mergeItemPtr:
        - name: 2
          mergingIntList:
            - 1
            - 3
            - 2
        - name: 1
          mergingIntList:
            - 2
            - 1
    sorted:
      mergeItemPtr:
        - name: 1
          mergingIntList:
            - 1
            - 2
        - name: 2
          mergingIntList:
            - 1
            - 2
            - 3
  - description: sort merging list by pointer
    original:
      mergeItemPtr:
        - name: 1
        - name: 3
        - name: 2
    sorted:
      mergeItemPtr:
        - name: 1
        - name: 2
        - name: 3
`)

func TestSortMergeLists(t *testing.T) {
	tc := SortMergeListTestCases{}
	err := yaml.Unmarshal(sortMergeListTestCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v", err)
		return
	}

	var e MergeItem
	for _, c := range tc.TestCases {
		sorted, err := sortMergeListsByName(toJSONOrFail(c.Original, t), e)
		if err != nil {
			t.Errorf("sort arrays returned error: %v", err)
		}

		if !reflect.DeepEqual(sorted, toJSONOrFail(c.Sorted, t)) {
			t.Errorf("sorting failed: %v\ntried to sort:\n%vexpected:\n%vgot:\n%v",
				c.Description, toYAMLOrError(c.Original), toYAMLOrError(c.Sorted), jsonToYAMLOrError(sorted))
		}
	}
}

// These are test cases for StrategicMergePatch that cannot be generated using
// CreateStrategicMergePatch because it doesn't use the replace directive, generate
// duplicate integers for a merging list patch, or generate empty merging lists.
var customStrategicMergePatchTestCaseData = []byte(`
testCases:
  - description: unique scalars when merging lists
    original:
      mergingIntList:
        - 1
        - 2
    patch:
      mergingIntList:
        - 2
        - 3
    modified:
      mergingIntList:
        - 1
        - 2
        - 3
  - description: delete all items from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - $patch: replace
    modified:
      mergingList: []
  - description: merge empty merging lists
    original:
      mergingList: []
    patch:
      mergingList: []
    modified:
      mergingList: []
  - description: delete all keys from map
    original:
      name: 1
      value: 1
    patch:
      $patch: replace
    modified: {}
  - description: add key and delete all keys from map
    original:
      name: 1
      value: 1
    patch:
      other: a
      $patch: replace
    modified:
      other: a
`)

func TestCustomStrategicMergePatch(t *testing.T) {
	tc := StrategicMergePatchTestCases{}
	err := yaml.Unmarshal(customStrategicMergePatchTestCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v", err)
		return
	}

	for _, c := range tc.TestCases {
		cOriginal, cPatch, cModified := testCaseToJSONOrFail(t, c)
		testPatchApplication(t, cOriginal, cPatch, cModified, c.Description)
	}
}

func testCaseToJSONOrFail(t *testing.T, c StrategicMergePatchTestCase) ([]byte, []byte, []byte) {
	var e MergeItem
	cOriginal := toJSONOrFail(c.Original, t)
	cPatch, err := sortMergeListsByName(toJSONOrFail(c.Patch, t), e)
	if err != nil {
		t.Errorf("error:%v sorting patch object:\n%v", err, c.Patch)
	}

	cModified, err := sortMergeListsByName(toJSONOrFail(c.Modified, t), e)
	if err != nil {
		t.Errorf("error: %v sorting modified object:\n%v", err, c.Modified)
	}

	return cOriginal, cPatch, cModified
}

func testPatchApplication(t *testing.T, cOriginal, cPatch, cModified []byte, description string) {
	var e MergeItem
	modified, err := StrategicMergePatch(cOriginal, cPatch, e)
	if err != nil {
		t.Errorf("error applying patch: %v:\noriginal:\n%vpatch:\n%v",
			err, jsonToYAMLOrError(cOriginal), jsonToYAMLOrError(cPatch))
	}

	// Sort the lists that have merged maps, since order is not significant.
	modified, err = sortMergeListsByName(modified, e)
	if err != nil {
		t.Errorf("error: %v sorting modified object:\n%v", err, modified)
	}

	if !reflect.DeepEqual(modified, cModified) {
		t.Errorf("patch application failed: %v\noriginal:\n%vpatch:\n%vexpected modified:\n%vgot modified:\n%v",
			description, jsonToYAMLOrError(cOriginal), jsonToYAMLOrError(cPatch),
			jsonToYAMLOrError(cModified), jsonToYAMLOrError(modified))
	}
}

// These are test cases for CreateStrategicMergePatch, used to assert that it
// generates the correct patch for a given outcome. They are also test cases for
// StrategicMergePatch, used to assert that applying a patch yields the correct
// outcome.
var createStrategicMergePatchTestCaseData = []byte(`
testCases:
  - description: add field to map
    original:
      name: 1
    patch:
      value: 1
    modified:
      name: 1
      value: 1
  - description: add field and delete field from map
    original:
      name: 1
    patch:
      name: null
      value: 1
    modified:
      value: 1
  - description: delete field from nested map
    original:
      simpleMap:
        key1: 1
        key2: 1
    patch:
      simpleMap:
        key2: null
    modified:
      simpleMap:
        key1: 1
  - description: delete all fields from map
    original:
      name: 1
      value: 1
    patch:
      name: null
      value: null
    modified: {}
  - description: add field and delete all fields from map
    original:
      name: 1
      value: 1
    patch:
      other: a
      name: null
      value: null
    modified:
      other: a
  - description: replace list of scalars
    original:
      nonMergingIntList:
        - 1
        - 2
    patch:
      nonMergingIntList:
        - 2
        - 3
    modified:
      nonMergingIntList:
        - 2
        - 3
  - description: merge lists of scalars
    original:
      mergingIntList:
        - 1
        - 2
    patch:
      mergingIntList:
        - 3
    modified:
      mergingIntList:
        - 1
        - 2
        - 3
  - description: merge lists of maps
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    patch:
      mergingList:
        - name: 3
          value: 3
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
        - name: 3
          value: 3
  - description: add field to map in merging list
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    patch:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
  - description: add duplicate field to map in merging list
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    patch:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
  - description: replace map field value in merging list
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    patch:
      mergingList:
        - name: 1
          value: a
    modified:
      mergingList:
        - name: 1
          value: a
        - name: 2
          value: 2
  - description: delete map from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
  - description: delete missing map from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
  - description: add map and delete map from merging list
    original:
      merginglist:
        - name: 1
        - name: 2
    patch:
      merginglist:
        - name: 1
          $patch: delete
        - name: 3
    modified:
      merginglist:
        - name: 2
        - name: 3
  - description: delete all maps from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
  - description: delete all maps from partially empty merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
  - description: delete all maps from empty merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
  - description: delete field from map in merging list
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    patch:
      mergingList:
        - name: 1
          value: null
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
  - description: replace non merging list nested in merging list
    original:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    patch:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
    modified:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
        - name: 2
  - description: add field to map in merging list nested in merging list
    original:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    patch:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
    modified:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
            - name: 2
              value: 2
        - name: 2
  - description: merge empty merging lists
    original:
      mergingList: []
    patch:
      {}
    modified:
      mergingList: []
    current:
      mergingList: []
    result:
      mergingList: []
  - description: add map to merging list by pointer
    original:
      mergeItemPtr:
        - name: 1
    patch:
      mergeItemPtr:
        - name: 2
    modified:
      mergeItemPtr:
        - name: 1
        - name: 2
  - description: add field to map in merging list by pointer
    original:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    patch:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
              value: 1
    modified:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
              value: 1
            - name: 2
              value: 2
        - name: 2
`)

func TestStrategicMergePatch(t *testing.T) {
	tc := StrategicMergePatchTestCases{}
	err := yaml.Unmarshal(createStrategicMergePatchTestCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v", err)
		return
	}

	var e MergeItem
	for _, c := range tc.TestCases {
		cOriginal, cPatch, cModified := testCaseToJSONOrFail(t, c)

		// Test patch generation
		patch, err := CreateStrategicMergePatch(cOriginal, cModified, e)
		if err != nil {
			t.Errorf("error generating patch: %s:\n%v", err, toYAMLOrError(c.StrategicMergePatchTestCaseData))
		}

		// Sort the lists that have merged maps, since order is not significant.
		patch, err = sortMergeListsByName(patch, e)
		if err != nil {
			t.Errorf("error: %s sorting patch object:\n%v", err, patch)
		}

		if !reflect.DeepEqual(patch, cPatch) {
			t.Errorf("patch generation failed:\n%vgot patch:\n%v", toYAMLOrError(c.StrategicMergePatchTestCaseData), jsonToYAMLOrError(patch))
		}

		// Test patch application
		testPatchApplication(t, cOriginal, cPatch, cModified, c.Description)
	}
}

func toYAMLOrError(v interface{}) string {
	y, err := toYAML(v)
	if err != nil {
		return err.Error()
	}

	return y
}

func toJSONOrFail(v interface{}, t *testing.T) []byte {
	theJSON, err := toJSON(v)
	if err != nil {
		t.Error(err)
	}

	return theJSON
}

func jsonToYAMLOrError(j []byte) string {
	y, err := jsonToYAML(j)
	if err != nil {
		return err.Error()
	}

	return string(y)
}

func toYAML(v interface{}) (string, error) {
	y, err := yaml.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("yaml marshal failed: %v\n%v", err, spew.Sdump(v))
	}

	return string(y), nil
}

func toJSON(v interface{}) ([]byte, error) {
	j, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("json marshal failed: %v\n%v", err, spew.Sdump(v))
	}

	return j, nil
}

func jsonToYAML(j []byte) ([]byte, error) {
	y, err := yaml.JSONToYAML(j)
	if err != nil {
		return nil, fmt.Errorf("json to yaml failed: %v\n%v", err, j)
	}

	return y, nil
}
