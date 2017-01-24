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
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/ghodss/yaml"
)

type SortMergeListTestCases struct {
	TestCases []SortMergeListTestCase
}

type SortMergeListTestCase struct {
	Description string
	Original    map[string]interface{}
	Sorted      map[string]interface{}
}

type StrategicMergePatchTestCases struct {
	TestCases []StrategicMergePatchTestCase
}

type StrategicMergePatchTestCase struct {
	Description string
	StrategicMergePatchTestCaseData
}

type StrategicMergePatchRawTestCase struct {
	Description string
	StrategicMergePatchRawTestCaseData
}

type StrategicMergePatchTestCaseData struct {
	// Original is the original object (last-applied config in annotation)
	Original map[string]interface{}
	// Modified is the modified object (new config we want)
	Modified map[string]interface{}
	// Current is the current object (live config in the server)
	Current map[string]interface{}
	// TwoWay is the expected two-way merge patch diff between original and modified
	TwoWay map[string]interface{}
	// ThreeWay is the expected three-way merge patch
	ThreeWay map[string]interface{}
	// Result is the expected object after applying the three-way patch on current object.
	Result map[string]interface{}
}

// The meaning of each field is the same as StrategicMergePatchTestCaseData's.
// The difference is that all the fields in StrategicMergePatchRawTestCaseData are json-encoded data.
type StrategicMergePatchRawTestCaseData struct {
	Original []byte
	Modified []byte
	Current  []byte
	TwoWay   []byte
	ThreeWay []byte
	Result   []byte
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

var mergeItem MergeItem

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
		t.Errorf("can't unmarshal test cases: %s\n", err)
		return
	}

	for _, c := range tc.TestCases {
		original := testObjectToJSONOrFail(t, c.Original, c.Description)
		sorted := testObjectToJSONOrFail(t, c.Sorted, c.Description)
		if !reflect.DeepEqual(original, sorted) {
			t.Errorf("error in test case: %s\ncannot sort object:\n%s\nexpected:\n%s\ngot:\n%s\n",
				c.Description, toYAMLOrError(c.Original), toYAMLOrError(c.Sorted), jsonToYAMLOrError(original))
		}
	}
}

// These are test cases for StrategicMergePatch that cannot be generated using
// CreateTwoWayMergePatch because it doesn't use the replace directive, generate
// duplicate integers for a merging list patch, or generate empty merging lists.
var customStrategicMergePatchTestCaseData = []byte(`
testCases:
  - description: unique scalars when merging lists
    original:
      mergingIntList:
        - 1
        - 2
    twoWay:
      mergingIntList:
        - 2
        - 3
    modified:
      mergingIntList:
        - 1
        - 2
        - 3
  - description: delete map from nested map
    original:
      simpleMap:
        key1: 1
        key2: 1
    twoWay:
      simpleMap:
        $patch: delete
    modified:
      simpleMap:
        {}
  - description: delete all items from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - $patch: replace
    modified:
      mergingList: []
  - description: merge empty merging lists
    original:
      mergingList: []
    twoWay:
      mergingList: []
    modified:
      mergingList: []
  - description: delete all keys from map
    original:
      name: 1
      value: 1
    twoWay:
      $patch: replace
    modified: {}
  - description: add key and delete all keys from map
    original:
      name: 1
      value: 1
    twoWay:
      other: a
      $patch: replace
    modified:
      other: a
  - description: delete all duplicate entries in a merging list
    original:
      mergingList:
        - name: 1
        - name: 1
        - name: 2
          value: a
        - name: 3
        - name: 3
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 3
          $patch: delete
    modified:
      mergingList:
        - name: 2
          value: a
`)

func TestCustomStrategicMergePatch(t *testing.T) {
	tc := StrategicMergePatchTestCases{}
	err := yaml.Unmarshal(customStrategicMergePatchTestCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v\n", err)
		return
	}

	for _, c := range tc.TestCases {
		original, twoWay, modified := twoWayTestCaseToJSONOrFail(t, c)
		testPatchApplication(t, original, twoWay, modified, c.Description)
	}
}

// These are test cases for StrategicMergePatch, to assert that applying  a patch
// yields the correct outcome. They are also test cases for CreateTwoWayMergePatch
// and CreateThreeWayMergePatch, to assert that they both generate the correct patch
// for the given set of input documents.
//
var createStrategicMergePatchTestCaseData = []byte(`
testCases:
  - description: nil original
    twoWay:
      name: 1
      value: 1
    modified:
      name: 1
      value: 1
    current:
      name: 1
      other: a
    threeWay:
      value: 1
    result:
      name: 1
      value: 1
      other: a
  - description: nil patch
    original:
      name: 1
    twoWay:
      {}
    modified:
      name: 1
    current:
      name: 1
    threeWay:
      {}
    result:
      name: 1
  - description: add field to map
    original:
      name: 1
    twoWay:
      value: 1
    modified:
      name: 1
      value: 1
    current:
      name: 1
      other: a
    threeWay:
      value: 1
    result:
      name: 1
      value: 1
      other: a
  - description: add field to map with conflict
    original:
      name: 1
    twoWay:
      value: 1
    modified:
      name: 1
      value: 1
    current:
      name: a
      other: a
    threeWay:
      name: 1
      value: 1
    result:
      name: 1
      value: 1
      other: a
  - description: add field and delete field from map
    original:
      name: 1
    twoWay:
      name: null
      value: 1
    modified:
      value: 1
    current:
      name: 1
      other: a
    threeWay:
      name: null
      value: 1
    result:
      value: 1
      other: a
  - description: add field and delete field from map with conflict
    original:
      name: 1
    twoWay:
      name: null
      value: 1
    modified:
      value: 1
    current:
      name: a
      other: a
    threeWay:
      name: null
      value: 1
    result:
      value: 1
      other: a
  - description: delete field from nested map
    original:
      simpleMap:
        key1: 1
        key2: 1
    twoWay:
      simpleMap:
        key2: null
    modified:
      simpleMap:
        key1: 1
    current:
      simpleMap:
        key1: 1
        key2: 1
        other: a
    threeWay:
      simpleMap:
        key2: null
    result:
      simpleMap:
        key1: 1
        other: a
  - description: delete field from nested map with conflict
    original:
      simpleMap:
        key1: 1
        key2: 1
    twoWay:
      simpleMap:
        key2: null
    modified:
      simpleMap:
        key1: 1
    current:
      simpleMap:
        key1: a
        key2: 1
        other: a
    threeWay:
      simpleMap:
        key1: 1
        key2: null
    result:
      simpleMap:
        key1: 1
        other: a
  - description: delete all fields from map
    original:
      name: 1
      value: 1
    twoWay:
      name: null
      value: null
    modified: {}
    current:
      name: 1
      value: 1
      other: a
    threeWay:
      name: null
      value: null
    result: 
      other: a
  - description: delete all fields from map with conflict
    original:
      name: 1
      value: 1
    twoWay:
      name: null
      value: null
    modified: {}
    current:
      name: 1
      value: a
      other: a
    threeWay:
      name: null
      value: null
    result: 
      other: a
  - description: add field and delete all fields from map
    original:
      name: 1
      value: 1
    twoWay:
      name: null
      value: null
      other: a
    modified:
      other: a
    current:
      name: 1
      value: 1
      other: a
    threeWay:
      name: null
      value: null
    result:
      other: a
  - description: add field and delete all fields from map with conflict
    original:
      name: 1
      value: 1
    twoWay:
      name: null
      value: null
      other: a
    modified:
      other: a
    current:
      name: 1
      value: 1
      other: b
    threeWay:
      name: null
      value: null
      other: a
    result:
      other: a
  - description: replace list of scalars
    original:
      nonMergingIntList:
        - 1
        - 2
    twoWay:
      nonMergingIntList:
        - 2
        - 3
    modified:
      nonMergingIntList:
        - 2
        - 3
    current:
      nonMergingIntList:
        - 1
        - 2
    threeWay:
      nonMergingIntList:
        - 2
        - 3
    result:
      nonMergingIntList:
        - 2
        - 3
  - description: replace list of scalars with conflict
    original:
      nonMergingIntList:
        - 1
        - 2
    twoWay:
      nonMergingIntList:
        - 2
        - 3
    modified:
      nonMergingIntList:
        - 2
        - 3
    current:
      nonMergingIntList:
        - 1
        - 4
    threeWay:
      nonMergingIntList:
        - 2
        - 3
    result:
      nonMergingIntList:
        - 2
        - 3
  - description: merge lists of scalars
    original:
      mergingIntList:
        - 1
        - 2
    twoWay:
      mergingIntList:
        - 3
    modified:
      mergingIntList:
        - 1
        - 2
        - 3
    current:
      mergingIntList:
        - 1
        - 2
        - 4
    threeWay:
      mergingIntList:
        - 3
    result:
      mergingIntList:
        - 1
        - 2
        - 3
        - 4
  - description: merge lists of maps
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 3
          value: 3
        - name: 4 
          value: 4 
    modified:
      mergingList:
        - name: 4 
          value: 4 
        - name: 1
        - name: 2
          value: 2
        - name: 3
          value: 3
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 3
          value: 3
        - name: 4 
          value: 4 
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
        - name: 3
          value: 3
        - name: 4 
          value: 4 
  - description: merge lists of maps with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 3
          other: b
    threeWay:
      mergingList:
        - name: 2
          value: 2
        - name: 3
          value: 3
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
        - name: 3
          value: 3
  - description: add field to map in merging list
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: 1
    result:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: add field to map in merging list with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          other: a
        - name: 3
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    result:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
        - name: 3
          value: 2
          other: b
  - description: add duplicate field to map in merging list
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      {}
    result:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: add duplicate field to map in merging list with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: 1
    modified:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 3
          other: b
    threeWay:
      mergingList:
        - name: 2
          value: 2
    result:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: replace map field value in merging list
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: a
    modified:
      mergingList:
        - name: 1
          value: a
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: a
    result:
      mergingList:
        - name: 1
          value: a
          other: a
        - name: 2
          value: 2
          other: b
  - description: replace map field value in merging list with conflict
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: a
    modified:
      mergingList:
        - name: 1
          value: a
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: 3
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: a
    result:
      mergingList:
        - name: 1
          value: a
          other: a
        - name: 2
          value: 2
          other: b
  - description: delete map from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
    current:
      mergingList:
        - name: 1
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
    result:
      mergingList:
        - name: 2
          other: b
  - description: delete map from merging list with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
    result:
      mergingList:
        - name: 2
          other: b
  - description: delete missing map from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
    current:
      mergingList:
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
    result:
      mergingList:
        - name: 2
          other: b
  - description: delete missing map from merging list with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
    modified:
      mergingList:
        - name: 2
    current:
      mergingList:
        - name: 3
          other: a
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
    result:
      mergingList:
        - name: 2
        - name: 3
          other: a
  - description: add map and delete map from merging list
    original:
      merginglist:
        - name: 1
        - name: 2
    twoWay:
      merginglist:
        - name: 1
          $patch: delete
        - name: 3
    modified:
      merginglist:
        - name: 2
        - name: 3
    current:
      merginglist:
        - name: 1
        - name: 2
          other: b
        - name: 4
          other: c
    threeWay:
      merginglist:
        - name: 1
          $patch: delete
        - name: 3
    result:
      merginglist:
        - name: 2
          other: b
        - name: 3
        - name: 4
          other: c
  - description: add map and delete map from merging list with conflict
    original:
      merginglist:
        - name: 1
        - name: 2
    twoWay:
      merginglist:
        - name: 1
          $patch: delete
        - name: 3
    modified:
      merginglist:
        - name: 2
        - name: 3
    current:
      merginglist:
        - name: 1
          other: a
        - name: 4
          other: c
    threeWay:
      merginglist:
        - name: 1
          $patch: delete
        - name: 2
        - name: 3
    result:
      merginglist:
        - name: 2
        - name: 3
        - name: 4
          other: c
  - description: delete all maps from merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
    current:
      mergingList:
        - name: 1
        - name: 2
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    result:
      mergingList: []
  - description: delete all maps from merging list with conflict
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    result:
      mergingList: []
  - description: delete all maps from empty merging list
    original:
      mergingList:
        - name: 1
        - name: 2
    twoWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    modified:
      mergingList: []
    current:
      mergingList: []
    threeWay:
      mergingList:
        - name: 1
          $patch: delete
        - name: 2
          $patch: delete
    result:
      mergingList: []
  - description: delete field from map in merging list
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: null
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: null
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: delete field from map in merging list with conflict
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: null
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          value: a
          other: a
        - name: 2
          value: 2
    threeWay:
      mergingList:
        - name: 1
          value: null
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
  - description: delete missing field from map in merging list
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: null
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: null
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: delete missing field from map in merging list with conflict
    original:
      mergingList:
        - name: 1
          value: 1
        - name: 2
          value: 2
    twoWay:
      mergingList:
        - name: 1
          value: null
    modified:
      mergingList:
        - name: 1
        - name: 2
          value: 2
    current:
      mergingList:
        - name: 1
          other: a
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          value: null
        - name: 2
          value: 2
    result:
      mergingList:
        - name: 1
          other: a
        - name: 2
          value: 2
          other: b
  - description: replace non merging list nested in merging list
    original:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
    result:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 1
              value: 1
        - name: 2
          other: b
  - description: replace non merging list nested in merging list with value conflict
    original:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 1
              value: c
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
    result:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 1
              value: 1
        - name: 2
          other: b
  - description: replace non merging list nested in merging list with deletion conflict
    original:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 2
              value: 2
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
    result:
      mergingList:
        - name: 1
          other: a
          nonMergingList:
            - name: 1
              value: 1
        - name: 2
          other: b
  - description: add field to map in merging list nested in merging list
    original:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
    result:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 1
              value: 1
            - name: 2
              value: 2
        - name: 2
          other: b
  - description: add field to map in merging list nested in merging list with value conflict
    original:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 1
              value: a
              other: c
            - name: 2
              value: b
              other: d
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
            - name: 2
              value: 2
    result:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 1
              value: 1
              other: c
            - name: 2
              value: 2
              other: d
        - name: 2
          other: b
  - description: add field to map in merging list nested in merging list with deletion conflict
    original:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 2
              value: 2
              other: d
        - name: 2
          other: b
    threeWay:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
    result:
      mergingList:
        - name: 1
          other: a
          mergingList:
            - name: 1
              value: 1
            - name: 2
              value: 2
              other: d
        - name: 2
          other: b
  - description: merge empty merging lists
    original:
      mergingList: []
    twoWay:
      {}
    modified:
      mergingList: []
    current:
      mergingList: []
    threeWay:
      {}
    result:
      mergingList: []
  - description: add map to merging list by pointer
    original:
      mergeItemPtr:
        - name: 1
    twoWay:
      mergeItemPtr:
        - name: 2
    modified:
      mergeItemPtr:
        - name: 1
        - name: 2
    current:
      mergeItemPtr:
        - name: 1
          other: a
        - name: 3
    threeWay:
      mergeItemPtr:
        - name: 2
    result:
      mergeItemPtr:
        - name: 1
          other: a
        - name: 2
        - name: 3
  - description: add map to merging list by pointer with conflict
    original:
      mergeItemPtr:
        - name: 1
    twoWay:
      mergeItemPtr:
        - name: 2
    modified:
      mergeItemPtr:
        - name: 1
        - name: 2
    current:
      mergeItemPtr:
        - name: 3
    threeWay:
      mergeItemPtr:
        - name: 1
        - name: 2
    result:
      mergeItemPtr:
        - name: 1
        - name: 2
        - name: 3
  - description: add field to map in merging list by pointer
    original:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergeItemPtr:
        - name: 1
          other: a
          mergeItemPtr:
            - name: 1
              other: a
            - name: 2
              value: 2
              other: b
        - name: 2
          other: b
    threeWay:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
              value: 1
    result:
      mergeItemPtr:
        - name: 1
          other: a
          mergeItemPtr:
            - name: 1
              value: 1
              other: a
            - name: 2
              value: 2
              other: b
        - name: 2
          other: b
  - description: add field to map in merging list by pointer with conflict
    original:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
            - name: 2
              value: 2
        - name: 2
    twoWay:
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
    current:
      mergeItemPtr:
        - name: 1
          other: a
          mergeItemPtr:
            - name: 1
              value: a
            - name: 2
              value: 2
              other: b
        - name: 2
          other: b
    threeWay:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
              value: 1
    result:
      mergeItemPtr:
        - name: 1
          other: a
          mergeItemPtr:
            - name: 1
              value: 1
            - name: 2
              value: 2
              other: b
        - name: 2
          other: b
`)

var strategicMergePatchRawTestCases = []StrategicMergePatchRawTestCase{
	{
		Description: "delete items in lists of scalars",
		StrategicMergePatchRawTestCaseData: StrategicMergePatchRawTestCaseData{
			Original: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
`),
			TwoWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
`),
			Modified: []byte(`
mergingIntList:
  - 1
  - 2
`),
			Current: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
  - 4
`),
			ThreeWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
`),
			Result: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
		},
	},
	{
		Description: "delete all duplicate items in lists of scalars",
		StrategicMergePatchRawTestCaseData: StrategicMergePatchRawTestCaseData{
			Original: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
  - 3
`),
			TwoWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
`),
			Modified: []byte(`
mergingIntList:
  - 1
  - 2
`),
			Current: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
  - 3
  - 4
`),
			ThreeWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
`),
			Result: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
		},
	},
	{
		Description: "add and delete items in lists of scalars",
		StrategicMergePatchRawTestCaseData: StrategicMergePatchRawTestCaseData{
			Original: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
`),
			TwoWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
mergingIntList:
  - 4
`),
			Modified: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
			Current: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
  - 4
`),
			ThreeWay: []byte(`
$deleteFromPrimitiveList/mergingIntList:
  - 3
`),
			Result: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
		},
	},
}

func TestStrategicMergePatch(t *testing.T) {
	testStrategicMergePatchWithCustomArguments(t, "bad original",
		"<THIS IS NOT JSON>", "{}", mergeItem, errBadJSONDoc)
	testStrategicMergePatchWithCustomArguments(t, "bad patch",
		"{}", "<THIS IS NOT JSON>", mergeItem, errBadJSONDoc)
	testStrategicMergePatchWithCustomArguments(t, "bad struct",
		"{}", "{}", []byte("<THIS IS NOT A STRUCT>"), fmt.Errorf(errBadArgTypeFmt, "struct", "slice"))
	testStrategicMergePatchWithCustomArguments(t, "nil struct",
		"{}", "{}", nil, fmt.Errorf(errBadArgTypeFmt, "struct", "nil"))

	tc := StrategicMergePatchTestCases{}
	err := yaml.Unmarshal(createStrategicMergePatchTestCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %s\n", err)
		return
	}

	for _, c := range tc.TestCases {
		testTwoWayPatch(t, c)
		testThreeWayPatch(t, c)
	}

	for _, c := range strategicMergePatchRawTestCases {
		testTwoWayPatchForRawTestCase(t, c)
		testThreeWayPatchForRawTestCase(t, c)
	}
}

func testStrategicMergePatchWithCustomArguments(t *testing.T, description, original, patch string, dataStruct interface{}, err error) {
	_, err2 := StrategicMergePatch([]byte(original), []byte(patch), dataStruct)
	if err2 != err {
		if err2 == nil {
			t.Errorf("expected error: %s\ndid not occur in test case: %s", err, description)
			return
		}

		if err == nil || err2.Error() != err.Error() {
			t.Errorf("unexpected error: %s\noccurred in test case: %s", err2, description)
			return
		}
	}
}

func testTwoWayPatch(t *testing.T, c StrategicMergePatchTestCase) {
	original, expected, modified := twoWayTestCaseToJSONOrFail(t, c)

	actual, err := CreateTwoWayMergePatch(original, modified, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch: %s:\n%s\n",
			err, c.Description, original, toYAMLOrError(c.StrategicMergePatchTestCaseData))
		return
	}

	testPatchCreation(t, expected, actual, c.Description)
	testPatchApplication(t, original, actual, modified, c.Description)
}

func testTwoWayPatchForRawTestCase(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, expected, modified := twoWayRawTestCaseToJSONOrFail(t, c)

	actual, err := CreateTwoWayMergePatch(original, modified, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	testPatchCreation(t, expected, actual, c.Description)
	testPatchApplication(t, original, actual, modified, c.Description)
}

func twoWayTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchTestCase) ([]byte, []byte, []byte) {
	return testObjectToJSONOrFail(t, c.Original, c.Description),
		testObjectToJSONOrFail(t, c.TwoWay, c.Description),
		testObjectToJSONOrFail(t, c.Modified, c.Description)
}

func twoWayRawTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchRawTestCase) ([]byte, []byte, []byte) {
	return yamlToJSONOrError(t, c.Original),
		yamlToJSONOrError(t, c.TwoWay),
		yamlToJSONOrError(t, c.Modified)
}

func testThreeWayPatch(t *testing.T, c StrategicMergePatchTestCase) {
	original, modified, current, expected, result := threeWayTestCaseToJSONOrFail(t, c)
	actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, false)
	if err != nil {
		if !IsConflict(err) {
			t.Errorf("error: %s\nin test case: %s\ncannot create three way patch:\n%s\n",
				err, c.Description, toYAMLOrError(c.StrategicMergePatchTestCaseData))
			return
		}

		if !strings.Contains(c.Description, "conflict") {
			t.Errorf("unexpected conflict: %s\nin test case: %s\ncannot create three way patch:\n%s\n",
				err, c.Description, toYAMLOrError(c.StrategicMergePatchTestCaseData))
			return
		}

		if len(c.Result) > 0 {
			actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, true)
			if err != nil {
				t.Errorf("error: %s\nin test case: %s\ncannot force three way patch application:\n%s\n",
					err, c.Description, toYAMLOrError(c.StrategicMergePatchTestCaseData))
				return
			}

			testPatchCreation(t, expected, actual, c.Description)
			testPatchApplication(t, current, actual, result, c.Description)
		}

		return
	}

	if strings.Contains(c.Description, "conflict") || len(c.Result) < 1 {
		t.Errorf("error in test case: %s\nexpected conflict did not occur:\n%s\n",
			c.Description, toYAMLOrError(c.StrategicMergePatchTestCaseData))
		return
	}

	testPatchCreation(t, expected, actual, c.Description)
	testPatchApplication(t, current, actual, result, c.Description)
}

func testThreeWayPatchForRawTestCase(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, modified, current, expected, result := threeWayRawTestCaseToJSONOrFail(t, c)
	actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, false)
	if err != nil {
		if !IsConflict(err) {
			t.Errorf("error: %s\nin test case: %s\ncannot create three way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
				err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
			return
		}

		if !strings.Contains(c.Description, "conflict") {
			t.Errorf("unexpected conflict: %s\nin test case: %s\ncannot create three way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
				err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
			return
		}

		if len(c.Result) > 0 {
			actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, true)
			if err != nil {
				t.Errorf("error: %s\nin test case: %s\ncannot force three way patch application:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
					err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
				return
			}

			testPatchCreation(t, expected, actual, c.Description)
			testPatchApplication(t, current, actual, result, c.Description)
		}

		return
	}

	if strings.Contains(c.Description, "conflict") || len(c.Result) < 1 {
		t.Errorf("error: %s\nin test case: %s\nexpected conflict did not occur:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	testPatchCreation(t, expected, actual, c.Description)
	testPatchApplication(t, current, actual, result, c.Description)
}

func threeWayTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchTestCase) ([]byte, []byte, []byte, []byte, []byte) {
	return testObjectToJSONOrFail(t, c.Original, c.Description),
		testObjectToJSONOrFail(t, c.Modified, c.Description),
		testObjectToJSONOrFail(t, c.Current, c.Description),
		testObjectToJSONOrFail(t, c.ThreeWay, c.Description),
		testObjectToJSONOrFail(t, c.Result, c.Description)
}

func threeWayRawTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchRawTestCase) ([]byte, []byte, []byte, []byte, []byte) {
	return yamlToJSONOrError(t, c.Original),
		yamlToJSONOrError(t, c.Modified),
		yamlToJSONOrError(t, c.Current),
		yamlToJSONOrError(t, c.ThreeWay),
		yamlToJSONOrError(t, c.Result)
}

func testPatchCreation(t *testing.T, expected, actual []byte, description string) {
	sorted, err := sortMergeListsByName(actual, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot sort patch:\n%s\n",
			err, description, jsonToYAMLOrError(actual))
		return
	}

	if !reflect.DeepEqual(sorted, expected) {
		t.Errorf("error in test case: %s\nexpected patch:\n%s\ngot:\n%s\n",
			description, jsonToYAMLOrError(expected), jsonToYAMLOrError(sorted))
		return
	}
}

func testPatchApplication(t *testing.T, original, patch, expected []byte, description string) {
	result, err := StrategicMergePatch(original, patch, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot apply patch:\n%s\nto original:\n%s\n",
			err, description, jsonToYAMLOrError(patch), jsonToYAMLOrError(original))
		return
	}

	sorted, err := sortMergeListsByName(result, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot sort result object:\n%s\n",
			err, description, jsonToYAMLOrError(result))
		return
	}

	if !reflect.DeepEqual(sorted, expected) {
		format := "error in test case: %s\npatch application failed:\noriginal:\n%s\npatch:\n%s\nexpected:\n%s\ngot:\n%s\n"
		t.Errorf(format, description,
			jsonToYAMLOrError(original), jsonToYAMLOrError(patch),
			jsonToYAMLOrError(expected), jsonToYAMLOrError(sorted))
		return
	}
}

func testObjectToJSONOrFail(t *testing.T, o map[string]interface{}, description string) []byte {
	if o == nil {
		return nil
	}

	j, err := toJSON(o)
	if err != nil {
		t.Error(err)
	}

	r, err := sortMergeListsByName(j, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot sort object:\n%s\n", err, description, j)
		return nil
	}

	return r
}

func jsonToYAMLOrError(j []byte) string {
	y, err := jsonToYAML(j)
	if err != nil {
		return err.Error()
	}

	return string(y)
}

func toJSON(v interface{}) ([]byte, error) {
	j, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("json marshal failed: %v\n%v\n", err, spew.Sdump(v))
	}

	return j, nil
}

func jsonToYAML(j []byte) ([]byte, error) {
	y, err := yaml.JSONToYAML(j)
	if err != nil {
		return nil, fmt.Errorf("json to yaml failed: %v\n%v\n", err, j)
	}

	return y, nil
}

func yamlToJSON(y []byte) ([]byte, error) {
	j, err := yaml.YAMLToJSON(y)
	if err != nil {
		return nil, fmt.Errorf("yaml to json failed: %v\n%v\n", err, y)
	}

	return j, nil
}

func yamlToJSONOrError(t *testing.T, y []byte) []byte {
	j, err := yamlToJSON(y)
	if err != nil {
		t.Errorf("%v", err)
	}

	return j
}

func TestHasConflicts(t *testing.T) {
	testCases := []struct {
		A   interface{}
		B   interface{}
		Ret bool
	}{
		{A: "hello", B: "hello", Ret: false}, // 0
		{A: "hello", B: "hell", Ret: true},
		{A: "hello", B: nil, Ret: true},
		{A: "hello", B: 1, Ret: true},
		{A: "hello", B: float64(1.0), Ret: true},
		{A: "hello", B: false, Ret: true},
		{A: 1, B: 1, Ret: false},
		{A: false, B: false, Ret: false},
		{A: float64(3), B: float64(3), Ret: false},

		{A: "hello", B: []interface{}{}, Ret: true}, // 6
		{A: []interface{}{1}, B: []interface{}{}, Ret: true},
		{A: []interface{}{}, B: []interface{}{}, Ret: false},
		{A: []interface{}{1}, B: []interface{}{1}, Ret: false},
		{A: map[string]interface{}{}, B: []interface{}{1}, Ret: true},

		{A: map[string]interface{}{}, B: map[string]interface{}{"a": 1}, Ret: false}, // 11
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"a": 1}, Ret: false},
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"a": 2}, Ret: true},
		{A: map[string]interface{}{"a": 1}, B: map[string]interface{}{"b": 2}, Ret: false},

		{ // 15
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": []interface{}{1}},
			Ret: false,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": []interface{}{}},
			Ret: true,
		},
		{
			A:   map[string]interface{}{"a": []interface{}{1}},
			B:   map[string]interface{}{"a": 1},
			Ret: true,
		},
	}

	for i, testCase := range testCases {
		out, err := HasConflicts(testCase.A, testCase.B)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if out != testCase.Ret {
			t.Errorf("%d: expected %t got %t", i, testCase.Ret, out)
			continue
		}
		out, err = HasConflicts(testCase.B, testCase.A)
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}
		if out != testCase.Ret {
			t.Errorf("%d: expected reversed %t got %t", i, testCase.Ret, out)
		}
	}
}

type PrecisionItem struct {
	Name    string
	Int32   int32
	Int64   int64
	Float32 float32
	Float64 float64
}

var precisionItem PrecisionItem

func TestNumberConversion(t *testing.T) {
	testcases := map[string]struct {
		Old            string
		New            string
		ExpectedPatch  string
		ExpectedResult string
	}{
		"empty": {
			Old:            `{}`,
			New:            `{}`,
			ExpectedPatch:  `{}`,
			ExpectedResult: `{}`,
		},
		"int32 medium": {
			Old:            `{"int32":1000000}`,
			New:            `{"int32":1000000,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"int32":1000000,"name":"newname"}`,
		},
		"int32 max": {
			Old:            `{"int32":2147483647}`,
			New:            `{"int32":2147483647,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"int32":2147483647,"name":"newname"}`,
		},
		"int64 medium": {
			Old:            `{"int64":1000000}`,
			New:            `{"int64":1000000,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"int64":1000000,"name":"newname"}`,
		},
		"int64 max": {
			Old:            `{"int64":9223372036854775807}`,
			New:            `{"int64":9223372036854775807,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"int64":9223372036854775807,"name":"newname"}`,
		},
		"float32 max": {
			Old:            `{"float32":3.4028234663852886e+38}`,
			New:            `{"float32":3.4028234663852886e+38,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"float32":3.4028234663852886e+38,"name":"newname"}`,
		},
		"float64 max": {
			Old:            `{"float64":1.7976931348623157e+308}`,
			New:            `{"float64":1.7976931348623157e+308,"name":"newname"}`,
			ExpectedPatch:  `{"name":"newname"}`,
			ExpectedResult: `{"float64":1.7976931348623157e+308,"name":"newname"}`,
		},
	}

	for k, tc := range testcases {
		patch, err := CreateTwoWayMergePatch([]byte(tc.Old), []byte(tc.New), precisionItem)
		if err != nil {
			t.Errorf("%s: unexpected error %v", k, err)
			continue
		}
		if tc.ExpectedPatch != string(patch) {
			t.Errorf("%s: expected %s, got %s", k, tc.ExpectedPatch, string(patch))
			continue
		}

		result, err := StrategicMergePatch([]byte(tc.Old), patch, precisionItem)
		if err != nil {
			t.Errorf("%s: unexpected error %v", k, err)
			continue
		}
		if tc.ExpectedResult != string(result) {
			t.Errorf("%s: expected %s, got %s", k, tc.ExpectedResult, string(result))
			continue
		}
	}
}
