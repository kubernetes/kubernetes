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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/sets"
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
	// TwoWayResult is the expected object after applying the two-way patch on current object.
	// If nil, Modified is used.
	TwoWayResult map[string]interface{}
}

// The meaning of each field is the same as StrategicMergePatchTestCaseData's.
// The difference is that all the fields in StrategicMergePatchRawTestCaseData are json-encoded data.
type StrategicMergePatchRawTestCaseData struct {
	Original     []byte
	Modified     []byte
	Current      []byte
	TwoWay       []byte
	ThreeWay     []byte
	Result       []byte
	TwoWayResult []byte
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
	ReplacingItem     runtime.RawExtension `patchStrategy:"replace"`
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
				c.Description, mergepatch.ToYAMLOrError(c.Original), mergepatch.ToYAMLOrError(c.Sorted), jsonToYAMLOrError(original))
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
		original, expectedTwoWayPatch, _, expectedResult := twoWayTestCaseToJSONOrFail(t, c)
		testPatchApplication(t, original, expectedTwoWayPatch, expectedResult, c.Description)
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
  - description: defined null values should propagate overwrite current fields (with conflict)
    original:
      name: 2
    twoWay:
      name: 1
      value: 1
      other: null
    twoWayResult:
      name: 1
      value: 1
    modified:
      name: 1
      value: 1
      other: null
    current:
      name: a
      other: a
    threeWay:
      name: 1
      value: 1
      other: null
    result:
      name: 1
      value: 1
  - description: defined null values should propagate removing original fields
    original:
      name: original-name
      value: original-value
    current:
      name: original-name
      value: original-value
      other: current-other
    modified:
      name: modified-name
      value: null
    twoWay:
      name: modified-name
      value: null
    twoWayResult:
      name: modified-name
    threeWay:
      name: modified-name
      value: null
    result:
      name: modified-name
      other: current-other
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
		"<THIS IS NOT JSON>", "{}", mergeItem, mergepatch.ErrBadJSONDoc)
	testStrategicMergePatchWithCustomArguments(t, "bad patch",
		"{}", "<THIS IS NOT JSON>", mergeItem, mergepatch.ErrBadJSONDoc)
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
	original, expectedPatch, modified, expectedResult := twoWayTestCaseToJSONOrFail(t, c)

	actualPatch, err := CreateTwoWayMergePatch(original, modified, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch: %s:\n%s\n",
			err, c.Description, original, mergepatch.ToYAMLOrError(c.StrategicMergePatchTestCaseData))
		return
	}

	testPatchCreation(t, expectedPatch, actualPatch, c.Description)
	testPatchApplication(t, original, actualPatch, expectedResult, c.Description)
}

func testTwoWayPatchForRawTestCase(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, expectedPatch, modified, expectedResult := twoWayRawTestCaseToJSONOrFail(t, c)

	actualPatch, err := CreateTwoWayMergePatch(original, modified, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	testPatchCreation(t, expectedPatch, actualPatch, c.Description)
	testPatchApplication(t, original, actualPatch, expectedResult, c.Description)
}

func twoWayTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchTestCase) ([]byte, []byte, []byte, []byte) {
	expectedResult := c.TwoWayResult
	if expectedResult == nil {
		expectedResult = c.Modified
	}
	return testObjectToJSONOrFail(t, c.Original, c.Description),
		testObjectToJSONOrFail(t, c.TwoWay, c.Description),
		testObjectToJSONOrFail(t, c.Modified, c.Description),
		testObjectToJSONOrFail(t, expectedResult, c.Description)
}

func twoWayRawTestCaseToJSONOrFail(t *testing.T, c StrategicMergePatchRawTestCase) ([]byte, []byte, []byte, []byte) {
	expectedResult := c.TwoWayResult
	if expectedResult == nil {
		expectedResult = c.Modified
	}
	return yamlToJSONOrError(t, c.Original),
		yamlToJSONOrError(t, c.TwoWay),
		yamlToJSONOrError(t, c.Modified),
		yamlToJSONOrError(t, expectedResult)
}

func testThreeWayPatch(t *testing.T, c StrategicMergePatchTestCase) {
	original, modified, current, expected, result := threeWayTestCaseToJSONOrFail(t, c)
	actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, false)
	if err != nil {
		if !mergepatch.IsConflict(err) {
			t.Errorf("error: %s\nin test case: %s\ncannot create three way patch:\n%s\n",
				err, c.Description, mergepatch.ToYAMLOrError(c.StrategicMergePatchTestCaseData))
			return
		}

		if !strings.Contains(c.Description, "conflict") {
			t.Errorf("unexpected conflict: %s\nin test case: %s\ncannot create three way patch:\n%s\n",
				err, c.Description, mergepatch.ToYAMLOrError(c.StrategicMergePatchTestCaseData))
			return
		}

		if len(c.Result) > 0 {
			actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, true)
			if err != nil {
				t.Errorf("error: %s\nin test case: %s\ncannot force three way patch application:\n%s\n",
					err, c.Description, mergepatch.ToYAMLOrError(c.StrategicMergePatchTestCaseData))
				return
			}

			testPatchCreation(t, expected, actual, c.Description)
			testPatchApplication(t, current, actual, result, c.Description)
		}

		return
	}

	if strings.Contains(c.Description, "conflict") || len(c.Result) < 1 {
		t.Errorf("error in test case: %s\nexpected conflict did not occur:\n%s\n",
			c.Description, mergepatch.ToYAMLOrError(c.StrategicMergePatchTestCaseData))
		return
	}

	testPatchCreation(t, expected, actual, c.Description)
	testPatchApplication(t, current, actual, result, c.Description)
}

func testThreeWayPatchForRawTestCase(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, modified, current, expected, result := threeWayRawTestCaseToJSONOrFail(t, c)
	actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, false)
	if err != nil {
		if !mergepatch.IsConflict(err) {
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

var replaceRawExtensionPatchTestCases = []StrategicMergePatchRawTestCase{
	{
		Description: "replace RawExtension field, rest unchanched",
		StrategicMergePatchRawTestCaseData: StrategicMergePatchRawTestCaseData{
			Original: []byte(`
name: my-object
value: some-value
other: current-other
replacingItem:
  Some: Generic
  Yaml: Inside
  The: RawExtension
  Field: Period
`),
			Current: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
  - name: 3
replacingItem:
  Some: Generic
  Yaml: Inside
  The: RawExtension
  Field: Period
`),
			Modified: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
  - name: 3
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			TwoWay: []byte(`
merginglist:
  - name: 1
  - name: 2
  - name: 3
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			TwoWayResult: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
  - name: 3
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			ThreeWay: []byte(`
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			Result: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
  - name: 3
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
		},
	},
	{
		Description: "replace RawExtension field and merge list",
		StrategicMergePatchRawTestCaseData: StrategicMergePatchRawTestCaseData{
			Original: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
replacingItem:
  Some: Generic
  Yaml: Inside
  The: RawExtension
  Field: Period
`),
			Current: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 3
replacingItem:
  Some: Generic
  Yaml: Inside
  The: RawExtension
  Field: Period
`),
			Modified: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			TwoWay: []byte(`
merginglist:
  - name: 2
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			TwoWayResult: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 2
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			ThreeWay: []byte(`
merginglist:
  - name: 2
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
			Result: []byte(`
name: my-object
value: some-value
other: current-other
merginglist:
  - name: 1
  - name: 3
  - name: 2
replacingItem:
  Newly: Modified
  Yaml: Inside
  The: RawExtension
`),
		},
	},
}

func TestReplaceWithRawExtension(t *testing.T) {
	for _, c := range replaceRawExtensionPatchTestCases {
		testTwoWayPatchWithoutSorting(t, c)
		testThreeWayPatchWithoutSorting(t, c)
	}
}

func testTwoWayPatchWithoutSorting(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, expectedPatch, modified, expectedResult := twoWayRawTestCaseToJSONOrFail(t, c)

	actualPatch, err := CreateTwoWayMergePatch(original, modified, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	testPatchCreationWithoutSorting(t, expectedPatch, actualPatch, c.Description)
	testPatchApplicationWithoutSorting(t, original, actualPatch, expectedResult, c.Description)
}

func testThreeWayPatchWithoutSorting(t *testing.T, c StrategicMergePatchRawTestCase) {
	original, modified, current, expected, result := threeWayRawTestCaseToJSONOrFail(t, c)
	actual, err := CreateThreeWayMergePatch(original, modified, current, mergeItem, false)
	if err != nil {
		if !mergepatch.IsConflict(err) {
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

			testPatchCreationWithoutSorting(t, expected, actual, c.Description)
			testPatchApplicationWithoutSorting(t, current, actual, result, c.Description)
		}

		return
	}

	if strings.Contains(c.Description, "conflict") || len(c.Result) < 1 {
		t.Errorf("error: %s\nin test case: %s\nexpected conflict did not occur:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	testPatchCreationWithoutSorting(t, expected, actual, c.Description)
	testPatchApplicationWithoutSorting(t, current, actual, result, c.Description)
}

func testPatchCreationWithoutSorting(t *testing.T, expected, actual []byte, description string) {
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("error in test case: %s\nexpected patch:\n%s\ngot:\n%s\n",
			description, jsonToYAMLOrError(expected), jsonToYAMLOrError(actual))
		return
	}
}

func testPatchApplicationWithoutSorting(t *testing.T, original, patch, expected []byte, description string) {
	result, err := StrategicMergePatch(original, patch, mergeItem)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot apply patch:\n%s\nto original:\n%s\n",
			err, description, jsonToYAMLOrError(patch), jsonToYAMLOrError(original))
		return
	}

	if !reflect.DeepEqual(result, expected) {
		format := "error in test case: %s\npatch application failed:\noriginal:\n%s\npatch:\n%s\nexpected:\n%s\ngot:\n%s\n"
		t.Errorf(format, description,
			jsonToYAMLOrError(original), jsonToYAMLOrError(patch),
			jsonToYAMLOrError(expected), jsonToYAMLOrError(result))
		return
	}
}

func TestUnknownField(t *testing.T) {
	testcases := map[string]struct {
		Original string
		Current  string
		Modified string

		ExpectedTwoWay         string
		ExpectedTwoWayErr      string
		ExpectedTwoWayResult   string
		ExpectedThreeWay       string
		ExpectedThreeWayErr    string
		ExpectedThreeWayResult string
	}{
		// cases we can successfully strategically merge
		"no diff": {
			Original: `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			Current:  `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			Modified: `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,

			ExpectedTwoWay:         `{}`,
			ExpectedTwoWayResult:   `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			ExpectedThreeWay:       `{}`,
			ExpectedThreeWayResult: `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
		},
		"added only": {
			Original: `{"name":"foo"}`,
			Current:  `{"name":"foo"}`,
			Modified: `{"name":"foo","scalar":true,"complex":{"nested":true},"array":[1,2,3]}`,

			ExpectedTwoWay:         `{"array":[1,2,3],"complex":{"nested":true},"scalar":true}`,
			ExpectedTwoWayResult:   `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			ExpectedThreeWay:       `{"array":[1,2,3],"complex":{"nested":true},"scalar":true}`,
			ExpectedThreeWayResult: `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
		},
		"removed only": {
			Original: `{"name":"foo","scalar":true,"complex":{"nested":true}}`,
			Current:  `{"name":"foo","scalar":true,"complex":{"nested":true},"array":[1,2,3]}`,
			Modified: `{"name":"foo"}`,

			ExpectedTwoWay:         `{"complex":null,"scalar":null}`,
			ExpectedTwoWayResult:   `{"name":"foo"}`,
			ExpectedThreeWay:       `{"complex":null,"scalar":null}`,
			ExpectedThreeWayResult: `{"array":[1,2,3],"name":"foo"}`,
		},

		// cases we cannot successfully strategically merge (expect errors)
		"diff": {
			Original: `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			Current:  `{"array":[1,2,3],"complex":{"nested":true},"name":"foo","scalar":true}`,
			Modified: `{"array":[1,2,3],"complex":{"nested":false},"name":"foo","scalar":true}`,

			ExpectedTwoWayErr:   `unable to find api field`,
			ExpectedThreeWayErr: `unable to find api field`,
		},
	}

	for _, k := range sets.StringKeySet(testcases).List() {
		tc := testcases[k]
		func() {
			twoWay, err := CreateTwoWayMergePatch([]byte(tc.Original), []byte(tc.Modified), &MergeItem{})
			if err != nil {
				if len(tc.ExpectedTwoWayErr) == 0 {
					t.Errorf("%s: error making two-way patch: %v", k, err)
				}
				if !strings.Contains(err.Error(), tc.ExpectedTwoWayErr) {
					t.Errorf("%s: expected error making two-way patch to contain '%s', got %s", k, tc.ExpectedTwoWayErr, err)
				}
				return
			}

			if string(twoWay) != tc.ExpectedTwoWay {
				t.Errorf("%s: expected two-way patch:\n\t%s\ngot\n\t%s", k, string(tc.ExpectedTwoWay), string(twoWay))
				return
			}

			twoWayResult, err := StrategicMergePatch([]byte(tc.Original), twoWay, MergeItem{})
			if err != nil {
				t.Errorf("%s: error applying two-way patch: %v", k, err)
				return
			}
			if string(twoWayResult) != tc.ExpectedTwoWayResult {
				t.Errorf("%s: expected two-way result:\n\t%s\ngot\n\t%s", k, string(tc.ExpectedTwoWayResult), string(twoWayResult))
				return
			}
		}()

		func() {
			threeWay, err := CreateThreeWayMergePatch([]byte(tc.Original), []byte(tc.Modified), []byte(tc.Current), &MergeItem{}, false)
			if err != nil {
				if len(tc.ExpectedThreeWayErr) == 0 {
					t.Errorf("%s: error making three-way patch: %v", k, err)
				} else if !strings.Contains(err.Error(), tc.ExpectedThreeWayErr) {
					t.Errorf("%s: expected error making three-way patch to contain '%s', got %s", k, tc.ExpectedThreeWayErr, err)
				}
				return
			}

			if string(threeWay) != tc.ExpectedThreeWay {
				t.Errorf("%s: expected three-way patch:\n\t%s\ngot\n\t%s", k, string(tc.ExpectedThreeWay), string(threeWay))
				return
			}

			threeWayResult, err := StrategicMergePatch([]byte(tc.Current), threeWay, MergeItem{})
			if err != nil {
				t.Errorf("%s: error applying three-way patch: %v", k, err)
				return
			} else if string(threeWayResult) != tc.ExpectedThreeWayResult {
				t.Errorf("%s: expected three-way result:\n\t%s\ngot\n\t%s", k, string(tc.ExpectedThreeWayResult), string(threeWayResult))
				return
			}
		}()
	}
}
