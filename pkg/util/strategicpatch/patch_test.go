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

type TestCases struct {
	StrategicMergePatchCases []StrategicMergePatchCase
	SortMergeListTestCases   []SortMergeListCase
}

type StrategicMergePatchCase struct {
	Description string
	Patch       map[string]interface{}
	Original    map[string]interface{}
	Result      map[string]interface{}
}

type SortMergeListCase struct {
	Description string
	Original    map[string]interface{}
	Sorted      map[string]interface{}
}

type MergeItem struct {
	Name              string
	Value             string
	MergingList       []MergeItem `patchStrategy:"merge" patchMergeKey:"name"`
	NonMergingList    []MergeItem
	MergingIntList    []int `patchStrategy:"merge"`
	NonMergingIntList []int
	MergeItemPtr      *MergeItem `patchStrategy:"merge" patchMergeKey:"name"`
	SimpleMap         map[string]string
}

var testCaseData = []byte(`
strategicMergePatchCases:
  - description: add new field
    original:
      name: 1
    patch:
      value: 1
    result:
      name: 1
      value: 1
  - description: remove field and add new field
    original:
      name: 1
    patch:
      name: null
      value: 1
    result:
      value: 1
  - description: merge arrays of scalars
    original:
      mergingIntList:
        - 1
        - 2
    patch:
      mergingIntList:
        - 2
        - 3
    result:
      mergingIntList:
        - 1
        - 2
        - 3
  - description: replace arrays of scalars
    original:
      nonMergingIntList:
        - 1
        - 2
    patch:
      nonMergingIntList:
        - 2
        - 3
    result:
      nonMergingIntList:
        - 2
        - 3
  - description: update param of list that should be merged but had element added serverside
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
    result:
      mergingList:
        - name: 1
          value: a
        - name: 2
          value: 2
  - description: delete field when field is nested in a map
    original:
      simpleMap:
        key1: 1
        key2: 1
    patch:
      simpleMap:
        key2: null
    result:
      simpleMap:
        key1: 1
  - description: update nested list when nested list should not be merged
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
    result:
      mergingList:
        - name: 1
          nonMergingList:
            - name: 1
              value: 1
        - name: 2
  - description: update nested list when nested list should be merged
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
    result:
      mergingList:
        - name: 1
          mergingList:
            - name: 1
              value: 1
            - name: 2
              value: 2
        - name: 2
  - description: update map when map should be replaced
    original:
      name: 1
      value: 1
    patch:
      value: 1
      $patch: replace
    result:
      value: 1
  - description: merge empty merge lists
    original:
      mergingList: []
    patch:
      mergingList: []
    result:
      mergingList: []
  - description: delete others in a map
    original:
      name: 1
      value: 1
    patch:
      $patch: replace
    result: {}
  - description: delete item from a merge list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - $patch: delete
          name: 1
    result:
      mergingList:
        - name: 2
  - description: add and delete item from a merge list
    original:
      merginglist:
        - name: 1
        - name: 2
    patch:
      merginglist:
        - name: 3
        - $patch: delete
          name: 1
    result:
      merginglist:
        - name: 2
        - name: 3
  - description: delete all items from a merge list
    original:
      mergingList:
        - name: 1
        - name: 2
    patch:
      mergingList:
        - $patch: replace
    result:
      mergingList: []
  - description: add new field inside pointers
    original:
      mergeItemPtr:
        - name: 1
    patch:
      mergeItemPtr:
        - name: 2
    result:
      mergeItemPtr:
        - name: 1
        - name: 2
  - description: update nested pointers
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
    result:
      mergeItemPtr:
        - name: 1
          mergeItemPtr:
            - name: 1
              value: 1
            - name: 2
              value: 2
        - name: 2
sortMergeListTestCases:
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
  - description: merging list should NOT sort when nested in a non merging list
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
  - description: sort a very nested list of maps
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
  - description: sort one pointer of maps
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

func TestStrategicMergePatch(t *testing.T) {
	tc := TestCases{}
	err := yaml.Unmarshal(testCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v", err)
		return
	}

	var e MergeItem
	for _, c := range tc.StrategicMergePatchCases {
		result, err := StrategicMergePatchData(toJSON(c.Original), toJSON(c.Patch), e)
		if err != nil {
			t.Errorf("error patching: %v:\noriginal:\n%s\npatch:\n%s",
				err, toYAML(c.Original), toYAML(c.Patch))
		}

		// Sort the lists that have merged maps, since order is not significant.
		result, err = sortMergeListsByName(result, e)
		if err != nil {
			t.Errorf("error sorting result object: %v", err)
		}
		cResult, err := sortMergeListsByName(toJSON(c.Result), e)
		if err != nil {
			t.Errorf("error sorting result object: %v", err)
		}

		if !reflect.DeepEqual(result, cResult) {
			t.Errorf("patching failed: %s\noriginal:\n%s\npatch:\n%s\nexpected result:\n%s\ngot result:\n%s",
				c.Description, toYAML(c.Original), toYAML(c.Patch), jsonToYAML(cResult), jsonToYAML(result))
		}
	}
}

func TestSortMergeLists(t *testing.T) {
	tc := TestCases{}
	err := yaml.Unmarshal(testCaseData, &tc)
	if err != nil {
		t.Errorf("can't unmarshal test cases: %v", err)
		return
	}

	var e MergeItem
	for _, c := range tc.SortMergeListTestCases {
		sorted, err := sortMergeListsByName(toJSON(c.Original), e)
		if err != nil {
			t.Errorf("sort arrays returned error: %v", err)
		}

		if !reflect.DeepEqual(sorted, toJSON(c.Sorted)) {
			t.Errorf("sorting failed: %s\ntried to sort:\n%s\nexpected:\n%s\ngot:\n%s",
				c.Description, toYAML(c.Original), toYAML(c.Sorted), jsonToYAML(sorted))
		}
	}
}

func toYAML(v interface{}) string {
	y, err := yaml.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("yaml marshal failed: %v", err))
	}
	return string(y)
}

func toJSON(v interface{}) []byte {
	j, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("json marshal failed: %s", spew.Sdump(v)))
	}
	return j
}

func jsonToYAML(j []byte) []byte {
	y, err := yaml.JSONToYAML(j)
	if err != nil {
		panic(fmt.Sprintf("json to yaml failed: %v", err))
	}
	return y
}
