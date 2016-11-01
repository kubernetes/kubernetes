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
	"bytes"
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

type StrategicMergePatchTestCaseData struct {
	Original []byte
	TwoWay   []byte
	Modified []byte
	Current  []byte
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
var customStrategicMergePatchTestCaseData = StrategicMergePatchTestCases{
	TestCases: []StrategicMergePatchTestCase{
		{
			Description: "unique scalars when merging lists using SMPatchVersion_1_0",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
mergingIntList:
  - 2
  - 3
`),
				Modified: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
`),
			},
		},
		{
			Description: "unique scalars when merging lists for list of primitives",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
mergingIntList:
  $patch: mergeprimitiveslist
  "2": 2
  "3": 3
`),
				Modified: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
`),
			},
		},
		{
			Description: "delete map from nested map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
simpleMap:
  key1: 1
  key2: 1
`),
				TwoWay: []byte(`
simpleMap:
  $patch: delete
`),
				Modified: []byte(`
simpleMap:
  {}
`),
			},
		},
		{
			Description: "delete all items from merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - $patch: replace
`),
				Modified: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "merge empty merging lists",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList: []
`),
				TwoWay: []byte(`
mergingList: []
`),
				Modified: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "delete all keys from map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
$patch: replace
`),
				Modified: []byte(`
{}
`),
			},
		},
		{
			Description: "add key and delete all keys from map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
other: a
$patch: replace
`),
				Modified: []byte(`
other: a
`),
			},
		},
	},
}

func TestCustomStrategicMergePatch(t *testing.T) {
	for _, c := range customStrategicMergePatchTestCaseData.TestCases {
		originalJSON := yamlToJSONOrError(c.Original)
		twoWayJSON := yamlToJSONOrError(c.TwoWay)
		modifiedJSON := yamlToJSONOrError(c.Modified)
		testPatchApplication(t, originalJSON, twoWayJSON, modifiedJSON, c.Description)
	}
}

// These are test cases for StrategicMergePatch, to assert that applying  a patch
// yields the correct outcome. They are also test cases for CreateTwoWayMergePatch
// and CreateThreeWayMergePatch, to assert that they both generate the correct patch
// for the given set of input documents.
var createStrategicMergePatchTestCaseData = StrategicMergePatchTestCases{
	TestCases: []StrategicMergePatchTestCase{
		{
			Description: "nil original",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				TwoWay: []byte(`
name: 1
value: 1
`),
				Modified: []byte(`
name: 1
value: 1
`),
				Current: []byte(`
name: 1
other: a
`),
				ThreeWay: []byte(`
value: 1
`),
				Result: []byte(`
name: 1
value: 1
other: a
`),
			},
		},
		{
			Description: "nil patch",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
`),
				TwoWay: []byte(`
{}
`),
				Modified: []byte(`
name: 1
`),
				Current: []byte(`
name: 1
`),
				ThreeWay: []byte(`
{}
`),
				Result: []byte(`
name: 1
`),
			},
		},
		{
			Description: "add field to map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
`),
				TwoWay: []byte(`
value: 1
`),
				Modified: []byte(`
name: 1
value: 1
`),
				Current: []byte(`
name: 1
other: a
`),
				ThreeWay: []byte(`
value: 1
`),
				Result: []byte(`
name: 1
value: 1
other: a
`),
			},
		},
		{
			Description: "add field to map with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
`),
				TwoWay: []byte(`
value: 1
`),
				Modified: []byte(`
name: 1
value: 1
`),
				Current: []byte(`
name: a
other: a
`),
				ThreeWay: []byte(`
name: 1
value: 1
`),
				Result: []byte(`
name: 1
value: 1
other: a
`),
			},
		},
		{
			Description: "add field and delete field from map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
`),
				TwoWay: []byte(`
name: null
value: 1
`),
				Modified: []byte(`
value: 1
`),
				Current: []byte(`
name: 1
other: a
`),
				ThreeWay: []byte(`
name: null
value: 1
`),
				Result: []byte(`
value: 1
other: a
`),
			},
		},
		{
			Description: "add field and delete field from map with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
`),
				TwoWay: []byte(`
name: null
value: 1
`),
				Modified: []byte(`
value: 1
`),
				Current: []byte(`
name: a
other: a
`),
				ThreeWay: []byte(`
name: null
value: 1
`),
				Result: []byte(`
value: 1
other: a
`),
			},
		},
		{
			Description: "delete field from nested map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
simpleMap:
  key1: 1
  key2: 1
`),
				TwoWay: []byte(`
simpleMap:
  key2: null
`),
				Modified: []byte(`
simpleMap:
  key1: 1
`),
				Current: []byte(`
simpleMap:
  key1: 1
  key2: 1
  other: a
`),
				ThreeWay: []byte(`
simpleMap:
  key2: null
`),
				Result: []byte(`
simpleMap:
  key1: 1
  other: a
`),
			},
		},
		{
			Description: "delete field from nested map with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
simpleMap:
  key1: 1
  key2: 1
`),
				TwoWay: []byte(`
simpleMap:
  key2: null
`),
				Modified: []byte(`
simpleMap:
  key1: 1
`),
				Current: []byte(`
simpleMap:
  key1: a
  key2: 1
  other: a
`),
				ThreeWay: []byte(`
simpleMap:
  key1: 1
  key2: null
`),
				Result: []byte(`
simpleMap:
  key1: 1
  other: a
`),
			},
		},
		{
			Description: "delete all fields from map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
name: null
value: null
`),
				Modified: []byte(`
{}
`),
				Current: []byte(`
name: 1
value: 1
other: a
`),
				ThreeWay: []byte(`
name: null
value: null
`),
				Result: []byte(`
other: a
`),
			},
		},
		{
			Description: "delete all fields from map with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
name: null
value: null
`),
				Modified: []byte(`
{}
`),
				Current: []byte(`
name: 1
value: a
other: a
`),
				ThreeWay: []byte(`
name: null
value: null
`),
				Result: []byte(`
other: a
`),
			},
		},
		{
			Description: "add field and delete all fields from map",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
name: null
value: null
other: a
`),
				Modified: []byte(`
other: a
`),
				Current: []byte(`
name: 1
value: 1
other: a
`),
				ThreeWay: []byte(`
name: null
value: null
`),
				Result: []byte(`
other: a
`),
			},
		},
		{
			Description: "add field and delete all fields from map with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
name: 1
value: 1
`),
				TwoWay: []byte(`
name: null
value: null
other: a
`),
				Modified: []byte(`
other: a
`),
				Current: []byte(`
name: 1
value: 1
other: b
`),
				ThreeWay: []byte(`
name: null
value: null
other: a
`),
				Result: []byte(`
other: a
`),
			},
		},
		{
			Description: "replace list of scalars",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
nonMergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Modified: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Current: []byte(`
nonMergingIntList:
  - 1
  - 2
`),
				ThreeWay: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Result: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
			},
		},
		{
			Description: "replace list of scalars with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
nonMergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Modified: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Current: []byte(`
nonMergingIntList:
  - 1
  - 4
`),
				ThreeWay: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
				Result: []byte(`
nonMergingIntList:
  - 2
  - 3
`),
			},
		},
		{
			Description: "merge lists of scalars using SMPatchVersion_1_0",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
mergingIntList:
  - 3
`),
				Modified: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
`),
				Current: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
				ThreeWay: []byte(`
mergingIntList:
  - 3
`),
				Result: []byte(`
mergingIntList:
  - 1
  - 2
  - 3
  - 4
`),
			},
		},
		{
			Description: "merge lists of scalars for list of primitives",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
mergingIntList:
  $patch: mergeprimitiveslist
  "1": null
  "3": 3
`),
				Modified: []byte(`
mergingIntList:
  - 2
  - 3
`),
				Current: []byte(`
mergingIntList:
  - 1
  - 2
  - 4
`),
				ThreeWay: []byte(`
mergingIntList:
  $patch: mergeprimitiveslist
  "1": null
  "3": 3
`),
				Result: []byte(`
mergingIntList:
  - 2
  - 3
  - 4
`),
			},
		},
		{
			Description: "another merge lists of scalars for list of primitives",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingIntList:
  - 1
  - 2
`),
				TwoWay: []byte(`
mergingIntList:
  $patch: mergeprimitiveslist
  "1": null
  "3": 3
`),
				Modified: []byte(`
mergingIntList:
  - 2
  - 3
`),
				Current: []byte(`
mergingIntList:
  - 2
  - 4
`),
				ThreeWay: []byte(`
mergingIntList:
  $patch: mergeprimitiveslist
  "1": null
  "3": 3
`),
				Result: []byte(`
mergingIntList:
  - 2
  - 3
  - 4
`),
			},
		},
		{
			Description: "merge lists of maps",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 3
    value: 3
  - name: 4
    value: 4
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
  - name: 3
    value: 3
  - name: 4
    value: 4
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 3
    value: 3
  - name: 4
    value: 4
`),
				Result: []byte(`
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
`),
			},
		},
		{
			Description: "merge lists of maps with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 3
    value: 3
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
  - name: 3
    value: 3
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 3
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 2
    value: 2
  - name: 3
    value: 3
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
  - name: 3
    value: 3
`),
			},
		},
		{
			Description: "add field to map in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: 1
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "add field to map in merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 3
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
  - name: 3
    value: 2
    other: b
`),
			},
		},
		{
			Description: "add duplicate field to map in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
{}
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "add duplicate field to map in merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 3
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 2
    value: 2
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "replace map field value in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: a
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: a
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: a
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: a
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "replace map field value in merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: a
`),
				Modified: []byte(`
mergingList:
  - name: 1
    value: a
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: 3
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: a
`),
				Result: []byte(`
mergingList:
  - name: 1
    value: a
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "delete map from merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Modified: []byte(`
mergingList:
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Result: []byte(`
mergingList:
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "delete map from merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Modified: []byte(`
mergingList:
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Result: []byte(`
mergingList:
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "delete missing map from merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Modified: []byte(`
mergingList:
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Result: []byte(`
mergingList:
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "delete missing map from merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
`),
				Modified: []byte(`
mergingList:
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 3
    other: a
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
`),
				Result: []byte(`
mergingList:
  - name: 2
  - name: 3
    other: a
`),
			},
		},
		{
			Description: "add map and delete map from merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 3
`),
				Modified: []byte(`
mergingList:
  - name: 2
  - name: 3
`),
				Current: []byte(`
mergingList:
  - name: 1
  - name: 2
    other: b
  - name: 4
    other: c
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 3
`),
				Result: []byte(`
mergingList:
  - name: 2
    other: b
  - name: 3
  - name: 4
    other: c
`),
			},
		},
		{
			Description: "add map and delete map from merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 3
`),
				Modified: []byte(`
mergingList:
  - name: 2
  - name: 3
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 4
    other: c
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
  - name: 3
`),
				Result: []byte(`
mergingList:
  - name: 2
  - name: 3
  - name: 4
    other: c
`),
			},
		},
		{
			Description: "delete all maps from merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Modified: []byte(`
mergingList: []
`),
				Current: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Result: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "delete all maps from merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Modified: []byte(`
mergingList: []
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Result: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "delete all maps from empty merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Modified: []byte(`
mergingList: []
`),
				Current: []byte(`
mergingList: []
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    $patch: delete
  - name: 2
    $patch: delete
`),
				Result: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "delete field from map in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "delete field from map in merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    value: a
    other: a
  - name: 2
    value: 2
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
`),
			},
		},
		{
			Description: "delete missing field from map in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "delete missing field from map in merging list with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    value: 1
  - name: 2
    value: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    value: null
`),
				Modified: []byte(`
mergingList:
  - name: 1
  - name: 2
    value: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    value: null
  - name: 2
    value: 2
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
  - name: 2
    value: 2
    other: b
`),
			},
		},
		{
			Description: "replace non merging list nested in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "replace non merging list nested in merging list with value conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 1
        value: c
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "replace non merging list nested in merging list with deletion conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 2
        value: 2
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    nonMergingList:
      - name: 1
        value: 1
`),
				Result: []byte(`
mergingList:
  - name: 1
    other: a
    nonMergingList:
      - name: 1
        value: 1
  - name: 2
    other: b
`),
			},
		},
		{
			Description: "add field to map in merging list nested in merging list",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
      - name: 2
        value: 2
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
    mergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
`),
				Result: []byte(`
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
`),
			},
		},
		{
			Description: "add field to map in merging list nested in merging list with value conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
      - name: 2
        value: 2
  - name: 2
`),
				Current: []byte(`
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
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
      - name: 2
        value: 2
`),
				Result: []byte(`
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
`),
			},
		},
		{
			Description: "add field to map in merging list nested in merging list with deletion conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
      - name: 2
        value: 2
  - name: 2
`),
				Current: []byte(`
mergingList:
  - name: 1
    other: a
    mergingList:
      - name: 2
        value: 2
        other: d
  - name: 2
    other: b
`),
				ThreeWay: []byte(`
mergingList:
  - name: 1
    mergingList:
      - name: 1
        value: 1
`),
				Result: []byte(`
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
`),
			},
		},
		{
			Description: "merge empty merging lists",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergingList: []
`),
				TwoWay: []byte(`
{}
`),
				Modified: []byte(`
mergingList: []
`),
				Current: []byte(`
mergingList: []
`),
				ThreeWay: []byte(`
{}
`),
				Result: []byte(`
mergingList: []
`),
			},
		},
		{
			Description: "add map to merging list by pointer",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergeItemPtr:
  - name: 1
`),
				TwoWay: []byte(`
mergeItemPtr:
  - name: 2
`),
				Modified: []byte(`
mergeItemPtr:
  - name: 1
  - name: 2
`),
				Current: []byte(`
mergeItemPtr:
  - name: 1
    other: a
  - name: 3
`),
				ThreeWay: []byte(`
mergeItemPtr:
  - name: 2
`),
				Result: []byte(`
mergeItemPtr:
  - name: 1
    other: a
  - name: 2
  - name: 3
`),
			},
		},
		{
			Description: "add map to merging list by pointer with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergeItemPtr:
  - name: 1
`),
				TwoWay: []byte(`
mergeItemPtr:
  - name: 2
`),
				Modified: []byte(`
mergeItemPtr:
  - name: 1
  - name: 2
`),
				Current: []byte(`
mergeItemPtr:
  - name: 3
`),
				ThreeWay: []byte(`
mergeItemPtr:
  - name: 1
  - name: 2
`),
				Result: []byte(`
mergeItemPtr:
  - name: 1
  - name: 2
  - name: 3
`),
			},
		},
		{
			Description: "add field to map in merging list by pointer",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
      - name: 2
        value: 2
  - name: 2
`),
				Current: []byte(`
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
`),
				ThreeWay: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
`),
				Result: []byte(`
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
`),
			},
		},
		{
			Description: "add field to map in merging list by pointer with conflict",
			StrategicMergePatchTestCaseData: StrategicMergePatchTestCaseData{
				Original: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
      - name: 2
        value: 2
  - name: 2
`),
				TwoWay: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
`),
				Modified: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
      - name: 2
        value: 2
  - name: 2
`),
				Current: []byte(`
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
`),
				ThreeWay: []byte(`
mergeItemPtr:
  - name: 1
    mergeItemPtr:
      - name: 1
        value: 1
`),
				Result: []byte(`
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
`),
			},
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

	for _, c := range createStrategicMergePatchTestCaseData.TestCases {
		testTwoWayPatch(t, c)
		testThreeWayPatch(t, c)
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
	originalJSON := yamlToJSONOrError(c.Original)
	modifiedJSON := yamlToJSONOrError(c.Modified)
	expectedJSON := yamlToJSONOrError(c.TwoWay)
	smPatchVersion := SMPatchVersion_1_5
	if strings.Contains(c.Description, "using SMPatchVersion_1_0") {
		smPatchVersion = SMPatchVersion_1_0
	}
	actualJSON, err := CreateTwoWayMergePatch(originalJSON, modifiedJSON, mergeItem, smPatchVersion)
	if err != nil {
		t.Errorf("error: %s\nin test case: %s\ncannot create two way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		return
	}

	if !strings.Contains(c.Description, "for list of primitives") {
		testPatchCreation(t, expectedJSON, actualJSON, c.Description)
	} else {
		testPatchCreationWithoutSorting(t, expectedJSON, actualJSON, c.Description)
	}
	testPatchApplication(t, originalJSON, actualJSON, modifiedJSON, c.Description)
}

func testThreeWayPatch(t *testing.T, c StrategicMergePatchTestCase) {
	originalJSON := yamlToJSONOrError(c.Original)
	modifiedJSON := yamlToJSONOrError(c.Modified)
	currentJSON := yamlToJSONOrError(c.Current)
	expectedJSON := yamlToJSONOrError(c.ThreeWay)
	resultJSON := yamlToJSONOrError(c.Result)
	smPatchVersion := SMPatchVersion_1_5
	if strings.Contains(c.Description, "using SMPatchVersion_1_0") {
		smPatchVersion = SMPatchVersion_1_0
	}
	actualJSON, err := CreateThreeWayMergePatch(originalJSON, modifiedJSON, currentJSON, mergeItem, false, smPatchVersion)
	if err != nil {
		if !IsConflict(err) {
			t.Errorf("error: %s\nin test case: %s\ncannot create three way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
				err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
			return
		}

		if !strings.Contains(c.Description, "conflict") {
			t.Errorf("unexpected conflict: %s\nin test case: %s\ncannot create three way patch:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
				err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
		}

		if len(c.Result) > 0 {
			actualJSON, err := CreateThreeWayMergePatch(originalJSON, modifiedJSON, currentJSON, mergeItem, true, smPatchVersion)
			if err != nil {
				t.Errorf("error: %s\nin test case: %s\ncannot force three way patch application:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
					err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
				return
			}

			if !strings.Contains(c.Description, "for list of primitives") {
				testPatchCreation(t, expectedJSON, actualJSON, c.Description)
			} else {
				testPatchCreationWithoutSorting(t, expectedJSON, actualJSON, c.Description)
			}
			testPatchApplication(t, currentJSON, actualJSON, resultJSON, c.Description)
		}

		return
	}

	if strings.Contains(c.Description, "conflict") || len(c.Result) < 1 {
		t.Errorf("error: %s\nin test case: %s\nexpected conflict did not occur:\noriginal:%s\ntwoWay:%s\nmodified:%s\ncurrent:%s\nthreeWay:%s\nresult:%s\n",
			err, c.Description, c.Original, c.TwoWay, c.Modified, c.Current, c.ThreeWay, c.Result)
	}

	if !strings.Contains(c.Description, "for list of primitives") {
		testPatchCreation(t, expectedJSON, actualJSON, c.Description)
	} else {
		testPatchCreationWithoutSorting(t, expectedJSON, actualJSON, c.Description)
	}
	testPatchApplication(t, currentJSON, actualJSON, resultJSON, c.Description)
}

func testPatchCreationWithoutSorting(t *testing.T, expectedYAML, actualYAML []byte, description string) {
	if bytes.Compare(actualYAML, expectedYAML) != 0 {
		t.Errorf("error in test case: %s\nexpected patch:\n%s\ngot:\n%s\n",
			description, jsonToYAMLOrError(expectedYAML), jsonToYAMLOrError(actualYAML))
	}
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

func yamlToJSONOrError(y []byte) []byte {
	j, err := yamlToJSON(y)
	if err != nil {
		return []byte(err.Error())
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
		patch, err := CreateTwoWayMergePatch([]byte(tc.Old), []byte(tc.New), precisionItem, SMPatchVersionLatest)
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
