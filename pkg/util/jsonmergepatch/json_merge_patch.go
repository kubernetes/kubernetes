/*
Copyright 2017 The Kubernetes Authors.

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

package jsonmergepatch

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/evanphx/json-patch"
)

// Create a 3-way merge patch based-on JSON merge patch.
// Calculate addition-and-change patch between current and modified.
// Calculate deletion patch between original and modified.
func CreateThreeWayJSONMergePatch(original, modified, current []byte, overwrite bool) ([]byte, error) {
	addAndChangePatch, err := jsonpatch.CreateMergePatch(current, modified)
	if err != nil {
		return nil, err
	}
	// Only keep addition and changes
	addAndChangePatch, err = keepOrDeleteNullInJsonPatch(addAndChangePatch, false)
	if err != nil {
		return nil, err
	}

	deletePatch, err := jsonpatch.CreateMergePatch(original, modified)
	if err != nil {
		return nil, err
	}
	// Only keep deletion
	deletePatch, err = keepOrDeleteNullInJsonPatch(deletePatch, true)
	if err != nil {
		return nil, err
	}
	// Merge the 2 patches if no conflict
	if !overwrite && hasConflicts(addAndChangePatch, deletePatch) {
		return nil, fmt.Errorf("changes are in conflict")
	}
	return jsonpatch.MergePatch(deletePatch, addAndChangePatch)
}

func keepOrDeleteNullInJsonPatch(j []byte, keepNull bool) ([]byte, error) {
	var obj map[string]interface{}
	err := json.Unmarshal(j, &obj)
	if err != nil {
		return nil, err
	}
	filteredObj, err := keepOrDeleteNullInObj(obj, keepNull)
	if err != nil {
		return nil, err
	}
	return json.Marshal(filteredObj)
}

func keepOrDeleteNullInObj(m map[string]interface{}, keepNull bool) (map[string]interface{}, error) {
	filteredMap := make(map[string]interface{})
	var err error
	for key, val := range m {
		if keepNull && val == nil {
			filteredMap[key] = nil
			continue
		}
		if !keepNull && val != nil {
			switch typedVal := val.(type) {
			case map[string]interface{}:
				filteredMap[key], err = keepOrDeleteNullInObj(typedVal, keepNull)
				if err != nil {
					return nil, err
				}
			case []interface{}, string, float64, bool, int, int64, nil:
				// Lists are always replaced in Json, no need to check each item.
				filteredMap[key] = val
			default:
				panic(fmt.Sprintf("unknown type: %v", reflect.TypeOf(typedVal)))
			}
		}
	}
	return filteredMap, nil
}

// This function is borrowed from openshift/origin.
func hasConflicts(left, right interface{}) bool {
	switch typedLeft := left.(type) {
	case map[string]interface{}:
		switch typedRight := right.(type) {
		case map[string]interface{}:
			for key, leftValue := range typedLeft {
				if rightValue, ok := typedRight[key]; ok && hasConflicts(leftValue, rightValue) {
					return true
				}
			}
			return false
		default:
			return true
		}
	case []interface{}:
		switch typedRight := right.(type) {
		case []interface{}:
			if len(typedLeft) != len(typedRight) {
				return true
			}
			for i := range typedLeft {
				if hasConflicts(typedLeft[i], typedRight[i]) {
					return true
				}
			}
			return false
		default:
			return true
		}
	case string, float64, bool, int, int64, nil:
		return !reflect.DeepEqual(left, right)
	default:
		panic(fmt.Sprintf("unknown type: %v", reflect.TypeOf(left)))
	}
}
