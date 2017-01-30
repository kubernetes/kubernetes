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
	"k8s.io/apimachinery/pkg/util/strategicpatch"
)

// Create a 3-way merge patch based-on JSON merge patch.
// Calculate addition-and-change patch between current and modified.
// Calculate deletion patch between original and modified.
func CreateThreeWayJSONMergePatch(original, modified, current []byte) ([]byte, error) {
	if len(original) == 0 {
		original = []byte(`{}`)
	}
	if len(modified) == 0 {
		modified = []byte(`{}`)
	}
	if len(current) == 0 {
		current = []byte(`{}`)
	}

	addAndChangePatch, err := jsonpatch.CreateMergePatch(current, modified)
	if err != nil {
		return nil, err
	}
	// Only keep addition and changes
	addAndChangePatch, addAndChangePatchObj, err := keepOrDeleteNullInJsonPatch(addAndChangePatch, false)
	if err != nil {
		return nil, err
	}

	deletePatch, err := jsonpatch.CreateMergePatch(original, modified)
	if err != nil {
		return nil, err
	}
	// Only keep deletion
	deletePatch, deletePatchObj, err := keepOrDeleteNullInJsonPatch(deletePatch, true)
	if err != nil {
		return nil, err
	}

	hasConflicts, err := strategicpatch.HasConflicts(addAndChangePatchObj, deletePatchObj)
	if err != nil {
		return nil, err
	}
	if hasConflicts {
		return nil, fmt.Errorf("changes are in conflict")
	}
	return jsonpatch.MergePatch(deletePatch, addAndChangePatch)
}

// keepOrDeleteNullInJsonPatch takes a json-encoded byte array and a boolean.
// It returns a filtered object and its corresponding json-encoded byte array.
// It is a wrapper of func keepOrDeleteNullInObj
func keepOrDeleteNullInJsonPatch(j []byte, keepNull bool) ([]byte, interface{}, error) {
	var obj map[string]interface{}
	err := json.Unmarshal(j, &obj)
	if err != nil {
		return nil, nil, err
	}
	filteredObj, err := keepOrDeleteNullInObj(obj, keepNull)
	if err != nil {
		return nil, nil, err
	}
	o, err := json.Marshal(filteredObj)
	return o, filteredObj, err
}

// keepOrDeleteNullInObj will keep only the null value and delete all the others,
// if keepNull is true. Otherwise, it will delete all the null value and keep the others.
func keepOrDeleteNullInObj(m map[string]interface{}, keepNull bool) (map[string]interface{}, error) {
	filteredMap := make(map[string]interface{})
	var err error
	for key, val := range m {
		switch {
		case keepNull && val == nil:
			filteredMap[key] = nil
		case val != nil:
			switch typedVal := val.(type) {
			case map[string]interface{}:
				filteredMap[key], err = keepOrDeleteNullInObj(typedVal, keepNull)
				if err != nil {
					return nil, err
				}
			case []interface{}, string, float64, bool, int, int64, nil:
				// Lists are always replaced in Json, no need to check each entry in the list.
				if !keepNull {
					filteredMap[key] = val
				}
			default:
				panic(fmt.Sprintf("unknown type: %v", reflect.TypeOf(typedVal)))
			}
		}
	}
	return filteredMap, nil
}
