/*
Copyright The Kubernetes Authors.

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

package apidefinitions

import (
	"reflect"
)

// containsAll tests is a Kubernetes unstructured value is a strict subset of another
// unstructured value. containsAll returns true if subset equals value, or if all
// map and slices values in subset also exist in value.
func containsAll(subset, value any) bool {
	if subset == nil {
		return value == nil
	}
	switch subsetValue := subset.(type) {
	case map[string]any:
		valueMap, ok := value.(map[string]any)
		if !ok {
			return false
		}
		for k, v := range subsetValue {
			sv, exists := valueMap[k]
			if !exists || !containsAll(v, sv) {
				return false
			}
		}
		return true
	case []any:
		valueSlice, ok := value.([]any)
		if !ok {
			return false
		}
		for _, v := range subsetValue {
			matched := false
			for _, sv := range valueSlice {
				if containsAll(v, sv) {
					matched = true
					break
				}
			}
			if !matched {
				return false
			}
		}
		return true
	default:
		return reflect.DeepEqual(subset, value)
	}
}
