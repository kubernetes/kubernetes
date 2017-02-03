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

package mergepatchutil

import (
	"fmt"
	"reflect"

	"github.com/davecgh/go-spew/spew"
	"github.com/ghodss/yaml"
)

var ErrBadJSONDoc = fmt.Errorf("Invalid JSON document")
var ErrNoListOfLists = fmt.Errorf("Lists of lists are not supported")
var ErrBadPatchFormatForPrimitiveList = fmt.Errorf("Invalid patch format of primitive list")

func ToYAMLOrError(v interface{}) string {
	y, err := toYAML(v)
	if err != nil {
		return err.Error()
	}

	return y
}

func toYAML(v interface{}) (string, error) {
	y, err := yaml.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("yaml marshal failed:%v\n%v\n", err, spew.Sdump(v))
	}

	return string(y), nil
}

// HasConflicts returns true if the left and right JSON interface objects overlap with
// different values in any key. All keys are required to be strings. Since patches of the
// same Type have congruent keys, this is valid for multiple patch types. This method
// supports JSON merge patch semantics.
func HasConflicts(left, right interface{}) (bool, error) {
	switch typedLeft := left.(type) {
	case map[string]interface{}:
		switch typedRight := right.(type) {
		case map[string]interface{}:
			for key, leftValue := range typedLeft {
				rightValue, ok := typedRight[key]
				if !ok {
					return false, nil
				}
				return HasConflicts(leftValue, rightValue)
			}

			return false, nil
		default:
			return true, nil
		}
	case []interface{}:
		switch typedRight := right.(type) {
		case []interface{}:
			if len(typedLeft) != len(typedRight) {
				return true, nil
			}

			for i := range typedLeft {
				return HasConflicts(typedLeft[i], typedRight[i])
			}

			return false, nil
		default:
			return true, nil
		}
	case string, float64, bool, int, int64, nil:
		return !reflect.DeepEqual(left, right), nil
	default:
		return true, fmt.Errorf("unknown type: %v", reflect.TypeOf(left))
	}
}
