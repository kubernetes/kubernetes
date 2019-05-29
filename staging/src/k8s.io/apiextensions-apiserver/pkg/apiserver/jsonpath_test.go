/*
Copyright 2018 The Kubernetes Authors.

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

package apiserver

import (
	"bytes"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
)

type (
	jsonPathNode struct {
		index *int
		field string
	}
	JsonPath []jsonPathNode
)

func (p JsonPath) String() string {
	var buf bytes.Buffer
	for _, n := range p {
		if n.index == nil {
			buf.WriteString("." + n.field)
		} else {
			buf.WriteString(fmt.Sprintf("[%d]", *n.index))
		}
	}
	return buf.String()
}

func jsonPaths(base JsonPath, j map[string]interface{}) []JsonPath {
	res := make([]JsonPath, 0, len(j))
	for k, old := range j {
		kPth := append(append([]jsonPathNode(nil), base...), jsonPathNode{field: k})
		res = append(res, kPth)

		switch old := old.(type) {
		case map[string]interface{}:
			res = append(res, jsonPaths(kPth, old)...)
		case []interface{}:
			res = append(res, jsonIterSlice(kPth, old)...)
		}
	}
	return res
}

func jsonIterSlice(base JsonPath, j []interface{}) []JsonPath {
	res := make([]JsonPath, 0, len(j))
	for i, old := range j {
		index := i
		iPth := append(append([]jsonPathNode(nil), base...), jsonPathNode{index: &index})
		res = append(res, iPth)

		switch old := old.(type) {
		case map[string]interface{}:
			res = append(res, jsonPaths(iPth, old)...)
		case []interface{}:
			res = append(res, jsonIterSlice(iPth, old)...)
		}
	}
	return res
}

func JsonPathValue(j map[string]interface{}, pth JsonPath, base int) (interface{}, error) {
	if len(pth) == base {
		return nil, fmt.Errorf("empty json path is invalid for object")
	}
	if pth[base].index != nil {
		return nil, fmt.Errorf("index json path is invalid for object")
	}
	field, ok := j[pth[base].field]
	if !ok || len(pth) == base+1 {
		if len(pth) > base+1 {
			return nil, fmt.Errorf("invalid non-terminal json path %q for non-existing field", pth)
		}
		return j[pth[base].field], nil
	}
	switch field := field.(type) {
	case map[string]interface{}:
		return JsonPathValue(field, pth, base+1)
	case []interface{}:
		return jsonPathValueSlice(field, pth, base+1)
	default:
		return nil, fmt.Errorf("invalid non-terminal json path %q for field", pth[:base+1])
	}
}

func jsonPathValueSlice(j []interface{}, pth JsonPath, base int) (interface{}, error) {
	if len(pth) == base {
		return nil, fmt.Errorf("empty json path %q is invalid for object", pth)
	}
	if pth[base].index == nil {
		return nil, fmt.Errorf("field json path %q is invalid for object", pth[:base+1])
	}
	if *pth[base].index >= len(j) {
		return nil, fmt.Errorf("invalid index %q for array of size %d", pth[:base+1], len(j))
	}
	if len(pth) == base+1 {
		return j[*pth[base].index], nil
	}
	switch item := j[*pth[base].index].(type) {
	case map[string]interface{}:
		return JsonPathValue(item, pth, base+1)
	case []interface{}:
		return jsonPathValueSlice(item, pth, base+1)
	default:
		return nil, fmt.Errorf("invalid non-terminal json path %q for index", pth[:base+1])
	}
}

func SetJsonPath(j map[string]interface{}, pth JsonPath, base int, value interface{}) error {
	if len(pth) == base {
		return fmt.Errorf("empty json path is invalid for object")
	}
	if pth[base].index != nil {
		return fmt.Errorf("index json path is invalid for object")
	}
	field, ok := j[pth[base].field]
	if !ok || len(pth) == base+1 {
		if len(pth) > base+1 {
			return fmt.Errorf("invalid non-terminal json path %q for non-existing field", pth)
		}
		j[pth[base].field] = runtime.DeepCopyJSONValue(value)
		return nil
	}
	switch field := field.(type) {
	case map[string]interface{}:
		return SetJsonPath(field, pth, base+1, value)
	case []interface{}:
		return setJsonPathSlice(field, pth, base+1, value)
	default:
		return fmt.Errorf("invalid non-terminal json path %q for field", pth[:base+1])
	}
}

func setJsonPathSlice(j []interface{}, pth JsonPath, base int, value interface{}) error {
	if len(pth) == base {
		return fmt.Errorf("empty json path %q is invalid for object", pth)
	}
	if pth[base].index == nil {
		return fmt.Errorf("field json path %q is invalid for object", pth[:base+1])
	}
	if *pth[base].index >= len(j) {
		return fmt.Errorf("invalid index %q for array of size %d", pth[:base+1], len(j))
	}
	if len(pth) == base+1 {
		j[*pth[base].index] = runtime.DeepCopyJSONValue(value)
		return nil
	}
	switch item := j[*pth[base].index].(type) {
	case map[string]interface{}:
		return SetJsonPath(item, pth, base+1, value)
	case []interface{}:
		return setJsonPathSlice(item, pth, base+1, value)
	default:
		return fmt.Errorf("invalid non-terminal json path %q for index", pth[:base+1])
	}
}

func DeleteJsonPath(j map[string]interface{}, pth JsonPath, base int) error {
	if len(pth) == base {
		return fmt.Errorf("empty json path is invalid for object")
	}
	if pth[base].index != nil {
		return fmt.Errorf("index json path is invalid for object")
	}
	field, ok := j[pth[base].field]
	if !ok || len(pth) == base+1 {
		if len(pth) > base+1 {
			return fmt.Errorf("invalid non-terminal json path %q for non-existing field", pth)
		}
		delete(j, pth[base].field)
		return nil
	}
	switch field := field.(type) {
	case map[string]interface{}:
		return DeleteJsonPath(field, pth, base+1)
	case []interface{}:
		if len(pth) == base+2 {
			if pth[base+1].index == nil {
				return fmt.Errorf("field json path %q is invalid for object", pth)
			}
			j[pth[base].field] = append(field[:*pth[base+1].index], field[*pth[base+1].index+1:]...)
			return nil
		}
		return deleteJsonPathSlice(field, pth, base+1)
	default:
		return fmt.Errorf("invalid non-terminal json path %q for field", pth[:base+1])
	}
}

func deleteJsonPathSlice(j []interface{}, pth JsonPath, base int) error {
	if len(pth) == base {
		return fmt.Errorf("empty json path %q is invalid for object", pth)
	}
	if pth[base].index == nil {
		return fmt.Errorf("field json path %q is invalid for object", pth[:base+1])
	}
	if *pth[base].index >= len(j) {
		return fmt.Errorf("invalid index %q for array of size %d", pth[:base+1], len(j))
	}
	if len(pth) == base+1 {
		return fmt.Errorf("cannot delete item at index %q in-place", pth[:base])
	}
	switch item := j[*pth[base].index].(type) {
	case map[string]interface{}:
		return DeleteJsonPath(item, pth, base+1)
	case []interface{}:
		if len(pth) == base+2 {
			if pth[base+1].index == nil {
				return fmt.Errorf("field json path %q is invalid for object", pth)
			}
			j[*pth[base].index] = append(item[:*pth[base+1].index], item[*pth[base+1].index+1:])
			return nil
		}
		return deleteJsonPathSlice(item, pth, base+1)
	default:
		return fmt.Errorf("invalid non-terminal json path %q for index", pth[:base+1])
	}
}
