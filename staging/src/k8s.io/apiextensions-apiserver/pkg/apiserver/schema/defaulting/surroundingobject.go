/*
Copyright 2019 The Kubernetes Authors.

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

package defaulting

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// FocusAccessorFunc encodes a JSON path through the given object, returning a leave,
// with the bool being false if the path does not exist.
type FocusAccessorFunc func(map[string]interface{}) (interface{}, bool, error)

// SurroundingObjectFunc is a closure returning a test object (as JSON data structure), with the
// focus value plugged in at the leaf.
type SurroundingObjectFunc func(focus interface{}) (map[string]interface{}, FocusAccessorFunc, error)

// NewRootObjectFunc returns a root closure of an object. The passed focus value
// must be an object.
func NewRootObjectFunc() SurroundingObjectFunc {
	return func(focus interface{}) (map[string]interface{}, FocusAccessorFunc, error) {
		obj, ok := focus.(map[string]interface{})
		if !ok {
			return nil, nil, fmt.Errorf("object root default value must be of object type")
		}
		return obj, func(root map[string]interface{}) (interface{}, bool, error) {
			return root, true, nil
		}, nil
	}
}

// WithTypeMeta returns a closure with the TypeMeta fields set.
func (f SurroundingObjectFunc) WithTypeMeta(meta metav1.TypeMeta) SurroundingObjectFunc {
	return func(focus interface{}) (map[string]interface{}, FocusAccessorFunc, error) {
		obj, acc, err := f(focus)
		if err != nil {
			return nil, nil, err
		}
		if obj == nil {
			obj = map[string]interface{}{}
		}
		obj["kind"] = meta.Kind
		obj["apiVersion"] = meta.APIVersion
		return obj, acc, err
	}
}

// Child returns a closure with the focus object as the given child.
func (f SurroundingObjectFunc) Child(k string) SurroundingObjectFunc {
	return func(focus interface{}) (map[string]interface{}, FocusAccessorFunc, error) {
		obj, acc, err := f(map[string]interface{}{k: focus})
		if err != nil {
			return nil, nil, err
		}
		return obj, func(obj map[string]interface{}) (interface{}, bool, error) {
			x, found, err := acc(obj)
			if err != nil {
				return nil, false, fmt.Errorf(".%s%v", k, err)
			}
			if !found {
				return nil, false, nil
			}
			if x, ok := x.(map[string]interface{}); !ok {
				return nil, false, fmt.Errorf(".%s must be of object type", k)
			} else if v, found := x[k]; !found {
				return nil, false, nil
			} else {
				return v, true, nil
			}
		}, err
	}
}

// Index returns a closure with the focus object at index 0.
func (f SurroundingObjectFunc) Index() SurroundingObjectFunc {
	return func(focus interface{}) (map[string]interface{}, FocusAccessorFunc, error) {
		obj, acc, err := f([]interface{}{focus})
		if err != nil {
			return nil, nil, err
		}
		return obj, func(obj map[string]interface{}) (interface{}, bool, error) {
			x, found, err := acc(obj)
			if err != nil {
				return nil, false, fmt.Errorf("[]%s", err)
			}
			if !found {
				return nil, false, nil
			}
			if x, ok := x.([]interface{}); !ok {
				return nil, false, fmt.Errorf("[] must be of array type")
			} else if len(x) == 0 {
				return nil, false, nil
			} else {
				return x[0], true, nil
			}
		}, err
	}
}
