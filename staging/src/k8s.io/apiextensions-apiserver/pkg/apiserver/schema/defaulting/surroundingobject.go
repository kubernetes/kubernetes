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

// AccessorFunc returns a node x in obj on a fixed (implicitly encoded) JSON path
// if that path exists in obj (found==true). If it does not exist, found is false.
// If on the path the type of a field is wrong, an error is returned.
type AccessorFunc func(obj map[string]interface{}) (x interface{}, found bool, err error)

// SurroundingObjectFunc is a surrounding object builder with a given x at a leaf.
// Which leave is determined by the series of Index() and Child(k) calls.
// It also returns the inverse of the builder, namely the accessor that extracts x
// from the test object.
//
// With obj, acc, _ := someSurroundingObjectFunc(x) we get:
//
//   acc(obj) == x
//   reflect.DeepEqual(acc(DeepCopy(obj), x) == x
//
// where x is the original instance for slices and maps.
//
// If after computation of acc the node holding x in obj is mutated (e.g. pruned),
// the accessor will return that mutated node value (e.g. the pruned x).
//
// Example (ignoring the last two return values):
//
//   NewRootObjectFunc()(x) == x
//   NewRootObjectFunc().Index()(x) == [x]
//   NewRootObjectFunc().Index().Child("foo") == [{"foo": x}]
//   NewRootObjectFunc().Index().Child("foo").Child("bar") == [{"foo": {"bar":x}}]
//   NewRootObjectFunc().Index().Child("foo").Child("bar").Index() == [{"foo": {"bar":[x]}}]
//
// and:
//
//   NewRootObjectFunc(), then acc(x) == x
//   NewRootObjectFunc().Index(), then acc([x]) == x
//   NewRootObjectFunc().Index().Child("foo"), then acc([{"foo": x}]) == x
//   NewRootObjectFunc().Index().Child("foo").Child("bar"), then acc([{"foo": {"bar":x}}]) == x
//   NewRootObjectFunc().Index().Child("foo").Child("bar").Index(), then acc([{"foo": {"bar":[x]}}]) == x
type SurroundingObjectFunc func(focus interface{}) (map[string]interface{}, AccessorFunc, error)

// NewRootObjectFunc returns the identity function. The passed focus value
// must be an object.
func NewRootObjectFunc() SurroundingObjectFunc {
	return func(x interface{}) (map[string]interface{}, AccessorFunc, error) {
		obj, ok := x.(map[string]interface{})
		if !ok {
			return nil, nil, fmt.Errorf("object root default value must be of object type")
		}
		return obj, func(root map[string]interface{}) (interface{}, bool, error) {
			return root, true, nil
		}, nil
	}
}

// WithTypeMeta returns a closure with the TypeMeta fields set if they are defined.
// This mutates f(x).
func (f SurroundingObjectFunc) WithTypeMeta(meta metav1.TypeMeta) SurroundingObjectFunc {
	return func(x interface{}) (map[string]interface{}, AccessorFunc, error) {
		obj, acc, err := f(x)
		if err != nil {
			return nil, nil, err
		}
		if obj == nil {
			obj = map[string]interface{}{}
		}
		if _, found := obj["kind"]; !found {
			obj["kind"] = meta.Kind
		}
		if _, found := obj["apiVersion"]; !found {
			obj["apiVersion"] = meta.APIVersion
		}
		return obj, acc, err
	}
}

// Child returns a function x => f({k: x}) and the corresponding accessor.
func (f SurroundingObjectFunc) Child(k string) SurroundingObjectFunc {
	return func(x interface{}) (map[string]interface{}, AccessorFunc, error) {
		obj, acc, err := f(map[string]interface{}{k: x})
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

// Index returns a function x => f([x]) and the corresponding accessor.
func (f SurroundingObjectFunc) Index() SurroundingObjectFunc {
	return func(focus interface{}) (map[string]interface{}, AccessorFunc, error) {
		obj, acc, err := f([]interface{}{focus})
		if err != nil {
			return nil, nil, err
		}
		return obj, func(obj map[string]interface{}) (interface{}, bool, error) {
			x, found, err := acc(obj)
			if err != nil {
				return nil, false, fmt.Errorf("[]%v", err)
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
