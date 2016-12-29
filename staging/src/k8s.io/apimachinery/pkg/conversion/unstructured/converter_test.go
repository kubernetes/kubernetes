/*
Copyright 2015 The Kubernetes Authors.

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

package unstructured

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"
)

// Definte a number of test types.
type A struct {
	A int `json:"aa,omitempty"`
	B string `json:"ab,omitempty"`
	C bool `json:"ac,omitempty"`
}

type B struct {
	A A `json:"ba"`
	B string `json:"bb"`
	C map[string]string `json:"bc"`
	D []string `json:"bd"`
}

type C struct {
	A []A `json:"ca"`
	B B `json:",inline"`
	C string `json:"cc"`
	D *int64 `json:"cd"`
	E map[string]int `json:"ce"`
	F []bool `json:"cf"`
	G []int `json"cg"`
	H float32 `json:ch"`
	I []interface{} `json:"ci"`
}

// C needs to implement runtime.Object to make it usable for tests.
func (c *C) GetObjectKind() schema.ObjectKind {
	return schema.EmptyObjectKind
}

func doRoundTrip(t *testing.T, item runtime.Object) {
	data, err := json.Marshal(item)
	if err != nil {
		t.Errorf("Error when marshaling object: %v", err)
		return
	}

	unstr := make(map[string]interface{})
	err = json.Unmarshal(data, &unstr)
	if err != nil {
		t.Errorf("Error when unmarshaling to unstructured: %v", err)
		return
	}

	data, err = json.Marshal(unstr)
	if err != nil {
		t.Errorf("Error when marshaling unstructured: %v", err)
		return
	}
	unmarshalledObj := reflect.New(reflect.TypeOf(item).Elem()).Interface()
	err = json.Unmarshal(data, &unmarshalledObj)
	if err != nil {
		t.Errorf("Error when unmarshaling to object: %v", err)
		return
	}
	if !reflect.DeepEqual(item, unmarshalledObj) {
		t.Errorf("Object changed during JSON operations, diff: %v", diff.ObjectReflectDiff(item, unmarshalledObj))
		return
	}

	newUnstr := make(map[string]interface{})
	err = NewConverter().ToUnstructured(item, &newUnstr)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}

	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	err = NewConverter().FromUnstructured(newUnstr, newObj)
	if err != nil {
		t.Errorf("FromUnstructured failed: %v", err)
		return
	}

	if !reflect.DeepEqual(item, newObj) {
		t.Errorf("Object changed, diff: %v", diff.ObjectReflectDiff(item, newObj))
	}
}

func TestRoundTrip(t *testing.T) {
	intVal := int64(42)
	testCases := []struct{
		obj runtime.Object
	}{
		{
			// This (among others) tests nil map, slice and pointer.
			obj: &C{
				C: "ccc",
			},
		},
		{
			// This (among others) tests empty map and slice.
			obj: &C{
				A: []A{},
				C: "ccc",
				E: map[string]int{},
				I: []interface{}{},
			},
		},
		{
			obj: &C{
				A: []A{
					{
						A: 1,
						B: "11",
						C: true,
					},
					{
						A: 2,
						B: "22",
						C: false,
					},
				},
				B: B{
					A: A{
						A: 3,
						B: "33",
					},
					B: "bbb",
					C: map[string]string{
						"k1": "v1",
						"k2": "v2",
					},
					D: []string{"s1", "s2"},
				},
				C: "ccc",
				D: &intVal,
				E: map[string]int{
					"k1": 1,
					"k2": 2,
				},
				F: []bool{true, false, false},
				G: []int{1, 2, 5},
				H: 3.3,
				I: []interface{}{nil, nil, nil},
			},
		},
		{
			// Test slice of interface{} with empty slices.
			obj: &C{
				I: []interface{}{[]interface{}{}, []interface{}{}},
			},
		},
	}

	for i := range testCases {
		doRoundTrip(t, testCases[i].obj)
		if t.Failed() {
			break
		}
	}
}
