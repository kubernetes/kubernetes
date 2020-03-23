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

// These tests are in a separate package to break cyclic dependency in tests.
// Unstructured type depends on unstructured converter package but we want to test how the converter handles
// the Unstructured type so we need to import both.

package runtime_test

import (
	encodingjson "encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var simpleEquality = conversion.EqualitiesOrDie(
	func(a, b time.Time) bool {
		return a.UTC() == b.UTC()
	},
)

// Define a number of test types.
type A struct {
	A int    `json:"aa,omitempty"`
	B string `json:"ab,omitempty"`
	C bool   `json:"ac,omitempty"`
}

type B struct {
	A A                 `json:"ba"`
	B string            `json:"bb"`
	C map[string]string `json:"bc"`
	D []string          `json:"bd"`
}

type C struct {
	A []A `json:"ca"`
	B `json:",inline"`
	C string         `json:"cc"`
	D *int64         `json:"cd"`
	E map[string]int `json:"ce"`
	F []bool         `json:"cf"`
	G []int          `json:"cg"`
	H float32        `json:"ch"`
	I []interface{}  `json:"ci"`
}

type D struct {
	A []interface{} `json:"da"`
}

type E struct {
	A interface{} `json:"ea"`
}

type F struct {
	A string            `json:"fa"`
	B map[string]string `json:"fb"`
	C []A               `json:"fc"`
	D int               `json:"fd"`
	E float32           `json:"fe"`
	F []string          `json:"ff"`
	G []int             `json:"fg"`
	H []bool            `json:"fh"`
	I []float32         `json:"fi"`
}

type G struct {
	CustomValue1   CustomValue    `json:"customValue1"`
	CustomValue2   *CustomValue   `json:"customValue2"`
	CustomPointer1 CustomPointer  `json:"customPointer1"`
	CustomPointer2 *CustomPointer `json:"customPointer2"`
}

type CustomValue struct {
	data []byte
}

// MarshalJSON has a value receiver on this type.
func (c CustomValue) MarshalJSON() ([]byte, error) {
	return c.data, nil
}

type CustomPointer struct {
	data []byte
}

// MarshalJSON has a pointer receiver on this type.
func (c *CustomPointer) MarshalJSON() ([]byte, error) {
	return c.data, nil
}

func doRoundTrip(t *testing.T, item interface{}) {
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
	err = json.Unmarshal(data, unmarshalledObj)
	if err != nil {
		t.Errorf("Error when unmarshaling to object: %v", err)
		return
	}
	if !reflect.DeepEqual(item, unmarshalledObj) {
		t.Errorf("Object changed during JSON operations, diff: %v", diff.ObjectReflectDiff(item, unmarshalledObj))
		return
	}

	// TODO: should be using mismatch detection but fails due to another error
	newUnstr, err := runtime.DefaultUnstructuredConverter.ToUnstructured(item)
	if err != nil {
		t.Errorf("ToUnstructured failed: %v", err)
		return
	}

	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface()
	err = runtime.NewTestUnstructuredConverter(simpleEquality).FromUnstructured(newUnstr, newObj)
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
	testCases := []struct {
		obj interface{}
	}{
		{
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind": "List",
				},
				// Not testing a list with nil Items because items is a non-optional field and hence
				// is always marshaled into an empty array which is not equal to nil when unmarshalled and will fail.
				// That is expected.
				Items: []unstructured.Unstructured{},
			},
		},
		{
			obj: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind": "List",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"kind": "Pod",
						},
					},
				},
			},
		},
		{
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"kind": "Pod",
				},
			},
		},
		{
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Foo",
					"metadata": map[string]interface{}{
						"name": "foo1",
					},
				},
			},
		},
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
			obj: &D{
				A: []interface{}{[]interface{}{}, []interface{}{}},
			},
		},
		{
			// Test slice of interface{} with different values.
			obj: &D{
				A: []interface{}{3.0, "3.0", nil},
			},
		},
	}

	for i := range testCases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			doRoundTrip(t, testCases[i].obj)
		})
	}
}

// Verifies that:
// 1) serialized json -> object
// 2) serialized json -> map[string]interface{} -> object
// produces the same object.
func doUnrecognized(t *testing.T, jsonData string, item interface{}, expectedErr error) {
	unmarshalledObj := reflect.New(reflect.TypeOf(item).Elem()).Interface()
	err := json.Unmarshal([]byte(jsonData), unmarshalledObj)
	if (err != nil) != (expectedErr != nil) {
		t.Errorf("Unexpected error when unmarshaling to object: %v, expected: %v", err, expectedErr)
		return
	}

	unstr := make(map[string]interface{})
	err = json.Unmarshal([]byte(jsonData), &unstr)
	if err != nil {
		t.Errorf("Error when unmarshaling to unstructured: %v", err)
		return
	}
	newObj := reflect.New(reflect.TypeOf(item).Elem()).Interface()
	err = runtime.NewTestUnstructuredConverter(simpleEquality).FromUnstructured(unstr, newObj)
	if (err != nil) != (expectedErr != nil) {
		t.Errorf("Unexpected error in FromUnstructured: %v, expected: %v", err, expectedErr)
	}

	if expectedErr == nil && !reflect.DeepEqual(unmarshalledObj, newObj) {
		t.Errorf("Object changed, diff: %v", diff.ObjectReflectDiff(unmarshalledObj, newObj))
	}
}

func TestUnrecognized(t *testing.T) {
	testCases := []struct {
		data string
		obj  interface{}
		err  error
	}{
		{
			data: "{\"da\":[3.0,\"3.0\",null]}",
			obj:  &D{},
		},
		{
			data: "{\"ea\":[3.0,\"3.0\",null]}",
			obj:  &E{},
		},
		{
			data: "{\"ea\":[null,null,null]}",
			obj:  &E{},
		},
		{
			data: "{\"ea\":[[],[null]]}",
			obj:  &E{},
		},
		{
			data: "{\"ea\":{\"a\":[],\"b\":null}}",
			obj:  &E{},
		},
		{
			data: "{\"fa\":\"fa\",\"fb\":{\"a\":\"a\"}}",
			obj:  &F{},
		},
		{
			data: "{\"fa\":\"fa\",\"fb\":{\"a\":null}}",
			obj:  &F{},
		},
		{
			data: "{\"fc\":[null]}",
			obj:  &F{},
		},
		{
			data: "{\"fc\":[{\"aa\":123,\"ab\":\"bbb\"}]}",
			obj:  &F{},
		},
		{
			// Only unknown fields
			data: "{\"fx\":[{\"aa\":123,\"ab\":\"bbb\"}],\"fz\":123}",
			obj:  &F{},
		},
		{
			data: "{\"fc\":[{\"aa\":\"aaa\",\"ab\":\"bbb\"}]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type int"),
		},
		{
			data: "{\"fd\":123,\"fe\":3.5}",
			obj:  &F{},
		},
		{
			data: "{\"ff\":[\"abc\"],\"fg\":[123],\"fh\":[true,false]}",
			obj:  &F{},
		},
		{
			// Invalid string data
			data: "{\"fa\":123}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type string"),
		},
		{
			// Invalid string data
			data: "{\"fa\":13.5}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type string"),
		},
		{
			// Invalid string data
			data: "{\"fa\":true}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal bool into Go value of type string"),
		},
		{
			// Invalid []string data
			data: "{\"ff\":123}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type []string"),
		},
		{
			// Invalid []string data
			data: "{\"ff\":3.5}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type []string"),
		},
		{
			// Invalid []string data
			data: "{\"ff\":[123,345]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type string"),
		},
		{
			// Invalid []int data
			data: "{\"fg\":123}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type []int"),
		},
		{
			// Invalid []int data
			data: "{\"fg\":\"abc\"}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type []int"),
		},
		{
			// Invalid []int data
			data: "{\"fg\":[\"abc\"]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type int"),
		},
		{
			// Invalid []int data
			data: "{\"fg\":[3.5]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number 3.5 into Go value of type int"),
		},
		{
			// Invalid []int data
			data: "{\"fg\":[true,false]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number 3.5 into Go value of type int"),
		},
		{
			// Invalid []bool data
			data: "{\"fh\":123}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type []bool"),
		},
		{
			// Invalid []bool data
			data: "{\"fh\":\"abc\"}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type []bool"),
		},
		{
			// Invalid []bool data
			data: "{\"fh\":[\"abc\"]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type bool"),
		},
		{
			// Invalid []bool data
			data: "{\"fh\":[3.5]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type bool"),
		},
		{
			// Invalid []bool data
			data: "{\"fh\":[123]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type bool"),
		},
		{
			// Invalid []float data
			data: "{\"fi\":123}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal number into Go value of type []float32"),
		},
		{
			// Invalid []float data
			data: "{\"fi\":\"abc\"}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type []float32"),
		},
		{
			// Invalid []float data
			data: "{\"fi\":[\"abc\"]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal string into Go value of type float32"),
		},
		{
			// Invalid []float data
			data: "{\"fi\":[true]}",
			obj:  &F{},
			err:  fmt.Errorf("json: cannot unmarshal bool into Go value of type float32"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.data, func(t *testing.T) {
			doUnrecognized(t, tc.data, tc.obj, tc.err)
		})
	}
}

func TestDeepCopyJSON(t *testing.T) {
	src := map[string]interface{}{
		"a": nil,
		"b": int64(123),
		"c": map[string]interface{}{
			"a": "b",
		},
		"d": []interface{}{
			int64(1), int64(2),
		},
		"e": "estr",
		"f": true,
		"g": encodingjson.Number("123"),
	}
	deepCopy := runtime.DeepCopyJSON(src)
	assert.Equal(t, src, deepCopy)
}

func TestFloatIntConversion(t *testing.T) {
	unstr := map[string]interface{}{"fd": float64(3)}

	var obj F
	if err := runtime.NewTestUnstructuredConverter(simpleEquality).FromUnstructured(unstr, &obj); err != nil {
		t.Errorf("Unexpected error in FromUnstructured: %v", err)
	}

	data, err := json.Marshal(unstr)
	if err != nil {
		t.Fatalf("Error when marshaling unstructured: %v", err)
	}
	var unmarshalled F
	if err := json.Unmarshal(data, &unmarshalled); err != nil {
		t.Fatalf("Error when unmarshaling to object: %v", err)
	}

	if !reflect.DeepEqual(obj, unmarshalled) {
		t.Errorf("Incorrect conversion, diff: %v", diff.ObjectReflectDiff(obj, unmarshalled))
	}
}

func TestCustomToUnstructured(t *testing.T) {
	testcases := []struct {
		Data     string
		Expected interface{}
	}{
		{Data: `null`, Expected: nil},
		{Data: `true`, Expected: true},
		{Data: `false`, Expected: false},
		{Data: `[]`, Expected: []interface{}{}},
		{Data: `[1]`, Expected: []interface{}{int64(1)}},
		{Data: `{}`, Expected: map[string]interface{}{}},
		{Data: `{"a":1}`, Expected: map[string]interface{}{"a": int64(1)}},
		{Data: `0`, Expected: int64(0)},
		{Data: `0.0`, Expected: float64(0)},
	}

	for _, tc := range testcases {
		tc := tc
		t.Run(tc.Data, func(t *testing.T) {
			t.Parallel()
			result, err := runtime.NewTestUnstructuredConverter(simpleEquality).ToUnstructured(&G{
				CustomValue1:   CustomValue{data: []byte(tc.Data)},
				CustomValue2:   &CustomValue{data: []byte(tc.Data)},
				CustomPointer1: CustomPointer{data: []byte(tc.Data)},
				CustomPointer2: &CustomPointer{data: []byte(tc.Data)},
			})
			require.NoError(t, err)
			for field, fieldResult := range result {
				assert.Equal(t, tc.Expected, fieldResult, field)
			}
		})
	}
}

func TestCustomToUnstructuredTopLevel(t *testing.T) {
	// Only objects are supported at the top level
	topLevelCases := []interface{}{
		&CustomValue{data: []byte(`{"a":1}`)},
		&CustomPointer{data: []byte(`{"a":1}`)},
	}
	expected := map[string]interface{}{"a": int64(1)}
	for i, obj := range topLevelCases {
		obj := obj
		t.Run(strconv.Itoa(i), func(t *testing.T) {
			t.Parallel()
			result, err := runtime.NewTestUnstructuredConverter(simpleEquality).ToUnstructured(obj)
			require.NoError(t, err)
			assert.Equal(t, expected, result)
		})
	}
}
