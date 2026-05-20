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

package common

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
)

type customJSONMarshaler struct {
	Val string
}

func (c customJSONMarshaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"marshaled": c.Val})
}

type customUnstructuredWrapper struct {
	Val string
}

func (c *customUnstructuredWrapper) ToUnstructured() interface{} {
	return map[string]interface{}{"unstructured": c.Val}
}

type customPointerJSONMarshaler struct {
	Val string
}

func (c *customPointerJSONMarshaler) MarshalJSON() ([]byte, error) {
	return json.Marshal(map[string]string{"marshaled": c.Val})
}

type myInt int
type myInt32 int32
type myInt64 int64
type myFloat32 float32
type myString string
type myBool bool
type myFloat float64

type TestStruct struct {
	FieldA string `json:"fieldA,omitempty"`
	FieldB int    `json:"fieldB"`
	FieldC *int   `json:"fieldC,omitempty"`
}

type OmitTestStruct struct {
	FieldNoOmit *int `json:"fieldNoOmit"`
	FieldOmit   *int `json:"fieldOmit,omitempty"`
}

func TestSchemalessTypedToVal_Primitives(t *testing.T) {
	tests := []struct {
		name     string
		val      interface{}
		expected ref.Val
	}{
		{
			name:     "nil",
			val:      nil,
			expected: types.NullValue,
		},
		{
			name:     "nil pointer",
			val:      (*int)(nil),
			expected: types.NullValue,
		},
		{
			name:     "pointer to int",
			val:      func() *int { i := 42; return &i }(),
			expected: types.Int(42),
		},
		{
			name:     "bool true",
			val:      true,
			expected: types.Bool(true),
		},
		{
			name:     "bool false",
			val:      false,
			expected: types.Bool(false),
		},
		{
			name:     "int",
			val:      int(12),
			expected: types.Int(12),
		},
		{
			name:     "int32",
			val:      int32(32),
			expected: types.Int(32),
		},
		{
			name:     "int64",
			val:      int64(64),
			expected: types.Int(64),
		},
		{
			name:     "float32",
			val:      float32(3.14),
			expected: types.Double(float32(3.14)),
		},
		{
			name:     "float64",
			val:      float64(3.14159),
			expected: types.Double(3.14159),
		},
		{
			name:     "string",
			val:      "hello",
			expected: types.String("hello"),
		},
		{
			name:     "bytes slice nil",
			val:      ([]byte)(nil),
			expected: types.NullValue,
		},
		{
			name:     "bytes slice",
			val:      []byte("hello"),
			expected: types.String("aGVsbG8="),
		},
		{
			name:     "custom type myInt",
			val:      myInt(42),
			expected: types.Int(42),
		},
		{
			name:     "custom type myInt32",
			val:      myInt32(32),
			expected: types.Int(32),
		},
		{
			name:     "custom type myInt64",
			val:      myInt64(64),
			expected: types.Int(64),
		},
		{
			name:     "custom type myFloat32",
			val:      myFloat32(3.14),
			expected: types.Double(float32(3.14)),
		},
		{
			name:     "custom type myString",
			val:      myString("abc"),
			expected: types.String("abc"),
		},
		{
			name:     "custom type myBool",
			val:      myBool(true),
			expected: types.Bool(true),
		},
		{
			name:     "custom type myFloat",
			val:      myFloat(1.23),
			expected: types.Double(1.23),
		},
		{
			name:     "unsupported channel type",
			val:      make(chan int),
			expected: types.NewErr("unsupported Go type for CEL: chan int"),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SchemalessTypedToVal(tc.val)
			if got.Type() != tc.expected.Type() {
				t.Fatalf("expected type %s, got %s", tc.expected.Type(), got.Type())
			}
			if got.Value() != tc.expected.Value() {
				if gotErr, ok := got.(*types.Err); ok {
					if tcErr, ok := tc.expected.(*types.Err); ok {
						if gotErr.Error() == tcErr.Error() {
							return
						}
					}
					t.Fatalf("got error %v, expected %v", gotErr, tc.expected)
				}
				t.Fatalf("expected value %v, got %v", tc.expected.Value(), got.Value())
			}
		})
	}
}

func TestSchemalessTypedToVal_K8sTypes(t *testing.T) {
	timeVal := time.Date(2026, 5, 18, 12, 0, 0, 0, time.UTC)
	microTimeVal := time.Date(2026, 5, 18, 12, 0, 0, 1000, time.UTC)

	tests := []struct {
		name     string
		val      interface{}
		expected ref.Val
	}{
		{
			name:     "metav1.Time zero",
			val:      metav1.Time{},
			expected: types.NullValue,
		},
		{
			name:     "metav1.Time non-zero",
			val:      metav1.NewTime(timeVal),
			expected: types.String("2026-05-18T12:00:00Z"),
		},
		{
			name:     "metav1.MicroTime zero",
			val:      metav1.MicroTime{},
			expected: types.NullValue,
		},
		{
			name:     "metav1.MicroTime non-zero",
			val:      metav1.NewMicroTime(microTimeVal),
			expected: types.String("2026-05-18T12:00:00.000001Z"),
		},
		{
			name:     "metav1.Duration",
			val:      metav1.Duration{Duration: 5 * time.Minute},
			expected: types.String("5m0s"),
		},
		{
			name:     "intstr.IntOrString int",
			val:      intstr.FromInt32(100),
			expected: types.Int(100),
		},
		{
			name:     "intstr.IntOrString string",
			val:      intstr.FromString("http"),
			expected: types.String("http"),
		},
		{
			name:     "resource.Quantity",
			val:      resource.MustParse("200m"),
			expected: types.String("200m"),
		},
		{
			name:     "json.Marshaler",
			val:      customJSONMarshaler{Val: "foo"},
			expected: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"marshaled": "foo"}),
		},
		{
			name:     "customUnstructuredWrapper by value",
			val:      customUnstructuredWrapper{Val: "bar"},
			expected: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"unstructured": "bar"}),
		},
		{
			name:     "customPointerJSONMarshaler by value",
			val:      customPointerJSONMarshaler{Val: "baz"},
			expected: types.DefaultTypeAdapter.NativeToValue(map[string]interface{}{"marshaled": "baz"}),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := SchemalessTypedToVal(tc.val)
			if got.Type() != tc.expected.Type() {
				t.Fatalf("expected type %s, got %s", tc.expected.Type(), got.Type())
			}
			if !reflect.DeepEqual(got.Value(), tc.expected.Value()) {
				t.Fatalf("expected value %v, got %v", tc.expected.Value(), got.Value())
			}
		})
	}
}

func TestReflectSchemalessTypedList(t *testing.T) {
	val := []int{1, 2, 3}
	cellVal := SchemalessTypedToVal(val)
	list, ok := cellVal.(*reflectSchemalessTypedList)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedList, got %T", cellVal)
	}

	// Type and Value
	if list.Type() != types.ListType {
		t.Errorf("expected ListType, got %s", list.Type())
	}
	if !reflect.DeepEqual(list.Value(), val) {
		t.Errorf("expected underlying slice %v, got %v", val, list.Value())
	}

	// Size
	if list.Size() != types.Int(3) {
		t.Errorf("expected size 3, got %v", list.Size())
	}

	// Get
	if item := list.Get(types.Int(1)); item.Value() != int64(2) {
		t.Errorf("expected item at 1 to be 2, got %v", item.Value())
	}
	// Index out of bounds
	if errVal := list.Get(types.Int(5)); !types.IsError(errVal) {
		t.Errorf("expected error for out of bounds index, got %v", errVal)
	}
	// Invalid index type
	if errVal := list.Get(types.String("foo")); !types.IsError(errVal) {
		t.Errorf("expected error for string index, got %v", errVal)
	}

	// Contains
	if has := list.Contains(types.Int(2)); has != types.True {
		t.Errorf("expected list to contain 2")
	}
	if has := list.Contains(types.Int(4)); has != types.False {
		t.Errorf("expected list to not contain 4")
	}

	// Equal
	otherSame := types.DefaultTypeAdapter.NativeToValue([]int{1, 2, 3})
	if list.Equal(otherSame) != types.True {
		t.Errorf("expected lists to be equal")
	}
	otherDiffVal := types.DefaultTypeAdapter.NativeToValue([]int{1, 2, 4})
	if list.Equal(otherDiffVal) != types.False {
		t.Errorf("expected lists with different elements to not be equal")
	}
	otherDiffLen := types.DefaultTypeAdapter.NativeToValue([]int{1, 2})
	if list.Equal(otherDiffLen) != types.False {
		t.Errorf("expected lists with different lengths to not be equal")
	}
	if !types.IsError(list.Equal(types.Int(10))) {
		t.Errorf("expected overload error for different type comparison")
	}

	// Add / Lister concatenation
	otherListToAdd := SchemalessTypedToVal([]int{4, 5})
	combinedVal := list.Add(otherListToAdd)
	combined, ok := combinedVal.(*reflectSchemalessTypedList)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedList, got %T", combinedVal)
	}
	if combined.Size() != types.Int(5) {
		t.Errorf("expected combined list size to be 5, got %v", combined.Size())
	}
	var combinedItems []int64
	combinedIt := combined.Iterator()
	for combinedIt.HasNext() == types.True {
		combinedItems = append(combinedItems, combinedIt.Next().Value().(int64))
	}
	if !reflect.DeepEqual(combinedItems, []int64{1, 2, 3, 4, 5}) {
		t.Errorf("expected combined items %v, got %v", []int64{1, 2, 3, 4, 5}, combinedItems)
	}
	if errVal := list.Add(types.Int(10)); !types.IsError(errVal) {
		t.Errorf("expected error for adding non-list type, got %v", errVal)
	}

	// Iterator
	it := list.Iterator()
	var items []int64
	for it.HasNext() == types.True {
		items = append(items, it.Next().Value().(int64))
	}
	if !reflect.DeepEqual(items, []int64{1, 2, 3}) {
		t.Errorf("expected iterated items %v, got %v", []int64{1, 2, 3}, items)
	}
	if errVal := it.Next(); !types.IsError(errVal) {
		t.Errorf("expected error when calling Next on exhausted iterator, got %v", errVal)
	}

	// Type conversions
	convertedNative, err := list.ConvertToNative(reflect.TypeFor[[]int]())
	if err != nil {
		t.Errorf("unexpected error from ConvertToNative: %v", err)
	}
	if !reflect.DeepEqual(convertedNative, val) {
		t.Errorf("expected converted native %v, got %v", val, convertedNative)
	}
	if _, err := list.ConvertToNative(reflect.TypeFor[map[string]string]()); err == nil {
		t.Errorf("expected error from converting list to native map")
	}

	if convertedType := list.ConvertToType(types.ListType); convertedType != list {
		t.Errorf("expected ConvertToType(ListType) to return self")
	}
	if convertedType := list.ConvertToType(types.TypeType); convertedType != types.ListType {
		t.Errorf("expected ConvertToType(TypeType) to return ListType")
	}
	if convertedType := list.ConvertToType(types.MapType); !types.IsError(convertedType) {
		t.Errorf("expected error for converting list to MapType, got %v", convertedType)
	}
}

func TestReflectSchemalessTypedMap(t *testing.T) {
	val := map[string]interface{}{
		"a": "foo",
		"b": 42,
	}
	cellVal := SchemalessTypedToVal(val)
	mapper, ok := cellVal.(*reflectSchemalessTypedMap)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedMap, got %T", cellVal)
	}

	// Type and Value
	if mapper.Type() != types.MapType {
		t.Errorf("expected MapType, got %s", mapper.Type())
	}
	if !reflect.DeepEqual(mapper.Value(), val) {
		t.Errorf("expected underlying map %v, got %v", val, mapper.Value())
	}

	// Size
	if mapper.Size() != types.Int(2) {
		t.Errorf("expected size 2, got %v", mapper.Size())
	}

	// Contains
	if has := mapper.Contains(types.String("a")); has != types.True {
		t.Errorf("expected map to contain 'a'")
	}
	if has := mapper.Contains(types.String("c")); has != types.False {
		t.Errorf("expected map to not contain 'c'")
	}
	if hasErr := mapper.Contains(types.Int(10)); !types.IsError(hasErr) {
		t.Errorf("expected error checking non-string key contains, got %v", hasErr)
	}

	// Find
	v, found := mapper.Find(types.String("a"))
	if !found || v.Value() != "foo" {
		t.Errorf("expected to find 'a' as 'foo', got (%v, %t)", v, found)
	}
	v, found = mapper.Find(types.String("c"))
	if found || v != nil {
		t.Errorf("expected to not find 'c', got (%v, %t)", v, found)
	}
	vErr, found := mapper.Find(types.Int(10))
	if found {
		t.Errorf("expected false found for non-string key in Find")
	}
	if !types.IsError(vErr) {
		t.Errorf("expected error finding non-string key, got %v", vErr)
	}

	// Get
	if item := mapper.Get(types.String("b")); item.Value() != int64(42) {
		t.Errorf("expected key 'b' to be 42, got %v", item.Value())
	}
	if errVal := mapper.Get(types.String("c")); !types.IsError(errVal) {
		t.Errorf("expected error getting non-existent key, got %v", errVal)
	}
	if errVal := mapper.Get(types.Int(10)); !types.IsError(errVal) {
		t.Errorf("expected error getting non-string key, got %v", errVal)
	}

	// Equal
	otherSame := SchemalessTypedToVal(map[string]interface{}{"a": "foo", "b": 42})
	if mapper.Equal(otherSame) != types.True {
		t.Errorf("expected maps to be equal")
	}
	otherDiffVal := SchemalessTypedToVal(map[string]interface{}{"a": "foo", "b": 43})
	if mapper.Equal(otherDiffVal) != types.False {
		t.Errorf("expected maps with different values to not be equal")
	}
	otherDiffLen := SchemalessTypedToVal(map[string]interface{}{"a": "foo"})
	if mapper.Equal(otherDiffLen) != types.False {
		t.Errorf("expected maps with different lengths to not be equal")
	}
	if !types.IsError(mapper.Equal(types.Int(10))) {
		t.Errorf("expected overload error for different type comparison")
	}

	// Iterator
	it := mapper.Iterator()
	keys := make(map[string]bool)
	for it.HasNext() == types.True {
		keys[it.Next().Value().(string)] = true
	}
	expectedKeys := map[string]bool{"a": true, "b": true}
	if !reflect.DeepEqual(keys, expectedKeys) {
		t.Errorf("expected iterated keys %v, got %v", expectedKeys, keys)
	}
	if errVal := it.Next(); !types.IsError(errVal) {
		t.Errorf("expected error when calling Next on exhausted iterator, got %v", errVal)
	}

	// ConvertToNative
	convertedNative, err := mapper.ConvertToNative(reflect.TypeOf(map[string]interface{}{}))
	if err != nil {
		t.Errorf("unexpected error from ConvertToNative: %v", err)
	}
	if !reflect.DeepEqual(convertedNative, val) {
		t.Errorf("expected converted native %v, got %v", val, convertedNative)
	}
	if _, err := mapper.ConvertToNative(reflect.TypeFor[[]int]()); err == nil {
		t.Errorf("expected error from converting map to native slice")
	}

	// ConvertToType
	if convertedType := mapper.ConvertToType(types.MapType); convertedType != mapper {
		t.Errorf("expected ConvertToType(MapType) to return self")
	}
	if convertedType := mapper.ConvertToType(types.TypeType); convertedType != types.MapType {
		t.Errorf("expected ConvertToType(TypeType) to return MapType")
	}
	if convertedType := mapper.ConvertToType(types.ListType); !types.IsError(convertedType) {
		t.Errorf("expected error converting map to ListType, got %v", convertedType)
	}
}

func TestReflectSchemalessTypedMap_NonStringKey(t *testing.T) {
	val := map[int]string{
		1: "foo",
		2: "bar",
	}
	cellVal := SchemalessTypedToVal(val)
	mapper, ok := cellVal.(*reflectSchemalessTypedMap)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedMap, got %T", cellVal)
	}

	// Find with string key
	v, found := mapper.Find(types.String("1"))
	if found || v != nil {
		t.Errorf("expected not found for map with non-string key, got (%v, %t)", v, found)
	}

	// Get with string key
	errVal := mapper.Get(types.String("1"))
	if !types.IsError(errVal) {
		t.Errorf("expected error getting key from map with non-string key, got %v", errVal)
	}
}

func TestReflectSchemalessTypedStruct(t *testing.T) {
	fieldCVal := 100
	val := TestStruct{
		FieldA: "hello",
		FieldB: 42,
		FieldC: &fieldCVal,
	}
	cellVal := SchemalessTypedToVal(val)
	structVal, ok := cellVal.(*reflectSchemalessTypedStruct)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedStruct, got %T", cellVal)
	}

	// Type and Value
	if structVal.Type() != types.MapType {
		t.Errorf("expected MapType for Struct, got %s", structVal.Type())
	}
	if !reflect.DeepEqual(structVal.Value(), val) {
		t.Errorf("expected underlying struct %v, got %v", val, structVal.Value())
	}

	// Size
	if structVal.Size() != types.Int(3) {
		t.Errorf("expected size 3, got %v", structVal.Size())
	}

	// Contains / lookupField / Find
	if has := structVal.Contains(types.String("fieldA")); has != types.True {
		t.Errorf("expected struct to contain 'fieldA'")
	}
	if has := structVal.Contains(types.String("fieldC")); has != types.True {
		t.Errorf("expected struct to contain 'fieldC'")
	}
	if has := structVal.Contains(types.String("nonExistent")); has != types.False {
		t.Errorf("expected struct to not contain 'nonExistent'")
	}

	v, found := structVal.Find(types.String("fieldA"))
	if !found || v.Value() != "hello" {
		t.Errorf("expected 'fieldA' to be found and be 'hello', got (%v, %t)", v, found)
	}
	v, found = structVal.Find(types.Int(100))
	if !found {
		t.Errorf("expected found true for non-string key in Find of struct")
	}
	if !types.IsError(v) {
		t.Errorf("expected error finding with non-string field key, got %v", v)
	}

	// Get
	if item := structVal.Get(types.String("fieldB")); item.Value() != int64(42) {
		t.Errorf("expected fieldB to be 42, got %v", item.Value())
	}
	if errVal := structVal.Get(types.String("nonExistent")); !types.IsError(errVal) {
		t.Errorf("expected error getting non-existent field, got %v", errVal)
	}

	// IsSet
	if isSet := structVal.IsSet(types.String("fieldA")); isSet != types.True {
		t.Errorf("expected fieldA to be set")
	}
	if isSet := structVal.IsSet(types.String("nonExistent")); isSet != types.False {
		t.Errorf("expected nonExistent to be unset")
	}
	if isSet := structVal.IsSet(types.Int(100)); !types.IsError(isSet) {
		t.Errorf("expected error check for non-string key in IsSet, got %v", isSet)
	}

	// Equal
	otherSame := SchemalessTypedToVal(TestStruct{
		FieldA: "hello",
		FieldB: 42,
		FieldC: &fieldCVal,
	})
	if structVal.Equal(otherSame) != types.True {
		t.Errorf("expected structs to be equal")
	}
	otherDiff := SchemalessTypedToVal(TestStruct{
		FieldA: "diff",
		FieldB: 42,
		FieldC: &fieldCVal,
	})
	if structVal.Equal(otherDiff) != types.False {
		t.Errorf("expected different structs to not be equal")
	}
	if !types.IsError(structVal.Equal(types.Int(10))) {
		t.Errorf("expected overload error for struct comparison with int")
	}

	// Equal with different size mapper
	diffSizeMap := SchemalessTypedToVal(map[string]interface{}{
		"fieldA": "hello",
	})
	if structVal.Equal(diffSizeMap) != types.False {
		t.Errorf("expected False when comparing struct to map of different size")
	}

	// Equal with same size mapper but different value
	diffValMap := SchemalessTypedToVal(map[string]interface{}{
		"fieldA": "hello",
		"fieldB": 43,
		"fieldC": 100,
	})
	if structVal.Equal(diffValMap) != types.False {
		t.Errorf("expected False when comparing struct to map of same size but different value")
	}

	// Iterator
	it := structVal.Iterator()
	keys := make(map[string]bool)
	for it.HasNext() == types.True {
		keys[it.Next().Value().(string)] = true
	}
	expectedKeys := map[string]bool{"fieldA": true, "fieldB": true, "fieldC": true}
	if !reflect.DeepEqual(keys, expectedKeys) {
		t.Errorf("expected iterated keys %v, got %v", expectedKeys, keys)
	}
	if errVal := it.Next(); !types.IsError(errVal) {
		t.Errorf("expected error when calling Next on exhausted iterator, got %v", errVal)
	}

	// ConvertToNative
	convertedNative, err := structVal.ConvertToNative(reflect.TypeFor[TestStruct]())
	if err != nil {
		t.Errorf("unexpected error from ConvertToNative struct: %v", err)
	}
	if !reflect.DeepEqual(convertedNative, val) {
		t.Errorf("expected converted native struct %v, got %v", val, convertedNative)
	}
	if _, err := structVal.ConvertToNative(reflect.TypeFor[map[string]string]()); err == nil {
		t.Errorf("expected error converting struct to native map")
	}

	// ConvertToType
	if convertedType := structVal.ConvertToType(types.MapType); convertedType != structVal {
		t.Errorf("expected ConvertToType(MapType) to return self")
	}
	if convertedType := structVal.ConvertToType(types.TypeType); convertedType != types.MapType {
		t.Errorf("expected ConvertToType(TypeType) to return MapType")
	}
	if convertedType := structVal.ConvertToType(types.ListType); !types.IsError(convertedType) {
		t.Errorf("expected error converting struct to ListType, got %v", convertedType)
	}
}

func TestSchemalessTypedToVal_Equivalence(t *testing.T) {
	pod := &corev1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "default",
			Labels: map[string]string{
				"app": "test",
			},
			CreationTimestamp: metav1.Time{Time: time.Date(2026, 5, 18, 12, 0, 0, 0, time.UTC)},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:  "test-container",
					Image: "nginx",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				},
			},
		},
	}

	reflectVal := SchemalessTypedToVal(pod)

	unstrMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(pod)
	if err != nil {
		t.Fatalf("failed to convert to unstructured: %v", err)
	}
	nativeVal := types.DefaultTypeAdapter.NativeToValue(unstrMap)

	// They must evaluate to exact deep equality via CEL
	if reflectVal.Equal(nativeVal) != types.True {
		t.Errorf("SchemalessTypedToVal does not semantically equal K8s ToUnstructured!")
	}
}

func TestReflectSchemalessTypedStruct_OmitAndNil(t *testing.T) {
	val := OmitTestStruct{
		FieldNoOmit: nil,
		FieldOmit:   nil,
	}
	cellVal := SchemalessTypedToVal(val)
	structVal, ok := cellVal.(*reflectSchemalessTypedStruct)
	if !ok {
		t.Fatalf("expected *reflectSchemalessTypedStruct, got %T", cellVal)
	}

	// FieldNoOmit should be set and contained, and its value should be NullValue.
	if isSet := structVal.IsSet(types.String("fieldNoOmit")); isSet != types.True {
		t.Errorf("expected fieldNoOmit to be set")
	}
	if has := structVal.Contains(types.String("fieldNoOmit")); has != types.True {
		t.Errorf("expected fieldNoOmit to be contained")
	}
	if v, found := structVal.Find(types.String("fieldNoOmit")); !found || v != types.NullValue {
		t.Errorf("expected fieldNoOmit to be found and equal to NullValue, got (%v, %t)", v, found)
	}
	if val := structVal.Get(types.String("fieldNoOmit")); val != types.NullValue {
		t.Errorf("expected fieldNoOmit value to be NullValue, got %v", val)
	}

	// FieldOmit is nil and omitempty is present, so it can be omitted.
	if isSet := structVal.IsSet(types.String("fieldOmit")); isSet != types.False {
		t.Errorf("expected fieldOmit to be unset")
	}
	if has := structVal.Contains(types.String("fieldOmit")); has != types.False {
		t.Errorf("expected fieldOmit to not be contained")
	}
	if v, found := structVal.Find(types.String("fieldOmit")); found || v != nil {
		t.Errorf("expected fieldOmit not to be found")
	}
	if errVal := structVal.Get(types.String("fieldOmit")); !types.IsError(errVal) {
		t.Errorf("expected error getting fieldOmit, got %v", errVal)
	}
}
