/*
Copyright 2025 The Kubernetes Authors.

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

package common_test

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apiserver/pkg/cel/common"
	"k8s.io/apiserver/pkg/cel/library"
	"k8s.io/apiserver/pkg/cel/openapi"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"testing"
	"time"
)

// TestToValue tests that both UnstructuredToValue and TypedToValue correctly
// convert Kubernetes typed objects and unstructured objects to CEL values.
//
// TestToValue also tests that UnstructuredToValue and TypedToValue behave identically
// for a Kubernetes object.
func TestToValue(t *testing.T) {
	struct1value := Struct{S: "hello", I: 10, B: true, F: 1.5}
	struct1 := typedValue{value: struct1value, schema: structSchema}
	structOmitEmpty1value := StructOmitEmpty{}
	structOmitEmpty1 := typedValue{value: structOmitEmpty1value, schema: structSchema}
	structOmitZero1value := StructOmitZero{}
	structOmitZero1 := typedValue{value: structOmitZero1value, schema: structSchema}
	struct1Ptr := typedValue{value: &struct1value, schema: structSchema}
	struct2value := Struct{S: "world", I: 20, B: false, F: 2.5}
	struct2 := typedValue{value: struct2value, schema: structSchema}
	struct1Again := typedValue{value: struct1value, schema: structSchema}
	zeroStruct := typedValue{value: Struct{}, schema: structSchema}
	zeroStructPtr := typedValue{value: Struct{}, schema: structSchema}

	now := metav1.NewTime(time.Now().Truncate(0))
	duration1 := metav1.Duration{Duration: 5 * time.Second}
	time1Parsed, err := time.Parse(RFC3339, "2000-01-01T12:00:00Z")
	if err != nil {
		t.Fatal(err)
	}
	time1 := metav1.Time{Time: time1Parsed}
	microTime1Parsed, err := time.Parse(RFC3339Micro, "2000-01-01T12:00:00.000001Z")
	if err != nil {
		t.Fatal(err)
	}
	microTime1 := metav1.MicroTime{Time: microTime1Parsed}

	nested1value := Nested{Name: "nested1", Info: Struct{S: "hello", I: 10, B: true, F: 1.5}}

	complex1value := Complex{
		TypeMeta:    metav1.TypeMeta{Kind: "Complex", APIVersion: "v1"},
		ObjectMeta:  metav1.ObjectMeta{Name: "complex1"},
		ID:          "c1",
		Tags:        []string{"a", "b", "c"},
		Labels:      map[string]string{"key1": "val1", "key2": "val2"},
		NestedObj:   nested1value,
		NestedEmpty: Nested{},
		NestedZero:  Nested{},
		Timeout:     duration1,
		Time:        time1,
		MicroTime:   microTime1,
		RawBytes:    []byte("bytes1"),
		NilBytes:    nil,
		ChildPtr:    &struct2value,
		NilPtr:      nil,
		EmptySlice:  []int{},
		NilSlice:    nil,
		EmptyMap:    map[string]int{},
		NilMap:      nil,
		IntOrString: intstr.FromInt32(5),
		Quantity:    resource.MustParse("100m"),
		I32:         int32(32),
		I64:         int64(64),
		F32:         float32(32.5),
		Enum:        EnumTypeA,
		MapList: []MapListEntry{
			{
				Key1:  "k1v1",
				Key2:  "k2v1",
				Value: 1,
			},
			{
				Key1:  "k1v2",
				Key2:  "k2v2",
				Value: 2,
			},
		},
		SetList: []SetEntry{1, 2, 3},
	}
	complex1 := typedValue{value: complex1value, schema: complexSchema}

	complex2value := Complex{
		TypeMeta:    metav1.TypeMeta{Kind: "Complex2", APIVersion: "v1"},
		ObjectMeta:  metav1.ObjectMeta{Name: "complex2"},
		ID:          "c2",
		Tags:        []string{"x", "y"},
		Labels:      map[string]string{"key3": "val3"},
		NestedObj:   Nested{Name: "nested2", Info: struct2value},
		NestedEmpty: Nested{Name: "nested3"},
		NestedZero:  Nested{Name: "nested4"},
		Timeout:     metav1.Duration{Duration: 10 * time.Second},
		RawBytes:    []byte("bytes2"),
		NilBytes:    []byte{}, // Non-nil but empty
		ChildPtr:    &struct1value,
		NilPtr:      nil,
		EmptySlice:  []int{1},               // Non-empty
		NilSlice:    []int{1},               // Non-nil
		EmptyMap:    map[string]int{"a": 1}, // Non-empty
		NilMap:      map[string]int{"a": 1}, // Non-nil
		IntOrString: intstr.FromString("port"),
		Quantity:    resource.MustParse("200m"),
		I32:         int32(42),
		I64:         int64(200),
		F32:         float32(42.5),
		Enum:        EnumTypeB,
		MapList: []MapListEntry{
			{
				Key1:  "k1v2",
				Key2:  "k2v2",
				Value: 2,
			},
			{
				Key1:  "k1v1",
				Key2:  "k2v1",
				Value: 1,
			},
		},
		SetList: []SetEntry{3, 2, 1},
	}
	complex2 := typedValue{value: complex2value, schema: complexSchema}

	complex3value := Complex{
		MapList: []MapListEntry{
			{
				Key1:  "k1v3",
				Key2:  "k2v3",
				Value: 3,
			},
			{
				Key1:  "k1v1",
				Key2:  "k2v1",
				Value: 1,
			},
		},
		SetList: []SetEntry{4, 1},
	}
	complex3 := typedValue{value: complex3value, schema: complexSchema}

	complex4value := Complex{
		MapList: []MapListEntry{
			{
				Key1:  "k1v3",
				Key2:  "k2v3",
				Value: 3,
			},
			{
				Key1:  "k1v2",
				Key2:  "k2v2",
				Value: 2,
			},
			{
				Key1:  "k1v1",
				Key2:  "k2v1",
				Value: 1,
			},
		},
		SetList: []SetEntry{4, 3, 2, 1},
	}
	complex4 := typedValue{value: complex4value, schema: complexSchema}

	complex1Again := complex1 // Create a copy for equality checks

	slice1 := []int{1, 2, 3}
	slice1Again := []int{1, 2, 3}
	slice2 := []int{1, 2, 4}
	slice3 := []string{"a", "b"}

	map1 := map[string]int{"a": 1, "b": 2}
	map1Again := map[string]int{"b": 2, "a": 1}
	map2 := map[string]int{"a": 1, "b": 3}        // Different value
	map3 := map[string]int{"a": 1, "c": 2}        // Different key
	map4 := map[string]string{"a": "1", "b": "2"} // Different value type

	tests := []testCase{
		// Basic Type Conversions
		{
			name:       "basic: int32",
			expression: "c.i32 == 32",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "basic: int64",
			expression: "c.i64 == 64",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "basic: float32",
			expression: "c.f32 == 32.5",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "basic: enum",
			expression: "c.enum == 'a'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "basic: nil bytes",
			expression: "!has(c.nilBytes)",
			activation: map[string]typedValue{"c": complex1},
		},

		// Struct Tests
		{
			name:       "struct: zero value struct",
			expression: "obj.s == '' && obj.i == 0 && obj.b == false && obj.f == 0.0",
			activation: map[string]typedValue{"obj": zeroStruct},
		},
		{
			name:       "struct: zero value struct pointer",
			expression: "obj.s == '' && obj.i == 0 && obj.b == false && obj.f == 0.0",
			activation: map[string]typedValue{"obj": zeroStructPtr},
		},
		{
			name:       "struct: populated struct field access",
			expression: "obj.s == 'hello' && obj.i == 10 && obj.b == true && obj.f == 1.5",
			activation: map[string]typedValue{"obj": struct1},
		},
		{
			name:       "struct: populated struct pointer field access",
			expression: "obj.s == 'hello' && obj.i == 10 && obj.b == true && obj.f == 1.5",
			activation: map[string]typedValue{"obj": struct1Ptr},
		},
		{
			name:       "struct: access omitempty field (has)",
			expression: "!has(obj.so)",
			activation: map[string]typedValue{"obj": struct1},
		},
		{
			name:       "struct: access non-existent field (has)",
			expression: "!has(obj.nonExistent)",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "struct: access non-existent field direct (error)",
			expression: "obj.nonExistent",
			activation: map[string]typedValue{"obj": struct2},
			wantErr:    "no such key: nonExistent",
		},
		{
			name:       "struct: access with non-string key (get) (error)",
			expression: "obj[1]",
			activation: map[string]typedValue{"obj": struct2},
			wantErr:    "no such overload",
		},
		{
			name:       "struct: check contains non-string key (error)",
			expression: "1 in obj",
			activation: map[string]typedValue{"obj": struct2},
			wantErr:    "no such overload",
		},
		{
			name:       "struct: convert to its own type",
			expression: "type(obj) == type(obj)",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "struct: embedded inline",
			expression: "c.apiVersion == 'v1' && c.kind == 'Complex'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "struct: embedded inline: omitempty",
			expression: "!has(c.apiVersion)",
			activation: map[string]typedValue{"c": struct2},
		},
		{
			name:       "struct: embedded struct",
			expression: "c.metadata.name == 'complex1'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "struct: omitempty: zero valued: has struct field",
			expression: "has(c.nestedEmpty)",
			activation: map[string]typedValue{"c": complex2},
		},
		{
			name:       "struct: omitempty: zero valued: does not have scalar fields",
			expression: "!has(c.s) && !has(c.i) && !has(c.b) && !has(c.f)",
			activation: map[string]typedValue{"c": structOmitEmpty1},
		},
		{
			name:       "struct: omitempty: zero valued: does not have pointer to scalar fields",
			expression: "!has(c.sp) && !has(c.ip) && !has(c.bp) && !has(c.fp)",
			activation: map[string]typedValue{"c": structOmitEmpty1},
		},
		{
			name:       "struct: omitzero: zero valued: does not have struct field",
			expression: "!has(c.nestedZero)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "struct: omitzero: zero valued: does not have scalar fields",
			expression: "!has(c.s) && !has(c.i) && !has(c.b) && !has(c.f)",
			activation: map[string]typedValue{"c": structOmitZero1},
		},
		{
			name:       "struct: omitzero: zero valued: does not have pointer to scalar fields",
			expression: "!has(c.sp) && !has(c.ip) && !has(c.bp) && !has(c.fp)",
			activation: map[string]typedValue{"c": structOmitZero1},
		},
		{
			name:       "struct: omitzero: non-zero valued: has struct field",
			expression: "has(c.nestedZero)",
			activation: map[string]typedValue{"c": complex2},
		},
		{
			name:       "struct: omitempty: zero valued: has embedded struct field",
			expression: "has(c.metadata)",
			activation: map[string]typedValue{"c": structOmitEmpty1},
		},
		{
			name:       "struct: omitzero: zero valued: does not have embedded struct field",
			expression: "!has(c.metadata)",
			activation: map[string]typedValue{"c": structOmitZero1},
		},

		// Comparison Tests
		{
			name:       "compare: identity (struct)",
			expression: "s1 == s1",
			activation: map[string]typedValue{"s1": struct1},
		},
		{
			name:       "compare: identical structs",
			expression: "s1 == s1_again",
			activation: map[string]typedValue{
				"s1":       struct1,
				"s1_again": struct1Again,
			},
		},
		{
			name:       "compare: different structs",
			expression: "s1 != s2",
			activation: map[string]typedValue{
				"s1": struct1,
				"s2": struct2,
			},
		},
		{
			name:       "compare: identical complex structs",
			expression: "c1 == c2",
			activation: map[string]typedValue{"c1": complex1, "c2": complex1Again},
		},
		{
			name:       "compare: different complex structs",
			expression: "c1 != c2",
			activation: map[string]typedValue{"c1": complex1, "c2": complex2},
		},
		{
			name:       "compare: struct and pointer to identical struct",
			expression: "s1 == s1_ptr",
			activation: map[string]typedValue{
				"s1":     struct1,
				"s1_ptr": struct1Ptr,
			},
		},
		{
			name:       "compare: struct and nil",
			expression: "s1 != null",
			activation: map[string]typedValue{"s1": struct1},
		},
		{
			name:       "compare: struct and different type",
			expression: "s1 != 10",
			activation: map[string]typedValue{"s1": struct1},
		},
		{
			name:       "compare: nil struct pointer and null",
			expression: "nil_obj == null",
			activation: map[string]typedValue{"nil_obj": {value: nil, schema: structSchema}},
		},
		{
			name:       "compare: identical slices (activation)",
			expression: "sl1 == sl1a",
			activation: map[string]typedValue{
				"sl1":  {value: slice1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
				"sl1a": {value: slice1Again, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: different slices (activation)",
			expression: "sl1 != sl2",
			activation: map[string]typedValue{
				"sl1": {value: slice1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
				"sl2": {value: slice2, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: slices of different types",
			expression: "sl1 != sl3",
			activation: map[string]typedValue{
				"sl1": {value: slice1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
				"sl3": {value: slice3, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: stringSchema}}}},
			},
		},
		{
			name:       "compare: slice and non-list",
			expression: "sl1 != 1",
			activation: map[string]typedValue{
				"sl1": {value: slice1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: identical maps (activation)",
			expression: "m1 == m1a",
			activation: map[string]typedValue{
				"m1":  {value: map1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
				"m1a": {value: map1Again, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: different maps (value) (activation)",
			expression: "m1 != m2",
			activation: map[string]typedValue{
				"m1": {value: map1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
				"m2": {value: map2, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: different maps (key) (activation)",
			expression: "m1 != m3",
			activation: map[string]typedValue{
				"m1": {value: map1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
				"m3": {value: map3, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: different maps (value type)",
			expression: "m1 != m4",
			activation: map[string]typedValue{
				"m1": {value: map1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
				"m4": {value: map4, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: stringSchema}}}},
			},
		},
		{
			name:       "compare: map and non-map",
			expression: "m1 != 1",
			activation: map[string]typedValue{
				"m1": {value: map1, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
			},
		},
		{
			name:       "compare: time instances (equal)",
			expression: "t1 == t2",
			activation: map[string]typedValue{
				"t1": {value: now, schema: &timeFormat},
				"t2": {value: now, schema: &timeFormat},
			},
		},
		{
			name:       "compare: time instances (different)",
			expression: "t1 != t2",
			activation: map[string]typedValue{
				"t1": {value: now, schema: &timeFormat},
				"t2": {value: metav1.MicroTime{Time: now.Add(time.Nanosecond)}, schema: stringSchema},
			},
		},
		{
			name:       "compare: microTime instances (equal)",
			expression: "t1 == t2",
			activation: map[string]typedValue{
				"t1": {value: now, schema: stringSchema},
				"t2": {value: now, schema: stringSchema},
			},
		},
		{
			name:       "compare: microTime instances (different)",
			expression: "t1 != t2",
			activation: map[string]typedValue{
				"t1": {value: now, schema: stringSchema},
				"t2": {value: metav1.MicroTime{Time: now.Add(time.Nanosecond)}, schema: stringSchema},
			},
		},
		{
			name:       "compare: duration instances (equal)",
			expression: "d1 == d2",
			activation: map[string]typedValue{
				"d1": {value: duration1, schema: stringSchema},
				"d2": {value: metav1.Duration{Duration: 5 * time.Second}, schema: stringSchema},
			},
		},
		{
			name:       "compare: duration instances (different)",
			expression: "d1 != d2",
			activation: map[string]typedValue{
				"d1": {value: duration1, schema: stringSchema},
				"d2": {value: metav1.Duration{Duration: 6 * time.Second}, schema: stringSchema},
			},
		},
		{
			name:       "compare: bytes instances (equal)",
			expression: "b1 == b2",
			activation: map[string]typedValue{
				"b1": {value: []byte("abc"), schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "byte"}}},
				"b2": {value: []byte("abc"), schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "byte"}}},
			},
		},
		{
			name:       "compare: bytes instances (different)",
			expression: "b1 != b2",
			activation: map[string]typedValue{
				"b1": {value: []byte("abc"), schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "byte"}}},
				"b2": {value: []byte("abd"), schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "byte"}}},
			},
		},
		{
			name:       "compare: empty slices (different underlying types)",
			expression: "e1 == e2",
			activation: map[string]typedValue{
				"e1": {value: []int{}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: int64Schema}}}},
				"e2": {value: []string{}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: stringSchema}}}},
			},
		},
		{
			name:       "compare: empty maps (different underlying types)",
			expression: "m1 == m2",
			activation: map[string]typedValue{
				"m1": {value: map[string]int{}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: int64Schema}}}},
				"m2": {value: map[string]bool{}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"object"}, AdditionalProperties: &spec.SchemaOrBool{Schema: boolSchema}}}},
			},
		},

		// Nested Struct Tests
		{
			name:       "nested: access field",
			expression: "c.nestedObj.info.s == 'hello'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "nested: compare nested struct",
			expression: "c1.nestedObj != c2.nestedObj",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "nested: compare identical nested struct",
			expression: "c1.nestedObj == c1_again.nestedObj",
			activation: map[string]typedValue{
				"c1":       complex1,
				"c1_again": complex1Again,
			},
		},

		// Slice Tests
		{
			name:       "slice: access element",
			expression: "c.tags[1] == 'b'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: size",
			expression: "size(c.tags) == 3",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: contains ('in')",
			expression: "'b' in c.tags",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: not contains ('in')",
			expression: "!('d' in c.tags)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: contains with non-primitive (struct)",
			expression: "s1 in structs",
			activation: map[string]typedValue{
				"structs": {value: []Struct{struct2value, struct1value}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: structSchema}}}},
				"s1":      struct1,
			},
		},
		{
			name:       "slice: contains with non-primitive (struct ptr)",
			expression: "s1 in structs",
			activation: map[string]typedValue{
				"structs": {value: []*Struct{&struct2value, &struct1value}, schema: &spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{Schema: structSchema}}}},
				"s1":      struct1,
			},
		},
		{
			name:       "slice: add",
			expression: "size(c1.tags + c2.tags) == 5 && (c1.tags + c2.tags)[3] == 'x'",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "slice: add non-list (error)",
			expression: "c.tags + 1",
			activation: map[string]typedValue{
				"c": complex1,
			},
			wantErr: "no such overload",
		},
		{
			name:       "slice: get with non-int index (error)",
			expression: `c.tags['a']`,
			activation: map[string]typedValue{
				"c": complex1,
			},
			wantErr: `unsupported index type 'string' in list`,
		},
		{
			name:       "slice: all() true",
			expression: "c.tags.all(t, t.startsWith(''))",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: all() false",
			expression: "!c.tags.all(t, t == 'a')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: exists() true",
			expression: "c.tags.exists(t, t == 'c')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: exists() false",
			expression: "!c.tags.exists(t, t == 'z')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: out of bounds access",
			expression: "c.tags[5]",
			activation: map[string]typedValue{"c": complex1},
			wantErr:    "index out of bounds: 5",
		},
		{
			name:       "slice: empty slice size",
			expression: "size(c.emptySlice) == 0",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: exists() on empty",
			expression: "!c.emptySlice.exists(x, true)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: all() on empty",
			expression: "c.emptySlice.all(x, false)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: convert to list type",
			expression: "type(c.tags) == list",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "slice: convert list to type type",
			expression: "type(c.tags) == list",
			activation: map[string]typedValue{"c": complex1},
		},

		// Map Tests
		{
			name:       "map: access element",
			expression: "c.labels['key1'] == 'val1'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: size",
			expression: "size(c.labels) == 2",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: contains key ('in')",
			expression: "'key1' in c.labels",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: not contains key ('in')",
			expression: "!('key3' in c.labels)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: has() key",
			expression: "has(c.labels.key1)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: has() non-existent key",
			expression: "!has(c.labels.key3)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: access non-existent key (error)",
			expression: "c.labels['key3']",
			activation: map[string]typedValue{"c": complex1},
			wantErr:    "no such key: key3",
		},
		{
			name:       "map: all() on keys true",
			expression: "c.labels.all(name, name.startsWith('key'))",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: all() on keys false",
			expression: "!c.labels.all(name, name == 'key1')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: exists() on keys true",
			expression: "c.labels.exists(name, name == 'key2')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: exists() on keys false",
			expression: "!c.labels.exists(name, name == 'key3')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: empty map size",
			expression: "size(c.emptyMap) == 0",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: exists() on empty",
			expression: "!c.emptyMap.exists(name, true)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: all() on empty",
			expression: "c.emptyMap.all(name, false)",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: convert to map type",
			expression: "type(c.labels) == map",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "map: convert map to type type",
			expression: "type(c.labels) == map",
			activation: map[string]typedValue{"c": complex1},
		},

		// Pointer Tests
		{
			name:       "pointer: access through non-nil pointer field",
			expression: "c.childPtr.s == 'world'",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "pointer: compare non-nil pointer field",
			expression: "c.childPtr == s2",
			activation: map[string]typedValue{
				"c":  complex1,
				"s2": struct2,
			},
		},
		{
			name:       "pointer: access through nil pointer field (error)",
			expression: "c.nilPtr.s",
			activation: map[string]typedValue{"c": complex1},
			wantErr:    "no such key: nilPtr", // Accessing field 's' on a null object
		},
		{
			name:       "pointer: check has() nil pointer",
			expression: "!has(c.nilPtr)",
			activation: map[string]typedValue{"c": complex1},
		},

		// Type Tests
		{
			name:       "type: string",
			expression: "type(obj.s) == string",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "type: int",
			expression: "type(obj.i) == int",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "type: bool",
			expression: "type(obj.b) == bool",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "type: float",
			expression: "type(obj.f) == double",
			activation: map[string]typedValue{"obj": struct2},
		},
		{
			name:       "type: slice",
			expression: "type(c.tags) == list",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: map",
			expression: "type(c.labels) == map",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: time",
			expression: "type(c.time) == google.protobuf.Timestamp",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: microTime",
			expression: "type(c.microTime) == google.protobuf.Timestamp",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: duration",
			expression: "type(c.timeout) == google.protobuf.Duration",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: bytes",
			expression: "type(c.rawBytes) == bytes",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: int32",
			expression: "type(c.i32) == int",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: int64",
			expression: "type(c.i64) == int",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: float32",
			expression: "type(c.f32) == double",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "type: enum",
			expression: "type(c.enum) == string",
			activation: map[string]typedValue{"c": complex1},
		},

		// listType=map
		{
			name:       "listType=map: equal",
			expression: "c1.mapList == c2.mapList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "listType=map: not equal",
			expression: "c1.mapList != c3.mapList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c3": complex3,
			},
		},
		{
			name:       "listType=map: add overlapping",
			expression: "c1.mapList + c2.mapList == c1.mapList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "listType=map: add non-overlapping",
			expression: "c1.mapList + c3.mapList == c4.mapList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c3": complex3,
				"c4": complex4,
			},
		},

		// listType=set
		{
			name:       "listType=set: equal",
			expression: "c1.setList == c2.setList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "listType=set: not equal",
			expression: "c1.setList != c3.setList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c3": complex3,
			},
		},
		{
			name:       "listType=set: add overlapping",
			expression: "c1.setList + c2.setList == c1.setList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c2": complex2,
			},
		},
		{
			name:       "listType=set: add non-overlapping",
			expression: "c1.setList + c3.setList == c4.setList",
			activation: map[string]typedValue{
				"c1": complex1,
				"c3": complex3,
				"c4": complex4,
			},
		},

		// Special K8s Types
		{
			name:       "time: comparison equals",
			expression: "c.time == timestamp('2000-01-01T12:00:00Z')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "microTime: comparison equals",
			expression: "c.microTime == timestamp('2000-01-01T12:00:00.000001Z')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "duration: comparison equals",
			expression: "c.timeout == duration('5s')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "duration: comparison greater",
			expression: "c.timeout > duration('1s')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "intOrString: int comparison",
			expression: "c.intOrString == 5",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "intOrString: string comparison",
			expression: "c.intOrString == 'port'",
			activation: map[string]typedValue{"c": complex2},
		},
		{
			name:       "quantity: comparison",
			expression: "quantity(c.quantity).isGreaterThan(quantity('99m')) && quantity(c.quantity).isLessThan(quantity('101m'))",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "quantity: equality",
			expression: "quantity(c.quantity) == quantity('100m')",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "bytes: size",
			expression: "size(c.rawBytes) == 6",
			activation: map[string]typedValue{"c": complex1},
		},
		{
			name:       "bytes: equality",
			expression: "c.rawBytes == b'bytes1'",
			activation: map[string]typedValue{"c": complex1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var opts []cel.EnvOption
			for k := range tt.activation {
				opts = append(opts, cel.Variable(k, cel.DynType))
			}
			opts = append(opts, cel.StdLib(), library.Quantity())

			env, err := cel.NewEnv(opts...)
			if err != nil {
				t.Fatalf("Env creation error: %v", err)
			}

			t.Run("TypedToVal", func(t *testing.T) {
				testTypedToVal(t, env, tt)
			})

			t.Run("UnstructuredToVal", func(t *testing.T) {
				testUnstructuredToVal(t, tt, env)
			})
		})
	}
}

func testTypedToVal(t *testing.T, env *cel.Env, tt testCase) {
	typedOut, typedErr := evalExpression(t, env, tt.expression, typedToValActivation(tt.activation))
	if typedErr != nil && len(tt.wantErr) == 0 {
		t.Fatalf("Unexpected err with typed values: %v", typedErr)
	}
	if len(tt.wantErr) > 0 {
		if typedErr == nil {
			t.Fatalf("Expected error '%s' during evaluation with typed values, but got none", tt.wantErr)
		}
		if typedErr.Error() != tt.wantErr {
			t.Fatalf("Expected error '%s' during evaluation with typed values, but got: %v", tt.wantErr, typedErr)
		}
	}
	if len(tt.wantErr) == 0 && typedOut != types.True {
		t.Errorf("Expected true with typed values but got %v", typedOut)
	}
}

func testUnstructuredToVal(t *testing.T, tt testCase, env *cel.Env) {
	a, err := unstructuredToValActivation(tt.activation)
	if err != nil {
		t.Fatalf("Unexpected error converting activation to unstructured: %v", err)
	}
	unstructuredOut, unstructuredErr := evalExpression(t, env, tt.expression, a)
	if unstructuredErr != nil && len(tt.wantErr) == 0 {
		t.Fatalf("Unexpected err with unstructured values: %v", unstructuredErr)
	}
	if len(tt.wantErr) > 0 {
		if unstructuredErr == nil {
			t.Fatalf("Expected error '%s' during evaluation with unstructured values, but got none", tt.wantErr)
		}
		if unstructuredErr.Error() != tt.wantErr {
			t.Fatalf("Expected error '%s' during evaluation with unstructured values, but got: %v", tt.wantErr, unstructuredErr)
		}
	}
	if len(tt.wantErr) == 0 && unstructuredOut != types.True {
		t.Errorf("Expected true with unstructured values but got %v", unstructuredOut)
	}
}

func evalExpression(t *testing.T, env *cel.Env, expression string, activation map[string]interface{}) (ref.Val, error) {
	ast, iss := env.Compile(expression)
	if iss.Err() != nil {
		t.Fatalf("Compile error: %v :: %s", iss.Err(), expression)
	}

	prg, err := env.Program(ast)
	if err != nil {
		t.Fatalf("Program error: %v :: %s", err, expression)
	}

	out, _, err := prg.Eval(activation)
	return out, err
}

type Struct struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata"`

	S string  `json:"s"`
	I int     `json:"i"`
	B bool    `json:"b"`
	F float64 `json:"f"`

	SP string  `json:"sp"`
	IP int     `json:"ip"`
	BP bool    `json:"bp"`
	FP float64 `json:"fp"`
}

type StructOmitEmpty struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	S string  `json:"s,omitempty"`
	I int     `json:"i,omitempty"`
	B bool    `json:"b,omitempty"`
	F float64 `json:"f,omitempty"`

	SP string  `json:"sp,omitempty"`
	IP int     `json:"ip,omitempty"`
	BP bool    `json:"bp,omitempty"`
	FP float64 `json:"fp,omitempty"`
}

type StructOmitZero struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitzero"`

	S string  `json:"s,omitzero"`
	I int     `json:"i,omitzero"`
	B bool    `json:"b,omitzero"`
	F float64 `json:"f,omitzero"`

	SP string  `json:"sp,omitzero"`
	IP int     `json:"ip,omitzero"`
	BP bool    `json:"bp,omitzero"`
	FP float64 `json:"fp,omitzero"`
}

var structSchema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type: []string{"object"},
		Properties: map[string]spec.Schema{
			"kind":       *stringSchema,
			"apiVersion": *stringSchema,
			"metadata": {
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					Properties: map[string]spec.Schema{
						"name": *stringSchema,
					},
				},
			},
			"s":  *stringSchema,
			"i":  *int64Schema,
			"b":  *boolSchema,
			"f":  *float64Schema,
			"so": *stringSchema,
			"io": *int64Schema,
			"bo": *boolSchema,
			"fo": *float64Schema,
		},
	},
}

func (s Struct) GetObjectKind() schema.ObjectKind {
	panic("not implemented")
}

func (s Struct) DeepCopyObject() runtime.Object {
	panic("not implemented")
}

type Nested struct {
	Name string `json:"name"`
	Info Struct `json:"info"`
}

var nestedSchema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type: []string{"object"},
		Properties: map[string]spec.Schema{
			"name": *stringSchema,
			"info": *structSchema,
		},
	},
}

func (s Nested) GetObjectKind() schema.ObjectKind {
	panic("not implemented")
}

func (s Nested) DeepCopyObject() runtime.Object {
	panic("not implemented")
}

type Complex struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	ID          string             `json:"id"`
	Tags        []string           `json:"tags"`
	Labels      map[string]string  `json:"labels"`
	NestedObj   Nested             `json:"nestedObj"`
	NestedEmpty Nested             `json:"nestedEmpty,omitempty"`
	NestedZero  Nested             `json:"nestedZero,omitzero"`
	Timeout     metav1.Duration    `json:"timeout"`
	Time        metav1.Time        `json:"time"`
	MicroTime   metav1.MicroTime   `json:"microTime"`
	RawBytes    []byte             `json:"rawBytes"`
	NilBytes    []byte             `json:"nilBytes"` // Always nil
	ChildPtr    *Struct            `json:"childPtr"`
	NilPtr      *Struct            `json:"nilPtr"` // Always nil
	EmptySlice  []int              `json:"emptySlice"`
	NilSlice    []int              `json:"nilSlice"` // Always nil
	EmptyMap    map[string]int     `json:"emptyMap"`
	NilMap      map[string]int     `json:"nilMap"` // Always nil
	IntOrString intstr.IntOrString `json:"intOrString"`
	Quantity    resource.Quantity  `json:"quantity"`
	I32         int32              `json:"i32"`
	I64         int64              `json:"i64"`
	F32         float32            `json:"f32"`
	Enum        EnumType           `json:"enum"`
	MapList     []MapListEntry     `json:"mapList"`
	SetList     []SetEntry         `json:"setList"`
}

type SetEntry int

type MapListEntry struct {
	Key1  string `json:"key1"`
	Key2  string `json:"key2"`
	Value int    `json:"value"`
}

var complexSchema = &spec.Schema{
	SchemaProps: spec.SchemaProps{
		Type: []string{"object"},
		Properties: map[string]spec.Schema{
			"kind":       *stringSchema,
			"apiVersion": *stringSchema,
			"metadata": {
				SchemaProps: spec.SchemaProps{
					Type: []string{"object"},
					Properties: map[string]spec.Schema{
						"name": *stringSchema,
					},
				},
			},
			"id":          *stringSchema,
			"tags":        *stringArraySchema,
			"labels":      *stringMapSchema,
			"nestedObj":   *nestedSchema,
			"nestedEmpty": *nestedSchema,
			"nestedZero":  *nestedSchema,
			"timeout":     durationFormat,
			"time":        timeFormat,
			"microTime":   timeFormat,
			"rawBytes":    bytesFormat,
			"nilBytes":    bytesFormat,
			"childPtr":    *structSchema,
			"nilPtr":      *structSchema,
			"emptySlice":  *intArraySchema,
			"nilSlice":    *intArraySchema,
			"emptyMap":    *intMapSchema,
			"nilMap":      *intMapSchema,
			"intOrString": {
				VendorExtensible: intOrStringSchema,
			},
			"quantity": *stringSchema, // TODO: If we add a quantity format to OpenAPI, test it here
			"i32":      *int32Schema,
			"i64":      *int64Schema,
			"f32":      *float32Schema,
			"enum":     *stringSchema,
			"mapList": {
				VendorExtensible: spec.VendorExtensible{Extensions: map[string]interface{}{
					"x-kubernetes-list-type":     "map",
					"x-kubernetes-list-map-keys": []any{"key1", "key2"},
				}},
				SchemaProps: spec.SchemaProps{Type: []string{"array"}, Items: &spec.SchemaOrArray{
					Schema: &spec.Schema{SchemaProps: spec.SchemaProps{
						Type: []string{"object"},
						Properties: map[string]spec.Schema{
							"key1":  *stringSchema,
							"key2":  *stringSchema,
							"value": *int64Schema,
						},
					}},
				}},
			},
			"setList": {
				VendorExtensible: spec.VendorExtensible{Extensions: map[string]interface{}{
					"x-kubernetes-list-type": "set",
				}},
				SchemaProps: intArraySchema.SchemaProps,
			},
		},
	},
}

func (c Complex) GetObjectKind() schema.ObjectKind {
	panic("not implemented")
}

func (c Complex) DeepCopyObject() runtime.Object {
	panic("not implemented")
}

type EnumType string

const (
	EnumTypeA EnumType = "a"
	EnumTypeB EnumType = "b"
)

var (
	stringSchema      = spec.StringProperty()
	bytesFormat       = spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "byte"}}
	durationFormat    = spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "duration"}}
	timeFormat        = spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{"string"}, Format: "date-time"}}
	intOrStringSchema = spec.VendorExtensible{Extensions: map[string]interface{}{"x-kubernetes-int-or-string": true}}
	int32Schema       = spec.Int32Property()
	int64Schema       = spec.Int64Property()
	boolSchema        = spec.BoolProperty()
	float32Schema     = spec.Float32Property()
	float64Schema     = spec.Float64Property()

	stringArraySchema = spec.ArrayProperty(stringSchema)
	intArraySchema    = spec.ArrayProperty(int64Schema)
	stringMapSchema   = spec.MapProperty(stringSchema)
	intMapSchema      = spec.MapProperty(int64Schema)
)

func typedToValActivation(vals map[string]typedValue) map[string]interface{} {
	activation := make(map[string]interface{}, len(vals))
	for k, tv := range vals {
		s := &openapi.Schema{Schema: tv.schema}
		activation[k] = common.TypedToVal(tv.value, s)
	}
	return activation
}

// unstructuredToValActivation converts the values in the activation map to map[string]interface{}.
func unstructuredToValActivation(vals map[string]typedValue) (map[string]interface{}, error) {
	activation := make(map[string]interface{}, len(vals))
	for k, tv := range vals {
		s := &openapi.Schema{Schema: tv.schema}
		switch v := tv.value.(type) {
		case runtime.Object:
			u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&v)
			if err != nil {
				return nil, err
			}
			activation[k] = common.UnstructuredToVal(u, s)
		case *runtime.Object:
			u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(v)
			if err != nil {
				return nil, err
			}
			activation[k] = common.UnstructuredToVal(u, s)
		default:
			u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(&wrap{Value: &v})
			if err != nil {
				return nil, err
			}
			if uv, ok := u["value"]; ok && uv != nil {
				activation[k] = common.UnstructuredToVal(uv, s)
			} else {
				activation[k] = types.NullValue
			}
		}
	}
	return activation, nil
}

type wrap struct {
	Value any `json:"value"`
}

type typedValue struct {
	value  any
	schema *spec.Schema
}

const RFC3339Micro = "2006-01-02T15:04:05.000000Z07:00"
const RFC3339 = "2006-01-02T15:04:05Z07:00"

type testCase struct {
	name       string
	expression string
	activation map[string]typedValue
	wantErr    string
}
