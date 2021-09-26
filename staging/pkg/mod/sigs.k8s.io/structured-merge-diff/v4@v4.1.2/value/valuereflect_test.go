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

package value

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"
)

func MustReflect(i interface{}) Value {
	if i == nil {
		return NewValueInterface(nil)
	}
	v, err := wrapValueReflect(reflect.ValueOf(i), nil, nil)
	if err != nil {
		panic(err)
	}
	return v
}

func TestReflectPrimitives(t *testing.T) {
	rv := MustReflect("string")
	if !rv.IsString() {
		t.Error("expected IsString to be true")
	}
	if rv.AsString() != "string" {
		t.Errorf("expected rv.String to be 'string' but got %v", rv.Unstructured())
	}

	rv = MustReflect([]byte("string"))
	if !rv.IsString() {
		t.Error("expected IsString to be true")
	}
	if rv.IsList() {
		t.Error("expected IsList to be false ([]byte is represented as a base64 encoded string)")
	}
	encoded := base64.StdEncoding.EncodeToString([]byte("string"))
	if rv.AsString() != encoded {
		t.Errorf("expected rv.String to be %v but got %v", []byte(encoded), rv.Unstructured())
	}

	rv = MustReflect(1)
	if !rv.IsInt() {
		t.Error("expected IsInt to be true")
	}
	if rv.AsInt() != 1 {
		t.Errorf("expected rv.Int to be 1 but got %v", rv.Unstructured())
	}

	rv = MustReflect(uint32(3000000000))
	if !rv.IsInt() {
		t.Error("expected IsInt to be true")
	}
	if rv.AsInt() != 3000000000 {
		t.Errorf("expected rv.Int to be 3000000000 but got %v", rv.Unstructured())
	}

	rv = MustReflect(1.5)
	if !rv.IsFloat() {
		t.Error("expected IsFloat to be true")
	}
	if rv.AsFloat() != 1.5 {
		t.Errorf("expected rv.Float to be 1.1 but got %v", rv.Unstructured())
	}

	rv = MustReflect(true)
	if !rv.IsBool() {
		t.Error("expected IsBool to be true")
	}
	if rv.AsBool() != true {
		t.Errorf("expected rv.Bool to be true but got %v", rv.Unstructured())
	}

	rv = MustReflect(nil)
	if !rv.IsNull() {
		t.Error("expected IsNull to be true")
	}
}

type Convertable struct {
	Value interface{}
}

func (t Convertable) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.Value)
}

func (t Convertable) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &t.Value)
}

type PtrConvertable struct {
	Value interface{}
}

func (t *PtrConvertable) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.Value)
}

func (t *PtrConvertable) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &t.Value)
}

type StringConvertable struct {
	Value string
}

func (t StringConvertable) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.Value)
}

func (t StringConvertable) ToUnstructured() (string, bool) {
	return t.Value, true
}

type PtrStringConvertable struct {
	Value string
}

func (t PtrStringConvertable) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.Value)
}

func (t *PtrStringConvertable) ToUnstructured() (string, bool) {
	return t.Value, true
}

func TestReflectCustomStringConversion(t *testing.T) {
	dateTime, err := time.Parse(time.RFC3339, "2006-01-02T15:04:05+07:00")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name        string
		convertable interface{}
		expected    interface{}
	}{
		{
			name:        "marshalable-struct",
			convertable: Convertable{Value: "struct-test"},
			expected:    "struct-test",
		},
		{
			name:        "marshalable-pointer",
			convertable: &PtrConvertable{Value: "pointer-test"},
			expected:    "pointer-test",
		},
		{
			name:        "pointer-to-marshalable-struct",
			convertable: &Convertable{Value: "pointer-test"},
			expected:    "pointer-test",
		},
		{
			name:        "string-convertable-struct",
			convertable: StringConvertable{Value: "struct-test"},
			expected:    "struct-test",
		},
		{
			name:        "string-convertable-pointer",
			convertable: &PtrStringConvertable{Value: "struct-test"},
			expected:    "struct-test",
		},
		{
			name:        "pointer-to-string-convertable-struct",
			convertable: &StringConvertable{Value: "pointer-test"},
			expected:    "pointer-test",
		},
		{
			name:        "time",
			convertable: dateTime,
			expected:    "2006-01-02T15:04:05+07:00",
		},
		{
			name:        "nil-marshalable-struct",
			convertable: Convertable{Value: nil},
			expected:    nil,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rv := MustReflect(tc.convertable)
			if rv.Unstructured() != tc.expected {
				t.Errorf("expected rv.String to be %v but got %s", tc.expected, rv.AsString())
			}
		})
	}
}

func TestReflectPointers(t *testing.T) {
	s := "string"
	rv := MustReflect(&s)
	if !rv.IsString() {
		t.Error("expected IsString to be true")
	}
	if rv.AsString() != "string" {
		t.Errorf("expected rv.String to be 'string' but got %s", rv.AsString())
	}
}

type T struct {
	I int64 `json:"int"`
}

type emptyStruct struct{}
type testBasicStruct struct {
	I int64 `json:"int"`
	S string
}
type testOmitStruct struct {
	I int64 `json:"-"`
	S string
}
type testInlineStruct struct {
	Inline T `json:",inline"`
	S      string
}
type testOmitemptyStruct struct {
	Noomit *string `json:"noomit"`
	Omit   *string `json:"omit,omitempty"`
}
type testEmbeddedStruct struct {
	*testBasicStruct `json:",inline"`
}

func TestReflectStruct(t *testing.T) {
	cases := []struct {
		name                 string
		val                  interface{}
		expectedMap          map[string]interface{}
		expectedUnstructured interface{}
	}{
		{
			name:                 "empty",
			val:                  emptyStruct{},
			expectedMap:          map[string]interface{}{},
			expectedUnstructured: map[string]interface{}{},
		},
		{
			name:                 "basic",
			val:                  testBasicStruct{I: 10, S: "string"},
			expectedMap:          map[string]interface{}{"int": int64(10), "S": "string"},
			expectedUnstructured: map[string]interface{}{"int": int64(10), "S": "string"},
		},
		{
			name:                 "pointerToBasic",
			val:                  &testBasicStruct{I: 10, S: "string"},
			expectedMap:          map[string]interface{}{"int": int64(10), "S": "string"},
			expectedUnstructured: map[string]interface{}{"int": int64(10), "S": "string"},
		},
		{
			name:                 "omit",
			val:                  testOmitStruct{I: 10, S: "string"},
			expectedMap:          map[string]interface{}{"S": "string"},
			expectedUnstructured: map[string]interface{}{"S": "string"},
		},
		{
			name:                 "inline",
			val:                  &testInlineStruct{Inline: T{I: 10}, S: "string"},
			expectedMap:          map[string]interface{}{"int": int64(10), "S": "string"},
			expectedUnstructured: map[string]interface{}{"int": int64(10), "S": "string"},
		},
		{
			name:                 "omitempty",
			val:                  testOmitemptyStruct{Noomit: nil, Omit: nil},
			expectedMap:          map[string]interface{}{"noomit": (*string)(nil)},
			expectedUnstructured: map[string]interface{}{"noomit": nil},
		},
		{
			name:                 "embedded",
			val:                  testEmbeddedStruct{&testBasicStruct{I: 10, S: "string"}},
			expectedMap:          map[string]interface{}{"int": int64(10), "S": "string"},
			expectedUnstructured: map[string]interface{}{"int": int64(10), "S": "string"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rv := MustReflect(tc.val)
			if !rv.IsMap() {
				t.Error("expected IsMap to be true")
			}
			m := rv.AsMap()
			if m.Length() != len(tc.expectedMap) {
				t.Errorf("expected map to be of length %d but got %d", len(tc.expectedMap), m.Length())
			}
			iterateResult := map[string]interface{}{}
			m.Iterate(func(s string, value Value) bool {
				iterateResult[s] = value.(*valueReflect).Value.Interface()
				return true
			})
			if !reflect.DeepEqual(iterateResult, tc.expectedMap) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedMap, iterateResult)
			}

			unstructured := rv.Unstructured()
			if !reflect.DeepEqual(unstructured, tc.expectedUnstructured) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedUnstructured, unstructured)
			}
		})
	}
}

type testMutateStruct struct {
	I1 int64  `json:"key1,omitempty"`
	S1 string `json:"key2,omitempty"`
	S2 string `json:"key3,omitempty"`
	S3 string `json:"key4,omitempty"`
}

func TestReflectStructMutate(t *testing.T) {
	rv := MustReflect(&testMutateStruct{I1: 1, S1: "string1"})
	if !rv.IsMap() {
		t.Error("expected IsMap to be true")
	}
	m := rv.AsMap()
	atKey1, ok := m.Get("key1")
	if !ok {
		t.Fatalf("expected map.Get(key1) to be 1 but got !ok")
	}
	if atKey1.AsInt() != 1 {
		t.Fatalf("expected map.Get(key1) to be 1 but got: %v", atKey1)
	}
	m.Set("key1", NewValueInterface(int64(2)))
	m.Delete("key2")
	m.Delete("key3")
	m.Set("key4", NewValueInterface("string4"))

	expectedMap := map[string]interface{}{"key1": int64(2), "key4": "string4"}
	unstructured := rv.Unstructured()
	if !reflect.DeepEqual(unstructured, expectedMap) {
		t.Errorf("expected %v but got: %v", expectedMap, unstructured)
	}
}

// TestReflectMutateNestedStruct ensures a structs field within various typed can be modified.
func TestReflectMutateNestedStruct(t *testing.T) {
	type field struct {
		S string `json:"s,omitempty"`
	}

	cases := []struct {
		fieldName     string
		root          Value
		lookupField   func(root Value) Value
		expectUpdated interface{}
		expectDeleted interface{}
	}{
		{
			fieldName: "field",
			root: MustReflect(&struct {
				Field field `json:"field,omitempty"`
			}{
				Field: field{S: "field"},
			}),
			lookupField: func(rv Value) Value {
				field, _ := rv.AsMap().Get("field")
				return field
			},
			expectUpdated: map[string]interface{}{
				"field": map[string]interface{}{"s": "updatedValue"},
			},
			expectDeleted: map[string]interface{}{
				"field": map[string]interface{}{},
			},
		},
		{
			fieldName: "map",
			root: MustReflect(&struct {
				Map map[string]field `json:"map,omitempty"`
			}{
				Map: map[string]field{"mapKey": {S: "mapItem"}},
			}),
			lookupField: func(rv Value) Value {
				m, _ := rv.AsMap().Get("map")
				mapItem, _ := m.AsMap().Get("mapKey")
				return mapItem
			},
			expectUpdated: map[string]interface{}{
				"map": map[string]interface{}{"mapKey": map[string]interface{}{"s": "updatedValue"}},
			},
			expectDeleted: map[string]interface{}{
				"map": map[string]interface{}{"mapKey": map[string]interface{}{}},
			},
		},
		{
			fieldName: "mapiter",
			root: MustReflect(&struct {
				Mapiter map[string]field `json:"mapiter,omitempty"`
			}{
				Mapiter: map[string]field{"mapKey": {S: "mapItem"}},
			}),
			lookupField: func(rv Value) Value {
				mapItem := &valueReflect{}
				m, _ := rv.AsMap().Get("mapiter")
				m.AsMap().Iterate(func(key string, value Value) bool {
					if key == "mapKey" {
						*mapItem = *value.(*valueReflect)
						return false
					}
					return true
				})
				if !mapItem.Value.IsValid() {
					t.Fatal("map item not found")
				}
				return mapItem
			},
			expectUpdated: map[string]interface{}{
				"mapiter": map[string]interface{}{"mapKey": map[string]interface{}{"s": "updatedValue"}},
			},
			expectDeleted: map[string]interface{}{
				"mapiter": map[string]interface{}{"mapKey": map[string]interface{}{}},
			},
		},
		{
			fieldName: "list",
			root: MustReflect(&struct {
				List []field `json:"list,omitempty"`
			}{
				List: []field{{S: "listItem"}},
			}),
			lookupField: func(rv Value) Value {
				list, _ := rv.AsMap().Get("list")
				return list.AsList().At(0)
			},
			expectUpdated: map[string]interface{}{
				"list": []interface{}{map[string]interface{}{"s": "updatedValue"}},
			},
			expectDeleted: map[string]interface{}{
				"list": []interface{}{map[string]interface{}{}},
			},
		},
		{
			fieldName: "mapOfMaps",
			root: MustReflect(&struct {
				MapOfMaps map[string]map[string]field `json:"mapOfMaps,omitempty"`
			}{
				MapOfMaps: map[string]map[string]field{"outer": {"inner": {S: "mapOfMapItem"}}},
			}),
			lookupField: func(rv Value) Value {
				mapOfMaps, _ := rv.AsMap().Get("mapOfMaps")
				innerMap, _ := mapOfMaps.AsMap().Get("outer")
				mapOfMapsItem, _ := innerMap.AsMap().Get("inner")
				return mapOfMapsItem
			},
			expectUpdated: map[string]interface{}{
				"mapOfMaps": map[string]interface{}{"outer": map[string]interface{}{"inner": map[string]interface{}{"s": "updatedValue"}}},
			},
			expectDeleted: map[string]interface{}{
				"mapOfMaps": map[string]interface{}{"outer": map[string]interface{}{"inner": map[string]interface{}{}}},
			},
		},
		{
			fieldName: "mapOfLists",
			root: MustReflect(&struct {
				MapOfLists map[string][]field `json:"mapOfLists,omitempty"`
			}{
				MapOfLists: map[string][]field{"outer": {{S: "mapOfListsItem"}}},
			}),
			lookupField: func(rv Value) Value {
				mapOfLists, _ := rv.AsMap().Get("mapOfLists")
				innerList, _ := mapOfLists.AsMap().Get("outer")
				mapOfListsItem := innerList.AsList().At(0)
				return mapOfListsItem
			},

			expectUpdated: map[string]interface{}{
				"mapOfLists": map[string]interface{}{"outer": []interface{}{map[string]interface{}{"s": "updatedValue"}}},
			},
			expectDeleted: map[string]interface{}{
				"mapOfLists": map[string]interface{}{"outer": []interface{}{map[string]interface{}{}}},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.fieldName, func(t *testing.T) {
			root := tc.root
			field := tc.lookupField(root)
			field.AsMap().Set("s", NewValueInterface("updatedValue"))
			unstructured := root.Unstructured()
			if !reflect.DeepEqual(unstructured, tc.expectUpdated) {
				t.Errorf("expected %v but got: %v", tc.expectUpdated, unstructured)
			}

			field.AsMap().Delete("s")
			unstructured = root.Unstructured()
			if !reflect.DeepEqual(unstructured, tc.expectDeleted) {
				t.Errorf("expected %v but got: %v", tc.expectDeleted, unstructured)
			}
		})
	}
}

func TestReflectMap(t *testing.T) {
	cases := []struct {
		name                 string
		val                  interface{}
		expectedMap          map[string]interface{}
		expectedUnstructured interface{}
		length               int
	}{
		{
			name:                 "empty",
			val:                  map[string]string{},
			expectedMap:          map[string]interface{}{},
			expectedUnstructured: map[string]interface{}{},
			length:               0,
		},
		{
			name:                 "stringMap",
			val:                  map[string]string{"key1": "value1", "key2": "value2"},
			expectedMap:          map[string]interface{}{"key1": "value1", "key2": "value2"},
			expectedUnstructured: map[string]interface{}{"key1": "value1", "key2": "value2"},
			length:               2,
		},
		{
			name:                 "convertableMap",
			val:                  map[string]Convertable{"key1": {"converted1"}, "key2": {"converted2"}},
			expectedMap:          map[string]interface{}{"key1": "converted1", "key2": "converted2"},
			expectedUnstructured: map[string]interface{}{"key1": "converted1", "key2": "converted2"},
			length:               2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rv := MustReflect(tc.val)
			if !rv.IsMap() {
				t.Error("expected IsMap to be true")
			}
			m := rv.AsMap()
			if m.Length() != tc.length {
				t.Errorf("expected map to be of length %d but got %d", tc.length, m.Length())
			}
			iterateResult := map[string]interface{}{}
			m.Iterate(func(s string, value Value) bool {
				iterateResult[s] = value.AsString()
				return true
			})
			if !reflect.DeepEqual(iterateResult, tc.expectedMap) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedMap, iterateResult)
			}
			unstructured := rv.Unstructured()
			if !reflect.DeepEqual(unstructured, tc.expectedUnstructured) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedUnstructured, unstructured)
			}
		})
	}
}

func TestReflectMapMutate(t *testing.T) {
	rv := MustReflect(map[string]string{"key1": "value1", "key2": "value2"})
	if !rv.IsMap() {
		t.Error("expected IsMap to be true")
	}
	m := rv.AsMap()
	atKey1, ok := m.Get("key1")
	if !ok {
		t.Errorf("expected map.Get(key1) to be 'value1' but got !ok")
	}
	if atKey1.AsString() != "value1" {
		t.Errorf("expected map.Get(key1) to be 'value1' but got: %v", atKey1)
	}
	m.Set("key1", NewValueInterface("replacement"))
	m.Delete("key2")
	m.Delete("key3")
	m.Set("key4", NewValueInterface("value4"))

	expectedMap := map[string]interface{}{"key1": "replacement", "key4": "value4"}
	unstructured := rv.Unstructured()
	if !reflect.DeepEqual(unstructured, expectedMap) {
		t.Errorf("expected %v but got: %v", expectedMap, unstructured)
	}
}

func TestReflectList(t *testing.T) {
	cases := []struct {
		name                 string
		val                  interface{}
		expectedIterate      []interface{}
		expectedUnstructured interface{}
		length               int
	}{
		{
			name:                 "empty",
			val:                  []string{},
			expectedIterate:      []interface{}{},
			expectedUnstructured: []interface{}{},
			length:               0,
		},
		{
			name:                 "stringList",
			val:                  []string{"value1", "value2"},
			expectedIterate:      []interface{}{"value1", "value2"},
			expectedUnstructured: []interface{}{"value1", "value2"},
			length:               2,
		},
		{
			name:                 "convertableList",
			val:                  []Convertable{{"converted1"}, {"converted2"}},
			expectedIterate:      []interface{}{"converted1", "converted2"},
			expectedUnstructured: []interface{}{"converted1", "converted2"},
			length:               2,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rv := MustReflect(tc.val)
			if !rv.IsList() {
				t.Error("expected IsList to be true")
			}
			m := rv.AsList()
			if m.Length() != tc.length {
				t.Errorf("expected list to be of length %d but got %d", tc.length, m.Length())
			}

			l := m.Length()
			iterateResult := make([]interface{}, l)
			for i := 0; i < l; i++ {
				iterateResult[i] = m.At(i).AsString()
			}
			if !reflect.DeepEqual(iterateResult, tc.expectedIterate) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedIterate, iterateResult)
			}

			iter := m.Range()
			iterateResult = make([]interface{}, l)
			for iter.Next() {
				i, val := iter.Item()
				iterateResult[i] = val.AsString()
			}
			if !reflect.DeepEqual(iterateResult, tc.expectedIterate) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedIterate, iterateResult)
			}

			unstructured := rv.Unstructured()
			if !reflect.DeepEqual(unstructured, tc.expectedUnstructured) {
				t.Errorf("expected iterate to produce %#v but got %#v", tc.expectedUnstructured, unstructured)
			}
		})
	}
}

func TestReflectListAt(t *testing.T) {
	rv := MustReflect([]string{"one", "two"})
	if !rv.IsList() {
		t.Error("expected IsList to be true")
	}
	list := rv.AsList()
	atOne := list.At(1)
	if atOne.AsString() != "two" {
		t.Errorf("expected list.At(1) to be 'two' but got: %v", atOne)
	}
}

func TestMapZip(t *testing.T) {
	type entry struct {
		key      string
		lhs, rhs interface{}
	}

	type s struct {
		// deliberately unordered
		C string `json:"c,omitempty"`
		B string `json:"b,omitempty"`
		D string `json:"d,omitempty"`
		A string `json:"a,omitempty"`
	}
	cases := []struct {
		name           string
		lhs            interface{}
		rhs            interface{}
		expectedZipped []entry
	}{
		{
			name: "structZip",
			lhs:  &s{A: "1", B: "3", C: "5"},
			rhs:  &s{A: "2", B: "4", D: "6"},
			expectedZipped: []entry{
				{"a", "1", "2"},
				{"b", "3", "4"},
				{"c", "5", interface{}(nil)},
				{"d", interface{}(nil), "6"},
			},
		},
		{
			name: "mapZip",
			lhs:  &map[string]interface{}{"a": "1", "b": "3", "c": "5"},
			rhs:  &map[string]interface{}{"a": "2", "b": "4", "d": "6"},
			expectedZipped: []entry{
				{"a", "1", "2"},
				{"b", "3", "4"},
				{"c", "5", interface{}(nil)},
				{"d", interface{}(nil), "6"},
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			lhs := MustReflect(tc.lhs)
			rhs := MustReflect(tc.rhs)
			for _, lhs := range []Value{lhs, NewValueInterface(lhs.Unstructured())} {
				for _, rhs := range []Value{rhs, NewValueInterface(rhs.Unstructured())} {
					t.Run(fmt.Sprintf("%s-%s", reflect.TypeOf(lhs).Elem().Name(), reflect.TypeOf(rhs).Elem().Name()), func(t *testing.T) {
						for _, order := range []MapTraverseOrder{Unordered, LexicalKeyOrder} {
							var zipped []entry
							var name string
							switch order {
							case Unordered:
								name = "Unordered"
							case LexicalKeyOrder:
								name = "LexicalKeyOrder"
							}
							t.Run(name, func(t *testing.T) {
								MapZip(lhs.AsMap(), rhs.AsMap(), order, func(key string, lhs, rhs Value) bool {
									var li, ri interface{}
									if lhs != nil {
										li = lhs.Unstructured()
									}
									if rhs != nil {
										ri = rhs.Unstructured()
									}
									zipped = append(zipped, entry{key, li, ri})
									return true
								})
								if order == Unordered {
									sort.Slice(zipped, func(i, j int) bool {
										return zipped[i].key < zipped[j].key
									})
								}
								if !reflect.DeepEqual(zipped, tc.expectedZipped) {
									t.Errorf("expected zip to produce:\n%#v\nbut got\n%#v", tc.expectedZipped, zipped)
								}
							})
						}
					})
				}
			}
		})
	}
}
