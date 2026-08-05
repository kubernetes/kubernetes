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

// This file directly tests construction of the golang map keys used to implement
// x-kubernetes-list-type=map semantics (merge on Add, order-insensitive Equal) in
// valuesreflect.go. TestListAdd in values_test.go covers the same dimensions end-to-end
// through CEL expression evaluation; the tests here assert the exact keys produced —
// including the fmt %#v serialized string form used for more than 3 key props — so that the
// constructed keys are directly visible in the test expectations.
//
// Two key constructors exist, and keys from one are never compared with keys from the other:
//
//   - typedMapList.toMapKey reads struct fields reflectively by unescaped JSON property name
//     and preserves the declared Go types of the fields (e.g. an int32 key field contributes
//     an int32). Pointer fields are dereferenced so keys compare by value, not by address.
//
//   - refValMapKey reads key values through CEL accessors by escaped property name. The
//     accessors return CEL-normalized values (all ints as int64, named string types as string,
//     pointers dereferenced), which is what makes keys agree across typed struct, unstructured
//     map and CEL literal map representations of the same logical element.

import (
	"reflect"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
)

type keyTestEntry struct {
	Name     string  `json:"name"`
	Port     int32   `json:"port"`
	Protocol string  `json:"protocol"`
	Extra    string  `json:"extra"`
	If       string  `json:"if"` // CEL reserved word, escapes to __if__
	PtrName  *string `json:"ptrName"`
	PtrPort  *int32  `json:"ptrPort"`
	Raw      []byte  `json:"raw"` // not comparable in go
}

type keyTestWideEntry struct {
	K1 string `json:"k1"`
	K2 string `json:"k2"`
	K3 string `json:"k3"`
	K4 string `json:"k4"`
}

func TestTypedMapListToMapKey(t *testing.T) {
	cases := []struct {
		name                string
		keyProps            []string // unescaped JSON names
		element             interface{}
		want                interface{}
		wantErr             bool
		sameKeyElement      interface{} // If set, test that this key is considered equivalent to keyProps (useful for testing pointer keys)
		differentKeyElement interface{} // If set, test that this key is considered non-equivalent to keyProps (provides a negative test when testing pointer keys)
	}{
		{
			name:     "one key prop",
			keyProps: []string{"name"},
			element:  keyTestEntry{Name: "a"},
			want:     "a",
		},
		{
			name:     "two key props preserve declared go types",
			keyProps: []string{"name", "port"},
			element:  keyTestEntry{Name: "a", Port: 8080},
			want:     [2]interface{}{"a", int32(8080)},
		},
		{
			name:     "three key props",
			keyProps: []string{"name", "port", "protocol"},
			element:  keyTestEntry{Name: "a", Port: 8080, Protocol: "TCP"},
			want:     [3]interface{}{"a", int32(8080), "TCP"},
		},
		{
			name:     "four key props serialize to a string",
			keyProps: []string{"name", "port", "protocol", "extra"},
			element:  keyTestEntry{Name: "a", Port: 8080, Protocol: "TCP", Extra: "x"},
			want:     `[]interface {}{"a", 8080, "TCP", "x"}`,
		},
		{
			name:                "serialized keys keep whitespace boundaries distinct",
			keyProps:            []string{"k1", "k2", "k3", "k4"},
			element:             keyTestWideEntry{K1: "a b", K2: "c", K3: "d", K4: "e"},
			want:                `[]interface {}{"a b", "c", "d", "e"}`,
			differentKeyElement: keyTestWideEntry{K1: "a", K2: "b c", K3: "d", K4: "e"},
		},
		{
			name:     "key prop that requires escaping in CEL is unescaped here",
			keyProps: []string{"if"},
			element:  keyTestEntry{If: "cond"},
			want:     "cond",
		},
		{
			name:           "pointer key prop is dereferenced",
			keyProps:       []string{"ptrName"},
			element:        keyTestEntry{PtrName: new("a")},
			want:           "a",
			sameKeyElement: keyTestEntry{PtrName: new("a")},
		},
		{
			name:                "two pointer key props compare by pointed-to values",
			keyProps:            []string{"ptrName", "ptrPort"},
			element:             keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8080))},
			want:                [2]interface{}{"a", int32(8080)},
			sameKeyElement:      keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8080))},
			differentKeyElement: keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8081))},
		},
		{
			name:     "nil pointer key props contribute nil",
			keyProps: []string{"ptrName", "ptrPort"},
			element:  keyTestEntry{},
			want:     [2]interface{}{nil, nil},
		},
		{
			name:           "serialized keys contain pointed-to values, not addresses",
			keyProps:       []string{"ptrName", "ptrPort", "name", "extra"},
			element:        keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8080)), Name: "n", Extra: "x"},
			want:           `[]interface {}{"a", 8080, "n", "x"}`,
			sameKeyElement: keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8080)), Name: "n", Extra: "x"},
		},
		{
			name:     "non-comparable key value is serialized",
			keyProps: []string{"raw"},
			element:  keyTestEntry{Raw: []byte("ab")},
			want:     `[]byte{0x61, 0x62}`,
		},
		{
			name:     "non-struct element returns an error key",
			keyProps: []string{"name"},
			element:  "not a struct",
			wantErr:  true,
		},
		{
			name:     "unknown key prop returns an error key",
			keyProps: []string{"nosuch"},
			element:  keyTestEntry{Name: "a"},
			wantErr:  true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			list := &typedMapList{
				typedList:       typedList{value: reflect.ValueOf([]interface{}{tc.element})},
				keyProps:        tc.keyProps,
				escapedKeyProps: escapeKeyProps(tc.keyProps),
			}
			got := list.toMapKey(reflect.ValueOf(tc.element))
			if tc.wantErr {
				if _, isErr := got.(*types.Err); !isErr {
					t.Errorf("toMapKey() = %#v (%T), want *types.Err", got, got)
				}
				return
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("toMapKey() = %#v (%T), want %#v (%T)", got, got, tc.want, tc.want)
			}
			if tc.sameKeyElement != nil {
				same := list.toMapKey(reflect.ValueOf(tc.sameKeyElement))
				if got != same {
					t.Errorf("elements with equal key values produced distinct keys: %#v != %#v", got, same)
				}
			}
			if tc.differentKeyElement != nil {
				different := list.toMapKey(reflect.ValueOf(tc.differentKeyElement))
				if got == different {
					t.Errorf("elements with different key values produced the same key: %#v", got)
				}
			}
		})
	}
}

func TestRefValMapKey(t *testing.T) {
	typedStructOf := func(v interface{}) ref.Val {
		return &typedStruct{
			value:      reflect.ValueOf(v),
			propSchema: func(key string) (Schema, bool) { return nil, true },
		}
	}

	cases := []struct {
		name             string
		escapedKeyProps  []string // escaped CEL identifiers
		element          interface{}
		want             interface{}
		equivalentCELMap map[string]interface{} // If set, is the same logical map.
	}{
		{
			name:             "one key prop",
			escapedKeyProps:  []string{"name"},
			element:          keyTestEntry{Name: "a"},
			want:             "a",
			equivalentCELMap: map[string]interface{}{"name": "a"},
		},
		{
			name:             "int key props normalize to int64 across representations",
			escapedKeyProps:  []string{"name", "port"},
			element:          keyTestEntry{Name: "a", Port: 8080}, // int32 field
			want:             [2]interface{}{"a", int64(8080)},
			equivalentCELMap: map[string]interface{}{"name": "a", "port": 8080}, // go int
		},
		{
			name:             "reserved word key prop is looked up by escaped name",
			escapedKeyProps:  []string{"__if__"},
			element:          keyTestEntry{If: "cond"},
			want:             "cond",
			equivalentCELMap: map[string]interface{}{"__if__": "cond"},
		},
		{
			name:            "three key props",
			escapedKeyProps: []string{"name", "port", "protocol"},
			element:         keyTestEntry{Name: "a", Port: 8080, Protocol: "TCP"},
			want:            [3]interface{}{"a", int64(8080), "TCP"},
		},
		{
			name:             "four key props serialize to a string",
			escapedKeyProps:  []string{"name", "port", "protocol", "extra"},
			element:          keyTestEntry{Name: "a", Port: 8080, Protocol: "TCP", Extra: "x"},
			want:             `[]interface {}{"a", 8080, "TCP", "x"}`,
			equivalentCELMap: map[string]interface{}{"name": "a", "port": 8080, "protocol": "TCP", "extra": "x"},
		},
		{
			name:             "serialized map key with whitespace in values",
			escapedKeyProps:  []string{"k1", "k2", "k3", "k4"},
			element:          keyTestWideEntry{K1: "a b", K2: "c", K3: "d", K4: "e"},
			want:             `[]interface {}{"a b", "c", "d", "e"}`,
			equivalentCELMap: map[string]interface{}{"k1": "a b", "k2": "c", "k3": "d", "k4": "e"},
		},
		{
			name:            "serialized map key with shifted whitespace boundary is distinct",
			escapedKeyProps: []string{"k1", "k2", "k3", "k4"},
			element:         keyTestWideEntry{K1: "a", K2: "b c", K3: "d", K4: "e"},
			want:            `[]interface {}{"a", "b c", "d", "e"}`,
		},
		{
			name:             "pointer key props are dereferenced and normalized",
			escapedKeyProps:  []string{"ptrName", "ptrPort"},
			element:          keyTestEntry{PtrName: new("a"), PtrPort: new(int32(8080))},
			want:             [2]interface{}{"a", int64(8080)},
			equivalentCELMap: map[string]interface{}{"ptrName": "a", "ptrPort": 8080},
		},
		{
			name:            "unset pointer key props contribute nil",
			escapedKeyProps: []string{"ptrName", "ptrPort"},
			element:         keyTestEntry{},
			want:            [2]interface{}{nil, nil},
		},
		{
			name:            "unresolvable key prop contributes nil",
			escapedKeyProps: []string{"nosuch"},
			element:         keyTestEntry{Name: "a"},
			want:            nil,
		},
		{
			name:            "non-comparable key value is serialized",
			escapedKeyProps: []string{"raw"},
			element:         keyTestEntry{Raw: []byte("ab")},
			want:            `[]byte{0x61, 0x62}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := refValMapKey(typedStructOf(tc.element), tc.escapedKeyProps)
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("refValMapKey() = %#v (%T), want %#v (%T)", got, got, tc.want, tc.want)
			}
			if tc.equivalentCELMap != nil {
				celMapKey := refValMapKey(types.DefaultTypeAdapter.NativeToValue(tc.equivalentCELMap), tc.escapedKeyProps)
				if got != celMapKey {
					t.Errorf("typed element key %#v differs from equivalent CEL map key %#v", got, celMapKey)
				}
			}
		})
	}
}
