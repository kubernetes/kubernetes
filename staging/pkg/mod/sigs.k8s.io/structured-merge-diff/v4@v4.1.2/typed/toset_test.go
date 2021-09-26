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

package typed_test

import (
	"fmt"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

type objSetPair struct {
	object typed.YAMLObject
	set    *fieldpath.Set
}

type fieldsetTestCase struct {
	name         string
	rootTypeName string
	schema       typed.YAMLObject
	pairs        []objSetPair
}

var (
	// Short names for readable test cases.
	_NS  = fieldpath.NewSet
	_P   = fieldpath.MakePathOrDie
	_KBF = fieldpath.KeyByFields
	_V   = value.NewValueInterface
)

var fieldsetCases = []fieldsetTestCase{{
	name:         "simple pair",
	rootTypeName: "stringPair",
	schema: `types:
- name: stringPair
  map:
    fields:
    - name: key
      type:
        scalar: string
    - name: value
      type:
        namedType: __untyped_atomic_
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
`,
	pairs: []objSetPair{
		{`{"key":"foo","value":1}`, _NS(_P("key"), _P("value"))},
		{`{"key":"foo","value":{"a": "b"}}`, _NS(_P("key"), _P("value"))},
		{`{"key":"foo","value":null}`, _NS(_P("key"), _P("value"))},
		{`{"key":"foo"}`, _NS(_P("key"))},
		{`{"key":"foo","value":true}`, _NS(_P("key"), _P("value"))},
	},
}, {
	name:         "struct grab bag",
	rootTypeName: "myStruct",
	schema: `types:
- name: myStruct
  map:
    fields:
    - name: numeric
      type:
        scalar: numeric
    - name: string
      type:
        scalar: string
    - name: bool
      type:
        scalar: boolean
    - name: setStr
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: associative
    - name: setBool
      type:
        list:
          elementType:
            scalar: boolean
          elementRelationship: associative
    - name: setNumeric
      type:
        list:
          elementType:
            scalar: numeric
          elementRelationship: associative
    - name: color
      type:
        map:
          fields:
          - name: R
            type:
              scalar: numeric
          - name: G
            type:
              scalar: numeric
          - name: B
            type:
              scalar: numeric
          elementRelationship: atomic
    - name: arbitraryWavelengthColor
      type:
        map:
          elementType:
            scalar: numeric
          elementRelationship: atomic
    - name: args
      type:
        list:
          elementType:
            map:
              fields:
              - name: key
                type:
                  scalar: string
              - name: value
                type:
                  scalar: string
          elementRelationship: atomic
`,
	pairs: []objSetPair{
		{`{"numeric":1}`, _NS(_P("numeric"))},
		{`{"numeric":3.14159}`, _NS(_P("numeric"))},
		{`{"string":"aoeu"}`, _NS(_P("string"))},
		{`{"bool":true}`, _NS(_P("bool"))},
		{`{"bool":false}`, _NS(_P("bool"))},
		{`{"setStr":["a","b","c"]}`, _NS(
			_P("setStr", _V("a")),
			_P("setStr", _V("b")),
			_P("setStr", _V("c")),
		)},
		{`{"setBool":[true,false]}`, _NS(
			_P("setBool", _V(true)),
			_P("setBool", _V(false)),
		)},
		{`{"setNumeric":[1,2,3,3.14159]}`, _NS(
			_P("setNumeric", _V(1)),
			_P("setNumeric", _V(2)),
			_P("setNumeric", _V(3)),
			_P("setNumeric", _V(3.14159)),
		)},
		{`{"color":{}}`, _NS(_P("color"))},
		{`{"color":null}`, _NS(_P("color"))},
		{`{"color":{"R":255,"G":0,"B":0}}`, _NS(_P("color"))},
		{`{"arbitraryWavelengthColor":{}}`, _NS(_P("arbitraryWavelengthColor"))},
		{`{"arbitraryWavelengthColor":null}`, _NS(_P("arbitraryWavelengthColor"))},
		{`{"arbitraryWavelengthColor":{"IR":255}}`, _NS(_P("arbitraryWavelengthColor"))},
		{`{"args":[]}`, _NS(_P("args"))},
		{`{"args":null}`, _NS(_P("args"))},
		{`{"args":[null]}`, _NS(_P("args"))},
		{`{"args":[{"key":"a","value":"b"},{"key":"c","value":"d"}]}`, _NS(_P("args"))},
	},
}, {
	name:         "associative list",
	rootTypeName: "myRoot",
	schema: `types:
- name: myRoot
  map:
    fields:
    - name: list
      type:
        namedType: myList
    - name: atomicList
      type:
        namedType: mySequence
- name: myList
  list:
    elementType:
      namedType: myElement
    elementRelationship: associative
    keys:
    - key
    - id
- name: mySequence
  list:
    elementType:
      scalar: string
    elementRelationship: atomic
- name: myElement
  map:
    fields:
    - name: key
      type:
        scalar: string
    - name: id
      type:
        scalar: numeric
    - name: value
      type:
        namedType: myValue
    - name: bv
      type:
        scalar: boolean
    - name: nv
      type:
        scalar: numeric
- name: myValue
  map:
    elementType:
      scalar: string
`,
	pairs: []objSetPair{
		{`{"list":[]}`, _NS()},
		{`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`, _NS(
			_P("list", _KBF("key", "a", "id", 1)),
			_P("list", _KBF("key", "a", "id", 1), "key"),
			_P("list", _KBF("key", "a", "id", 1), "id"),
			_P("list", _KBF("key", "a", "id", 1), "value", "a"),
		)},
		{`{"list":[{"key":"a","id":1},{"key":"a","id":2},{"key":"b","id":1}]}`, _NS(
			_P("list", _KBF("key", "a", "id", 1)),
			_P("list", _KBF("key", "a", "id", 2)),
			_P("list", _KBF("key", "b", "id", 1)),
			_P("list", _KBF("key", "a", "id", 1), "key"),
			_P("list", _KBF("key", "a", "id", 1), "id"),
			_P("list", _KBF("key", "a", "id", 2), "key"),
			_P("list", _KBF("key", "a", "id", 2), "id"),
			_P("list", _KBF("key", "b", "id", 1), "key"),
			_P("list", _KBF("key", "b", "id", 1), "id"),
		)},
		{`{"atomicList":["a","a","a"]}`, _NS(_P("atomicList"))},
	},
}}

func (tt fieldsetTestCase) test(t *testing.T) {
	parser, err := typed.NewParser(tt.schema)
	if err != nil {
		t.Fatalf("failed to create schema: %v", err)
	}
	for i, v := range tt.pairs {
		v := v
		t.Run(fmt.Sprintf("%v-%v", tt.name, i), func(t *testing.T) {
			t.Parallel()
			tv, err := parser.Type(tt.rootTypeName).FromYAML(v.object)
			if err != nil {
				t.Errorf("failed to parse object: %v", err)
			}
			fs, err := tv.ToFieldSet()
			if err != nil {
				t.Fatalf("got validation errors: %v", err)
			}
			if !fs.Equals(v.set) {
				t.Errorf("wanted\n%s\ngot\n%s\n", v.set, fs)
			}
		})
	}
}

func TestToFieldSet(t *testing.T) {
	for _, tt := range fieldsetCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.test(t)
		})
	}
}
