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
	"strings"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

type validationTestCase struct {
	name           string
	rootTypeName   string
	schema         typed.YAMLObject
	validObjects   []typed.YAMLObject
	invalidObjects []typed.YAMLObject
}

var validationCases = []validationTestCase{{
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
	validObjects: []typed.YAMLObject{
		`{"key":"foo","value":1}`,
		`{"key":"foo","value":{}}`,
		`{"key":"foo","value":null}`,
		`{"key":"foo"}`,
		`{"key":"foo","value":true}`,
		`{"key":"foo","value":true}`,
		`{"key":null}`,
	},
	invalidObjects: []typed.YAMLObject{
		`{"key":true,"value":1}`,
		`{"key":1,"value":{}}`,
		`{"key":false,"value":null}`,
		`{"key":[1, 2]}`,
		`{"key":{"foo":true}}`,
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
`,
	validObjects: []typed.YAMLObject{
		`{"numeric":null}`,
		`{"numeric":1}`,
		`{"numeric":3.14159}`,
		`{"string":null}`,
		`{"string":"aoeu"}`,
		`{"bool":null}`,
		`{"bool":true}`,
		`{"bool":false}`,
		`{"setStr":["a","b","c"]}`,
		`{"setBool":[true,false]}`,
		`{"setNumeric":[1,2,3,3.14159]}`,
	},
	invalidObjects: []typed.YAMLObject{
		`{"numeric":["foo"]}`,
		`{"numeric":{"a":1}}`,
		`{"numeric":"foo"}`,
		`{"numeric":true}`,
		`{"string":1}`,
		`{"string":3.5}`,
		`{"string":true}`,
		`{"string":{"a":1}}`,
		`{"string":["foo"]}`,
		`{"bool":1}`,
		`{"bool":3.5}`,
		`{"bool":"aoeu"}`,
		`{"bool":{"a":1}}`,
		`{"bool":["foo"]}`,
		`{"setStr":["a","a"]}`,
		`{"setBool":[true,false,true]}`,
		`{"setNumeric":[1,2,3,3.14159,1]}`,
		`{"setStr":[1]}`,
		`{"setStr":[true]}`,
		`{"setStr":[1.5]}`,
		`{"setStr":[null]}`,
		`{"setStr":[{}]}`,
		`{"setStr":[[]]}`,
		`{"setBool":[true,false,true]}`,
		`{"setBool":[1]}`,
		`{"setBool":[1.5]}`,
		`{"setBool":[null]}`,
		`{"setBool":[{}]}`,
		`{"setBool":[[]]}`,
		`{"setBool":["a"]}`,
		`{"setNumeric":[1,2,3,3.14159,1]}`,
		`{"setNumeric":[null]}`,
		`{"setNumeric":[true]}`,
		`{"setNumeric":["a"]}`,
		`{"setNumeric":[[]]}`,
		`{"setNumeric":[{}]}`,
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
	validObjects: []typed.YAMLObject{
		`{"list":[]}`,
		`{"list":[{"key":"a","id":1,"value":{"a":"a"}}]}`,
		`{"list":[{"key":"a","id":1},{"key":"a","id":2},{"key":"b","id":1}]}`,
		`{"atomicList":["a","a","a"]}`,
	},
	invalidObjects: []typed.YAMLObject{
		`{"key":true,"value":1}`,
		`{"list":{"key":true,"value":1}}`,
		`{"list":true}`,
		`true`,
		`{"list":[{"key":true,"value":1}]}`,
		`{"list":[{"key":[],"value":1}]}`,
		`{"list":[{"key":{},"value":1}]}`,
		`{"list":[{"key":1.5,"value":1}]}`,
		`{"list":[{"key":1,"value":1}]}`,
		`{"list":[{"key":null,"value":1}]}`,
		`{"list":[{},{}]}`,
		`{"list":[{},null]}`,
		`{"list":[[]]}`,
		`{"list":[null]}`,
		`{"list":[{}]}`,
		`{"list":[{"value":{"a":"a"},"bv":true,"nv":3.14}]}`,
		`{"list":[{"key":"a","id":1,"value":{"a":1}}]}`,
		`{"list":[{"key":"a","id":1},{"key":"a","id":1}]}`,
		`{"list":[{"key":"a","id":1,"value":{"a":"a"},"bv":"true","nv":3.14}]}`,
		`{"list":[{"key":"a","id":1,"value":{"a":"a"},"bv":true,"nv":false}]}`,
	},
}}

func (tt validationTestCase) test(t *testing.T) {
	parser, err := typed.NewParser(tt.schema)
	if err != nil {
		t.Fatalf("failed to create schema: %v", err)
	}
	pt := parser.Type(tt.rootTypeName)

	for i, v := range tt.validObjects {
		v := v
		t.Run(fmt.Sprintf("%v-valid-%v", tt.name, i), func(t *testing.T) {
			t.Parallel()
			_, err := pt.FromYAML(v)
			if err != nil {
				t.Errorf("failed to parse/validate yaml: %v\n%v", err, v)
			}
		})
	}

	for i, iv := range tt.invalidObjects {
		iv := iv
		t.Run(fmt.Sprintf("%v-invalid-%v", tt.name, i), func(t *testing.T) {
			t.Parallel()
			_, err := pt.FromYAML(iv)
			if err == nil {
				t.Errorf("Object should fail: %v\n%v", err, iv)
			}
			if strings.Contains(err.Error(), "invalid atom") {
				t.Errorf("Error should be useful, but got: %v\n%v", err, iv)
			}
		})
	}
}

func TestSchemaValidation(t *testing.T) {
	for _, tt := range validationCases {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			tt.test(t)
		})
	}
}

func TestSchemaSchema(t *testing.T) {
	// Verify that the schema schema validates itself.
	_, err := typed.NewParser(typed.YAMLObject(schema.SchemaSchemaYAML))
	if err != nil {
		t.Fatalf("failed to create schemaschema: %v", err)
	}
}

func BenchmarkValidateStructured(b *testing.B) {
	type Primitives struct {
		s string
		i int64
		f float64
		b bool
	}

	primitive1 := Primitives{s: "string1"}
	primitive2 := Primitives{i: 100}
	primitive3 := Primitives{f: 3.14}
	primitive4 := Primitives{b: true}

	type Example struct {
		listOfPrimitives []Primitives
		mapOfPrimitives  map[string]Primitives
		mapOfLists       map[string][]Primitives
	}

	tests := []struct {
		name         string
		rootTypeName string
		schema       typed.YAMLObject
		object       interface{}
	}{
		{
			name:         "struct",
			rootTypeName: "example",
			schema: `types:
- name: example
  map:
    fields:
    - name: listOfPrimitives
      type:
        list:
          elementType:
            namedType: primitives
    - name: mapOfPrimitives
      type:
        map:
          elementType:
            namedType: primitives
    - name: mapOfLists
      type:
        map:
          elementType:
            list:
              elementType:
                namedType: primitives
- name: primitives
  map:
    fields:
    - name: s
      type:
        scalar: string
    - name: i
      type:
        scalar: numeric
    - name: f
      type:
        scalar: numeric
    - name: b
      type:
        scalar: boolean
`,
			object: &Example{
				listOfPrimitives: []Primitives{primitive1, primitive2, primitive3, primitive4},
				mapOfPrimitives:  map[string]Primitives{"1": primitive1, "2": primitive2, "3": primitive3, "4": primitive4},
				mapOfLists: map[string][]Primitives{
					"1": {primitive1, primitive2, primitive3, primitive4},
					"2": {primitive1, primitive2, primitive3, primitive4},
				},
			},
		},
	}

	for _, test := range tests {
		b.Run(test.name, func(b *testing.B) {
			parser, err := typed.NewParser(test.schema)
			if err != nil {
				b.Fatalf("failed to create schema: %v", err)
			}
			pt := parser.Type(test.rootTypeName)

			b.ReportAllocs()
			for n := 0; n < b.N; n++ {
				tv, err := pt.FromStructured(test.object)
				if err != nil {
					b.Errorf("failed to parse/validate yaml: %v\n%v", err, tv)
				}
			}
		})
	}
}
