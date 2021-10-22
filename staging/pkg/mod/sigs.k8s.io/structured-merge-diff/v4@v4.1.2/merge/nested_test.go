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

package merge_test

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var nestedTypeParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: type
  map:
    fields:
      - name: listOfLists
        type:
          namedType: listOfLists
      - name: listOfMaps
        type:
          namedType: listOfMaps
      - name: mapOfLists
        type:
          namedType: mapOfLists
      - name: mapOfMaps
        type:
          namedType: mapOfMaps
      - name: mapOfMapsRecursive
        type:
          namedType: mapOfMapsRecursive
      - name: struct
        type:
          namedType: struct
- name: struct
  map:
    fields:
    - name: name
      type:
        scalar: string
    - name: value
      type:
        scalar: number
- name: listOfLists
  list:
    elementType:
      map:
        fields:
        - name: name
          type:
            scalar: string
        - name: value
          type:
            namedType: list
    elementRelationship: associative
    keys:
    - name
- name: list
  list:
    elementType:
      scalar: string
    elementRelationship: associative
- name: listOfMaps
  list:
    elementType:
      map:
        fields:
        - name: name
          type:
            scalar: string
        - name: value
          type:
            namedType: map
    elementRelationship: associative
    keys:
    - name
- name: map
  map:
    elementType:
      scalar: string
    elementRelationship: associative
- name: mapOfLists
  map:
    elementType:
      namedType: list
    elementRelationship: associative
- name: mapOfMaps
  map:
    elementType:
      namedType: map
    elementRelationship: associative
- name: mapOfMapsRecursive
  map:
    elementType:
      namedType: mapOfMapsRecursive
    elementRelationship: associative
`)
	if err != nil {
		panic(err)
	}
	return SameVersionParser{T: parser.Type("type")}
}()

func TestUpdateNestedType(t *testing.T) {
	tests := map[string]TestCase{
		"listOfLists_change_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - a
						  - c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				listOfLists:
				- name: a
				  value:
				  - a
				  - c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
						_P("listOfLists", _KBF("name", "a"), "value", _V("a")),
						_P("listOfLists", _KBF("name", "a"), "value", _V("c")),
					),
					"v1",
					false,
				),
			},
		},
		"listOfLists_change_key_and_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						listOfLists:
						- name: b
						  value:
						  - a
						  - c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				listOfLists:
				- name: b
				  value:
				  - a
				  - c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "b")),
						_P("listOfLists", _KBF("name", "b"), "name"),
						_P("listOfLists", _KBF("name", "b"), "value", _V("a")),
						_P("listOfLists", _KBF("name", "b"), "value", _V("c")),
					),
					"v1",
					false,
				),
			},
		},
		"listOfMaps_change_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						listOfMaps:
						- name: a
						  value:
						    b: "x"
						    c: "y"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						listOfMaps:
						- name: a
						  value:
						    a: "x"
						    c: "z"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				listOfMaps:
				- name: a
				  value:
				    a: "x"
				    c: "z"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfMaps", _KBF("name", "a")),
						_P("listOfMaps", _KBF("name", "a"), "name"),
						_P("listOfMaps", _KBF("name", "a"), "value", "a"),
						_P("listOfMaps", _KBF("name", "a"), "value", "c"),
					),
					"v1",
					true,
				),
			},
		},
		"listOfMaps_change_key_and_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						listOfMaps:
						- name: a
						  value:
						    b: "x"
						    c: "y"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						listOfMaps:
						- name: b
						  value:
						    a: "x"
						    c: "z"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				listOfMaps:
				- name: b
				  value:
				    a: "x"
				    c: "z"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfMaps", _KBF("name", "b")),
						_P("listOfMaps", _KBF("name", "b"), "name"),
						_P("listOfMaps", _KBF("name", "b"), "value", "a"),
						_P("listOfMaps", _KBF("name", "b"), "value", "c"),
					),
					"v1",
					false,
				),
			},
		},
		"mapOfLists_change_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						mapOfLists:
						  a:
						  - b
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						mapOfLists:
						  a:
						  - a
						  - c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				mapOfLists:
				  a:
				  - a
				  - c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfLists", "a"),
						_P("mapOfLists", "a", _V("a")),
						_P("mapOfLists", "a", _V("c")),
					),
					"v1",
					true,
				),
			},
		},
		"mapOfLists_change_key_and_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						mapOfLists:
						  a:
						  - b
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						mapOfLists:
						  b:
						  - a
						  - c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				mapOfLists:
				  b:
				  - a
				  - c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfLists", "b"),
						_P("mapOfLists", "b", _V("a")),
						_P("mapOfLists", "b", _V("c")),
					),
					"v1",
					false,
				),
			},
		},
		"mapOfMaps_change_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						mapOfMaps:
						  a:
						    b: "x"
						    c: "y"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						mapOfMaps:
						  a:
						    a: "x"
						    c: "z"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				mapOfMaps:
				  a:
				    a: "x"
				    c: "z"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMaps", "a"),
						_P("mapOfMaps", "a", "a"),
						_P("mapOfMaps", "a", "c"),
					),
					"v1",
					false,
				),
			},
		},
		"mapOfMaps_change_key_and_value": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						mapOfMaps:
						  a:
						    b: "x"
						    c: "y"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						mapOfMaps:
						  b:
						    a: "x"
						    c: "z"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				mapOfMaps:
				  b:
				    a: "x"
				    c: "z"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMaps", "b"),
						_P("mapOfMaps", "b", "a"),
						_P("mapOfMaps", "b", "c"),
					),
					"v1",
					false,
				),
			},
		},
		"mapOfMapsRecursive_change_middle_key": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						mapOfMapsRecursive:
						  a:
						    d:
						      c:
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  a:
				    d:
				      c:
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "d"),
						_P("mapOfMapsRecursive", "a", "d", "c"),
					),
					"v1",
					false,
				),
			},
		},
		"struct_apply_remove_all": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
					`,
					APIVersion: "v1",
				},
			},
			Object: `
			`,
			APIVersion: "v1",
			Managed:    fieldpath.ManagedFields{},
		},
		"struct_apply_remove_dangling": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						struct:
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("struct"),
					),
					"v1",
					true,
				),
			},
		},
		"struct_apply_update_remove_all": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						struct:
						  name: a
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  value: 1
			`,
			APIVersion: "v1",
		},
		"struct_apply_update_dict_dangling": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						struct:
						  name: a
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						struct: {}
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  value: 1
			`,
			APIVersion: "v1",
		},
		"struct_apply_update_dict_null": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						struct:
						  name: a
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						struct:
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  value: 1
			`,
			APIVersion: "v1",
		},
		"struct_apply_update_took_over": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						struct:
						  name: a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						struct:
						  name: b
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						struct:
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  name: b
				  value: 1
			`,
			APIVersion: "v1",
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(nestedTypeParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
