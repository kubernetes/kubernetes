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
	"fmt"
	"strconv"
	"strings"
	"testing"

	"gopkg.in/yaml.v2"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

func TestMultipleAppliersSet(t *testing.T) {
	tests := map[string]TestCase{
		"remove_one": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						list:
						- name: a
						- name: b
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v2",
					Object: `
						list:
						- name: c
					`,
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: "v3",
					Object: `
						list:
						- name: a
					`,
				},
			},
			Object: `
				list:
				- name: a
				- name: c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "a")),
						_P("list", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "c")),
						_P("list", _KBF("name", "c"), "name"),
					),
					"v2",
					false,
				),
			},
		},
		"same_value_no_conflict": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						list:
						- name: a
						  value: 0
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v2",
					Object: `
						list:
						- name: a
						  value: 0
					`,
				},
			},
			Object: `
				list:
				- name: a
				  value: 0
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "a")),
						_P("list", _KBF("name", "a"), "name"),
						_P("list", _KBF("name", "a"), "value"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "a")),
						_P("list", _KBF("name", "a"), "name"),
						_P("list", _KBF("name", "a"), "value"),
					),
					"v2",
					false,
				),
			},
		},
		"change_value_yes_conflict": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						list:
						- name: a
						  value: 0
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v2",
					Object: `
						list:
						- name: a
						  value: 1
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "apply-one", Path: _P("list", _KBF("name", "a"), "value")},
					},
				},
			},
			Object: `
				list:
				- name: a
				  value: 0
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "a")),
						_P("list", _KBF("name", "a"), "name"),
						_P("list", _KBF("name", "a"), "value"),
					),
					"v1",
					false,
				),
			},
		},
		"remove_one_keep_one": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						list:
						- name: a
						- name: b
						- name: c
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v2",
					Object: `
						list:
						- name: c
						- name: d
					`,
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: "v3",
					Object: `
						list:
						- name: a
					`,
				},
			},
			Object: `
				list:
				- name: a
				- name: c
				- name: d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "a")),
						_P("list", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", "c")),
						_P("list", _KBF("name", "d")),
						_P("list", _KBF("name", "c"), "name"),
						_P("list", _KBF("name", "d"), "name"),
					),
					"v2",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(associativeListParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}

var structMultiversionParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: struct
        type:
          namedType: struct
      - name: version
        type:
          scalar: string
- name: struct
  map:
    fields:
      - name: name
        type:
          scalar: string
      - name: scalarField_v1
        type:
          scalar: string
      - name: complexField_v1
        type:
          namedType: complex
- name: complex
  map:
    fields:
      - name: name
        type:
          scalar: string
- name: v2
  map:
    fields:
      - name: struct
        type:
          namedType: struct_v2
      - name: version
        type:
          scalar: string
- name: struct_v2
  map:
    fields:
      - name: name
        type:
          scalar: string
      - name: scalarField_v2
        type:
          scalar: string
      - name: complexField_v2
        type:
          namedType: complex_v2
- name: complex_v2
  map:
    fields:
      - name: name
        type:
          scalar: string
- name: v3
  map:
    fields:
      - name: struct
        type:
          namedType: struct_v3
      - name: version
        type:
          scalar: string
- name: struct_v3
  map:
    fields:
      - name: name
        type:
          scalar: string
      - name: scalarField_v3
        type:
          scalar: string
      - name: complexField_v3
        type:
          namedType: complex_v3
- name: complex_v3
  map:
    fields:
      - name: name
        type:
          scalar: string
`)
	if err != nil {
		panic(err)
	}
	return parser
}()

func TestMultipleAppliersFieldUnsetting(t *testing.T) {
	versions := []fieldpath.APIVersion{"v1", "v2", "v3"}
	for _, v1 := range versions {
		for _, v2 := range versions {
			for _, v3 := range versions {
				t.Run(fmt.Sprintf("%s-%s-%s", v1, v2, v3), func(t *testing.T) {
					testMultipleAppliersFieldUnsetting(t, v1, v2, v3)
				})
			}
		}
	}
}

func testMultipleAppliersFieldUnsetting(t *testing.T, v1, v2, v3 fieldpath.APIVersion) {
	tests := map[string]TestCase{
		"unset_scalar_sole_owner": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v1)),
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: v2,
					Object: `
						struct:
						  name: a
					`,
				},
			},
			Object: `
				struct:
				  name: a
			`,
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v2,
					false,
				),
			},
		},
		"unset_scalar_shared_with_applier": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v1)),
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: v2,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  scalarField_%s: a
					`, v2)),
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: v3,
					Object: `
						struct:
						  name: a
					`,
				},
			},
			Object: typed.YAMLObject(fmt.Sprintf(`
				struct:
				  name: a
				  scalarField_%s: a
			`, v3)),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v3,
					true,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", fmt.Sprintf("scalarField_%s", v2)),
					),
					v2,
					false,
				),
			},
		},
		"unset_scalar_shared_with_updater": {
			Ops: []Operation{
				Update{
					Manager:    "updater",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v1)),
				},
				Apply{
					Manager:    "applier",
					APIVersion: v2,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v2)),
				},
				Apply{
					Manager:    "applier",
					APIVersion: v3,
					Object: `
						struct:
						  name: a
					`,
				},
			},
			Object: typed.YAMLObject(fmt.Sprintf(`
				struct:
				  name: a
				  scalarField_%s: a
			`, v3)),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"updater": fieldpath.NewVersionedSet(
					_NS(
						_P("struct"),
						_P("struct", "name"),
						_P("struct", fmt.Sprintf("scalarField_%s", v1)),
					),
					v1,
					false,
				),
				"applier": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v3,
					true,
				),
			},
		},
		"updater_claims_field": {
			Ops: []Operation{
				Apply{
					Manager:    "applier",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v1)),
				},
				Update{
					Manager:    "updater",
					APIVersion: v2,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: b
					`, v2)),
				},
			},
			Object: typed.YAMLObject(fmt.Sprintf(`
				struct:
				  name: a
				  scalarField_%s: b
			`, v3)),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"updater": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", fmt.Sprintf("scalarField_%s", v2)),
					),
					v2,
					false,
				),
				"applier": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v1,
					true,
				),
			},
		},
		"unset_complex_sole_owner": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  complexField_%s:
						    name: b
					`, v1)),
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: v2,
					Object: `
						struct:
						  name: a
					`,
				},
			},
			Object: typed.YAMLObject(`
				struct:
				  name: a
			`),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v2,
					false,
				),
			},
		},
		"unset_complex_shared_with_applier": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  complexField_%s:
						    name: b
					`, v1)),
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: v2,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  complexField_%s:
						    name: b
					`, v2)),
				},
				Apply{
					Manager:    "apply-one",
					APIVersion: v3,
					Object: `
						struct:
						  name: a
					`,
				},
			},
			Object: typed.YAMLObject(fmt.Sprintf(`
				struct:
				  name: a
				  complexField_%s:
				    name: b
			`, v3)),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v3,
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", fmt.Sprintf("complexField_%s", v2), "name"),
					),
					v2,
					false,
				),
			},
		},
	}

	converter := renamingConverter{structMultiversionParser}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.TestWithConverter(structMultiversionParser, converter); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMultipleAppliersNestedType(t *testing.T) {
	tests := map[string]TestCase{
		"remove_one_keep_one_with_two_sub_items": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						listOfLists:
						- name: b
						  value:
						  - d
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				listOfLists:
				- name: a
				- name: b
				  value:
				  - d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "b")),
						_P("listOfLists", _KBF("name", "b"), "name"),
						_P("listOfLists", _KBF("name", "b"), "value", _V("d")),
					),
					"v2",
					false,
				),
			},
		},
		"remove_one_keep_one_with_dangling_subitem": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						listOfLists:
						- name: b
						  value:
						  - d
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
						  - d
						  - e
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				listOfLists:
				- name: a
				- name: b
				  value:
				  - d
				  - e
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "b")),
						_P("listOfLists", _KBF("name", "b"), "name"),
						_P("listOfLists", _KBF("name", "b"), "value", _V("d")),
					),
					"v2",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "b"), "value", _V("e")),
					),
					"v2",
					false,
				),
			},
		},
		"remove_one_with_dangling_subitem_keep_one": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
						- name: b
						  value:
						  - c
						  - d
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				listOfLists:
				- name: a
				  value:
				  - b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
						_P("listOfLists", _KBF("name", "a"), "value", _V("b")),
					),
					"v2",
					false,
				),
			},
		},
		"remove_one_with_managed_subitem_keep_one": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller",
					Object: `
						listOfLists:
						- name: a
						  value:
						  - b
						- name: b
						  value:
						  - c
						  - d
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				listOfLists:
				- name: a
				  value:
				  - b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
						_P("listOfLists", _KBF("name", "a"), "value", _V("b")),
					),
					"v2",
					false,
				),
			},
		},
		"remove_one_keep_one_with_sub_item": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
						- name: b
						  value:
						  - c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						listOfLists:
						- name: b
						  value:
						  - d
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						listOfLists:
						- name: a
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				listOfLists:
				- name: a
				- name: b
				  value:
				  - d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "a")),
						_P("listOfLists", _KBF("name", "a"), "name"),
					),
					"v3",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("listOfLists", _KBF("name", "b")),
						_P("listOfLists", _KBF("name", "b"), "name"),
						_P("listOfLists", _KBF("name", "b"), "value", _V("d")),
					),
					"v2",
					false,
				),
			},
		},
		"multiple_appliers_recursive_map": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						  c:
						    d:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						mapOfMapsRecursive:
						  a:
						  c:
						    d:
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller-one",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
						  c:
						    d:
						      e:
					`,
					APIVersion: "v3",
				},
				Update{
					Manager: "controller-two",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
						        d:
						  c:
						    d:
						      e:
						        f:
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller-one",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
						        d:
						          e:
						  c:
						    d:
						      e:
						        f:
						          g:
					`,
					APIVersion: "v3",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						mapOfMapsRecursive:
					`,
					APIVersion: "v4",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  a:
				  c:
				    d:
				      e:
				        f:
				          g:
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive"),
					),
					"v4",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "c"),
						_P("mapOfMapsRecursive", "c", "d"),
					),
					"v2",
					false,
				),
				"controller-one": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "c", "d", "e"),
						_P("mapOfMapsRecursive", "c", "d", "e", "f", "g"),
					),
					"v3",
					false,
				),
				"controller-two": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "c", "d", "e", "f"),
					),
					"v2",
					false,
				),
			},
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

func TestMultipleAppliersDeducedType(t *testing.T) {
	tests := map[string]TestCase{
		"multiple_appliers_recursive_map_deduced": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						a:
						  b:
						c:
						  d:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						a:
						c:
						  d:
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller-one",
					Object: `
						a:
						  b:
						    c:
						c:
						  d:
						    e:
					`,
					APIVersion: "v3",
				},
				Update{
					Manager: "controller-two",
					Object: `
						a:
						  b:
						    c:
						      d:
						c:
						  d:
						    e:
						      f:
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller-one",
					Object: `
						a:
						  b:
						    c:
						      d:
						        e:
						c:
						  d:
						    e:
						      f:
						        g:
					`,
					APIVersion: "v3",
				},
				Apply{
					Manager:    "apply-one",
					Object:     ``,
					APIVersion: "v4",
				},
			},
			Object: `
				a:
				c:
				  d:
				    e:
				      f:
				        g:
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("a"),
						_P("c"),
						_P("c", "d"),
					),
					"v2",
					false,
				),
				"controller-one": fieldpath.NewVersionedSet(
					_NS(
						_P("c", "d", "e"),
						_P("c", "d", "e", "f", "g"),
					),
					"v3",
					false,
				),
				"controller-two": fieldpath.NewVersionedSet(
					_NS(
						_P("c", "d", "e", "f"),
					),
					"v2",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(DeducedParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMultipleAppliersRealConversion(t *testing.T) {
	tests := map[string]TestCase{
		"multiple_appliers_recursive_map_real_conversion": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						  c:
						    d:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						mapOfMapsRecursive:
						  aa:
						  cc:
						    dd:
					`,
					APIVersion: "v2",
				},
				Update{
					Manager: "controller",
					Object: `
						mapOfMapsRecursive:
						  aaa:
						    bbb:
						      ccc:
						        ddd:
						  ccc:
						    ddd:
						      eee:
						        fff:
					`,
					APIVersion: "v3",
				},
				Apply{
					Manager: "apply-one",
					Object: `
						mapOfMapsRecursive:
					`,
					APIVersion: "v4",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  aaaa:
				  cccc:
				    dddd:
				      eeee:
				        ffff:
			`,
			APIVersion: "v4",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive"),
					),
					"v4",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "aa"),
						_P("mapOfMapsRecursive", "cc"),
						_P("mapOfMapsRecursive", "cc", "dd"),
					),
					"v2",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "ccc", "ddd", "eee"),
						_P("mapOfMapsRecursive", "ccc", "ddd", "eee", "fff"),
					),
					"v3",
					false,
				),
			},
		},
		"appliers_remove_from_controller_real_conversion": {
			// Ensures that an applier can delete associative map items it created after a controller
			// modifies them.
			Ops: []Operation{
				Apply{
					Manager: "apply",
					Object: `
						mapOfMapsRecursive:
						  aaa:
						    bbb:
					`,
					APIVersion: "v3",
				},
				Update{
					Manager: "controller",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply",
					Object: `
						mapOfMapsRecursive:
						  aa:
						    bb:
						  cc:
						    dd:
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply",
					Object: `
						mapOfMapsRecursive:
						  aaa:
						  ccc:
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  aaa:
				  ccc:
			`,
			APIVersion: "v3",
			Managed: fieldpath.ManagedFields{
				"apply": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "aaa"),
						_P("mapOfMapsRecursive", "ccc"),
					),
					"v3",
					false,
				),
			},
		},
		"applier_updater_shared_ownership_real_conversion": {
			// Ensures that when an updater creates maps that they are not deleted when
			// an applier shares ownership in them and then later removes them from its applied
			// configuration
			Ops: []Operation{
				Update{
					Manager: "updater",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						      c:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply",
					Object: `
						mapOfMapsRecursive:
						  aa:
						    bb:
						  cc:
						    dd:
					`,
					APIVersion: "v2",
				},
				Apply{
					Manager: "apply",
					Object: `
						mapOfMapsRecursive:
						  aaa:
						  ccc:
					`,
					APIVersion: "v3",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  aaa:
				    bbb:
				      ccc:
				  ccc:
			`,
			APIVersion: "v3",
			Managed: fieldpath.ManagedFields{
				"updater": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive"),
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "b"),
						_P("mapOfMapsRecursive", "a", "b", "c"),
					),
					"v1",
					false,
				),
				"apply": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "aaa"),
						_P("mapOfMapsRecursive", "ccc"),
					),
					"v3",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.TestWithConverter(nestedTypeParser, repeatingConverter{nestedTypeParser}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestMultipleAppliersFieldRenameConversions(t *testing.T) {
	versions := []fieldpath.APIVersion{"v1", "v2", "v3"}
	for _, v1 := range versions {
		for _, v2 := range versions {
			for _, v3 := range versions {
				t.Run(fmt.Sprintf("%s-%s-%s", v1, v2, v3), func(t *testing.T) {
					testMultipleAppliersFieldRenameConversions(t, v1, v2, v3)
				})
			}
		}
	}
}

func testMultipleAppliersFieldRenameConversions(t *testing.T, v1, v2, v3 fieldpath.APIVersion) {
	tests := map[string]TestCase{
		"updater_claims_field": {
			Ops: []Operation{
				Apply{
					Manager:    "applier",
					APIVersion: v1,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: a
					`, v1)),
				},
				Update{
					Manager:    "updater",
					APIVersion: v2,
					Object: typed.YAMLObject(fmt.Sprintf(`
						struct:
						  name: a
						  scalarField_%s: b
					`, v2)),
				},
			},
			Object: typed.YAMLObject(fmt.Sprintf(`
				struct:
				  name: a
				  scalarField_%s: b
			`, v3)),
			APIVersion: v3,
			Managed: fieldpath.ManagedFields{
				"updater": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", fmt.Sprintf("scalarField_%s", v2)),
					),
					v2,
					false,
				),
				"applier": fieldpath.NewVersionedSet(
					_NS(
						_P("struct", "name"),
					),
					v1,
					true,
				),
			},
		},
	}

	converter := renamingConverter{structMultiversionParser}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.TestWithConverter(structMultiversionParser, converter); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// repeatingConverter repeats a single letterkey v times, where v is the version.
type repeatingConverter struct {
	parser Parser
}

var _ merge.Converter = repeatingConverter{}

var missingVersionError error = fmt.Errorf("cannot convert to invalid version")

// Convert implements merge.Converter
func (r repeatingConverter) Convert(v *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	if len(version) < 2 || string(version)[0] != 'v' {
		return nil, missingVersionError
	}
	versionNumber, err := strconv.Atoi(string(version)[1:len(version)])
	if err != nil {
		return nil, missingVersionError
	}
	y, err := yaml.Marshal(v.AsValue().Unstructured())
	if err != nil {
		return nil, err
	}
	str := string(y)
	var str2 string
	for i, line := range strings.Split(str, "\n") {
		if i == 0 {
			str2 = line
		} else {
			spaces := strings.Repeat(" ", countLeadingSpace(line))
			if len(spaces) == 0 {
				break
			}
			c := line[len(spaces) : len(spaces)+1]
			c = strings.Repeat(c, versionNumber)
			str2 = fmt.Sprintf("%v\n%v%v:", str2, spaces, c)
		}
	}
	v2, err := r.parser.Type(string(version)).FromYAML(typed.YAMLObject(str2))
	if err != nil {
		return nil, err
	}
	return v2, nil
}

func countLeadingSpace(line string) int {
	spaces := 0
	for _, letter := range line {
		if letter == ' ' {
			spaces++
		} else {
			break
		}
	}
	return spaces
}

// Convert implements merge.Converter
func (r repeatingConverter) IsMissingVersionError(err error) bool {
	return err == missingVersionError
}

// renamingConverter renames fields by substituting the version suffix of the field name. E.g.
// converting a map  with a field named "name_v1" from v1 to v2 renames the field to "name_v2".
// Fields without a version suffix are not converted; they are the same in all versions.
// When parsing, this converter will look for the type by using the APIVersion of the
// object it's trying to parse. If trying to parse a "v1" object, a corresponding "v1" type
// should exist in the schema of the provided parser.
type renamingConverter struct {
	parser Parser
}

// Convert implements merge.Converter
func (r renamingConverter) Convert(v *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	inVersion := fieldpath.APIVersion(*v.TypeRef().NamedType)
	outType := r.parser.Type(string(version))
	return outType.FromUnstructured(renameFields(v.AsValue(), string(inVersion), string(version)))
}

func renameFields(v value.Value, oldSuffix, newSuffix string) interface{} {
	if v.IsMap() {
		out := map[string]interface{}{}
		v.AsMap().Iterate(func(key string, value value.Value) bool {
			if strings.HasSuffix(key, oldSuffix) {
				out[strings.TrimSuffix(key, oldSuffix)+newSuffix] = renameFields(value, oldSuffix, newSuffix)
			} else {
				out[key] = renameFields(value, oldSuffix, newSuffix)
			}
			return true
		})
		return out
	}
	if v.IsList() {
		var out []interface{}
		ri := v.AsList().Range()
		for ri.Next() {
			_, v := ri.Item()
			out = append(out, renameFields(v, oldSuffix, newSuffix))
		}
		return out
	}
	return v.Unstructured()
}

// Convert implements merge.Converter
func (r renamingConverter) IsMissingVersionError(err error) bool {
	return err == missingVersionError
}

var atomicMapParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: atomicMap
        type:
          namedType: atomicMap
- name: atomicMap
  map:
    fields:
      - name: field1
        type:
          scalar: string
      - name: field2
        type:
          scalar: string
    elementRelationship: atomic
`)
	if err != nil {
		panic(err)
	}
	return parser
}()

func TestMultipleApplierAtomicMaps(t *testing.T) {
	tests := map[string]TestCase{
		"force": {
			Ops: []Operation{
				Apply{
					Manager:    "apply-one",
					APIVersion: "v1",
					Object: `
						atomicMap:
						  field1: a
					`,
				},
				Apply{
					Manager:    "apply-two",
					APIVersion: "v1",
					Object: `
						atomicMap:
						  field2: b
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "apply-one", Path: _P("atomicMap")},
					},
				},
				ForceApply{
					Manager:    "apply-two",
					APIVersion: "v1",
					Object: `
						atomicMap:
						  field2: b
					`,
				},
			},
			Object: `
				atomicMap:
				  field2: b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("atomicMap"),
					),
					"v1",
					false,
				),
			},
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(atomicMapParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func BenchmarkMultipleApplierRecursiveRealConversion(b *testing.B) {
	test := TestCase{
		Ops: []Operation{
			Apply{
				Manager: "apply-one",
				Object: `
					mapOfMapsRecursive:
					  a:
					    b:
					  c:
					    d:
				`,
				APIVersion: "v1",
			},
			Apply{
				Manager: "apply-two",
				Object: `
					mapOfMapsRecursive:
					  aa:
					  cc:
					    dd:
				`,
				APIVersion: "v2",
			},
			Update{
				Manager: "controller",
				Object: `
					mapOfMapsRecursive:
					  aaa:
					    bbb:
					      ccc:
					        ddd:
					  ccc:
					    ddd:
					      eee:
					        fff:
					`,
				APIVersion: "v3",
			},
			Apply{
				Manager: "apply-one",
				Object: `
					mapOfMapsRecursive:
				`,
				APIVersion: "v4",
			},
		},
		Object: `
			mapOfMapsRecursive:
			  aaaa:
			  cccc:
			    dddd:
			      eeee:
			        ffff:
		`,
		APIVersion: "v4",
		Managed: fieldpath.ManagedFields{
			"apply-one": fieldpath.NewVersionedSet(
				_NS(
					_P("mapOfMapsRecursive"),
				),
				"v4",
				false,
			),
			"apply-two": fieldpath.NewVersionedSet(
				_NS(
					_P("mapOfMapsRecursive", "aa"),
					_P("mapOfMapsRecursive", "cc"),
					_P("mapOfMapsRecursive", "cc", "dd"),
				),
				"v2",
				false,
			),
			"controller": fieldpath.NewVersionedSet(
				_NS(
					_P("mapOfMapsRecursive", "ccc", "ddd", "eee"),
					_P("mapOfMapsRecursive", "ccc", "ddd", "eee", "fff"),
				),
				"v3",
				false,
			),
		},
	}

	// Make sure this passes...
	if err := test.TestWithConverter(nestedTypeParser, repeatingConverter{nestedTypeParser}); err != nil {
		b.Fatal(err)
	}

	test.PreprocessOperations(nestedTypeParser)

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if err := test.BenchWithConverter(nestedTypeParser, repeatingConverter{nestedTypeParser}); err != nil {
			b.Fatal(err)
		}
	}
}
