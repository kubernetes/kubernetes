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

var extractParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: sets
  map:
    fields:
    - name: list
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: associative
    - name: atomicList
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: atomic
    - name: map
      type:
        map:
          elementType:
            scalar: string
          elementRelationship: separable
    - name: atomicMap
      type:
        map:
          elementType:
            scalar: string
          elementRelationship: atomic`)
	if err != nil {
		panic(err)
	}
	return SameVersionParser{T: parser.Type("sets")}
}()

func TestExtractApply(t *testing.T) {
	tests := map[string]TestCase{
		"apply_one_extract_apply_one_own_both": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "default",
					Object: `
							list:
							- b
						`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_from_beginning": {
			Ops: []Operation{
				ExtractApply{
					Manager: "default",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "default",
					Object: `
							list:
							- b
						`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_after_extract_remove_fields": {
			Ops: []Operation{
				ExtractApply{
					Manager: "default",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						list:
						- b
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_one_controller_remove_extract_apply_one": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						list:
						 - b
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "default",
					Object: `
						list:
						- c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_retain_ownership_after_controller_update": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "controller",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "default",
					Object: `
						list:
						- c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_share_ownership_after_another_apply": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
						- c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_two_cant_delete_object_also_owned_by_extract_apply": {
			Ops: []Operation{
				Apply{
					Manager: "apply-one",
					Object: `
						list:
						- a
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
						- c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - b
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_list": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- a
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_remove_list": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - b
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_add_later_list": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - a
						 - b
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "apply-one",
					Object: `
						list:
						- c
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						list:
						 - b
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_map": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						map:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						map:
						 a: c
						 b: d
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				map:
				  a: c
				  b: d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("map"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("map", "a"),
						_P("map", "b"),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_remove_map": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						map:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						map:
						 a: c
						 b: d
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						map:
						 b: d
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				map:
				  b: d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("map"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("map", "b"),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_empty_structure_add_later_map": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						map:
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						map:
						 a: c
						 b: d
					`,
					APIVersion: "v1",
				},
				ExtractApply{
					Manager: "apply-one",
					Object: `
						map:
						  e: f
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "apply-two",
					Object: `
						map:
						 b: d
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				map:
				  b: d
				  e: f
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("map", "e"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("map", "b"),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_atomic_list": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						atomicList:
						- a
						- b
						- c
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				atomicList:
				- a
				- b
				- c
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("atomicList"),
					),
					"v1",
					false,
				),
			},
		},
		"extract_apply_atomic_map": {
			Ops: []Operation{
				ExtractApply{
					Manager: "apply-one",
					Object: `
						atomicMap:
						 a: c
						 b: d
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				atomicMap:
				 a: c
				 b: d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
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
			if err := test.Test(extractParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
