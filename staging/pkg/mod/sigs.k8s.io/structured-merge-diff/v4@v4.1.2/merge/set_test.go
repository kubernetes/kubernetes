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

package merge_test

import (
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var setFieldsParser = func() Parser {
	parser, err := typed.NewParser(`types:
- name: sets
  map:
    fields:
    - name: list
      type:
        list:
          elementType:
            scalar: string
          elementRelationship: associative`)
	if err != nil {
		panic(err)
	}
	return SameVersionParser{T: parser.Type("sets")}
}()

func TestUpdateSet(t *testing.T) {
	tests := map[string]TestCase{
		"apply_twice": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
			},
			Object: `
				list:
				- a
				- b
				- c
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
						_P("list", _V("d")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_no_overlap": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- aprime
						- c
						- cprime
					`,
				},
			},
			Object: `
				list:
				- a
				- aprime
				- b
				- c
				- cprime
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("aprime")),
						_P("list", _V("c")),
						_P("list", _V("cprime")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
						_P("list", _V("d")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_no_overlap_and_different_version": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v2",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- aprime
						- c
						- cprime
					`,
				},
			},
			Object: `
				list:
				- a
				- aprime
				- b
				- c
				- cprime
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("aprime")),
						_P("list", _V("c")),
						_P("list", _V("cprime")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
						_P("list", _V("d")),
					),
					"v2",
					false,
				),
			},
		},
		"apply_update_apply_with_overlap": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
					`,
				},
			},
			Object: `
				list:
				- a
				- b
				- c
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
						_P("list", _V("d")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_with_overlap_and_different_version": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v2",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
					`,
				},
			},
			Object: `
				list:
				- a
				- b
				- c
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("b")),
						_P("list", _V("d")),
					),
					"v2",
					false,
				),
			},
		},
		"apply_twice_reorder": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- d
						- c
						- b
					`,
				},
			},
			Object: `
				list:
				- a
				- d
				- c
				- b
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
						_P("list", _V("d")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_reorder": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- d
						- c
						- b
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
			},
			Object: `
				list:
				- a
				- b
				- c
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
						_P("list", _V("d")),
					),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_reorder_across_versions": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- d
						- c
						- b
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v2",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
			},
			Object: `
				list:
				- a
				- b
				- c
				- d
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("b")),
						_P("list", _V("c")),
						_P("list", _V("d")),
					),
					"v2",
					false,
				),
			},
		},
		"apply_twice_remove": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- c
					`,
				},
			},
			Object: `
				list:
				- a
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
			},
		},
		"apply_twice_remove_across_versions": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						list:
						- a
						- b
						- c
						- d
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v2",
					Object: `
						list:
						- a
						- c
						- e
					`,
				},
			},
			Object: `
				list:
				- a
				- c
				- e
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _V("a")),
						_P("list", _V("c")),
						_P("list", _V("e")),
					),
					"v2",
					false,
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(setFieldsParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
