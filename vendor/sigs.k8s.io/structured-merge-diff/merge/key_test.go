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

	"sigs.k8s.io/structured-merge-diff/fieldpath"
	. "sigs.k8s.io/structured-merge-diff/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/typed"
)

var associativeListParser = func() typed.ParseableType {
	parser, err := typed.NewParser(`types:
- name: type
  map:
    fields:
      - name: list
        type:
          namedType: associativeList
- name: associativeList
  list:
    elementType:
      namedType: myElement
    elementRelationship: associative
    keys:
    - name
- name: myElement
  map:
    fields:
    - name: name
      type:
        scalar: string
    - name: value
      type:
        scalar: numeric
`)
	if err != nil {
		panic(err)
	}
	return parser.Type("type")
}()

func TestUpdateAssociativeLists(t *testing.T) {
	tests := map[string]TestCase{
		"removing_obsolete_applied_structs": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						list:
						- name: a
						  value: 1
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						list:
						- name: b
						  value: 2
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				list:
				- name: b
				  value: 2
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("list", _KBF("name", _SV("b"))),
						_P("list", _KBF("name", _SV("b")), "name"),
						_P("list", _KBF("name", _SV("b")), "value"),
					),
					"v1",
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
