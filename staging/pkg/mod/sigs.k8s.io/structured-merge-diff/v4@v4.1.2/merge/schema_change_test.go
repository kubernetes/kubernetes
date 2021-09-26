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
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

var structParser = func() *typed.Parser {
	oldParser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: struct
        type:
          namedType: struct
- name: struct
  map:
    fields:
    - name: numeric
      type:
        scalar: numeric
    - name: string
      type:
        scalar: string`)
	if err != nil {
		panic(err)
	}
	return oldParser
}()

var structWithAtomicParser = func() *typed.Parser {
	newParser, err := typed.NewParser(`types:
- name: v1
  map:
    fields:
      - name: struct
        type:
          namedType: struct
- name: struct
  map:
    fields:
    - name: numeric
      type:
        scalar: numeric
    - name: string
      type:
        scalar: string
    elementRelationship: atomic`)
	if err != nil {
		panic(err)
	}
	return newParser
}()

func TestGranularToAtomicSchemaChanges(t *testing.T) {
	tests := map[string]TestCase{
		"to-atomic": {
			Ops: []Operation{
				Apply{
					Manager: "one",
					Object: `
						struct:
						  numeric: 1
					`,
					APIVersion: "v1",
				},
				ChangeParser{Parser: structWithAtomicParser},
				Apply{
					Manager: "two",
					Object: `
						struct:
						  string: "string"
					`,
					APIVersion: "v1",
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "one", Path: _P("struct")},
					},
				},
				ForceApply{
					Manager: "two",
					Object: `
						struct:
						  string: "string"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  string: "string"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"two": fieldpath.NewVersionedSet(_NS(
					_P("struct"),
				), "v1", true),
			},
		},
		"to-atomic-owner-with-no-child-fields": {
			Ops: []Operation{
				Apply{
					Manager: "one",
					Object: `
						struct:
						  numeric: 1
					`,
					APIVersion: "v1",
				},
				ForceApply{ // take the only child field from manager "one"
					Manager: "two",
					Object: `
						struct:
						  numeric: 2
					`,
					APIVersion: "v1",
				},
				ChangeParser{Parser: structWithAtomicParser},
				Apply{
					Manager: "three",
					Object: `
						struct:
						  string: "string"
					`,
					APIVersion: "v1",
					Conflicts: merge.Conflicts{
						// We expect no conflict with "one" because we do not allow a manager
						// to own a map without owning any of the children.
						merge.Conflict{Manager: "two", Path: _P("struct")},
					},
				},
				ForceApply{
					Manager: "two",
					Object: `
						struct:
						  string: "string"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  string: "string"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"two": fieldpath.NewVersionedSet(_NS(
					_P("struct"),
				), "v1", true),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(structParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestAtomicToGranularSchemaChanges(t *testing.T) {
	tests := map[string]TestCase{
		"to-granular": {
			Ops: []Operation{
				Apply{
					Manager: "one",
					Object: `
						struct:
						  numeric: 1
						  string: "a"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "two",
					Object: `
						struct:
						  string: "b"
					`,
					APIVersion: "v1",
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "one", Path: _P("struct")},
					},
				},
				ChangeParser{Parser: structParser},
				// No conflict after changing struct to a granular schema
				Apply{
					Manager: "two",
					Object: `
						struct:
						  string: "b"
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				struct:
				  numeric: 1
				  string: "b"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				// Note that manager one previously owned
				// the top level _P("struct")
				// which included all of its subfields
				// when the struct field was atomic.
				//
				// Upon changing the schema of struct from
				// atomic to granular, manager one continues
				// to own the same fieldset as before,
				// but does not retain ownership of any of the subfields.
				//
				// This is a known limitation due to the inability
				// to accurately determine whether an empty field
				// was previously atomic or not.
				"one": fieldpath.NewVersionedSet(_NS(
					_P("struct"),
				), "v1", true),
				"two": fieldpath.NewVersionedSet(_NS(
					_P("struct", "string"),
				), "v1", true),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(structWithAtomicParser); err != nil {
				t.Fatal(err)
			}
		})
	}
}
