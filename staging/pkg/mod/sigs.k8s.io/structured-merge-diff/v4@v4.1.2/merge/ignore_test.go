/*
Copyright 2020 The Kubernetes Authors.

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
)

func TestIgnoredFields(t *testing.T) {
	tests := map[string]TestCase{
		"update_does_not_own_ignored": {
			APIVersion: "v1",
			Ops: []Operation{
				Update{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "some string"
					`,
				},
			},
			Object: `
				numeric: 1
				string: "some string"
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"),
					),
					"v1",
					false,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v1": _NS(
					_P("string"),
				),
			},
		},
		"update_does_not_own_deep_ignored": {
			APIVersion: "v1",
			Ops: []Operation{
				Update{
					Manager:    "default",
					APIVersion: "v1",
					Object:     `{"numeric": 1, "obj": {"string": "foo", "numeric": 2}}`,
				},
			},
			Object: `{"numeric": 1, "obj": {"string": "foo", "numeric": 2}}`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"),
					),
					"v1",
					false,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v1": _NS(
					_P("obj"),
				),
			},
		},
		"apply_does_not_own_ignored": {
			APIVersion: "v1",
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "some string"
					`,
				},
			},
			Object: `
				numeric: 1
				string: "some string"
			`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"),
					),
					"v1",
					true,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v1": _NS(
					_P("string"),
				),
			},
		},
		"apply_does_not_own_deep_ignored": {
			APIVersion: "v1",
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object:     `{"numeric": 1, "obj": {"string": "foo", "numeric": 2}}`,
				},
			},
			Object: `{"numeric": 1, "obj": {"string": "foo", "numeric": 2}}`,
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"),
					),
					"v1",
					true,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v1": _NS(
					_P("obj"),
				),
			},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			if err := test.Test(DeducedParser); err != nil {
				t.Fatal("Should fail:", err)
			}
		})
	}
}

func TestIgnoredFieldsUsesVersions(t *testing.T) {
	tests := map[string]TestCase{
		"does_use_ignored_fields_versions": {
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
				Apply{
					Manager: "apply-one",
					Object: `
						mapOfMapsRecursive:
					`,
					APIVersion: "v4",
				},
			},
			// note that this still contains cccc due to ignored fields not being removed from the update result
			Object: `
				mapOfMapsRecursive:
				  aaaa:
				  cccc:
				    dddd:
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
					),
					"v2",
					false,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v1": _NS(
					_P("mapOfMapsRecursive", "c"),
				),
				"v2": _NS(
					_P("mapOfMapsRecursive", "cc"),
				),
				"v3": _NS(
					_P("mapOfMapsRecursive", "ccc"),
				),
				"v4": _NS(
					_P("mapOfMapsRecursive", "cccc"),
				),
			},
		},
		"update_does_not_steal_ignored": {
			APIVersion: "v1",
			Ops: []Operation{
				Update{
					Manager: "update-one",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						  c:
						    d:
					`,
					APIVersion: "v1",
				},
				Update{
					Manager: "update-two",
					Object: `
						mapOfMapsRecursive:
						  a:
						    b:
						  c:
						    e:
					`,
					APIVersion: "v2",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  a:
				    b:
				  c:
				    e:
			`,
			Managed: fieldpath.ManagedFields{
				"update-one": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive"),
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "b"),
						_P("mapOfMapsRecursive", "c"),
					),
					"v1",
					false,
				),
				"update-two": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "b"),
					),
					"v2",
					false,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v2": _NS(
					_P("mapOfMapsRecursive", "c"),
				),
			},
		},
		"apply_does_not_steal_ignored": {
			APIVersion: "v1",
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
						    b:
						  c:
						    e:
					`,
					APIVersion: "v2",
				},
			},
			Object: `
				mapOfMapsRecursive:
				  a:
				    b:
				  c:
				    d:
			`,
			Managed: fieldpath.ManagedFields{
				"apply-one": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "b"),
						_P("mapOfMapsRecursive", "c"),
						_P("mapOfMapsRecursive", "c", "d"),
					),
					"v1",
					false,
				),
				"apply-two": fieldpath.NewVersionedSet(
					_NS(
						_P("mapOfMapsRecursive", "a"),
						_P("mapOfMapsRecursive", "a", "b"),
					),
					"v2",
					false,
				),
			},
			IgnoredFields: map[fieldpath.APIVersion]*fieldpath.Set{
				"v2": _NS(
					_P("mapOfMapsRecursive", "c"),
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
