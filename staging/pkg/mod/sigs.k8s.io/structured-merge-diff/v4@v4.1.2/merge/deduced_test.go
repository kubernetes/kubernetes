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
	"sigs.k8s.io/structured-merge-diff/v4/merge"
)

func TestDeduced(t *testing.T) {
	tests := map[string]TestCase{
		"leaf_apply_twice": {
			Ops: []Operation{
				Apply{
					Manager: "default",
					Object: `
						numeric: 1
						string: "string"
					`,
					APIVersion: "v1",
				},
				Apply{
					Manager: "default",
					Object: `
						numeric: 2
						string: "string"
						bool: false
					`,
					APIVersion: "v1",
				},
			},
			Object: `
				numeric: 2
				string: "string"
				bool: false
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"), _P("string"), _P("bool"),
					),
					"v1",
					false,
				),
			},
		},
		"leaf_apply_update_apply_no_conflict": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "string"
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "string"
						bool: true
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 2
						string: "string"
					`,
				},
			},
			Object: `
				numeric: 2
				string: "string"
				bool: true
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"), _P("string"),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("bool"),
					),
					"v1",
					false,
				),
			},
		},
		"leaf_apply_update_apply_with_conflict": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "string"
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "controller string"
						bool: true
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 2
						string: "user string"
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "controller", Path: _P("string")},
					},
				},
				ForceApply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 2
						string: "user string"
					`,
				},
			},
			Object: `
				numeric: 2
				string: "user string"
				bool: true
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("numeric"), _P("string"),
					),
					"v1",
					false,
				),
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("bool"),
					),
					"v1",
					false,
				),
			},
		},
		"leaf_apply_twice_remove": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						numeric: 1
						string: "string"
						bool: false
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						string: "new string"
					`,
				},
			},
			Object: `
				string: "new string"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(
						_P("string"),
					),
					"v1",
					false,
				),
			},
		},
		"leaf_update_remove_empty_set": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						string: "string"
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						string: "new string"
					`,
				},
			},
			Object: `
				string: "new string"
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"controller": fieldpath.NewVersionedSet(
					_NS(
						_P("string"),
					),
					"v1",
					false,
				),
			},
		},
		"apply_twice_list_is_atomic": {
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
					_NS(_P("list")),
					"v1",
					false,
				),
			},
		},
		"apply_update_apply_list": {
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
				ForceApply{
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
			`,
			APIVersion: "v1",
			Managed: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("list")),
					"v1",
					false,
				),
			},
		},
		"leaf_apply_remove_empty_set": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						string: "string"
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object:     ``,
				},
			},
			Object:     ``,
			APIVersion: "v1",
			Managed:    fieldpath.ManagedFields{},
		},
		"apply_update_apply_nested": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						a: 1
						b:
						  c:
						    d: 2
						    e:
						    - 1
						    - 2
						    - 3
						    f:
						    - name: n
						      value: 1
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v1",
					Object: `
						a: 1
						b:
						  c:
						    d: 3
						    e:
						    - 1
						    - 2
						    - 3
						    - 4
						    f:
						    - name: n
						      value: 2
						g: 5
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "d")},
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "e")},
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "f")},
					},
				},
				ForceApply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
				},
			},
			Object: `
				a: 2
				b:
				  c:
				    d: 2
				    e:
				    - 3
				    - 2
				    - 1
				    f:
				    - name: n
				      value: 1
				g: 5
			`,
			APIVersion: "v1",
		},
		"apply_update_apply_nested_different_version": {
			Ops: []Operation{
				Apply{
					Manager:    "default",
					APIVersion: "v1",
					Object: `
						a: 1
						b:
						  c:
						    d: 2
						    e:
						    - 1
						    - 2
						    - 3
						    f:
						    - name: n
						      value: 1
					`,
				},
				Update{
					Manager:    "controller",
					APIVersion: "v2",
					Object: `
						a: 1
						b:
						  c:
						    d: 3
						    e:
						    - 1
						    - 2
						    - 3
						    - 4
						    f:
						    - name: n
						      value: 2
						g: 5
					`,
				},
				Apply{
					Manager:    "default",
					APIVersion: "v3",
					Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
					Conflicts: merge.Conflicts{
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "d")},
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "e")},
						merge.Conflict{Manager: "controller", Path: _P("b", "c", "f")},
					},
				},
				ForceApply{
					Manager:    "default",
					APIVersion: "v3",
					Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
				},
			},
			Object: `
				a: 2
				b:
				  c:
				    d: 2
				    e:
				    - 3
				    - 2
				    - 1
				    f:
				    - name: n
				      value: 1
				g: 5
			`,
			APIVersion: "v1",
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

func BenchmarkDeducedSimple(b *testing.B) {
	test := TestCase{
		Ops: []Operation{
			Apply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						numeric: 1
						string: "string"
					`,
			},
			Update{
				Manager:    "controller",
				APIVersion: "v1",
				Object: `
						numeric: 1
						string: "controller string"
						bool: true
					`,
			},
			Apply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						numeric: 2
						string: "user string"
					`,
				Conflicts: merge.Conflicts{
					merge.Conflict{Manager: "controller", Path: _P("string")},
				},
			},
			ForceApply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						numeric: 2
						string: "user string"
					`,
			},
		},
		Object: `
				numeric: 2
				string: "user string"
				bool: true
			`,
		APIVersion: "v1",
		Managed: fieldpath.ManagedFields{
			"default": fieldpath.NewVersionedSet(
				_NS(
					_P("numeric"), _P("string"),
				),
				"v1",
				false,
			),
			"controller": fieldpath.NewVersionedSet(
				_NS(
					_P("bool"),
				),
				"v1",
				false,
			),
		},
	}

	// Make sure this passes...
	if err := test.Test(DeducedParser); err != nil {
		b.Fatal(err)
	}

	test.PreprocessOperations(DeducedParser)

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if err := test.Bench(DeducedParser); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDeducedNested(b *testing.B) {
	test := TestCase{
		Ops: []Operation{
			Apply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						a: 1
						b:
						  c:
						    d: 2
						    e:
						    - 1
						    - 2
						    - 3
						    f:
						    - name: n
						      value: 1
					`,
			},
			Update{
				Manager:    "controller",
				APIVersion: "v1",
				Object: `
						a: 1
						b:
						  c:
						    d: 3
						    e:
						    - 1
						    - 2
						    - 3
						    - 4
						    f:
						    - name: n
						      value: 2
						g: 5
					`,
			},
			Apply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
				Conflicts: merge.Conflicts{
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "d")},
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "e")},
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "f")},
				},
			},
			ForceApply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
						a: 2
						b:
						  c:
						    d: 2
						    e:
						    - 3
						    - 2
						    - 1
						    f:
						    - name: n
						      value: 1
					`,
			},
		},
		Object: `
				a: 2
				b:
				  c:
				    d: 2
				    e:
				    - 3
				    - 2
				    - 1
				    f:
				    - name: n
				      value: 1
				g: 5
			`,
		APIVersion: "v1",
	}

	// Make sure this passes...
	if err := test.Test(DeducedParser); err != nil {
		b.Fatal(err)
	}

	test.PreprocessOperations(DeducedParser)

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if err := test.Bench(DeducedParser); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDeducedNestedAcrossVersion(b *testing.B) {
	test := TestCase{
		Ops: []Operation{
			Apply{
				Manager:    "default",
				APIVersion: "v1",
				Object: `
					a: 1
					b:
					  c:
					    d: 2
					    e:
					    - 1
					    - 2
					    - 3
					    f:
					    - name: n
					      value: 1
				`,
			},
			Update{
				Manager:    "controller",
				APIVersion: "v2",
				Object: `
					a: 1
					b:
					  c:
					    d: 3
					    e:
					    - 1
					    - 2
					    - 3
					    - 4
					    f:
					    - name: n
					      value: 2
					g: 5
				`,
			},
			Apply{
				Manager:    "default",
				APIVersion: "v3",
				Object: `
					a: 2
					b:
					  c:
					    d: 2
					    e:
					    - 3
					    - 2
					    - 1
					    f:
					    - name: n
					      value: 1
				`,
				Conflicts: merge.Conflicts{
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "d")},
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "e")},
					merge.Conflict{Manager: "controller", Path: _P("b", "c", "f")},
				},
			},
			ForceApply{
				Manager:    "default",
				APIVersion: "v3",
				Object: `
					a: 2
					b:
					  c:
					    d: 2
					    e:
					    - 3
					    - 2
					    - 1
					    f:
					    - name: n
					      value: 1
				`,
			},
		},
		Object: `
			a: 2
			b:
			  c:
			    d: 2
			    e:
			    - 3
			    - 2
			    - 1
			    f:
			    - name: n
			      value: 1
			g: 5
		`,
		APIVersion: "v1",
	}

	// Make sure this passes...
	if err := test.Test(DeducedParser); err != nil {
		b.Fatal(err)
	}

	test.PreprocessOperations(DeducedParser)

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		if err := test.Bench(DeducedParser); err != nil {
			b.Fatal(err)
		}
	}
}
