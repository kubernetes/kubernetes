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

package fieldpath_test

import (
	"fmt"
	"reflect"
	"testing"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

var (
	// Short names for readable test cases.
	_NS = fieldpath.NewSet
	_P  = fieldpath.MakePathOrDie
)

func TestManagersEquals(t *testing.T) {
	tests := []struct {
		name string
		lhs  fieldpath.ManagedFields
		rhs  fieldpath.ManagedFields
		out  fieldpath.ManagedFields
	}{
		{
			name: "Empty sets",
			out:  fieldpath.ManagedFields{},
		},
		{
			name: "Empty RHS",
			lhs: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			out: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
		},
		{
			name: "Empty LHS",
			rhs: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			out: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
		},
		{
			name: "Different managers",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"two": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			out: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
				"two": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
		},
		{
			name: "Same manager, different version",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("integer")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v2",
					false,
				),
			},
			out: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v2",
					false,
				),
			},
		},
		{
			name: "Set difference",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			out: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("bool")),
					"v1",
					false,
				),
			},
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf(test.name), func(t *testing.T) {
			want := test.out
			got := test.lhs.Difference(test.rhs)
			if !reflect.DeepEqual(want, got) {
				t.Errorf("want %v, got %v", want, got)
			}
		})
	}
}

func TestManagersDifference(t *testing.T) {
	tests := []struct {
		name  string
		lhs   fieldpath.ManagedFields
		rhs   fieldpath.ManagedFields
		equal bool
	}{
		{
			name:  "Empty sets",
			equal: true,
		},
		{
			name: "Same everything",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			equal: true,
		},
		{
			name: "Empty RHS",
			lhs: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			equal: false,
		},
		{
			name: "Empty LHS",
			rhs: fieldpath.ManagedFields{
				"default": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			equal: false,
		},
		{
			name: "Different managers",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"two": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			equal: false,
		},
		{
			name: "Same manager, different version",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("integer")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string"), _P("bool")),
					"v2",
					false,
				),
			},
			equal: false,
		},
		{
			name: "Set difference",
			lhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("numeric"), _P("string")),
					"v1",
					false,
				),
			},
			rhs: fieldpath.ManagedFields{
				"one": fieldpath.NewVersionedSet(
					_NS(_P("string"), _P("bool")),
					"v1",
					false,
				),
			},
			equal: false,
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf(test.name), func(t *testing.T) {
			equal := test.lhs.Equals(test.rhs)
			if test.equal && !equal {
				difference := test.lhs.Difference(test.rhs)
				t.Errorf("should be equal, but are different: %v", difference)
			} else if !test.equal && equal {
				t.Errorf("should not be equal, but they are")
			}
		})
	}
}
