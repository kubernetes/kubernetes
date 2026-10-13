/*
Copyright 2024 The Kubernetes Authors.

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

package generators

import (
	"reflect"
	"testing"

	"k8s.io/code-generator/cmd/validation-gen/validators"
	"k8s.io/gengo/v2/types"
)

// gengo has `PointerTo()` but not the rest, so keep this here for consistency.
func ptrTo(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "*" + t.Name.String(),
		},
		Kind: types.Pointer,
		Elem: t,
	}
}

func sliceOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "[]" + t.Name.String(),
		},
		Kind: types.Slice,
		Elem: t,
	}
}

func mapOf(t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "map[string]" + t.Name.String(),
		},
		Kind: types.Map,
		Key:  types.String,
		Elem: t,
	}
}

func aliasOf(name string, t *types.Type) *types.Type {
	return &types.Type{
		Name: types.Name{
			Package: "",
			Name:    "Alias_" + name,
		},
		Kind:       types.Alias,
		Underlying: t,
	}
}

func TestGetLeafTypeAndPrefixes(t *testing.T) {

	cases := []struct {
		in              *types.Type
		expectedType    *types.Type
		expectedTypePfx string
		expectedExprPfx string
	}{{
		// string
		in:              types.String,
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// *string
		in:              ptrTo(types.String),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **string
		in:              ptrTo(ptrTo(types.String)),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***string
		in:              ptrTo(ptrTo(ptrTo(types.String))),
		expectedType:    types.String,
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []string
		in:              sliceOf(types.String),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]string
		in:              ptrTo(sliceOf(types.String)),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]string
		in:              ptrTo(ptrTo(sliceOf(types.String))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    sliceOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]string
		in:              mapOf(types.String),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]string
		in:              ptrTo(mapOf(types.String)),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]string
		in:              ptrTo(ptrTo(mapOf(types.String))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]string
		in:              ptrTo(ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    mapOf(types.String),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// alias of string
		in:              aliasOf("s", types.String),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "&",
	}, {
		// alias of *string
		in:              aliasOf("ps", ptrTo(types.String)),
		expectedType:    aliasOf("ps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **string
		in:              aliasOf("pps", ptrTo(ptrTo(types.String))),
		expectedType:    aliasOf("pps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***string
		in:              aliasOf("ppps", ptrTo(ptrTo(ptrTo(types.String)))),
		expectedType:    aliasOf("ppps", types.String),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of []string
		in:              aliasOf("ls", sliceOf(types.String)),
		expectedType:    aliasOf("ls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *[]string
		in:              aliasOf("pls", ptrTo(sliceOf(types.String))),
		expectedType:    aliasOf("pls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **[]string
		in:              aliasOf("ppls", ptrTo(ptrTo(sliceOf(types.String)))),
		expectedType:    aliasOf("ppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***[]string
		in:              aliasOf("pppls", ptrTo(ptrTo(ptrTo(sliceOf(types.String))))),
		expectedType:    aliasOf("pppls", sliceOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of map[string]string
		in:              aliasOf("ms", mapOf(types.String)),
		expectedType:    aliasOf("ms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of *map[string]string
		in:              aliasOf("pms", ptrTo(mapOf(types.String))),
		expectedType:    aliasOf("pms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of **map[string]string
		in:              aliasOf("ppms", ptrTo(ptrTo(mapOf(types.String)))),
		expectedType:    aliasOf("ppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// alias of ***map[string]string
		in:              aliasOf("pppms", ptrTo(ptrTo(ptrTo(mapOf(types.String))))),
		expectedType:    aliasOf("pppms", mapOf(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *alias-of-string
		in:              ptrTo(aliasOf("s", types.String)),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "",
	}, {
		// **alias-of-string
		in:              ptrTo(ptrTo(aliasOf("s", types.String))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "*",
	}, {
		// ***alias-of-string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("s", types.String)))),
		expectedType:    aliasOf("s", types.String),
		expectedTypePfx: "*",
		expectedExprPfx: "**",
	}, {
		// []alias-of-string
		in:              sliceOf(aliasOf("s", types.String)),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-string
		in:              ptrTo(sliceOf(aliasOf("s", types.String))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("s", types.String)))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("s", types.String))))),
		expectedType:    sliceOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-string
		in:              mapOf(aliasOf("s", types.String)),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-string
		in:              ptrTo(mapOf(aliasOf("s", types.String))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-string
		in:              ptrTo(ptrTo(mapOf(aliasOf("s", types.String)))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("s", types.String))))),
		expectedType:    mapOf(aliasOf("s", types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// *alias-of-*string
		in:              ptrTo(aliasOf("ps", ptrTo(types.String))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **alias-of-*string
		in:              ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    aliasOf("ps", ptrTo(types.String)),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// []alias-of-*string
		in:              sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *[]alias-of-*string
		in:              ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **[]alias-of-*string
		in:              ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***[]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(sliceOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    sliceOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}, {
		// map[string]alias-of-*string
		in:              mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "",
	}, {
		// *map[string]alias-of-*string
		in:              ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "*",
	}, {
		// **map[string]alias-of-*string
		in:              ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "**",
	}, {
		// ***map[string]alias-of-*string
		in:              ptrTo(ptrTo(ptrTo(mapOf(aliasOf("ps", ptrTo(types.String)))))),
		expectedType:    mapOf(aliasOf("ps", ptrTo(types.String))),
		expectedTypePfx: "",
		expectedExprPfx: "***",
	}}

	for _, tc := range cases {
		leafType, typePfx, exprPfx := getLeafTypeAndPrefixes(tc.in)
		if got, want := leafType.Name.String(), tc.expectedType.Name.String(); got != want {
			t.Errorf("%q: wrong leaf type: expected %q, got %q", tc.in, want, got)
		}
		if got, want := typePfx, tc.expectedTypePfx; got != want {
			t.Errorf("%q: wrong type prefix: expected %q, got %q", tc.in, want, got)
		}
		if got, want := exprPfx, tc.expectedExprPfx; got != want {
			t.Errorf("%q: wrong expr prefix: expected %q, got %q", tc.in, want, got)
		}
	}
}

func TestSortIntoCohorts(t *testing.T) {
	cases := []struct {
		in       []validators.FunctionGen
		expected [][]validators.FunctionGen
	}{{
		// empty
		in:       []validators.FunctionGen{},
		expected: [][]validators.FunctionGen{},
	}, {
		// default cohort
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, not already sorted by name
		in: []validators.FunctionGen{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, with a short-circuit
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default cohort, with 2 short-circuits
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "", Flags: validators.ShortCircuit},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "b", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "d", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}},
	}, {
		// default and non-default cohorts
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "e", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "f", Cohort: "", Flags: validators.DefaultFlags},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "f", Cohort: "", Flags: validators.DefaultFlags},
		}, {
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.DefaultFlags},
		}, {
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "e", Cohort: "bar", Flags: validators.DefaultFlags},
		}},
	}, {
		// default and non-default cohorts with short-circuit
		in: []validators.FunctionGen{
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
			{TagName: "d", Cohort: "foo", Flags: validators.ShortCircuit},
			{TagName: "e", Cohort: "bar", Flags: validators.ShortCircuit},
			{TagName: "f", Cohort: "", Flags: validators.ShortCircuit},
		},
		expected: [][]validators.FunctionGen{{
			{TagName: "f", Cohort: "", Flags: validators.ShortCircuit},
			{TagName: "c", Cohort: "", Flags: validators.DefaultFlags},
		}, {
			{TagName: "d", Cohort: "foo", Flags: validators.ShortCircuit},
			{TagName: "a", Cohort: "foo", Flags: validators.DefaultFlags},
		}, {
			{TagName: "e", Cohort: "bar", Flags: validators.ShortCircuit},
			{TagName: "b", Cohort: "bar", Flags: validators.DefaultFlags},
		}},
	}}

	for _, tc := range cases {
		out := sortIntoCohorts(tc.in)
		if !reflect.DeepEqual(out, tc.expected) {
			t.Errorf("expected %v, got %v", tc.expected, out)
		}
	}
}

func TestNearestPkg(t *testing.T) {
	const (
		modAScheduling = "mod.a/scheduling/validation"
		modBScheduling = "mod.b/apis/scheduling"
		modCMeta       = "mod.c/meta/validation"
	)
	cases := []struct {
		name       string
		pkg        string // the package being generated
		candidates []string
		want       string
	}{{
		name:       "single candidate is always chosen, even in another module",
		pkg:        "mod.b/apis/batch",
		candidates: []string{modCMeta},
		want:       modCMeta,
	}, {
		// The motivating case: a caller must not resolve to a copy in another
		// module, which it may be unable to import.
		name:       "caller prefers the candidate in its own module",
		pkg:        "mod.b/apis/batch",
		candidates: []string{modBScheduling, modAScheduling},
		want:       modBScheduling,
	}, {
		name:       "a caller in another module prefers that module's candidate",
		pkg:        "mod.a/batch/validation",
		candidates: []string{modBScheduling, modAScheduling},
		want:       modAScheduling,
	}, {
		name:       "deeper shared prefix wins over a shallower one",
		pkg:        "root/nearby/client",
		candidates: []string{"root/registered", "root/external", "root/nearby/lib"},
		want:       "root/nearby/lib",
	}, {
		// Heuristic limit: with no prefix to tell the candidates apart,
		// canonical wins. Doing better would need the module graph.
		name:       "ambiguous shallow tie falls back to canonical",
		pkg:        "mod.d/x/validation",
		candidates: []string{modBScheduling, modAScheduling},
		want:       modBScheduling,
	}}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := nearestPkg(tc.pkg, tc.candidates); got != tc.want {
				t.Errorf("nearestPkg(%q, %v) = %q, want %q", tc.pkg, tc.candidates, got, tc.want)
			}
		})
	}
}

func TestCommonPathDepth(t *testing.T) {
	cases := []struct {
		a, b string
		want int
	}{
		{"a/b/c", "a/b/d", 2},
		{"a/b/c", "a/b/c", 3},
		{"a/b/c", "a/b", 2}, // stops at the shorter path
		{"x/y", "a/b", 0},
		{"k8s.io/api", "k8s.io/apimachinery", 1}, // whole segments only
	}
	for _, tc := range cases {
		t.Run(tc.a+"|"+tc.b, func(t *testing.T) {
			if got := commonPathDepth(tc.a, tc.b); got != tc.want {
				t.Errorf("commonPathDepth(%q, %q) = %d, want %d", tc.a, tc.b, got, tc.want)
			}
		})
	}
}
