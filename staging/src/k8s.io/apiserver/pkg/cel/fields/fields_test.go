/*
Copyright 2025 The Kubernetes Authors.

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

package fields

import (
	"github.com/google/cel-go/cel"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/cel/environment"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"testing"
)

func TestReachableFields(t *testing.T) {
	testCases := []struct {
		name       string
		expression string
		variables  sets.Set[string]
		expect     *fieldpath.Set
	}{
		{
			name:       "variable",
			expression: "a",
			variables:  sets.New("a"),
			expect:     fieldpath.NewSet(fieldpath.MakePathOrDie("a")),
		},
		{
			name:       "select",
			expression: "a.b.c.d.e.f",
			variables:  sets.New("a"),
			expect:     fieldpath.NewSet(fieldpath.MakePathOrDie("a", "b", "c", "d", "e", "f")),
		},
		{
			name:       "test",
			expression: "has(a.b.c.d.e.f)",
			variables:  sets.New("a"),
			expect:     fieldpath.NewSet(fieldpath.MakePathOrDie("a", "b", "c", "d", "e", "f")),
		},
		{
			name:       "index",
			expression: "a[c].x",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", Wildcard, "x"),
				fieldpath.MakePathOrDie("c"),
			),
		},
		{
			name:       "data literal",
			expression: "1",
			variables:  sets.New[string](),
			expect:     fieldpath.NewSet(),
		},
		{
			name:       "infix operator",
			expression: "a.b < c.d",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
			),
		},
		{
			name:       "index same path multiple times",
			expression: "a.b == a.b",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
			),
		},
		{
			name:       "index parent and child",
			expression: "true ? a.b : a.b.c",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("a", "b", "c"),
			),
		},
		{
			name:       "boolean operators",
			expression: "a.b || c.d && e.f",
			variables:  sets.New("a", "c", "e"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
				fieldpath.MakePathOrDie("e", "f"),
			),
		},
		{
			name:       "ternary operator",
			expression: "a.b ? c.d : e.f",
			variables:  sets.New("a", "c", "e"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
				fieldpath.MakePathOrDie("e", "f"),
			),
		},
		{
			name:       "list literal: empty",
			expression: "[]",
			variables:  sets.New[string](),
			expect:     fieldpath.NewSet(),
		},
		{
			name:       "list literal",
			expression: "[a.b, c.d]",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
			),
		},
		{
			name:       "list literal: nested",
			expression: "[[a.b], [c.d]]",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
			),
		},
		{
			name:       "list literal: selector",
			expression: "[a.b, c.d][0].x",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, "x"),
				fieldpath.MakePathOrDie("c", "d", Wildcard, "x"),
			),
		},
		{
			name:       "map literal",
			expression: "{a.k1: c.v1, a.k2: c.v2}",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "k1"),
				fieldpath.MakePathOrDie("c", "v1"),
				fieldpath.MakePathOrDie("a", "k2"),
				fieldpath.MakePathOrDie("c", "v2"),
			),
		},
		{
			name:       "map literal: selector",
			expression: "{a.k1: c.v1, a.k2: c.v2}['foo'].z",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "k1"),
				fieldpath.MakePathOrDie("c", "v1", Wildcard, "z"),
				fieldpath.MakePathOrDie("a", "k2"),
				fieldpath.MakePathOrDie("c", "v2", Wildcard, "z"),
			),
		},
		{
			name:       "map literal: empty",
			expression: "{}",
			variables:  sets.New[string](),
			expect:     fieldpath.NewSet(),
		},
		{
			name:       "call 1 arg",
			expression: "size(a.b.c.d)",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", "c", "d"),
			),
		},
		{
			name:       "call 2 arg",
			expression: "matches(a.b, c.d)",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
			),
		},
		{
			name:       "call with infix",
			expression: "size(a.b + c.d)",
			variables:  sets.New("a", "c"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b"),
				fieldpath.MakePathOrDie("c", "d"),
			),
		},
		{
			name:       "call target",
			expression: "a.b.c.d.size()",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", "c", "d"),
			),
		},
		{
			name:       "1 var macro",
			expression: "a.b.all(x, x.y)",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, "y"),
			),
		},
		{
			name:       "1 var macro: nested",
			expression: "a.b.all(x, x.all(y, y.z))",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, Wildcard, "z"),
			),
		},
		{
			name:       "1 var macro: chained",
			expression: "a.b.map(x, x.y).map(x2, x2.z)",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, "y", Wildcard, "z"),
			),
		},
		{
			name:       "1 var macro: selector",
			expression: "a.b.map(x, x.y)[0].z",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, "y", Wildcard, "z"),
			),
		},
		{
			name:       "2 var macro",
			expression: "a.b.all(k, v, k + v.y)",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard),
				fieldpath.MakePathOrDie("a", "b", Wildcard, "y"),
			),
		},
		{
			name:       "2 var macro: nested",
			expression: "a.b.all(k, v, v.all(k2, v2, v2.y2 + v2.x2))",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard, Wildcard, "x2"),
				fieldpath.MakePathOrDie("a", "b", Wildcard, Wildcard, "y2"),
			),
		},
		{
			name:       "2 var macro: nested key access",
			expression: "a.b.all(k, v, v.all(k2, v2, k + k2))",
			variables:  sets.New("a"),
			expect: fieldpath.NewSet(
				fieldpath.MakePathOrDie("a", "b", Wildcard),
				fieldpath.MakePathOrDie("a", "b", Wildcard, Wildcard),
			),
		},
	}
	envSet := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true)
	env := envSet.StoredExpressionsEnv()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []cel.EnvOption
			for varName := range tc.variables {
				opts = append(opts, cel.Variable(varName, cel.DynType))
			}
			env, err := env.Extend(opts...)
			if err != nil {
				t.Fatalf("Error updating env: %v", err)
			}
			ast, issues := env.Compile(tc.expression)
			if issues.Err() != nil {
				t.Fatalf("Error compiling: %v", issues.Err())
			}
			rep := ast.NativeRep()
			result := ReachableFields(rep.Expr())
			if !tc.expect.Equals(result) {
				t.Errorf("Expected %v, got %v\ngot unexpected: %v\nmissing: %v\n",
					tc.expect, result, result.Difference(tc.expect), tc.expect.Difference(result))
			}
		})
	}
}
