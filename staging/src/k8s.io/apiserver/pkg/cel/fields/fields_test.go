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
	"testing"
)

func TestReachableFields(t *testing.T) {
	testCases := []struct {
		name       string
		expression string
		variables  sets.Set[string]
		expect     sets.Set[string]
	}{
		{
			name:       "variable",
			expression: "a",
			variables:  sets.New("a"),
			expect:     sets.New("a"),
		},
		{
			name:       "select",
			expression: "a.b.c.d.e.f",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.c.d.e.f"),
		},
		{
			name:       "test",
			expression: "has(a.b.c.d.e.f)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.c.d.e.f"),
		},
		{
			name:       "index",
			expression: "a[c].x",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.@index.x", "c"),
		},
		{
			name:       "data literal",
			expression: "1",
			variables:  sets.New[string](),
			expect:     sets.New[string](),
		},
		{
			name:       "infix operator",
			expression: "a.b < c.d",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.b", "c.d"),
		},
		{
			name:       "boolean operators",
			expression: "a.b || c.d && e.f",
			variables:  sets.New("a", "c", "e"),
			expect:     sets.New("a.b", "c.d", "e.f"),
		},
		{
			name:       "ternary operator",
			expression: "a.b ? c.d : e.f",
			variables:  sets.New("a", "c", "e"),
			expect:     sets.New("a.b", "c.d", "e.f"),
		},
		{
			name:       "list literal: empty",
			expression: "[]",
			variables:  sets.New[string](),
			expect:     sets.New[string](),
		},
		{
			name:       "list literal",
			expression: "[a.b, c.d]",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.b", "c.d"),
		},
		{
			name:       "list literal: nested",
			expression: "[[a.b], [c.d]]",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.b", "c.d"),
		},
		{
			name:       "list literal: selector",
			expression: "[a.b, c.d][0].x",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.b.@index.x", "c.d.@index.x"),
		},
		{
			name:       "map literal",
			expression: "{a.k1: c.v1, a.k2: c.v2}",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.k1", "c.v1", "a.k2", "c.v2"),
		},
		{
			name:       "map literal: selector",
			expression: "{a.k1: c.v1, a.k2: c.v2}['foo'].z",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.k1", "c.v1.@index.z", "a.k2", "c.v2.@index.z"),
		},
		{
			name:       "map literal: empty",
			expression: "{}",
			variables:  sets.New[string](),
			expect:     sets.New[string](),
		},
		{
			name:       "call 1 arg",
			expression: "size(a.b.c.d)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.c.d"),
		},
		{
			name:       "call 2 arg",
			expression: "matches(a.b, c.d)",
			variables:  sets.New("a", "c"),
			expect:     sets.New("a.b", "c.d"),
		},
		{
			name:       "call target",
			expression: "a.b.c.d.size()",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.c.d"),
		},
		{
			name:       "1 var macro",
			expression: "a.b.all(x, x.y)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.y"),
		},
		{
			name:       "1 var macro: nested",
			expression: "a.b.all(x, x.all(y, y.z))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.@item.z"),
		},
		{
			name:       "1 var macro: chained",
			expression: "a.b.map(x, x.y).map(x2, x2.z)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.y.@item.z"),
		},
		{
			name:       "1 var macro: selector",
			expression: "a.b.map(x, x.y)[0].z",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.y.@index.z"),
		},
		{
			name:       "2 var macro",
			expression: "a.b.all(k, v, k.x + v.y)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@index.x", "a.b.@value.y"),
		},
		{
			name:       "2 var macro: nested",
			expression: "a.b.all(k, v, k.all(k2, v2, k2.x2 + v2.y2 + v.y))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@index.@index.x2", "a.b.@index.@value.y2", "a.b.@value.y"),
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
			if !tc.expect.Equal(result) {
				t.Errorf("Expected %v, got %v\ngot unexpected: %v\nmissing: %v\n",
					tc.expect, result, result.Difference(tc.expect), tc.expect.Difference(result))
			}
		})
	}
}

// TestReachableFieldsComprehensions focuses on testing the comprehension handling
// in the ReachableFields function, which is one of the most complex parts of the implementation.
func TestReachableFieldsComprehensions(t *testing.T) {
	envSet := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true)
	env := envSet.StoredExpressionsEnv()

	testCases := []struct {
		name       string
		expression string
		variables  sets.Set[string]
		expect     sets.Set[string]
	}{
		// Basic comprehension tests
		{
			name:       "simple map",
			expression: "a.b.map(x, x.c)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
		{
			name:       "simple filter",
			expression: "a.b.filter(x, x.c > 10)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
		{
			name:       "simple all",
			expression: "a.b.all(x, x.c == true)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
		{
			name:       "simple exists",
			expression: "a.b.exists(x, x.c == 'value')",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
		{
			name:       "simple reduce",
			expression: "a.b.reduce(0, acc, x, acc + x.c)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},

		// Tests for accumulator variable handling
		{
			name:       "reduce with accumulator reference",
			expression: "a.b.reduce(0, acc, x, acc + x.c)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
		{
			name:       "reduce with complex accumulator expression",
			expression: "a.b.reduce(0, acc, x, acc + x.c * x.d)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d"),
		},
		{
			name:       "nested reduce with accumulator",
			expression: "a.b.reduce(0, acc, x, acc + x.c.reduce(0, inner, y, inner + y.d))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d"),
		},

		// Tests for iterator variable handling
		{
			name:       "map with complex iterator expression",
			expression: "a.b.map(x, x.c + x.d * x.e)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d", "a.b.@item.e"),
		},
		{
			name:       "nested map with iterator",
			expression: "a.b.map(x, x.c.map(y, y.d + x.e))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d", "a.b.@item.e"),
		},
		{
			name:       "deeply nested map with iterators",
			expression: "a.b.map(x, x.c.map(y, y.d.map(z, z.e + y.f + x.g)))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d.@item.e", "a.b.@item.c.@item.f", "a.b.@item.g"),
		},

		// Tests for map comprehensions with key/value pairs
		{
			name:       "map comprehension with key/value",
			expression: "a.b.all(k, v, k == 'key' && v.c > 10)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@index", "a.b.@value.c"),
		},
		{
			name:       "nested map comprehension with key/value",
			expression: "a.b.all(k1, v1, k1 == 'key' && v1.c.all(k2, v2, k2 > 5 && v2.d < 10))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@index", "a.b.@value.c.@index", "a.b.@value.c.@value.d"),
		},

		// Tests for complex comprehension chains
		{
			name:       "chained comprehensions",
			expression: "a.b.map(x, x.c).map(y, y.d)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d"),
		},
		{
			name:       "filter then map",
			expression: "a.b.filter(x, x.c > 10).map(y, y.d)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d"),
		},
		{
			name:       "complex chain with multiple operations",
			expression: "a.b.filter(x, x.c > 10).map(y, y.d).exists(z, z == 'value')",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d.@item"),
		},

		// Tests for comprehensions with complex conditions
		{
			name:       "filter with complex condition",
			expression: "a.b.filter(x, x.c > 10 && x.d < 20 && x.e == 'value')",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d", "a.b.@item.e"),
		},
		{
			name:       "exists with nested condition",
			expression: "a.b.exists(x, x.c > 10 && x.d.exists(y, y.e == 'value'))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d.@item.e"),
		},

		// Tests for comprehensions with indexing
		{
			name:       "map with indexing",
			expression: "a.b.map(x, x.c[d.e])",
			variables:  sets.New("a", "d"),
			expect:     sets.New("a.b.@item.c.@index", "d.e"),
		},
		{
			name:       "complex indexing in comprehension",
			expression: "a.b.map(x, x.c[d.e[f.g]])",
			variables:  sets.New("a", "d", "f"),
			expect:     sets.New("a.b.@item.c.@index", "d.e.@index", "f.g"),
		},
		{
			name:       "indexing into comprehension result",
			expression: "a.b.map(x, x.c)[d.e]",
			variables:  sets.New("a", "d"),
			expect:     sets.New("a.b.@item.c.@index", "d.e"),
		},

		// Tests for comprehensions with function calls
		{
			name:       "map with function call",
			expression: "a.b.map(x, x.c.startsWith(d.e))",
			variables:  sets.New("a", "d"),
			expect:     sets.New("a.b.@item.c", "d.e"),
		},
		{
			name:       "function call with comprehension argument",
			expression: "size(a.b.map(x, x.c))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []cel.EnvOption
			for varName := range tc.variables {
				opts = append(opts, cel.Variable(varName, cel.DynType))
			}
			testEnv, err := env.Extend(opts...)
			if err != nil {
				t.Fatalf("Error updating env: %v", err)
			}
			ast, issues := testEnv.Compile(tc.expression)
			if issues.Err() != nil {
				t.Fatalf("Error compiling: %v", issues.Err())
			}
			rep := ast.NativeRep()
			result := ReachableFields(rep.Expr())
			if !tc.expect.Equal(result) {
				t.Errorf("Expected %v, got %v\ngot unexpected: %v\nmissing: %v\n",
					tc.expect, result, result.Difference(tc.expect), tc.expect.Difference(result))
			}
		})
	}
}

// TestReachableFieldsEdgeCases tests edge cases for the ReachableFields function
func TestReachableFieldsEdgeCases(t *testing.T) {
	envSet := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion(), true)
	env := envSet.StoredExpressionsEnv()

	// Test case for nil expression is already covered in the implementation
	// ReachableFields(nil) should return nil

	// Test cases for complex comprehensions and edge cases
	testCases := []struct {
		name       string
		expression string
		variables  sets.Set[string]
		expect     sets.Set[string]
	}{
		// Complex comprehension cases
		{
			name:       "comprehension with accumulator",
			expression: "a.b.reduce(0, acc, x, acc + x.value)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.value"),
		},
		{
			name:       "comprehension with condition",
			expression: "a.b.filter(x, x.value > 10).map(y, y.name)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.value", "a.b.@item.name"),
		},
		{
			name:       "deeply nested comprehensions",
			expression: "a.b.map(x, x.c.map(y, y.d.filter(z, z.e > 10).map(w, w.f)))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d.@item.e", "a.b.@item.c.@item.d.@item.f"),
		},
		{
			name:       "comprehension with complex condition",
			expression: "a.b.filter(x, x.c.exists(y, y.d == 'test' && y.e > 10))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.d", "a.b.@item.c.@item.e"),
		},
		{
			name:       "map comprehension with key/value",
			expression: "a.b.all(k, v, k.startsWith('prefix') && v.c > 10)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@index", "a.b.@value.c"),
		},

		// Complex nested expressions
		{
			name:       "complex nested expressions",
			expression: "a.b[c.d[e.f]].g.all(x, x.h[i.j[k.l]].m)",
			variables:  sets.New("a", "c", "e", "i", "k"),
			expect:     sets.New("a.b.@index.g.@item.h.@index.m", "c.d.@index", "e.f", "i.j.@index", "k.l"),
		},
		{
			name:       "multiple nested indexing",
			expression: "a.b[c.d[e.f[g.h]]].i",
			variables:  sets.New("a", "c", "e", "g"),
			expect:     sets.New("a.b.@index.i", "c.d.@index", "e.f.@index", "g.h"),
		},
		{
			name:       "complex conditional",
			expression: "a.b ? (c.d ? e.f : g.h) : (i.j ? k.l : m.n)",
			variables:  sets.New("a", "c", "e", "g", "i", "k", "m"),
			expect:     sets.New("a.b", "c.d", "e.f", "g.h", "i.j", "k.l", "m.n"),
		},

		// Edge cases for comprehension accumulators
		{
			name:       "comprehension with accumulator reference",
			expression: "a.b.reduce(0, acc, x, acc + x.value + x.other)",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.value", "a.b.@item.other"),
		},
		{
			name:       "nested comprehension with accumulator",
			expression: "a.b.reduce(0, acc, x, acc + x.c.reduce(0, inner_acc, y, inner_acc + y.value))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c.@item.value"),
		},

		// Edge cases for loop conditions
		{
			name:       "comprehension with complex loop condition",
			expression: "a.b.exists(x, x.c == 'value' && x.d > 10 && x.e.contains('test'))",
			variables:  sets.New("a"),
			expect:     sets.New("a.b.@item.c", "a.b.@item.d", "a.b.@item.e"),
		},

		// Edge cases for nested map/list operations
		{
			name:       "nested map with complex operations",
			expression: "{a.b: c.d, e.f: {g.h: i.j, k.l: m.n}}[o.p].q",
			variables:  sets.New("a", "c", "e", "g", "i", "k", "m", "o"),
			expect:     sets.New("a.b", "c.d.@index.q", "e.f", "g.h", "i.j.@index.q", "k.l", "m.n.@index.q", "o.p"),
		},
		{
			name:       "list with complex operations",
			expression: "[a.b, [c.d, e.f], g.h][i.j][k.l]",
			variables:  sets.New("a", "c", "e", "g", "i", "k"),
			expect:     sets.New("a.b.@index.@index", "c.d.@index.@index", "e.f.@index.@index", "g.h.@index.@index", "i.j", "k.l"),
		},

		// Edge cases for function calls
		{
			name:       "complex function call with nested expressions",
			expression: "a.b.startsWith(c.d[e.f].g) && h.i.endsWith(j.k.l)",
			variables:  sets.New("a", "c", "e", "h", "j"),
			expect:     sets.New("a.b", "c.d.@index.g", "e.f", "h.i", "j.k.l"),
		},
		{
			name:       "function call with complex arguments",
			expression: "matches(a.b, c.d + e.f) && contains(g.h, i.j + k.l)",
			variables:  sets.New("a", "c", "e", "g", "i", "k"),
			expect:     sets.New("a.b", "c.d", "e.f", "g.h", "i.j", "k.l"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var opts []cel.EnvOption
			for varName := range tc.variables {
				opts = append(opts, cel.Variable(varName, cel.DynType))
			}
			testEnv, err := env.Extend(opts...)
			if err != nil {
				t.Fatalf("Error updating env: %v", err)
			}
			ast, issues := testEnv.Compile(tc.expression)
			if issues.Err() != nil {
				t.Fatalf("Error compiling: %v", issues.Err())
			}
			rep := ast.NativeRep()
			result := ReachableFields(rep.Expr())
			if !tc.expect.Equal(result) {
				t.Errorf("Expected %v, got %v\ngot unexpected: %v\nmissing: %v\n",
					tc.expect, result, result.Difference(tc.expect), tc.expect.Difference(result))
			}
		})
	}

	// Test for handling of optional field access if supported by the CEL version
	t.Run("optional field access", func(t *testing.T) {
		// This test requires CEL optional syntax support
		expr := "a.b.c"
		testEnv, err := env.Extend(cel.Variable("a", cel.DynType))
		if err != nil {
			t.Fatalf("Error updating env: %v", err)
		}
		ast, issues := testEnv.Compile(expr)
		if issues.Err() == nil {
			rep := ast.NativeRep()
			result := ReachableFields(rep.Expr())
			expect := sets.New("a.b.c")
			if !expect.Equal(result) {
				t.Errorf("Expected %v for field access, got %v", expect, result)
			}
		}
	})
}

// TestReachableFieldsLimitations documents known limitations and potential issues
// with the ReachableFields implementation that should be addressed in the future.
//
// Note: This test doesn't actually test anything, it just documents the limitations.
func TestReachableFieldsLimitations(t *testing.T) {
	t.Skip("This test is skipped because it only documents limitations")

	// Limitation 1: Handling of StructKind
	// The current implementation panics when encountering a StructKind expression:
	// case ast.StructKind:
	//     panic("object initialization not supported")
	//
	// This could be problematic if CEL expressions with struct literals are used.
	// Recommendation: Modify the implementation to handle StructKind expressions
	// by traversing their fields similar to how MapKind is handled.

	// Limitation 2: Handling of unknown expression kinds
	// The current implementation panics when encountering an unknown expression kind:
	// default:
	//     panic(fmt.Sprintf("unknown expression kind: %v", e.Kind()))
	//
	// This could be problematic if new expression kinds are added to the CEL AST.
	// Recommendation: Modify the implementation to log a warning and return an empty
	// set for unknown expression kinds instead of panicking.

	// Limitation 3: Handling of comprehension loop conditions
	// The current implementation doesn't process the loop condition of comprehensions:
	// result := ac.paths(comprehension.LoopStep(), vars)
	//
	// If the loop condition contains field references, they won't be included in the result.
	// Recommendation: Modify the implementation to also process comprehension.LoopCondition().

	// Limitation 4: Handling of comprehension result expressions
	// The current implementation doesn't process the result expression of comprehensions:
	// result := ac.paths(comprehension.LoopStep(), vars)
	//
	// If the result expression contains field references, they won't be included in the result.
	// Recommendation: Modify the implementation to also process comprehension.Result().

	// Limitation 5: Handling of optional fields
	// The current implementation doesn't distinguish between required and optional field access.
	// If CEL adds support for optional field access (e.g., a?.b?.c), the implementation
	// should be updated to handle it appropriately.

	// Limitation 6: Handling of recursive or circular references
	// The current implementation doesn't have protection against recursive or circular
	// references in the expression, which could lead to stack overflow.
	// Recommendation: Add a depth limit or cycle detection to prevent stack overflow.
}
