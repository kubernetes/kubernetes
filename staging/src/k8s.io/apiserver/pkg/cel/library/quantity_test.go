/*
Copyright 2023 The Kubernetes Authors.

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

package library_test

import (
	"regexp"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"github.com/google/cel-go/ext"
	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/library"
)

func testQuantity(t *testing.T, expr string, expectResult ref.Val, expectRuntimeErrPattern string, expectCompileErrs []string) {
	env, err := cel.NewEnv(
		cel.OptionalTypes(),
		ext.Strings(),
		library.URLs(),
		library.Regex(),
		library.Lists(),
		library.Quantity(),
		library.Format(),
	)
	if err != nil {
		t.Fatalf("%v", err)
	}
	compiled, issues := env.Compile(expr)

	if len(expectCompileErrs) > 0 {
		missingCompileErrs := []string{}
		matchedCompileErrs := sets.New[int]()
		for _, expectedCompileErr := range expectCompileErrs {
			compiledPattern, err := regexp.Compile(expectedCompileErr)
			if err != nil {
				t.Fatalf("failed to compile expected err regex: %v", err)
			}

			didMatch := false

			for i, compileError := range issues.Errors() {
				if compiledPattern.Match([]byte(compileError.Message)) {
					didMatch = true
					matchedCompileErrs.Insert(i)
				}
			}

			if !didMatch {
				missingCompileErrs = append(missingCompileErrs, expectedCompileErr)
			} else if len(matchedCompileErrs) != len(issues.Errors()) {
				unmatchedErrs := []cel.Error{}
				for i, issue := range issues.Errors() {
					if !matchedCompileErrs.Has(i) {
						unmatchedErrs = append(unmatchedErrs, *issue)
					}
				}
				require.Empty(t, unmatchedErrs, "unexpected compilation errors")
			}
		}

		require.Empty(t, missingCompileErrs, "expected compilation errors")
		return
	} else if len(issues.Errors()) > 0 {
		errorStrings := []string{}
		source := common.NewTextSource(expr)
		for _, issue := range issues.Errors() {
			errorStrings = append(errorStrings, issue.ToDisplayString(source))
		}
		t.Fatalf("%v", errorStrings)
	}

	// Typecheck expression
	_, err = cel.AstToCheckedExpr(compiled)
	if err != nil {
		t.Fatalf("%v", err)
	}

	prog, err := env.Program(compiled)
	if err != nil {
		t.Fatalf("%v", err)
	}
	res, _, err := prog.Eval(map[string]interface{}{})
	if len(expectRuntimeErrPattern) > 0 {
		if err == nil {
			t.Fatalf("no runtime error thrown. Expected: %v", expectRuntimeErrPattern)
		} else if matched, regexErr := regexp.MatchString(expectRuntimeErrPattern, err.Error()); regexErr != nil {
			t.Fatalf("failed to compile expected err regex: %v", regexErr)
		} else if !matched {
			t.Fatalf("unexpected err: %v", err)
		}
	} else if err != nil {
		t.Fatalf("%v", err)
	} else if expectResult != nil {
		converted := res.Equal(expectResult).Value().(bool)
		require.True(t, converted, "expectation not equal to output: %v", cmp.Diff(expectResult.Value(), res.Value()))
	} else {
		t.Fatal("expected result must not be nil")
	}

}

func TestQuantity(t *testing.T) {
	twelveMi := resource.MustParse("12Mi")
	trueVal := types.Bool(true)
	falseVal := types.Bool(false)

	cases := []struct {
		name               string
		expr               string
		expectValue        ref.Val
		expectedCompileErr []string
		expectedRuntimeErr string
	}{
		{
			name:        "parse",
			expr:        `quantity("12Mi")`,
			expectValue: apiservercel.Quantity{Quantity: &twelveMi},
		},
		{
			name:               "parseInvalidSuffix",
			expr:               `quantity("10Mo")`,
			expectedRuntimeErr: "quantities must match the regular expression.*",
		},
		{
			// The above case fails due to a regex check. This case passes the
			// regex check and fails a suffix check
			name:               "parseInvalidSuffixPassesRegex",
			expr:               `quantity("10Mm")`,
			expectedRuntimeErr: "unable to parse quantity's suffix",
		},
		{
			name:        "isQuantity",
			expr:        `isQuantity("20")`,
			expectValue: trueVal,
		},
		{
			name:        "isQuantity_megabytes",
			expr:        `isQuantity("20M")`,
			expectValue: trueVal,
		},
		{
			name:        "isQuantity_mebibytes",
			expr:        `isQuantity("20Mi")`,
			expectValue: trueVal,
		},
		{
			name:        "isQuantity_invalidSuffix",
			expr:        `isQuantity("20Mo")`,
			expectValue: falseVal,
		},
		{
			name:        "isQuantity_passingRegex",
			expr:        `isQuantity("10Mm")`,
			expectValue: falseVal,
		},
		{
			name:               "isQuantity_noOverload",
			expr:               `isQuantity([1, 2, 3])`,
			expectedCompileErr: []string{"found no matching overload for 'isQuantity' applied to.*"},
		},
		{
			name:        "equality_reflexivity",
			expr:        `quantity("200M") == quantity("200M")`,
			expectValue: trueVal,
		},
		{
			name:        "equality_symmetry",
			expr:        `quantity("200M") == quantity("0.2G") && quantity("0.2G") == quantity("200M")`,
			expectValue: trueVal,
		},
		{
			name:        "equality_transitivity",
			expr:        `quantity("2M") == quantity("0.002G") && quantity("2000k") == quantity("2M") && quantity("0.002G") == quantity("2000k")`,
			expectValue: trueVal,
		},
		{
			name:        "inequality",
			expr:        `quantity("200M") == quantity("0.3G")`,
			expectValue: falseVal,
		},
		{
			name:        "quantity_less",
			expr:        `quantity("50M").isLessThan(quantity("50Mi"))`,
			expectValue: trueVal,
		},
		{
			name:        "quantity_less_obvious",
			expr:        `quantity("50M").isLessThan(quantity("100M"))`,
			expectValue: trueVal,
		},
		{
			name:        "quantity_less_false",
			expr:        `quantity("100M").isLessThan(quantity("50M"))`,
			expectValue: falseVal,
		},
		{
			name:        "quantity_greater",
			expr:        `quantity("50Mi").isGreaterThan(quantity("50M"))`,
			expectValue: trueVal,
		},
		{
			name:        "quantity_greater_obvious",
			expr:        `quantity("150Mi").isGreaterThan(quantity("100Mi"))`,
			expectValue: trueVal,
		},
		{
			name:        "quantity_greater_false",
			expr:        `quantity("50M").isGreaterThan(quantity("100M"))`,
			expectValue: falseVal,
		},
		{
			name:        "compare_equal",
			expr:        `quantity("200M").compareTo(quantity("0.2G"))`,
			expectValue: types.Int(0),
		},
		{
			name:        "compare_less",
			expr:        `quantity("50M").compareTo(quantity("50Mi"))`,
			expectValue: types.Int(-1),
		},
		{
			name:        "compare_greater",
			expr:        `quantity("50Mi").compareTo(quantity("50M"))`,
			expectValue: types.Int(1),
		},
		{
			name:        "add_quantity",
			expr:        `quantity("50k").add(quantity("20")) == quantity("50.02k")`,
			expectValue: trueVal,
		},
		{
			name:        "add_int",
			expr:        `quantity("50k").add(20).isLessThan(quantity("50020"))`,
			expectValue: falseVal,
		},
		{
			name:        "sub_quantity",
			expr:        `quantity("50k").sub(quantity("20")) == quantity("49.98k")`,
			expectValue: trueVal,
		},
		{
			name:        "sub_int",
			expr:        `quantity("50k").sub(20) == quantity("49980")`,
			expectValue: trueVal,
		},
		{
			name:        "arith_chain_1",
			expr:        `quantity("50k").add(20).sub(quantity("100k")).asInteger()`,
			expectValue: types.Int(-49980),
		},
		{
			name:        "arith_chain",
			expr:        `quantity("50k").add(20).sub(quantity("100k")).sub(-50000).asInteger()`,
			expectValue: types.Int(20),
		},
		{
			name:        "as_integer",
			expr:        `quantity("50k").asInteger()`,
			expectValue: types.Int(50000),
		},
		{
			name:               "as_integer_error",
			expr:               `quantity("9999999999999999999999999999999999999G").asInteger()`,
			expectedRuntimeErr: `cannot convert value to integer`,
		},
		{
			name:        "is_integer",
			expr:        `quantity("9999999999999999999999999999999999999G").isInteger()`,
			expectValue: falseVal,
		},
		{
			name:        "is_integer",
			expr:        `quantity("50").isInteger()`,
			expectValue: trueVal,
		},
		{
			name:        "as_float",
			expr:        `quantity("50.703k").asApproximateFloat()`,
			expectValue: types.Double(50703),
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			testQuantity(t, c.expr, c.expectValue, c.expectedRuntimeErr, c.expectedCompileErr)
		})
	}
}
