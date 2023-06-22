package library_test

import (
	"reflect"
	"regexp"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common"
	"github.com/google/cel-go/ext"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/cel/library"
)

func testQuantity(t *testing.T, expr string, expectResult any, expectRuntimeErrPattern string, expectCompileErrs []string) {
	env, err := cel.NewEnv(
		ext.Strings(),
		library.URLs(),
		library.Regex(),
		library.Lists(),
		library.Quantity(),
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
				unmatchedErrs := []common.Error{}
				for i, issue := range issues.Errors() {
					if !matchedCompileErrs.Has(i) {
						unmatchedErrs = append(unmatchedErrs, issue)
					}
				}
				require.Empty(t, unmatchedErrs, "unexpected compilation errors")
			}
		}

		require.Empty(t, missingCompileErrs, "expected compilation errors")
		return
	} else if len(issues.Errors()) > 0 {
		t.Fatalf("%v", issues.Errors())
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
	}

	if expectResult != nil {
		converted, err := res.ConvertToNative(reflect.TypeOf(expectResult))
		if err != nil {
			t.Fatalf("cannot convert evaluation result to expected result type %T", expectResult)
		}
		require.Equal(t, expectResult, converted, "expected result")
	}
}

func TestQuantity(t *testing.T) {
	cases := []struct {
		name               string
		expr               string
		expectValue        interface{}
		expectedCompileErr []string
		expectedRuntimeErr string
	}{
		{
			name: "parse",
			expr: `quantity("12Mi")`,
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
			expectValue: true,
		},
		{
			name:        "isQuantity_megabytes",
			expr:        `isQuantity("20M")`,
			expectValue: true},
		{
			name:        "isQuantity_mebibytes",
			expr:        `isQuantity("20Mi")`,
			expectValue: true,
		},
		{
			name:        "isQuantity_invalidSuffix",
			expr:        `isQuantity("20Mo")`,
			expectValue: false,
		},
		{
			name:        "isQuantity_passingRegex",
			expr:        `isQuantity("10Mm")`,
			expectValue: false,
		},
		{
			name:               "isQuantity_noOverload",
			expr:               `isQuantity([1, 2, 3])`,
			expectedCompileErr: []string{"found no matching overload for 'isQuantity' applied to.*"},
		},
		{
			name:        "equality_reflexivity",
			expr:        `quantity("200M") == quantity("200M")`,
			expectValue: true,
		},
		{
			name:        "equality_symmetry",
			expr:        `quantity("200M") == quantity("0.2G") && quantity("0.2G") == quantity("200M")`,
			expectValue: true,
		},
		{
			name:        "equality_transitivity",
			expr:        `quantity("2M") == quantity("0.002G") && quantity("2000k") == quantity("2M") && quantity("0.002G") == quantity("2000k")`,
			expectValue: true,
		},
		{
			name:        "inequality",
			expr:        `quantity("200M") == quantity("0.3G")`,
			expectValue: false,
		},
		{
			name: "quantity_less",
			expr: `quantity("50M").isLessThan(quantity("50Mi"))`,
		},
		{
			name: "quantity_greater",
			expr: `quantity("50Mi").isGreaterThan(quantity("50M"))`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			testQuantity(t, c.expr, c.expectValue, c.expectedRuntimeErr, c.expectedCompileErr)
		})
	}
}
