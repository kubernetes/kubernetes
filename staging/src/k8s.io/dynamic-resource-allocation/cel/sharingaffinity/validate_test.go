/*
Copyright The Kubernetes Authors.

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

package sharingaffinity

import (
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
)

// Admission and runtime must use parameters as the root variable, not object.
func TestValidateExpression_ParametersEnvironment(t *testing.T) {
	for _, mode := range []environment.Type{environment.NewExpressions, environment.StoredExpressions} {
		t.Run(string(mode), func(t *testing.T) {
			for _, tc := range []struct {
				name, expression string
				wantError        bool
			}{
				{"optional access", `parameters.?subnetID.orValue("")`, false},
				{"driver private field", `parameters.private.subnet`, false},
				{"guarded wildcard kind", `has(parameters.kind) && parameters.kind == "NICConfig" ? parameters.?subnetID.orValue("") : ""`, false},
				{"bindings", `cel.bind(value, parameters.?subnetID.orValue(""), value)`, false},
				{"obsolete object root", `object.x`, true},
				{"device root", `device.x`, true},
				{"syntax error", `parameters[`, true},
			} {
				t.Run(tc.name, func(t *testing.T) {
					admission := ValidateExpression(tc.expression, mode)
					if got := admission.Error != nil; got != tc.wantError {
						t.Fatalf("ValidateExpression(%q) error = %v, want error %v", tc.expression, admission.Error, tc.wantError)
					}
					_, err := compileExpression(tc.expression, mode)
					if got := err != nil; got != tc.wantError {
						t.Errorf("runtime compiler error = %v, want error %v", err, tc.wantError)
					}
				})
			}
		})
	}
}

func TestValidateExpression_OutputType(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		wantError        bool
	}{
		{"string", `"blue"`, false},
		{"empty string result", `""`, false},
		{"dynamic field", `parameters.value`, false},
		{"empty expression", ``, true},
		{"int", `1`, true},
		{"bool", `true`, true},
		{"list", `["blue"]`, true},
		{"map", `{"subnet":"blue"}`, true},
		{"null", `null`, true},
		{"statically invalid call", `"blue" + 1`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []environment.Type{environment.NewExpressions, environment.StoredExpressions} {
				result := ValidateExpression(tc.expression, mode)
				if got := result.Error != nil; got != tc.wantError {
					t.Errorf("ValidateExpression(%q, %s) error = %v, want error %v", tc.expression, mode, result.Error, tc.wantError)
				}
			}
		})
	}
}

// Unknown parameters sizes need finite conservative bounds, and computed
// intermediate sizes must not be mistaken for a one-element input.
func TestValidateExpression_CostEstimate(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		parameters       map[string]any
		want             string
	}{
		{"dynamic ASCII", `parameters.s.matches("a+") ? "yes" : "no"`, map[string]any{"s": strings.Repeat("a", 100000)}, "yes"},
		{"dynamic Unicode", `parameters.s.matches("界+") ? "yes" : "no"`, map[string]any{"s": strings.Repeat("界", 100000)}, "yes"},
		{"computed ASCII", `(parameters.s + parameters.s).endsWith("a") ? "yes" : "no"`, map[string]any{"s": strings.Repeat("a", 100000)}, "yes"},
		{"computed Unicode", `(parameters.s + parameters.s).endsWith("界") ? "yes" : "no"`, map[string]any{"s": strings.Repeat("界", 100000)}, "yes"},
		{"nested input", `parameters.private.s.matches("a+") ? "yes" : "no"`, map[string]any{"private": map[string]any{"s": strings.Repeat("a", 100000)}}, "yes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result := ValidateExpression(tc.expression, environment.NewExpressions)
			if result.Error != nil {
				t.Fatalf("ValidateExpression() error = %v", result.Error)
			}
			if result.MaxCost == 0 || result.MaxCost == math.MaxUint64 {
				t.Fatalf("estimated cost = %d, want finite positive parameters-root bound", result.MaxCost)
			}
			compiled, err := compileExpression(tc.expression, environment.StoredExpressions)
			if err != nil {
				t.Fatalf("compileExpression() error = %v", err)
			}
			program, err := newBudget().program(compiled)
			if err != nil {
				t.Fatalf("budget.program() error = %v", err)
			}
			value, details, err := program.ContextEval(testContext(t), map[string]any{"parameters": tc.parameters})
			if err != nil {
				t.Fatalf("ContextEval() error = %v", err)
			}
			if value != types.String(tc.want) {
				t.Errorf("value = %v, want %q", value, tc.want)
			}
			if details == nil || details.ActualCost() == nil {
				t.Fatal("evaluation did not record actual cost")
			}
			if actual := *details.ActualCost(); actual > result.MaxCost {
				t.Errorf("actual cost %d exceeds estimated maximum %d", actual, result.MaxCost)
			}
		})
	}
}

// The eight-entry admission cap does not apply to a combined pool set.
func TestValidateExtractorSet_Definitions(t *testing.T) {
	entry := resourceapi.SharingAffinityExtractor{Name: "subnet", APIGroup: "*", Kind: "*", Expression: `"blue"`}
	nine := make([]resourceapi.SharingAffinityExtractor, 9)
	for i := range nine {
		nine[i] = entry
		nine[i].Name = fmt.Sprintf("key-%d", i)
	}
	changed := entry
	changed.Expression = `"red"`
	invalid := entry
	invalid.Expression = `parameters[`
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		wantError  bool
	}{
		{"empty", nil, false},
		{"one", []resourceapi.SharingAffinityExtractor{entry}, false},
		{"nine across slices", nine, false},
		{"duplicate equal expressions", []resourceapi.SharingAffinityExtractor{entry, entry}, true},
		{"duplicate different expressions", []resourceapi.SharingAffinityExtractor{entry, changed}, true},
		{"invalid CEL", []resourceapi.SharingAffinityExtractor{invalid}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateExtractorSet(tc.extractors)
			if got := err != nil; got != tc.wantError {
				t.Fatalf("ValidateExtractorSet() error = %v, want error %v", err, tc.wantError)
			}
			if err != nil && !errors.Is(err, ErrInvalidDefinition) {
				t.Errorf("error = %v, want ErrInvalidDefinition", err)
			}
		})
	}
}

func TestValidateExtractorSet_AggregateCostBoundary(t *testing.T) {
	at, over := estimatedCostBoundaryExtractors(t)
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		wantError  bool
	}{
		{"exactly at combined limit", at, false},
		{"one unit over combined limit", over, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateExtractorSet(tc.extractors)
			if got := err != nil; got != tc.wantError {
				t.Fatalf("ValidateExtractorSet() error = %v, want error %v", err, tc.wantError)
			}
			if err != nil {
				if !errors.Is(err, ErrInvalidDefinition) || !errors.Is(err, ErrCostExceeded) {
					t.Errorf("aggregate error = %v, want both ErrInvalidDefinition and ErrCostExceeded", err)
				}
				if got := Classify(err); got != OutcomeInvalidDefinition {
					t.Errorf("Classify() = %v, want OutcomeInvalidDefinition", got)
				}
			}
		})
	}
}

func TestValidateExtractorSet_PerExpressionCost(t *testing.T) {
	extractors := []resourceapi.SharingAffinityExtractor{{
		Name: "subnet", APIGroup: "*", Kind: "*",
		// Both operands have input-dependent size; the maximum product
		// exceeds the budget even though a small concrete input is cheap.
		Expression: `parameters.s.contains(parameters.pattern) ? "x" : ""`,
	}}
	err := ValidateExtractorSet(extractors)
	if !errors.Is(err, ErrInvalidDefinition) || !errors.Is(err, ErrCostExceeded) {
		t.Fatalf("ValidateExtractorSet() error = %v, want invalid definition wrapping cost exceeded", err)
	}
	if got := Classify(err); got != OutcomeInvalidDefinition {
		t.Errorf("Classify() = %v, want OutcomeInvalidDefinition", got)
	}
}

// New publications use admission's environment; runtime validation must
// continue using stored expressions without tolerating invalid definitions.
func TestValidatePublishedExtractorSet(t *testing.T) {
	at, over := estimatedCostBoundaryExtractors(t)
	for _, tc := range []struct {
		name        string
		entries     []resourceapi.SharingAffinityExtractor
		stored      []bool
		wantKind    error
		wantCompile bool
	}{
		{"nil", nil, nil, nil, false},
		{"empty", []resourceapi.SharingAffinityExtractor{}, []bool{}, nil, false},
		{"valid", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"blue"`)}, nil, nil, false},
		{"at-cost-limit", at, nil, nil, false},
		{"over-cost-limit", over, nil, ErrCostExceeded, false},
		{"stored-cost-at-limit", at, []bool{true, true, true}, nil, false},
		{"stored-cost-over-limit", over, []bool{true, true, true}, ErrCostExceeded, false},
		{"mixed-stored-new-over-limit", over, []bool{true, true, false}, ErrCostExceeded, false},
		{"syntax-error", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "parameters[")}, nil, ErrInvalidDefinition, true},
		{"non-string-type", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "1")}, nil, ErrInvalidDefinition, true},
		{"stored-uncompilable-tolerated", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "parameters[")}, []bool{true}, nil, false},
		{"stored-non-string-tolerated", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "1")}, []bool{true}, nil, false},
		{"duplicate", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"a"`), testExtractor("subnet", `"b"`)}, nil, ErrInvalidDefinition, false},
		{"stored-duplicate", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"a"`), testExtractor("subnet", `"a"`)}, []bool{true, true}, ErrInvalidDefinition, false},
		{"markers-too-short", at, []bool{true}, ErrInvalidDefinition, false},
		{"markers-too-long", at, []bool{true, true, true, true}, ErrInvalidDefinition, false},
		{"non-nil-empty-markers", at, []bool{}, ErrInvalidDefinition, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePublishedExtractorSet(tc.entries, tc.stored)
			if tc.wantKind == nil {
				if err != nil {
					t.Fatalf("valid publication rejected: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantKind) || !errors.Is(err, ErrInvalidDefinition) {
				t.Fatalf("error = %v, want invalid definition wrapping %v", err, tc.wantKind)
			}
			var compileErr *CompileError
			if got := errors.As(err, &compileErr); got != tc.wantCompile {
				t.Errorf("compiler cause reachable = %t, want %t", got, tc.wantCompile)
			}
			if tc.wantCompile {
				if !strings.Contains(err.Error(), "subnet") || !strings.Contains(err.Error(), "CEL expression could not be compiled") {
					t.Errorf("error lacks extractor attribution or redacted summary: %v", err)
				}
				if strings.Contains(err.Error(), tc.entries[0].Expression) {
					t.Errorf("compiler source leaked into public error: %v", err)
				}
			}
		})
	}
	t.Run("per-expression-admission-parity", func(t *testing.T) {
		for _, expression := range []string{
			`parameters.?subnetID.orValue("")`, `"blue"`, `parameters.private.subnet`,
			`cel.bind(value, parameters.?subnetID.orValue(""), value)`,
			`parameters[`, `object.x`, `true`, `["blue"]`, `{"subnet":"blue"}`, `null`,
			`parameters.s.contains(parameters.pattern) ? "x" : ""`,
		} {
			t.Run(expression, func(t *testing.T) {
				result := ValidateExpression(expression, environment.NewExpressions)
				err := ValidatePublishedExtractorSet([]resourceapi.SharingAffinityExtractor{testExtractor("subnet", expression)}, nil)
				wantError := result.Error != nil || result.MaxCost > resourceapi.SharingAffinityCELMaxCost
				if (err != nil) != wantError {
					t.Fatalf("publisher = %v, admission compiler = %+v", err, result)
				}
				if result.Error != nil {
					var compileErr *CompileError
					if !errors.As(err, &compileErr) || compileErr.Error() != result.Error.Error() {
						t.Errorf("publisher lost admission compiler diagnostics: %v, want %v", err, result.Error)
					}
				}
			})
		}
	})
	t.Run("environment-selection", func(t *testing.T) {
		entries := []resourceapi.SharingAffinityExtractor{testExtractor("a", `"a"`), testExtractor("b", `"b"`)}
		for _, tc := range []struct {
			name     string
			selector envSelector
			want     []environment.Type
		}{
			{"all-new", publishedEnv(nil), []environment.Type{environment.NewExpressions, environment.NewExpressions}},
			{"per-index", publishedEnv([]bool{true, false}), []environment.Type{environment.StoredExpressions, environment.NewExpressions}},
			{"runtime", storedEnv, []environment.Type{environment.StoredExpressions, environment.StoredExpressions}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var got []environment.Type
				_, err := compileExtractorSet(entries, tc.selector, func(expression string, env environment.Type) (*compiledExpression, error) {
					got = append(got, env)
					return &compiledExpression{maxCost: 1}, nil
				})
				if err != nil {
					t.Fatalf("compileExtractorSet: %v", err)
				}
				if len(got) != len(tc.want) {
					t.Fatalf("compiler calls = %d, want %d", len(got), len(tc.want))
				}
				for i := range got {
					if got[i] != tc.want[i] {
						t.Errorf("environment[%d] = %v, want %v", i, got[i], tc.want[i])
					}
				}
			})
		}
	})
	t.Run("stored-only-expression", func(t *testing.T) {
		// Pin a rollback environment instead of depending on the checkout's
		// moving compatibility version. No parallel tests mutate affinityEnv.
		env, err := environment.MustBaseEnvSet(version.MajorMinor(1, 31)).Extend(environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 31),
			EnvOptions:        []cel.EnvOption{cel.Variable(parametersVar, cel.DynType)},
		})
		if err != nil {
			t.Fatal(err)
		}
		original := affinityEnv
		affinityEnv = func() (*environment.EnvSet, error) { return env, nil }
		t.Cleanup(func() { affinityEnv = original })
		entry := testExtractor("subnet", `[1, 2].includes(2) ? "x" : ""`)
		if got := ValidateExpression(entry.Expression, environment.NewExpressions); got.Error == nil {
			t.Fatal("fixture must be unavailable in the rollback admission environment")
		}
		if got := ValidateExpression(entry.Expression, environment.StoredExpressions); got.Error != nil || got.MaxCost == 0 {
			t.Fatalf("stored-only fixture must compile with positive cost: %+v", got)
		}
		if err := ValidatePublishedExtractorSet([]resourceapi.SharingAffinityExtractor{entry}, []bool{true}); err != nil {
			t.Errorf("unchanged stored-only expression rejected: %v", err)
		}
		if err := ValidatePublishedExtractorSet([]resourceapi.SharingAffinityExtractor{entry}, nil); !errors.Is(err, ErrInvalidDefinition) {
			t.Errorf("new stored-only expression accepted: %v", err)
		}
		if err := ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
			t.Errorf("runtime stopped using StoredExpressions: %v", err)
		}
		entry.Expression = "parameters["
		if err := ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); !errors.Is(err, ErrInvalidDefinition) {
			t.Errorf("runtime incorrectly tolerates compilation failures: %v", err)
		}
	})
	t.Run("stored-uncompilable-cost-zero", func(t *testing.T) {
		entries := append([]resourceapi.SharingAffinityExtractor{testExtractor("legacy", "parameters[")}, at...)
		if err := ValidatePublishedExtractorSet(entries, []bool{true, false, false, false}); err != nil {
			t.Errorf("uncompilable stored expression must contribute zero cost: %v", err)
		}
		entries = append([]resourceapi.SharingAffinityExtractor{testExtractor("legacy", "parameters[")}, over...)
		if err := ValidatePublishedExtractorSet(entries, []bool{true, false, false, false}); !errors.Is(err, ErrCostExceeded) {
			t.Errorf("tolerating one expression bypassed other expressions' cost: %v", err)
		}
	})
	t.Run("stored-cost-counted-and-overflow-safe", func(t *testing.T) {
		entries := []resourceapi.SharingAffinityExtractor{testExtractor("a", `"a"`), testExtractor("b", `"b"`)}
		for _, costs := range [][2]uint64{
			{resourceapi.SharingAffinityCELMaxCost, 1},
			{1, math.MaxUint64},
		} {
			call := 0
			_, err := compileExtractorSet(entries, publishedEnv([]bool{true, false}), func(_ string, env environment.Type) (*compiledExpression, error) {
				cost := costs[call]
				call++
				return &compiledExpression{maxCost: cost}, nil
			})
			if !errors.Is(err, ErrCostExceeded) {
				t.Errorf("costs %v accepted: %v", costs, err)
			}
		}
	})
}
func estimatedCostBoundaryExtractors(t *testing.T) (at, over []resourceapi.SharingAffinityExtractor) {
	t.Helper()
	cost := func(expression string) uint64 {
		t.Helper()
		result := ValidateExpression(expression, environment.NewExpressions)
		if result.Error != nil {
			t.Fatalf("boundary fixture compilation failed: %v", result.Error)
		}
		if len(expression) > resourceapi.SharingAffinityCELExpressionMaxLength {
			t.Fatalf("boundary fixture length %d exceeds expression byte limit", len(expression))
		}
		return result.MaxCost
	}
	regex := func(input, pattern int) string {
		return fmt.Sprintf("'%s'.matches('%s') ? 'x' : 'x'", strings.Repeat("a", input), strings.Repeat("a", pattern))
	}
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	// Measure costs with the compiler under test; no cached probe numbers are
	// assumptions about the current CEL version's estimates.
	low, high := 1, 5000
	for low < high {
		mid := (low + high + 1) / 2
		if 2*cost(regex(4000, mid)) <= limit-10 {
			low = mid
		} else {
			high = mid - 1
		}
	}
	base := regex(4000, low)
	remainder := limit - 2*cost(base)
	low, high = 0, 10000
	for low < high {
		mid := (low + high) / 2
		if cost(regex(mid, 1)) < remainder {
			low = mid + 1
		} else {
			high = mid
		}
	}
	filler := regex(low, 1)
	if got := 2*cost(base) + cost(filler); got != limit {
		t.Fatalf("could not construct exact aggregate boundary: got %d, want %d", got, limit)
	}
	for i, expression := range []string{base, base, filler} {
		if got := cost(expression); got >= limit {
			t.Fatalf("individual fixture cost %d must be below %d", got, limit)
		}
		at = append(at, resourceapi.SharingAffinityExtractor{
			Name: fmt.Sprintf("key-%d", i), APIGroup: "*", Kind: "*", Expression: expression,
		})
	}
	over = append([]resourceapi.SharingAffinityExtractor(nil), at...)
	over[2].Expression = "(" + filler + ") + ''"
	if got := 2*cost(base) + cost(over[2].Expression); got != limit+1 {
		t.Fatalf("could not construct one-unit-over boundary: got %d, want %d", got, limit+1)
	}
	return at, over
}
