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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"math"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"cel.dev/cel-go/cel"
	"cel.dev/cel-go/common/types"
	"cel.dev/cel-go/interpreter"
	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/version"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	sigsjson "sigs.k8s.io/json"
)

// Admission and runtime must use parameters as the root variable, not object.
func TestValidateExpression_ParametersEnvironment(t *testing.T) {
	for _, mode := range []environment.Type{environment.NewExpressions, environment.StoredExpressions} {
		t.Run(string(mode), func(t *testing.T) {
			for _, tc := range []struct {
				name, expression string
				wantErrorDetail  string
			}{
				{"optional access", `parameters.?subnetID.orValue("")`, ""},
				{"driver private field", `parameters.private.subnet`, ""},
				{"guarded wildcard kind", `has(parameters.kind) && parameters.kind == "NICConfig" ? parameters.?subnetID.orValue("") : ""`, ""},
				{"bindings", `cel.bind(value, parameters.?subnetID.orValue(""), value)`, ""},
				{"obsolete object root", `object.x`, "compile CEL expression: ERROR: <input>:1:1: undeclared reference to 'object'\n | object.x\n | ^"},
				{"device root", `device.x`, "compile CEL expression: ERROR: <input>:1:1: undeclared reference to 'device'\n | device.x\n | ^"},
				{"syntax error", `parameters[`, "compile CEL expression: ERROR: <input>:1:12: Syntax error: mismatched input '<EOF>' expecting {'[', '{', '(', '.', '-', '!', '?', 'true', 'false', 'null', NUM_FLOAT, NUM_INT, NUM_UINT, STRING, BYTES, IDENTIFIER}\n | parameters[\n | ...........^"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					admission := ValidateExpression(tc.expression, mode)
					assertCompileError(t, mode, admission.Error, tc.wantErrorDetail)
					_, err := compileExpression(tc.expression, mode)
					assertCompileError(t, mode, err, tc.wantErrorDetail)
				})
			}
		})
	}
}

func TestValidateExpression_OutputType(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		wantErrorDetail  string
	}{
		{"string", `"blue"`, ""},
		{"empty string result", `""`, ""},
		{"dynamic field", `parameters.value`, ""},
		{"empty expression", ``, "compile CEL expression: ERROR: <input>:1:0: Syntax error: mismatched input '<EOF>' expecting {'[', '{', '(', '.', '-', '!', 'true', 'false', 'null', NUM_FLOAT, NUM_INT, NUM_UINT, STRING, BYTES, IDENTIFIER}"},
		{"int", `1`, "CEL expression must evaluate to string, not int"},
		{"double", `1.5`, "CEL expression must evaluate to string, not double"},
		{"bool", `true`, "CEL expression must evaluate to string, not bool"},
		{"bytes", `b"value"`, "CEL expression must evaluate to string, not bytes"},
		{"list", `["blue"]`, "CEL expression must evaluate to string, not list(string)"},
		{"map", `{"subnet":"blue"}`, "CEL expression must evaluate to string, not map(string, string)"},
		{"null", `null`, "CEL expression must evaluate to string, not null_type"},
		{"statically invalid call", `"blue" + 1`, "compile CEL expression: ERROR: <input>:1:8: found no matching overload for '_+_' applied to '(string, int)'\n | \"blue\" + 1\n | .......^"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, mode := range []environment.Type{environment.NewExpressions, environment.StoredExpressions} {
				result := ValidateExpression(tc.expression, mode)
				assertCompileError(t, mode, result.Error, tc.wantErrorDetail)
			}
		})
	}
}

func assertCompileError(t *testing.T, mode environment.Type, err error, wantDetail string) {
	t.Helper()
	if wantDetail == "" {
		if err != nil {
			t.Errorf("%s: unexpected compile error: %v", mode, err)
		}
		return
	}
	var celErr *apiservercel.Error
	if !errors.As(err, &celErr) {
		t.Fatalf("%s: compile error = %v, want an apiserver CEL error", mode, err)
	}
	if celErr.Type != apiservercel.ErrorTypeInvalid {
		t.Errorf("%s: CEL error type = %v, want %v", mode, celErr.Type, apiservercel.ErrorTypeInvalid)
	}
	if celErr.Detail != wantDetail {
		t.Errorf("%s: CEL error detail = %q, want %q", mode, celErr.Detail, wantDetail)
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

// Name uniqueness and the eight-entry admission cap are not CEL constraints.
func TestValidateExpressions_Inputs(t *testing.T) {
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
		{"duplicate equal expressions", []resourceapi.SharingAffinityExtractor{entry, entry}, false},
		{"duplicate different expressions", []resourceapi.SharingAffinityExtractor{entry, changed}, false},
		{"invalid CEL behind duplicate name", []resourceapi.SharingAffinityExtractor{entry, invalid}, true},
		{"invalid CEL", []resourceapi.SharingAffinityExtractor{invalid}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateExpressions(tc.extractors)
			if got := err != nil; got != tc.wantError {
				t.Fatalf("ValidateExpressions() error = %v, want error %v", err, tc.wantError)
			}
			if err != nil && !errors.Is(err, ErrInvalidDefinition) {
				t.Errorf("error = %v, want ErrInvalidDefinition", err)
			}
		})
	}
}

func TestValidateExpressions_AggregateCostBoundary(t *testing.T) {
	at, over := estimatedCostBoundaryExtractors(t)
	withDuplicateNames := func(entries []resourceapi.SharingAffinityExtractor) []resourceapi.SharingAffinityExtractor {
		entries = slices.Clone(entries)
		for i := range entries {
			entries[i].Name = "subnet"
		}
		return entries
	}
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		wantError  bool
	}{
		{"exactly at combined limit", at, false},
		{"one unit over combined limit", over, true},
		{"duplicate names exactly at combined limit", withDuplicateNames(at), false},
		{"duplicate names one unit over combined limit", withDuplicateNames(over), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateExpressions(tc.extractors)
			if got := err != nil; got != tc.wantError {
				t.Fatalf("ValidateExpressions() error = %v, want error %v", err, tc.wantError)
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

func TestValidateExpressions_PerExpressionCost(t *testing.T) {
	extractors := []resourceapi.SharingAffinityExtractor{{
		Name: "subnet", APIGroup: "*", Kind: "*",
		// Both operands have input-dependent size; the maximum product
		// exceeds the budget even though a small concrete input is cheap.
		Expression: `parameters.s.contains(parameters.pattern) ? "x" : ""`,
	}}
	err := ValidateExpressions(extractors)
	if !errors.Is(err, ErrInvalidDefinition) || !errors.Is(err, ErrCostExceeded) {
		t.Fatalf("ValidateExpressions() error = %v, want invalid definition wrapping cost exceeded", err)
	}
	if got := Classify(err); got != OutcomeInvalidDefinition {
		t.Errorf("Classify() = %v, want OutcomeInvalidDefinition", got)
	}
}

// Pool validation shares runtime's environment without admission's ratcheting.
func TestValidateExpressions_Compilation(t *testing.T) {
	at, over := estimatedCostBoundaryExtractors(t)
	for _, tc := range []struct {
		name        string
		entries     []resourceapi.SharingAffinityExtractor
		wantKind    error
		wantCompile bool
	}{
		{"nil", nil, nil, false},
		{"empty", []resourceapi.SharingAffinityExtractor{}, nil, false},
		{"valid", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"blue"`)}, nil, false},
		{"at-cost-limit", at, nil, false},
		{"over-cost-limit", over, ErrCostExceeded, false},
		{"syntax-error", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "parameters[")}, ErrInvalidDefinition, true},
		{"non-string-type", []resourceapi.SharingAffinityExtractor{testExtractor("subnet", "1")}, ErrInvalidDefinition, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateExpressions(tc.entries)
			if tc.wantKind == nil {
				if err != nil {
					t.Fatalf("valid pool rejected: %v", err)
				}
				return
			}
			if !errors.Is(err, tc.wantKind) || !errors.Is(err, ErrInvalidDefinition) {
				t.Fatalf("error = %v, want invalid definition wrapping %v", err, tc.wantKind)
			}
			var compileErr *apiservercel.Error
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
	t.Run("per-expression-compilation-parity", func(t *testing.T) {
		for _, expression := range []string{
			`parameters.?subnetID.orValue("")`, `"blue"`, `parameters.private.subnet`,
			`cel.bind(value, parameters.?subnetID.orValue(""), value)`,
			`parameters[`, `object.x`, `true`, `["blue"]`, `{"subnet":"blue"}`, `null`,
			`parameters.s.contains(parameters.pattern) ? "x" : ""`,
		} {
			t.Run(expression, func(t *testing.T) {
				result := ValidateExpression(expression, environment.StoredExpressions)
				err := ValidateExpressions([]resourceapi.SharingAffinityExtractor{testExtractor("subnet", expression)})
				wantError := result.Error != nil || result.MaxCost > resourceapi.SharingAffinityCELMaxCost
				if (err != nil) != wantError {
					t.Fatalf("pool validation = %v, compiler = %+v", err, result)
				}
				if result.Error != nil {
					var compileErr *apiservercel.Error
					if !errors.As(err, &compileErr) || compileErr.Error() != result.Error.Error() {
						t.Errorf("pool validation lost compiler diagnostics: %v, want %v", err, result.Error)
					}
				}
			})
		}
	})
	t.Run("environment-selection", func(t *testing.T) {
		entries := []resourceapi.SharingAffinityExtractor{testExtractor("a", `"a"`), testExtractor("b", `"b"`)}
		calls := 0
		_, err := compileExpressions(entries, func(expression string, env environment.Type) (*compiledExpression, error) {
			calls++
			if env != environment.StoredExpressions {
				t.Errorf("environment = %v, want StoredExpressions", env)
			}
			return &compiledExpression{maxCost: 1}, nil
		})
		if err != nil {
			t.Fatalf("compileExpressions: %v", err)
		}
		if calls != len(entries) {
			t.Errorf("compiler calls = %d, want %d", calls, len(entries))
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
		if err := ValidateExpressions([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
			t.Errorf("pool validation stopped using StoredExpressions: %v", err)
		}
		entry.Expression = "parameters["
		if err := ValidateExpressions([]resourceapi.SharingAffinityExtractor{entry}); !errors.Is(err, ErrInvalidDefinition) {
			t.Errorf("runtime incorrectly tolerates compilation failures: %v", err)
		}
	})
	t.Run("uncompilable-expression-blocks-pool", func(t *testing.T) {
		entries := append([]resourceapi.SharingAffinityExtractor{testExtractor("legacy", "parameters[")}, at...)
		err := ValidateExpressions(entries)
		var compileErr *apiservercel.Error
		if !errors.Is(err, ErrInvalidDefinition) || !errors.As(err, &compileErr) {
			t.Errorf("uncompilable expression must fail instead of contributing zero cost: %v", err)
		}
	})
	t.Run("cost-counted-and-overflow-safe", func(t *testing.T) {
		entries := []resourceapi.SharingAffinityExtractor{testExtractor("a", `"a"`), testExtractor("b", `"b"`)}
		for _, costs := range [][2]uint64{
			{resourceapi.SharingAffinityCELMaxCost, 1},
			{1, math.MaxUint64},
		} {
			call := 0
			_, err := compileExpressions(entries, func(_ string, env environment.Type) (*compiledExpression, error) {
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

const testDriver = "dra.example.com"

func TestExtract_ParametersEnvironment(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		wantErr          error
	}{
		{"parameters", `parameters.subnetID`, nil},
		{"optional parameters", `parameters.?subnetID.orValue("")`, nil},
		{"guarded parameters", `has(parameters.subnetID) ? parameters.subnetID : ""`, nil},
		{"no object alias", `object.subnetID`, ErrInvalidDefinition},
		{"no device variable", `device.attributes["dra.example.com"].model`, ErrInvalidDefinition},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
				opaqueConfig(nil, `{"subnetID":"blue","private":{"token":"not-a-selector"}}`),
			})
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, OutcomeInvalidDefinition)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
		})
	}
}

func TestExtract_Outcomes(t *testing.T) {
	value := testExtractor("subnet", `parameters.?subnetID.orValue("")`)
	empty := testExtractor("pkey", `""`)
	exact := value
	exact.APIGroup, exact.Kind = "example.com", "Config"
	blue := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`)
	green := opaqueConfig(nil, `{"apiVersion":"example.com/v2","kind":"Config","subnetID":"green"}`)
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		configs    []resourceapi.DeviceClaimConfiguration
		want       Result
		wantErr    error
		outcome    Outcome
	}{
		{name: "empty set", want: Result{Satisfied: true}},
		{name: "empty set skips malformed parameters", configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{`)}, want: Result{Satisfied: true}},
		{name: "no configs", extractors: []resourceapi.SharingAffinityExtractor{value}, want: Result{Constrained: true, Unsatisfied: []string{"subnet"}}},
		{name: "no metadata matches", extractors: []resourceapi.SharingAffinityExtractor{exact}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)}, want: Result{Constrained: true, Unsatisfied: []string{"subnet"}}},
		{name: "all empty", extractors: []resourceapi.SharingAffinityExtractor{empty}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Unsatisfied: []string{"pkey"}}},
		{name: "partial satisfaction", extractors: []resourceapi.SharingAffinityExtractor{value, empty}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Values: map[string]string{"subnet": "blue"}, Unsatisfied: []string{"pkey"}}},
		{name: "full satisfaction", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("pkey", `"p0"`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue", "pkey": "p0"}}},
		{name: "equal contributions", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "empty then value", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`), blue}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "value then empty", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, opaqueConfig(nil, `{}`)}, want: Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}},
		{name: "conflicting contributions", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{blue, green}, wantErr: ErrClaimSelfInconsistent, outcome: OutcomeClaimInconsistent},
		{name: "invalid CEL without configs", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.[`)}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "empty expression", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", ``)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "static non-string", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `42`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrInvalidDefinition, outcome: OutcomeInvalidDefinition},
		{name: "missing field", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "dynamic non-string", extractors: []resourceapi.SharingAffinityExtractor{value}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"subnetID":42}`)}, wantErr: ErrNonStringReturn, outcome: OutcomeRuntime},
		{name: "runtime evaluation failure", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `string(1 / parameters.zero)`)}, configs: []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"zero":0}`)}, wantErr: ErrCELEval, outcome: OutcomeRuntime},
		{name: "later failure discards previous values", extractors: []resourceapi.SharingAffinityExtractor{value, testExtractor("pkey", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "unsatisfied does not hide later runtime error", extractors: []resourceapi.SharingAffinityExtractor{empty, testExtractor("subnet", `parameters.missing`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, wantErr: ErrMissingField, outcome: OutcomeRuntime},
		{name: "unsatisfied names sorted", extractors: []resourceapi.SharingAffinityExtractor{testExtractor("z-last", `""`), testExtractor("a-first", `""`)}, configs: []resourceapi.DeviceClaimConfiguration{blue}, want: Result{Constrained: true, Unsatisfied: []string{"a-first", "z-last"}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, tc.extractors), Request{Name: "req"}, tc.configs)
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, tc.outcome)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, tc.want)
		})
	}
}

func TestExtract_DefinitionBeforeMatching(t *testing.T) {
	for _, expression := range []string{`parameters.[`, `object.x`, `device.x`, `42`, ``} {
		t.Run(expression, func(t *testing.T) {
			entry := testExtractor("subnet", expression)
			entry.APIGroup, entry.Kind = "example.com", "Config"
			for _, raw := range []string{`{"apiVersion":"other.example.com/v1","kind":"Other"}`, `{`} {
				got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
				assertExtractError(t, got, err, ErrInvalidDefinition, OutcomeInvalidDefinition)
			}
		})
	}
}

func TestExtract_NineCombinedExtractors(t *testing.T) {
	slices := make([]resourceapi.ResourceSlice, 2)
	want := map[string]string{}
	for i := range 9 {
		name := fmt.Sprintf("key-%d", i)
		slices[i/8].Spec.SharingAffinity = append(slices[i/8].Spec.SharingAffinity, testExtractor(name, `"value"`))
		want[name] = "value"
	}
	combined := append(slices[0].Spec.SharingAffinity, slices[1].Spec.SharingAffinity...)
	if err := ValidateExpressions(combined); err != nil {
		t.Fatalf("ValidateExpressions(8 + 1 entries) error = %v; eight is a per-slice cap", err)
	}
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, combined), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	if err != nil {
		t.Fatalf("Extract(9 combined entries) error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: want})
}

func TestExtract_Scoping(t *testing.T) {
	for _, tc := range []struct {
		name     string
		req      Request
		requests []string
		driver   string
		opaque   bool
		matches  bool
	}{
		{"whole claim", Request{Name: "main"}, nil, testDriver, true, true},
		{"empty driver", Request{Name: "main"}, nil, "", true, true},
		{"top level", Request{Name: "main"}, []string{"main"}, testDriver, true, true},
		{"unrelated top level", Request{Name: "main"}, []string{"other"}, testDriver, true, false},
		{"subrequest whole claim", Request{Name: "slice", ParentName: "main"}, nil, testDriver, true, true},
		{"subrequest parent", Request{Name: "slice", ParentName: "main"}, []string{"main"}, testDriver, true, true},
		{"qualified subrequest", Request{Name: "slice", ParentName: "main"}, []string{"main/slice"}, testDriver, true, true},
		{"multiple targets", Request{Name: "slice", ParentName: "main"}, []string{"other", "main/slice"}, testDriver, true, true},
		{"bare subrequest not qualified", Request{Name: "slice", ParentName: "main"}, []string{"slice"}, testDriver, true, false},
		{"sibling subrequest", Request{Name: "slice", ParentName: "main"}, []string{"main/other"}, testDriver, true, false},
		{"other parent", Request{Name: "slice", ParentName: "main"}, []string{"other/slice"}, testDriver, true, false},
		{"top level excludes subrequest target", Request{Name: "main"}, []string{"main/slice"}, testDriver, true, false},
		{"other driver", Request{Name: "main"}, nil, "other.example.com", true, false},
		{"non opaque", Request{Name: "main"}, nil, testDriver, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := opaqueConfig(tc.requests, `{"subnetID":"blue"}`)
			cfg.Opaque.Driver = tc.driver
			if !tc.opaque {
				cfg.Opaque = nil
			}
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.subnetID`)})
			got, err := newTestExtractor().Extract(testContext(t), pool, tc.req, []resourceapi.DeviceClaimConfiguration{cfg})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.matches {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_MatchedButMissingFields(t *testing.T) {
	for _, tc := range []struct {
		name, expression, fields string
		wantErr                  error
		wantValue                string
	}{
		{"optional absence", `parameters.?subnetID.orValue("")`, ``, nil, ""},
		{"guarded absence", `has(parameters.subnetID) ? parameters.subnetID : ""`, ``, nil, ""},
		{"unguarded absence", `parameters.subnetID`, ``, ErrMissingField, ""},
		{"renamed field takes precedence", `has(parameters.subnet) ? parameters.subnet : (has(parameters.subnetID) ? parameters.subnetID : "")`, `,"subnet":"blue","subnetID":"old"`, nil, "blue"},
		{"renamed field fallback", `has(parameters.subnet) ? parameters.subnet : (has(parameters.subnetID) ? parameters.subnetID : "")`, `,"subnetID":"blue"`, nil, "blue"},
		{"optional present integer", `parameters.?subnetID.orValue("")`, `,"subnetID":7`, ErrNonStringReturn, ""},
		{"present boolean", `parameters.subnetID`, `,"subnetID":true`, ErrNonStringReturn, ""},
		{"present null", `parameters.subnetID`, `,"subnetID":null`, ErrNonStringReturn, ""},
		{"present object", `parameters.subnetID`, `,"subnetID":{}`, ErrNonStringReturn, ""},
		{"present list", `parameters.subnetID`, `,"subnetID":[]`, ErrNonStringReturn, ""},
		{"wrong type method", `parameters.subnetID.lowerAscii()`, `,"subnetID":7`, ErrCELEval, ""},
		{"private nested field available", `parameters.private.subnetID`, `,"private":{"subnetID":"blue"}`, nil, "blue"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entry := testExtractor("subnet", tc.expression)
			entry.APIGroup, entry.Kind = "example.com", "Config"
			raw := `{"apiVersion":"example.com/v2","kind":"Config"` + tc.fields + `}`
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
			if tc.wantErr != nil {
				assertExtractError(t, got, err, tc.wantErr, OutcomeRuntime)
				return
			}
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.wantValue != "" {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": tc.wantValue}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_WildcardKindGuard(t *testing.T) {
	entry := testExtractor("subnet", `has(parameters.kind) && (parameters.kind == "NICConfig" || parameters.kind == "NICConfig2") ? parameters.subnetID : ""`)
	entry.APIGroup = "example.com"
	for _, tc := range []struct {
		name, raw string
		satisfied bool
	}{
		{"first kind", `{"apiVersion":"example.com/v1","kind":"NICConfig","subnetID":"blue"}`, true},
		{"second kind", `{"apiVersion":"example.com/v2","kind":"NICConfig2","subnetID":"blue"}`, true},
		{"missing kind", `{"apiVersion":"example.com/v1"}`, false},
		{"unrelated kind", `{"apiVersion":"example.com/v1","kind":"Other"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.satisfied {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_MultiVersionGuard(t *testing.T) {
	entry := testExtractor("subnet", `parameters.apiVersion == "example.com/v1" ? parameters.subnetID : ""`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"example.com/v1beta1","kind":"Config","subnetID":"old"}`),
		opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`),
	})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
}

func TestExtract_LargeIntegers(t *testing.T) {
	for _, id := range []string{"9007199254740993", "9223372036854775807", "-9223372036854775808"} {
		t.Run(id, func(t *testing.T) {
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("id", `string(parameters.id)`)})
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"id":`+id+`}`)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"id": id}})
		})
	}
}

func TestExtract_MalformedParameters(t *testing.T) {
	for _, tc := range []struct{ name, raw string }{
		{"empty", ""}, {"whitespace", " \t\n"}, {"null", "null"}, {"array", "[]"},
		{"string", `"s3cr3t-marker"`}, {"number", "1"}, {"boolean", "true"},
		{"truncated", `{"secret":"s3cr3t-marker"`}, {"trailing data", `{} {}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, scope := range []string{"in scope", "other driver", "other request"} {
				t.Run(scope, func(t *testing.T) {
					cfg := opaqueConfig(nil, tc.raw)
					if scope == "other driver" {
						cfg.Opaque.Driver = "other.example.com"
					}
					if scope == "other request" {
						cfg.Requests = []string{"other"}
					}
					pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"blue"`)})
					got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{cfg})
					if scope == "in scope" {
						assertExtractError(t, got, err, ErrParameters, OutcomeRuntime)
						if strings.Contains(err.Error(), "s3cr3t-marker") {
							t.Error("error disclosed opaque parameters")
						}
						return
					}
					if err != nil {
						t.Fatalf("out-of-scope parameters were decoded: %v", err)
					}
					assertExtractResult(t, got, Result{Constrained: true, Unsatisfied: []string{"subnet"}})
				})
			}
		})
	}
}

func TestExtract_ErrorRedaction(t *testing.T) {
	// Integer-preserving JSON falls back to float64 beyond int64, so the
	// token must overflow float64 to exercise a decoder error.
	overflow := strings.Repeat("1234567890", 40)
	for _, tc := range []struct {
		name, expression, raw, secret  string
		wantErr                        error
		jsonTypeError, jsonSyntaxError bool
	}{
		{"numeric overflow", `"blue"`, `{"n":` + overflow + `}`, overflow, ErrParameters, true, false},
		{"dynamic missing key", `parameters[parameters.secret]`, `{"secret":"s3cr3t-marker"}`, "s3cr3t-marker", ErrMissingField, false, false},
		{"JSON syntax", `"blue"`, `{"secret":"s3cr3t-marker","broken":}`, "s3cr3t-marker", ErrParameters, false, true},
		{"dynamic conversion", `string(int(parameters.secret))`, `{"secret":"s3cr3t-marker"}`, "s3cr3t-marker", ErrCELEval, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(10)))
			ctx := klog.NewContext(context.Background(), logger)
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			assertExtractError(t, got, err, tc.wantErr, OutcomeRuntime)
			if strings.Contains(err.Error(), tc.secret) || strings.Contains(err.Error(), tc.raw) {
				t.Error("error disclosed opaque parameters")
			}
			buffer := logger.GetSink().(ktesting.Underlier).GetBuffer()
			logs := buffer.String()
			if logs == "" {
				t.Error("expected captured extraction logs at verbosity 10")
			}
			if strings.Contains(logs, tc.secret) || strings.Contains(logs, tc.raw) {
				t.Error("logs disclosed opaque parameters")
			}
			if tc.jsonTypeError {
				var cause *json.UnmarshalTypeError
				if !errors.As(err, &cause) {
					t.Fatalf("error lost JSON type-error cause: %v", err)
				}
				if !strings.Contains(cause.Value, tc.secret) {
					t.Error("redaction discarded the original numeric decoder cause")
				}
			}
			if tc.jsonSyntaxError {
				found := false
				for cause := err; cause != nil; cause = errors.Unwrap(cause) {
					if syntax, offset := sigsjson.SyntaxErrorOffset(cause); syntax && offset > 0 {
						found = true
					}
				}
				if !found {
					t.Fatalf("error lost JSON syntax-error cause: %v", err)
				}
			}
			if tc.wantErr == ErrMissingField {
				found := false
				for cause := errors.Unwrap(err); cause != nil; cause = errors.Unwrap(cause) {
					if strings.Contains(cause.Error(), tc.secret) {
						found = true
					}
				}
				if !found {
					t.Error("redaction discarded the underlying CEL missing-key error")
				}
			}
		})
	}
}

func TestExtract_LoggingRedaction(t *testing.T) {
	logger := ktesting.NewLogger(t, ktesting.NewConfig(ktesting.BufferLogs(true), ktesting.Verbosity(10)))
	ctx := klog.NewContext(context.Background(), logger)
	entry := testExtractor("subnet", `parameters.subnetID`)
	raw := `{"subnetID":"blue","privateToken":"s3cr3t-marker"}`
	got, err := newTestExtractor().Extract(ctx, testPool(7, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, raw)})
	if err != nil {
		t.Fatalf("Extract() error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
	logs := logger.GetSink().(ktesting.Underlier).GetBuffer().String()
	if logs == "" {
		t.Fatal("expected captured extraction logs at verbosity 10")
	}
	for _, forbidden := range []string{raw, "s3cr3t-marker", entry.Expression} {
		if strings.Contains(logs, forbidden) {
			t.Errorf("logs contain private payload or expression %q", forbidden)
		}
	}
}

func TestExtract_Cancellation(t *testing.T) {
	entry := testExtractor("subnet", `parameters.subnetID`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	matching := opaqueConfig(nil, `{"apiVersion":"example.com/v1","kind":"Config","subnetID":"blue"}`)
	for _, tc := range []struct {
		name    string
		configs []resourceapi.DeviceClaimConfiguration
	}{
		{"matching", []resourceapi.DeviceClaimConfiguration{matching}},
		{"zero configs", nil},
		{"out of scope", []resourceapi.DeviceClaimConfiguration{opaqueConfig([]string{"other"}, `{`)}},
		{"metadata nonmatch", []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"kind":"Other"}`)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cause := errors.New("test cancellation cause")
			ctx, cancel := context.WithCancelCause(testContext(t))
			cancel(cause)
			got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, tc.configs)
			assertExtractError(t, got, err, context.Canceled, OutcomeRuntime)
			if !errors.Is(err, cause) {
				t.Fatalf("Extract() error = %v, want cancellation cause", err)
			}
		})
	}
	t.Run("deadline and cause", func(t *testing.T) {
		cause := errors.New("test deadline cause")
		ctx, cancel := context.WithDeadlineCause(testContext(t), time.Now().Add(-time.Second), cause)
		defer cancel()
		got, err := newTestExtractor().Extract(ctx, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{matching})
		assertExtractError(t, got, err, context.DeadlineExceeded, OutcomeRuntime)
		if !errors.Is(err, cause) {
			t.Fatalf("Extract() error = %v, want deadline cause", err)
		}
	})
	t.Run("empty extractor fast path", func(t *testing.T) {
		ctx, cancel := context.WithCancel(testContext(t))
		cancel()
		got, err := newTestExtractor().Extract(ctx, testPool(1, nil), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{`)})
		if err != nil {
			t.Fatalf("empty extractor set consulted cancellation: %v", err)
		}
		assertExtractResult(t, got, Result{Satisfied: true})
	})
}

func TestExtract_CancellationDuringEvaluation(t *testing.T) {
	// CEL obtains Done inside ContextEval, after Extract's pre-evaluation
	// checks. Cancel there to exercise the real interpreter interrupt without
	// racing a timer against compilation or exhausting the cost budget.
	cause := errors.New("interpreter cancellation cause")
	ctx, cancel := context.WithCancelCause(testContext(t))
	defer cancel(nil)
	interrupt := &cancelOnEvalContext{Context: ctx, cancel: func() { cancel(cause) }}
	items := strings.TrimSuffix(strings.Repeat(`1,`, 4096), ",")
	entry := testExtractor("subnet", `[`+items+`].all(x, x == 1) ? "blue" : ""`)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	if err := ValidateExpressions(pool.Extractors); err != nil {
		t.Fatalf("cancellation fixture must be statically valid: %v", err)
	}
	got, err := newTestExtractor().Extract(interrupt, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	assertExtractError(t, got, err, context.Canceled, OutcomeRuntime)
	if !errors.Is(err, cause) {
		t.Fatalf("Extract() error = %v, want interpreter cancellation cause", err)
	}
	var detail *Error
	if !errors.As(err, &detail) || detail.Extractor != "subnet" || detail.ConfigIndex != 0 {
		t.Fatalf("cancellation lost evaluation attribution: %#v", detail)
	}
}

type cancelOnEvalContext struct {
	context.Context
	once   sync.Once
	cancel func()
}

func (c *cancelOnEvalContext) Done() <-chan struct{} {
	c.once.Do(c.cancel)
	return c.Context.Done()
}

func TestExtract_TypedErrorPropagation(t *testing.T) {
	for _, tc := range []struct {
		name, expression string
		raws             []string
		want             error
		index            int
	}{
		{"missing field", `parameters.subnetID`, []string{`{"subnetID":"blue"}`, `{}`}, ErrMissingField, 1},
		{"non-string", `parameters.subnetID`, []string{`{"subnetID":7}`}, ErrNonStringReturn, 0},
		{"evaluation", `string(1 / parameters.zero)`, []string{`{"zero":0}`}, ErrCELEval, 0},
		{"parameters", `"blue"`, []string{`{}`, `{`}, ErrParameters, 1},
		{"inconsistent", `parameters.subnetID`, []string{`{"subnetID":"blue"}`, `{"subnetID":"green"}`}, ErrClaimSelfInconsistent, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pool := testPool(7, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", tc.expression)})
			configs := []resourceapi.DeviceClaimConfiguration{opaqueConfig([]string{"unrelated"}, `{`)}
			for _, raw := range tc.raws {
				configs = append(configs, opaqueConfig(nil, raw))
			}
			got, err := newTestExtractor().Extract(testContext(t), pool, Request{Name: "slice", ParentName: "main"}, configs)
			outcome := OutcomeRuntime
			if tc.want == ErrClaimSelfInconsistent {
				outcome = OutcomeClaimInconsistent
			}
			assertExtractError(t, got, err, tc.want, outcome)
			var detail *Error
			if !errors.As(err, &detail) {
				t.Fatalf("Extract() error %v does not wrap *Error", err)
			}
			if detail.Pool != pool.Key || detail.Request != "main/slice" || detail.ConfigIndex != tc.index+1 {
				t.Errorf("error attribution = %#v, want pool %#v, request main/slice, original config index %d", detail, pool.Key, tc.index+1)
			}
			if tc.want != ErrParameters && detail.Extractor != "subnet" {
				t.Errorf("extractor attribution = %q, want subnet", detail.Extractor)
			}
		})
	}
}

func TestError_WrappingAndAttribution(t *testing.T) {
	for _, kind := range []error{ErrClaimSelfInconsistent, ErrInvalidDefinition, ErrParameters, ErrCELEval, ErrNonStringReturn, ErrCostExceeded, ErrMissingField} {
		t.Run(kind.Error(), func(t *testing.T) {
			cause := errors.New("underlying cause")
			detail := &Error{Kind: kind, Pool: PoolKey{Driver: testDriver, Name: "pool-a", Generation: 7}, Request: "main/slice", Extractor: "subnet", ConfigIndex: 2, Err: cause}
			err := fmt.Errorf("outer wrapper: %w", detail)
			if !errors.Is(err, kind) || !errors.Is(err, cause) {
				t.Errorf("wrapped error does not preserve kind and cause: %v", err)
			}
			var got *Error
			if !errors.As(err, &got) || got != detail || errors.Unwrap(detail) != cause {
				t.Error("typed error or direct cause was lost")
			}
			for _, text := range []string{testDriver, "pool-a", "7", "main/slice", "subnet", "2"} {
				if !strings.Contains(err.Error(), text) {
					t.Errorf("error %q lacks attribution %q", err, text)
				}
			}
		})
	}
}

func TestClassify_Outcomes(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want Outcome
	}{
		{"claim inconsistency", ErrClaimSelfInconsistent, OutcomeClaimInconsistent},
		{"invalid definition", ErrInvalidDefinition, OutcomeInvalidDefinition},
		{"estimated cost is definition error", &Error{Kind: ErrInvalidDefinition, Err: ErrCostExceeded}, OutcomeInvalidDefinition},
		{"definition takes precedence", errors.Join(ErrClaimSelfInconsistent, ErrInvalidDefinition), OutcomeInvalidDefinition},
		{"parameters", ErrParameters, OutcomeRuntime},
		{"evaluation", ErrCELEval, OutcomeRuntime},
		{"missing field", ErrMissingField, OutcomeRuntime},
		{"non-string", ErrNonStringReturn, OutcomeRuntime},
		{"runtime cost", ErrCostExceeded, OutcomeRuntime},
		{"cancellation", context.Canceled, OutcomeRuntime},
		{"deadline", context.DeadlineExceeded, OutcomeRuntime},
		{"unknown", errors.New("unexpected failure"), OutcomeRuntime},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, err := range []error{tc.err, fmt.Errorf("wrapped: %w", tc.err)} {
				if got := Classify(err); got != tc.want {
					t.Errorf("Classify(%v) = %v, want %v", err, got, tc.want)
				}
			}
		})
	}
}

func TestExtract_StoredExpressionLengthCompatibility(t *testing.T) {
	value := strings.Repeat("x", resourceapi.SharingAffinityCELExpressionMaxLength+1)
	entry := testExtractor("subnet", `"`+value+`"`)
	if compiled, err := compileExpression(entry.Expression, environment.StoredExpressions); err != nil || compiled == nil {
		t.Fatalf("stored long expression must compile without admission length check: %v", err)
	}
	got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
	if err != nil {
		t.Fatalf("Extract() newly rejected stored expression by length: %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": value}})
}

func testContext(t *testing.T) context.Context {
	t.Helper()
	_, ctx := ktesting.NewTestContext(t)
	return ctx
}

func newTestExtractor() *Extractor {
	return NewExtractor(32)
}

func testPool(generation int64, extractors []resourceapi.SharingAffinityExtractor) PoolExtractors {
	return PoolExtractors{Key: PoolKey{Driver: testDriver, Name: "pool-a", Generation: generation}, Extractors: extractors}
}

func testExtractor(name, expression string) resourceapi.SharingAffinityExtractor {
	return resourceapi.SharingAffinityExtractor{Name: name, APIGroup: "*", Kind: "*", Expression: expression}
}

func opaqueConfig(requests []string, raw string) resourceapi.DeviceClaimConfiguration {
	return resourceapi.DeviceClaimConfiguration{
		Requests: requests,
		DeviceConfiguration: resourceapi.DeviceConfiguration{
			Opaque: &resourceapi.OpaqueDeviceConfiguration{Driver: testDriver, Parameters: runtime.RawExtension{Raw: []byte(raw)}},
		},
	}
}

func assertExtractResult(t *testing.T, got, want Result) {
	t.Helper()
	if got.Constrained != want.Constrained || got.Satisfied != want.Satisfied || !maps.Equal(got.Values, want.Values) || !slices.Equal(got.Unsatisfied, want.Unsatisfied) {
		t.Errorf("Extract() = %#v, want %#v", got, want)
	}
}

func assertExtractError(t *testing.T, got Result, err, want error, outcome Outcome) {
	t.Helper()
	if !errors.Is(err, want) {
		t.Fatalf("Extract() error = %v, want errors.Is(..., %v)", err, want)
	}
	if !reflect.DeepEqual(got, Result{}) {
		t.Errorf("failed extraction returned nonzero Result: %#v", got)
	}
	if actual := Classify(err); actual != outcome {
		t.Errorf("Classify(%v) = %v, want %v", err, actual, outcome)
	}
}

func TestMetadataOf_Fields(t *testing.T) {
	for _, tc := range []struct {
		name   string
		params map[string]any
		want   configMetadata
	}{
		{"nil", nil, configMetadata{}},
		{"empty", map[string]any{}, configMetadata{}},
		{"v1", map[string]any{"apiVersion": "example.com/v1", "kind": "Config"}, configMetadata{group: "example.com", kind: "Config", hasGroup: true, hasKind: true}},
		{"v2", map[string]any{"apiVersion": "example.com/v2", "kind": "Config"}, configMetadata{group: "example.com", kind: "Config", hasGroup: true, hasKind: true}},
		{"groupless", map[string]any{"apiVersion": "v1"}, configMetadata{}},
		{"empty version", map[string]any{"apiVersion": ""}, configMetadata{}},
		{"empty group", map[string]any{"apiVersion": "/v1"}, configMetadata{}},
		{"last slash", map[string]any{"apiVersion": "example.com/extra/v1"}, configMetadata{group: "example.com/extra", hasGroup: true}},
		{"ignore version suffix", map[string]any{"apiVersion": "example.com/"}, configMetadata{group: "example.com", hasGroup: true}},
		{"kind only", map[string]any{"kind": "Config"}, configMetadata{kind: "Config", hasKind: true}},
		{"integer version", map[string]any{"apiVersion": int64(5), "kind": "Config"}, configMetadata{kind: "Config", hasKind: true}},
		{"boolean kind", map[string]any{"apiVersion": "example.com/v1", "kind": true}, configMetadata{group: "example.com", hasGroup: true}},
		{"nulls", map[string]any{"apiVersion": nil, "kind": nil}, configMetadata{}},
		{"structured metadata", map[string]any{"apiVersion": map[string]any{"group": "example.com"}, "kind": []any{"Config"}}, configMetadata{}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := metadataOf(tc.params); got != tc.want {
				t.Errorf("metadataOf(%#v) = %#v, want %#v", tc.params, got, tc.want)
			}
		})
	}
}

func TestMatcher_Matrix(t *testing.T) {
	for _, tc := range []struct {
		name, group, kind, raw string
		want                   bool
	}{
		{"exact v1", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Config"}`, true},
		{"exact v2", "example.com", "Config", `{"apiVersion":"example.com/v2","kind":"Config"}`, true},
		{"wrong group", "example.com", "Config", `{"apiVersion":"other.example.com/v1","kind":"Config"}`, false},
		{"group case mismatch", "example.com", "Config", `{"apiVersion":"EXAMPLE.com/v1","kind":"Config"}`, false},
		{"wrong kind", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"Other"}`, false},
		{"kind case mismatch", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":"config"}`, false},
		{"group wildcard", "*", "Config", `{"apiVersion":"other.example.com/v1","kind":"Config"}`, true},
		{"group wildcard still checks kind", "*", "Config", `{"kind":"Other"}`, false},
		{"kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1","kind":"Other"}`, true},
		{"kind wildcard still checks group", "example.com", "*", `{"apiVersion":"other.example.com/v1"}`, false},
		{"both wildcards", "*", "*", `{"apiVersion":"other.example.com/v1","kind":"Other"}`, true},
		{"no metadata wildcards", "*", "*", `{}`, true},
		{"missing apiVersion exact", "example.com", "Config", `{"kind":"Config"}`, false},
		{"missing apiVersion wildcard", "*", "Config", `{"kind":"Config"}`, true},
		{"groupless exact", "example.com", "Config", `{"apiVersion":"v1","kind":"Config"}`, false},
		{"groupless wildcard", "*", "Config", `{"apiVersion":"v1","kind":"Config"}`, true},
		{"missing kind exact", "example.com", "Config", `{"apiVersion":"example.com/v1"}`, false},
		{"missing kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1"}`, true},
		{"numeric apiVersion exact", "example.com", "Config", `{"apiVersion":5,"kind":"Config"}`, false},
		{"numeric apiVersion wildcard", "*", "Config", `{"apiVersion":5,"kind":"Config"}`, true},
		{"boolean kind exact", "example.com", "Config", `{"apiVersion":"example.com/v1","kind":true}`, false},
		{"boolean kind wildcard", "example.com", "*", `{"apiVersion":"example.com/v1","kind":true}`, true},
		{"null metadata exact", "example.com", "Config", `{"apiVersion":null,"kind":null}`, false},
		{"null metadata wildcard", "*", "*", `{"apiVersion":null,"kind":null}`, true},
		{"structured metadata wildcard", "*", "*", `{"apiVersion":{},"kind":[]}`, true},
		{"structured metadata exact", "example.com", "Config", `{"apiVersion":{},"kind":[]}`, false},
		{"empty metadata exact", "example.com", "Config", `{"apiVersion":"","kind":""}`, false},
		{"empty metadata wildcard", "*", "*", `{"apiVersion":"","kind":""}`, true},
		{"extra slash not valid group", "example.com", "Config", `{"apiVersion":"example.com/extra/v1","kind":"Config"}`, false},
		{"extra slash wildcard", "*", "Config", `{"apiVersion":"example.com/extra/v1","kind":"Config"}`, true},
		{"empty group exact", "example.com", "Config", `{"apiVersion":"/v1","kind":"Config"}`, false},
		{"empty group wildcard", "*", "Config", `{"apiVersion":"/v1","kind":"Config"}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var params map[string]any
			if err := json.Unmarshal([]byte(tc.raw), &params); err != nil {
				t.Fatalf("invalid fixture: %v", err)
			}
			if got := (matcher{apiGroup: tc.group, kind: tc.kind}).matches(metadataOf(params)); got != tc.want {
				t.Errorf("matches(%s) = %v, want %v", tc.raw, got, tc.want)
			}
			entry := testExtractor("subnet", `"blue"`)
			entry.APIGroup, entry.Kind = tc.group, tc.kind
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, []resourceapi.SharingAffinityExtractor{entry}), Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, tc.raw)})
			if err != nil {
				t.Fatalf("Extract() error = %v", err)
			}
			want := Result{Constrained: true, Unsatisfied: []string{"subnet"}}
			if tc.want {
				want = Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}}
			}
			assertExtractResult(t, got, want)
		})
	}
}

func TestExtract_NonmatchDoesNotEvaluate(t *testing.T) {
	entry := testExtractor("subnet", `parameters.mustExist`)
	entry.APIGroup, entry.Kind = "example.com", "Config"
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	for _, raw := range []string{
		`{"apiVersion":"other.example.com/v1","kind":"Config"}`,
		`{"apiVersion":"example.com/v1","kind":"Other"}`,
		`{"apiVersion":"example.com/v1","kind":"config"}`,
		`{"kind":"Config"}`,
		`{"apiVersion":"v1","kind":"Config"}`,
		`{"apiVersion":"example.com/v1"}`,
		`{"apiVersion":7,"kind":true}`,
		`{}`,
	} {
		t.Run(raw, func(t *testing.T) {
			cfg := opaqueConfig(nil, raw)
			extractor := newTestExtractor()
			got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{cfg})
			if err != nil {
				t.Fatalf("metadata nonmatch evaluated missing-field expression: %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Unsatisfied: []string{"subnet"}})
			got, err = extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{
				cfg,
				opaqueConfig(nil, `{"apiVersion":"example.com/v2","kind":"Config","mustExist":"blue"}`),
				cfg,
			})
			if err != nil {
				t.Fatalf("unrelated config prevented matching extraction: %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
		})
	}
}

func TestBudget_ChargeDoesNotOverflow(t *testing.T) {
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	for _, tc := range []struct {
		name    string
		charges []uint64
		spent   uint64
	}{
		{"zero cost", []uint64{0}, 0},
		{"partial", []uint64{1, 2}, 3},
		{"exact boundary", []uint64{limit - 1, 1, 0}, limit},
		{"defensive saturation", []uint64{limit - 1, math.MaxUint64, 1}, limit},
	} {
		t.Run(tc.name, func(t *testing.T) {
			b := newBudget()
			for _, charge := range tc.charges {
				b.charge(charge)
			}
			if b.spent != tc.spent || b.remaining != limit-tc.spent {
				t.Errorf("budget = spent %d, remaining %d; want %d, %d", b.spent, b.remaining, tc.spent, limit-tc.spent)
			}
		})
	}
}

// A single statically valid expression must not receive a fresh limit
// for each config. The input sizes below are measured, not fixed CEL costs.
func TestExtract_SharedRuntimeBudgetNoFreshAllowance(t *testing.T) {
	entry, compiled := runtimeBudgetExpression(t)
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	raw, parameters := parametersWithActualCost(t, compiled, 2*limit/5)
	b := newBudget()
	for i := 0; i < 2; i++ {
		if _, err := evaluateBudget(t, b, compiled, parameters); err != nil {
			t.Fatalf("matching evaluation %d error = %v", i, err)
		}
	}
	spent, remaining := b.spent, b.remaining
	failedCost, err := evaluateBudget(t, b, compiled, parameters)
	var cancelled interpreter.EvalCancelledError
	if !errors.As(err, &cancelled) || cancelled.Cause != interpreter.CostLimitExceeded {
		t.Fatalf("third evaluation error = %v, want CEL CostLimitExceeded", err)
	}
	// CEL checks after incrementing cost. The failed observation can exceed
	// remaining; only successful cost is charged.
	if failedCost <= remaining {
		t.Errorf("failed evaluation actual cost = %d, want greater than remaining %d", failedCost, remaining)
	}
	if b.spent != spent || b.remaining != remaining || b.spent > limit {
		t.Errorf("failed evaluation changed budget: spent %d/%d, remaining %d/%d", b.spent, spent, b.remaining, remaining)
	}

	extractor := newTestExtractor()
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	config := opaqueConfig(nil, raw)
	for count := 1; count <= 3; count++ {
		t.Run(fmt.Sprintf("configs-%d", count), func(t *testing.T) {
			configs := make([]resourceapi.DeviceClaimConfiguration, count)
			for i := range configs {
				configs[i] = config
			}
			got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, configs)
			if count < 3 {
				if err != nil || !got.Constrained || !got.Satisfied || got.Values[entry.Name] != "x" {
					t.Fatalf("Extract() = %#v, %v; want satisfied constrained result", got, err)
				}
			} else {
				if !errors.Is(err, ErrCostExceeded) || Classify(err) != OutcomeRuntime {
					t.Fatalf("Extract() error = %v, want runtime ErrCostExceeded", err)
				}
				if !reflect.DeepEqual(got, Result{}) {
					t.Errorf("failed extraction returned usable result %#v", got)
				}
				var attributed *Error
				if !errors.As(err, &attributed) || attributed.Extractor != entry.Name || attributed.ConfigIndex != 2 {
					t.Errorf("error attribution = %#v, want extractor %q, config 2", attributed, entry.Name)
				}
			}
		})
	}
}

func TestExtract_BudgetZeroCostAtBoundary(t *testing.T) {
	entry, compiled := runtimeBudgetExpression(t)
	limit := uint64(resourceapi.SharingAffinityCELMaxCost)
	raw, large := parametersWithActualCost(t, compiled, 2*limit/5)
	lastRaw, last := parametersWithActualCost(t, compiled, limit-4*limit/5)
	b := newBudget()
	for _, parameters := range []map[string]any{large, large, last} {
		if _, err := evaluateBudget(t, b, compiled, parameters); err != nil {
			t.Fatalf("evaluation at boundary failed: %v", err)
		}
	}
	if b.remaining != 0 || b.spent != limit {
		t.Fatalf("boundary budget = remaining %d, spent %d; want 0, %d", b.remaining, b.spent, limit)
	}
	zero, err := compileExpression(`"k"`, environment.StoredExpressions)
	if err != nil {
		t.Fatal(err)
	}
	if actual, err := evaluateBudget(t, b, zero, nil); err != nil || actual != 0 {
		t.Fatalf("constant at exhausted budget: cost %d, error %v; want 0, nil", actual, err)
	}
	positive, err := compileExpression(`parameters.value`, environment.StoredExpressions)
	if err != nil {
		t.Fatal(err)
	}
	_, err = evaluateBudget(t, b, positive, map[string]any{"value": "x"})
	var cancelled interpreter.EvalCancelledError
	if !errors.As(err, &cancelled) || cancelled.Cause != interpreter.CostLimitExceeded {
		t.Fatalf("positive-cost evaluation at exhausted budget error = %v, want CostLimitExceeded", err)
	}
	if b.remaining != 0 || b.spent != limit {
		t.Errorf("failed boundary evaluation changed successful cost: %#v", b)
	}

	configs := []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, raw), opaqueConfig(nil, raw), opaqueConfig(nil, lastRaw),
		// This would dereference an absent field if the matcher invoked CEL.
		opaqueConfig(nil, `{"apiVersion":"unrelated.example.com/v2","kind":"Config"}`),
	}
	constant := entry
	constant.Name, constant.Expression = "constant", `"k"`
	extra := entry
	extra.Name, extra.Expression = "positive", `parameters.value`
	for _, tc := range []struct {
		name       string
		extractors []resourceapi.SharingAffinityExtractor
		wantError  bool
	}{
		{"exact limit without later evaluation", []resourceapi.SharingAffinityExtractor{entry}, false},
		{"zero-cost and nonmatch after limit", []resourceapi.SharingAffinityExtractor{entry, constant}, false},
		{"positive-cost after limit", []resourceapi.SharingAffinityExtractor{entry, constant, extra}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := ValidateExpressions(tc.extractors); err != nil {
				t.Fatalf("runtime fixture is not statically valid: %v", err)
			}
			got, err := newTestExtractor().Extract(testContext(t), testPool(1, tc.extractors), Request{Name: "req"}, configs)
			if tc.wantError {
				if !errors.Is(err, ErrCostExceeded) {
					t.Fatalf("Extract() error = %v, want ErrCostExceeded", err)
				}
				if !reflect.DeepEqual(got, Result{}) {
					t.Errorf("failed Extract() returned %#v, want zero Result", got)
				}
				return
			}
			if err != nil || !got.Constrained || !got.Satisfied || len(got.Values) != len(tc.extractors) {
				t.Errorf("Extract() = %#v, %v; want all extractors satisfied exactly at limit", got, err)
			}
		})
	}
}

// Reuse programs concurrently, never the mutable request budget.
func TestExtract_BudgetIsolationConcurrent(t *testing.T) {
	entry, compiled := runtimeBudgetExpression(t)
	raw, _ := parametersWithActualCost(t, compiled, 2*uint64(resourceapi.SharingAffinityCELMaxCost)/5)
	config := opaqueConfig(nil, raw)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	extractor := newTestExtractor()
	ctx := testContext(t)
	var wg sync.WaitGroup
	start := make(chan struct{})
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			configs := []resourceapi.DeviceClaimConfiguration{config, config}
			if i%2 == 0 {
				configs = append(configs, config)
			}
			got, err := extractor.Extract(ctx, pool, Request{Name: fmt.Sprintf("req-%d", i)}, configs)
			if i%2 == 0 {
				if !errors.Is(err, ErrCostExceeded) {
					t.Errorf("request %d error = %v, want ErrCostExceeded", i, err)
				}
			} else if err != nil || !got.Constrained || !got.Satisfied || got.Values[entry.Name] != "x" {
				t.Errorf("request %d inherited another request's budget: %#v, %v", i, got, err)
			}
		}(i)
	}
	close(start)
	wg.Wait()
}

func runtimeBudgetExpression(t *testing.T) (resourceapi.SharingAffinityExtractor, *compiledExpression) {
	t.Helper()
	entry := resourceapi.SharingAffinityExtractor{
		Name: "subnet", APIGroup: "example.com", Kind: "Config",
		Expression: `parameters.s.matches("a{1,3}") ? "x" : "x"`,
	}
	if err := ValidateExpressions([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
		t.Fatalf("shared runtime-budget fixture must pass static validation: %v", err)
	}
	compiled, err := compileExpression(entry.Expression, environment.StoredExpressions)
	if err != nil {
		t.Fatalf("compile runtime budget fixture: %v", err)
	}
	return entry, compiled
}

func parametersWithActualCost(t *testing.T, compiled *compiledExpression, target uint64) (string, map[string]any) {
	t.Helper()
	parameters := func(size int) map[string]any {
		return map[string]any{"apiVersion": "example.com/v1", "kind": "Config", "s": strings.Repeat("a", size), "value": "x"}
	}
	low, high := 0, int(celconfig.MaxRequestSizeBytes)-1024
	for low < high {
		mid := (low + high) / 2
		cost, err := evaluateBudget(t, newBudget(), compiled, parameters(mid))
		if err != nil {
			t.Fatalf("measure runtime budget fixture: %v", err)
		}
		if cost < target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	params := parameters(low)
	cost, err := evaluateBudget(t, newBudget(), compiled, params)
	if err != nil || cost != target {
		t.Fatalf("cannot construct exact runtime-cost fixture: cost %d, error %v; want %d", cost, err, target)
	}
	return fmt.Sprintf(`{"apiVersion":"example.com/v1","kind":"Config","s":"%s","value":"x"}`, params["s"]), params
}

func evaluateBudget(t *testing.T, b *budget, compiled *compiledExpression, parameters map[string]any) (uint64, error) {
	t.Helper()
	program, err := b.program(compiled)
	if err != nil {
		return 0, err
	}
	value, details, err := program.ContextEval(testContext(t), map[string]any{"parameters": parameters})
	if details == nil || details.ActualCost() == nil {
		t.Fatal("cost-tracked program returned no actual cost")
	}
	actual := *details.ActualCost()
	if err == nil {
		if _, ok := value.(types.String); !ok {
			t.Fatalf("budget fixture returned %T, want CEL string", value)
		}
		b.charge(actual)
	}
	return actual, err
}

func TestProgramCache_ReuseAcrossPoolsAndGenerations(t *testing.T) {
	ctx := testContext(t)
	extractor := newTestExtractor()
	entry := testExtractor("subnet", `parameters.subnetID`)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{entry})
	var first *compiledExpression
	for _, key := range []PoolKey{
		pool.Key,
		pool.Key,
		{Driver: testDriver, Name: "pool-a", Generation: 2},
		{Driver: testDriver, Name: "pool-b", Generation: 2},
		{Driver: "other.example.com", Name: "pool-b", Generation: 2},
	} {
		pool.Key = key
		cfg := opaqueConfig(nil, `{"subnetID":"blue"}`)
		cfg.Opaque.Driver = key.Driver
		got, err := extractor.Extract(ctx, pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{cfg})
		if err != nil {
			t.Fatalf("Extract(%#v) error = %v", key, err)
		}
		assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
		cached := cachedProgram(t, extractor, entry.Expression, environment.StoredExpressions)
		if first == nil {
			first = cached
		} else if cached != first {
			t.Errorf("pool/generation change recompiled expression: %p != %p", cached, first)
		}
		assertCacheLength(t, extractor, 1)
	}
}

func TestProgramCache_ReuseAcrossMatchers(t *testing.T) {
	ctx := testContext(t)
	extractor := newTestExtractor()
	entry := testExtractor("subnet", `parameters.value`)
	pool := testPool(1, nil)
	configs := []resourceapi.DeviceClaimConfiguration{
		opaqueConfig(nil, `{"apiVersion":"a.example.com/v1","kind":"First","value":"blue"}`),
		opaqueConfig(nil, `{"apiVersion":"b.example.com/v2","kind":"First","value":"green"}`),
		opaqueConfig(nil, `{"apiVersion":"b.example.com/v2","kind":"Second","value":"red"}`),
	}
	var first *compiledExpression
	for _, tc := range []struct{ name, group, kind, value string }{
		{"subnet", "a.example.com", "First", "blue"},
		{"subnet", "b.example.com", "First", "green"},
		{"subnet", "b.example.com", "Second", "red"},
		{"renamed", "b.example.com", "Second", "red"},
		{"renamed", "c.example.com", "Second", ""},
		{"subnet", "a.example.com", "First", "blue"},
	} {
		entry.Name, entry.APIGroup, entry.Kind = tc.name, tc.group, tc.kind
		pool.Extractors = []resourceapi.SharingAffinityExtractor{entry}
		got, err := extractor.Extract(ctx, pool, Request{Name: "req"}, configs)
		if err != nil {
			t.Fatalf("Extract(%s/%s) error = %v", tc.group, tc.kind, err)
		}
		want := Result{Constrained: true, Unsatisfied: []string{tc.name}}
		if tc.value != "" {
			want = Result{Constrained: true, Satisfied: true, Values: map[string]string{tc.name: tc.value}}
		}
		assertExtractResult(t, got, want)
		cached := cachedProgram(t, extractor, entry.Expression, environment.StoredExpressions)
		if first == nil {
			first = cached
		} else if first != cached {
			t.Error("changing metadata or name recompiled an identical expression")
		}
		assertCacheLength(t, extractor, 1)
	}
	// Two descriptors can share one program within a single extraction too.
	a, b := entry, entry
	a.Name, a.APIGroup, a.Kind = "first", "a.example.com", "First"
	b.Name, b.APIGroup, b.Kind = "second", "b.example.com", "Second"
	pool.Extractors = []resourceapi.SharingAffinityExtractor{a, b}
	got, err := extractor.Extract(ctx, pool, Request{Name: "req"}, configs)
	if err != nil {
		t.Fatalf("Extract(shared expression, independent matchers) error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"first": "blue", "second": "red"}})
	assertCacheLength(t, extractor, 1)
}

func TestProgramCache_ChangedExpression(t *testing.T) {
	extractor := newTestExtractor()
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `parameters.first`)})
	configs := []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"first":"blue","second":"green"}`)}
	var first *compiledExpression
	for _, tc := range []struct{ expression, value string }{
		{`parameters.first`, "blue"},
		{`parameters.second`, "green"},
		{`parameters.first`, "blue"},
	} {
		pool.Extractors[0].Expression = tc.expression
		got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, configs)
		if err != nil {
			t.Fatalf("Extract() error = %v", err)
		}
		assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": tc.value}})
		cached := cachedProgram(t, extractor, tc.expression, environment.StoredExpressions)
		if first == nil {
			first = cached
		} else if tc.expression == `parameters.second` && cached == first {
			t.Error("changed expression reused stale program")
		} else if tc.expression == `parameters.first` && cached != first {
			t.Error("original expression was recompiled")
		}
	}
	assertCacheLength(t, extractor, 2)
}

func TestProgramCache_Eviction(t *testing.T) {
	for _, capacity := range []int{0, -1, 2} {
		t.Run(fmt.Sprintf("capacity-%d", capacity), func(t *testing.T) {
			extractor := NewExtractor(capacity)
			effective := max(1, capacity)
			var first *compiledExpression
			for i := range effective + 2 {
				expression := fmt.Sprintf(`"value-%d"`, i)
				pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", expression)})
				got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
				if err != nil {
					t.Fatalf("Extract() error = %v", err)
				}
				assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": fmt.Sprintf("value-%d", i)}})
				if i == 0 {
					first = cachedProgram(t, extractor, expression, environment.StoredExpressions)
				}
				assertCacheLength(t, extractor, min(i+1, effective))
			}
			extractor.cacheMutex.RLock()
			_, found := extractor.cache.Get(cacheKey{env: environment.StoredExpressions, expr: `"value-0"`})
			extractor.cacheMutex.RUnlock()
			if found {
				t.Fatal("oldest expression survived bounded eviction")
			}
			pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", `"value-0"`)})
			got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{}`)})
			if err != nil {
				t.Fatalf("Extract(evicted expression) error = %v", err)
			}
			assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "value-0"}})
			if cachedProgram(t, extractor, `"value-0"`, environment.StoredExpressions) == first {
				t.Error("evicted expression unexpectedly retained its original program")
			}
			assertCacheLength(t, extractor, effective)
		})
	}
}

func TestProgramCache_EnvironmentSeparation(t *testing.T) {
	extractor := newTestExtractor()
	expression := `parameters.?subnetID.orValue("")`
	newProgram, err := extractor.getOrCompile(expression, environment.NewExpressions)
	if err != nil {
		t.Fatalf("new-expression compile error = %v", err)
	}
	storedProgram, err := extractor.getOrCompile(expression, environment.StoredExpressions)
	if err != nil {
		t.Fatalf("stored-expression compile error = %v", err)
	}
	if newProgram == storedProgram {
		t.Error("new and stored environment modes shared a cache entry")
	}
	for mode, want := range map[environment.Type]*compiledExpression{
		environment.NewExpressions: newProgram, environment.StoredExpressions: storedProgram,
	} {
		got, err := extractor.getOrCompile(expression, mode)
		if err != nil || got != want {
			t.Errorf("getOrCompile(mode %v) = %p, %v; want cached %p", mode, got, err, want)
		}
		if cachedProgram(t, extractor, expression, mode) != want {
			t.Errorf("mode %v cache entry differs from compiled program", mode)
		}
	}
	assertCacheLength(t, extractor, 2)
	pool := testPool(1, []resourceapi.SharingAffinityExtractor{testExtractor("subnet", expression)})
	got, err := extractor.Extract(testContext(t), pool, Request{Name: "req"}, []resourceapi.DeviceClaimConfiguration{opaqueConfig(nil, `{"subnetID":"blue"}`)})
	if err != nil {
		t.Fatalf("Extract() after warming both environment modes error = %v", err)
	}
	assertExtractResult(t, got, Result{Constrained: true, Satisfied: true, Values: map[string]string{"subnet": "blue"}})
	if cachedProgram(t, extractor, expression, environment.StoredExpressions) != storedProgram {
		t.Error("runtime extraction replaced the stored-environment program")
	}
	assertCacheLength(t, extractor, 2)
}

func TestProgramCache_CompileFailureNotCached(t *testing.T) {
	extractor := newTestExtractor()
	for range 2 {
		for _, mode := range []environment.Type{environment.NewExpressions, environment.StoredExpressions} {
			compiled, err := extractor.getOrCompile(`parameters.[`, mode)
			if err == nil || compiled != nil {
				t.Fatalf("invalid expression compilation = %p, %v; want nil, error", compiled, err)
			}
			assertCacheLength(t, extractor, 0)
		}
	}
}

func TestProgramCache_Concurrent(t *testing.T) {
	ctx := testContext(t)
	extractor := newTestExtractor()
	const workers = 24
	start := make(chan struct{})
	programs := make(chan *compiledExpression, workers)
	failures := make(chan error, workers)
	var wg sync.WaitGroup
	for i := range workers {
		wg.Go(func() {
			<-start
			expression := `parameters.value`
			name := fmt.Sprintf("key-%d", i)
			entry := testExtractor(name, expression)
			entry.APIGroup, entry.Kind = fmt.Sprintf("group-%d.example.com", i), "Config"
			pool := testPool(int64(i), []resourceapi.SharingAffinityExtractor{entry})
			pool.Key.Name = fmt.Sprintf("pool-%d", i)
			for j := range 5 {
				value := fmt.Sprintf("worker-%d-call-%d", i, j)
				raw := fmt.Sprintf(`{"apiVersion":%q,"kind":"Config","value":%q}`, entry.APIGroup+"/v1", value)
				got, err := extractor.Extract(ctx, pool, Request{Name: fmt.Sprintf("req-%d", j)}, []resourceapi.DeviceClaimConfiguration{
					opaqueConfig(nil, `{"apiVersion":"unrelated.example.com/v1","kind":"Other"}`),
					opaqueConfig(nil, raw),
				})
				if err != nil || !got.Constrained || !got.Satisfied || len(got.Values) != 1 || got.Values[name] != value || len(got.Unsatisfied) != 0 {
					failures <- fmt.Errorf("worker %d call %d: result %#v, error %v", i, j, got, err)
					return
				}
			}
			program, err := extractor.getOrCompile(expression, environment.StoredExpressions)
			if err != nil {
				failures <- err
				return
			}
			programs <- program
		})
	}
	close(start)
	wg.Wait()
	close(failures)
	close(programs)
	for err := range failures {
		t.Error(err)
	}
	var first *compiledExpression
	count := 0
	for program := range programs {
		count++
		if program == nil {
			t.Error("concurrent compile returned nil program")
		} else if first == nil {
			first = program
		} else if first != program {
			t.Error("concurrent compilation returned different cached programs")
		}
	}
	if count != workers {
		t.Errorf("successful workers = %d, want %d", count, workers)
	}
	assertCacheLength(t, extractor, 1)
}

func cachedProgram(t *testing.T, extractor *Extractor, expression string, mode environment.Type) *compiledExpression {
	t.Helper()
	extractor.cacheMutex.RLock()
	defer extractor.cacheMutex.RUnlock()
	value, ok := extractor.cache.Get(cacheKey{env: mode, expr: expression})
	if !ok {
		t.Fatalf("expression %q not cached for environment %v", expression, mode)
	}
	compiled, ok := value.(*compiledExpression)
	if !ok || compiled == nil {
		t.Fatalf("cache value = %#v, want compiled expression", value)
	}
	return compiled
}

func assertCacheLength(t *testing.T, extractor *Extractor, want int) {
	t.Helper()
	extractor.cacheMutex.RLock()
	defer extractor.cacheMutex.RUnlock()
	if got := extractor.cache.Len(); got != want {
		t.Errorf("program cache length = %d, want %d", got, want)
	}
}
