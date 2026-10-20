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
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/interpreter"

	resourceapi "k8s.io/api/resource/v1"
	celconfig "k8s.io/apiserver/pkg/apis/cel"
	"k8s.io/apiserver/pkg/cel/environment"
)

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
			if err := ValidateExtractorSet(tc.extractors); err != nil {
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
	if err := ValidateExtractorSet([]resourceapi.SharingAffinityExtractor{entry}); err != nil {
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
