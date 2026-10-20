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
	"fmt"
	"sync"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apiserver/pkg/cel/environment"
)

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
