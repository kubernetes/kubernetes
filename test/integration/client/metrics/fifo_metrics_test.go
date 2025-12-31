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

package metrics

import (
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

func TestRealFIFO_Metrics(t *testing.T) {
	tests := []struct {
		name           string
		actions        []func(f *cache.RealFIFO)
		expectedMetric int
	}{
		{
			name:           "empty queue has zero metric",
			actions:        []func(f *cache.RealFIFO){},
			expectedMetric: 0,
		},
		{
			name: "Add increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
			},
			expectedMetric: 1,
		},
		{
			name: "multiple Adds increase metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("bar", 2)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("baz", 3)) },
			},
			expectedMetric: 3,
		},
		{
			name: "Update increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Update(mkFifoObj("foo", 2)) },
			},
			expectedMetric: 2,
		},
		{
			name: "Delete increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Delete(mkFifoObj("foo", 2)) },
			},
			expectedMetric: 2,
		},
		{
			name: "Pop decreases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("bar", 2)) },
				func(f *cache.RealFIFO) { cache.Pop(f) },
			},
			expectedMetric: 1,
		},
		{
			name: "Replace sets metric to new count",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("old", 1)) },
				func(f *cache.RealFIFO) {
					_ = f.Replace([]interface{}{
						mkFifoObj("foo", 1),
						mkFifoObj("bar", 2),
					}, "0")
				},
			},
			// 1 (Add) + 1 (Delete for "old") + 2 (Replace items) = 4
			expectedMetric: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cache.ResetIdentity()
			metricsProvider := newTestFIFOMetricsProvider()
			id, err := cache.NewIdentifier("test-fifo", &v1.Pod{})
			if err != nil {
				t.Fatalf("NewIdentifier() unexpected error: %v", err)
			}

			f := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
				KeyFunction:     testFifoObjectKeyFunc,
				KnownObjects:    emptyKnownObjects(),
				Identifier:      id,
				MetricsProvider: metricsProvider,
			})

			for _, action := range tt.actions {
				action(f)
			}

			want := fmt.Sprintf(`# HELP fifo_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE fifo_queued_items gauge
fifo_queued_items{item_type="*v1.Pod",name="test-fifo"} %d
`, tt.expectedMetric)
			if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "fifo_queued_items"); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestRealFIFO_MetricsNotPublishedForUnnamedFIFO(t *testing.T) {
	cache.ResetIdentity()
	metricsProvider := newTestFIFOMetricsProvider()

	// Identifier with empty name - should not be unique
	id, err := cache.NewIdentifier("", &v1.Pod{})
	if err == nil {
		t.Fatalf("NewIdentifier() expected error for empty name, got nil")
	}
	f := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id,
		MetricsProvider: metricsProvider,
	})

	// Perform operations
	_ = f.Add(mkFifoObj("foo", 1))
	_ = f.Add(mkFifoObj("bar", 2))

	// No metrics should be created because the identifier is not unique (empty name)
	want := ""
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "fifo_queued_items"); err != nil {
		t.Fatal(err)
	}
}

func TestRealFIFO_MetricsNotPublishedForDuplicateIdentifier(t *testing.T) {
	cache.ResetIdentity()
	metricsProvider := newTestFIFOMetricsProvider()

	// Create first FIFO with a name - this should be unique
	id1, err := cache.NewIdentifier("duplicate-name", &v1.Pod{})
	if err != nil {
		t.Fatalf("NewIdentifier() unexpected error: %v", err)
	}
	f1 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id1,
		MetricsProvider: metricsProvider,
	})

	// Create second FIFO with the same name - this should NOT be unique
	id2, err := cache.NewIdentifier("duplicate-name", &v1.Pod{})
	if err == nil {
		t.Fatalf("NewIdentifier() expected error for duplicate name, got nil")
	}
	f2 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id2,
		MetricsProvider: metricsProvider,
	})

	// Add items to both FIFOs
	_ = f1.Add(mkFifoObj("foo", 1))
	_ = f2.Add(mkFifoObj("bar", 2))

	// Only f1's metric should be published, f2 uses noopMetric
	want := `# HELP fifo_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE fifo_queued_items gauge
fifo_queued_items{item_type="*v1.Pod",name="duplicate-name"} 1
`
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "fifo_queued_items"); err != nil {
		t.Fatal(err)
	}
}

func TestRealFIFO_MetricsTrackedIndependentlyForDifferentFIFOs(t *testing.T) {
	cache.ResetIdentity()
	metricsProvider := newTestFIFOMetricsProvider()

	// Create two FIFOs with different names - both should be unique
	id1, err := cache.NewIdentifier("fifo-1", &v1.Pod{})
	if err != nil {
		t.Fatalf("NewIdentifier() unexpected error: %v", err)
	}
	f1 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id1,
		MetricsProvider: metricsProvider,
	})

	id2, err := cache.NewIdentifier("fifo-2", &v1.Pod{})
	if err != nil {
		t.Fatalf("NewIdentifier() unexpected error: %v", err)
	}
	f2 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id2,
		MetricsProvider: metricsProvider,
	})

	// Add items to f1
	_ = f1.Add(mkFifoObj("foo", 1))
	_ = f1.Add(mkFifoObj("bar", 2))

	// Add items to f2
	_ = f2.Add(mkFifoObj("baz", 3))

	// Verify metrics are tracked independently
	want := `# HELP fifo_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE fifo_queued_items gauge
fifo_queued_items{item_type="*v1.Pod",name="fifo-1"} 2
fifo_queued_items{item_type="*v1.Pod",name="fifo-2"} 1
`
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "fifo_queued_items"); err != nil {
		t.Fatal(err)
	}

	// Pop from f1 and verify its metric decreases while f2's stays the same
	cache.Pop(f1)

	wantAfterPop := `# HELP fifo_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE fifo_queued_items gauge
fifo_queued_items{item_type="*v1.Pod",name="fifo-1"} 1
fifo_queued_items{item_type="*v1.Pod",name="fifo-2"} 1
`
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(wantAfterPop), "fifo_queued_items"); err != nil {
		t.Fatal(err)
	}
}

type testFifoObject struct {
	name string
	val  interface{}
}

func testFifoObjectKeyFunc(obj interface{}) (string, error) {
	return obj.(testFifoObject).name, nil
}

func mkFifoObj(name string, val interface{}) testFifoObject {
	return testFifoObject{name: name, val: val}
}

type literalListerGetter func() []testFifoObject

func (l literalListerGetter) List() []interface{} {
	if l == nil {
		return nil
	}
	result := []interface{}{}
	for _, item := range l() {
		result = append(result, item)
	}
	return result
}

func (l literalListerGetter) ListKeys() []string {
	if l == nil {
		return nil
	}
	result := []string{}
	for _, item := range l() {
		result = append(result, item.name)
	}
	return result
}

func (l literalListerGetter) Get(key string) (interface{}, bool, error) {
	for _, item := range l() {
		if item.name == key {
			return item, true, nil
		}
	}
	return nil, false, nil
}

func (l literalListerGetter) GetByKey(key string) (interface{}, bool, error) {
	return l.Get(key)
}

func emptyKnownObjects() cache.KeyListerGetter {
	return literalListerGetter(
		func() []testFifoObject {
			return []testFifoObject{}
		},
	)
}

// testFIFOMetricsProvider is a test implementation of cache.FIFOMetricsProvider
// that uses real component-base metrics registered with a custom registry.
type testFIFOMetricsProvider struct {
	registry metrics.KubeRegistry
	gauge    *metrics.GaugeVec
}

func newTestFIFOMetricsProvider() *testFIFOMetricsProvider {
	registry := metrics.NewKubeRegistry()
	gauge := metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "fifo_queued_items",
			Help:           "Number of items currently queued in the FIFO.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "item_type"},
	)
	registry.MustRegister(gauge)
	return &testFIFOMetricsProvider{
		registry: registry,
		gauge:    gauge,
	}
}

func (p *testFIFOMetricsProvider) NewQueuedItemMetric(id *cache.Identifier) cache.GaugeMetric {
	return p.gauge.WithLabelValues(id.Name(), id.ItemType())
}
