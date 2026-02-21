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

package metrics

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
)

var podsGVR = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}

func TestRealFIFO_Metrics(t *testing.T) {
	tests := []struct {
		name                        string
		actions                     []func(f *cache.RealFIFO)
		expectedQueuedItems         int
		expectedLatencyObservations uint64
	}{
		{
			name:                "empty queue has zero metric",
			actions:             []func(f *cache.RealFIFO){},
			expectedQueuedItems: 0,
		},
		{
			name: "Add increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
			},
			expectedQueuedItems: 1,
		},
		{
			name: "multiple Adds increase metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("bar", 2)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("baz", 3)) },
			},
			expectedQueuedItems: 3,
		},
		{
			name: "Update increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Update(mkFifoObj("foo", 2)) },
			},
			expectedQueuedItems: 2,
		},
		{
			name: "Delete increases metric",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Delete(mkFifoObj("foo", 2)) },
			},
			expectedQueuedItems: 2,
		},
		{
			name: "Pop decreases metric and records latency",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("bar", 2)) },
				func(f *cache.RealFIFO) {
					_, _ = f.Pop(func(obj interface{}, isInInitialList bool) error { return nil })
				},
			},
			expectedQueuedItems:         1,
			expectedLatencyObservations: 1,
		},
		{
			name: "PopBatch decreases metric and records latency",
			actions: []func(f *cache.RealFIFO){
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("foo", 1)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("bar", 2)) },
				func(f *cache.RealFIFO) { _ = f.Add(mkFifoObj("baz", 3)) },
				func(f *cache.RealFIFO) {
					_ = f.PopBatch(
						func(deltas []cache.Delta, isInInitialList bool) error { return nil },
						func(obj interface{}, isInInitialList bool) error { return nil },
					)
				},
			},
			expectedQueuedItems:         0,
			expectedLatencyObservations: 1,
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
			expectedQueuedItems: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metricsProvider := newTestFIFOMetricsProvider()
			informerName, err := cache.NewInformerName("test-fifo")
			if err != nil {
				t.Fatalf("NewInformerName() unexpected error: %v", err)
			}
			defer informerName.Release()
			id := informerName.WithResource(podsGVR)

			f := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
				KeyFunction:     testFifoObjectKeyFunc,
				KnownObjects:    emptyKnownObjects(),
				Identifier:      id,
				MetricsProvider: metricsProvider,
			})

			for _, action := range tt.actions {
				action(f)
			}

			// Verify queued items metric
			verifyQueuedItems(t, metricsProvider, "test-fifo", podsGVR, tt.expectedQueuedItems)

			// Verify processing latency observations
			verifyLatencyObservations(t, metricsProvider, "test-fifo", podsGVR, tt.expectedLatencyObservations)
		})
	}
}

func TestRealFIFO_MetricsNotPublishedForUnnamedFIFO(t *testing.T) {
	metricsProvider := newTestFIFOMetricsProvider()

	// No InformerName configured - should not publish metrics
	var id cache.InformerNameAndResource
	f := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id,
		MetricsProvider: metricsProvider,
	})

	// Perform operations
	_ = f.Add(mkFifoObj("foo", 1))
	_ = f.Add(mkFifoObj("bar", 2))

	// No metrics should be created because there's no identifier configured
	want := ""
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "informer_queued_items"); err != nil {
		t.Fatal(err)
	}
}

func TestRealFIFO_MetricsNotPublishedForDuplicateGVR(t *testing.T) {
	metricsProvider := newTestFIFOMetricsProvider()

	// Create InformerName
	informerName, err := cache.NewInformerName("duplicate-test")
	if err != nil {
		t.Fatalf("NewInformerName() unexpected error: %v", err)
	}
	defer informerName.Release()
	// Create first FIFO with a GVR - this should be reserved
	id1 := informerName.WithResource(podsGVR)
	if !id1.Reserved() {
		t.Fatal("Expected first identifier to be reserved")
	}
	f1 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id1,
		MetricsProvider: metricsProvider,
	})

	// Create second FIFO with the same GVR - this should NOT be reserved
	id2 := informerName.WithResource(podsGVR)
	if id2.Reserved() {
		t.Fatal("Expected second identifier with same GVR to not be reserved")
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
	verifyQueuedItems(t, metricsProvider, "duplicate-test", podsGVR, 1)
}

func TestRealFIFO_MetricsTrackedIndependentlyForDifferentFIFOs(t *testing.T) {
	metricsProvider := newTestFIFOMetricsProvider()

	// Create two InformerNames with different names - both should be unique
	informerName1, err := cache.NewInformerName("fifo-1")
	if err != nil {
		t.Fatalf("NewInformerName() unexpected error: %v", err)
	}
	defer informerName1.Release()

	id1 := informerName1.WithResource(podsGVR)
	f1 := cache.NewRealFIFOWithOptions(cache.RealFIFOOptions{
		KeyFunction:     testFifoObjectKeyFunc,
		KnownObjects:    emptyKnownObjects(),
		Identifier:      id1,
		MetricsProvider: metricsProvider,
	})

	informerName2, err := cache.NewInformerName("fifo-2")
	if err != nil {
		t.Fatalf("NewInformerName() unexpected error: %v", err)
	}
	defer informerName2.Release()

	id2 := informerName2.WithResource(podsGVR)
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
	wantInformerQueuedItemsMetric := `# HELP informer_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE informer_queued_items gauge
informer_queued_items{group="",name="fifo-1",resource="pods",version="v1"} 2
informer_queued_items{group="",name="fifo-2",resource="pods",version="v1"} 1
`
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(wantInformerQueuedItemsMetric), "informer_queued_items"); err != nil {
		t.Fatal(err)
	}

	// Pop from f1 and verify its metric decreases while f2's stays the same
	_, _ = f1.Pop(func(obj interface{}, isInInitialList bool) error { return nil })

	wantInformerQueuedItemsMetricAfterPop := `# HELP informer_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE informer_queued_items gauge
informer_queued_items{group="",name="fifo-1",resource="pods",version="v1"} 1
informer_queued_items{group="",name="fifo-2",resource="pods",version="v1"} 1
`
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(wantInformerQueuedItemsMetricAfterPop), "informer_queued_items"); err != nil {
		t.Fatal(err)
	}

	// Verify that processing latency was recorded for f1 only after the Pop.
	// fifo-2 has count=0 because no Pop was called for it.
	wantLatencyMetricAfterPop := `# HELP informer_processing_latency_seconds [ALPHA] Time taken to process events after popping from the queue.
# TYPE informer_processing_latency_seconds histogram
informer_processing_latency_seconds_count{group="",name="fifo-1",resource="pods",version="v1"} 1
informer_processing_latency_seconds_count{group="",name="fifo-2",resource="pods",version="v1"} 0
`
	if err := testutil.GatherAndCompare(metricsProvider.gatherWithoutDurations(), strings.NewReader(wantLatencyMetricAfterPop), "informer_processing_latency_seconds"); err != nil {
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
	registry  metrics.KubeRegistry
	gauge     *metrics.GaugeVec
	histogram *metrics.HistogramVec
}

func newTestFIFOMetricsProvider() *testFIFOMetricsProvider {
	registry := metrics.NewKubeRegistry()
	gauge := metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Subsystem:      "informer",
			Name:           "queued_items",
			Help:           "Number of items currently queued in the FIFO.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "group", "version", "resource"},
	)
	histogram := metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Subsystem:      "informer",
			Name:           "processing_latency_seconds",
			Help:           "Time taken to process events after popping from the queue.",
			Buckets:        []float64{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10},
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"name", "group", "version", "resource"},
	)
	registry.MustRegister(gauge)
	registry.MustRegister(histogram)
	return &testFIFOMetricsProvider{
		registry:  registry,
		gauge:     gauge,
		histogram: histogram,
	}
}

func (p *testFIFOMetricsProvider) NewQueuedItemMetric(id cache.InformerNameAndResource) cache.GaugeMetric {
	return p.gauge.WithLabelValues(id.Name(), id.GroupVersionResource().Group, id.GroupVersionResource().Version, id.GroupVersionResource().Resource)
}

func (p *testFIFOMetricsProvider) NewProcessingLatencyMetric(id cache.InformerNameAndResource) cache.HistogramMetric {
	return p.histogram.WithLabelValues(id.Name(), id.GroupVersionResource().Group, id.GroupVersionResource().Version, id.GroupVersionResource().Resource)
}

func verifyQueuedItems(t *testing.T, metricsProvider *testFIFOMetricsProvider, informerName string, gvr schema.GroupVersionResource, expected int) {
	t.Helper()
	want := fmt.Sprintf(`# HELP informer_queued_items [ALPHA] Number of items currently queued in the FIFO.
# TYPE informer_queued_items gauge
informer_queued_items{group="%s",name="%s",resource="%s",version="%s"} %d
`, gvr.Group, informerName, gvr.Resource, gvr.Version, expected)
	if err := testutil.GatherAndCompare(metricsProvider.registry, strings.NewReader(want), "informer_queued_items"); err != nil {
		t.Fatal(err)
	}
}

// verifyLatencyObservations checks the histogram observation count using a custom gatherer
// that strips timing-dependent values (bucket counts and sum) from the comparison, so that
// we only verify the number of observations without being affected by the duration values.
func verifyLatencyObservations(t *testing.T, metricsProvider *testFIFOMetricsProvider, informerName string, gvr schema.GroupVersionResource, expected uint64) {
	t.Helper()
	if expected == 0 {
		return
	}

	want := fmt.Sprintf(`# HELP informer_processing_latency_seconds [ALPHA] Time taken to process events after popping from the queue.
# TYPE informer_processing_latency_seconds histogram
informer_processing_latency_seconds_count{group="%s",name="%s",resource="%s",version="%s"} %d
`, gvr.Group, informerName, gvr.Resource, gvr.Version, expected)
	if err := testutil.GatherAndCompare(metricsProvider.gatherWithoutDurations(), strings.NewReader(want), "informer_processing_latency_seconds"); err != nil {
		t.Fatal(err)
	}
}

func (p *testFIFOMetricsProvider) gatherWithoutDurations() testutil.GathererFunc {
	return func() ([]*testutil.MetricFamily, error) {
		got, err := p.registry.Gather()
		for _, mf := range got {
			for _, m := range mf.Metric {
				if m.Histogram == nil {
					continue
				}
				// Remove everything from a histogram that depends on timing.
				m.Histogram.SampleSum = nil
				m.Histogram.Bucket = nil
			}
		}
		return got, err
	}
}
