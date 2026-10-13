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

package podautoscaler

import (
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/controller/util/selectors"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test verifies helper function PutIfAbsent used during enqueue is not overwriting values
// that are updated during reconciliation phase
func TestHPASelectorStorePutIfAbsent(t *testing.T) {
	store := newHPASelectorStore()
	key := selectors.Key{Name: "hpa-1", Namespace: "ns-1"}
	sel1, _ := labels.Parse("app=hakuna")
	sel2, _ := labels.Parse("app=matata")

	// This should succeed.
	assert.True(t, store.PutIfAbsent("ns-1", key, sel1))

	// This should be a no-op, should not overwrite.
	assert.False(t, store.PutIfAbsent("ns-1", key, sel2))

	// Verify original selector is still in place.
	pods := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "ns-1", Labels: map[string]string{"app": "hakuna"}}},
	}
	assert.Len(t, store.HPAsMatchingPods("ns-1", pods, 0), 1)

	pods2 := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "ns-1", Labels: map[string]string{"app": "matata"}}},
	}
	assert.Empty(t, store.HPAsMatchingPods("ns-1", pods2, 0))
}

// This verifies helper function PutIfPresent used during reconciliation phase
// is not accidentally adding HPA that was deleted from store
func TestHPASelectorStorePutIfPresent(t *testing.T) {
	store := newHPASelectorStore()
	key := selectors.Key{Name: "hpa-1", Namespace: "ns-1"}
	sel1, _ := labels.Parse("app=hakuna")
	sel2, _ := labels.Parse("app=matata")

	// Update on non-existent key is a no-op.
	assert.False(t, store.PutIfPresent("ns-1", key, sel2))

	// Insert then update.
	store.PutIfAbsent("ns-1", key, sel1)
	assert.True(t, store.PutIfPresent("ns-1", key, sel2))

	// Verify selector was updated.
	pods := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "ns-1", Labels: map[string]string{"app": "hakuna"}}},
	}
	assert.Empty(t, store.HPAsMatchingPods("ns-1", pods, 0))

	pods2 := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "ns-1", Labels: map[string]string{"app": "matata"}}},
	}
	assert.Len(t, store.HPAsMatchingPods("ns-1", pods2, 0), 1)
}

// This test ensures we are not leaking empty namespace entries when deleted
func TestHPASelectorStoreDeleteCleansUpNamespace(t *testing.T) {
	store := newHPASelectorStore()
	sel, _ := labels.Parse("app=hakuna")

	key1 := selectors.Key{Name: "hpa-1", Namespace: "ns-1"}
	key2 := selectors.Key{Name: "hpa-2", Namespace: "ns-1"}
	store.PutIfAbsent("ns-1", key1, sel)
	store.PutIfAbsent("ns-1", key2, sel)

	// Delete first HPA, namespace still exists.
	store.Delete("ns-1", key1)
	store.mu.RLock()
	require.Contains(t, store.namespaces, "ns-1")
	store.mu.RUnlock()

	// Delete the last HPA, namespace entry is removed.
	store.Delete("ns-1", key2)
	store.mu.RLock()
	require.NotContains(t, store.namespaces, "ns-1")
	store.mu.RUnlock()
}

// This verifies that it is not iterating the entire selectorStore when matching work is done
func TestHPASelectorStoreHPAsMatchingPodsLimit(t *testing.T) {
	store := newHPASelectorStore()

	// Both HPAs select "app=hakuna".
	sel, _ := labels.Parse("app=hakuna")
	store.PutIfAbsent("ns-1", selectors.Key{Name: "hpa-1", Namespace: "ns-1"}, sel)
	store.PutIfAbsent("ns-1", selectors.Key{Name: "hpa-2", Namespace: "ns-1"}, sel)

	pods := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "ns-1", Labels: map[string]string{"app": "hakuna"}}},
	}

	// Without limit, returns all matches.
	assert.Len(t, store.HPAsMatchingPods("ns-1", pods, 0), 2)

	// With limit=2, returns exactly 2 (early exit).
	assert.Len(t, store.HPAsMatchingPods("ns-1", pods, 2), 2)

	// With limit=1, returns only 1.
	assert.Len(t, store.HPAsMatchingPods("ns-1", pods, 1), 1)
}

// Test ensures we are not iterating through all the HPA's across all namespaces
func TestHPASelectorStoreHPAsMatchingPodsNamespaceIsolation(t *testing.T) {
	store := newHPASelectorStore()
	sel, _ := labels.Parse("app=hakuna")

	store.PutIfAbsent("ns-1", selectors.Key{Name: "hpa-1", Namespace: "ns-1"}, sel)
	store.PutIfAbsent("ns-2", selectors.Key{Name: "hpa-2", Namespace: "ns-2"}, sel)

	pods := []*v1.Pod{
		{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "ns-1", Labels: map[string]string{"app": "hakuna"}}},
	}

	// Only returns HPAs from the queried namespace.
	result := store.HPAsMatchingPods("ns-1", pods, 0)
	assert.Len(t, result, 1)
	assert.Equal(t, "hpa-1", result[0].Name)

	// Different namespace returns nothing for ns-1 pods.
	assert.Empty(t, store.HPAsMatchingPods("ns-3", pods, 0))
}

func BenchmarkHPAsMatchingPodsParallel(b *testing.B) {
	benchmarkMatchingPodsParallel(b, newHPASelectorStore())
}

// TODO: Remove once HPAOptimizedSelectorStore graduates to GA and the legacy
// BiMultimap tracker is removed.
func BenchmarkHPAsMatchingPodsParallelBiMultimap(b *testing.B) {
	benchmarkMatchingPodsParallel(b, newBiMultimapSelectorTracker())
}

func benchmarkMatchingPodsParallel(b *testing.B, tracker hpaSelectorTracker) {
	scenarios := []struct {
		name     string
		hpaCount int
		podCount int
	}{
		{name: "100hpas_10pods", hpaCount: 100, podCount: 10},
		{name: "1000hpas_10pods", hpaCount: 1000, podCount: 10},
		{name: "1000hpas_100pods", hpaCount: 1000, podCount: 100},
		{name: "10000hpas_100pods", hpaCount: 10000, podCount: 100},
	}

	const namespace = "hakuna"
	for _, s := range scenarios {
		b.Run(s.name, func(b *testing.B) {
			for i := 0; i < s.hpaCount; i++ {
				key := selectors.Key{Name: fmt.Sprintf("hpa-%d", i), Namespace: namespace}
				sel, _ := labels.Parse(fmt.Sprintf("app=matata-%d", i))
				tracker.PutIfAbsent(namespace, key, sel)
			}

			// Pods labels match to first HPA's selector
			pods := make([]*v1.Pod, 0, s.podCount)
			for i := 0; i < s.podCount; i++ {
				pods = append(pods, &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("pod-%d", i),
						Namespace: namespace,
						Labels:    map[string]string{"app": "matata-0"},
					},
				})
			}

			b.ResetTimer()
			b.ReportAllocs()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					tracker.HPAsMatchingPods(namespace, pods, 2)
				}
			})
		})
	}
}
