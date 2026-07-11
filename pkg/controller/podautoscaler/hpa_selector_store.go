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
	"maps"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/controller/util/selectors"
)

// hpaSelectorTracker has two implementations based on HPAOptimizedSelectorStore feature gate,
// hpaSelectorStore (optimized) when feature gate is true
// and biMultimapSelectorTracker when feature gate is false.
// TODO: Remove Bimultimap implementation once HPAOptimizedSelectorStore
// graduates to GA.
type hpaSelectorTracker interface {
	PutIfAbsent(namespace string, key selectors.Key, selector labels.Selector) bool

	PutIfPresent(namespace string, key selectors.Key, selector labels.Selector) bool

	Delete(namespace string, key selectors.Key)

	HPAsMatchingPods(namespace string, pods []*v1.Pod, limit int) []selectors.Key
}

var _ hpaSelectorTracker = &hpaSelectorStore{}

// hpaSelectorStore is a per-namespace store of HPA selectors that supports
// concurrent read-only overlap detection.
type hpaSelectorStore struct {
	mu         sync.RWMutex
	namespaces map[string]map[selectors.Key]labels.Selector
}

func newHPASelectorStore() *hpaSelectorStore {
	return &hpaSelectorStore{
		namespaces: make(map[string]map[selectors.Key]labels.Selector),
	}
}

// PutIfAbsent inserts an HPA's selector only if the key is not already present.
// Returns true if the selector was inserted (i.e., the key was new).
func (s *hpaSelectorStore) PutIfAbsent(namespace string, key selectors.Key, selector labels.Selector) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ns := s.namespaces[namespace]; ns != nil {
		if _, exists := ns[key]; exists {
			return false
		}
		ns[key] = selector
	} else {
		s.namespaces[namespace] = map[selectors.Key]labels.Selector{key: selector}
	}
	return true
}

// PutIfPresent updates an HPA's selector only if the key already exists.
// Returns true if the selector was updated.
func (s *hpaSelectorStore) PutIfPresent(namespace string, key selectors.Key, selector labels.Selector) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ns := s.namespaces[namespace]; ns != nil {
		if _, exists := ns[key]; exists {
			ns[key] = selector
			return true
		}
	}
	return false
}

// Delete removes an HPA's selector.
func (s *hpaSelectorStore) Delete(namespace string, key selectors.Key) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if ns := s.namespaces[namespace]; ns != nil {
		delete(ns, key)
		if len(ns) == 0 {
			delete(s.namespaces, namespace)
		}
	}
}

// HPAsMatchingPods returns the keys of all HPAs in the given namespace whose
// selectors match at least one of the provided pods. This is a read-only
// operation that holds only a shared read lock, allowing full concurrency
// across all workers. If limit > 0, it short-circuits after finding that many distinct matches.
func (s *hpaSelectorStore) HPAsMatchingPods(namespace string, pods []*v1.Pod, limit int) []selectors.Key {
	s.mu.RLock()
	defer s.mu.RUnlock()

	nsSelectors := s.namespaces[namespace]
	if len(nsSelectors) == 0 {
		return nil
	}

	// Deduplicate pod label sets to avoid redundant selector matching.
	// Pods from the same ReplicaSet typically share identical labels.
	uniqueLabels := make(map[string]labels.Set)
	for _, pod := range pods {
		key := labels.Set(pod.Labels).String()
		if _, seen := uniqueLabels[key]; !seen {
			uniqueLabels[key] = labels.Set(pod.Labels)
		}
	}

	hpas := make(map[selectors.Key]struct{})
	for _, set := range uniqueLabels {
		for key, sel := range nsSelectors {
			if sel.Matches(set) {
				hpas[key] = struct{}{}
				if limit > 0 && len(hpas) >= limit {
					return slices.Collect(maps.Keys(hpas))
				}
			}
		}
	}
	return slices.Collect(maps.Keys(hpas))
}

var _ hpaSelectorTracker = &biMultimapSelectorTracker{}

// biMultimapSelectorTracker is the legacy hpaSelectorTracker backed by a
// BiMultimap. It is used when HPAOptimizedSelectorStore is disabled.
// TODO: Remove once HPAOptimizedSelectorStore graduates to GA.
type biMultimapSelectorTracker struct {
	mu        sync.Mutex
	selectors *selectors.BiMultimap
}

func newBiMultimapSelectorTracker() *biMultimapSelectorTracker {
	return &biMultimapSelectorTracker{
		selectors: selectors.NewBiMultimap(),
	}
}

func (t *biMultimapSelectorTracker) PutIfAbsent(namespace string, key selectors.Key, selector labels.Selector) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.selectors.SelectorExists(key) {
		return false
	}
	t.selectors.PutSelector(key, selector)
	return true
}

func (t *biMultimapSelectorTracker) PutIfPresent(namespace string, key selectors.Key, selector labels.Selector) bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	if !t.selectors.SelectorExists(key) {
		return false
	}
	t.selectors.PutSelector(key, selector)
	return true
}

func (t *biMultimapSelectorTracker) Delete(namespace string, key selectors.Key) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.selectors.DeleteSelector(key)
}

func (t *biMultimapSelectorTracker) HPAsMatchingPods(namespace string, pods []*v1.Pod, limit int) []selectors.Key {
	t.mu.Lock()
	defer t.mu.Unlock()

	hpas := map[selectors.Key]struct{}{}
	for _, p := range pods {
		podKey := selectors.Key{Name: p.Name, Namespace: p.Namespace}
		t.selectors.Put(podKey, p.Labels)

		selectingHpas, ok := t.selectors.ReverseSelect(podKey)
		if !ok {
			continue
		}
		for _, hpa := range selectingHpas {
			hpas[hpa] = struct{}{}
		}
	}
	// Clean up all added pods.
	t.selectors.KeepOnly([]selectors.Key{})

	hpaList := []selectors.Key{}
	for hpa := range hpas {
		hpaList = append(hpaList, hpa)
	}
	return hpaList
}
