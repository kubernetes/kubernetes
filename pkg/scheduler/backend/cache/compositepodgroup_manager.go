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

package cache

import (
	"fmt"

	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// compositePodGroupState holds the runtime state of a composite pod group in the tree.
type compositePodGroupState struct {
	name          string
	namespace     string
	minGroupCount int
	childrenCPGs  sets.Set[podGroupKey]
	childrenPGs   sets.Set[podGroupKey]
	parentCPG     *podGroupKey
	generation    int64
}

// snapshot returns a deep copy of the live composite pod group state as an immutable snapshot.
func (n *compositePodGroupState) snapshot() *compositePodGroupStateSnapshot {
	return &compositePodGroupStateSnapshot{
		name:          n.name,
		namespace:     n.namespace,
		minGroupCount: n.minGroupCount,
		childrenCPGs:  n.childrenCPGs.Clone(),
		childrenPGs:   n.childrenPGs.Clone(),
		parentCPG:     n.parentCPG,
		generation:    n.generation,
	}
}

// compositePodGroupStateSnapshot is an immutable copy of compositePodGroupState for export.
type compositePodGroupStateSnapshot struct {
	name          string
	namespace     string
	minGroupCount int
	childrenCPGs  sets.Set[podGroupKey]
	childrenPGs   sets.Set[podGroupKey]
	parentCPG     *podGroupKey
	generation    int64
}

func (n compositePodGroupStateSnapshot) GetName() string       { return n.name }
func (n compositePodGroupStateSnapshot) GetNamespace() string  { return n.namespace }
func (n compositePodGroupStateSnapshot) GetMinGroupCount() int { return n.minGroupCount }

func (n compositePodGroupStateSnapshot) GetChildrenCPGs() []string {
	var children []string
	for k := range n.childrenCPGs {
		children = append(children, k.name)
	}
	return children
}

func (n compositePodGroupStateSnapshot) GetChildrenPGs() []string {
	var children []string
	for k := range n.childrenPGs {
		children = append(children, k.name)
	}
	return children
}

// GetCompositePodGroupState returns a snapshot of the composite pod group node.
func (cache *cacheImpl) GetCompositePodGroupState(namespace, name string) (fwk.CompositePodGroupState, error) {
	cache.mu.RLock()
	defer cache.mu.RUnlock()

	key := newPodGroupKey(namespace, name)
	cpg, exists := cache.cpgStates[key]
	if !exists {
		return nil, fmt.Errorf("composite pod group %s not found", key)
	}

	return compositePodGroupStateSnapshot{
		name:          cpg.name,
		namespace:     cpg.namespace,
		minGroupCount: cpg.minGroupCount,
		childrenCPGs:  cpg.childrenCPGs.Clone(),
		childrenPGs:   cpg.childrenPGs.Clone(),
	}, nil
}



// AddCompositePodGroup adds a composite pod group to the cache.
func (cache *cacheImpl) AddCompositePodGroup(cpg *schedulingapi.CompositePodGroup) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := newPodGroupKey(cpg.Namespace, cpg.Name)
	cpgState, exists := cache.cpgStates[key]
	if !exists {
		cpgState = &compositePodGroupState{
			name:         cpg.Name,
			namespace:    cpg.Namespace,
			childrenCPGs: sets.New[podGroupKey](),
			childrenPGs:  sets.New[podGroupKey](),
		}
		cache.cpgStates[key] = cpgState
	}
	if cpg.Spec.SchedulingPolicy.Gang != nil {
		cpgState.minGroupCount = int(cpg.Spec.SchedulingPolicy.Gang.MinGroupCount)
	}

	// Handle parent link
	if cpg.Spec.ParentCompositePodGroupName != nil {
		parentKey := newPodGroupKey(cpg.Namespace, *cpg.Spec.ParentCompositePodGroupName)
		cpgState.parentCPG = &parentKey

		parent, parentExists := cache.cpgStates[parentKey]
		if parentExists {
			parent.childrenCPGs.Insert(key)
		} else {
			// Orphan
			if _, ok := cache.orphanCPGs[parentKey]; !ok {
				cache.orphanCPGs[parentKey] = sets.New[podGroupKey]()
			}
			cache.orphanCPGs[parentKey].Insert(key)
		}
	}

	// Process orphans
	if orphans, ok := cache.orphanCPGs[key]; ok {
		for orphanKey := range orphans {
			if _, ok := cache.cpgStates[orphanKey]; ok {
				cpgState.childrenCPGs.Insert(orphanKey)
			}
		}
		delete(cache.orphanCPGs, key)
	}
	if orphans, ok := cache.orphanPGs[key]; ok {
		for orphanKey := range orphans {
			if _, ok := cache.podGroupStates[orphanKey]; ok {
				cpgState.childrenPGs.Insert(orphanKey)
			}
		}
		delete(cache.orphanPGs, key)
	}

	cpgState.generation = nextPodGroupGeneration()
}

// RemoveCompositePodGroup removes a composite pod group from the cache.
func (cache *cacheImpl) RemoveCompositePodGroup(cpg *schedulingapi.CompositePodGroup) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := newPodGroupKey(cpg.Namespace, cpg.Name)
	cpgState, exists := cache.cpgStates[key]
	if !exists {
		return
	}

	// Remove from parent's children list
	if cpgState.parentCPG != nil {
		if parent, ok := cache.cpgStates[*cpgState.parentCPG]; ok {
			parent.childrenCPGs.Delete(key)
		}
	}

	// Move children CPGs to orphans
	for childKey := range cpgState.childrenCPGs {
		if _, ok := cache.orphanCPGs[key]; !ok {
			cache.orphanCPGs[key] = sets.New[podGroupKey]()
		}
		cache.orphanCPGs[key].Insert(childKey)
	}

	// Move children PGs to orphans
	for childKey := range cpgState.childrenPGs {
		if _, ok := cache.orphanPGs[key]; !ok {
			cache.orphanPGs[key] = sets.New[podGroupKey]()
		}
		cache.orphanPGs[key].Insert(childKey)
	}

	delete(cache.cpgStates, key)
}

// AddPodGroup adds a pod group to the cache.
func (cache *cacheImpl) AddPodGroup(pg *schedulingapi.PodGroup) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := newPodGroupKey(pg.Namespace, pg.Name)

	podGroupState, exists := cache.podGroupStates[key]
	if !exists {
		podGroupState = newPodGroupState()
		cache.podGroupStates[key] = podGroupState
	}

	if pg.Spec.SchedulingPolicy.Gang != nil {
		podGroupState.minCount = int(pg.Spec.SchedulingPolicy.Gang.MinCount)
	}

	// Check if it has a parent CPG
	if pg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := newPodGroupKey(pg.Namespace, *pg.Spec.ParentCompositePodGroupName)
	podGroupState.parentCPG = &parentKey

	parent, parentExists := cache.cpgStates[parentKey]
	if parentExists {
		parent.childrenPGs.Insert(key)
	} else {
		// Orphan
		if _, ok := cache.orphanPGs[parentKey]; !ok {
			cache.orphanPGs[parentKey] = sets.New[podGroupKey]()
		}
		cache.orphanPGs[parentKey].Insert(key)
	}
}

// RemovePodGroup removes a pod group from the cache.
func (cache *cacheImpl) RemovePodGroup(pg *schedulingapi.PodGroup) {
	cache.mu.Lock()
	defer cache.mu.Unlock()

	key := newPodGroupKey(pg.Namespace, pg.Name)

	if podGroupState, exists := cache.podGroupStates[key]; exists {
		podGroupState.minCount = 0
		podGroupState.parentCPG = nil
	}

	// Remove from parent if exists
	if pg.Spec.ParentCompositePodGroupName == nil {
		return
	}

	parentKey := newPodGroupKey(pg.Namespace, *pg.Spec.ParentCompositePodGroupName)
	if parent, ok := cache.cpgStates[parentKey]; ok {
		parent.childrenPGs.Delete(key)
	}
}

// UpdateCompositePodGroup updates a composite pod group in the cache.
func (cache *cacheImpl) UpdateCompositePodGroup(oldCPG, newCPG *schedulingapi.CompositePodGroup) {
	cache.mu.Lock()
	key := newPodGroupKey(newCPG.Namespace, newCPG.Name)
	cpgState, exists := cache.cpgStates[key]
	if !exists {
		cache.mu.Unlock()
		cache.AddCompositePodGroup(newCPG)
		return
	}

	// Update MinGroupCount
	if newCPG.Spec.SchedulingPolicy.Gang != nil {
		cpgState.minGroupCount = int(newCPG.Spec.SchedulingPolicy.Gang.MinGroupCount)
	} else {
		cpgState.minGroupCount = 1
	}
	cpgState.generation = nextPodGroupGeneration()
	cache.mu.Unlock()
}

// UpdatePodGroup updates a pod group in the cache.
func (cache *cacheImpl) UpdatePodGroup(oldPG, newPG *schedulingapi.PodGroup) {
	cache.mu.Lock()
	key := newPodGroupKey(newPG.Namespace, newPG.Name)
	podGroupState, exists := cache.podGroupStates[key]
	if !exists {
		cache.mu.Unlock()
		cache.AddPodGroup(newPG)
		return
	}

	// Update MinCount
	if newPG.Spec.SchedulingPolicy.Gang != nil {
		podGroupState.minCount = int(newPG.Spec.SchedulingPolicy.Gang.MinCount)
	} else {
		podGroupState.minCount = 1
	}
	cache.mu.Unlock()
}
