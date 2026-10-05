/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// Note: CycleState uses a sync.Map to back the storage, because it is thread safe. It's aimed to optimize for the "write once and read many times" scenarios.
// It is the recommended pattern used in all in-tree plugins - plugin-specific state is written once in PreFilter/PreScore and afterward read many times in Filter/Score.
type CycleState struct {
	// storage is keyed with StateKey, and valued with StateData.
	storage sync.Map
	// if recordPluginMetrics is true, metrics.PluginExecutionDuration will be recorded for this cycle.
	recordPluginMetrics bool
	// skipFilterPlugins are plugins that will be skipped in the Filter extension point.
	skipFilterPlugins sets.Set[string]
	// skipScorePlugins are plugins that will be skipped in the Score extension point.
	skipScorePlugins sets.Set[string]
	// skipPreBindPlugins are plugins that will be skipped in the PreBind extension point.
	skipPreBindPlugins sets.Set[string]
	// skipAllPostFilterPlugins indicates whether to skip all plugins in the PostFilter extension point.
	skipAllPostFilterPlugins bool
	// GetParallelPreBindPlugins returns plugins that can be run in parallel with other plugins
	// in the PreBind extension point.
	parallelPreBindPlugins sets.Set[string]
	// podGroupCycleState contains the CycleState for this pod's PodGroup.
	// If set to nil, it means that the pod referencing this CycleState either passed the pod group cycle
	// or doesn't belong to any pod group.
	// This field can only be non-nil when GenericWorkload feature flag is enabled.
	podGroupCycleState fwk.PodGroupCycleState
	// placementCycleState contains the CycleState for the current Placement being evaluated.
	// If set to nil, it means this pod is not being scheduled within a placement context.
	// This field can only be non-nil when GenericWorkload feature flag is enabled.
	placementCycleState fwk.PlacementCycleState
	// placementStates holds per-placement state keyed by Placement pointer.
	// PlacementGeneratePlugins populate it via PlacementState during placement generation.
	// The framework merges entries across plugins and copies the state for a placement
	// into the per-placement CycleState before placement evaluation.
	// Only populated on a PodGroup-level CycleState.
	placementStates map[*fwk.Placement]*CycleState
	// placementStatesMu guards placementStates.
	placementStatesMu sync.Mutex
}

// NewCycleState initializes a new CycleState and returns its pointer.
func NewCycleState() *CycleState {
	return &CycleState{}
}

// ShouldRecordPluginMetrics returns whether metrics.PluginExecutionDuration metrics should be recorded.
func (c *CycleState) ShouldRecordPluginMetrics() bool {
	if c == nil {
		return false
	}
	return c.recordPluginMetrics
}

// SetRecordPluginMetrics sets recordPluginMetrics to the given value.
func (c *CycleState) SetRecordPluginMetrics(flag bool) {
	if c == nil {
		return
	}
	c.recordPluginMetrics = flag
}

func (c *CycleState) SetSkipFilterPlugins(plugins sets.Set[string]) {
	c.skipFilterPlugins = plugins
}

func (c *CycleState) GetSkipFilterPlugins() sets.Set[string] {
	return c.skipFilterPlugins
}

func (c *CycleState) SetSkipScorePlugins(plugins sets.Set[string]) {
	c.skipScorePlugins = plugins
}

func (c *CycleState) GetSkipScorePlugins() sets.Set[string] {
	return c.skipScorePlugins
}

func (c *CycleState) SetSkipPreBindPlugins(plugins sets.Set[string]) {
	c.skipPreBindPlugins = plugins
}

func (c *CycleState) GetSkipPreBindPlugins() sets.Set[string] {
	return c.skipPreBindPlugins
}

func (c *CycleState) SetParallelPreBindPlugins(plugins sets.Set[string]) {
	c.parallelPreBindPlugins = plugins
}

func (c *CycleState) GetParallelPreBindPlugins() sets.Set[string] {
	return c.parallelPreBindPlugins
}

func (c *CycleState) IsPodGroupSchedulingCycle() bool {
	return c.podGroupCycleState != nil
}

func (c *CycleState) SetPodGroupCycleState(podGroupCycleState fwk.PodGroupCycleState) {
	c.podGroupCycleState = podGroupCycleState
}

func (c *CycleState) GetPodGroupCycleState() fwk.PodGroupCycleState {
	return c.podGroupCycleState
}

func (c *CycleState) GetPlacementCycleState() fwk.PlacementCycleState {
	return c.placementCycleState
}

// GetParentPlacementCycleState implements [fwk.PodGroupCycleState.GetParentPlacementCycleState].
// We can reuse the same placementCycleState field as GetPlacementCycleState, as it implements a different interface
// and the two are mutually exclusive.
func (c *CycleState) GetParentPlacementCycleState() fwk.PlacementCycleState {
	return c.placementCycleState
}

func (c *CycleState) SetPlacementCycleState(placementCycleState fwk.PlacementCycleState) {
	c.placementCycleState = placementCycleState
}

// PlacementState returns the PlacementCycleState for the given placement.
// If no state exists for the placement, the method initializes and returns a new one.
// If placement is nil, the method returns nil.
func (c *CycleState) PlacementState(placement *fwk.Placement) fwk.PlacementCycleState {
	if c == nil || placement == nil {
		return nil
	}
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	if c.placementStates == nil {
		c.placementStates = make(map[*fwk.Placement]*CycleState)
	}
	ps := c.placementStates[placement]
	if ps == nil {
		ps = NewCycleState()
		c.placementStates[placement] = ps
	}
	return ps
}

// DeletePlacementStates removes the PlacementCycleState entries for the given placements.
// The framework calls this method after merging placements to remove intermediate parent states.
func (c *CycleState) DeletePlacementStates(placements ...*fwk.Placement) {
	if c == nil {
		return
	}
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	if len(c.placementStates) == 0 {
		return
	}
	for _, p := range placements {
		delete(c.placementStates, p)
	}
}

// CopyPlacementDataInto clones every StateData entry registered for placement and writes
// it into dst. The scheduler calls this method to seed a per-placement CycleState before
// placement evaluation.
func (c *CycleState) CopyPlacementDataInto(placement *fwk.Placement, dst *CycleState) {
	if c == nil || placement == nil || dst == nil {
		return
	}
	c.placementStatesMu.Lock()
	src := c.placementStates[placement]
	c.placementStatesMu.Unlock()
	if src == nil {
		return
	}
	src.storage.Range(func(k, v interface{}) bool {
		dst.storage.Store(k, v.(fwk.StateData).Clone())
		return true
	})
}

// MergePlacementStatesInto combines the placement states for srcs into a single
// PlacementCycleState for dst. The method clones each StateData entry so the merged
// placement does not share mutable state with the source placements. If no source
// holds state, the method leaves dst unregistered and allocates nothing on the heap.
// If two source states contain the same StateKey, the method returns an error.
func (c *CycleState) MergePlacementStatesInto(dst *fwk.Placement, srcs ...*fwk.Placement) error {
	if c == nil || dst == nil {
		return nil
	}
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	if len(c.placementStates) == 0 {
		return nil
	}

	var merged *CycleState
	for _, srcPlacement := range srcs {
		src := c.placementStates[srcPlacement]
		if src == nil {
			continue
		}
		var conflict *fwk.StateKey
		src.storage.Range(func(k, v interface{}) bool {
			key := k.(fwk.StateKey)
			if merged == nil {
				merged = NewCycleState()
			}
			if _, loaded := merged.storage.LoadOrStore(key, v.(fwk.StateData).Clone()); loaded {
				keyCopy := key
				conflict = &keyCopy
				return false
			}
			return true
		})
		if conflict != nil {
			return fmt.Errorf("conflicting placement cycle state key %q while merging into placement %q", *conflict, dst.Name)
		}
	}
	if merged == nil {
		return nil
	}
	c.placementStates[dst] = merged
	return nil
}

func (c *CycleState) SetSkipAllPostFilterPlugins(flag bool) {
	c.skipAllPostFilterPlugins = flag
}

func (c *CycleState) ShouldSkipAllPostFilterPlugins() bool {
	return c.skipAllPostFilterPlugins
}

// Clone creates a copy of CycleState and returns its pointer. Clone returns
// nil if the context being cloned is nil.
func (c *CycleState) Clone() fwk.CycleState {
	if c == nil {
		return nil
	}
	copy := NewCycleState()
	// Safe copy storage in case of overwriting.
	c.storage.Range(func(k, v interface{}) bool {
		copy.storage.Store(k, v.(fwk.StateData).Clone())
		return true
	})
	// The below are not mutated, so we don't have to safe copy.
	copy.recordPluginMetrics = c.recordPluginMetrics
	copy.skipFilterPlugins = c.skipFilterPlugins
	copy.skipScorePlugins = c.skipScorePlugins
	copy.skipPreBindPlugins = c.skipPreBindPlugins
	copy.parallelPreBindPlugins = c.parallelPreBindPlugins
	copy.podGroupCycleState = c.podGroupCycleState
	copy.placementCycleState = c.placementCycleState
	copy.skipAllPostFilterPlugins = c.skipAllPostFilterPlugins

	// Deep copy the per-placement states so the clone does not share mutable StateData.
	c.placementStatesMu.Lock()
	if len(c.placementStates) > 0 {
		copy.placementStates = make(map[*fwk.Placement]*CycleState, len(c.placementStates))
		for p, st := range c.placementStates {
			if st == nil {
				continue
			}
			dup := NewCycleState()
			st.storage.Range(func(k, v interface{}) bool {
				dup.storage.Store(k, v.(fwk.StateData).Clone())
				return true
			})
			copy.placementStates[p] = dup
		}
	}
	c.placementStatesMu.Unlock()

	return copy
}

// Read retrieves data with the given "key" from CycleState. If the key is not
// present, ErrNotFound is returned.
//
// See CycleState for notes on concurrency.
func (c *CycleState) Read(key fwk.StateKey) (fwk.StateData, error) {
	if v, ok := c.storage.Load(key); ok {
		return v.(fwk.StateData), nil
	}
	return nil, fwk.ErrNotFound
}

// Write stores the given "val" in CycleState with the given "key".
//
// See CycleState for notes on concurrency.
func (c *CycleState) Write(key fwk.StateKey, val fwk.StateData) {
	c.storage.Store(key, val)
}

// Delete deletes data with the given key from CycleState.
//
// See CycleState for notes on concurrency.
func (c *CycleState) Delete(key fwk.StateKey) {
	c.storage.Delete(key)
}
