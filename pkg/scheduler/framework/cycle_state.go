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
	// placementCycleStatesByName holds per-placement state keyed by placement name. It is
	// populated by PlacementGeneratePlugins (via SetPlacementCycleStateForName) during
	// placement generation, combined by the framework when merging placements, and consumed
	// to seed the per-placement CycleState before each placement is simulated.
	// Only ever populated on a PodGroup-level CycleState.
	placementCycleStatesByName map[string]fwk.PlacementCycleState
	// placementStatesMu guards placementCycleStatesByName.
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

func (c *CycleState) SetPodGroupSchedulingCycle(podGroupCycleState fwk.PodGroupCycleState) {
	c.podGroupCycleState = podGroupCycleState
}

func (c *CycleState) GetPodGroupSchedulingCycle() fwk.PodGroupCycleState {
	return c.podGroupCycleState
}

func (c *CycleState) GetPlacementCycleState() fwk.PlacementCycleState {
	return c.placementCycleState
}

func (c *CycleState) SetPlacementCycleState(placementCycleState fwk.PlacementCycleState) {
	c.placementCycleState = placementCycleState
}

// GetPlacementCycleStateForName returns the PlacementCycleState registered under the given
// placement name, or nil if none is registered.
func (c *CycleState) GetPlacementCycleStateForName(placementName string) fwk.PlacementCycleState {
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	return c.placementCycleStatesByName[placementName]
}

// SetPlacementCycleStateForName registers the PlacementCycleState for the given placement name.
func (c *CycleState) SetPlacementCycleStateForName(placementName string, state fwk.PlacementCycleState) {
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	if c.placementCycleStatesByName == nil {
		c.placementCycleStatesByName = make(map[string]fwk.PlacementCycleState)
	}
	c.placementCycleStatesByName[placementName] = state
}

// DeletePlacementCycleStateForName removes the PlacementCycleState for the given placement name.
func (c *CycleState) DeletePlacementCycleStateForName(placementName string) {
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()
	delete(c.placementCycleStatesByName, placementName)
}

// CopyPlacementDataInto clones every StateData entry stored in this state and writes it
// into dst. It is used by the framework to seed a per-placement CycleState from the named
// placement state produced during placement generation.
func (c *CycleState) CopyPlacementDataInto(dst *CycleState) {
	if c == nil || dst == nil {
		return
	}
	c.storage.Range(func(k, v interface{}) bool {
		dst.storage.Store(k, v.(fwk.StateData).Clone())
		return true
	})
}

// MergePlacementStatesInto combines the placement states registered under srcNames into a
// single PlacementCycleState registered under dstName. Each StateData entry is cloned so the
// merged placement does not share mutable state with the source placements (which may be
// reused across other merged placements). It returns an error if two source states write the
// same StateKey, which signals that plugins are not using disjoint keys as required.
func (c *CycleState) MergePlacementStatesInto(dstName string, srcNames ...string) error {
	c.placementStatesMu.Lock()
	defer c.placementStatesMu.Unlock()

	merged := NewCycleState()
	for _, name := range srcNames {
		src, ok := c.placementCycleStatesByName[name].(*CycleState)
		if !ok || src == nil {
			continue
		}
		var conflict *fwk.StateKey
		src.storage.Range(func(k, v interface{}) bool {
			key := k.(fwk.StateKey)
			if _, loaded := merged.storage.Load(key); loaded {
				keyCopy := key
				conflict = &keyCopy
				return false
			}
			merged.storage.Store(key, v.(fwk.StateData).Clone())
			return true
		})
		if conflict != nil {
			return fmt.Errorf("conflicting placement cycle state key %q while merging placements %v into %q", *conflict, srcNames, dstName)
		}
	}
	if c.placementCycleStatesByName == nil {
		c.placementCycleStatesByName = make(map[string]fwk.PlacementCycleState)
	}
	c.placementCycleStatesByName[dstName] = merged
	return nil
}

// clonePlacementCycleState returns a deep copy of the placement-scoped StateData held by
// state, or returns state unchanged if it is not backed by a *CycleState.
func clonePlacementCycleState(state fwk.PlacementCycleState) fwk.PlacementCycleState {
	cs, ok := state.(*CycleState)
	if !ok || cs == nil {
		return state
	}
	dup := NewCycleState()
	cs.CopyPlacementDataInto(dup)
	return dup
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

	// Deep copy the named placement states so the clone does not share mutable StateData.
	c.placementStatesMu.Lock()
	if c.placementCycleStatesByName != nil {
		copy.placementCycleStatesByName = make(map[string]fwk.PlacementCycleState, len(c.placementCycleStatesByName))
		for name, st := range c.placementCycleStatesByName {
			copy.placementCycleStatesByName[name] = clonePlacementCycleState(st)
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
