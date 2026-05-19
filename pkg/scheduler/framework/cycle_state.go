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
