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

package framework

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// ExtraNodeAllocatableResourcesStateKey is the CycleState key for
// storing extra node allocatable resources reported by plugins.
var ExtraNodeAllocatableResourcesStateKey fwk.StateKey = "kubernetes.io/extra-node-allocatable-resources"

// ExtraNodeAllocatableResourcesState stores per-node extra node
// allocatable resources reported by plugins during a scheduling cycle.
type ExtraNodeAllocatableResourcesState struct {
	mu sync.RWMutex
	// nodeToPluginResources maps node name to per-plugin extra node allocatable resources.
	nodeToPluginResources map[string]map[string][]v1.ExtraNodeAllocatableResource
}

// NewExtraNodeAllocatableResourcesState instantiates an ExtraNodeAllocatableResourcesState object.
func NewExtraNodeAllocatableResourcesState() *ExtraNodeAllocatableResourcesState {
	return &ExtraNodeAllocatableResourcesState{
		nodeToPluginResources: make(map[string]map[string][]v1.ExtraNodeAllocatableResource),
	}
}

// Clone just returns the same state.
func (s *ExtraNodeAllocatableResourcesState) Clone() fwk.StateData {
	return s
}

// Set records entries for the given nodeName and pluginName, replacing any existing entries for that plugin on the node.
func (s *ExtraNodeAllocatableResourcesState) Set(nodeName, pluginName string, entries []v1.ExtraNodeAllocatableResource) {
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	pluginRes := s.nodeToPluginResources[nodeName]
	if len(entries) == 0 {
		delete(pluginRes, pluginName)
		if len(pluginRes) == 0 {
			delete(s.nodeToPluginResources, nodeName)
		}
		return
	}
	if pluginRes == nil {
		pluginRes = make(map[string][]v1.ExtraNodeAllocatableResource)
		s.nodeToPluginResources[nodeName] = pluginRes
	}
	pluginRes[pluginName] = entries
}

// Get returns a copy of all entries recorded for nodeName.
func (s *ExtraNodeAllocatableResourcesState) Get(nodeName string) []v1.ExtraNodeAllocatableResource {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	pluginRes := s.nodeToPluginResources[nodeName]
	if len(pluginRes) == 0 {
		return nil
	}
	var out []v1.ExtraNodeAllocatableResource
	// TODO(pravk03): Re-evaluate if this deep copy is required. Current consumers don't modify the
	// returned resources.
	for _, entries := range pluginRes {
		for i := range entries {
			var entry v1.ExtraNodeAllocatableResource
			entries[i].DeepCopyInto(&entry)
			out = append(out, entry)
		}
	}
	return out
}

// Has returns true if any entries are recorded for nodeName.
func (s *ExtraNodeAllocatableResourcesState) Has(nodeName string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.nodeToPluginResources[nodeName]) > 0
}

// GetExtraNodeAllocatableResourcesState reads ExtraNodeAllocatableResourcesState from CycleState.
func GetExtraNodeAllocatableResourcesState(cs fwk.CycleState) *ExtraNodeAllocatableResourcesState {
	if cs == nil {
		return nil
	}
	data, err := cs.Read(ExtraNodeAllocatableResourcesStateKey)
	if err != nil {
		return nil
	}
	s, _ := data.(*ExtraNodeAllocatableResourcesState)
	return s
}

// GetOrCreateExtraNodeAllocatableResourcesState returns the existing
// ExtraNodeAllocatableResourcesState from CycleState, or creates and stores a new one.
func GetOrCreateExtraNodeAllocatableResourcesState(cs fwk.CycleState) *ExtraNodeAllocatableResourcesState {
	if s := GetExtraNodeAllocatableResourcesState(cs); s != nil {
		return s
	}
	s := NewExtraNodeAllocatableResourcesState()
	cs.Write(ExtraNodeAllocatableResourcesStateKey, s)
	return s
}
