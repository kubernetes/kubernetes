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

// AdditionalNodeAllocatableResourcesStateKey is the CycleState key for
// storing additional node allocatable resources reported by plugins.
var AdditionalNodeAllocatableResourcesStateKey fwk.StateKey = "kubernetes.io/additional-node-allocatable-resources"

// AdditionalNodeAllocatableResourcesState stores per-node additional node
// allocatable resources reported by plugins during a scheduling cycle.
type AdditionalNodeAllocatableResourcesState struct {
	mu sync.RWMutex
	// nodeToPluginResources maps node name to per-plugin additional node allocatable resources.
	nodeToPluginResources map[string]map[string][]v1.AdditionalNodeAllocatableResource
}

// NewAdditionalNodeAllocatableResourcesState instantiates an AdditionalNodeAllocatableResourcesState object.
func NewAdditionalNodeAllocatableResourcesState() *AdditionalNodeAllocatableResourcesState {
	return &AdditionalNodeAllocatableResourcesState{
		nodeToPluginResources: make(map[string]map[string][]v1.AdditionalNodeAllocatableResource),
	}
}

// Clone just returns the same state.
func (s *AdditionalNodeAllocatableResourcesState) Clone() fwk.StateData {
	return s
}

// Set records entries for the given nodeName and pluginName, replacing any existing entries for that plugin on the node.
func (s *AdditionalNodeAllocatableResourcesState) Set(nodeName, pluginName string, entries []v1.AdditionalNodeAllocatableResource) {
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
		pluginRes = make(map[string][]v1.AdditionalNodeAllocatableResource)
		s.nodeToPluginResources[nodeName] = pluginRes
	}
	pluginRes[pluginName] = entries
}

// Get returns a copy of all entries recorded for nodeName.
func (s *AdditionalNodeAllocatableResourcesState) Get(nodeName string) []v1.AdditionalNodeAllocatableResource {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()

	pluginRes := s.nodeToPluginResources[nodeName]
	if len(pluginRes) == 0 {
		return nil
	}
	var out []v1.AdditionalNodeAllocatableResource
	// TODO(pravk03): Re-evaluate if this deep copy is required. Current consumers don't modify the
	// returned resources.
	for _, entries := range pluginRes {
		for i := range entries {
			var entry v1.AdditionalNodeAllocatableResource
			entries[i].DeepCopyInto(&entry)
			out = append(out, entry)
		}
	}
	return out
}

// Has returns true if any entries are recorded for nodeName.
func (s *AdditionalNodeAllocatableResourcesState) Has(nodeName string) bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.nodeToPluginResources[nodeName]) > 0
}

// GetAdditionalNodeAllocatableResourcesState reads AdditionalNodeAllocatableResourcesState from CycleState.
func GetAdditionalNodeAllocatableResourcesState(cs fwk.CycleState) *AdditionalNodeAllocatableResourcesState {
	if cs == nil {
		return nil
	}
	data, err := cs.Read(AdditionalNodeAllocatableResourcesStateKey)
	if err != nil {
		return nil
	}
	s, _ := data.(*AdditionalNodeAllocatableResourcesState)
	return s
}

// GetOrCreateAdditionalNodeAllocatableResourcesState returns the existing
// AdditionalNodeAllocatableResourcesState from CycleState, or creates and stores a new one.
func GetOrCreateAdditionalNodeAllocatableResourcesState(cs fwk.CycleState) *AdditionalNodeAllocatableResourcesState {
	if s := GetAdditionalNodeAllocatableResourcesState(cs); s != nil {
		return s
	}
	s := NewAdditionalNodeAllocatableResourcesState()
	cs.Write(AdditionalNodeAllocatableResourcesStateKey, s)
	return s
}
