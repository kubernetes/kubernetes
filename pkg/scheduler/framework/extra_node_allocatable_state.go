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
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fwk "k8s.io/kube-scheduler/framework"
)

// ExtraNodeAllocatableResourcesStateKey is the CycleState key for
// storing extra node allocatable resources reported by plugins.
var ExtraNodeAllocatableResourcesStateKey fwk.StateKey = "kubernetes.io/extra-node-allocatable-resources"

// ExtraNodeAllocatableResourcesState stores per-node extra node
// allocatable resources reported by plugins during a scheduling cycle,
// and which plugins wrote them to the pod status in PreBind.
//
// status.extraNodeAllocatableResources is one list, and each write replaces it. A plugin that
// contributes entries:
//  1. In Filter, records its entries for each node with Set.
//  2. In PreBind, writes its own recorded entries plus the ones that other plugins wrote in one
//     request, with UpdatePersisted.
//
// Entries that plugins only recorded are left out, because their PreBind has not written them
// yet and may still fail.
type ExtraNodeAllocatableResourcesState struct {
	mu sync.RWMutex
	// nodeToPluginResources maps node name to per-plugin extra node allocatable resources.
	nodeToPluginResources map[string]map[string][]v1.ExtraNodeAllocatableResource
	// persisted holds the plugins whose PreBind wrote their entries to the pod status. PreBind
	// runs only for the node that the pod is bound to, so we do not need to key this by node name.
	// Nothing removes plugins from it: a failed attempt drops this state.
	persisted sets.Set[string]
}

// NewExtraNodeAllocatableResourcesState instantiates an ExtraNodeAllocatableResourcesState object.
func NewExtraNodeAllocatableResourcesState() *ExtraNodeAllocatableResourcesState {
	return &ExtraNodeAllocatableResourcesState{
		nodeToPluginResources: make(map[string]map[string][]v1.ExtraNodeAllocatableResource),
		persisted:             sets.New[string](),
	}
}

// Clone returns the same state.
func (s *ExtraNodeAllocatableResourcesState) Clone() fwk.StateData {
	return s
}

// Set records entries for the given nodeName and pluginName, replacing any existing entries for that plugin on the node.
func (s *ExtraNodeAllocatableResourcesState) Set(nodeName, pluginName string, entries []v1.ExtraNodeAllocatableResource) {
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
	copied := make([]v1.ExtraNodeAllocatableResource, len(entries))
	for i := range entries {
		entries[i].DeepCopyInto(&copied[i])
	}
	pluginRes[pluginName] = copied
}

// Get returns a copy of all entries recorded for nodeName.
func (s *ExtraNodeAllocatableResourcesState) Get(nodeName string) []v1.ExtraNodeAllocatableResource {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return flattenByPluginName(s.nodeToPluginResources[nodeName], "")
}

// GetExcludingPlugin returns a copy of the entries recorded for nodeName by all plugins except pluginName.
func (s *ExtraNodeAllocatableResourcesState) GetExcludingPlugin(nodeName, pluginName string) []v1.ExtraNodeAllocatableResource {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return flattenByPluginName(s.nodeToPluginResources[nodeName], pluginName)
}

// GetFromPlugin returns a copy of the entries recorded for nodeName by pluginName.
func (s *ExtraNodeAllocatableResourcesState) GetFromPlugin(nodeName, pluginName string) []v1.ExtraNodeAllocatableResource {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return copyEntries(s.nodeToPluginResources[nodeName][pluginName])
}

// GetPersisted returns a copy of the entries recorded for nodeName by plugins whose PreBind
// wrote them to the pod status, ordered by plugin name.
func (s *ExtraNodeAllocatableResourcesState) GetPersisted(nodeName string) []v1.ExtraNodeAllocatableResource {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.getPersistedLocked(nodeName)
}

// UpdatePersisted calls update with copies of the entries that plugins persisted so far for nodeName,
// ordered by plugin name, and adds the entries of pluginName to them if update succeeds. PreBind of
// plugins can run in parallel and each write of the field replaces the whole list, so update runs
// under the lock of the state. update must not call the methods of the state.
func (s *ExtraNodeAllocatableResourcesState) UpdatePersisted(nodeName, pluginName string, update func(persisted []v1.ExtraNodeAllocatableResource) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := update(s.getPersistedLocked(nodeName)); err != nil {
		return err
	}
	s.setPersistedLocked(pluginName)
	return nil
}

// getPersistedLocked returns copies of the entries recorded for nodeName by plugins whose PreBind
// wrote them to the pod status, ordered by plugin name. The caller holds the lock.
func (s *ExtraNodeAllocatableResourcesState) getPersistedLocked(nodeName string) []v1.ExtraNodeAllocatableResource {
	var out []v1.ExtraNodeAllocatableResource
	for _, pluginName := range sets.List(s.persisted) {
		out = append(out, copyEntries(s.nodeToPluginResources[nodeName][pluginName])...)
	}
	return MergeExtraNodeAllocatableResources(out)
}

// setPersistedLocked marks the entries of pluginName as written to the pod status. The caller holds the lock.
func (s *ExtraNodeAllocatableResourcesState) setPersistedLocked(pluginName string) {
	s.persisted.Insert(pluginName)
}

// flattenByPluginName returns copies of the entries of all plugins in entriesByPlugin except
// excludedPlugin, sorted by plugin name.
func flattenByPluginName(entriesByPlugin map[string][]v1.ExtraNodeAllocatableResource, excludedPlugin string) []v1.ExtraNodeAllocatableResource {
	if len(entriesByPlugin) == 0 {
		return nil
	}
	pluginNames := make([]string, 0, len(entriesByPlugin))
	for pluginName := range entriesByPlugin {
		if pluginName != excludedPlugin {
			pluginNames = append(pluginNames, pluginName)
		}
	}
	// Map order is random, and PreBind in DRA plugin compares the recorded list with the copy that
	// the assumed pod got.
	slices.Sort(pluginNames)

	var out []v1.ExtraNodeAllocatableResource
	for _, pluginName := range pluginNames {
		out = append(out, copyEntries(entriesByPlugin[pluginName])...)
	}
	return MergeExtraNodeAllocatableResources(out)
}

func copyEntries(entries []v1.ExtraNodeAllocatableResource) []v1.ExtraNodeAllocatableResource {
	var out []v1.ExtraNodeAllocatableResource
	for _, entry := range entries {
		out = append(out, *entry.DeepCopy())
	}
	return out
}

// MergeExtraNodeAllocatableResources returns copies of the entries in lists, in order, with one
// entry per container name. The pod status has one entry per container name, so the requests and
// limits of entries with the same container name are added up.
func MergeExtraNodeAllocatableResources(lists ...[]v1.ExtraNodeAllocatableResource) []v1.ExtraNodeAllocatableResource {
	var out []v1.ExtraNodeAllocatableResource
	indexByContainerName := make(map[string]int)
	for _, list := range lists {
		for _, entry := range list {
			i, ok := indexByContainerName[entry.ContainerName]
			if !ok {
				indexByContainerName[entry.ContainerName] = len(out)
				out = append(out, *entry.DeepCopy())
				continue
			}
			addResources(&out[i].Requests, entry.Requests)
			addResources(&out[i].Limits, entry.Limits)
		}
	}
	return out
}

// addResources adds the quantities in src to *dst.
func addResources(dst *v1.ResourceList, src v1.ResourceList) {
	for name, quantity := range src {
		if *dst == nil {
			*dst = v1.ResourceList{}
		}
		sum := (*dst)[name]
		sum.Add(quantity)
		(*dst)[name] = sum
	}
}

// Has returns true if any entries are recorded for nodeName.
func (s *ExtraNodeAllocatableResourcesState) Has(nodeName string) bool {
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

// createExtraNodeAllocatableResourcesStateMu makes the read and the write in
// GetOrCreateExtraNodeAllocatableResourcesState atomic in case the function in called
// from a stage which can run in parallel for multiple nodes.
var createExtraNodeAllocatableResourcesStateMu sync.Mutex

// GetOrCreateExtraNodeAllocatableResourcesState returns the existing
// ExtraNodeAllocatableResourcesState from CycleState, or creates and stores a new one.
func GetOrCreateExtraNodeAllocatableResourcesState(cs fwk.CycleState) *ExtraNodeAllocatableResourcesState {
	createExtraNodeAllocatableResourcesStateMu.Lock()
	defer createExtraNodeAllocatableResourcesStateMu.Unlock()
	if s := GetExtraNodeAllocatableResourcesState(cs); s != nil {
		return s
	}
	s := NewExtraNodeAllocatableResourcesState()
	cs.Write(ExtraNodeAllocatableResourcesStateKey, s)
	return s
}

// GetExtraNodeAllocatableResources returns the entries that plugins recorded in cs for nodeName.
func GetExtraNodeAllocatableResources(cs fwk.CycleState, nodeName string) []v1.ExtraNodeAllocatableResource {
	if s := GetExtraNodeAllocatableResourcesState(cs); s != nil {
		return s.Get(nodeName)
	}
	return nil
}
