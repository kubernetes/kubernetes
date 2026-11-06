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

package runtime

import (
	"context"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// fillAllocationExtensions collects the plugins which implement
// framework.AllocationExtensions or framework.VictimExtensions. They are not
// configured per extension point, so every instantiated plugin of the
// profile is considered.
func (f *frameworkImpl) fillAllocationExtensions() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAPreemption) {
		return
	}
	names := make([]string, 0, len(f.pluginsMap))
	for name := range f.pluginsMap {
		names = append(names, name)
	}
	// Sorted for a deterministic call order.
	slices.Sort(names)
	for _, name := range names {
		if ext, ok := f.pluginsMap[name].(framework.AllocationExtensions); ok {
			f.allocationExtensions = append(f.allocationExtensions, ext)
		}
		if ext, ok := f.pluginsMap[name].(framework.VictimExtensions); ok {
			f.victimExtensions = append(f.victimExtensions, ext)
		}
	}
	if len(f.allocationExtensions) > 0 {
		f.nominatingCycleStates = &nominatingCycleStates{states: make(map[types.UID]*nominatingCycleState)}
	}
}

// nominatingCycleStates holds the CycleState of scheduling cycles whose
// nominations the PodNominator is about to apply. It is nil when no plugin
// implements framework.AllocationExtensions.
type nominatingCycleStates struct {
	mutex  sync.Mutex
	states map[types.UID]*nominatingCycleState
}

type nominatingCycleState struct {
	nodeName string
	state    fwk.CycleState
}

// SetNominatingCycleState makes state available to the AddAllocation call
// for the nomination of pod to nodeName until done gets called.
func (f *frameworkImpl) SetNominatingCycleState(pod *v1.Pod, nodeName string, state fwk.CycleState) (done func()) {
	c := f.nominatingCycleStates
	if c == nil || nodeName == "" || state == nil {
		return func() {}
	}
	entry := &nominatingCycleState{nodeName: nodeName, state: state}
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.states[pod.UID] = entry
	return func() {
		c.mutex.Lock()
		defer c.mutex.Unlock()
		// A later call for the same pod may have replaced the entry.
		if c.states[pod.UID] == entry {
			delete(c.states, pod.UID)
		}
	}
}

// TakeNominatingCycleState returns the state set for the nomination of pod to
// nodeName and drops it, so that only the first AddAllocation for it gets it.
func (f *frameworkImpl) TakeNominatingCycleState(pod *v1.Pod, nodeName string) fwk.CycleState {
	c := f.nominatingCycleStates
	if c == nil {
		return nil
	}
	c.mutex.Lock()
	defer c.mutex.Unlock()
	entry, ok := c.states[pod.UID]
	if !ok || entry.nodeName != nodeName {
		return nil
	}
	delete(c.states, pod.UID)
	return entry.state
}

// RunAllocationExtensionsAddAllocation calls AddAllocation of all plugins and
// returns the events of all of them.
func (f *frameworkImpl) RunAllocationExtensionsAddAllocation(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) []framework.AllocationEvent {
	var events []framework.AllocationEvent
	for _, ext := range f.allocationExtensions {
		events = append(events, ext.AddAllocation(f.extensionsContext(ctx, "AllocationExtensions", ext), state, pod, nodeName).Events...)
	}
	return events
}

// RunAllocationExtensionsRemoveAllocation calls RemoveAllocation of all
// plugins and returns the events of all of them.
func (f *frameworkImpl) RunAllocationExtensionsRemoveAllocation(ctx context.Context, pod *v1.Pod, nodeName string) []framework.AllocationEvent {
	var events []framework.AllocationEvent
	for _, ext := range f.allocationExtensions {
		events = append(events, ext.RemoveAllocation(f.extensionsContext(ctx, "AllocationExtensions", ext), pod, nodeName).Events...)
	}
	return events
}

func (f *frameworkImpl) extensionsContext(ctx context.Context, kind string, ext fwk.Plugin) context.Context {
	logger := klog.FromContext(ctx)
	if !logger.V(4).Enabled() {
		return ctx
	}
	return klog.NewContext(ctx, klog.LoggerWithName(klog.LoggerWithName(logger, kind), ext.Name()))
}
