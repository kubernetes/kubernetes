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
		f.preparedAllocations = &preparedAllocations{byPod: make(map[types.UID]*preparedAllocation)}
	}
}

// preparedAllocations holds the results of PrepareAllocation for nominations
// which the PodNominator is about to make, by pod UID.
type preparedAllocations struct {
	mutex sync.Mutex
	byPod map[types.UID]*preparedAllocation
}

type preparedAllocation struct {
	nodeName string
	// results has one entry per plugin in allocationExtensions.
	results []any
}

// RunAllocationExtensionsPrepareAllocation calls PrepareAllocation of all
// plugins and keeps the results for the nomination of pod to nodeName until
// done gets called.
func (f *frameworkImpl) RunAllocationExtensionsPrepareAllocation(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (done func()) {
	p := f.preparedAllocations
	if p == nil {
		return func() {}
	}
	entry := &preparedAllocation{nodeName: nodeName, results: make([]any, len(f.allocationExtensions))}
	for i, ext := range f.allocationExtensions {
		entry.results[i] = ext.PrepareAllocation(f.extensionsContext(ctx, "AllocationExtensions", ext), state, pod, nodeName)
	}
	p.mutex.Lock()
	defer p.mutex.Unlock()
	p.byPod[pod.UID] = entry
	return func() {
		p.mutex.Lock()
		defer p.mutex.Unlock()
		// A later call for the same pod may have replaced the entry.
		if p.byPod[pod.UID] == entry {
			delete(p.byPod, pod.UID)
		}
	}
}

// takePreparedAllocation returns and drops the results of PrepareAllocation
// for the nomination of the pod to the node, nil if there are none.
func (f *frameworkImpl) takePreparedAllocation(podUID types.UID, nodeName string) []any {
	p := f.preparedAllocations
	if p == nil {
		return nil
	}
	p.mutex.Lock()
	defer p.mutex.Unlock()
	entry := p.byPod[podUID]
	if entry == nil || entry.nodeName != nodeName {
		return nil
	}
	delete(p.byPod, podUID)
	return entry.results
}

// RunAllocationExtensionsAddAllocation calls AddAllocation of all plugins,
// with the results of their PrepareAllocation for the nomination, if they are
// still kept, and returns the events of all of them.
func (f *frameworkImpl) RunAllocationExtensionsAddAllocation(ctx context.Context, pod *v1.Pod, nodeName string) []framework.AllocationEvent {
	prepared := f.takePreparedAllocation(pod.UID, nodeName)
	var events []framework.AllocationEvent
	for i, ext := range f.allocationExtensions {
		var result any
		if prepared != nil {
			result = prepared[i]
		}
		events = append(events, ext.AddAllocation(f.extensionsContext(ctx, "AllocationExtensions", ext), pod, nodeName, result).Events...)
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
