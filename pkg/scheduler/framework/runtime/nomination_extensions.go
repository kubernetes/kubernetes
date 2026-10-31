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

// fillNominationExtensions collects the plugins which implement
// framework.NominationExtensions. They are not configured per extension
// point, so every instantiated plugin of the profile is considered.
func (f *frameworkImpl) fillNominationExtensions() {
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
		if ext, ok := f.pluginsMap[name].(framework.NominationExtensions); ok {
			f.nominationExtensions = append(f.nominationExtensions, ext)
		}
	}
	if len(f.nominationExtensions) > 0 {
		f.preparedNominations = &preparedNominations{byPod: make(map[types.UID]*preparedNomination)}
	}
}

// preparedNominations holds the results of PrepareNomination for nominations
// which the PodNominator is about to make, by pod UID.
type preparedNominations struct {
	mutex sync.Mutex
	byPod map[types.UID]*preparedNomination
}

type preparedNomination struct {
	nodeName string
	// results has one entry per plugin in nominationExtensions.
	results []any
}

// RunNominationExtensionsPrepareNomination calls PrepareNomination of all
// plugins and keeps the results for the nomination of pod to nodeName until
// done gets called.
func (f *frameworkImpl) RunNominationExtensionsPrepareNomination(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (done func()) {
	p := f.preparedNominations
	if p == nil {
		return func() {}
	}
	entry := &preparedNomination{nodeName: nodeName, results: make([]any, len(f.nominationExtensions))}
	for i, ext := range f.nominationExtensions {
		entry.results[i] = ext.PrepareNomination(f.nominationExtensionsContext(ctx, ext), state, pod, nodeName)
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

// takePreparedNomination returns and drops the results of PrepareNomination
// for the nomination of the pod to the node, nil if there are none.
func (f *frameworkImpl) takePreparedNomination(podUID types.UID, nodeName string) []any {
	p := f.preparedNominations
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

// RunNominationExtensionsAddNomination calls AddNomination of all plugins,
// with the results of their PrepareNomination for the nomination, if they are
// still kept, and returns the events of all of them.
func (f *frameworkImpl) RunNominationExtensionsAddNomination(ctx context.Context, pod *v1.Pod, nodeName string) []framework.NominationEvent {
	prepared := f.takePreparedNomination(pod.UID, nodeName)
	var events []framework.NominationEvent
	for i, ext := range f.nominationExtensions {
		var result any
		if prepared != nil {
			result = prepared[i]
		}
		events = append(events, ext.AddNomination(f.nominationExtensionsContext(ctx, ext), pod, nodeName, result).Events...)
	}
	return events
}

// RunNominationExtensionsRemoveNomination calls RemoveNomination of all
// plugins and returns the events of all of them.
func (f *frameworkImpl) RunNominationExtensionsRemoveNomination(ctx context.Context, pod *v1.Pod, nodeName string) []framework.NominationEvent {
	var events []framework.NominationEvent
	for _, ext := range f.nominationExtensions {
		events = append(events, ext.RemoveNomination(f.nominationExtensionsContext(ctx, ext), pod, nodeName).Events...)
	}
	return events
}

func (f *frameworkImpl) nominationExtensionsContext(ctx context.Context, ext fwk.Plugin) context.Context {
	logger := klog.FromContext(ctx)
	if !logger.V(4).Enabled() {
		return ctx
	}
	return klog.NewContext(ctx, klog.LoggerWithName(klog.LoggerWithName(logger, "NominationExtensions"), ext.Name()))
}
