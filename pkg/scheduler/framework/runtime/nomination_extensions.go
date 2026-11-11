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
	"maps"
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
	// Sorted for a deterministic call order.
	for _, name := range slices.Sorted(maps.Keys(f.pluginsMap)) {
		if ext, ok := f.pluginsMap[name].(framework.NominationExtensions); ok {
			f.nominationExtensions = append(f.nominationExtensions, ext)
		}
	}
	if len(f.nominationExtensions) > 0 {
		f.pendingNominations = &pendingNominations{byPod: make(map[types.UID]*pendingNomination)}
	}
	f.genericWorkloadEnabled = utilfeature.DefaultFeatureGate.Enabled(features.GenericWorkload)
}

// pendingNominations buffers per-plugin data returned by PrepareNomination
// until PodNominator commits the nomination and calls AddNomination.
type pendingNominations struct {
	mu    sync.Mutex
	byPod map[types.UID]*pendingNomination
}

type pendingNomination struct {
	nodeName string
	// pluginData holds the opaque per-plugin nomination data returned by
	// PrepareNomination, indexed in the same order as f.nominationExtensions.
	pluginData []any
}

// skipNominationExtensions reports whether NominationExtensions calls must be
// skipped for pod. Currently, pods belonging to a PodGroup (when
// GenericWorkload is enabled) are skipped because PodGroup scheduling cycles do
// not track nominations in memory yet.
func (f *frameworkImpl) skipNominationExtensions(pod *v1.Pod) bool {
	return f.genericWorkloadEnabled && pod.Spec.SchedulingGroup != nil
}

// RunNominationExtensionsPrepareNomination calls PrepareNomination of all
// plugins and buffers the returned nomination data for pod on nodeName until
// AddNomination consumes it or done is called.
func (f *frameworkImpl) RunNominationExtensionsPrepareNomination(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (done func()) {
	p := f.pendingNominations
	if p == nil || f.skipNominationExtensions(pod) {
		return func() {}
	}
	entry := &pendingNomination{nodeName: nodeName, pluginData: make([]any, len(f.nominationExtensions))}
	for i, ext := range f.nominationExtensions {
		entry.pluginData[i] = ext.PrepareNomination(f.nominationExtensionsContext(ctx, ext), state, pod, nodeName)
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.byPod[pod.UID] = entry
	return func() {
		p.mu.Lock()
		defer p.mu.Unlock()
		// A later call for the same pod may have replaced the entry.
		if p.byPod[pod.UID] == entry {
			delete(p.byPod, pod.UID)
		}
	}
}

// popPendingNomination returns and drops the per-plugin nomination data from
// PrepareNomination for the nomination of the pod to the node, or nil if there
// is none.
func (f *frameworkImpl) popPendingNomination(podUID types.UID, nodeName string) []any {
	p := f.pendingNominations
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	entry := p.byPod[podUID]
	if entry == nil || entry.nodeName != nodeName {
		return nil
	}
	delete(p.byPod, podUID)
	return entry.pluginData
}

// RunNominationExtensionsAddNomination calls AddNomination of all plugins,
// passing each plugin its pending nomination data for pod on nodeName (if
// any), and returns all emitted events.
func (f *frameworkImpl) RunNominationExtensionsAddNomination(ctx context.Context, pod *v1.Pod, nodeName string) []framework.NominationEvent {
	if len(f.nominationExtensions) == 0 || f.skipNominationExtensions(pod) {
		return nil
	}
	pluginData := f.popPendingNomination(pod.UID, nodeName)
	var events []framework.NominationEvent
	for i, ext := range f.nominationExtensions {
		var nominationData any
		if pluginData != nil {
			nominationData = pluginData[i]
		}
		events = append(events, ext.AddNomination(f.nominationExtensionsContext(ctx, ext), pod, nodeName, nominationData).Events...)
	}
	return events
}

// RunNominationExtensionsRemoveNomination calls RemoveNomination of all
// plugins and returns all emitted events.
func (f *frameworkImpl) RunNominationExtensionsRemoveNomination(ctx context.Context, pod *v1.Pod, nodeName string) []framework.NominationEvent {
	if len(f.nominationExtensions) == 0 || f.skipNominationExtensions(pod) {
		return nil
	}
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
