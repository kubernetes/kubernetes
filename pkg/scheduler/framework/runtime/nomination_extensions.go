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

	v1 "k8s.io/api/core/v1"
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
}

// RunNominationExtensionsAddNominatedPod calls AddNominatedPod of all plugins.
func (f *frameworkImpl) RunNominationExtensionsAddNominatedPod(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string, victims []*v1.Pod) {
	for _, ext := range f.nominationExtensions {
		ext.AddNominatedPod(f.nominationExtensionsContext(ctx, ext), state, pod, nodeName, victims)
	}
}

// RunNominationExtensionsRemoveNominatedPod calls RemoveNominatedPod of all plugins.
func (f *frameworkImpl) RunNominationExtensionsRemoveNominatedPod(ctx context.Context, pod *v1.Pod, nodeName string, preemptionFailed bool) {
	for _, ext := range f.nominationExtensions {
		ext.RemoveNominatedPod(f.nominationExtensionsContext(ctx, ext), pod, nodeName, preemptionFailed)
	}
}

// RunNominationExtensionsPodEligibleToPreempt returns the result of the first
// plugin which reports that the pod is not eligible to preempt.
func (f *frameworkImpl) RunNominationExtensionsPodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	for _, ext := range f.nominationExtensions {
		if eligible, reason := ext.PodEligibleToPreempt(f.nominationExtensionsContext(ctx, ext), pod, nodeName); !eligible {
			return false, reason
		}
	}
	return true, ""
}

func (f *frameworkImpl) nominationExtensionsContext(ctx context.Context, ext framework.NominationExtensions) context.Context {
	logger := klog.FromContext(ctx)
	if !logger.V(4).Enabled() {
		return ctx
	}
	return klog.NewContext(ctx, klog.LoggerWithName(klog.LoggerWithName(logger, "NominationExtensions"), ext.Name()))
}
