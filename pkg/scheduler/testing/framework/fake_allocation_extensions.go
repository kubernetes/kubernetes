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
	"context"
	"fmt"
	"slices"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

// FakeAllocationExtensionsName is the default name of FakeAllocationExtensionsPlugin.
const FakeAllocationExtensionsName = "FakeAllocationExtensions"

var _ schedulerframework.AllocationExtensions = &FakeAllocationExtensionsPlugin{}
var _ schedulerframework.VictimExtensions = &FakeAllocationExtensionsPlugin{}
var _ fwk.PreFilterPlugin = &FakeAllocationExtensionsPlugin{}

// FakeAllocationExtensionsPlugin implements framework.AllocationExtensions
// and framework.VictimExtensions and records the calls. It is also a PreFilter plugin which skips all pods,
// so that it can be enabled in a profile.
type FakeAllocationExtensionsPlugin struct {
	// PluginName overrides FakeAllocationExtensionsName.
	PluginName string
	// NotEligibleReason, if set, makes PodEligibleToPreempt return false with this reason.
	NotEligibleReason string
	// ReleaseEvents get returned by RemoveAllocation.
	ReleaseEvents []schedulerframework.AllocationEvent

	mutex sync.Mutex
	calls []string
}

// NewFakeAllocationExtensionsPlugin returns a factory which always returns pl.
func NewFakeAllocationExtensionsPlugin(pl *FakeAllocationExtensionsPlugin) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return pl, nil
	}
}

// Name returns name of the plugin.
func (pl *FakeAllocationExtensionsPlugin) Name() string {
	if pl.PluginName != "" {
		return pl.PluginName
	}
	return FakeAllocationExtensionsName
}

// PreFilter skips the plugin for all pods.
func (pl *FakeAllocationExtensionsPlugin) PreFilter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, fwk.NewStatus(fwk.Skip)
}

// PreFilterExtensions returns nil.
func (pl *FakeAllocationExtensionsPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

// PrepareAllocation records the call as
// "PrepareAllocation <pod> <node> assigned=<pod.spec.nodeName>".
func (pl *FakeAllocationExtensionsPlugin) PrepareAllocation(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) {
	pl.record(fmt.Sprintf("PrepareAllocation %s %s assigned=%s", klog.KObj(pod), nodeName, pod.Spec.NodeName))
}

// AddAllocation records the call as
// "AddAllocation <pod> <node> assigned=<pod.spec.nodeName>".
func (pl *FakeAllocationExtensionsPlugin) AddAllocation(_ context.Context, pod *v1.Pod, nodeName string) schedulerframework.AllocationChangeResult {
	pl.record(fmt.Sprintf("AddAllocation %s %s assigned=%s", klog.KObj(pod), nodeName, pod.Spec.NodeName))
	return schedulerframework.AllocationChangeResult{}
}

// RemoveAllocation records the call as
// "RemoveAllocation <pod> <node> assigned=<pod.spec.nodeName>" and returns
// ReleaseEvents.
func (pl *FakeAllocationExtensionsPlugin) RemoveAllocation(_ context.Context, pod *v1.Pod, nodeName string) schedulerframework.AllocationChangeResult {
	pl.record(fmt.Sprintf("RemoveAllocation %s %s assigned=%s", klog.KObj(pod), nodeName, pod.Spec.NodeName))
	return schedulerframework.AllocationChangeResult{Events: pl.ReleaseEvents}
}

// AssumeVictim records the call as "AssumeVictim <victim>".
func (pl *FakeAllocationExtensionsPlugin) AssumeVictim(_ context.Context, victim *v1.Pod) {
	pl.record(fmt.Sprintf("AssumeVictim %s", klog.KObj(victim)))
}

// PodEligibleToPreempt records the call as "PodEligibleToPreempt <pod> <node>".
func (pl *FakeAllocationExtensionsPlugin) PodEligibleToPreempt(_ context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	pl.record(fmt.Sprintf("PodEligibleToPreempt %s %s", klog.KObj(pod), nodeName))
	return pl.NotEligibleReason == "", pl.NotEligibleReason
}

// Calls returns the recorded calls in the order in which they happened.
func (pl *FakeAllocationExtensionsPlugin) Calls() []string {
	pl.mutex.Lock()
	defer pl.mutex.Unlock()
	return slices.Clone(pl.calls)
}

func (pl *FakeAllocationExtensionsPlugin) record(call string) {
	pl.mutex.Lock()
	defer pl.mutex.Unlock()
	pl.calls = append(pl.calls, call)
}
