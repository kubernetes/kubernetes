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
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulerframework "k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

// FakeNominationExtensionsName is the default name of FakeNominationExtensionsPlugin.
const FakeNominationExtensionsName = "FakeNominationExtensions"

var _ schedulerframework.NominationExtensions = &FakeNominationExtensionsPlugin{}
var _ fwk.PreFilterPlugin = &FakeNominationExtensionsPlugin{}

// FakeNominationExtensionsPlugin implements framework.NominationExtensions
// and records the calls. It is also a PreFilter plugin which skips all pods,
// so that it can be enabled in a profile.
type FakeNominationExtensionsPlugin struct {
	// PluginName overrides FakeNominationExtensionsName.
	PluginName string
	// NotEligibleReason, if set, makes PodEligibleToPreempt return false with this reason.
	NotEligibleReason string

	mutex sync.Mutex
	calls []string
}

// NewFakeNominationExtensionsPlugin returns a factory which always returns pl.
func NewFakeNominationExtensionsPlugin(pl *FakeNominationExtensionsPlugin) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ fwk.Handle) (fwk.Plugin, error) {
		return pl, nil
	}
}

// Name returns name of the plugin.
func (pl *FakeNominationExtensionsPlugin) Name() string {
	if pl.PluginName != "" {
		return pl.PluginName
	}
	return FakeNominationExtensionsName
}

// PreFilter skips the plugin for all pods.
func (pl *FakeNominationExtensionsPlugin) PreFilter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	return nil, fwk.NewStatus(fwk.Skip)
}

// PreFilterExtensions returns nil.
func (pl *FakeNominationExtensionsPlugin) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

// AddNominatedPod records the call as "AddNominatedPod <pod> <node> <victims>".
func (pl *FakeNominationExtensionsPlugin) AddNominatedPod(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string, victims []*v1.Pod) {
	names := make([]string, 0, len(victims))
	for _, victim := range victims {
		names = append(names, klog.KObj(victim).String())
	}
	slices.Sort(names)
	pl.record(fmt.Sprintf("AddNominatedPod %s %s [%s]", klog.KObj(pod), nodeName, strings.Join(names, " ")))
}

// RemoveNominatedPod records the call as
// "RemoveNominatedPod <pod> <node> assigned=<pod.spec.nodeName> failed=<preemptionFailed>".
func (pl *FakeNominationExtensionsPlugin) RemoveNominatedPod(_ context.Context, pod *v1.Pod, nodeName string, preemptionFailed bool) {
	pl.record(fmt.Sprintf("RemoveNominatedPod %s %s assigned=%s failed=%t", klog.KObj(pod), nodeName, pod.Spec.NodeName, preemptionFailed))
}

// PodEligibleToPreempt records the call as "PodEligibleToPreempt <pod> <node>".
func (pl *FakeNominationExtensionsPlugin) PodEligibleToPreempt(_ context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	pl.record(fmt.Sprintf("PodEligibleToPreempt %s %s", klog.KObj(pod), nodeName))
	return pl.NotEligibleReason == "", pl.NotEligibleReason
}

// Calls returns the recorded calls in the order in which they happened.
func (pl *FakeNominationExtensionsPlugin) Calls() []string {
	pl.mutex.Lock()
	defer pl.mutex.Unlock()
	return slices.Clone(pl.calls)
}

func (pl *FakeNominationExtensionsPlugin) record(call string) {
	pl.mutex.Lock()
	defer pl.mutex.Unlock()
	pl.calls = append(pl.calls, call)
}
