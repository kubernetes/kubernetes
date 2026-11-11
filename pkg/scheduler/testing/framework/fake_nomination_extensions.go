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
	// ReleaseEvents get returned by RemoveNomination.
	ReleaseEvents []schedulerframework.NominationEvent

	mu    sync.Mutex
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

// PrepareNomination records the call as
// "PrepareNomination <pod> <node> assigned=<pod.spec.nodeName>" and returns
// "prepared <pod> <node>".
func (pl *FakeNominationExtensionsPlugin) PrepareNomination(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) any {
	pl.record(fmt.Sprintf("PrepareNomination %s %s assigned=%s", klog.KObj(pod), nodeName, pod.Spec.NodeName))
	return fmt.Sprintf("prepared %s %s", klog.KObj(pod), nodeName)
}

// AddNomination records the call as
// "AddNomination <pod> <node> assigned=<pod.spec.nodeName> prepared=<prepared>".
func (pl *FakeNominationExtensionsPlugin) AddNomination(_ context.Context, pod *v1.Pod, nodeName string, prepared any) schedulerframework.NominationChangeResult {
	pl.record(fmt.Sprintf("AddNomination %s %s assigned=%s prepared=%v", klog.KObj(pod), nodeName, pod.Spec.NodeName, prepared))
	return schedulerframework.NominationChangeResult{}
}

// RemoveNomination records the call as
// "RemoveNomination <pod> <node> assigned=<pod.spec.nodeName>" and returns
// ReleaseEvents.
func (pl *FakeNominationExtensionsPlugin) RemoveNomination(_ context.Context, pod *v1.Pod, nodeName string) schedulerframework.NominationChangeResult {
	pl.record(fmt.Sprintf("RemoveNomination %s %s assigned=%s", klog.KObj(pod), nodeName, pod.Spec.NodeName))
	return schedulerframework.NominationChangeResult{Events: pl.ReleaseEvents}
}

// Calls returns the recorded calls in the order in which they happened.
func (pl *FakeNominationExtensionsPlugin) Calls() []string {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	return slices.Clone(pl.calls)
}

func (pl *FakeNominationExtensionsPlugin) record(call string) {
	pl.mu.Lock()
	defer pl.mu.Unlock()
	pl.calls = append(pl.calls, call)
}
