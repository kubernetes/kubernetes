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
var _ schedulerframework.UnreserveExtensions = &FakeNominationExtensionsPlugin{}
var _ schedulerframework.VictimExtensions = &FakeNominationExtensionsPlugin{}
var _ fwk.PreFilterPlugin = &FakeNominationExtensionsPlugin{}
var _ fwk.ReservePlugin = &FakeNominationExtensionsPlugin{}

// FakeNominationExtensionsPlugin implements framework.NominationExtensions,
// framework.UnreserveExtensions, and framework.VictimExtensions and records
// calls for tests. It also implements PreFilterPlugin and ReservePlugin so it
// can be enabled in a scheduler profile.
type FakeNominationExtensionsPlugin struct {
	// PluginName overrides FakeNominationExtensionsName.
	PluginName string
	// NotEligibleReason, if set, makes PodEligibleToPreempt return false with this reason.
	NotEligibleReason string
	// ReleaseEvents get returned by RemoveNomination and UnreserveEvents.
	ReleaseEvents []schedulerframework.ResourceEvent

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

// Reserve succeeds for all pods.
func (pl *FakeNominationExtensionsPlugin) Reserve(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) *fwk.Status {
	return nil
}

// Unreserve is a no-op; UnreserveEvents records unreserve rollback.
func (pl *FakeNominationExtensionsPlugin) Unreserve(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) {
}

// AddNomination records the call as
// "AddNomination <pod> <node> hasState=<bool>".
func (pl *FakeNominationExtensionsPlugin) AddNomination(_ context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) schedulerframework.NominationChangeResult {
	pl.record(fmt.Sprintf("AddNomination %s %s hasState=%v", klog.KObj(pod), nodeName, state != nil))
	return schedulerframework.NominationChangeResult{}
}

// RemoveNomination records the call as
// "RemoveNomination <pod> <node>" and returns ReleaseEvents.
func (pl *FakeNominationExtensionsPlugin) RemoveNomination(_ context.Context, pod *v1.Pod, nodeName string) schedulerframework.NominationChangeResult {
	pl.record(fmt.Sprintf("RemoveNomination %s %s", klog.KObj(pod), nodeName))
	return schedulerframework.NominationChangeResult{Events: pl.ReleaseEvents}
}

// UnreserveEvents records the call as
// "UnreserveEvents <pod> <node>" and returns ReleaseEvents.
func (pl *FakeNominationExtensionsPlugin) UnreserveEvents(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeName string) []schedulerframework.ResourceEvent {
	pl.record(fmt.Sprintf("UnreserveEvents %s %s", klog.KObj(pod), nodeName))
	return pl.ReleaseEvents
}

// AssumeVictims records the call as "AssumeVictims <node> [<victim>,...]".
func (pl *FakeNominationExtensionsPlugin) AssumeVictims(_ context.Context, nodeName string, victims []*v1.Pod) {
	names := make([]string, len(victims))
	for i, v := range victims {
		names[i] = klog.KObj(v).String()
	}
	pl.record(fmt.Sprintf("AssumeVictims %s [%s]", nodeName, strings.Join(names, ",")))
}

// ForgetVictim records the call as "ForgetVictim <node> <victim>".
func (pl *FakeNominationExtensionsPlugin) ForgetVictim(_ context.Context, nodeName string, victim *v1.Pod) {
	pl.record(fmt.Sprintf("ForgetVictim %s %s", nodeName, klog.KObj(victim)))
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
