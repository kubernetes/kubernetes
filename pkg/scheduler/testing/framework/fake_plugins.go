/*
Copyright 2020 The Kubernetes Authors.

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
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
)

// ErrReasonFake is a fake error message denotes the filter function errored.
const ErrReasonFake = "Nodes failed the fake plugin"

// FalseFilterPlugin is a filter plugin which always return Unschedulable when Filter function is called.
type FalseFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *FalseFilterPlugin) Name() string {
	return "FalseFilter"
}

// Filter invoked at the filter extension point.
func (pl *FalseFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return fwk.NewStatus(fwk.Unschedulable, ErrReasonFake)
}

// NewFalseFilterPlugin initializes a FalseFilterPlugin and returns it.
func NewFalseFilterPlugin(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &FalseFilterPlugin{}, nil
}

// TrueFilterPlugin is a filter plugin which always return Success when Filter function is called.
type TrueFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *TrueFilterPlugin) Name() string {
	return "TrueFilter"
}

// Filter invoked at the filter extension point.
func (pl *TrueFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	return nil
}

// NewTrueFilterPlugin initializes a TrueFilterPlugin and returns it.
func NewTrueFilterPlugin(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &TrueFilterPlugin{}, nil
}

type FakePreFilterAndFilterPlugin struct {
	*FakePreFilterPlugin
	*FakeFilterPlugin
}

// Name returns name of the plugin.
func (pl FakePreFilterAndFilterPlugin) Name() string {
	return "FakePreFilterAndFilterPlugin"
}

// FakeFilterPlugin is a test filter plugin to record how many times its Filter() function have
// been called, and it returns different 'Code' depending on its internal 'failedNodeReturnCodeMap'.
type FakeFilterPlugin struct {
	NumFilterCalled         int32
	FailedNodeReturnCodeMap map[string]fwk.Code
}

// Name returns name of the plugin.
func (pl *FakeFilterPlugin) Name() string {
	return "FakeFilter"
}

// Filter invoked at the filter extension point.
func (pl *FakeFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	atomic.AddInt32(&pl.NumFilterCalled, 1)

	if returnCode, ok := pl.FailedNodeReturnCodeMap[nodeInfo.Node().Name]; ok {
		return fwk.NewStatus(returnCode, fmt.Sprintf("injecting failure for pod %v", pod.Name))
	}

	return nil
}

// NewFakeFilterPlugin initializes a fakeFilterPlugin and returns it.
func NewFakeFilterPlugin(failedNodeReturnCodeMap map[string]fwk.Code) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakeFilterPlugin{
			FailedNodeReturnCodeMap: failedNodeReturnCodeMap,
		}, nil
	}
}

// MatchFilterPlugin is a filter plugin which return Success when the evaluated pod and node
// have the same name; otherwise return Unschedulable.
type MatchFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *MatchFilterPlugin) Name() string {
	return "MatchFilter"
}

// Filter invoked at the filter extension point.
func (pl *MatchFilterPlugin) Filter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	node := nodeInfo.Node()
	if node == nil {
		return fwk.NewStatus(fwk.Error, "node not found")
	}
	if pod.Name == node.Name {
		return nil
	}
	return fwk.NewStatus(fwk.Unschedulable, ErrReasonFake)
}

// NewMatchFilterPlugin initializes a MatchFilterPlugin and returns it.
func NewMatchFilterPlugin(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &MatchFilterPlugin{}, nil
}

// FakePreFilterPlugin is a test filter plugin.
type FakePreFilterPlugin struct {
	Result *framework.PreFilterResult
	Status *fwk.Status
	name   string
}

// Name returns name of the plugin.
func (pl *FakePreFilterPlugin) Name() string {
	return pl.name
}

// PreFilter invoked at the PreFilter extension point.
func (pl *FakePreFilterPlugin) PreFilter(_ context.Context, _ fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	return pl.Result, pl.Status
}

// PreFilterExtensions no extensions implemented by this plugin.
func (pl *FakePreFilterPlugin) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

// NewFakePreFilterPlugin initializes a fakePreFilterPlugin and returns it.
func NewFakePreFilterPlugin(name string, result *framework.PreFilterResult, status *fwk.Status) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakePreFilterPlugin{
			Result: result,
			Status: status,
			name:   name,
		}, nil
	}
}

// FakeReservePlugin is a test reserve plugin.
type FakeReservePlugin struct {
	Status *fwk.Status
}

// Name returns name of the plugin.
func (pl *FakeReservePlugin) Name() string {
	return "FakeReserve"
}

// Reserve invoked at the Reserve extension point.
func (pl *FakeReservePlugin) Reserve(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) *fwk.Status {
	return pl.Status
}

// Unreserve invoked at the Unreserve extension point.
func (pl *FakeReservePlugin) Unreserve(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) {
}

// NewFakeReservePlugin initializes a fakeReservePlugin and returns it.
func NewFakeReservePlugin(status *fwk.Status) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakeReservePlugin{
			Status: status,
		}, nil
	}
}

// FakePreBindPlugin is a test prebind plugin.
type FakePreBindPlugin struct {
	PreBindPreFlightStatus *fwk.Status
	PreBindStatus          *fwk.Status
}

// Name returns name of the plugin.
func (pl *FakePreBindPlugin) Name() string {
	return "FakePreBind"
}

// PreBindPreFlight invoked at the PreBind extension point.
func (pl *FakePreBindPlugin) PreBindPreFlight(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) *fwk.Status {
	return pl.PreBindPreFlightStatus
}

// PreBind invoked at the PreBind extension point.
func (pl *FakePreBindPlugin) PreBind(_ context.Context, _ fwk.CycleState, _ *v1.Pod, _ string) *fwk.Status {
	return pl.PreBindStatus
}

// NewFakePreBindPlugin initializes a fakePreBindPlugin and returns it.
func NewFakePreBindPlugin(preBindPreFlightStatus, preBindStatus *fwk.Status) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakePreBindPlugin{
			PreBindPreFlightStatus: preBindPreFlightStatus,
			PreBindStatus:          preBindStatus,
		}, nil
	}
}

// FakePermitPlugin is a test permit plugin.
type FakePermitPlugin struct {
	Handle  framework.Handle
	Status  *fwk.Status
	Timeout time.Duration
}

// Name returns name of the plugin.
func (pl *FakePermitPlugin) Name() string {
	return "FakePermit"
}

// Permit invoked at the Permit extension point.
func (pl *FakePermitPlugin) Permit(_ context.Context, _ fwk.CycleState, p *v1.Pod, _ string) (*fwk.Status, time.Duration) {
	return pl.Status, pl.Timeout
}

// NewFakePermitPlugin initializes a fakePermitPlugin and returns it.
func NewFakePermitPlugin(status *fwk.Status, timeout time.Duration) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, h framework.Handle) (framework.Plugin, error) {
		return &FakePermitPlugin{
			Status:  status,
			Timeout: timeout,
			Handle:  h,
		}, nil
	}
}

type FakePreScoreAndScorePlugin struct {
	name           string
	score          int64
	preScoreStatus *fwk.Status
	scoreStatus    *fwk.Status
}

// Name returns name of the plugin.
func (pl *FakePreScoreAndScorePlugin) Name() string {
	return pl.name
}

func (pl *FakePreScoreAndScorePlugin) Score(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeInfo fwk.NodeInfo) (int64, *fwk.Status) {
	return pl.score, pl.scoreStatus
}

func (pl *FakePreScoreAndScorePlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (pl *FakePreScoreAndScorePlugin) PreScore(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) *fwk.Status {
	return pl.preScoreStatus
}

func NewFakePreScoreAndScorePlugin(name string, score int64, preScoreStatus, scoreStatus *fwk.Status) frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakePreScoreAndScorePlugin{
			name:           name,
			score:          score,
			preScoreStatus: preScoreStatus,
			scoreStatus:    scoreStatus,
		}, nil
	}
}

// NewEqualPrioritizerPlugin returns a factory function to build equalPrioritizerPlugin.
func NewEqualPrioritizerPlugin() frameworkruntime.PluginFactory {
	return func(_ context.Context, _ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &FakePreScoreAndScorePlugin{
			name:  "EqualPrioritizerPlugin",
			score: 1,
		}, nil
	}
}
