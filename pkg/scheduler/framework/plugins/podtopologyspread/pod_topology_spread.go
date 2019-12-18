/*
Copyright 2019 The Kubernetes Authors.

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

package podtopologyspread

import (
	"context"
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// PodTopologySpread is a plugin that ensures pod's topologySpreadConstraints is satisfied.
type PodTopologySpread struct {
	snapshotSharedLister schedulerlisters.SharedLister
}

var _ framework.PreFilterPlugin = &PodTopologySpread{}
var _ framework.FilterPlugin = &PodTopologySpread{}
var _ framework.ScorePlugin = &PodTopologySpread{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "PodTopologySpread"

	// preFilterStateKey is the key in CycleState to PodTopologySpread pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *PodTopologySpread) Name() string {
	return Name
}

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	// `nil` represents the meta is not set at all (in PreFilter phase)
	// An empty `PodTopologySpreadMetadata` object denotes it's a legit meta and is set in PreFilter phase.
	meta *predicates.PodTopologySpreadMetadata
}

// Clone makes a copy of the given state.
func (s *preFilterState) Clone() framework.StateData {
	copy := &preFilterState{
		meta: s.meta.Clone(),
	}
	return copy
}

// PreFilter invoked at the prefilter extension point.
func (pl *PodTopologySpread) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	var meta *predicates.PodTopologySpreadMetadata
	var allNodes []*nodeinfo.NodeInfo
	var err error

	if allNodes, err = pl.snapshotSharedLister.NodeInfos().List(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos: %v", err))
	}

	if meta, err = predicates.GetPodTopologySpreadMetadata(pod, allNodes); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("Error calculating podTopologySpreadMetadata: %v", err))
	}

	s := &preFilterState{
		meta: meta,
	}
	cycleState.Write(preFilterStateKey, s)

	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *PodTopologySpread) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *PodTopologySpread) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodTopologySpreadMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	meta.AddPod(podToAdd, podToSchedule, nodeInfo.Node())
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *PodTopologySpread) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodTopologySpreadMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	meta.RemovePod(podToRemove, podToSchedule, nodeInfo.Node())
	return nil
}

func getPodTopologySpreadMetadata(cycleState *framework.CycleState) (*predicates.PodTopologySpreadMetadata, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// The metadata wasn't pre-computed in prefilter. We ignore the error for now since
		// we are able to handle that by computing it again (e.g. in Filter()).
		klog.V(5).Infof("Error reading %q from cycleState: %v", preFilterStateKey, err)
		return nil, nil
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v convert to podtopologyspread.state error", c)
	}
	return s.meta, nil
}

// Filter invoked at the filter extension point.
func (pl *PodTopologySpread) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodTopologySpreadMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	_, reasons, err := predicates.PodTopologySpreadPredicate(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *PodTopologySpread) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.snapshotSharedLister.NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	meta := migration.PriorityMetadata(state)
	s, err := priorities.CalculateEvenPodsSpreadPriorityMap(pod, meta, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// NormalizeScore invoked after scoring all nodes.
func (pl *PodTopologySpread) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	meta := migration.PriorityMetadata(state)
	err := priorities.CalculateEvenPodsSpreadPriorityReduce(pod, meta, pl.snapshotSharedLister, scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *PodTopologySpread) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	return &PodTopologySpread{snapshotSharedLister: h.SnapshotSharedLister()}, nil
}
