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

package interpodaffinity

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

// InterPodAffinity is a plugin that checks inter pod affinity
type InterPodAffinity struct {
	snapshotSharedLister schedulerlisters.SharedLister
	podAffinityChecker   *predicates.PodAffinityChecker
}

var _ framework.PreFilterPlugin = &InterPodAffinity{}
var _ framework.FilterPlugin = &InterPodAffinity{}
var _ framework.ScorePlugin = &InterPodAffinity{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "InterPodAffinity"

	// preFilterStateKey is the key in CycleState to InterPodAffinity pre-computed data.
	// Using the name of the plugin will likely help us avoid collisions with other plugins.
	preFilterStateKey = "PreFilter" + Name
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	meta *predicates.PodAffinityMetadata
}

// Clone the prefilter state.
func (s *preFilterState) Clone() framework.StateData {
	copy := &preFilterState{
		meta: s.meta.Clone(),
	}
	return copy
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *InterPodAffinity) Name() string {
	return Name
}

// PreFilter invoked at the prefilter extension point.
func (pl *InterPodAffinity) PreFilter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod) *framework.Status {
	var meta *predicates.PodAffinityMetadata
	var allNodes []*nodeinfo.NodeInfo
	var havePodsWithAffinityNodes []*nodeinfo.NodeInfo
	var err error
	if allNodes, err = pl.snapshotSharedLister.NodeInfos().List(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos: %v", err))
	}
	if havePodsWithAffinityNodes, err = pl.snapshotSharedLister.NodeInfos().HavePodsWithAffinityList(); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("failed to list NodeInfos with pods with affinity: %v", err))
	}
	if meta, err = predicates.GetPodAffinityMetadata(pod, allNodes, havePodsWithAffinityNodes); err != nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("Error calculating podAffinityMetadata: %v", err))
	}

	s := &preFilterState{
		meta: meta,
	}
	cycleState.Write(preFilterStateKey, s)
	return nil
}

// PreFilterExtensions returns prefilter extensions, pod add and remove.
func (pl *InterPodAffinity) PreFilterExtensions() framework.PreFilterExtensions {
	return pl
}

// AddPod from pre-computed data in cycleState.
func (pl *InterPodAffinity) AddPod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToAdd *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodAffinityMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	meta.UpdateWithPod(podToAdd, podToSchedule, nodeInfo.Node(), 1)
	return nil
}

// RemovePod from pre-computed data in cycleState.
func (pl *InterPodAffinity) RemovePod(ctx context.Context, cycleState *framework.CycleState, podToSchedule *v1.Pod, podToRemove *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodAffinityMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	meta.UpdateWithPod(podToRemove, podToSchedule, nodeInfo.Node(), -1)
	return nil
}

func getPodAffinityMetadata(cycleState *framework.CycleState) (*predicates.PodAffinityMetadata, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// The metadata wasn't pre-computed in prefilter. We ignore the error for now since
		// Filter is able to handle that by computing it again.
		klog.V(5).Infof("Error reading %q from cycleState: %v", preFilterStateKey, err)
		return nil, nil
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("%+v  convert to interpodaffinity.state error", c)
	}
	return s.meta, nil
}

// Filter invoked at the filter extension point.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, err := getPodAffinityMetadata(cycleState)
	if err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}
	_, reasons, err := pl.podAffinityChecker.InterPodAffinityMatches(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *InterPodAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.snapshotSharedLister.NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	meta := migration.PriorityMetadata(state)
	s, err := priorities.CalculateInterPodAffinityPriorityMap(pod, meta, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// NormalizeScore invoked after scoring all nodes.
func (pl *InterPodAffinity) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	meta := migration.PriorityMetadata(state)
	err := priorities.CalculateInterPodAffinityPriorityReduce(pod, meta, pl.snapshotSharedLister, scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *InterPodAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	if h.SnapshotSharedLister() == nil {
		return nil, fmt.Errorf("SnapshotSharedlister is nil")
	}
	return &InterPodAffinity{
		snapshotSharedLister: h.SnapshotSharedLister(),
		podAffinityChecker:   predicates.NewPodAffinityChecker(h.SnapshotSharedLister()),
	}, nil
}
