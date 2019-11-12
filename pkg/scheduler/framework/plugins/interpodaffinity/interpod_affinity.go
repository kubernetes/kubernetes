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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// InterPodAffinity is a plugin that checks inter pod affinity
type InterPodAffinity struct {
	handle    framework.FrameworkHandle
	predicate predicates.FitPredicate
}

var _ framework.FilterPlugin = &InterPodAffinity{}
var _ framework.ScorePlugin = &InterPodAffinity{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "InterPodAffinity"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *InterPodAffinity) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *InterPodAffinity) Filter(ctx context.Context, cycleState *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	meta, ok := migration.PredicateMetadata(cycleState).(predicates.Metadata)
	if !ok {
		return migration.ErrorToFrameworkStatus(fmt.Errorf("%+v convert to predicates.Metadata error", cycleState))
	}
	_, reasons, err := pl.predicate(pod, meta, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the Score extension point.
// The "score" returned in this function is the matching number of pods on the `nodeName`,
// it is normalized later.
func (pl *InterPodAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
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
	err := priorities.CalculateInterPodAffinityPriorityReduce(pod, meta, pl.handle.SnapshotSharedLister(), scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *InterPodAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &InterPodAffinity{
		handle:    h,
		predicate: predicates.NewPodAffinityPredicate(h.SnapshotSharedLister().NodeInfos(), h.SnapshotSharedLister().Pods()),
	}, nil
}
