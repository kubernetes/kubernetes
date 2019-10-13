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

package tainttoleration

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// TaintToleration is a plugin that checks if a pod tolerates a node's taints.
type TaintToleration struct {
	handle framework.FrameworkHandle
}

var _ = framework.FilterPlugin(&TaintToleration{})
var _ = framework.ScorePlugin(&TaintToleration{})

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = "TaintToleration"

// Name returns name of the plugin. It is used in logs, etc.
func (pl *TaintToleration) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *TaintToleration) Filter(state *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	// Note that PodToleratesNodeTaints doesn't use predicate metadata, hence passing nil here.
	_, reasons, err := predicates.PodToleratesNodeTaints(pod, nil, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the Score extension point.
func (pl *TaintToleration) Score(state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, exist := pl.handle.NodeInfoSnapshot().NodeInfoMap[nodeName]
	if !exist {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("node %q does not exist in NodeInfoSnapshot", nodeName))
	}
	meta := migration.PriorityMetadata(state)
	s, err := priorities.ComputeTaintTolerationPriorityMap(pod, meta, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// NormalizeScore invoked after scoring all nodes.
func (pl *TaintToleration) NormalizeScore(_ *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	// Note that ComputeTaintTolerationPriorityReduce doesn't use priority metadata, hence passing nil here.
	err := priorities.ComputeTaintTolerationPriorityReduce(pod, nil, pl.handle.NodeInfoSnapshot().NodeInfoMap, scores)
	return migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *TaintToleration) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ *runtime.Unknown, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &TaintToleration{handle: h}, nil
}
