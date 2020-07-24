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

package nodeaffinity

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	pluginhelper "k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// NodeAffinity is a plugin that checks if a pod node selector matches the node label.
type NodeAffinity struct {
	handle framework.FrameworkHandle
}

var _ framework.FilterPlugin = &NodeAffinity{}
var _ framework.ScorePlugin = &NodeAffinity{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "NodeAffinity"

	// ErrReason for node affinity/selector not matching.
	ErrReason = "node(s) didn't match node selector"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeAffinity) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *NodeAffinity) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	if !pluginhelper.PodMatchesNodeSelectorAndAffinityTerms(pod, node) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReason)
	}
	return nil
}

// Score invoked at the Score extension point.
func (pl *NodeAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}

	affinity := pod.Spec.Affinity

	var count int64
	// A nil element of PreferredDuringSchedulingIgnoredDuringExecution matches no objects.
	// An element of PreferredDuringSchedulingIgnoredDuringExecution that refers to an
	// empty PreferredSchedulingTerm matches all objects.
	if affinity != nil && affinity.NodeAffinity != nil && affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		// Match PreferredDuringSchedulingIgnoredDuringExecution term by term.
		for i := range affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution {
			preferredSchedulingTerm := &affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution[i]
			if preferredSchedulingTerm.Weight == 0 {
				continue
			}

			// TODO: Avoid computing it for all nodes if this becomes a performance problem.
			nodeSelector, err := v1helper.NodeSelectorRequirementsAsSelector(preferredSchedulingTerm.Preference.MatchExpressions)
			if err != nil {
				return 0, framework.NewStatus(framework.Error, err.Error())
			}

			if nodeSelector.Matches(labels.Set(node.Labels)) {
				count += int64(preferredSchedulingTerm.Weight)
			}
		}
	}

	return count, nil
}

// NormalizeScore invoked after scoring all nodes.
func (pl *NodeAffinity) NormalizeScore(ctx context.Context, state *framework.CycleState, pod *v1.Pod, scores framework.NodeScoreList) *framework.Status {
	return pluginhelper.DefaultNormalizeScore(framework.MaxNodeScore, false, scores)
}

// ScoreExtensions of the Score plugin.
func (pl *NodeAffinity) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, h framework.FrameworkHandle) (framework.Plugin, error) {
	return &NodeAffinity{handle: h}, nil
}
