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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	pluginhelper "k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
)

// NodeAffinity is a plugin that checks if a pod node selector matches the node label.
type NodeAffinity struct {
	handle              framework.Handle
	addedNodeSelector   *nodeaffinity.NodeSelector
	addedPrefSchedTerms *nodeaffinity.PreferredSchedulingTerms
}

var _ framework.FilterPlugin = &NodeAffinity{}
var _ framework.ScorePlugin = &NodeAffinity{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = "NodeAffinity"

	// ErrReasonPod is the reason for Pod's node affinity/selector not matching.
	ErrReasonPod = "node(s) didn't match Pod's node affinity"

	// errReasonEnforced is the reason for added node affinity not matching.
	errReasonEnforced = "node(s) didn't match scheduler-enforced node affinity"
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeAffinity) Name() string {
	return Name
}

// Filter checks if the Node matches the Pod .spec.affinity.nodeAffinity and
// the plugin's added affinity.
func (pl *NodeAffinity) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	if pl.addedNodeSelector != nil && !pl.addedNodeSelector.Match(node) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, errReasonEnforced)
	}
	if !pluginhelper.PodMatchesNodeSelectorAndAffinityTerms(pod, node) {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonPod)
	}
	return nil
}

// Score returns the sum of the weights of the terms that match the Node.
// Terms came from the Pod .spec.affinity.nodeAffinity and from the plugin's
// default affinity.
func (pl *NodeAffinity) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}

	affinity := pod.Spec.Affinity

	var count int64
	if pl.addedPrefSchedTerms != nil {
		count += pl.addedPrefSchedTerms.Score(node)
	}
	// A nil element of PreferredDuringSchedulingIgnoredDuringExecution matches no objects.
	// An element of PreferredDuringSchedulingIgnoredDuringExecution that refers to an
	// empty PreferredSchedulingTerm matches all objects.
	if affinity != nil && affinity.NodeAffinity != nil && affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution != nil {
		// TODO(#96164): Do this in PreScore to avoid computing it for all nodes.
		preferredNodeAffinity, err := nodeaffinity.NewPreferredSchedulingTerms(affinity.NodeAffinity.PreferredDuringSchedulingIgnoredDuringExecution)
		if err != nil {
			return 0, framework.AsStatus(err)
		}
		count += preferredNodeAffinity.Score(node)
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
func New(plArgs runtime.Object, h framework.Handle) (framework.Plugin, error) {
	args, err := getArgs(plArgs)
	if err != nil {
		return nil, err
	}
	pl := &NodeAffinity{
		handle: h,
	}
	if args.AddedAffinity != nil {
		if ns := args.AddedAffinity.RequiredDuringSchedulingIgnoredDuringExecution; ns != nil {
			pl.addedNodeSelector, err = nodeaffinity.NewNodeSelector(ns)
			if err != nil {
				return nil, fmt.Errorf("parsing addedAffinity.requiredDuringSchedulingIgnoredDuringExecution: %w", err)
			}
		}
		// TODO: parse requiredDuringSchedulingRequiredDuringExecution when it gets added to the API.
		if terms := args.AddedAffinity.PreferredDuringSchedulingIgnoredDuringExecution; len(terms) != 0 {
			pl.addedPrefSchedTerms, err = nodeaffinity.NewPreferredSchedulingTerms(terms)
			if err != nil {
				return nil, fmt.Errorf("parsing addedAffinity.preferredDuringSchedulingIgnoredDuringExecution: %w", err)
			}
		}
	}
	return pl, nil
}

func getArgs(obj runtime.Object) (config.NodeAffinityArgs, error) {
	ptr, ok := obj.(*config.NodeAffinityArgs)
	if !ok {
		return config.NodeAffinityArgs{}, fmt.Errorf("args are not of type NodeAffinityArgs, got %T", obj)
	}
	return *ptr, validation.ValidateNodeAffinityArgs(ptr)
}
