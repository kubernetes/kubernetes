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

package nodelabel

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// Name of this plugin.
const Name = "NodeLabel"

const (
	// ErrReasonPresenceViolated is used for CheckNodeLabelPresence predicate error.
	ErrReasonPresenceViolated = "node(s) didn't have the requested labels"
)

// New initializes a new plugin and returns it.
func New(plArgs runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	args, err := getArgs(plArgs)
	if err != nil {
		return nil, err
	}

	if err := validation.ValidateNodeLabelArgs(args); err != nil {
		return nil, err
	}

	return &NodeLabel{
		handle: handle,
		args:   args,
	}, nil
}

func getArgs(obj runtime.Object) (config.NodeLabelArgs, error) {
	ptr, ok := obj.(*config.NodeLabelArgs)
	if !ok {
		return config.NodeLabelArgs{}, fmt.Errorf("want args to be of type NodeLabelArgs, got %T", obj)
	}
	return *ptr, nil
}

// NodeLabel checks whether a pod can fit based on the node labels which match a filter that it requests.
type NodeLabel struct {
	handle framework.Handle
	args   config.NodeLabelArgs
}

var _ framework.FilterPlugin = &NodeLabel{}
var _ framework.ScorePlugin = &NodeLabel{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeLabel) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
// It checks whether all of the specified labels exists on a node or not, regardless of their value
//
// Consider the cases where the nodes are placed in regions/zones/racks and these are identified by labels
// In some cases, it is required that only nodes that are part of ANY of the defined regions/zones/racks be selected
//
// Alternately, eliminating nodes that have a certain label, regardless of value, is also useful
// A node may have a label with "retiring" as key and the date as the value
// and it may be desirable to avoid scheduling new pods on this node.
func (pl *NodeLabel) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}

	size := int64(len(pl.args.PresentLabels) + len(pl.args.AbsentLabels))
	if size == 0 {
		return nil
	}

	nodeLabels := labels.Set(node.Labels)
	check := func(labels []string, presence bool) bool {
		for _, label := range labels {
			exists := nodeLabels.Has(label)
			if (exists && !presence) || (!exists && presence) {
				return false
			}
		}
		return true
	}
	if check(pl.args.PresentLabels, true) && check(pl.args.AbsentLabels, false) {
		return nil
	}

	return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonPresenceViolated)
}

// Score invoked at the score extension point.
func (pl *NodeLabel) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.AsStatus(fmt.Errorf("getting node %q from Snapshot: %w", nodeName, err))
	}

	node := nodeInfo.Node()
	if node == nil {
		return 0, framework.NewStatus(framework.Error, "node not found")
	}

	size := int64(len(pl.args.PresentLabelsPreference) + len(pl.args.AbsentLabelsPreference))
	if size == 0 {
		return 0, nil
	}

	score := int64(0)
	for _, label := range pl.args.PresentLabelsPreference {
		if labels.Set(node.Labels).Has(label) {
			score += framework.MaxNodeScore
		}
	}
	for _, label := range pl.args.AbsentLabelsPreference {
		if !labels.Set(node.Labels).Has(label) {
			score += framework.MaxNodeScore
		}
	}

	// Take average score for each label to ensure the score doesn't exceed MaxNodeScore.
	score /= size

	return score, nil
}

// ScoreExtensions of the Score plugin.
func (pl *NodeLabel) ScoreExtensions() framework.ScoreExtensions {
	return nil
}
