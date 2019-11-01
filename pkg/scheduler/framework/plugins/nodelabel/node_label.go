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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Name of this plugin.
const Name = "NodeLabel"

// Args holds the args that are used to configure the plugin.
type Args struct {
	// The list of labels that identify node "groups"
	// All of the labels should be either present (or absent) for the node to be considered a fit for hosting the pod
	Labels []string `json:"labels,omitempty"`
	// The boolean flag that indicates whether the labels should be present or absent from the node
	Presence bool `json:"presence,omitempty"`

	// The parameters that are used to configure the priority function
	// The label that identify node "groups"
	PreferenceLabel string `json:"preferenceLabel,omitempty"`
	// The boolean flag that indicates whether the label should be present or absent from the node
	PreferenceLabelPresence bool `json:"preferenceLabelPresence,omitempty"`
}

// New initializes a new plugin and returns it.
func New(plArgs *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	args := &Args{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}
	// Note that the reduce function is always nil therefore it's ignored.
	prioritize, _ := priorities.NewNodeLabelPriority(args.PreferenceLabel, args.PreferenceLabelPresence)
	return &NodeLabel{
		handle:     handle,
		predicate:  predicates.NewNodeLabelPredicate(args.Labels, args.Presence),
		prioritize: prioritize,
	}, nil
}

// NodeLabel checks whether a pod can fit based on the node labels which match a filter that it requests.
type NodeLabel struct {
	handle     framework.FrameworkHandle
	predicate  predicates.FitPredicate
	prioritize priorities.PriorityMapFunction
}

var _ framework.FilterPlugin = &NodeLabel{}
var _ framework.ScorePlugin = &NodeLabel{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeLabel) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *NodeLabel) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	// Note that NodeLabelPredicate doesn't use predicate metadata, hence passing nil here.
	_, reasons, err := pl.predicate(pod, nil, nodeInfo)
	return migration.PredicateResultToFrameworkStatus(reasons, err)
}

// Score invoked at the score extension point.
func (pl *NodeLabel) Score(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeName string) (int64, *framework.Status) {
	nodeInfo, err := pl.handle.SnapshotSharedLister().NodeInfos().Get(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("getting node %q from Snapshot: %v", nodeName, err))
	}
	// Note that node label priority function doesn't use metadata, hence passing nil here.
	s, err := pl.prioritize(pod, nil, nodeInfo)
	return s.Score, migration.ErrorToFrameworkStatus(err)
}

// ScoreExtensions of the Score plugin.
func (pl *NodeLabel) ScoreExtensions() framework.ScoreExtensions {
	return nil
}
