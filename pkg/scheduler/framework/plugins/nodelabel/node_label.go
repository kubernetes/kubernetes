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
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

// Name of this plugin.
const Name = "NodeLabel"

// Args holds the args that are used to configure the plugin.
type Args struct {
	// PresentLabels should be present for the node to be considered a fit for hosting the pod
	PresentLabels []string `json:"presentLabels,omitempty"`
	// AbsentLabels should be absent for the node to be considered a fit for hosting the pod
	AbsentLabels []string `json:"absentLabels,omitempty"`
}

// validateArgs validates that PresentLabels and AbsentLabels do not conflict.
func validateArgs(args *Args) error {
	presentLabels := make(map[string]struct{}, len(args.PresentLabels))
	for _, l := range args.PresentLabels {
		presentLabels[l] = struct{}{}
	}
	for _, l := range args.AbsentLabels {
		if _, ok := presentLabels[l]; ok {
			return fmt.Errorf("detecting at least one label (e.g., %q) that exist in both the present and absent label list: %+v", l, args)
		}
	}
	return nil
}

// New initializes a new plugin and returns it.
func New(plArgs *runtime.Unknown, _ framework.FrameworkHandle) (framework.Plugin, error) {
	args := &Args{}
	if err := framework.DecodeInto(plArgs, args); err != nil {
		return nil, err
	}
	if err := validateArgs(args); err != nil {
		return nil, err
	}
	return &NodeLabel{
		predicate: predicates.NewNodeLabelPredicate(args.PresentLabels, args.AbsentLabels),
	}, nil
}

// NodeLabel checks whether a pod can fit based on the node labels which match a filter that it requests.
type NodeLabel struct {
	predicate predicates.FitPredicate
}

var _ framework.FilterPlugin = &NodeLabel{}

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
