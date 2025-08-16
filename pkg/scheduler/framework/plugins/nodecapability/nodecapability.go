/*
Copyright 2025 The Kubernetes Authors.

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

package nodecapability

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	versionutil "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/version"
	"k8s.io/component-helpers/nodecapabilities"
	fwk "k8s.io/kube-scheduler/framework"
	nodecapabilitiesregistry "k8s.io/kubernetes/pkg/features/nodecapabilities"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.NodeCapability
	// preFilterStateKey is the key in CycleState used to store the pod's capability requirements.
	preFilterStateKey fwk.StateKey = "PreFilter" + Name
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	reqs *nodecapabilities.PodRequirements
}

// Clone implements StateData.
func (s *preFilterState) Clone() fwk.StateData {
	return s
}

// NodeCapability is a plugin that checks if a node has all the capabilities required by a pod.
type NodeCapability struct {
	helper *nodecapabilities.NodeCapabilityHelper
}

var _ framework.PreFilterPlugin = &NodeCapability{}
var _ framework.FilterPlugin = &NodeCapability{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeCapability) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	ver, err := versionutil.Parse(version.Get().String())
	if err != nil {
		return nil, fmt.Errorf("failed to parse version: %w", err)
	}
	helper, err := nodecapabilities.NewNodeCapabilityHelper(nodecapabilitiesregistry.NewRegistry(), ver)
	if err != nil {
		return nil, fmt.Errorf("failed to create node capability helper: %w", err)
	}
	return &NodeCapability{helper: helper}, nil
}

// PreFilter checks if the pod has any capability requirements.
func (pl *NodeCapability) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*framework.PreFilterResult, *fwk.Status) {
	reqs, err := pl.helper.InferPodCreateRequirements(ctx, pod)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("inferring pod requirements: %w", err))
	}
	if reqs == nil || len(reqs.Capabilities) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	cycleState.Write(preFilterStateKey, &preFilterState{reqs: reqs})
	return nil, fwk.NewStatus(fwk.Success)
}

// PreFilterExtensions returns pre-filter extensions, pod add and remove.
func (pl *NodeCapability) PreFilterExtensions() framework.PreFilterExtensions {
	return nil
}

func getPreFilterState(cycleState fwk.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		// preFilterState doesn't exist, that means PreFilter rejected the pod.
		return nil, fmt.Errorf("error reading %q from cycle-state: %w", preFilterStateKey, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("invalid PreFilter state, got type %T, expected %T", c, &preFilterState{})
	}
	return s, nil
}

// Filter checks if the node has the required capabilities.
func (pl *NodeCapability) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if nodeInfo.Node() == nil {
		return fwk.NewStatus(fwk.Error, "node not found")
	}
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	match, err := pl.helper.MatchNode(ctx, s.reqs, nodeInfo.Node())
	if err != nil {
		return fwk.AsStatus(err)
	}
	if !match {
		return fwk.NewStatus(fwk.Unschedulable, "node does not have all required capabilities")
	}
	return nil
}
