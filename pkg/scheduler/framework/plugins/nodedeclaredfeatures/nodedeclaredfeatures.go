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

package nodedeclaredfeatures

import (
	"context"
	"fmt"
	"reflect"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-base/version"
	"k8s.io/component-helpers/nodedeclaredfeatures"
	"k8s.io/component-helpers/nodedeclaredfeatures/features"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.NodeDeclaredFeatures
	// preFilterStateKey is the key in CycleState used to store the pod's feature requirements.
	preFilterStateKey fwk.StateKey = "PreFilter" + Name
)

// preFilterState computed at PreFilter and used at Filter.
type preFilterState struct {
	reqs []string
}

// Clone implements StateData.
func (s *preFilterState) Clone() fwk.StateData {
	return s
}

// NodeDeclaredFeatures is a plugin that checks if a node has all the features required by a pod.
type NodeDeclaredFeatures struct {
	helper  *nodedeclaredfeatures.Helper
	version string
}

var _ fwk.PreFilterPlugin = &NodeDeclaredFeatures{}
var _ fwk.FilterPlugin = &NodeDeclaredFeatures{}
var _ fwk.EnqueueExtensions = &NodeDeclaredFeatures{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeDeclaredFeatures) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, handle fwk.Handle) (fwk.Plugin, error) {
	helper, err := nodedeclaredfeatures.NewHelper(features.AllFeatures)
	if err != nil {
		return nil, fmt.Errorf("failed to create node feature helper: %w", err)
	}
	return &NodeDeclaredFeatures{helper: helper, version: version.Get().String()}, nil
}

// PreFilter checks if the pod has any feature requirements.
func (pl *NodeDeclaredFeatures) PreFilter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	reqs, err := pl.helper.InferForPodCreate(pod, pl.version)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("inferring pod requirements: %w", err))
	}
	if len(reqs) == 0 {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	cycleState.Write(preFilterStateKey, &preFilterState{reqs: reqs})
	return nil, fwk.NewStatus(fwk.Success)
}

// PreFilterExtensions returns pre-filter extensions, pod add and remove.
func (pl *NodeDeclaredFeatures) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

func getPreFilterState(cycleState fwk.CycleState) (*preFilterState, error) {
	c, err := cycleState.Read(preFilterStateKey)
	if err != nil {
		return nil, fmt.Errorf("error reading %q from cycle-state: %w", preFilterStateKey, err)
	}

	s, ok := c.(*preFilterState)
	if !ok {
		return nil, fmt.Errorf("invalid PreFilter state, got type %T, expected %T", c, &preFilterState{})
	}
	return s, nil
}

// Filter checks if the node has the required features.
func (pl *NodeDeclaredFeatures) Filter(ctx context.Context, cycleState fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if nodeInfo.Node() == nil {
		return fwk.NewStatus(fwk.Error, "node not found")
	}
	s, err := getPreFilterState(cycleState)
	if err != nil {
		return fwk.AsStatus(err)
	}

	result, err := pl.helper.MatchNode(s.reqs, nodeInfo.Node())
	if err != nil {
		return fwk.AsStatus(err)
	}

	if !result.IsMatch {
		return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("node declared features check failed: %s", strings.Join(result.UnsatisfiedRequirements, ", ")))
	}

	return nil
}

// EventsToRegister returns events that may make a pod schedulable. It is required for the EnqueueExtensions interface.
func (pl *NodeDeclaredFeatures) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add | fwk.UpdateNodeDeclaredFeature},
			QueueingHintFn: pl.isSchedulableAfterNodeChange,
		},
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.UpdatePodFeatureRequirement},
			QueueingHintFn: pl.isSchedulableAfterPodUpdate,
		},
	}, nil
}

func (pl *NodeDeclaredFeatures) isSchedulableAfterPodUpdate(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldPod, newPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	newReqs, err := pl.helper.InferForPodUpdate(oldPod, newPod, pl.version)
	if err != nil {
		logger.Error(err, "Failed to infer pod requirements for queueing hint", "pod", klog.KObj(pod))
		return fwk.QueueSkip, nil
	}

	// No new requirements, no need to requeue.
	if len(newReqs) == 0 {
		return fwk.QueueSkip, nil
	}

	logger.V(4).Info("Pod update is relevant for pod, queueing", "pod", klog.KObj(pod))
	return fwk.Queue, nil
}

func (pl *NodeDeclaredFeatures) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldNode, newNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}
	if reflect.DeepEqual(oldNode.Status.DeclaredFeatures, newNode.Status.DeclaredFeatures) {
		return fwk.QueueSkip, nil
	}
	logger.V(4).Info("Node declared features updated, queueing", "pod", klog.KObj(pod))
	return fwk.Queue, nil
}
