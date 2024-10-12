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

package nodeunschedulable

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	v1helper "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// NodeUnschedulable plugin filters nodes that set node.Spec.Unschedulable=true unless
// the pod tolerates {key=node.kubernetes.io/unschedulable, effect:NoSchedule} taint.
type NodeUnschedulable struct {
	enableSchedulingQueueHint bool
}

var _ framework.FilterPlugin = &NodeUnschedulable{}
var _ framework.EnqueueExtensions = &NodeUnschedulable{}

// Name is the name of the plugin used in the plugin registry and configurations.
const Name = names.NodeUnschedulable

const (
	// ErrReasonUnknownCondition is used for NodeUnknownCondition predicate error.
	ErrReasonUnknownCondition = "node(s) had unknown conditions"
	// ErrReasonUnschedulable is used for NodeUnschedulable predicate error.
	ErrReasonUnschedulable = "node(s) were unschedulable"
)

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *NodeUnschedulable) EventsToRegister(_ context.Context) ([]framework.ClusterEventWithHint, error) {
	if !pl.enableSchedulingQueueHint {
		return []framework.ClusterEventWithHint{
			// A note about UpdateNodeLabel event:
			// Ideally, it's supposed to register only Add | UpdateNodeTaint because UpdateNodeLabel will never change the result from this plugin.
			// But, we may miss Node/Add event due to preCheck, and we decided to register UpdateNodeTaint | UpdateNodeLabel for all plugins registering Node/Add.
			// See: https://github.com/kubernetes/kubernetes/issues/109437
			{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeTaint | framework.UpdateNodeLabel}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
		}, nil
	}

	return []framework.ClusterEventWithHint{
		// When QueueingHint is enabled, we don't use preCheck and we don't need to register UpdateNodeLabel event.
		{Event: framework.ClusterEvent{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeTaint}, QueueingHintFn: pl.isSchedulableAfterNodeChange},
		// When the QueueingHint feature is enabled,
		// the scheduling queue uses Pod/Update Queueing Hint
		// to determine whether a Pod's update makes the Pod schedulable or not.
		// https://github.com/kubernetes/kubernetes/pull/122234
		{Event: framework.ClusterEvent{Resource: framework.Pod, ActionType: framework.UpdatePodTolerations}, QueueingHintFn: pl.isSchedulableAfterPodTolerationChange},
	}, nil
}

// isSchedulableAfterPodTolerationChange is invoked whenever a pod's toleration changed.
func (pl *NodeUnschedulable) isSchedulableAfterPodTolerationChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	_, modifiedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	if pod.UID == modifiedPod.UID {
		// Note: we don't need to check oldPod tolerations the taint because:
		// - Taint can be added, but can't be modified nor removed.
		// - If the Pod already has the toleration, it shouldn't have rejected by this plugin in the first place.
		//   Meaning, here this Pod has been rejected by this plugin, and hence it shouldn't have the toleration yet.
		if v1helper.TolerationsTolerateTaint(modifiedPod.Spec.Tolerations, &v1.Taint{
			Key:    v1.TaintNodeUnschedulable,
			Effect: v1.TaintEffectNoSchedule,
		}) {
			// This update makes the pod tolerate the unschedulable taint.
			logger.V(5).Info("a new toleration is added for the unschedulable Pod, and it may make it schedulable", "pod", klog.KObj(modifiedPod))
			return framework.Queue, nil
		}
		logger.V(5).Info("a new toleration is added for the unschedulable Pod, but it's an unrelated toleration", "pod", klog.KObj(modifiedPod))
		return framework.QueueSkip, nil
	}

	logger.V(5).Info("a new toleration is added for a Pod, but it's an unrelated Pod and wouldn't change the TaintToleration plugin's decision", "pod", klog.KObj(modifiedPod))

	return framework.QueueSkip, nil
}

// isSchedulableAfterNodeChange is invoked for all node events reported by
// an informer. It checks whether that change made a previously unschedulable
// pod schedulable.
func (pl *NodeUnschedulable) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (framework.QueueingHint, error) {
	originalNode, modifiedNode, err := util.As[*v1.Node](oldObj, newObj)
	if err != nil {
		return framework.Queue, err
	}

	// We queue this Pod when -
	// 1. the node is updated from unschedulable to schedulable.
	// 2. the node is added and is schedulable.
	if (originalNode != nil && originalNode.Spec.Unschedulable && !modifiedNode.Spec.Unschedulable) ||
		(originalNode == nil && !modifiedNode.Spec.Unschedulable) {
		logger.V(5).Info("node was created or updated, pod may be schedulable now", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
		return framework.Queue, nil
	}

	logger.V(5).Info("node was created or updated, but it doesn't make this pod schedulable", "pod", klog.KObj(pod), "node", klog.KObj(modifiedNode))
	return framework.QueueSkip, nil
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *NodeUnschedulable) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
func (pl *NodeUnschedulable) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	node := nodeInfo.Node()

	if !node.Spec.Unschedulable {
		return nil
	}

	// If pod tolerate unschedulable taint, it's also tolerate `node.Spec.Unschedulable`.
	podToleratesUnschedulable := v1helper.TolerationsTolerateTaint(pod.Spec.Tolerations, &v1.Taint{
		Key:    v1.TaintNodeUnschedulable,
		Effect: v1.TaintEffectNoSchedule,
	})
	if !podToleratesUnschedulable {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonUnschedulable)
	}

	return nil
}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, _ framework.Handle, fts feature.Features) (framework.Plugin, error) {
	return &NodeUnschedulable{enableSchedulingQueueHint: fts.EnableSchedulingQueueHint}, nil
}
