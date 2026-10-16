/*
Copyright The Kubernetes Authors.

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

package deferredpodscheduling

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

type DeferredPodScheduling struct {
	handle                                             fwk.Handle
	enableInPlacePodVerticalScalingSchedulerPreemption bool
}

var _ fwk.PreFilterPlugin = &DeferredPodScheduling{}
var _ fwk.FilterPlugin = &DeferredPodScheduling{}
var _ fwk.EnqueueExtensions = &DeferredPodScheduling{}
var _ fwk.BindPlugin = &DeferredPodScheduling{}
var _ fwk.RelevancePlugin = &DeferredPodScheduling{}

const (
	Name                                  = names.DeferredPodScheduling
	ErrReasonNodeDisablesResizePreemption = "node had resize preemption disabled"
)

func (pl *DeferredPodScheduling) Name() string {
	return Name
}

// IsRelevantForPodResize returns true to indicate that DeferredPodScheduling is relevant for pod resizes.
func (pl *DeferredPodScheduling) IsRelevantForPodResize() bool {
	return true
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *DeferredPodScheduling) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.UpdateNodePreemptionPolicy},
			QueueingHintFn: pl.isSchedulableAfterNodeChange,
		},
		{
			Event:          fwk.ClusterEvent{Resource: fwk.Node, ActionType: fwk.Add},
			QueueingHintFn: pl.isSchedulableAfterNodeAdd,
		},
	}, nil
}

func (pl *DeferredPodScheduling) isSchedulableAfterNodeChange(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(pod) {
		return fwk.QueueSkip, nil
	}
	oldNode, ok := oldObj.(*v1.Node)
	if !ok {
		return fwk.Queue, fmt.Errorf("got object of type %T, want *v1.Node", oldObj)
	}
	newNode, ok := newObj.(*v1.Node)
	if !ok {
		return fwk.Queue, fmt.Errorf("got object of type %T, want *v1.Node", newObj)
	}

	if pod.Spec.NodeName != newNode.Name {
		return fwk.QueueSkip, nil
	}

	oldNodeDisabled := oldNode.Spec.PodPreemptionPolicy != nil && len(oldNode.Spec.PodPreemptionPolicy.DisableResizePreemption) > 0
	newNodeDisabled := newNode.Spec.PodPreemptionPolicy != nil && len(newNode.Spec.PodPreemptionPolicy.DisableResizePreemption) > 0

	// If resize preemption was disabled and is now enabled, the pod may be schedulable.
	if oldNodeDisabled && !newNodeDisabled {
		logger.V(5).Info("Node preemption policy changed from disabled to enabled, queueing pod", "pod", klog.KObj(pod), "node", newNode.Name)
		return fwk.Queue, nil
	}

	return fwk.QueueSkip, nil
}

func (pl *DeferredPodScheduling) isSchedulableAfterNodeAdd(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(pod) {
		return fwk.QueueSkip, nil
	}
	node, ok := newObj.(*v1.Node)
	if !ok {
		return fwk.Queue, fmt.Errorf("got object of type %T, want *v1.Node", newObj)
	}

	if pod.Spec.NodeName != node.Name {
		return fwk.QueueSkip, nil
	}

	// If the added node has preemption enabled, the pod may be schedulable.
	if node.Spec.PodPreemptionPolicy == nil || len(node.Spec.PodPreemptionPolicy.DisableResizePreemption) == 0 {
		logger.V(5).Info("Node added with preemption enabled, queueing pod", "pod", klog.KObj(pod), "node", node.Name)
		return fwk.Queue, nil
	}

	return fwk.QueueSkip, nil
}

func (pl *DeferredPodScheduling) PreFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodes []fwk.NodeInfo) (*fwk.PreFilterResult, *fwk.Status) {
	if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(pod) {
		return nil, fwk.NewStatus(fwk.Skip)
	}
	return nil, nil
}

func (pl *DeferredPodScheduling) PreFilterExtensions() fwk.PreFilterExtensions {
	return nil
}

func (pl *DeferredPodScheduling) Filter(ctx context.Context, _ fwk.CycleState, pod *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(pod) {
		return nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return fwk.NewStatus(fwk.Error, "node not found")
	}

	if pod.Spec.NodeName != "" && pod.Spec.NodeName != node.Name {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "pod assigned to different node")
	}

	if node.Spec.PodPreemptionPolicy != nil && len(node.Spec.PodPreemptionPolicy.DisableResizePreemption) > 0 {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, ErrReasonNodeDisablesResizePreemption)
	}
	return nil
}

// Bind sets the SchedulerEvaluationCompleted condition on deferred resize pods that pass
// scheduling evaluation, skipping DefaultBinder because the pod is already bound to its node.
func (pl *DeferredPodScheduling) Bind(ctx context.Context, state fwk.CycleState, p *v1.Pod, nodeName string) *fwk.Status {
	if !pl.enableInPlacePodVerticalScalingSchedulerPreemption || !resource.IsPodResizeDeferred(p) {
		return fwk.NewStatus(fwk.Skip)
	}

	cachedPod, err := pl.handle.SharedInformerFactory().Core().V1().Pods().Lister().Pods(p.Namespace).Get(p.Name)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fwk.AsStatus(err)
	}
	// Complete the binding cycle without a patch if the pod was deleted or recreated, its deferred resize was
	// resolved while in flight, or an earlier cycle already set the condition: bindingCycle marks the pod Done
	// before Bind, so an update in between can queue it again.
	if cachedPod.UID != p.UID || !resource.IsPodResizeDeferred(cachedPod) || resource.IsPodResizeSchedulerEvaluationCompleted(cachedPod) {
		return nil
	}
	// The queue drops updates received while the pod is in flight, so retry the new spec after backoff
	// instead of marking an evaluation of the old one complete.
	if cachedPod.Generation != p.Generation {
		return fwk.AsStatus(fmt.Errorf("pod spec changed from generation %d to %d during scheduling, retrying", p.Generation, cachedPod.Generation))
	}
	p = cachedPod

	condition := &v1.PodCondition{
		Type:               v1.SchedulerEvaluationCompleted,
		Status:             v1.ConditionTrue,
		Reason:             v1.PodReasonDeferredPodResize,
		ObservedGeneration: podutil.CalculatePodConditionObservedGeneration(&p.Status, p.Generation, v1.SchedulerEvaluationCompleted),
	}

	if pl.handle.APICacher() != nil {
		onFinish, err := pl.handle.APICacher().PatchPodStatus(p, []*v1.PodCondition{condition}, nil)
		if err != nil {
			return fwk.AsStatus(err)
		}
		if err := pl.handle.APICacher().WaitOnFinish(ctx, onFinish); err != nil {
			return fwk.AsStatus(err)
		}
		return nil
	}

	podStatusCopy := p.Status.DeepCopy()
	if !podutil.UpdatePodCondition(podStatusCopy, condition) {
		return nil
	}
	if err := util.PatchPodStatus(ctx, pl.handle.ClientSet(), p.Name, p.Namespace, &p.Status, podStatusCopy); err != nil {
		return fwk.AsStatus(err)
	}
	return nil
}

func New(_ context.Context, _ runtime.Object, h fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &DeferredPodScheduling{
		handle: h,
		enableInPlacePodVerticalScalingSchedulerPreemption: fts.EnableInPlacePodVerticalScalingSchedulerPreemption,
	}, nil
}
