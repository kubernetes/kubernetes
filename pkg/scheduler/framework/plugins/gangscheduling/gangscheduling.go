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

package gangscheduling

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.GangScheduling
)

// GangScheduling is a plugin that enforces "all-or-nothing" scheduling for pods
// belonging to a Workload with a Gang scheduling policy.
type GangScheduling struct {
	handle         fwk.Handle
	workloadLister schedulinglisters.WorkloadLister
}

var _ fwk.EnqueueExtensions = &GangScheduling{}
var _ fwk.PreEnqueuePlugin = &GangScheduling{}
var _ fwk.ReservePlugin = &GangScheduling{}
var _ fwk.PermitPlugin = &GangScheduling{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &GangScheduling{
		handle:         fh,
		workloadLister: fh.SharedInformerFactory().Scheduling().V1alpha1().Workloads().Lister(),
	}, nil
}

// Name returns name of the plugin.
func (pl *GangScheduling) Name() string {
	return Name
}

// EventsToRegister returns the possible events that may make a Pod failed by this plugin schedulable.
func (pl *GangScheduling) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	return []fwk.ClusterEventWithHint{
		// A new pod being added might be the one that completes a gang, meeting its MinCount requirement.
		// Workload reference is immutable, so there is no need to subscribe on Pod/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterPodAdded},
		// A Workload being added can be making a waiting gang schedulable.
		// Workload's PodGroups are immutable, so there's no need to handle Workload/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.Workload, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterWorkloadAdded},
	}, nil
}

// matchingWorkloadReference returns true if two pods belong to the same workload, including their pod group and replica key.
func matchingWorkloadReference(pod1, pod2 *v1.Pod) bool {
	return pod1.Spec.WorkloadRef != nil && pod2.Spec.WorkloadRef != nil && pod1.Namespace == pod2.Namespace && *pod1.Spec.WorkloadRef == *pod2.Spec.WorkloadRef
}

func (pl *GangScheduling) isSchedulableAfterPodAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if !matchingWorkloadReference(pod, addedPod) {
		logger.V(5).Info("another pod was added but it doesn't match the target pod's workload",
			"pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "addedPod", klog.KObj(addedPod), "addedWorkloadRef", pod.Spec.WorkloadRef)
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("another pod was added and it matches the target pod's workload, which may make the pod schedulable",
		"pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "addedPod", klog.KObj(addedPod), "addedWorkloadRef", pod.Spec.WorkloadRef)
	return fwk.Queue, nil
}

func (pl *GangScheduling) isSchedulableAfterWorkloadAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedWorkload, err := util.As[*schedulingapi.Workload](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.WorkloadRef == nil || pod.Namespace != addedWorkload.Namespace || pod.Spec.WorkloadRef.Name != addedWorkload.Name {
		logger.V(5).Info("workload was added but it doesn't match the target pod's workloadRef",
			"pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "addedWorkload", klog.KObj(addedWorkload))
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("workload was added and it matches the target pod's workload, which may make the pod schedulable",
		"pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "addedWorkload", klog.KObj(addedWorkload))
	return fwk.Queue, nil
}

// PreEnqueue checks if the pod belongs to a gang and, if so, whether the gang has met its MinCount of available pods.
// If not, the pod is rejected until more pods arrive.
func (pl *GangScheduling) PreEnqueue(ctx context.Context, pod *v1.Pod) *fwk.Status {
	if pod.Spec.WorkloadRef == nil {
		return nil
	}

	namespace := pod.Namespace
	workloadRef := pod.Spec.WorkloadRef

	workload, err := pl.workloadLister.Workloads(namespace).Get(workloadRef.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// The pod is unschedulable until its Workload object is created.
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for pods's workload %q to appear in scheduling queue", workloadRef.Name))
		}
		klog.FromContext(ctx).Error(err, "Failed to get workload", "pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef)
		return fwk.AsStatus(fmt.Errorf("failed to get workload %s/%s", namespace, workloadRef.Name))
	}

	policy, ok := podGroupPolicy(workload, workloadRef.PodGroup)
	if !ok {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("pod group %q doesn't exist for a workload %q", workloadRef.PodGroup, workload.Name))
	}
	// This plugin only cares about pods with a Gang scheduling policy.
	if policy.Gang == nil {
		return nil
	}

	podGroupInfo, err := pl.handle.WorkloadManager().PodGroupInfo(namespace, workloadRef)
	if err != nil {
		return fwk.AsStatus(err)
	}
	allPods := podGroupInfo.AllPods()
	if len(allPods) < int(policy.Gang.MinCount) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue")
	}

	// The quorum is met, allow the pod to enter the scheduling queue.
	return nil
}

// Reserve is called after a node has been selected for the pod. For gang pods,
// this stage marks the pod as "assumed" in the WorkloadManager,
// contributing to the count of pods ready to be co-scheduled at the Permit stage.
func (pl *GangScheduling) Reserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) *fwk.Status {
	if pod.Spec.WorkloadRef == nil {
		return nil
	}
	podGroupInfo, err := pl.handle.WorkloadManager().PodGroupInfo(pod.Namespace, pod.Spec.WorkloadRef)
	if err != nil {
		return fwk.AsStatus(err)
	}
	podGroupInfo.AssumePod(pod.UID)
	return nil
}

// Unreserve removes the gang pod from the "assumed" state in the WorkloadManager,
// ensuring it doesn't count towards the Permit quorum.
func (pl *GangScheduling) Unreserve(ctx context.Context, cs fwk.CycleState, pod *v1.Pod, nodeName string) {
	if pod.Spec.WorkloadRef == nil {
		return
	}
	podGroupInfo, err := pl.handle.WorkloadManager().PodGroupInfo(pod.Namespace, pod.Spec.WorkloadRef)
	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Failed to get pod group info", "pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef)
		return
	}
	podGroupInfo.ForgetPod(pod.UID)
}

// podGroupPolicy is a helper to find the policy for a specific pod group name in a workload.
func podGroupPolicy(workload *schedulingapi.Workload, podGroupName string) (schedulingapi.PodGroupPolicy, bool) {
	for _, podGroup := range workload.Spec.PodGroups {
		if podGroup.Name == podGroupName {
			return podGroup.Policy, true
		}
	}
	return schedulingapi.PodGroupPolicy{}, false
}

// Permit forces all pods in a gang to wait at this stage. Once the number of waiting (assumed) pods
// reaches the gang's MinCount, all pods in the gang are permitted to proceed to binding simultaneously.
func (pl *GangScheduling) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if pod.Spec.WorkloadRef == nil {
		return nil, 0
	}

	logger := klog.FromContext(ctx)
	namespace := pod.Namespace
	workloadRef := pod.Spec.WorkloadRef

	workload, err := pl.workloadLister.Workloads(namespace).Get(workloadRef.Name)
	if err != nil {
		// It's likely that the workload was removed or another error happened.
		// Returning error to retry the Pod when the Workload is added again.
		return fwk.AsStatus(fmt.Errorf("failed to get workload %s/%s: %w", namespace, workloadRef.Name, err)), 0
	}

	policy, ok := podGroupPolicy(workload, workloadRef.PodGroup)
	if !ok {
		return fwk.AsStatus(fmt.Errorf("pod group %q doesn't exist for a workload %q", workloadRef.PodGroup, workload.Name)), 0
	}
	// This plugin only cares about pods with a Gang scheduling policy.
	if policy.Gang == nil {
		return nil, 0
	}

	podGroupInfo, err := pl.handle.WorkloadManager().PodGroupInfo(namespace, workloadRef)
	if err != nil {
		return fwk.AsStatus(err), 0
	}
	assumedPods := podGroupInfo.AssumedPods()
	assumedOrAssignedPods := assumedPods.Union(podGroupInfo.AssignedPods())
	if len(assumedOrAssignedPods) < int(policy.Gang.MinCount) {
		// Activate unscheduled pods from this pod group in case they were waiting for this pod to be scheduled.
		unscheduledPods := podGroupInfo.UnscheduledPods()
		pl.handle.Activate(klog.FromContext(ctx), unscheduledPods)
		logger.V(4).Info("Quorum is not met for a gang. Waiting for another pod to allow", "pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "activatedPods", len(unscheduledPods))
		return fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be waiting on permit"), podGroupInfo.SchedulingTimeout()
	}

	logger.V(4).Info("Quorum is met for a gang. Allowing other pods from a gang waiting on permit", "pod", klog.KObj(pod), "workloadRef", pod.Spec.WorkloadRef, "allowedPods", len(assumedPods))

	// The quorum is met. Allow this pod and signal all other waiting pods from the same gang to proceed.
	for podUID := range assumedPods {
		waitingPod := pl.handle.GetWaitingPod(podUID)
		if waitingPod != nil {
			waitingPod.Allow(Name)
		}
	}

	return nil, 0
}
