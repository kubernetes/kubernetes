/*
Copyright 2014 The Kubernetes Authors.

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

package scheduler

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/utils/ptr"
)

// errPodGroupUnschedulable is used to describe that the pod group is unschedulable.
var errPodGroupUnschedulable = fmt.Errorf("other pods from a pod group are unschedulable")

// scheduleOnePod does the entire workload-aware scheduling workflow for a single pod group.
func (sched *Scheduler) scheduleOnePodGroup(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo) {
	logger := klog.FromContext(ctx)
	// TODO(knelasevero): Remove duplicated keys from log entry calls
	// When contextualized logging hits GA
	// https://github.com/kubernetes/kubernetes/issues/111672
	logger = klog.LoggerWithValues(logger, "podGroup", klog.KObj(podGroupInfo))
	ctx = klog.NewContext(ctx, logger)

	logger.V(3).Info("Attempting to schedule pod group", "podGroup", klog.KObj(podGroupInfo))

	// Synchronously attempt to find a fit for the pod group.
	start := time.Now()

	sched.podGroupCycle(ctx, podGroupInfo, start)
}

// podGroupInfoForPod is a temporary function that obtains the QueuedPodGroupInfo based on the pod that got popped from the scheduling queue.
// Ultimately, scheduling queue, adjusted with workload-awareness will return such QueuedPodGroupInfo directly.
func (sched *Scheduler) podGroupInfoForPod(ctx context.Context, pInfo *framework.QueuedPodInfo) (*framework.QueuedPodGroupInfo, error) {
	logger := klog.FromContext(ctx)

	// Get the actual pod group state
	podGroupState, err := sched.WorkloadManager.PodGroupState(pInfo.Pod.Namespace, pInfo.Pod.Spec.WorkloadRef)
	if err != nil {
		return nil, fmt.Errorf("error while retrieving pod group state: %w", err)
	}
	unscheduledPods := podGroupState.UnscheduledPods()

	podGroupInfo := &framework.QueuedPodGroupInfo{
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace:   pInfo.Pod.Namespace,
			WorkloadRef: pInfo.Pod.Spec.WorkloadRef,
		},
		QueuedPodInfos: make([]*framework.QueuedPodInfo, 0, len(unscheduledPods)+1),
	} // TODO: Get the PodGroup object from the informer and write it to the PodGroupInfo, if needed
	podGroupInfo.QueuedPodInfos = append(podGroupInfo.QueuedPodInfos, pInfo)

	// Pop all unscheduled pods from the scheduling queue
	for _, pod := range unscheduledPods {
		unscheduledPodInfo := sched.SchedulingQueue.PopSpecificPod(logger, pod)
		if unscheduledPodInfo == nil {
			logger.V(5).Info("Pod available in pod group state not available in scheduling queue", "podGroup", klog.KObj(podGroupInfo), "pod", klog.KObj(pod))
			continue
		}
		podGroupInfo.QueuedPodInfos = append(podGroupInfo.QueuedPodInfos, unscheduledPodInfo)
	}
	// Sort the pods in deterministic order. First by priority, then by their InitialAttemptTimestamp.
	sort.Slice(podGroupInfo.QueuedPodInfos, func(i, j int) bool {
		pInfo1 := podGroupInfo.QueuedPodInfos[i]
		pInfo2 := podGroupInfo.QueuedPodInfos[j]
		p1 := corev1helpers.PodPriority(pInfo1.GetPodInfo().GetPod())
		p2 := corev1helpers.PodPriority(pInfo2.GetPodInfo().GetPod())
		// Timestamps should be set, but dereferencing them for safety.
		p1Timestamp := ptr.Deref(pInfo1.InitialAttemptTimestamp, time.Time{})
		p2Timestamp := ptr.Deref(pInfo2.InitialAttemptTimestamp, time.Time{})
		return (p1 > p2) || (p1 == p2 && p1Timestamp.Before(p2Timestamp))
	})

	// Populate UnscheduledPods based on the QueuedPodInfos.
	podGroupInfo.UnscheduledPods = make([]*v1.Pod, 0, len(podGroupInfo.QueuedPodInfos))
	for _, pInfo := range podGroupInfo.QueuedPodInfos {
		podGroupInfo.UnscheduledPods = append(podGroupInfo.UnscheduledPods, pInfo.Pod)
	}

	return podGroupInfo, nil
}

// podSchedulingContext holds the precomputed data needed to handle the pod scheduling.
// Each scheduling attempt in the same pod group scheduling cycle for the same pod
// should call the withNewState method to recreate the podSchedulingContext with a new state.
type podSchedulingContext struct {
	logger         klog.Logger
	fwk            framework.Framework
	state          *framework.CycleState
	podsToActivate *framework.PodsToActivate
}

// withNewState returns a new podSchedulingContext instance with a new CycleState.
// This should be called before each scheduling attempt in the same pod group scheduling cycle
// for the same pod
func (psc *podSchedulingContext) withNewState() *podSchedulingContext {
	// Synchronously attempt to find a fit for the pod.
	state := framework.NewCycleState()
	// For the sake of performance, scheduler does not measure and export the scheduler_plugin_execution_duration metric
	// for every plugin execution in each scheduling cycle. Instead it samples a portion of scheduling cycles - percentage
	// determined by pluginMetricsSamplePercent. The line below helps to randomly pick appropriate scheduling cycles.
	state.SetRecordPluginMetrics(rand.Intn(100) < pluginMetricsSamplePercent)

	// Initialize an empty podsToActivate struct, which will be filled up by plugins or stay empty.
	// During pod group scheduling cycle, no pods will be activated using this,
	// as they will be eventually activated in the pod-by-pod phase.
	// We however still need to provide this storage, not to break any existing plugin
	// that may rely on this and run in the pod group scheduling cycle.
	podsToActivate := framework.NewPodsToActivate()
	state.Write(framework.PodsToActivateKey, podsToActivate)

	return &podSchedulingContext{
		logger:         psc.logger,
		fwk:            psc.fwk,
		state:          state,
		podsToActivate: podsToActivate,
	}
}

// initPodSchedulingContext initializes the scheduling context of a single pod for pod group scheduling cycle.
func (sched *Scheduler) initPodSchedulingContext(ctx context.Context, pod *v1.Pod) (*podSchedulingContext, bool) {
	logger := klog.FromContext(ctx)
	// TODO(knelasevero): Remove duplicated keys from log entry calls
	// When contextualized logging hits GA
	// https://github.com/kubernetes/kubernetes/issues/111672
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod))
	ctx = klog.NewContext(ctx, logger)

	podFwk, err := sched.frameworkForPod(pod)
	if err != nil {
		logger.Error(err, "Error occurred")
		return nil, false
	}
	if sched.skipPodSchedule(ctx, podFwk, pod) {
		// We don't put this Pod back to the queue, but we have to cleanup the in-flight pods/events.
		return nil, false
	}

	podCtx := &podSchedulingContext{
		logger: logger,
		fwk:    podFwk,
	}
	return podCtx.withNewState(), true
}

// initPodGroupSchedulingContexts initializes and returns the pod scheduling contexts for the pod group.
// If the individual pod shouldn't be scheduled anymore, for example, because it's terminating,
// it's removed entirely from the podGroupInfo.
func (sched *Scheduler) initPodGroupSchedulingContexts(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo) []*podSchedulingContext {
	filteredQueuedPodInfos := make([]*framework.QueuedPodInfo, 0, len(podGroupInfo.QueuedPodInfos))
	podCtxs := make([]*podSchedulingContext, 0, len(podGroupInfo.QueuedPodInfos))
	for _, podInfo := range podGroupInfo.QueuedPodInfos {
		pod := podInfo.Pod
		podCtx, ok := sched.initPodSchedulingContext(ctx, pod)
		if !ok {
			sched.SchedulingQueue.Done(pod.UID)
			continue
		}
		filteredQueuedPodInfos = append(filteredQueuedPodInfos, podInfo)
		podCtxs = append(podCtxs, podCtx)
	}
	if len(filteredQueuedPodInfos) != len(podGroupInfo.QueuedPodInfos) {
		podGroupInfo.QueuedPodInfos = filteredQueuedPodInfos
		podGroupInfo.UnscheduledPods = make([]*v1.Pod, 0, len(podGroupInfo.QueuedPodInfos))
		for _, pInfo := range podGroupInfo.QueuedPodInfos {
			podGroupInfo.UnscheduledPods = append(podGroupInfo.UnscheduledPods, pInfo.Pod)
		}
	}
	return podCtxs
}

// podGroupCycle runs a pod group scheduling cycle for the given pod group in a single cluster snapshot.
func (sched *Scheduler) podGroupCycle(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, start time.Time) {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	logger := klog.FromContext(podGroupCycleCtx)

	podCtxs := sched.initPodGroupSchedulingContexts(podGroupCycleCtx, podGroupInfo)

	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		logger.Error(err, "Error updating snapshot")
		for i, pInfo := range podGroupInfo.QueuedPodInfos {
			podCtx := podCtxs[i]
			sched.FailureHandler(podGroupCycleCtx, podCtx.fwk, pInfo, fwk.AsStatus(err), nil, time.Time{})
		}
		return
	}

	result := sched.podGroupSchedulingDefaultAlgorithm(podGroupCycleCtx, podGroupInfo, podCtxs)

	// applyPodGroupAlgorithmResult can dispatch binding goroutines, so should be called with the noncancelable ctx.
	sched.applyPodGroupAlgorithmResult(ctx, podGroupInfo, result)

	// TBD: Record PodGroup-level metrics and logs, return podGroupInfo to the queue.
}

// algorithmResult stores the scheduling result and status for a scheduling attempt of a single pod.
type algorithmResult struct {
	// scheduleResult is a scheduling algorithm result.
	scheduleResult ScheduleResult
	// podCtx is a specific pod scheduling context used for the scheduling algorithm.
	podCtx *podSchedulingContext
	// requiresPreemption determines whether this pod requires a preemption to proceed or not.
	requiresPreemption bool
	// status is a scheduling algorithm status.
	status *fwk.Status
	// permitStatus is a status of the permit check.
	// This is only set when the `status` is success or the `requiresPreemption` is true.
	permitStatus *fwk.Status
	// TBD: Victims for Delayed Preemption
}

// podGroupAlgorithmStatus is a status of a pod group scheduling algorithm.
type podGroupAlgorithmStatus string

const (
	// podGroupFeasible means that the pod group is schedulable, doesn't require any preemption
	// and its feasible pods should be moved to the binding cycle.
	podGroupFeasible podGroupAlgorithmStatus = "feasible"
	// podGroupUnschedulable means that the pod group is unschedulable
	// and all its pods should be moved back to the scheduling queue as unschedulable.
	podGroupUnschedulable podGroupAlgorithmStatus = "unschedulable"
	// podGroupRequiresPreemption means that the pod group requires preemption,
	// so all its pods should be moved back to the scheduling queue,
	// waiting for resources to be released.
	podGroupRequiresPreemption podGroupAlgorithmStatus = "requires_preemption"
)

// podGroupAlgorithmResult stores the scheduling pod scheduling results for a pod group
// and any information needed to act on these results.
type podGroupAlgorithmResult struct {
	// podResults is the list of scheduling results for each pod in the group.
	podResults []algorithmResult
	// status is the final status of the pod group algorithm.
	status podGroupAlgorithmStatus
}

// podGroupSchedulingDefaultAlgorithm runs the default algorithm for scheduling a pod group.
// It tries to schedule each pod using standard filtering and scoring logic in a fixed order.
// If a pod requires preemption to be schedulable, subsequent pods in the algorithm
// treat that pod as already scheduled on that node with victims being already removed in memory.
func (sched *Scheduler) podGroupSchedulingDefaultAlgorithm(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, podCtxs []*podSchedulingContext) podGroupAlgorithmResult {
	result := podGroupAlgorithmResult{
		podResults: make([]algorithmResult, 0, len(podGroupInfo.QueuedPodInfos)),
		status:     podGroupUnschedulable,
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running a pod group scheduling algorithm", "podGroup", klog.KObj(podGroupInfo), "unscheduledPodsCount", len(podGroupInfo.QueuedPodInfos))

	requiresPreemption := false
	for i, podInfo := range podGroupInfo.QueuedPodInfos {
		podCtx := podCtxs[i].withNewState()

		podResult, revertFn := sched.podGroupPodSchedulingAlgorithm(ctx, podGroupInfo, podInfo, podCtx)
		result.podResults = append(result.podResults, podResult)
		if !podResult.status.IsSuccess() && !podResult.requiresPreemption {
			// When a pod is not feasible and doesn't require preemption, it means that it failed scheduling.
			// However, the pod group can still be schedulable as long as the permit check can succeed.
			continue
		}
		// We unreserve the pod at the end of the whole algorithm (via defer) because it should be ultimately returned to the queue,
		// without binding it yet. We only assumed the pod to check feasibility of subsequent pods in the group.
		defer revertFn()

		requiresPreemption = requiresPreemption || podResult.requiresPreemption
		if podResult.permitStatus.IsSuccess() {
			// When the permit returns success for any pod, the pod group is schedulable.
			if requiresPreemption {
				// If any preemption is required the whole pod group requires it to be feasible.
				result.status = podGroupRequiresPreemption
			} else {
				result.status = podGroupFeasible
			}
		}
	}

	return result
}

// podGroupPodSchedulingAlgorithm runs a scheduling algorithm for individual pod from a pod group.
// It returns the algorithm result and, if successful or the preemption is required, the permit status together with the revert function.
func (sched *Scheduler) podGroupPodSchedulingAlgorithm(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, podInfo *framework.QueuedPodInfo, podCtx *podSchedulingContext) (algorithmResult, func()) {
	pod := podInfo.Pod
	logger := podCtx.logger
	ctx = klog.NewContext(ctx, logger)
	start := time.Now()

	logger.V(4).Info("Attempting to schedule a pod belonging to a pod group", "podGroup", klog.KObj(podGroupInfo), "pod", klog.KObj(pod))

	requiresPreemption := false
	scheduleResult, status := sched.schedulingAlgorithm(ctx, podCtx.state, podCtx.fwk, podInfo, start)
	if !status.IsSuccess() {
		if scheduleResult.nominatingInfo != nil && scheduleResult.nominatingInfo.NominatedNodeName != "" {
			// If the NominatedNodeName is set, the preemption is requried.
			// Continue with assuming and reserving, because the subsequent pods from this group
			// have to see this one as already scheduled on its nominated place.
			// Set SuggestedHost to NominatedNodeName to handle the pod similarly to one that is feasible.
			scheduleResult.SuggestedHost = scheduleResult.nominatingInfo.NominatedNodeName
			requiresPreemption = true
		} else {
			// In case of pod being just unschedulable or having an error, just return now.
			return algorithmResult{
				scheduleResult: scheduleResult,
				podCtx:         podCtx,
				status:         status,
			}, nil
		}
	}
	assumedPodInfo, assumeStatus := sched.assumeAndReserve(ctx, podCtx.state, podCtx.fwk, podInfo, scheduleResult)
	if !assumeStatus.IsSuccess() {
		return algorithmResult{
			scheduleResult: ScheduleResult{nominatingInfo: clearNominatedNode},
			podCtx:         podCtx,
			status:         assumeStatus,
		}, nil
	}

	revertFn := func() {
		err := sched.unreserveAndForget(ctx, podCtx.state, podCtx.fwk, assumedPodInfo, scheduleResult.SuggestedHost)
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Unreserve and forget failed")
		}
	}

	permitStatus := podCtx.fwk.RunPermitPluginsWithoutWaiting(ctx, podCtx.state, assumedPodInfo.Pod, scheduleResult.SuggestedHost)
	if !permitStatus.IsWait() && !permitStatus.IsSuccess() {
		revertFn()
		if permitStatus.IsRejected() {
			fitErr := &framework.FitError{
				NumAllNodes: 1,
				Pod:         assumedPodInfo.Pod,
				Diagnosis: framework.Diagnosis{
					NodeToStatus: framework.NewDefaultNodeToStatus(),
				},
			}
			fitErr.Diagnosis.NodeToStatus.Set(scheduleResult.SuggestedHost, permitStatus)
			fitErr.Diagnosis.AddPluginStatus(permitStatus)
			permitStatus = fwk.NewStatus(permitStatus.Code()).WithError(fitErr)
		}
		return algorithmResult{
			scheduleResult: ScheduleResult{nominatingInfo: clearNominatedNode},
			podCtx:         podCtx,
			status:         permitStatus,
		}, nil
	}

	return algorithmResult{
		scheduleResult:     scheduleResult,
		podCtx:             podCtx,
		status:             status,
		permitStatus:       permitStatus,
		requiresPreemption: requiresPreemption,
	}, revertFn
}

// applyPodGroupAlgorithmResult applies the result of the pod group scheduling algorithm.
// If that algorithm succedeed, the schedulable pods proceed to the binding cycle.
// Unschedulable pods are moved back to the scheduling queue and need to wait
// for the next pod group scheduling cycle.
// If the preemption is required for this pod group, all pods are moved back to the scheduling queue
// and require the next pod group scheduling cycle to verify the preemption outcome.
func (sched *Scheduler) applyPodGroupAlgorithmResult(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, result podGroupAlgorithmResult) {
	for i, podResult := range result.podResults {
		pInfo := podGroupInfo.QueuedPodInfos[i]
		podCtx := podResult.podCtx
		ctx := klog.NewContext(ctx, podCtx.logger)

		if podResult.status.IsSuccess() {
			nominatingInfo := &fwk.NominatingInfo{
				NominatingMode:    fwk.ModeOverride,
				NominatedNodeName: podResult.scheduleResult.SuggestedHost,
			}
			switch result.status {
			case podGroupFeasible:
				// Pod no longer needs a pod group scheduling cycle. Setting it to false to disable any checks in further functions.
				pInfo.NeedsPodGroupCycle = false
				// Schedule result is applied for pod and its binding cycle executes.
				assumedPodInfo, status := sched.prepareForBindingCycle(ctx, podCtx.state, podCtx.fwk, pInfo, podCtx.podsToActivate, podResult.scheduleResult)
				if !status.IsSuccess() {
					// In such unlikely situation just reject this pod.
					sched.FailureHandler(ctx, podCtx.fwk, pInfo, status, clearNominatedNode, time.Time{})
					continue
				}
				go sched.runBindingCycle(ctx, podCtx.state, podCtx.fwk, podResult.scheduleResult, assumedPodInfo, time.Time{}, podCtx.podsToActivate)
			case podGroupUnschedulable:
				// Pod group is unschedulable, so the pod has to be marked as unschedulable.
				// Its rejection status is set to its permit status message.
				status := fwk.NewStatus(fwk.UnschedulableAndUnresolvable, podResult.permitStatus.Message()).WithError(errPodGroupUnschedulable)
				sched.FailureHandler(ctx, podCtx.fwk, pInfo, status, clearNominatedNode, time.Time{})
			case podGroupRequiresPreemption:
				// Pod has to come back to the scheduling queue as unschedulable, waiting for preemption to complete.
				status := fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "preemption is required for other pods from a pod group").WithError(errPodGroupUnschedulable)
				sched.FailureHandler(ctx, podCtx.fwk, pInfo, status, nominatingInfo, time.Time{})
			}
		} else {
			// When an error occured or preemption is required for this pod, just call the FailureHandler.
			// TBD: Add a message to status if the pod used features for which finding a placement cannot be guaranteed,
			// such as heterogeneous pod group or using inter-pod dependencies.
			sched.FailureHandler(ctx, podCtx.fwk, pInfo, podResult.status, podResult.scheduleResult.nominatingInfo, time.Time{})
		}
	}
}
