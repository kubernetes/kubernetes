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

package scheduler

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/utils/ptr"
)

// errPodGroupUnschedulable is used to describe that the pod group is unschedulable.
var errPodGroupUnschedulable = fmt.Errorf("pod group is unschedulable")

// scheduleOnePodGroup does the entire workload-aware scheduling workflow for a single pod group.
func (sched *Scheduler) scheduleOnePodGroup(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo) {
	logger := klog.FromContext(ctx)
	// TODO(knelasevero): Remove duplicated keys from log entry calls
	// When contextualized logging hits GA
	// https://github.com/kubernetes/kubernetes/issues/111672
	logger = klog.LoggerWithValues(logger, "podGroup", klog.KObj(podGroupInfo))
	ctx = klog.NewContext(ctx, logger)

	schedFwk, err := sched.frameworkForPodGroup(podGroupInfo)
	if err != nil {
		for _, podInfo := range podGroupInfo.QueuedPodInfos {
			podFwk, podFwkErr := sched.frameworkForPod(podInfo.Pod)
			if podFwkErr != nil {
				// This shouldn't happen, because we only accept for scheduling the pods
				// which specify a scheduler name that matches one of the profiles.
				logger.Error(podFwkErr, "Error occurred")
				sched.SchedulingQueue.Done(podInfo.Pod.UID)
				return
			}
			sched.FailureHandler(ctx, podFwk, podInfo, fwk.AsStatus(err), clearNominatedNode, time.Now())
		}
		return
	}
	sched.skipPodGroupPodSchedule(ctx, schedFwk, podGroupInfo)
	// skipPodGroupPodSchedule could remove some pods from the pod group.
	// Pod group constraints will be re-evaluated on a Permit phase.
	// Now, verify if it has any pods left.
	if len(podGroupInfo.QueuedPodInfos) == 0 {
		return
	}

	logger.V(3).Info("Attempting to schedule pod group", "podGroup", klog.KObj(podGroupInfo))

	sched.podGroupCycle(ctx, schedFwk, podGroupInfo)
}

// frameworkForPodGroup obtains the concrete scheduler framework for the entire pod group.
func (sched *Scheduler) frameworkForPodGroup(podGroupInfo *framework.QueuedPodGroupInfo) (framework.Framework, error) {
	schedulerName := ""
	for _, pInfo := range podGroupInfo.QueuedPodInfos {
		if schedulerName == "" {
			schedulerName = pInfo.Pod.Spec.SchedulerName
		} else if pInfo.Pod.Spec.SchedulerName != schedulerName {
			return nil, fmt.Errorf("all pods in a single pod group should have the same .spec.schedulerName set, got: %q and %q", pInfo.Pod.Spec.SchedulerName, schedulerName)
		}
	}
	fwk, ok := sched.Profiles[schedulerName]
	if !ok {
		return nil, fmt.Errorf("profile not found for scheduler name %q", schedulerName)
	}
	return fwk, nil
}

// skipPodGroupPodSchedule skips the scheduling of particular pods from the group when they should no longer be considered.
// This can happen when the pod is already being deleted (i.e., when its deletionTimestamp is set)
// or when the pod has already been assumed.
func (sched *Scheduler) skipPodGroupPodSchedule(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) {
	filteredQueuedPodInfos := make([]*framework.QueuedPodInfo, 0, len(podGroupInfo.QueuedPodInfos))
	for _, podInfo := range podGroupInfo.QueuedPodInfos {
		if sched.skipPodSchedule(ctx, schedFwk, podInfo.Pod) {
			// We don't put this Pod back to the queue, but we have to cleanup the in-flight pods/events.
			sched.SchedulingQueue.Done(podInfo.Pod.UID)
			continue
		}
		filteredQueuedPodInfos = append(filteredQueuedPodInfos, podInfo)
	}
	if len(filteredQueuedPodInfos) != len(podGroupInfo.QueuedPodInfos) {
		podGroupInfo.QueuedPodInfos = filteredQueuedPodInfos
		podGroupInfo.UnscheduledPods = make([]*v1.Pod, 0, len(podGroupInfo.QueuedPodInfos))
		for _, pInfo := range podGroupInfo.QueuedPodInfos {
			podGroupInfo.UnscheduledPods = append(podGroupInfo.UnscheduledPods, pInfo.Pod)
		}
	}
}

// podGroupInfoForPod is a temporary function that obtains the QueuedPodGroupInfo based on the pod that got popped from the scheduling queue.
// Ultimately, scheduling queue, adjusted with workload-awareness will return such QueuedPodGroupInfo directly.
func (sched *Scheduler) podGroupInfoForPod(ctx context.Context, pInfo *framework.QueuedPodInfo) (*framework.QueuedPodGroupInfo, error) {
	logger := klog.FromContext(ctx)

	// Get the actual pod group state
	podGroupState, err := sched.PodGroupManager.PodGroupState(pInfo.Pod.Namespace, pInfo.Pod.Spec.SchedulingGroup)
	if err != nil {
		return nil, fmt.Errorf("error while retrieving pod group state: %w", err)
	}
	unscheduledPods := podGroupState.UnscheduledPods()

	podGroupInfo := &framework.QueuedPodGroupInfo{
		PodGroupInfo: &framework.PodGroupInfo{
			Namespace: pInfo.Pod.Namespace,
			Name:      *pInfo.Pod.Spec.SchedulingGroup.PodGroupName,
		},
		QueuedPodInfos: make([]*framework.QueuedPodInfo, 0, len(unscheduledPods)+1),
	}
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
// should use a new podSchedulingContext.
type podSchedulingContext struct {
	logger         klog.Logger
	state          *framework.CycleState
	podsToActivate *framework.PodsToActivate
}

// initPodSchedulingContext initializes the scheduling context of a single pod for pod group scheduling cycle.
func initPodSchedulingContext(ctx context.Context, pod *v1.Pod) *podSchedulingContext {
	logger := klog.FromContext(ctx)
	// TODO(knelasevero): Remove duplicated keys from log entry calls
	// When contextualized logging hits GA
	// https://github.com/kubernetes/kubernetes/issues/111672
	logger = klog.LoggerWithValues(logger, "pod", klog.KObj(pod))

	// Synchronously attempt to find a fit for the pod.
	state := framework.NewCycleState()
	// For the sake of performance, scheduler does not measure and export the scheduler_plugin_execution_duration metric
	// for every plugin execution in each scheduling cycle. Instead it samples a portion of scheduling cycles - percentage
	// determined by pluginMetricsSamplePercent. The line below helps to randomly pick appropriate scheduling cycles.
	state.SetRecordPluginMetrics(rand.Intn(100) < pluginMetricsSamplePercent)

	// Initialize an empty podsToActivate struct, which will be filled up by plugins or stay empty.
	podsToActivate := framework.NewPodsToActivate()
	state.Write(framework.PodsToActivateKey, podsToActivate)

	return &podSchedulingContext{
		logger:         logger,
		state:          state,
		podsToActivate: podsToActivate,
	}
}

// podGroupCycle runs a pod group scheduling cycle for the given pod group in a single cluster snapshot.
func (sched *Scheduler) podGroupCycle(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) {
	// Synchronously attempt to find a fit for the pod group.
	start := time.Now()

	logger := klog.FromContext(ctx)
	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		logger.Error(err, "Error updating snapshot", "podGroup", klog.KObj(podGroupInfo))
		result := podGroupAlgorithmResult{
			status: fwk.AsStatus(err),
		}
		sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupInfo, result, start)
		return
	}

	result := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupInfo)
	metrics.PodGroupSchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))

	sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupInfo, result, start)
}

// algorithmResult stores the scheduling result and status for a scheduling attempt of a single pod.
type algorithmResult struct {
	// scheduleResult is a scheduling algorithm result.
	scheduleResult ScheduleResult
	// podCtx is a specific pod scheduling context used for the scheduling algorithm.
	podCtx *podSchedulingContext
	// schedulingDuration is a pod scheduling duration used for metrics recording.
	schedulingDuration time.Duration
	// requiresPreemption determines whether this pod requires a preemption to proceed or not.
	requiresPreemption bool
	// status is a scheduling algorithm status.
	status *fwk.Status
	// permitStatus is a status of the permit check.
	// This is only set when the `status` is success or the `requiresPreemption` is true.
	permitStatus *fwk.Status
}

// podGroupAlgorithmResult stores the scheduling pod scheduling results for a pod group
// and any information needed to act on these results.
type podGroupAlgorithmResult struct {
	// podResults is the list of scheduling results for each pod in the group.
	// Only in the case of a pod group-wide Unschedulable or Error status can it contain fewer pods.
	podResults []algorithmResult
	// status is the final status of the pod group algorithm.
	//
	// Success code indicates that the pod group is schedulable and does not require any preemption.
	// Its feasible pods should be moved to the binding cycle.
	// This should only be set when the pod group is feasible and `waitingOnPreemption` is false.
	//
	// Unschedulable code indicates that the pod group is unschedulable,
	// and all its pods should be moved back to the scheduling queue as unschedulable.
	// Result with `waitingOnPreemption` set to true should have the Unschedulable status.
	//
	// Error code means that pod group scheduling failed due to an unexpected error,
	// and no pods will be scheduled this attempt.
	status *fwk.Status
	// waitingOnPreemption indicates whether this pod group requires or is waiting for preemption to complete.
	// This can only be set to true when the status is Unschedulable.
	waitingOnPreemption bool
}

// podGroupSchedulingDefaultAlgorithm runs the default algorithm for scheduling a pod group.
// It tries to schedule each pod using standard filtering and scoring logic in a fixed order.
// If a pod requires preemption to be schedulable, subsequent pods in the algorithm
// treat that pod as already scheduled on that node with victims being already removed in memory.
func (sched *Scheduler) podGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) podGroupAlgorithmResult {
	result := podGroupAlgorithmResult{
		podResults:          make([]algorithmResult, 0, len(podGroupInfo.QueuedPodInfos)),
		status:              fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable),
		waitingOnPreemption: false,
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running a pod group scheduling algorithm", "podGroup", klog.KObj(podGroupInfo), "unscheduledPodsCount", len(podGroupInfo.QueuedPodInfos))

	requiresPreemption := false
	for _, podInfo := range podGroupInfo.QueuedPodInfos {
		podResult, revertFn := sched.podGroupPodSchedulingAlgorithm(ctx, schedFwk, podGroupInfo, podInfo)
		result.podResults = append(result.podResults, podResult)
		if !podResult.status.IsSuccess() && !podResult.requiresPreemption {
			// When a pod is not feasible and doesn't require preemption, it means that it failed scheduling.
			if podResult.status.IsRejected() {
				// If the pod is rejected, the pod group can still be schedulable as long as the permit check can succeed.
				continue
			}
			// When the algorithm returns error or unexpected status, stop evaluating the rest of the pods.
			result.status = fwk.AsStatus(fmt.Errorf("failed to schedule other pod from a pod group: %w", podResult.status.AsError()))
			// Clear the waiting on preemption flag that could have been set by previous pods.
			result.waitingOnPreemption = false
			break
		}
		// At this point, the pod has passed the scheduling algorithm with the Permit status being either Success or Wait.
		// We unreserve the pod at the end of the whole algorithm (via defer) because it should be ultimately returned to the queue,
		// without binding it yet. We only assumed the pod to check feasibility of subsequent pods in the group.
		defer revertFn()

		requiresPreemption = requiresPreemption || podResult.requiresPreemption
		if podResult.permitStatus.IsSuccess() {
			// When the permit returns success for any pod, the pod group is schedulable.
			if requiresPreemption {
				// If any preemption is required, the whole pod group requires it to be feasible.
				result.status = fwk.NewStatus(fwk.Unschedulable, "pod group is waiting for preemption to complete").WithError(errPodGroupUnschedulable)
				// Set the waitingOnPreemption to true iff the pod group is feasible (Permit returned Success) and requires preemption.
				result.waitingOnPreemption = true
			} else {
				result.status = nil // Success
			}
		}
	}

	return result
}

// podGroupPodSchedulingAlgorithm runs a scheduling algorithm for individual pod from a pod group.
// It returns the algorithm result and, if successful or the preemption is required, the permit status together with the revert function.
func (sched *Scheduler) podGroupPodSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo, podInfo *framework.QueuedPodInfo) (algorithmResult, func()) {
	pod := podInfo.Pod
	podCtx := initPodSchedulingContext(ctx, pod)
	logger := podCtx.logger
	ctx = klog.NewContext(ctx, logger)
	start := time.Now()

	logger.V(4).Info("Attempting to schedule a pod belonging to a pod group", "podGroup", klog.KObj(podGroupInfo), "pod", klog.KObj(pod))

	requiresPreemption := false
	scheduleResult, status := sched.schedulingAlgorithm(ctx, podCtx.state, schedFwk, podInfo, start)
	if !status.IsSuccess() {
		if scheduleResult.nominatingInfo != nil && scheduleResult.nominatingInfo.NominatedNodeName != "" {
			// If the NominatedNodeName is set, the preemption is required.
			// Continue with assuming and reserving, because the subsequent pods from this group
			// have to see this one as already scheduled on its nominated place.
			// Set SuggestedHost to NominatedNodeName to handle the pod similarly to one that is feasible.
			scheduleResult.SuggestedHost = scheduleResult.nominatingInfo.NominatedNodeName
			requiresPreemption = true
		} else {
			// In case of pod being just unschedulable or having an error, just return now.
			return algorithmResult{
				scheduleResult:     scheduleResult,
				podCtx:             podCtx,
				schedulingDuration: time.Since(start),
				status:             status,
			}, nil
		}
	}
	assumedPodInfo, assumeStatus := sched.assumeAndReserve(ctx, podCtx.state, schedFwk, podInfo, scheduleResult)
	if !assumeStatus.IsSuccess() {
		return algorithmResult{
			scheduleResult:     ScheduleResult{nominatingInfo: clearNominatedNode},
			podCtx:             podCtx,
			schedulingDuration: time.Since(start),
			status:             assumeStatus,
		}, nil
	}

	revertFn := func() {
		err := sched.unreserveAndForget(ctx, podCtx.state, schedFwk, assumedPodInfo, scheduleResult.SuggestedHost)
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "ForgetPod failed")
		}
	}

	_, permitStatus := schedFwk.RunPermitPlugins(ctx, podCtx.state, assumedPodInfo.Pod, scheduleResult.SuggestedHost)
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
			scheduleResult:     ScheduleResult{nominatingInfo: clearNominatedNode},
			podCtx:             podCtx,
			schedulingDuration: time.Since(start),
			status:             permitStatus,
		}, nil
	}

	return algorithmResult{
		scheduleResult:     scheduleResult,
		podCtx:             podCtx,
		schedulingDuration: time.Since(start),
		status:             status,
		permitStatus:       permitStatus,
		requiresPreemption: requiresPreemption,
	}, revertFn
}

// submitPodGroupAlgorithmResult submits the result of the pod group scheduling algorithm.
// If that algorithm succedeed, the schedulable pods proceed to the binding cycle.
// Unschedulable pods are moved back to the scheduling queue and need to wait
// for the next pod group scheduling cycle.
// If the preemption is required for this pod group, all pods are moved back to the scheduling queue
// and require the next pod group scheduling cycle to verify the preemption outcome.
func (sched *Scheduler) submitPodGroupAlgorithmResult(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo, podGroupResult podGroupAlgorithmResult, start time.Time) {
	var scheduledPods, unschedulablePods int
	for i, pInfo := range podGroupInfo.QueuedPodInfos {
		var podResult algorithmResult
		if len(podGroupResult.podResults) > i {
			podResult = podGroupResult.podResults[i]
		} else {
			// In pod group-level unschedulable or error cases, podResult may not be defined.
			// Initialize it now to handle pod failure correctly.
			podResult = algorithmResult{
				podCtx: initPodSchedulingContext(ctx, pInfo.Pod),
				status: podGroupResult.status.Clone(),
			}
		}
		podCtx := podResult.podCtx
		ctx := klog.NewContext(ctx, podCtx.logger)
		// To be consistent with pod-by-pod scheduling, construct pod scheduling start time as `now - scheduling duration`.
		podSchedulingStart := time.Now().Add(-podResult.schedulingDuration)

		if podGroupResult.status.IsError() {
			if podResult.status.IsError() {
				// If this exact pod failed with an error, use its status instead.
				sched.FailureHandler(ctx, schedFwk, pInfo, podResult.status, clearNominatedNode, podSchedulingStart)
				continue
			}
			// Pod group failed with an error. Reject all pods with its status.
			sched.FailureHandler(ctx, schedFwk, pInfo, podGroupResult.status, clearNominatedNode, podSchedulingStart)
			continue
		}
		if podResult.status.IsSuccess() {
			nominatingInfo := &fwk.NominatingInfo{
				NominatingMode:    fwk.ModeOverride,
				NominatedNodeName: podResult.scheduleResult.SuggestedHost,
			}
			switch {
			case podGroupResult.status.IsSuccess():
				// Pod no longer needs a pod group scheduling cycle. Setting it to false to disable any checks in further functions.
				pInfo.NeedsPodGroupScheduling = false
				// Schedule result is applied for pod and its binding cycle executes.
				assumedPodInfo, status := sched.prepareForBindingCycle(ctx, podCtx.state, schedFwk, pInfo, podCtx.podsToActivate, podResult.scheduleResult)
				if !status.IsSuccess() {
					// In such unlikely situation just reject this pod.
					sched.FailureHandler(ctx, schedFwk, pInfo, status, clearNominatedNode, podSchedulingStart)
					unschedulablePods++
					continue
				}
				go sched.runBindingCycle(ctx, podCtx.state, schedFwk, podResult.scheduleResult, assumedPodInfo, podSchedulingStart, podCtx.podsToActivate)
				scheduledPods++
			case podGroupResult.status.IsRejected():
				if podGroupResult.waitingOnPreemption {
					// Pod has to come back to the scheduling queue as unschedulable, waiting for preemption to complete.
					sched.FailureHandler(ctx, schedFwk, pInfo, podGroupResult.status, nominatingInfo, podSchedulingStart)
				} else {
					// Pod group is unschedulable, so the pod has to be marked as unschedulable.
					// Its rejection status is set to its permit status message.
					status := fwk.NewStatus(fwk.Unschedulable, podResult.permitStatus.Message()).WithError(errPodGroupUnschedulable)
					sched.FailureHandler(ctx, schedFwk, pInfo, status, clearNominatedNode, podSchedulingStart)
				}
				unschedulablePods++
			default:
				err := fmt.Errorf("received unexpected pod group scheduling algorithm status code: %s", podGroupResult.status.Code())
				sched.FailureHandler(ctx, schedFwk, pInfo, fwk.AsStatus(err), clearNominatedNode, podSchedulingStart)
				unschedulablePods++
			}
		} else {
			// TBD: Add a message to status if the pod used features for which finding a placement cannot be guaranteed,
			// such as heterogeneous pod group or using inter-pod dependencies.
			if podResult.requiresPreemption && !podGroupResult.waitingOnPreemption {
				// Pod group is unschedulable, so the pod has to be marked as unschedulable, even if it just required preemption.
				// Its rejection status is set to its permit status message, as the preemption message is no longer relevant.
				status := fwk.NewStatus(fwk.Unschedulable, podResult.permitStatus.Message()).WithError(errPodGroupUnschedulable)
				sched.FailureHandler(ctx, schedFwk, pInfo, status, clearNominatedNode, podSchedulingStart)
			} else {
				// When a pod is unschedulable or preemption is required, just call the FailureHandler.
				sched.FailureHandler(ctx, schedFwk, pInfo, podResult.status, podResult.scheduleResult.nominatingInfo, podSchedulingStart)
			}
			unschedulablePods++
		}
	}
	logger := klog.FromContext(ctx)
	switch {
	case podGroupResult.status.IsSuccess():
		logger.V(2).Info("Successfully scheduled a pod group", "podGroup", klog.KObj(podGroupInfo), "scheduledPods", scheduledPods, "unschedulablePods", unschedulablePods)
		metrics.PodGroupScheduled(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
	case podGroupResult.status.IsRejected():
		if podGroupResult.waitingOnPreemption {
			logger.V(2).Info("Pod group is waiting for preemption", "podGroup", klog.KObj(podGroupInfo), "unschedulablePods", unschedulablePods)
			metrics.PodGroupWaitingOnPreemption(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		} else {
			logger.V(2).Info("Unable to schedule a pod group", "podGroup", klog.KObj(podGroupInfo), "unschedulablePods", unschedulablePods)
			metrics.PodGroupUnschedulable(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		}
	default:
		utilruntime.HandleErrorWithContext(ctx, podGroupResult.status.AsError(), "Error scheduling pod group", "podGroup", klog.KObj(podGroupInfo), "errorPods", len(podGroupInfo.QueuedPodInfos))
		metrics.PodGroupScheduleError(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
	}
}

// placementResult associates pod group algorithm result with the placement.
// The placement information can be used by the placement score plugins.
type placementResult struct {
	podGroupAlgorithmResult
	placement *fwk.Placement
}

// podGroupSchedulingPlacementAlgorithm tries several different combinations for scheduling the pod group and selects the best one.
// First it runs placement generator plugins to create a list of placements.
// Placement is a set of nodes that will be considered when scheduling a pod group.
// Then for each placement it tries to schedule the pod group through podGroupSchedulingDefaultAlgorithm.
// Finally, it runs placement scorer plugins to select the best placement.
func (sched *Scheduler) podGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) podGroupAlgorithmResult {
	logger := klog.FromContext(ctx)
	allNodes, err := sched.nodeInfoSnapshot.NodeInfos().List()
	if err != nil {
		return podGroupAlgorithmResult{
			status: fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}
	}

	// TODO: kubernetes/enhancements#5732 - run placement generator plugins to get the set of placements
	placements := []*fwk.Placement{
		{
			Nodes: allNodes,
		},
	}

	results := make([]placementResult, len(placements))
	successfulResults := make([]placementResult, 0, len(placements))

	for i, placement := range placements {
		logger.V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
		err := sched.nodeInfoSnapshot.AssumePlacement(placement)
		if err != nil {
			return podGroupAlgorithmResult{
				status: fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
			}
		}
		result := sched.podGroupSchedulingDefaultAlgorithm(ctx, schedFwk, podGroupInfo)
		sched.nodeInfoSnapshot.ForgetPlacement()
		if result.status.IsError() {
			return result
		}

		results[i] = placementResult{
			podGroupAlgorithmResult: result,
			placement:               placement,
		}

		if result.status.IsSuccess() || result.waitingOnPreemption {
			successfulResults = append(successfulResults, results[i])
		}
	}

	if len(successfulResults) == 0 {
		// We need to send events and set the status for pods in case all simulations were infeasible.
		// The selection of which simulation we report is arbitrary for now, but may change in the future.
		return results[0].podGroupAlgorithmResult
	}

	// TODO: kubernetes/enhancements#5732 - run placement scorer plugins to select the best placement
	return successfulResults[0].podGroupAlgorithmResult
}

// podGroupSchedulingAlgorithm attempts to schedule pods in the pod group according to the policy and constraints and returns the scheduling result for each pod in the pod group.
func (sched *Scheduler) podGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) podGroupAlgorithmResult {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		return sched.podGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupInfo)
	} else {
		return sched.podGroupSchedulingDefaultAlgorithm(podGroupCycleCtx, schedFwk, podGroupInfo)
	}
}
