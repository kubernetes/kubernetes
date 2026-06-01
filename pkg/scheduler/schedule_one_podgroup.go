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
	"time"

	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
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
		err := sched.SchedulingQueue.AddAttemptedPodGroupIfNeeded(logger, podGroupInfo, sched.SchedulingQueue.SchedulingCycle())
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Failed to pod group back to scheduling queue", "podGroup", klog.KObj(podGroupInfo))
		}
		return
	}
	sched.skipPodGroupPodSchedule(ctx, schedFwk, podGroupInfo)
	// skipPodGroupPodSchedule could remove some pods from the pod group.
	// Pod group constraints will be re-evaluated on a PlacementFeasible phase.
	// Now, verify if it has any pods left.
	if len(podGroupInfo.QueuedPodInfos) == 0 {
		return
	}

	logger.V(3).Info("Attempting to schedule pod group", "podGroup", klog.KObj(podGroupInfo))

	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo)
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

// podSchedulingContext holds the precomputed data needed to handle the pod scheduling.
// Each scheduling attempt in the same pod group scheduling cycle for the same pod
// should use a new podSchedulingContext.
type podSchedulingContext struct {
	logger         klog.Logger
	state          *framework.CycleState
	podsToActivate *framework.PodsToActivate
}

// initPodSchedulingContext initializes the scheduling context of a single pod for pod group scheduling cycle.
func initPodSchedulingContext(ctx context.Context, pod *v1.Pod, placementCycleState *framework.CycleState, postFilterMode podGroupPostFilterMode) *podSchedulingContext {
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

	podGroupCycleState := placementCycleState.GetPodGroupSchedulingCycle()
	// Marks this cycle as a pod group scheduling cycle.
	state.SetPodGroupSchedulingCycle(podGroupCycleState)
	// Set the placement cycle state so per-pod plugins can access placement-scoped data.
	state.SetPlacementCycleState(placementCycleState)

	// Skip post filters if requested.
	switch postFilterMode {
	case runWithoutPostFilters:
		state.SetSkipAllPostFilterPlugins(true)
	case runAllPostFilters:
		// Default podCtx state will run all post filters.
	}

	return &podSchedulingContext{
		logger:         logger,
		state:          state,
		podsToActivate: podsToActivate,
	}
}

// podGroupCycle runs a pod group scheduling cycle for the given pod group in a single cluster snapshot.
func (sched *Scheduler) podGroupCycle(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo) {
	// Synchronously attempt to find a fit for the pod group.
	start := time.Now()

	logger := klog.FromContext(ctx)
	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		logger.Error(err, "Error updating snapshot", "podGroup", klog.KObj(podGroupInfo))
		result := podGroupAlgorithmResult{
			status: fwk.AsStatus(err),
		}
		// Ensure podResults has an entry for each pod in the pod group with Error status.
		result = completePodGroupAlgorithmResult(ctx, podGroupInfo, podGroupCycleState, runAllPostFilters, result)
		sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, podGroupInfo, result, start)
		return
	}

	result := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, podGroupInfo, runAllPostFilters)

	// Ensure podResults has an entry for each pod in the pod group with a status.
	result = completePodGroupAlgorithmResult(ctx, podGroupInfo, podGroupCycleState, runAllPostFilters, result)
	metrics.PodGroupSchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))

	// Run workload aware preemption if required. If the preemption is successful,
	// we need to put the pods from pod group back into the scheduling queue.
	if sched.workloadAwarePreemptionEnabled && result.status.Code() == fwk.Unschedulable {
		pgPostFilterResult, status := sched.runWorkloadAwarePreemption(ctx, schedFwk, podGroupCycleState, podGroupInfo)
		if status.IsSuccess() {
			result.waitingOnPreemption = true
			for i := range result.podResults {
				if nodeNameInfo, ok := pgPostFilterResult.NominatedNodeNames[result.podResults[i].pod]; ok {
					result.podResults[i].scheduleResult.nominatingInfo = nodeNameInfo
				}
			}
		} else if status.IsError() {
			result.status = status
		} else {
			result.status.AppendReason(status.String())
		}
	}

	sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, podGroupInfo, result, start)
}

// runWorkloadAwarePreemption runs workload-aware preemption for the given pod group.
// It saves the current snapshot of node infos, runs a PodGroupPostFilter
// which modifies the node infos to check feasibility of the
// pod group scheduling with some pods removed and reverts the snapshot to the
// original state.
// The function used for evaluating feasibility of pod group scheduling is
// scheduler.podGroupSchedulingAlgorithm run without any post filters.
func (sched *Scheduler) runWorkloadAwarePreemption(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo) (*framework.PodGroupPostFilterResult, *fwk.Status) {
	// Default preemption should be the only pod group post filter registered plugin.
	plugins := schedFwk.PodGroupPostFilterPlugins()
	if len(plugins) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "default preemption plugin is not registered, workload aware preemption is disabled")
	}

	pg, err := schedFwk.SharedInformerFactory().Scheduling().V1alpha3().PodGroups().Lister().PodGroups(podGroupInfo.Namespace).Get(podGroupInfo.Name)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to get pod group object: %w", err))
	}
	if pg.Spec.SchedulingConstraints != nil && len(pg.Spec.SchedulingConstraints.Topology) > 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "workload aware preemption is not supported for pod groups with scheduling constraints")
	}

	restoreFn, err := sched.nodeInfoSnapshot.BackupSnapshot()
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to backup snapshot: %w", err))
	}
	defer restoreFn()

	var pgSchedulingFunc framework.PodGroupSchedulingFunc = func(_ context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
		res := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, podGroupInfo, runWithoutPostFilters)
		return &fwk.PodGroupAssignments{
			// We do not fill the Placement struct, because we do not need it.
			ProposedAssignments: makeProposedAssignments(&res),
		}, res.status
	}
	return plugins[0].PodGroupPostFilter(ctx, pg, podGroupInfo.UnscheduledPods, pgSchedulingFunc)
}

// algorithmResult stores the scheduling result and status for a scheduling attempt of a single pod.
type algorithmResult struct {
	// pod is the pod the result applies to.
	pod *v1.Pod
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
}

// podGroupPostFilterMode defines how the pod group algorithm should run post filters plugins.
type podGroupPostFilterMode int

const (
	// The pod group algorithm should try to run all post filters in pod-by-pod cycle.
	runAllPostFilters podGroupPostFilterMode = iota
	// The pod group algorithm should not try post filter at all. This can be used
	// by workload aware preemption that tries to check if after removing some
	// pods the pod group can be scheduled.
	runWithoutPostFilters
)

func (ar *algorithmResult) GetPod() *v1.Pod {
	return ar.pod
}

func (ar *algorithmResult) GetNodeName() string {
	return ar.scheduleResult.SuggestedHost
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
	// placementCycleState is the state with which this placement was processed.
	placementCycleState fwk.PlacementCycleState
}

// podGroupSchedulingDefaultAlgorithm runs the default algorithm for scheduling a pod group.
// It tries to schedule each pod using standard filtering and scoring logic in a fixed order.
// If a pod requires preemption to be schedulable, subsequent pods in the algorithm
// treat that pod as already scheduled on that node with victims being already removed in memory.
func (sched *Scheduler) podGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo, postFilterMode podGroupPostFilterMode) podGroupAlgorithmResult {
	result := podGroupAlgorithmResult{
		podResults:          make([]algorithmResult, 0, len(podGroupInfo.QueuedPodInfos)),
		status:              fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable),
		waitingOnPreemption: false,
		placementCycleState: placementCycleState,
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running a pod group scheduling algorithm", "podGroup", klog.KObj(podGroupInfo), "unscheduledPodsCount", len(podGroupInfo.QueuedPodInfos))

	requiresPreemption := false
	anyScheduled := false
	for _, podInfo := range podGroupInfo.QueuedPodInfos {
		podResult, revertFn := sched.podGroupPodSchedulingAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, podInfo, postFilterMode)
		result.podResults = append(result.podResults, podResult)
		if revertFn != nil {
			// We unreserve the pod at the end of the whole algorithm (via defer) because it should be ultimately returned to the queue,
			// without binding it yet. We only assumed the pod to check feasibility of subsequent pods in the group.
			defer revertFn()
		}

		if !podResult.status.IsSuccess() && !podResult.status.IsRejected() {
			// When the algorithm returns error or unexpected status, stop evaluating the rest of the pods.
			result.status = fwk.AsStatus(fmt.Errorf("failed to schedule other pod from a pod group: %w", podResult.status.AsError()))
			break
		}

		// PlacementFeasible plugins check if the pod group can meet its constraints.
		// Those plugins need to be run after each pod is scheduled.
		placementFeasibleStatus := schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo)

		if placementFeasibleStatus.IsError() {
			// When the algorithm returns error or unexpected status, stop evaluating the rest of the pods.
			result.status = fwk.AsStatus(fmt.Errorf("failed to evaluate placement feasibility: %w", placementFeasibleStatus.AsError()))
			break
		}

		// UnschedulableAndUnresolvable from PlacementFeasible plugins indicates that the pod group
		// cannot meet its constraints regardless of how many more pods we check.
		// We can stop the scheduling loop early.
		if placementFeasibleStatus.Code() == fwk.UnschedulableAndUnresolvable {
			// We need to change the code to Unschedulable to make sure preemption can be fired.
			result.status = fwk.NewStatus(fwk.Unschedulable).WithError(placementFeasibleStatus.AsError())
			break
		}

		result.status = placementFeasibleStatus
		requiresPreemption = requiresPreemption || podResult.requiresPreemption
		anyScheduled = anyScheduled || podResult.status.IsSuccess()
	}

	if result.status.IsSuccess() {
		if requiresPreemption {
			// If any preemption is required, the whole pod group requires it to be feasible.
			result.status = fwk.NewStatus(fwk.Unschedulable, "pod group is waiting for preemption to complete").WithError(errPodGroupUnschedulable)
			result.waitingOnPreemption = true
		} else if !anyScheduled {
			// The framework requires at least 1 pod to be scheduled in order to return a success status.
			result.status = fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable)
		}
	}

	return result
}

// podGroupPodSchedulingAlgorithm runs a scheduling algorithm for individual pod from a pod group.
// It returns the algorithm result together with the revert function.
func (sched *Scheduler) podGroupPodSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo, podInfo *framework.QueuedPodInfo, postFilterMode podGroupPostFilterMode) (algorithmResult, func()) {
	pod := podInfo.Pod
	podCtx := initPodSchedulingContext(ctx, pod, placementCycleState, postFilterMode)
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
				pod:                pod,
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
			pod:                pod,
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

	return algorithmResult{
		pod:                pod,
		scheduleResult:     scheduleResult,
		podCtx:             podCtx,
		schedulingDuration: time.Since(start),
		status:             status,
		requiresPreemption: requiresPreemption,
	}, revertFn
}

// completePodGroupAlgorithmResult ensures that the podGroupAlgorithmResult contains the same number of podResults as there are pods in QueuedPodInfos.
func completePodGroupAlgorithmResult(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, podGroupState *framework.CycleState, postFilterMode podGroupPostFilterMode, podGroupResult podGroupAlgorithmResult) podGroupAlgorithmResult {
	numInResult := len(podGroupResult.podResults)
	numInQueue := len(podGroupInfo.QueuedPodInfos)
	if numInResult == numInQueue {
		return podGroupResult
	}
	newResults := make([]algorithmResult, numInQueue)
	copy(newResults, podGroupResult.podResults)
	for i := numInResult; i < numInQueue; i++ {
		pInfo := podGroupInfo.QueuedPodInfos[i]
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupSchedulingCycle(podGroupState)
		newResults[i] = algorithmResult{
			pod:    pInfo.Pod,
			podCtx: initPodSchedulingContext(ctx, pInfo.Pod, placementCycleState, postFilterMode),
			status: podGroupResult.status.Clone(),
		}
	}
	podGroupResult.podResults = newResults
	return podGroupResult
}

// submitPodGroupAlgorithmResult submits the result of the pod group scheduling algorithm.
// It assumes that podGroupResult contains results for all pods from the pod group,
// if it does not, podGroupCondition will be updated to reflect the error.
// If that algorithm succedeed, the schedulable pods proceed to the binding cycle.
// Unschedulable pods are moved back to the scheduling queue and need to wait
// for the next pod group scheduling cycle.
// If the preemption is required for this pod group, all pods are moved back to the scheduling queue
// and require the next pod group scheduling cycle to verify the preemption outcome.
func (sched *Scheduler) submitPodGroupAlgorithmResult(ctx context.Context, schedFwk framework.Framework, podGroupState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo, podGroupResult podGroupAlgorithmResult, start time.Time) {
	logger := klog.FromContext(ctx)

	if len(podGroupResult.podResults) != len(podGroupInfo.QueuedPodInfos) {
		// This should never happen, but if it does, complete the result with the error status.
		logger.Error(fmt.Errorf("some pods were not processed"), "scheduling error for pod group", "podGroup", klog.KObj(podGroupInfo))
		podGroupResult.status = fwk.NewStatus(fwk.Error, "scheduling error for pod group, some pods were not processed")
		podGroupResult.podResults = nil
		podGroupResult = completePodGroupAlgorithmResult(ctx, podGroupInfo, podGroupState, runAllPostFilters, podGroupResult)
	}
	var scheduledPods, unschedulablePods int
	for i, pInfo := range podGroupInfo.QueuedPodInfos {
		podResult := podGroupResult.podResults[i]
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
				// Disable pod group scheduling in cycle state before binding.
				podCtx.state.SetPodGroupSchedulingCycle(nil)
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
					// Its rejection status is set to the pod group's status message.
					status := fwk.NewStatus(fwk.Unschedulable, podGroupResult.status.Message()).WithError(errPodGroupUnschedulable)
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
				// Its rejection status is set to the pod group's status message, as the preemption message is no longer relevant.
				status := fwk.NewStatus(fwk.Unschedulable, podGroupResult.status.Message()).WithError(errPodGroupUnschedulable)
				sched.FailureHandler(ctx, schedFwk, pInfo, status, clearNominatedNode, podSchedulingStart)
			} else {
				// When a pod is unschedulable or preemption is required, just call the FailureHandler.
				sched.FailureHandler(ctx, schedFwk, pInfo, podResult.status, podResult.scheduleResult.nominatingInfo, podSchedulingStart)
			}
			unschedulablePods++
		}
	}

	var condition *metav1.Condition
	switch {
	case podGroupResult.status.IsSuccess():
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupScheduled,
			Status:  metav1.ConditionTrue,
			Reason:  "Scheduled",
			Message: podGroupResult.status.Message(),
		}
		logger.V(2).Info("Successfully scheduled a pod group", "podGroup", klog.KObj(podGroupInfo), "scheduledPods", scheduledPods, "unschedulablePods", unschedulablePods)
		metrics.PodGroupScheduled(schedFwk.ProfileName(), metrics.SinceInSeconds(start))

	case podGroupResult.status.IsRejected():
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.PodGroupReasonUnschedulable,
			Message: podGroupResult.status.Message(),
		}
		if podGroupResult.waitingOnPreemption {
			logger.V(2).Info("Pod group is waiting for preemption", "podGroup", klog.KObj(podGroupInfo), "unschedulablePods", unschedulablePods, "err", podGroupResult.status.Message())
			metrics.PodGroupWaitingOnPreemption(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		} else {
			logger.V(2).Info("Unable to schedule a pod group", "podGroup", klog.KObj(podGroupInfo), "unschedulablePods", unschedulablePods, "err", podGroupResult.status.Message())
			metrics.PodGroupUnschedulable(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		}

	default:
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.PodGroupReasonSchedulerError,
			Message: podGroupResult.status.AsError().Error(),
		}
		utilruntime.HandleErrorWithContext(ctx, podGroupResult.status.AsError(), "Error scheduling pod group", "podGroup", klog.KObj(podGroupInfo), "errorPods", len(podGroupInfo.QueuedPodInfos))
		metrics.PodGroupScheduleError(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
	}
	sched.updatePodGroupCondition(ctx, podGroupInfo, condition)

	err := sched.SchedulingQueue.AddAttemptedPodGroupIfNeeded(logger, podGroupInfo, sched.SchedulingQueue.SchedulingCycle())
	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Failed to add attempted pod group to scheduling queue", "podGroup", klog.KObj(podGroupInfo))
	}
}

// updatePodGroupCondition patches the given condition on a PodGroup.
func (sched *Scheduler) updatePodGroupCondition(ctx context.Context,
	podGroupInfo *framework.QueuedPodGroupInfo, condition *metav1.Condition) {
	logger := klog.FromContext(ctx)

	// If the PodGroup was already successfully scheduled, don't regress the
	// condition back to False on a subsequent cycle for extra pods.
	pg, err := sched.podGroupLister.PodGroups(podGroupInfo.Namespace).Get(podGroupInfo.Name)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get PodGroup for status update", "podGroup", klog.KObj(podGroupInfo))
		return
	}

	existing := apimeta.FindStatusCondition(pg.Status.Conditions, condition.Type)
	if existing != nil && existing.Status == metav1.ConditionTrue && condition.Status != metav1.ConditionTrue {
		return
	}

	condition.ObservedGeneration = pg.Generation
	newStatus := pg.Status.DeepCopy()
	if !apimeta.SetStatusCondition(&newStatus.Conditions, *condition) {
		return
	}

	if err := util.PatchPodGroupStatus(ctx, sched.client, podGroupInfo.Name, podGroupInfo.Namespace, &pg.Status, newStatus); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to update PodGroup status", "podGroup", klog.KObj(podGroupInfo))
	}
}

// podGroupSchedulingPlacementAlgorithm tries several different combinations for scheduling the pod group and selects the best one.
// First it runs placement generator plugins to create a list of placements.
// Placement is a set of nodes that will be considered when scheduling a pod group.
// Then for each placement it tries to schedule the pod group through podGroupSchedulingDefaultAlgorithm.
// Finally, it runs placement scorer plugins to select the best placement.
func (sched *Scheduler) podGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo, postFilterMode podGroupPostFilterMode) podGroupAlgorithmResult {
	logger := klog.FromContext(ctx)
	allNodes, err := sched.nodeInfoSnapshot.NodeInfos().List()
	if err != nil {
		return podGroupAlgorithmResult{
			status: fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}
	}

	// For now, always record plugin metrics until we understand its impact on performance.
	podGroupCycleState.SetRecordPluginMetrics(true)
	placements, status := schedFwk.RunPlacementGeneratePlugins(ctx, podGroupCycleState, podGroupInfo.PodGroupInfo, allNodes)
	if !status.IsSuccess() {
		return podGroupAlgorithmResult{
			status: status,
		}
	}

	var anyResult *podGroupAlgorithmResult
	successfulResults := make(map[*fwk.Placement]*podGroupAlgorithmResult)

	for _, placement := range placements {
		logger.V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
		err := sched.nodeInfoSnapshot.AssumePlacement(placement)
		if err != nil {
			return podGroupAlgorithmResult{
				status: fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
			}
		}
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
		result := sched.podGroupSchedulingDefaultAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, postFilterMode)
		sched.nodeInfoSnapshot.ForgetPlacement()
		if result.status.IsError() {
			return result
		}

		if anyResult == nil {
			anyResult = &result
		}

		if result.status.IsSuccess() || result.waitingOnPreemption {
			successfulResults[placement] = &result
		}
	}

	if len(successfulResults) == 0 {
		// We need to send events and set the status for pods in case all simulations were infeasible.
		// The selection of which simulation we report is arbitrary for now, but may change in the future.
		anyResult.status = fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("0/%d placements are available, first placement status: %v", len(placements), anyResult.status.AsError()))
		return *anyResult
	}

	if len(successfulResults) == 1 {
		for _, result := range successfulResults {
			return *result
		}
	}

	bestPlacement, status := sched.findBestPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, successfulResults)
	if !status.IsSuccess() {
		return podGroupAlgorithmResult{
			status: status,
		}
	}

	return *successfulResults[bestPlacement]
}

// findBestPlacement uses PlacementScore plugins to determine the best placement based on the scheduling results.
func (sched *Scheduler) findBestPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.QueuedPodGroupInfo, successfulResults map[*fwk.Placement]*podGroupAlgorithmResult) (*fwk.Placement, *fwk.Status) {
	placementPodGroupAssignments, placementStates := makePodGroupAssignments(successfulResults)

	scores, status := schedFwk.RunPlacementScorePlugins(ctx, podGroupCycleState, podGroupInfo, placementPodGroupAssignments, placementStates)
	if !status.IsSuccess() {
		return nil, status
	}

	for i := range scores {
		scores[i].Randomizer = rand.Int()
	}

	loggerVTen := klog.FromContext(ctx).V(10)
	if loggerVTen.Enabled() {
		for _, score := range scores {
			for _, pluginScore := range score.Scores {
				loggerVTen.Info("Plugin scored placement for podGroup", "podGroup", klog.KObj(podGroupInfo), "plugin", pluginScore.Name, "placement", score.Placement.Name, "score", pluginScore.Score)
			}
			loggerVTen.Info("Calculated placement's final score for podGroup", "podGroup", klog.KObj(podGroupInfo), "placement", score.Placement.Name, "score", score.TotalScore)
		}
	}

	bestScore := &scores[0]
	for _, score := range scores[1:] {
		if score.TotalScore > bestScore.TotalScore ||
			score.TotalScore == bestScore.TotalScore &&
				score.Randomizer > bestScore.Randomizer {
			bestScore = &score
		}
	}
	return bestScore.Placement, nil
}

func makePodGroupAssignments(successfulResults map[*fwk.Placement]*podGroupAlgorithmResult) ([]*fwk.PodGroupAssignments, []fwk.PlacementCycleState) {
	placementPodGroupAssignments := make([]*fwk.PodGroupAssignments, 0, len(successfulResults))
	placementStates := make([]fwk.PlacementCycleState, 0, len(successfulResults))
	for placement, result := range successfulResults {
		proposedAssignments := makeProposedAssignments(result)
		placementPodGroupAssignments = append(placementPodGroupAssignments, &fwk.PodGroupAssignments{
			Placement:           placement,
			ProposedAssignments: proposedAssignments,
		})
		placementStates = append(placementStates, result.placementCycleState)
	}
	return placementPodGroupAssignments, placementStates
}

// makeProposedAssignments builds a list of proposedAssignments from the result of a pod group scheduling attempt.
func makeProposedAssignments(res *podGroupAlgorithmResult) []fwk.ProposedAssignment {
	proposedAssignments := make([]fwk.ProposedAssignment, 0)
	for _, podRes := range res.podResults {
		if podRes.GetNodeName() != "" {
			proposedAssignments = append(proposedAssignments, &podRes)
		}
	}
	return proposedAssignments
}

// podGroupSchedulingAlgorithm attempts to schedule pods in the pod group according to the policy and constraints and returns the scheduling result for all evaluated pods in the pod group, not necessarily all pods in the pod group.
func (sched *Scheduler) podGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.QueuedPodGroupInfo, postFilterMode podGroupPostFilterMode) podGroupAlgorithmResult {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		return sched.podGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupCycleState, podGroupInfo, postFilterMode)
	}
	// The non-TAS default algorithm does not evaluate placement candidates, but it
	// still runs in a single implicit placement context so placement-scoped
	// extension points can use the same state plumbing as TAS.
	placementCycleState := framework.NewCycleState()
	placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
	return sched.podGroupSchedulingDefaultAlgorithm(podGroupCycleCtx, schedFwk, placementCycleState, podGroupInfo, postFilterMode)
}
