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
	"time"

	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	schedulingapi "k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/utils/ptr"
)

// scheduleOnePodGroup does the entire workload-aware scheduling workflow for a single pod group.
func (sched *Scheduler) scheduleOnePodGroup(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo) {
	logger := klog.FromContext(ctx)
	// TODO(knelasevero): Remove duplicated keys from log entry calls
	// When contextualized logging hits GA
	// https://github.com/kubernetes/kubernetes/issues/111672
	logger = klog.LoggerWithValues(logger, "podGroupType", podGroupInfo.GetType(), "podGroup", klog.KObj(podGroupInfo))
	ctx = klog.NewContext(ctx, logger)
	start := time.Now()

	if err := sched.Cache.UpdateSnapshot(logger, sched.nodeInfoSnapshot); err != nil {
		logger.Error(err, "Error updating snapshot")
		sched.handlePodGroupFailureBeforeScheduling(ctx, podGroupInfo, err)
		return
	}

	// PodGroupInfo popped from the queue can have older (Composite)PodGroup objects.
	// Override it here with the snapshotted version to ensure consistency throughout the cycle.
	if err := sched.reconcilePodGroupWithSnapshot(podGroupInfo.PodGroupInfo); err != nil {
		// It can happen that the hierarchy was popped from the scheduling queue before it observed the change of shape.
		// (Composite)PodGroup should come back to the scheduling queue.
		sched.handlePodGroupFailureBeforeScheduling(ctx, podGroupInfo, err)
		return
	}
	if err := sched.validatePodGroup(podGroupInfo); err != nil {
		sched.handlePodGroupFailureBeforeScheduling(ctx, podGroupInfo, err)
		return
	}

	schedFwk := sched.frameworkForPodGroup(podGroupInfo)
	sched.skipPodGroupPodSchedule(ctx, schedFwk, podGroupInfo)
	// skipPodGroupPodSchedule could remove some pods from the pod group.
	// Pod group constraints will be re-evaluated on a PlacementFeasible phase.
	// Now, verify if it has any pods left.
	if !podGroupInfo.HasQueuedPodInfos() {
		// Finish the in-flight attempt so members that arrived while these pods were
		// being skipped can be requeued instead of remaining pending indefinitely.
		if err := sched.SchedulingQueue.AddAttemptedPodGroupIfNeeded(logger, podGroupInfo, sched.SchedulingQueue.SchedulingCycle(), fwk.NewStatus(fwk.Success)); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Failed to finish skipped pod group scheduling attempt", "podGroup", klog.KObj(podGroupInfo))
		}
		return
	}

	logger.V(3).Info("Attempting to schedule pod group", "podGroupType", podGroupInfo.GetType(), "podGroup", klog.KObj(podGroupInfo))

	sched.podGroupCycle(ctx, schedFwk, framework.NewCycleState(), podGroupInfo, start)
}

// reconcilePodGroupWithSnapshot overrides the objects with the snapshotted versions to ensure consistency throughout the cycle.
// This is needed because PodGroupInfo popped from the queue can have older PodGroup/CompositePodGroup objects.
// Any differences in the hierarchy shape (added or removed subtrees) will result in error.
func (sched *Scheduler) reconcilePodGroupWithSnapshot(pgi *framework.PodGroupInfo) error {
	if pgi.GetType() == fwk.CompositePodGroupKeyType {
		compositePodGroup, err := sched.nodeInfoSnapshot.CompositePodGroups().Get(pgi.GetNamespace(), pgi.GetName())
		if err != nil {
			return err
		}
		if pgi.CompositePodGroup != nil && !ptr.Equal(compositePodGroup.Spec.ParentCompositePodGroupName, pgi.CompositePodGroup.Spec.ParentCompositePodGroupName) {
			return fmt.Errorf("different parent in composite pod group between snapshot (%s) and queued entity (%s)",
				ptr.Deref(compositePodGroup.Spec.ParentCompositePodGroupName, "[unset]"),
				ptr.Deref(pgi.CompositePodGroup.Spec.ParentCompositePodGroupName, "[unset]"))
		}
		cpgs, err := sched.nodeInfoSnapshot.CompositePodGroupStates().Get(pgi.GetNamespace(), pgi.GetName())
		if err != nil {
			return err
		}
		cpgsChildren := cpgs.GetChildren()
		if len(cpgsChildren) != len(pgi.Children) {
			return fmt.Errorf("different number of children in composite pod group between snapshot (%d) and queued entity (%d)", len(cpgsChildren), len(pgi.Children))
		}
		for i := range pgi.Children {
			err := sched.reconcilePodGroupWithSnapshot(pgi.Children[i])
			if err != nil {
				return err
			}
		}
		pgi.GenericPodGroup = fwk.NewGenericCompositePodGroup(compositePodGroup)
	} else {
		podGroup, err := sched.nodeInfoSnapshot.PodGroups().Get(pgi.GetNamespace(), pgi.GetName())
		if err != nil {
			return err
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) &&
			!ptr.Equal(podGroup.Spec.ParentCompositePodGroupName, pgi.PodGroup.Spec.ParentCompositePodGroupName) {
			return fmt.Errorf("different parent in pod group between snapshot (%s) and queued entity (%s)",
				ptr.Deref(podGroup.Spec.ParentCompositePodGroupName, "[unset]"),
				ptr.Deref(pgi.PodGroup.Spec.ParentCompositePodGroupName, "[unset]"))
		}
		pgi.GenericPodGroup = fwk.NewGenericPodGroup(podGroup)
	}
	return nil
}

// handlePodGroupFailureBeforeScheduling handles the failure of a (composite) pod group that occurred before scheduling.
func (sched *Scheduler) handlePodGroupFailureBeforeScheduling(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, err error) {
	logger := klog.FromContext(ctx)
	for podInfo := range podGroupInfo.ForEachPodInfo() {
		podFwk, podFwkErr := sched.frameworkForPod(podInfo.Pod)
		if podFwkErr != nil {
			// This shouldn't happen, because we only accept for scheduling the pods
			// which specify a scheduler name that matches one of the profiles.
			logger.Error(podFwkErr, "Error occurred")
			sched.SchedulingQueue.Done(podInfo.Pod.UID)
			continue
		}
		sched.FailureHandler(ctx, podFwk, podInfo, fwk.AsStatus(err), clearNominatedNode, time.Now())
	}
	sched.updatePodGroupConditionWithError(ctx, podGroupInfo.PodGroupInfo, err)
	err = sched.SchedulingQueue.AddAttemptedPodGroupIfNeeded(logger, podGroupInfo, sched.SchedulingQueue.SchedulingCycle(), fwk.AsStatus(err))
	if err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Failed to add pod group back to scheduling queue", "podGroupType", podGroupInfo.GetType(), "podGroup", klog.KObj(podGroupInfo))
	}
}

func (sched *Scheduler) updatePodGroupConditionWithError(ctx context.Context, pgi *framework.PodGroupInfo, err error) {
	if pgi.PodGroup != nil {
		sched.updatePodGroupCondition(ctx, pgi, &metav1.Condition{
			Type:    schedulingapi.PodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.PodGroupReasonSchedulerError,
			Message: err.Error(),
		})
		return
	}
	sched.updateCompositePodGroupCondition(ctx, pgi, &metav1.Condition{
		Type:    schedulingapi.CompositePodGroupInitiallyScheduled,
		Status:  metav1.ConditionFalse,
		Reason:  schedulingapi.CompositePodGroupReasonSchedulerError,
		Message: err.Error(),
	})
	for _, child := range pgi.GetChildGroups() {
		sched.updatePodGroupConditionWithError(ctx, child, err)
	}
}

// validatePodGroup ensures that:
// - all Pods in a group hierarchy have matching scheduler name,
// - all entities in a group hierarchy have the same priority as the root group,
// - all entities in a group hierarchy have the same preemption policy.
func (sched *Scheduler) validatePodGroup(rootInfo *framework.QueuedPodGroupInfo) error {
	rootPriority := rootInfo.GetPriority()

	var rootPreemptionPolicy v1.PreemptionPolicy
	if sched.podGroupPreemptionPolicyEnabled {
		rootPreemptionPolicy = rootInfo.GetPreemptionPolicy()
	}
	schedulerName := ""
	firstPodName := ""
	for pInfo := range rootInfo.ForEachPodInfo() {
		schedulerName = pInfo.Pod.Spec.SchedulerName
		if !sched.podGroupPreemptionPolicyEnabled {
			rootPreemptionPolicy = util.PodPreemptionPolicy(pInfo.Pod)
		}
		firstPodName = pInfo.Pod.Name
		break
	}

	validatePodGroup := func(pgi *framework.PodGroupInfo) error {
		pgPriority := pgi.GetPriority()
		if pgPriority != rootPriority {
			return fmt.Errorf("all pod groups in a hierarchy should have the same priority as the root pod group's priority, got %d (%q) and %d (%q)",
				pgPriority, pgi.GetKey(), rootPriority, rootInfo.GetKey())
		}

		if sched.podGroupPreemptionPolicyEnabled {
			pgPreemptionPolicy := pgi.GetPreemptionPolicy()
			if pgPreemptionPolicy != rootPreemptionPolicy {
				return fmt.Errorf("all pod groups in a hierarchy should have the same preemption policy as the root pod group's preemption policy, got %v (%q) and %v (%q)",
					pgPreemptionPolicy, pgi.GetKey(), rootPreemptionPolicy, rootInfo.GetKey())
			}
		}
		return nil
	}

	validatePod := func(pod *v1.Pod) error {
		if pod.Spec.SchedulerName != schedulerName {
			return fmt.Errorf("all pods in a pod group hierarchy should have the same .spec.schedulerName set, got: %q (%q) and %q (%q)",
				pod.Spec.SchedulerName, pod.Name, schedulerName, firstPodName)
		}
		podPriority := corev1helpers.PodPriority(pod)
		if podPriority != rootPriority {
			return fmt.Errorf("all pods in a pod group hierarchy should have the same priority as the root pod group's priority, got %d (%q) and %d (%q)",
				podPriority, pod.Name, rootPriority, rootInfo.GetKey())
		}

		podPreemptionPolicy := util.PodPreemptionPolicy(pod)
		if podPreemptionPolicy != rootPreemptionPolicy {
			if sched.podGroupPreemptionPolicyEnabled {
				// If the PodGroupPreemptionPolicy feature is enabled, validate that the pod's preemption policy
				// matches the root group's preemption policy.
				return fmt.Errorf("all pods in a pod group hierarchy should have the same preemption policy as the root pod group's preemption policy, got %v (%q) and %v (%q)",
					podPreemptionPolicy, pod.Name, rootPreemptionPolicy, rootInfo.GetKey())
			} else {
				// If the PodGroupPreemptionPolicy feature is disabled, the preemption policy is determined by the first pod in the group.
				// Validate that preemption policy is the same across all pods in the pod group.
				return fmt.Errorf("all pods in a pod group hierarchy should have the same preemption policy, got %v (%q) and %v (%q)",
					podPreemptionPolicy, pod.Name, rootPreemptionPolicy, firstPodName)
			}
		}
		return nil
	}

	if err := sched.validatePodGroupHierarchy(rootInfo.PodGroupInfo, validatePodGroup, validatePod); err != nil {
		return err
	}

	if _, ok := sched.Profiles[schedulerName]; !ok {
		return fmt.Errorf("profile not found for scheduler name %q", schedulerName)
	}

	return nil
}

// validatePodGroupHierarchy recursively validates that all entities in the hierarchy
// (composite pod groups, pod groups, and both unscheduled and scheduled pods)
// conform to the group-wide constraints.
func (sched *Scheduler) validatePodGroupHierarchy(podGroupInfo *framework.PodGroupInfo, validatePodGroup func(pgi *framework.PodGroupInfo) error, validatePod func(pod *v1.Pod) error) error {
	if err := validatePodGroup(podGroupInfo); err != nil {
		return err
	}

	if podGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		for _, child := range podGroupInfo.GetChildGroups() {
			if err := sched.validatePodGroupHierarchy(child, validatePodGroup, validatePod); err != nil {
				return err
			}
		}
		return nil
	}

	for _, pod := range podGroupInfo.UnscheduledPods {
		if err := validatePod(pod); err != nil {
			return err
		}
	}
	podGroupState, err := sched.nodeInfoSnapshot.PodGroupStates().Get(podGroupInfo.GetNamespace(), podGroupInfo.GetName())
	if err != nil {
		return fmt.Errorf("failed to get pod group state: %w", err)
	}
	for _, pod := range podGroupState.ScheduledPods() {
		if err := validatePod(pod); err != nil {
			return err
		}
	}
	return nil
}

// frameworkForPodGroup obtains the concrete scheduler framework for the entire pod group.
// Assumes [Scheduler.validatePodGroup] has been called before.
func (sched *Scheduler) frameworkForPodGroup(podGroupInfo *framework.QueuedPodGroupInfo) framework.Framework {
	for pInfo := range podGroupInfo.ForEachPodInfo() {
		return sched.Profiles[pInfo.Pod.Spec.SchedulerName]
	}
	return nil
}

// skipPodGroupPodSchedule skips the scheduling of particular pods from the group when they should no longer be considered.
// This can happen when the pod is already being deleted (i.e., when its deletionTimestamp is set)
// or when the pod has already been assumed.
func (sched *Scheduler) skipPodGroupPodSchedule(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) {
	var podsToRemove []*v1.Pod
	for pInfo := range podGroupInfo.ForEachPodInfo() {
		if sched.skipPodSchedule(ctx, schedFwk, pInfo.Pod) {
			// We don't put this Pod back to the queue, but we have to cleanup the in-flight pods/events.
			sched.SchedulingQueue.Done(pInfo.Pod.UID)
			podsToRemove = append(podsToRemove, pInfo.Pod)
		}
	}
	// The following code removes each pod from the QueuedPodGroupInfo object individually.
	// This simplified approach may result in O(n^2) complexity. However, since
	// it is unlikely that any given pod will be filtered out using skipPodGroupPodSchedule,
	// such complexity is acceptable.
	for _, pod := range podsToRemove {
		podGroupInfo.RemovePod(pod)
	}
}

// podGroupCycle runs a pod group scheduling cycle for the given pod group.
// Cluster state should be snapshotted before calling this method.
func (sched *Scheduler) podGroupCycle(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo, start time.Time) {
	pgResults := sched.podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo)
	rootStatus := pgResults[rootPodGroupInfo.PodGroupInfo.GetKey()].status
	var completePGResults map[fwk.EntityKey]*PodGroupScheduledResult
	if rootPodGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		completePGResults = completeCompositePodGroupAlgorithmResult(ctx, rootPodGroupInfo, podGroupCycleState, pgResults)
	} else {
		// pgResults has exactly 1 element.
		queuedPodInfos := rootPodGroupInfo.PodInfosForGroup(rootPodGroupInfo.PodGroupInfo.GetKey())
		result := completePodGroupAlgorithmResult(ctx, queuedPodInfos, podGroupCycleState, pgResults[rootPodGroupInfo.PodGroupInfo.GetKey()])
		completePGResults = map[fwk.EntityKey]*PodGroupScheduledResult{rootPodGroupInfo.PodGroupInfo.GetKey(): result}
	}

	metrics.PodGroupSchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))

	// Run pod group post filter plugins if scheduling failed. If any of the plugins is successful,
	// we need to put the pods from pod group back into the scheduling queue.
	if rootStatus.Code() == fwk.Unschedulable {
		var pgSchedulingFunc fwk.PodGroupSchedulingFunc = func(ctx context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
			results := sched.podGroupAlgorithm.RunRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), rootPodGroupInfo)
			proposedAssignments := make([]fwk.ProposedAssignment, 0)
			for _, res := range results {
				proposedAssignments = append(proposedAssignments, makeProposedAssignments(res)...)
			}
			return &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			}, results[rootPodGroupInfo.PodGroupInfo.GetKey()].status
		}
		pgPostFilterResult, status := schedFwk.RunPodGroupPostFilterPlugins(ctx, podGroupCycleState, rootPodGroupInfo.PodGroupInfo, pgSchedulingFunc)
		applyPodGroupPostFilterResult(completePGResults, pgPostFilterResult, status)
	}

	sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo, completePGResults, start, rootStatus)
}

// completePodGroupAlgorithmResult ensures that the podGroupAlgorithmResult contains the same number of podResults as there are queued pods in the pod group.
func completePodGroupAlgorithmResult(ctx context.Context, queuedPodInfos []*framework.QueuedPodInfo, podGroupState *framework.CycleState, podGroupResult *PodGroupScheduledResult) *PodGroupScheduledResult {
	numInResult := len(podGroupResult.podResults)
	numInQueue := len(queuedPodInfos)
	if numInResult == numInQueue {
		return podGroupResult
	}
	newResults := make([]algorithmResult, numInQueue)
	copy(newResults, podGroupResult.podResults)
	for i := numInResult; i < numInQueue; i++ {
		pInfo := queuedPodInfos[i]
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupCycleState(podGroupState)
		newResults[i] = algorithmResult{
			podInfo: pInfo,
			podCtx:  initPodSchedulingContext(ctx, pInfo.Pod, placementCycleState),
			status:  podGroupResult.status.Clone(),
		}
	}
	podGroupResult.podResults = newResults
	return podGroupResult
}

// completeCompositePodGroupAlgorithmResult post-processes scheduling results for a composite pod group.
// It ensures that every pod in every subgroup has a fully populated status and that failure statuses
// are propagated down the tree before finalizing the cycle.
func completeCompositePodGroupAlgorithmResult(ctx context.Context, rootPodGroupInfo *framework.QueuedPodGroupInfo, rootCycleState *framework.CycleState, pgResults map[fwk.EntityKey]*PodGroupScheduledResult) map[fwk.EntityKey]*PodGroupScheduledResult {
	completeCompositePodGroupAlgorithmResultMap(ctx, rootPodGroupInfo.PodGroupInfo, pgResults, &PodGroupScheduledResult{})
	for pgKey, queuedPodInfos := range rootPodGroupInfo.ForEachGroupAndPodInfos() {
		pgResult := pgResults[pgKey]
		// Ensure podResults has an entry for each pod in the pod group with a status.
		completePodGroupAlgorithmResult(ctx, queuedPodInfos, rootCycleState, pgResult)
	}
	return pgResults
}

// completeCompositePodGroupAlgorithmResultMap propagates scheduling failures from parents to children.
// This is necessary because child pod groups cannot be committed or bound if their parent composite
// pod group fails to meet its scheduling requirements.
func completeCompositePodGroupAlgorithmResultMap(ctx context.Context, podGroupInfo *framework.PodGroupInfo, pgResults map[fwk.EntityKey]*PodGroupScheduledResult, parentResult *PodGroupScheduledResult) {
	key := podGroupInfo.GetKey()
	result, ok := pgResults[key]
	if !ok {
		// In case the pod group wasn't processed, create the result and set its status to parent.
		result = &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       parentResult.status.Clone(),
		}
		pgResults[key] = result
	} else if !parentResult.status.IsSuccess() && result.status.IsSuccess() {
		// When a parent composite pod group fails, any child that previously succeeded during its own evaluation
		// must be invalidated with the parent's failure status to prevent its pods from proceeding to binding.
		// Preserve the old result, but just overwrite the status.
		result.status = parentResult.status.Clone()
	} else if parentResult.status.IsError() && !result.status.IsError() {
		// In case of an error, overwrite the status with an error.
		result.status = parentResult.status.Clone()
	}
	if podGroupInfo.CompositePodGroup != nil {
		for _, child := range podGroupInfo.GetChildGroups() {
			completeCompositePodGroupAlgorithmResultMap(ctx, child, pgResults, result)
		}
	}
}

// applyPodGroupPostFilterResult updates the final scheduling results of the pod group hierarchy
// based on the outcome of the PodGroupPostFilter plugin execution. It ensures that preemption
// nominations are properly registered so they can reclaim resources, and that scheduling failures,
// errors, and preemption states are propagated down the hierarchy (including composite groups)
// to prevent invalid bindings and ensure accurate root-level metrics.
func applyPodGroupPostFilterResult(completePGResults map[fwk.EntityKey]*PodGroupScheduledResult, pgPostFilterResult *fwk.PodGroupPostFilterResult, status *fwk.Status) {
	if status.IsError() {
		for _, pgResult := range completePGResults {
			pgResult.status = status.Clone()
		}
		return
	}
	if status.IsSuccess() {
		// Post-filter plugins successfully identified preemption candidates.
		// Mark all pod groups in the hierarchy as waiting on preemption.
		// Also associate nominated nodes with individual pods in the leaf groups to preserve placement decisions.
		for _, pgResult := range completePGResults {
			pgResult.waitingOnPreemption = true
			if pgResult.podGroupInfo.CompositePodGroup != nil {
				continue
			}
			for j := range pgResult.podResults {
				pod := pgResult.podResults[j].podInfo.Pod
				namespacedName := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
				if nodeNameInfo, ok := pgPostFilterResult.NominatingInfos[namespacedName]; ok {
					pgResult.podResults[j].scheduleResult.nominatingInfo = nodeNameInfo
				}
			}
		}
	}
	msg := status.Message()
	if msg == "" {
		return
	}
	for _, pgResult := range completePGResults {
		pgResult.status.AppendReason(msg)
	}
}

// submitPodGroupAlgorithmResult submits the result of the pod group scheduling algorithm.
// It assumes that podGroupResult contains results for all pods from the pod group,
// if it does not, podGroupCondition will be updated to reflect the error.
// If that algorithm succedeed, the schedulable pods proceed to the binding cycle.
// Unschedulable pods are moved back to the scheduling queue and need to wait
// for the next pod group scheduling cycle.
// If the preemption is required for this pod group, all pods are moved back to the scheduling queue
// and require the next pod group scheduling cycle to verify the preemption outcome.
//
// TODO(#142269): Group statuses are updated sequentially and synchronously on the main
// scheduling goroutine. Offload or parallelize these updates for larger hierarchies to avoid
// head-of-line blocking in the scheduling queue.
func (sched *Scheduler) submitPodGroupAlgorithmResult(ctx context.Context, schedFwk framework.Framework, podGroupState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo, podGroupResults map[fwk.EntityKey]*PodGroupScheduledResult, start time.Time, rootStatus *fwk.Status) {
	logger := klog.FromContext(ctx)

	for _, podGroupResult := range podGroupResults {
		pgi := podGroupResult.podGroupInfo
		if pgi.CompositePodGroup != nil {
			sched.submitCompositePodGroupResult(ctx, podGroupResult)
			continue
		}
		queuedPodInfos := rootPodGroupInfo.PodInfosForGroup(pgi.GetKey())
		sched.submitPodGroupResult(ctx, schedFwk, podGroupState, queuedPodInfos, podGroupResult)
	}

	rootResult := podGroupResults[rootPodGroupInfo.PodGroupInfo.GetKey()]
	switch {
	case rootResult.status.IsSuccess():
		metrics.PodGroupScheduled(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
	case rootResult.status.IsRejected():
		if rootResult.waitingOnPreemption {
			metrics.PodGroupWaitingOnPreemption(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		} else {
			metrics.PodGroupUnschedulable(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
		}
	default:
		metrics.PodGroupScheduleError(schedFwk.ProfileName(), metrics.SinceInSeconds(start))
	}

	if err := sched.SchedulingQueue.AddAttemptedPodGroupIfNeeded(logger, rootPodGroupInfo, sched.SchedulingQueue.SchedulingCycle(), rootStatus); err != nil {
		utilruntime.HandleErrorWithContext(ctx, err, "Failed to add attempted pod group to scheduling queue", rootPodGroupInfo.Type, klog.KObj(rootPodGroupInfo))
	}
}

// submitCompositePodGroupResult updates the CompositePodGroup status condition based on the scheduling outcome.
func (sched *Scheduler) submitCompositePodGroupResult(ctx context.Context, podGroupResult *PodGroupScheduledResult) {
	logger := klog.FromContext(ctx)
	pgi := podGroupResult.podGroupInfo
	var condition *metav1.Condition
	switch {
	case podGroupResult.status.IsSuccess():
		condition = &metav1.Condition{
			Type:    schedulingapi.CompositePodGroupInitiallyScheduled,
			Status:  metav1.ConditionTrue,
			Reason:  schedulingapi.CompositePodGroupReasonScheduled,
			Message: podGroupResult.status.Message(),
		}
		logger.V(2).Info("Successfully scheduled a composite pod group", "compositePodGroup", klog.KObj(pgi))

	case podGroupResult.status.IsRejected():
		condition = &metav1.Condition{
			Type:    schedulingapi.CompositePodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.CompositePodGroupReasonUnschedulable,
			Message: podGroupResult.status.Message(),
		}
		if podGroupResult.waitingOnPreemption {
			logger.V(2).Info("Composite pod group is waiting for preemption", "compositePodGroup", klog.KObj(pgi), "err", podGroupResult.status.Message())
		} else {
			logger.V(2).Info("Unable to schedule a composite pod group", "compositePodGroup", klog.KObj(pgi), "err", podGroupResult.status.Message())
		}

	default:
		condition = &metav1.Condition{
			Type:    schedulingapi.CompositePodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.CompositePodGroupReasonSchedulerError,
			Message: podGroupResult.status.Message(),
		}
		utilruntime.HandleErrorWithContext(ctx, podGroupResult.status.AsError(), "Error scheduling composite pod group", "compositePodGroup", klog.KObj(pgi))
	}
	sched.updateCompositePodGroupCondition(ctx, pgi, condition)
}

// submitPodGroupResult processes scheduling results for a leaf PodGroup and its member pods.
// Schedulable pods proceed to their binding cycles if the PodGroup succeeded, while unschedulable,
// preempting, or errored pods are passed to the FailureHandler for requeueing.
func (sched *Scheduler) submitPodGroupResult(ctx context.Context, schedFwk framework.Framework, podGroupState *framework.CycleState, queuedPodInfos []*framework.QueuedPodInfo, podGroupResult *PodGroupScheduledResult) {
	logger := klog.FromContext(ctx)
	pgi := podGroupResult.podGroupInfo

	if len(podGroupResult.podResults) != len(queuedPodInfos) {
		// This should never happen, but if it does, complete the result with the error status.
		logger.Error(fmt.Errorf("some pods were not processed"), "scheduling error for pod group", "podGroup", klog.KObj(pgi))
		podGroupResult.status = fwk.NewStatus(fwk.Error, "scheduling error for pod group, some pods were not processed")
		podGroupResult.podResults = nil
		completePodGroupAlgorithmResult(ctx, queuedPodInfos, podGroupState, podGroupResult)
	}
	var scheduledPods, unschedulablePods int
	for i, pInfo := range queuedPodInfos {
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
			switch {
			case podGroupResult.status.IsSuccess():
				// Disable pod group scheduling in cycle state before binding.
				podCtx.state.SetPodGroupCycleState(nil)
				podCtx.state.SetPlacementCycleState(nil)
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
					sched.FailureHandler(ctx, schedFwk, pInfo, podGroupResult.status.Clone(), podResult.scheduleResult.nominatingInfo, podSchedulingStart)
				} else {
					// Pod group is unschedulable, so the pod has to be marked as unschedulable.
					// Its rejection status is set to the pod group's status message.
					sched.FailureHandler(ctx, schedFwk, pInfo, podGroupResult.status.Clone(), clearNominatedNode, podSchedulingStart)
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
			// When a pod is unschedulable or preemption is required, just call the FailureHandler.
			sched.FailureHandler(ctx, schedFwk, pInfo, podResult.status, podResult.scheduleResult.nominatingInfo, podSchedulingStart)
			unschedulablePods++
		}
	}

	var condition *metav1.Condition
	switch {
	case podGroupResult.status.IsSuccess():
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupInitiallyScheduled,
			Status:  metav1.ConditionTrue,
			Reason:  schedulingapi.PodGroupReasonScheduled,
			Message: podGroupResult.status.Message(),
		}
		logger.V(2).Info("Successfully scheduled a pod group", "podGroup", klog.KObj(pgi), "scheduledPods", scheduledPods, "unschedulablePods", unschedulablePods)

	case podGroupResult.status.IsRejected():
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.PodGroupReasonUnschedulable,
			Message: podGroupResult.status.Message(),
		}
		if podGroupResult.waitingOnPreemption {
			logger.V(2).Info("Pod group is waiting for preemption", "podGroup", klog.KObj(pgi), "unschedulablePods", unschedulablePods, "err", podGroupResult.status.Message())
		} else {
			logger.V(2).Info("Unable to schedule a pod group", "podGroup", klog.KObj(pgi), "unschedulablePods", unschedulablePods, "err", podGroupResult.status.Message())
		}

	default:
		condition = &metav1.Condition{
			Type:    schedulingapi.PodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  schedulingapi.PodGroupReasonSchedulerError,
			Message: podGroupResult.status.Message(),
		}
		utilruntime.HandleErrorWithContext(ctx, podGroupResult.status.AsError(), "Error scheduling pod group", "podGroup", klog.KObj(pgi), "errorPods", len(queuedPodInfos))
	}
	sched.updatePodGroupCondition(ctx, pgi, condition)
}

// updatePodGroupCondition patches the given condition on a PodGroup.
func (sched *Scheduler) updatePodGroupCondition(ctx context.Context,
	podGroupInfo *framework.PodGroupInfo, condition *metav1.Condition) {
	logger := klog.FromContext(ctx)

	// Get the newest object from cache to ensure the update below serves on the newest object possible.
	pg, err := sched.Cache.PodGroups().Get(podGroupInfo.GetNamespace(), podGroupInfo.GetName())
	if err != nil {
		return
	}
	// If the PodGroup was already successfully scheduled, don't regress the
	// condition back to False on a subsequent cycle for extra pods.
	existing := apimeta.FindStatusCondition(pg.Status.Conditions, condition.Type)
	if existing != nil && existing.Status == metav1.ConditionTrue && condition.Status != metav1.ConditionTrue {
		return
	}

	condition.ObservedGeneration = pg.Generation
	newStatus := pg.Status.DeepCopy()
	if !apimeta.SetStatusCondition(&newStatus.Conditions, *condition) {
		return
	}

	if err := util.PatchPodGroupStatus(ctx, sched.client, podGroupInfo.GetName(), podGroupInfo.GetNamespace(), &pg.Status, newStatus); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to update PodGroup status", "podGroup", klog.KObj(podGroupInfo))
	}
}

// updateCompositePodGroupCondition patches the given condition on a CompositePodGroup.
func (sched *Scheduler) updateCompositePodGroupCondition(ctx context.Context,
	podGroupInfo *framework.PodGroupInfo, condition *metav1.Condition) {
	logger := klog.FromContext(ctx)

	// Get the newest object from cache to ensure the update below serves on the newest object possible.
	cpg, err := sched.Cache.CompositePodGroups().Get(podGroupInfo.GetNamespace(), podGroupInfo.GetName())
	if err != nil {
		return
	}
	// If the CompositePodGroup was already successfully scheduled, don't regress the
	// condition back to False on a subsequent cycle for extra pods.
	existing := apimeta.FindStatusCondition(cpg.Status.Conditions, condition.Type)
	if existing != nil && existing.Status == metav1.ConditionTrue && condition.Status != metav1.ConditionTrue {
		return
	}

	condition.ObservedGeneration = cpg.Generation
	newStatus := cpg.Status.DeepCopy()
	if !apimeta.SetStatusCondition(&newStatus.Conditions, *condition) {
		return
	}

	if err := util.PatchCompositePodGroupStatus(ctx, sched.client, podGroupInfo.GetName(), podGroupInfo.GetNamespace(), &cpg.Status, newStatus); err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to update CompositePodGroup status", "compositePodGroup", klog.KObj(podGroupInfo))
	}
}
