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
	"errors"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
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
	if err := sched.validatePodGroupHierarchy(podGroupInfo); err != nil {
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
		snapshotChildren := cpgs.GetChildren()
		queuedChildren := pgi.Children()
		if len(snapshotChildren) != len(queuedChildren) {
			return fmt.Errorf("different number of children in composite pod group between snapshot (%d) and queued entity (%d)", len(snapshotChildren), len(queuedChildren))
		}
		for _, child := range queuedChildren {
			if err := sched.reconcilePodGroupWithSnapshot(child); err != nil {
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
		pgReason := schedulingapi.PodGroupReasonSchedulerError
		if _, ok := errors.AsType[*validationError](err); ok {
			pgReason = schedulingapi.PodGroupReasonPodGroupError
		}
		sched.updatePodGroupCondition(ctx, pgi, &metav1.Condition{
			Type:    schedulingapi.PodGroupInitiallyScheduled,
			Status:  metav1.ConditionFalse,
			Reason:  pgReason,
			Message: err.Error(),
		})
		return
	}
	cpgReason := schedulingapi.CompositePodGroupReasonSchedulerError
	if _, ok := errors.AsType[*validationError](err); ok {
		cpgReason = schedulingapi.CompositePodGroupReasonCompositePodGroupError
	}
	sched.updateCompositePodGroupCondition(ctx, pgi, &metav1.Condition{
		Type:    schedulingapi.CompositePodGroupInitiallyScheduled,
		Status:  metav1.ConditionFalse,
		Reason:  cpgReason,
		Message: err.Error(),
	})
	for _, child := range pgi.Children() {
		sched.updatePodGroupConditionWithError(ctx, child, err)
	}
}

// validatePodGroupHierarchy ensures that:
// - the hierarchy does not exceed WorkloadMaxTreeDepth (and has no CompositePodGroup at depth WorkloadMaxTreeDepth),
// - all groups in the hierarchy reference the same Workload as the root group,
// - no gang parent group has a basic child group,
// - no parent group with All disruption mode has a child group with Single disruption mode,
// - all Pods in a group hierarchy have matching scheduler name,
// - all entities in a group hierarchy have the same priority as the root group,
// - all entities in a group hierarchy have the same preemption policy.
func (sched *Scheduler) validatePodGroupHierarchy(rootInfo *framework.QueuedPodGroupInfo) error {
	rootPriority := rootInfo.GetPriority()
	rootWorkloadName := rootInfo.GetWorkloadName()

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

	validateChildPodGroup := func(pgi, parent *framework.PodGroupInfo) error {
		if parent == nil {
			// It's a root, so we can skip the checks against itself.
			return nil
		}

		if pgWorkloadName := pgi.GetWorkloadName(); pgWorkloadName != rootWorkloadName {
			return newValidationErrorf("pod group workload does not match root workload, got: %q (%q) and %q (%q)",
				pgWorkloadName, pgi.GetKey(), rootWorkloadName, rootInfo.GetKey())
		}

		pgPriority := pgi.GetPriority()
		if pgPriority != rootPriority {
			return newValidationErrorf("all pod groups in a hierarchy should have the same priority as the root pod group's priority, got %d (%q) and %d (%q)",
				pgPriority, pgi.GetKey(), rootPriority, rootInfo.GetKey())
		}

		if sched.podGroupPreemptionPolicyEnabled {
			pgPreemptionPolicy := pgi.GetPreemptionPolicy()
			if pgPreemptionPolicy != rootPreemptionPolicy {
				return newValidationErrorf("all pod groups in a hierarchy should have the same preemption policy as the root pod group's preemption policy, got %v (%q) and %v (%q)",
					pgPreemptionPolicy, pgi.GetKey(), rootPreemptionPolicy, rootInfo.GetKey())
			}
		}

		if parent.IsGang() && pgi.IsBasic() {
			return newValidationErrorf("gang parent %q cannot have basic child %q", parent.GetKey(), pgi.GetKey())
		}

		if parent.HasDisruptionModeAll() && pgi.HasDisruptionModeSingle() {
			return newValidationErrorf("parent %q with All disruption mode cannot have child %q with Single disruption mode", parent.GetKey(), pgi.GetKey())
		}
		return nil
	}

	validatePod := func(pod *v1.Pod) error {
		if pod.Spec.SchedulerName != schedulerName {
			return newValidationErrorf("all pods in a pod group hierarchy should have the same .spec.schedulerName set, got: %q (%q) and %q (%q)",
				pod.Spec.SchedulerName, pod.Name, schedulerName, firstPodName)
		}
		podPriority := corev1helpers.PodPriority(pod)
		if podPriority != rootPriority {
			return newValidationErrorf("all pods in a pod group hierarchy should have the same priority as the root pod group's priority, got %d (%q) and %d (%q)",
				podPriority, pod.Name, rootPriority, rootInfo.GetKey())
		}

		podPreemptionPolicy := util.PodPreemptionPolicy(pod)
		if podPreemptionPolicy != rootPreemptionPolicy {
			if sched.podGroupPreemptionPolicyEnabled {
				// If the PodGroupPreemptionPolicy feature is enabled, validate that the pod's preemption policy
				// matches the root group's preemption policy.
				return newValidationErrorf("all pods in a pod group hierarchy should have the same preemption policy as the root pod group's preemption policy, got %v (%q) and %v (%q)",
					podPreemptionPolicy, pod.Name, rootPreemptionPolicy, rootInfo.GetKey())
			} else {
				// If the PodGroupPreemptionPolicy feature is disabled, the preemption policy is determined by the first pod in the group.
				// Validate that preemption policy is the same across all pods in the pod group.
				return newValidationErrorf("all pods in a pod group hierarchy should have the same preemption policy, got %v (%q) and %v (%q)",
					podPreemptionPolicy, pod.Name, rootPreemptionPolicy, firstPodName)
			}
		}
		return nil
	}

	if err := sched.validatePodGroupSubtree(rootInfo.PodGroupInfo, nil /* parent */, 1 /* depth */, validateChildPodGroup, validatePod); err != nil {
		return err
	}

	if _, ok := sched.Profiles[schedulerName]; !ok {
		return fmt.Errorf("profile not found for scheduler name %q", schedulerName)
	}

	return nil
}

// validatePodGroupSubtree recursively validates that all entities in the hierarchy
// (composite pod groups, pod groups, and both unscheduled and scheduled pods)
// conform to the group-wide constraints.
func (sched *Scheduler) validatePodGroupSubtree(
	podGroupInfo, parent *framework.PodGroupInfo,
	depth int,
	validateChildPodGroup func(pgi, parent *framework.PodGroupInfo) error,
	validatePod func(pod *v1.Pod) error,
) error {
	// CompositePodGroup nodes cannot be at depth WorkloadMaxTreeDepth as they have to contain a PodGroup as a child,
	// breaking the depth constraint.
	if podGroupInfo.GetType() == fwk.CompositePodGroupKeyType && depth == schedulingv1beta1.WorkloadMaxTreeDepth {
		return newValidationErrorf("hierarchy depth exceeds maximum allowed depth %d at %q", schedulingv1beta1.WorkloadMaxTreeDepth, podGroupInfo.GetKey())
	}

	if err := validateChildPodGroup(podGroupInfo, parent); err != nil {
		return err
	}

	if podGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		for _, child := range podGroupInfo.Children() {
			if err := sched.validatePodGroupSubtree(child, podGroupInfo, depth+1, validateChildPodGroup, validatePod); err != nil {
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
// Assumes [Scheduler.validatePodGroupHierarchy] has been called before.
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

	metrics.PodGroupAlgorithmLatency(schedFwk.ProfileName(), metrics.EntityTypeToLabel(rootPodGroupInfo.Type()), metrics.SinceInSeconds(start))

	// Run pod group post filter plugins if scheduling failed. If any of the plugins is successful,
	// we need to put the pods from pod group back into the scheduling queue.
	if rootStatus.Code() == fwk.Unschedulable {
		pgSchedulingFunc := sched.podGroupAlgorithm.BuildPodGroupSchedulingFunc(schedFwk, rootPodGroupInfo)
		pgPostFilterResult, status := schedFwk.RunPodGroupPostFilterPlugins(ctx, podGroupCycleState, rootPodGroupInfo.PodGroupInfo, pgSchedulingFunc)
		applyPodGroupPostFilterResult(completePGResults, pgPostFilterResult, status)
	}

	sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo, completePGResults, start, rootStatus)
}

// completePodGroupAlgorithmResult ensures that the PodGroupScheduledResult contains the same number of podResults as there are queued pods in the pod group.
func completePodGroupAlgorithmResult(ctx context.Context, queuedPodInfos []*framework.QueuedPodInfo, podGroupState *framework.CycleState, podGroupResult *PodGroupScheduledResult) *PodGroupScheduledResult {
	numInResult := len(podGroupResult.podResults)
	numInQueue := len(queuedPodInfos)
	if numInResult == numInQueue {
		return podGroupResult
	}
	skippedPodStatus := podGroupResult.status
	if skippedPodStatus.IsError() {
		skippedPodStatus = buildParentPodGroupErrorStatus(podGroupResult.podGroupInfo.GetName())
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
			status:  skippedPodStatus.Clone(),
		}
	}
	podGroupResult.podResults = newResults
	return podGroupResult
}

// completeCompositePodGroupAlgorithmResult post-processes scheduling results for a composite pod group.
// It ensures that every pod in every subgroup has a fully populated status and that failure statuses
// are propagated down the tree before finalizing the cycle.
func completeCompositePodGroupAlgorithmResult(ctx context.Context, rootPodGroupInfo *framework.QueuedPodGroupInfo, rootCycleState *framework.CycleState, pgResults map[fwk.EntityKey]*PodGroupScheduledResult) map[fwk.EntityKey]*PodGroupScheduledResult {
	completeCompositePodGroupAlgorithmResultMap(rootPodGroupInfo.PodGroupInfo, pgResults, nil)
	for pgKey, queuedPodInfos := range rootPodGroupInfo.ForEachGroupAndPodInfos() {
		pgResult := pgResults[pgKey]
		// Ensure podResults has an entry for each pod in the pod group with a status.
		completePodGroupAlgorithmResult(ctx, queuedPodInfos, rootCycleState, pgResult)
	}
	return pgResults
}

// completeCompositePodGroupAlgorithmResultMap propagates failures of composite pod groups to their
// descendants, so that no group is bound while one of its ancestors failed:
//   - A group that succeeded or was not evaluated gets a status derived from its nearest failed
//     ancestor (see buildAncestorFailureStatus).
//   - A group that failed on its own keeps its status, unless the ancestor failed with an Error
//     and the group did not, in which case the ancestor's Error takes precedence.
//   - A group that keeps its own failure becomes the nearest failed ancestor of its subtree.
func completeCompositePodGroupAlgorithmResultMap(
	podGroupInfo *framework.PodGroupInfo,
	pgResults map[fwk.EntityKey]*PodGroupScheduledResult,
	failedAncestor *PodGroupScheduledResult,
) {
	key := podGroupInfo.GetKey()
	result, evaluated := pgResults[key]
	if !evaluated {
		// Groups are only skipped after an ancestor failed.
		result = &PodGroupScheduledResult{podGroupInfo: podGroupInfo}
		pgResults[key] = result
	}
	if failedAncestor != nil {
		if !evaluated || result.status.IsSuccess() {
			// The group can't be bound while its ancestor failed.
			result.status = buildAncestorFailureStatus(failedAncestor)
		} else if failedAncestor.status.IsError() && !result.status.IsError() {
			// An error takes precedence over the group's own non-error failure.
			result.status = buildAncestorFailureStatus(failedAncestor)
		} else {
			// The group's own failure is the more specific reason, so its subtree inherits it.
			failedAncestor = result
		}
	} else if !result.status.IsSuccess() {
		// The subtree inherits the group's own failure.
		failedAncestor = result
	}

	for _, child := range podGroupInfo.Children() {
		completeCompositePodGroupAlgorithmResultMap(child, pgResults, failedAncestor)
	}
}

// buildParentPodGroupErrorStatus returns the status for a pod whose parent pod group encountered an error.
func buildParentPodGroupErrorStatus(podGroupName string) *fwk.Status {
	return fwk.AsStatus(fmt.Errorf("parent pod group %q encountered an error", podGroupName))
}

// buildAncestorFailureStatus returns the status for a pod group whose ancestor failed.
// The status only names the ancestor, without repeating the ancestor's failure reason,
// which is reported in the ancestor's own status.
// If the ancestor's status is an Error, it returns an Error.
// Otherwise, it returns ancestor's status code with a podGroupFitError that names the ancestor
// and carries its plugins.
func buildAncestorFailureStatus(failedAncestor *PodGroupScheduledResult) *fwk.Status {
	ancestorName := failedAncestor.podGroupInfo.GetName()
	if failedAncestor.status.IsError() {
		return fwk.AsStatus(fmt.Errorf("ancestor composite pod group %q encountered an error", ancestorName))
	}

	msg := fmt.Sprintf("ancestor composite pod group %q is unschedulable", ancestorName)
	fitError := newPodGroupFitError(fwk.NewStatus(fwk.Unschedulable, msg))
	fitError.addPluginStatus(failedAncestor.status)
	return fwk.NewStatus(failedAncestor.status.Code()).WithError(fitError)
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

	entityType := metrics.EntityTypeToLabel(rootPodGroupInfo.Type())
	rootResult := podGroupResults[rootPodGroupInfo.PodGroupInfo.GetKey()]
	switch {
	case rootResult.status.IsSuccess():
		metrics.PodGroupScheduled(schedFwk.ProfileName(), entityType, metrics.SinceInSeconds(start))
	case rootResult.status.IsRejected():
		if rootResult.waitingOnPreemption {
			metrics.PodGroupWaitingOnPreemption(schedFwk.ProfileName(), entityType, metrics.SinceInSeconds(start))
		} else {
			metrics.PodGroupUnschedulable(schedFwk.ProfileName(), entityType, metrics.SinceInSeconds(start))
		}
	default:
		metrics.PodGroupScheduleError(schedFwk.ProfileName(), entityType, metrics.SinceInSeconds(start))
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
			// Pod group failed with an error. Reject all pods with a status referencing the parent pod group.
			errStatus := buildParentPodGroupErrorStatus(pgi.GetName())
			sched.FailureHandler(ctx, schedFwk, pInfo, errStatus, clearNominatedNode, podSchedulingStart)
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
