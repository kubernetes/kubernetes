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
	"iter"
	"maps"
	"math/rand"
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

// revertFns is an aggregator of functions that undo the in-memory changes (such
// as assuming pods and calls to Reserve plugins) performed during the pod group
// scheduling algorithm simulation.
//
// These functions are executed:
//   - After the whole root is processed in the scheduling algorithm, to clean up the
//     simulation state.
//   - On failure in (composite) pod group algorithm, to immediately roll back partial
//     modifications.
//   - After each candidate placement is considered in the placement scheduling algorithm,
//     to reset the state before evaluating the next candidate placement.
type revertFns []func()

// append registers additional revert functions.
func (r *revertFns) append(other revertFns) {
	*r = append(*r, other...)
}

// revert executes the underlying reverting functions in reverse order of their registration
// (Last In, First Out). Reverting in LIFO order ensures that sequential operations are unwound
// correctly, preserving state integrity since later operations might depend on the side-effects
// established by earlier ones (similar to how deferred execution works in Go).
func (r *revertFns) revert() {
	if r == nil {
		return
	}
	for i := len(*r) - 1; i >= 0; i-- {
		(*r)[i]()
		(*r)[i] = nil // allow GC
	}
	*r = nil
}

// errPodGroupUnschedulable is used to describe that the pod group is unschedulable.
var errPodGroupUnschedulable = fmt.Errorf("pod group is unschedulable")

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
		// We set the underlying API object to nil to signify that we don't want to update its condition in failure handler.
		podGroupInfo.PodGroup = nil
		podGroupInfo.CompositePodGroup = nil
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
	if len(podGroupInfo.QueuedPodInfos) == 0 {
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
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) && (pgi.GetType() == fwk.CompositePodGroupKeyType || pgi.CompositePodGroup != nil) {
		compositePodGroup, err := sched.nodeInfoSnapshot.CompositePodGroups().Get(pgi.Namespace, pgi.Name)
		if err != nil {
			return err
		}
		if pgi.CompositePodGroup != nil && !ptr.Equal(compositePodGroup.Spec.ParentCompositePodGroupName, pgi.CompositePodGroup.Spec.ParentCompositePodGroupName) {
			return fmt.Errorf("different parent in composite pod group between snapshot (%s) and queued entity (%s)",
				ptr.Deref(compositePodGroup.Spec.ParentCompositePodGroupName, "[unset]"),
				ptr.Deref(pgi.CompositePodGroup.Spec.ParentCompositePodGroupName, "[unset]"))
		}
		cpgs, err := sched.nodeInfoSnapshot.CompositePodGroupStates().Get(pgi.Namespace, pgi.Name)
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
		pgi.CompositePodGroup = compositePodGroup
	} else {
		podGroup, err := sched.nodeInfoSnapshot.PodGroups().Get(pgi.Namespace, pgi.Name)
		if err != nil {
			return err
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) &&
			!ptr.Equal(podGroup.Spec.ParentCompositePodGroupName, pgi.PodGroup.Spec.ParentCompositePodGroupName) {
			return fmt.Errorf("different parent in pod group between snapshot (%s) and queued entity (%s)",
				ptr.Deref(podGroup.Spec.ParentCompositePodGroupName, "[unset]"),
				ptr.Deref(pgi.PodGroup.Spec.ParentCompositePodGroupName, "[unset]"))
		}
		pgi.PodGroup = podGroup
	}
	return nil
}

// handlePodGroupFailureBeforeScheduling handles the failure of a (composite) pod group that occurred before scheduling.
func (sched *Scheduler) handlePodGroupFailureBeforeScheduling(ctx context.Context, podGroupInfo *framework.QueuedPodGroupInfo, err error) {
	logger := klog.FromContext(ctx)
	podGroupInfo.ForEachPodInfo(func(podInfo *framework.QueuedPodInfo) bool {
		podFwk, podFwkErr := sched.frameworkForPod(podInfo.Pod)
		if podFwkErr != nil {
			// This shouldn't happen, because we only accept for scheduling the pods
			// which specify a scheduler name that matches one of the profiles.
			logger.Error(podFwkErr, "Error occurred")
			sched.SchedulingQueue.Done(podInfo.Pod.UID)
		} else {
			sched.FailureHandler(ctx, podFwk, podInfo, fwk.AsStatus(err), clearNominatedNode, time.Now())
		}
		return true
	})
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
	for _, child := range pgi.GetChildGroups() {
		sched.updatePodGroupConditionWithError(ctx, child, err)
	}
}

// validatePodGroup ensures that:
// - all Pods in a group hierarchy have matching scheduler name,
// - all Pods in a group hierarchy have the same preemption policy,
// - the root group has the same priority as all the Pods.
// - the root group has the same preemption policy as all the Pods.
func (sched *Scheduler) validatePodGroup(podGroupInfo *framework.QueuedPodGroupInfo) error {
	schedulerName := ""
	podGroupPriority := podGroupInfo.GetPriority()
	var err error

	var pgPreemptionPolicy v1.PreemptionPolicy
	if utilfeature.DefaultFeatureGate.Enabled(features.PodGroupPreemptionPolicy) {
		pgPreemptionPolicy = podGroupPreemptionPolicy(podGroupInfo)
	}

	validatePod := func(pod *v1.Pod) error {
		if pod.Spec.SchedulerName != schedulerName {
			return fmt.Errorf("all pods in a single pod group should have the same .spec.schedulerName set, got: %q and %q", pod.Spec.SchedulerName, schedulerName)
		}
		podPriority := corev1helpers.PodPriority(pod)
		if podPriority != podGroupPriority {
			return fmt.Errorf("all pods in a single pod group should have the same priority as the pod group's priority, got %d and %d", podPriority, podGroupPriority)
		}

		pPreemptionPolicy := podPreemptionPolicy(pod)
		if utilfeature.DefaultFeatureGate.Enabled(features.PodGroupPreemptionPolicy) {
			// If the PodGroupPreemptionPolicy feature is enabled, validate that the pod's preemption policy
			// matches the root group's preemption policy.
			if pPreemptionPolicy != pgPreemptionPolicy {
				return fmt.Errorf("all pods in a single pod group should have the same preemption policy as the pod group's preemption policy, got %v and %v", pPreemptionPolicy, pgPreemptionPolicy)
			}
		} else {
			// If the PodGroupPreemptionPolicy feature is disabled, the preemption policy is determined by the first pod in the group.
			// Validate that preemption policy is the same across all pods in the pod group.
			if pgPreemptionPolicy == "" {
				pgPreemptionPolicy = pPreemptionPolicy
			} else if pPreemptionPolicy != pgPreemptionPolicy {
				return fmt.Errorf("all pods in a single pod group should have the same preemption policy, got %v and %v", pPreemptionPolicy, pgPreemptionPolicy)
			}
		}
		return nil
	}
	podGroupInfo.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		if schedulerName == "" {
			schedulerName = pInfo.Pod.Spec.SchedulerName
		}
		err = validatePod(pInfo.Pod)
		return err == nil
	})
	if err != nil {
		return err
	}

	err = sched.validateScheduledPods(podGroupInfo.PodGroupInfo, validatePod)
	if err != nil {
		return err
	}

	if _, ok := sched.Profiles[schedulerName]; !ok {
		return fmt.Errorf("profile not found for scheduler name %q", schedulerName)
	}

	return nil
}

// podGroupPreemptionPolicy returns the PreemptionPolicy set in the pod group, or the default policy
// (PreemptLowerPriority) if not set.
func podGroupPreemptionPolicy(podGroupInfo *framework.QueuedPodGroupInfo) v1.PreemptionPolicy {
	if pg := podGroupInfo.PodGroup; pg != nil && pg.Spec.PreemptionPolicy != nil {
		return v1.PreemptionPolicy(*pg.Spec.PreemptionPolicy)
	}
	if cpg := podGroupInfo.CompositePodGroup; cpg != nil && cpg.Spec.PreemptionPolicy != nil {
		return v1.PreemptionPolicy(*cpg.Spec.PreemptionPolicy)
	}
	return v1.PreemptLowerPriority
}

// podPreemptionPolicy returns the PreemptionPolicy set in the pod, or the default policy
// (PreemptLowerPriority) if not set.
func podPreemptionPolicy(pod *v1.Pod) v1.PreemptionPolicy {
	if pod != nil && pod.Spec.PreemptionPolicy != nil {
		return *pod.Spec.PreemptionPolicy
	}
	return v1.PreemptLowerPriority
}

// validateScheduledPods validates that already-scheduled pods in the pod group hierarchy
// conform to the same group-wide constraints (like scheduler name and priority) as
// unscheduled pods. It recursively traverses the hierarchy to fetch and check the cached
// state for each leaf group.
func (sched *Scheduler) validateScheduledPods(podGroupInfo *framework.PodGroupInfo, validatePod func(pod *v1.Pod) error) error {
	if podGroupInfo.CompositePodGroup != nil {
		for _, child := range podGroupInfo.GetChildGroups() {
			if err := sched.validateScheduledPods(child, validatePod); err != nil {
				return err
			}
		}
		return nil
	}

	podGroupState, err := sched.nodeInfoSnapshot.PodGroupStates().Get(podGroupInfo.Namespace, podGroupInfo.Name)
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
	var result framework.Framework
	podGroupInfo.ForEachPodInfo(func(pInfo *framework.QueuedPodInfo) bool {
		result = sched.Profiles[pInfo.Pod.Spec.SchedulerName]
		return false
	})
	return result
}

// skipPodGroupPodSchedule skips the scheduling of particular pods from the group when they should no longer be considered.
// This can happen when the pod is already being deleted (i.e., when its deletionTimestamp is set)
// or when the pod has already been assumed.
func (sched *Scheduler) skipPodGroupPodSchedule(ctx context.Context, schedFwk framework.Framework, podGroupInfo *framework.QueuedPodGroupInfo) {
	queuedPodInfosToUpdate := map[fwk.EntityKey][]*framework.QueuedPodInfo{}
	for pgKey, pInfos := range podGroupInfo.QueuedPodInfos {
		filteredQueuedPodInfos := make([]*framework.QueuedPodInfo, 0, len(pInfos))
		for _, podInfo := range pInfos {
			if sched.skipPodSchedule(ctx, schedFwk, podInfo.Pod) {
				// We don't put this Pod back to the queue, but we have to cleanup the in-flight pods/events.
				sched.SchedulingQueue.Done(podInfo.Pod.UID)
				continue
			}
			filteredQueuedPodInfos = append(filteredQueuedPodInfos, podInfo)
		}
		if len(filteredQueuedPodInfos) != len(pInfos) {
			podGroupInfo.QueuedPodInfos[pgKey] = filteredQueuedPodInfos
			if len(filteredQueuedPodInfos) == 0 {
				delete(podGroupInfo.QueuedPodInfos, pgKey)
			}
			queuedPodInfosToUpdate[pgKey] = filteredQueuedPodInfos
		}
	}
	sched.updateUnscheduledPods(podGroupInfo.PodGroupInfo, queuedPodInfosToUpdate)
}

// updateUnscheduledPods synchronizes the list of unscheduled pods in the pod group hierarchy
// after filtering out pods that are deleted or already assumed. It recursively traverses the
// group hierarchy to update each leaf pod group's list of unscheduled pods.
func (sched *Scheduler) updateUnscheduledPods(pgi *framework.PodGroupInfo, queuedPodInfosToUpdate map[fwk.EntityKey][]*framework.QueuedPodInfo) {
	if len(queuedPodInfosToUpdate) == 0 {
		return
	}
	if pgi.CompositePodGroup != nil {
		for _, child := range pgi.GetChildGroups() {
			sched.updateUnscheduledPods(child, queuedPodInfosToUpdate)
		}
		return
	}
	key := pgKey(pgi)
	if podInfos, ok := queuedPodInfosToUpdate[key]; ok {
		pgi.UnscheduledPods = make([]*v1.Pod, 0, len(podInfos))
		for _, pInfo := range podInfos {
			pgi.UnscheduledPods = append(pgi.UnscheduledPods, pInfo.Pod)
		}
		delete(queuedPodInfosToUpdate, key)
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
func initPodSchedulingContext(ctx context.Context, pod *v1.Pod, placementCycleState *framework.CycleState) *podSchedulingContext {
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

	return &podSchedulingContext{
		logger:         logger,
		state:          state,
		podsToActivate: podsToActivate,
	}
}

// podGroupCycle runs a pod group scheduling cycle for the given pod group.
// Cluster state should be snapshotted before calling this method.
func (sched *Scheduler) podGroupCycle(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo, start time.Time) {
	pgResults := sched.runRootSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo)
	rootStatus := pgResults[pgKey(rootPodGroupInfo.PodGroupInfo)].status
	var completePGResults map[fwk.EntityKey]*podGroupAlgorithmResult
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) && rootPodGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		completePGResults = completeCompositePodGroupAlgorithmResult(ctx, rootPodGroupInfo, podGroupCycleState, pgResults)
	} else {
		// pgResults has exactly 1 element.
		queuedPodInfos := rootPodGroupInfo.QueuedPodInfos[pgKey(rootPodGroupInfo.PodGroupInfo)]
		result := completePodGroupAlgorithmResult(ctx, queuedPodInfos, podGroupCycleState, pgResults[pgKey(rootPodGroupInfo.PodGroupInfo)])
		completePGResults = map[fwk.EntityKey]*podGroupAlgorithmResult{pgKey(rootPodGroupInfo.PodGroupInfo): result}
	}

	metrics.PodGroupSchedulingAlgorithmLatency.Observe(metrics.SinceInSeconds(start))

	// Run pod group post filter plugins if scheduling failed. If any of the plugins is successful,
	// we need to put the pods from pod group back into the scheduling queue.
	if rootStatus.Code() == fwk.Unschedulable {
		var pgSchedulingFunc fwk.PodGroupSchedulingFunc = func(ctx context.Context) (*fwk.PodGroupAssignments, *fwk.Status) {
			results := sched.runRootSchedulingAlgorithm(ctx, schedFwk, framework.NewCycleState(), rootPodGroupInfo)
			proposedAssignments := make([]fwk.ProposedAssignment, 0)
			for _, res := range results {
				proposedAssignments = append(proposedAssignments, makeProposedAssignments(res)...)
			}
			return &fwk.PodGroupAssignments{
				ProposedAssignments: proposedAssignments,
			}, results[pgKey(rootPodGroupInfo.PodGroupInfo)].status
		}
		pgPostFilterResult, status := schedFwk.RunPodGroupPostFilterPlugins(ctx, podGroupCycleState, rootPodGroupInfo.PodGroupInfo, pgSchedulingFunc)
		applyPodGroupPostFilterResult(completePGResults, pgPostFilterResult, status)
	}

	sched.submitPodGroupAlgorithmResult(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo, completePGResults, start, rootStatus)
}

// runRootSchedulingAlgorithm orchestrates the scheduling attempt for a root pod group.
// It decides whether to evaluate a single group or recursively evaluate a composite group hierarchy
// and eventually cleans up the tentative reservations that the algorithm makes during its execution.
// The returned map aggregates scheduling results across the entire pod group hierarchy.
func (sched *Scheduler) runRootSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo) map[fwk.EntityKey]*podGroupAlgorithmResult {
	var revertFns revertFns
	defer revertFns.revert()
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) && rootPodGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		result := map[fwk.EntityKey]*podGroupAlgorithmResult{}
		rootResult, childRevertFns := sched.podGroupSchedulingRecursiveAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo, rootPodGroupInfo.PodGroupInfo, result)
		revertFns = childRevertFns
		if rootResult.status.IsSuccess() && !rootResult.anyScheduled {
			// The framework requires at least 1 pod to be scheduled in order to return a success status.
			rootResult.status = fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable)
		}
		return result
	}
	result, childRevertFns := sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo.PodGroupInfo, rootPodGroupInfo)
	revertFns = childRevertFns

	if result.status.IsSuccess() && !result.anyScheduled {
		// The framework requires at least 1 pod to be scheduled in order to return a success status.
		result.status = fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable)
	}
	return map[fwk.EntityKey]*podGroupAlgorithmResult{pgKey(result.podGroupInfo): result}
}

// algorithmResult stores the scheduling result and status for a scheduling attempt of a single pod.
type algorithmResult struct {
	// podInfo is the pod info for the pod the result applies to.
	podInfo *framework.QueuedPodInfo
	// scheduleResult is a scheduling algorithm result.
	scheduleResult ScheduleResult
	// podCtx is a specific pod scheduling context used for the scheduling algorithm.
	podCtx *podSchedulingContext
	// schedulingDuration is a pod scheduling duration used for metrics recording.
	schedulingDuration time.Duration
	// status is a scheduling algorithm status.
	status *fwk.Status
}

func (ar *algorithmResult) GetPod() *v1.Pod {
	return ar.podInfo.Pod
}

func (ar *algorithmResult) GetPodInfo() fwk.PodInfo {
	return ar.podInfo
}

func (ar *algorithmResult) GetNodeName() string {
	return ar.scheduleResult.SuggestedHost
}

func (ar *algorithmResult) GetCycleState() fwk.CycleState {
	return ar.podCtx.state
}

// podGroupAlgorithmResult stores the scheduling pod scheduling results for a pod group
// and any information needed to act on these results.
type podGroupAlgorithmResult struct {
	// podGroupInfo is the leaf pod group this result applies to.
	podGroupInfo *framework.PodGroupInfo
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
	// anyScheduled indicates if at least one pod was scheduled in this pod group during this cycle.
	anyScheduled bool
}

// podGroupSchedulingDefaultAlgorithm runs the default algorithm for scheduling a pod group.
// It tries to schedule each pod using standard filtering and scoring logic in a fixed order.
// If a pod requires preemption to be schedulable, subsequent pods in the algorithm
// treat that pod as already scheduled on that node with victims being already removed in memory.
// The returned revertFns accumulates revert functions for all scheduled pods, allowing the caller
// to rollback tentative reservations if the pod group scheduling cycle fails.
func (sched *Scheduler) podGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (result *podGroupAlgorithmResult, revertFns revertFns) {
	defer func() {
		if !result.status.IsSuccess() {
			revertFns.revert()
			result.anyScheduled = false
		}
	}()

	// Retrieve the queued podinfos for the given pod group from the root queuedPodGroupInfo.
	queuedPodInfos := queuedPodGroupInfo.QueuedPodInfos[pgKey(podGroupInfo)]
	result = &podGroupAlgorithmResult{
		podGroupInfo:        podGroupInfo,
		podResults:          make([]algorithmResult, 0, len(queuedPodInfos)),
		status:              fwk.NewStatus(fwk.Unschedulable).WithError(errPodGroupUnschedulable),
		waitingOnPreemption: false,
		placementCycleState: placementCycleState,
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running a pod group scheduling algorithm", "podGroup", klog.KObj(podGroupInfo), "unscheduledPodsCount", len(queuedPodInfos))

	podGroupState, err := sched.nodeInfoSnapshot.PodGroupStates().Get(podGroupInfo.Namespace, podGroupInfo.Name)
	if err != nil {
		result.status = fwk.AsStatus(fmt.Errorf("failed to get podGroup state for podGroup %s to compute gang feasibility: %w", klog.KObj(podGroupInfo), err))
		return result, nil
	}

	// Run PlacementFeasible plugins to check if the pod group can meet its constraints
	// before even attempting to schedule any pods.
	placementProgress := framework.PlacementProgress{
		Remaining: len(podGroupInfo.GetUnscheduledPods()),
		Scheduled: podGroupState.ScheduledPodsCount(),
	}
	placementFeasibleStatus := schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo, placementProgress)
	result.status = placementFeasibleStatus
	if placementFeasibleStatus.IsError() {
		// Do not evaluate any pods if PlacementFeasible plugins return error or unexpected status.
		result.status = fwk.AsStatus(fmt.Errorf("failed to evaluate placement feasibility: %w", placementFeasibleStatus.AsError()))
		return result, nil
	}
	if placementFeasibleStatus.Code() == fwk.Unschedulable {
		// Unschedulable from PlacementFeasible plugins indicates that the pod group
		// cannot meet its constraints, even if we succeed in scheduling all the pods.
		// Exit early from the pod group algorithm.
		result.status = fwk.NewStatus(fwk.Unschedulable, result.status.Reasons()...).WithError(errPodGroupUnschedulable)
		return result, nil
	}

	anyScheduled := false
	for _, podInfo := range queuedPodInfos {
		podResult, revertFn := sched.podGroupPodSchedulingAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, podInfo)
		result.podResults = append(result.podResults, podResult)
		if revertFn != nil {
			revertFns.append([]func(){revertFn})
		}

		if !podResult.status.IsSuccess() && !podResult.status.IsRejected() {
			// When the algorithm returns error or unexpected status, stop evaluating the rest of the pods.
			result.status = fwk.AsStatus(fmt.Errorf("failed to schedule other pod from a pod group: %w", podResult.status.AsError()))
			break
		}

		// Check if the pod group can still meet its constraints after scheduling the current pod.
		placementProgress.Remaining--
		if podResult.status.IsSuccess() {
			placementProgress.Scheduled++
		}
		placementFeasibleStatus := schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo, placementProgress)
		if placementFeasibleStatus.IsError() {
			// Stop evaluating the rest of the pods if PlacementFeasible plugins return error or unexpected status.
			result.status = fwk.AsStatus(fmt.Errorf("failed to evaluate placement feasibility: %w", placementFeasibleStatus.AsError()))
			break
		}

		result.status = placementFeasibleStatus

		// Unschedulable from PlacementFeasible plugins indicates that the pod group
		// cannot meet its constraints regardless of how many more pods we check.
		// We can stop the scheduling loop early.
		if placementFeasibleStatus.Code() == fwk.Unschedulable {
			break
		}

		anyScheduled = anyScheduled || podResult.status.IsSuccess()
	}

	if result.status.IsWait() {
		result.status = fwk.NewStatus(fwk.Unschedulable, result.status.Reasons()...).WithError(errPodGroupUnschedulable)
	}

	result.anyScheduled = anyScheduled
	return result, revertFns
}

// podGroupPodSchedulingAlgorithm runs a scheduling algorithm for individual pod from a pod group.
// It returns the algorithm result together with the revert function.
// The returned revert function rolls back tentative node reservations for the pod if the overall
// pod group fails to schedule.
func (sched *Scheduler) podGroupPodSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, podInfo *framework.QueuedPodInfo) (algorithmResult, func()) {
	pod := podInfo.Pod
	podCtx := initPodSchedulingContext(ctx, pod, placementCycleState)
	logger := podCtx.logger
	ctx = klog.NewContext(ctx, logger)
	start := time.Now()

	logger.V(4).Info("Attempting to schedule a pod belonging to a pod group", "podGroup", klog.KObj(podGroupInfo), "pod", klog.KObj(pod))

	scheduleResult, status := sched.schedulingAlgorithm(ctx, podCtx.state, schedFwk, podInfo, start)
	if !status.IsSuccess() {
		return algorithmResult{
			podInfo:            podInfo,
			scheduleResult:     scheduleResult,
			podCtx:             podCtx,
			schedulingDuration: time.Since(start),
			status:             status,
		}, nil
	}
	assumeStatus, revertFn := sched.assumeAndReserveWithRevert(ctx, podCtx.state, schedFwk, podInfo, scheduleResult)
	if !assumeStatus.IsSuccess() {
		return algorithmResult{
			podInfo:            podInfo,
			scheduleResult:     ScheduleResult{nominatingInfo: clearNominatedNode},
			podCtx:             podCtx,
			schedulingDuration: time.Since(start),
			status:             assumeStatus,
		}, nil
	}

	return algorithmResult{
		podInfo:            podInfo,
		scheduleResult:     scheduleResult,
		podCtx:             podCtx,
		schedulingDuration: time.Since(start),
		status:             status,
	}, revertFn
}

func (sched *Scheduler) assumeAndReserveWithRevert(ctx context.Context,
	state fwk.CycleState,
	schedFramework framework.Framework,
	podInfo *framework.QueuedPodInfo,
	scheduleResult ScheduleResult,
) (*fwk.Status, func()) {
	assumedPodInfo, assumeStatus := sched.assumeAndReserve(ctx, state, schedFramework, podInfo, scheduleResult)
	if !assumeStatus.IsSuccess() {
		return assumeStatus, nil
	}
	return assumeStatus, func() {
		err := sched.unreserveAndForget(ctx, state, schedFramework, assumedPodInfo, scheduleResult.SuggestedHost)
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "ForgetPod failed")
		}
	}
}

// completePodGroupAlgorithmResult ensures that the podGroupAlgorithmResult contains the same number of podResults as there are pods in QueuedPodInfos.
func completePodGroupAlgorithmResult(ctx context.Context, queuedPodInfos []*framework.QueuedPodInfo, podGroupState *framework.CycleState, podGroupResult *podGroupAlgorithmResult) *podGroupAlgorithmResult {
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
		placementCycleState.SetPodGroupSchedulingCycle(podGroupState)
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
func completeCompositePodGroupAlgorithmResult(ctx context.Context, rootPodGroupInfo *framework.QueuedPodGroupInfo, rootCycleState *framework.CycleState, pgResults map[fwk.EntityKey]*podGroupAlgorithmResult) map[fwk.EntityKey]*podGroupAlgorithmResult {
	completeCompositePodGroupAlgorithmResultMap(ctx, rootPodGroupInfo.PodGroupInfo, pgResults, &podGroupAlgorithmResult{})
	for pgKey, queuedPodInfos := range rootPodGroupInfo.QueuedPodInfos {
		pgResult := pgResults[pgKey]
		// Ensure podResults has an entry for each pod in the pod group with a status.
		completePodGroupAlgorithmResult(ctx, queuedPodInfos, rootCycleState, pgResult)
	}
	return pgResults
}

// completeCompositePodGroupAlgorithmResultMap propagates scheduling failures from parents to children.
// This is necessary because child pod groups cannot be committed or bound if their parent composite
// pod group fails to meet its scheduling requirements.
func completeCompositePodGroupAlgorithmResultMap(ctx context.Context, podGroupInfo *framework.PodGroupInfo, pgResults map[fwk.EntityKey]*podGroupAlgorithmResult, parentResult *podGroupAlgorithmResult) {
	key := pgKey(podGroupInfo)
	result, ok := pgResults[key]
	// When a parent composite pod group fails, any child that previously succeeded during its own evaluation
	// must be invalidated with the parent's failure status to prevent its pods from proceeding to binding.
	if !ok || (!parentResult.status.IsSuccess() && result.status.IsSuccess()) {
		result = &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       parentResult.status,
		}
		pgResults[key] = result
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
func applyPodGroupPostFilterResult(completePGResults map[fwk.EntityKey]*podGroupAlgorithmResult, pgPostFilterResult *fwk.PodGroupPostFilterResult, status *fwk.Status) {
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
	if status.IsError() {
		for _, pgResult := range completePGResults {
			pgResult.status = status
		}
	} else {
		for _, pgResult := range completePGResults {
			pgResult.status.AppendReason(status.String())
		}
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
func (sched *Scheduler) submitPodGroupAlgorithmResult(ctx context.Context, schedFwk framework.Framework, podGroupState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo, podGroupResults map[fwk.EntityKey]*podGroupAlgorithmResult, start time.Time, rootStatus *fwk.Status) {
	logger := klog.FromContext(ctx)

	for _, podGroupResult := range podGroupResults {
		pgi := podGroupResult.podGroupInfo
		if pgi.CompositePodGroup != nil {
			// Composite pod groups do not own any pods directly.
			continue
		}
		queuedPodInfos := rootPodGroupInfo.QueuedPodInfos[pgKey(pgi)]
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
						sched.FailureHandler(ctx, schedFwk, pInfo, podGroupResult.status, podResult.scheduleResult.nominatingInfo, podSchedulingStart)
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
				Reason:  "Scheduled",
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
				Message: podGroupResult.status.AsError().Error(),
			}
			utilruntime.HandleErrorWithContext(ctx, podGroupResult.status.AsError(), "Error scheduling pod group", "podGroup", klog.KObj(pgi), "errorPods", len(queuedPodInfos))
		}
		sched.updatePodGroupCondition(ctx, pgi, condition)
	}

	rootResult := podGroupResults[pgKey(rootPodGroupInfo.PodGroupInfo)]
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

// updatePodGroupCondition patches the given condition on a PodGroup.
func (sched *Scheduler) updatePodGroupCondition(ctx context.Context,
	podGroupInfo *framework.PodGroupInfo, condition *metav1.Condition) {
	logger := klog.FromContext(ctx)

	// If the PodGroup was already successfully scheduled, don't regress the
	// condition back to False on a subsequent cycle for extra pods.
	pg := podGroupInfo.PodGroup
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
// For a standalone PodGroup it evaluates the placement matching the pods' NominatedNodeName first
// and uses it if the gang is feasible there, short-circuiting the rest. Otherwise (or for a
// PodGroup that is part of a CompositePodGroup) it tries every placement through
// podGroupSchedulingDefaultAlgorithm and runs placement scorer plugins to select the best one.
func (sched *Scheduler) podGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (finalResult *podGroupAlgorithmResult, revertFns revertFns) {
	allNodes, err := sched.nodeInfoSnapshot.ListNodesInPlacement()
	if err != nil {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}, nil
	}

	// For now, always record plugin metrics until we understand its impact on performance.
	podGroupCycleState.SetRecordPluginMetrics(true)
	placements, status := schedFwk.RunPlacementGeneratePlugins(ctx, podGroupCycleState, podGroupInfo, allNodes)
	if !status.IsSuccess() {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}
	metrics.RecordGeneratedPlacements(schedFwk.ProfileName(), len(placements))

	var anyResult *podGroupAlgorithmResult
	successfulResults := make(map[*fwk.Placement]*podGroupAlgorithmResult)

	parentPlacement := sched.nodeInfoSnapshot.GetPlacement()
	defer func() {
		sched.nodeInfoSnapshot.ForgetPlacement()
		err := sched.nodeInfoSnapshot.AssumePlacement(parentPlacement)
		if err != nil {
			finalResult.status = fwk.AsStatus(fmt.Errorf("failed to restore parent pod group placement: %w", err))
			revertFns.revert()
		}
	}()

	// Try the placement matching the pods' NominatedNodeName first, mirroring pod-by-pod
	// scheduling, which evaluates the nominated node before all others. A nominated node is
	// usually set by a previous preemption cycle and is the placement the pod group is
	// expected to land on, so if the gang is feasible there we use it and skip the rest.
	//
	// This fast path is limited to standalone PodGroups. A PodGroup that is part of a
	// CompositePodGroup defers its feasibility verdict to the CPG root, where a Success status
	// with nothing scheduled is still meaningful. Short-circuiting on it (or dropping it) here
	// could wrongly report the whole CPG Unschedulable and stop sibling groups from being
	// evaluated, so CPG children evaluate every placement as before.
	// TODO(kubernetes/kubernetes#140863): extend NNN support to CompositePodGroups.
	nominatedFeasible := false
	var nominated *fwk.Placement
	if queuedPodGroupInfo.GetType() == fwk.PodGroupKeyType {
		nominated = nominatedPlacement(placements, podGroupInfo, queuedPodGroupInfo)
	}
	if nominated != nil {
		result := sched.evaluatePlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, queuedPodGroupInfo, nominated)
		if result.status.IsError() {
			return result, nil
		}
		anyResult = result
		// Honoring the nominated placement matters more than fitting more pods elsewhere: the NNN
		// was typically set by a prior preemption cycle. This mirrors pod-by-pod NNN, which can
		// pick a non-optimal node because it skips scoring. The tradeoff: when minCount < len(pods)
		// we may prefer the nominated placement over one that fits the whole gang. For a standalone
		// PodGroup a Success status implies at least minCount pods were placed. The anyScheduled
		// check prevents short-circuiting when the placement is feasible only because minCount pods
		// were already scheduled in previous cycles (for example minCount=3 with 3 pods already
		// running), but we failed to place any newly arriving pods on the nominated placement.
		if result.status.IsSuccess() && result.anyScheduled {
			successfulResults[nominated] = result
			nominatedFeasible = true
		}
	}

	// Only evaluate the remaining placements when the nominated one wasn't feasible.
	if !nominatedFeasible {
		for _, placement := range placements {
			if placement == nominated {
				continue
			}
			result := sched.evaluatePlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, queuedPodGroupInfo, placement)
			if result.status.IsError() {
				return result, nil
			}

			if anyResult == nil {
				anyResult = result
			}

			if result.status.IsSuccess() {
				successfulResults[placement] = result
			}
		}
	}

	if len(successfulResults) == 0 {
		// We need to send events and set the status for pods in case all simulations were infeasible.
		// anyResult is the nominated placement's result when one was evaluated, otherwise the first
		// placement tried. Which one we report is otherwise arbitrary and may change in the future.
		anyResult.status = fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("0/%d placements are available, reported placement status: %v", len(placements), anyResult.status.AsError())).WithError(errPodGroupUnschedulable)
		return anyResult, nil
	}

	bestPlacement, status := sched.findBestPodGroupPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, successfulResults)
	if !status.IsSuccess() {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}
	bestResult := successfulResults[bestPlacement]

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		revertFns, err = sched.assumeSubtreeWithRevert(ctx, schedFwk, podGroupInfo, map[fwk.EntityKey]*podGroupAlgorithmResult{pgKey(podGroupInfo): bestResult})
		if err != nil {
			return &podGroupAlgorithmResult{
				podGroupInfo: podGroupInfo,
				status:       fwk.AsStatus(fmt.Errorf("failed to assume the subtree: %w", err)),
			}, nil
		}

		return bestResult, revertFns
	}
	return bestResult, nil
}

// compositePodGroupSchedulingPlacementAlgorithm tries several different combinations for scheduling the child pod groups and selects the best one.
// First it runs placement generator plugins to create a list of placements.
// Placement is a set of nodes that will be considered when scheduling a pod group.
// Then for each placement it tries to schedule the pod group through podGroupSchedulingDefaultAlgorithm.
// Finally, it runs placement scorer plugins to select the best placement.
func (sched *Scheduler) compositePodGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) (finalResult *podGroupAlgorithmResult, revertFns revertFns) {
	defer func() {
		results[pgKey(podGroupInfo)] = finalResult
	}()
	logger := klog.FromContext(ctx)
	allNodes, err := sched.nodeInfoSnapshot.ListNodesInPlacement()
	if err != nil {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}, nil
	}

	// For now, always record plugin metrics until we understand its impact on performance.
	podGroupCycleState.SetRecordPluginMetrics(true)
	placements, status := schedFwk.RunPlacementGeneratePlugins(ctx, podGroupCycleState, podGroupInfo, allNodes)
	if !status.IsSuccess() {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}

	var anyResultSubtree map[fwk.EntityKey]*podGroupAlgorithmResult
	successfulResults := make(map[*fwk.Placement]map[fwk.EntityKey]*podGroupAlgorithmResult)

	parentPlacement := sched.nodeInfoSnapshot.GetPlacement()
	defer func() {
		sched.nodeInfoSnapshot.ForgetPlacement()
		err := sched.nodeInfoSnapshot.AssumePlacement(parentPlacement)
		if err != nil {
			finalResult.status = fwk.AsStatus(fmt.Errorf("failed to restore parent pod group placement: %w", err))
			revertFns.revert()
		}
	}()

	for _, placement := range placements {
		logger.V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
		err := sched.nodeInfoSnapshot.AssumePlacement(placement)
		if err != nil {
			return &podGroupAlgorithmResult{
				podGroupInfo: podGroupInfo,
				status:       fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
			}, nil
		}
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
		subtreeResult := map[fwk.EntityKey]*podGroupAlgorithmResult{}
		result, placementRevertFns := sched.compositePodGroupSchedulingDefaultAlgorithm(ctx, schedFwk, placementCycleState, root, podGroupInfo, subtreeResult)
		placementRevertFns.revert()

		if result.status.IsError() {
			return result, nil
		}

		if anyResultSubtree == nil {
			anyResultSubtree = subtreeResult
		}

		if result.status.IsSuccess() {
			successfulResults[placement] = subtreeResult
		}
	}

	if len(successfulResults) == 0 {
		// We need to send events and set the status for pods in case all simulations were infeasible.
		// The selection of which simulation we report is arbitrary for now, but may change in the future.
		anyResultRoot := anyResultSubtree[pgKey(podGroupInfo)]
		anyResultRoot.status = fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("0/%d placements are available, first placement status: %v", len(placements), anyResultRoot.status.AsError())).WithError(errPodGroupUnschedulable)
		// It is critical to copy the entire anyResultSubtree into results.
		// If omitted, the pod results are reconstructed later using the generic parent error
		// (errPodGroupUnschedulable) rather than their original *framework.FitError.
		// Losing the FitError means we lose the UnschedulablePlugins for each pod,
		// which breaks the QueueingHints.
		maps.Copy(results, anyResultSubtree)
		return anyResultRoot, nil
	}

	bestPlacement, status := sched.findBestCompositePodGroupPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, successfulResults)
	if !status.IsSuccess() {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}

	bestResult := successfulResults[bestPlacement]

	revertFns, err = sched.assumeSubtreeWithRevert(ctx, schedFwk, podGroupInfo, bestResult)
	if err != nil {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to assume the subtree: %w", err)),
		}, nil
	}
	maps.Copy(results, bestResult)

	return bestResult[pgKey(podGroupInfo)], revertFns
}

func (sched *Scheduler) findBestPodGroupPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, successfulResults map[*fwk.Placement]*podGroupAlgorithmResult) (*fwk.Placement, *fwk.Status) {
	if len(successfulResults) == 1 {
		for placement := range successfulResults {
			return placement, nil
		}
	}

	placementPodGroupAssignments, placementStates := makePodGroupAssignments(successfulResults)
	return sched.findBestPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, placementPodGroupAssignments, placementStates)
}

func (sched *Scheduler) findBestCompositePodGroupPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, successfulResults map[*fwk.Placement]map[fwk.EntityKey]*podGroupAlgorithmResult) (*fwk.Placement, *fwk.Status) {
	if len(successfulResults) == 1 {
		for placement := range successfulResults {
			return placement, nil
		}
	}

	placementPodGroupAssignments, placementStates := makeCompositePodGroupAssignments(podGroupInfo, successfulResults)
	return sched.findBestPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, placementPodGroupAssignments, placementStates)
}

// evaluatePlacement runs the pod group scheduling algorithm within a single placement,
// assuming the placement in the snapshot for the duration of the simulation.
func (sched *Scheduler) evaluatePlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo, placement *fwk.Placement) *podGroupAlgorithmResult {
	klog.FromContext(ctx).V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
	evaluationStart := time.Now()
	if err := sched.nodeInfoSnapshot.AssumePlacement(placement); err != nil {
		return &podGroupAlgorithmResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
		}
	}
	placementCycleState := framework.NewCycleState()
	placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
	// Seed the per-placement state with any data a PlacementGeneratePlugin attached to this
	// placement during generation, so the plugin can read it back in later phases.
	if named, ok := podGroupCycleState.GetPlacementCycleStateForName(placement.Name).(*framework.CycleState); ok {
		named.CopyPlacementDataInto(placementCycleState)
	}
	result, placementRevertFns := sched.podGroupSchedulingDefaultAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, queuedPodGroupInfo)
	placementRevertFns.revert()

	if result.status.IsError() {
		// Error results are internal faults, not feasibility verdicts, and callers early-return
		// on them, so skip the feasible/infeasible evaluation metric.
		return result
	}

	evaluationResult := metrics.InfeasibleResult
	if result.status.IsSuccess() {
		evaluationResult = metrics.FeasibleResult
	}
	metrics.ObservePlacementEvaluation(evaluationResult, schedFwk.ProfileName(), metrics.SinceInSeconds(evaluationStart))
	return result
}

// nominatedPlacement returns the placement that should be evaluated first because it best
// matches the pods' NominatedNodeName, or nil when no placement holds any nominated node.
// A nominated node is typically set by a previous preemption cycle, so preferring its
// placement mirrors pod-by-pod scheduling, which tries the nominated node before all others.
// When nominations span placements, the one honoring the most pods' nominations is chosen.
func nominatedPlacement(placements []*fwk.Placement, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) *fwk.Placement {
	// Count nominated nodes across the pods of the currently evaluated pod group node
	// (podGroupInfo), which for CPG TAS is the CPG or PG carrying the TAS constraints, not the
	// whole hierarchy rooted at queuedPodGroupInfo.
	//
	// We count instead of using a set because pods in the group can carry different
	// NominatedNodeNames when preemption nominated them independently. Counting lets us pick the
	// placement covering the most nominated pods; a set would drop those tallies and couldn't rank
	// placements that only partially overlap the nominated nodes.
	nominatedNodeCounts := make(map[string]int)
	for _, podInfo := range queuedPodGroupInfo.QueuedPodInfos[pgKey(podGroupInfo)] {
		if nnn := podInfo.Pod.Status.NominatedNodeName; nnn != "" {
			nominatedNodeCounts[nnn]++
		}
	}
	if len(nominatedNodeCounts) == 0 {
		return nil
	}

	var best *fwk.Placement
	bestCount := 0
	for _, placement := range placements {
		count := 0
		for _, node := range placement.Nodes {
			count += nominatedNodeCounts[node.Node().Name]
		}
		if count > bestCount {
			bestCount = count
			best = placement
		}
	}
	return best
}

// findBestPlacement uses PlacementScore plugins to determine the best placement based on the scheduling results.
func (sched *Scheduler) findBestPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, placementPodGroupAssignments []*fwk.PodGroupAssignments, placementStates []fwk.PlacementCycleState) (*fwk.Placement, *fwk.Status) {
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

// makePodGroupAssignments converts scheduling results for PodGroup from candidate placements into the format
// required by PlacementScore plugins to score and select the best placement for the pod group.
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

// makePodGroupAssignments converts scheduling results for CompositePodGroup from candidate placements into the format
// required by PlacementScore plugins to score and select the best placement for the composite pod group.
func makeCompositePodGroupAssignments(pgi *framework.PodGroupInfo, successfulResults map[*fwk.Placement]map[fwk.EntityKey]*podGroupAlgorithmResult) ([]*fwk.PodGroupAssignments, []fwk.PlacementCycleState) {
	placementPodGroupAssignments := make([]*fwk.PodGroupAssignments, 0)
	placementStates := make([]fwk.PlacementCycleState, 0)
	for placement, subtreeResults := range successfulResults {
		var combinedProposedAssignments []fwk.ProposedAssignment
		for result := range successfulLeafResults(pgi, subtreeResults) {
			combinedProposedAssignments = append(combinedProposedAssignments, makeProposedAssignments(result)...)
		}
		placementPodGroupAssignments = append(placementPodGroupAssignments, &fwk.PodGroupAssignments{
			Placement:           placement,
			ProposedAssignments: combinedProposedAssignments,
		})
		placementStates = append(placementStates, subtreeResults[pgKey(pgi)].placementCycleState)
	}
	return placementPodGroupAssignments, placementStates
}

// makeProposedAssignments builds a list of proposedAssignments from the result of a pod group scheduling attempt.
func makeProposedAssignments(res *podGroupAlgorithmResult) []fwk.ProposedAssignment {
	proposedAssignments := make([]fwk.ProposedAssignment, 0)
	for _, podRes := range res.podResults {
		if podRes.status.IsSuccess() && podRes.GetNodeName() != "" {
			proposedAssignments = append(proposedAssignments, &podRes)
		}
	}
	return proposedAssignments
}

// podGroupSchedulingAlgorithm attempts to schedule pods in the pod group according to the policy and constraints and returns the scheduling result for all evaluated pods in the pod group, not necessarily all pods in the pod group.
// The returned revertFns accumulates revert functions for all scheduled pods, allowing the caller to rollback tentative reservations if the pod group scheduling cycle fails.
func (sched *Scheduler) podGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (*podGroupAlgorithmResult, revertFns) {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		return sched.podGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupCycleState, podGroupInfo, queuedPodGroupInfo)
	}
	// The non-TAS default algorithm does not evaluate placement candidates, but it
	// still runs in a single implicit placement context so placement-scoped
	// extension points can use the same state plumbing as TAS.
	placementCycleState := framework.NewCycleState()
	placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
	return sched.podGroupSchedulingDefaultAlgorithm(podGroupCycleCtx, schedFwk, placementCycleState, podGroupInfo, queuedPodGroupInfo)
}

// podGroupSchedulingRecursiveAlgorithm runs a recursive pod group scheduling algorithm.
// If the pod group info wraps a composite pod group, it will recursively invoke the algorithm on its children.
// Otherwise, the pod group info wraps a leaf pod group for which we invoke the standard pod group scheduling algorithm.
// The returned revertFns propagates revert functions from all child pod group evaluations up to the root level.
func (sched *Scheduler) podGroupSchedulingRecursiveAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) (*podGroupAlgorithmResult, revertFns) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running recursive podgroup scheduling algorithm", "rootType", podGroupInfo.Type, "root", klog.KObj(podGroupInfo))

	var algorithmResult *podGroupAlgorithmResult
	var childRevertFns revertFns
	if podGroupInfo.Type == fwk.PodGroupKeyType {
		algorithmResult, childRevertFns = sched.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, podGroupInfo, root)
		results[pgKey(podGroupInfo)] = algorithmResult
	} else {
		algorithmResult, childRevertFns = sched.compositePodGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, root, podGroupInfo, results)
	}
	return algorithmResult, childRevertFns
}

func (sched *Scheduler) compositePodGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) (result *podGroupAlgorithmResult, revertFns revertFns) {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// CPG requires TopologyAwareWorkloadScheduling feature to be enabled
	return sched.compositePodGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupCycleState, root, podGroupInfo, results)
}

// compositePodGroupSchedulingDefaultAlgorithm schedules a composite pod group by recursively scheduling
// its children. It uses PlacementFeasible plugins to verify if the composite group constraints
// remain satisfiable at each step of the recursion, aborting and reverting early if they cannot be met.
// The returned revertFns propagates revert functions from all child pod group evaluations up to the root level.
func (sched *Scheduler) compositePodGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) (result *podGroupAlgorithmResult, revertFns revertFns) {
	logger := klog.FromContext(ctx)
	defer func() {
		results[pgKey(podGroupInfo)] = result
		if result.status.IsSuccess() {
			logger.V(5).Info("Composite podgroup scheduling algorithm succeeded", "compositePodGroup", klog.KObj(podGroupInfo))
		} else {
			logger.V(5).Info("Composite podgroup scheduling algorithm failed", "compositePodGroup", klog.KObj(podGroupInfo), "status", result.status)
			revertFns.revert()
			result.anyScheduled = false
		}
	}()

	placementProgress := framework.PlacementProgress{
		Remaining: len(podGroupInfo.Children),
	}
	placementFeasibleStatus := schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo, placementProgress)

	if placementFeasibleStatus.Code() == fwk.Unschedulable || placementFeasibleStatus.Code() == fwk.Error {
		return &podGroupAlgorithmResult{
			podGroupInfo:        podGroupInfo,
			status:              placementFeasibleStatus,
			placementCycleState: placementCycleState,
		}, revertFns
	}
	anyScheduled := false
	for _, childPGInfo := range podGroupInfo.GetChildGroups() {
		childPodGroupState := framework.NewCycleState()
		childPodGroupState.SetPlacementCycleState(placementCycleState)
		childResult, childRevertFns := sched.podGroupSchedulingRecursiveAlgorithm(ctx, schedFwk, childPodGroupState, root, childPGInfo, results)
		if childResult.status.IsError() {
			return &podGroupAlgorithmResult{
				podGroupInfo:        podGroupInfo,
				status:              fwk.AsStatus(fmt.Errorf("composite pod group evaluation failed due to child error: %w", childResult.status.AsError())),
				placementCycleState: placementCycleState,
			}, revertFns
		}
		anyScheduled = anyScheduled || childResult.anyScheduled
		revertFns.append(childRevertFns)
		placementProgress.Remaining--
		if childResult.status.IsSuccess() {
			placementProgress.Scheduled++
		}
		placementFeasibleStatus = schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo, placementProgress)
		if placementFeasibleStatus.Code() == fwk.Unschedulable || placementFeasibleStatus.Code() == fwk.Error {
			break
		}
	}

	if placementFeasibleStatus.IsWait() {
		placementFeasibleStatus = fwk.NewStatus(fwk.Unschedulable, placementFeasibleStatus.Reasons()...).WithError(errPodGroupUnschedulable)
	}
	return &podGroupAlgorithmResult{
		podGroupInfo:        podGroupInfo,
		status:              placementFeasibleStatus,
		placementCycleState: placementCycleState,
		anyScheduled:        anyScheduled,
	}, revertFns
}

func pgKey(pgi *framework.PodGroupInfo) fwk.EntityKey {
	if pgi.Type == fwk.CompositePodGroupKeyType {
		return fwk.CompositePodGroupKey(pgi.Namespace, pgi.Name)
	}
	return fwk.PodGroupKey(pgi.Namespace, pgi.Name)
}

// assumeSubtreeWithRevert runs assumeAndReserveWithRevert on all pods within the subtree.
// This is needed for placement-based algorithm, because after evaluating the results for all placements,
// the chosen result needs to be assumed for the other pods in the hierarchy to see the result.
func (sched *Scheduler) assumeSubtreeWithRevert(ctx context.Context, schedFwk framework.Framework, pgi *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) (_ revertFns, err error) {
	if results == nil {
		return nil, fmt.Errorf("results for the subtree are missing")
	}

	var revertFns revertFns
	defer func() {
		if err != nil {
			revertFns.revert()
		}
	}()
	for leafResult := range successfulLeafResults(pgi, results) {
		for _, podResult := range leafResult.podResults {
			if !podResult.status.IsSuccess() || podResult.GetNodeName() == "" {
				continue
			}
			status, revert := sched.assumeAndReserveWithRevert(ctx, podResult.podCtx.state, schedFwk, podResult.podInfo, podResult.scheduleResult)
			if revert != nil {
				revertFns = append(revertFns, revert)
			}
			if !status.IsSuccess() {
				return nil, status.AsError()
			}
		}
	}

	return revertFns, nil
}

// successfulLeafResults walks the tree down to the successful leafs.
// A leaf is only deemed successful if its ancestors are also successful.
// If the results are missing for a given subtree, that subtree is skipped.
func successfulLeafResults(root *framework.PodGroupInfo, results map[fwk.EntityKey]*podGroupAlgorithmResult) iter.Seq[*podGroupAlgorithmResult] {
	return func(yield func(*podGroupAlgorithmResult) bool) {
		var walk func(pgi *framework.PodGroupInfo) bool
		walk = func(pgi *framework.PodGroupInfo) bool {
			result, ok := results[pgKey(pgi)]
			// Result may be missing because it may have been skipped due to PlacementFeasible status.
			// If the result for a given subtree is non-success (e.g. actualCount < minGroupCount), we treat all of its descendants as non-success with 0 pods scheduled.
			if !ok || !result.status.IsSuccess() {
				return true
			}

			for _, child := range pgi.Children {
				if !walk(child) {
					return false
				}
			}

			if len(result.podResults) > 0 {
				return yield(result)
			}

			return true
		}
		walk(root)
	}
}
