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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// podScheduler abstracts single-pod scheduling and tentative snapshot reservation
// for PodGroupSchedulingAlgorithm.
//
// Decoupling single-pod evaluation behind this interface allows the pod group algorithm
// to orchestrate gang placement, constraints, and rollbacks without depending directly
// on the concrete SchedulingAlgorithm or its live cache and batching state.
type podScheduler interface {
	// SchedulePod evaluates candidate nodes for a single pod using the profile's
	// filtering and scoring plugins. It does not run single-pod PostFilter (preemption)
	// on failure, because gang preemption and feasibility decisions are handled at the
	// pod group level.
	SchedulePod(ctx context.Context, schedFramework framework.Framework,
		state fwk.CycleState, podInfo *framework.QueuedPodInfo) (result ScheduleResult, err error)

	// AssumeAndReserveInSnapshot tentatively reserves the pod on the suggested host in the
	// node snapshot rather than the live scheduler cache, running Reserve plugins.
	//
	// This ensures subsequent pods in the group observe the tentative allocation during the
	// group scheduling cycle without leaking state into the live cache before the whole
	// group succeeds. It returns a revert closure to cleanly unreserve plugins and restore
	// snapshot and nomination state if gang constraints fail or alternate placements are explored.
	AssumeAndReserveInSnapshot(ctx context.Context, state fwk.CycleState,
		schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
		scheduleResult ScheduleResult) (*fwk.Status, func())
}

var _ podScheduler = (*SchedulingAlgorithm)(nil)

// PodGroupSchedulingAlgorithm encapsulates the in-memory scheduling logic for pod groups
// and composite pod group hierarchies.
//
// It coordinates group-level constraint evaluation (PlacementFeasible plugins), optional
// topology-aware candidate placement generation and scoring, and serialized single-pod
// evaluations. Placements within the group are tentatively assumed in the node snapshot
// and rolled back if the group or any prerequisite constraint cannot be satisfied.
// Like SchedulingAlgorithm, it does not manage the scheduling queue, trigger binding,
// or execute failure handling.
//
// An instance operates directly on the shared nodeInfoSnapshot and is not safe for
// concurrent use across goroutines; invocations must be serialized by the caller.
type PodGroupSchedulingAlgorithm struct {
	// nodeInfoSnapshot is the point-in-time view of cluster nodes. Placement algorithms
	// inspect it to generate candidate node sets, and tentative per-pod reservations mutate
	// it directly so subsequent group members observe allocations without touching the
	// live cache.
	nodeInfoSnapshot *internalcache.Snapshot

	// podScheduler executes single-pod filtering, scoring, and tentative snapshot
	// assumptions for members of the pod group.
	podScheduler podScheduler
}

func NewPodGroupAlgorithm(snapshot *internalcache.Snapshot, podScheduler podScheduler) *PodGroupSchedulingAlgorithm {
	return &PodGroupSchedulingAlgorithm{nodeInfoSnapshot: snapshot, podScheduler: podScheduler}
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

// RunRootSchedulingAlgorithm orchestrates the scheduling attempt for a root pod group.
// It decides whether to evaluate a single group or recursively evaluate a composite group hierarchy
// and eventually cleans up the tentative reservations that the algorithm makes during its execution.
// The returned map aggregates scheduling results across the entire pod group hierarchy.
func (p *PodGroupSchedulingAlgorithm) RunRootSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, rootPodGroupInfo *framework.QueuedPodGroupInfo) map[fwk.EntityKey]*PodGroupScheduledResult {
	var revertFns revertFns
	defer revertFns.revert()

	if rootPodGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		result := map[fwk.EntityKey]*PodGroupScheduledResult{}
		rootResult, childRevertFns := p.podGroupSchedulingRecursiveAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo, rootPodGroupInfo.PodGroupInfo, result)
		revertFns = childRevertFns
		if rootResult.status.IsSuccess() && !rootResult.anyScheduled {
			// The framework requires at least 1 pod to be scheduled in order to return a success status.
			fitError := newPodGroupFitError(fwk.NewStatus(fwk.Unschedulable, "no pods were schedulable"))
			rootResult.status = fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
		}
		return result
	}
	result, childRevertFns := p.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, rootPodGroupInfo.PodGroupInfo, rootPodGroupInfo)
	revertFns = childRevertFns

	if result.status.IsSuccess() && !result.anyScheduled {
		// The framework requires at least 1 pod to be scheduled in order to return a success status.
		fitError := newPodGroupFitError(fwk.NewStatus(fwk.Unschedulable, "no pods were schedulable"))
		result.status = fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
	}

	return map[fwk.EntityKey]*PodGroupScheduledResult{result.podGroupInfo.GetKey(): result}
}

// podGroupSchedulingDefaultAlgorithm runs the default algorithm for scheduling a pod group.
// It tries to schedule each pod using standard filtering and scoring logic in a fixed order.
// If a pod requires preemption to be schedulable, subsequent pods in the algorithm
// treat that pod as already scheduled on that node with victims being already removed in memory.
// The returned revertFns accumulates revert functions for all scheduled pods, allowing the caller
// to rollback tentative reservations if the pod group scheduling cycle fails.
func (p *PodGroupSchedulingAlgorithm) podGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (result *PodGroupScheduledResult, revertFns revertFns) {
	defer func() {
		if !result.status.IsSuccess() {
			revertFns.revert()
			result.anyScheduled = false
		}
	}()

	// Retrieve the queued podinfos for the given pod group from the root queuedPodGroupInfo.
	queuedPodInfos := queuedPodGroupInfo.QueuedPodInfos[podGroupInfo.GetKey()]
	result = &PodGroupScheduledResult{
		podGroupInfo:        podGroupInfo,
		podResults:          make([]algorithmResult, 0, len(queuedPodInfos)),
		status:              fwk.NewStatus(fwk.Unschedulable),
		waitingOnPreemption: false,
		placementCycleState: placementCycleState,
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running a pod group scheduling algorithm", "podGroup", klog.KObj(podGroupInfo), "unscheduledPodsCount", len(queuedPodInfos))

	podGroupState, err := p.nodeInfoSnapshot.PodGroupStates().Get(podGroupInfo.GetNamespace(), podGroupInfo.GetName())
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
	proceed, placementFeasibleStatus := podGroupPotentiallyFeasible(ctx, schedFwk, placementCycleState, podGroupInfo, placementProgress)
	result.status = placementFeasibleStatus
	if !proceed {
		// PlacementFeasible plugins said not to proceed to the subsequent pods.
		// Exit early from the pod group algorithm.
		return result, nil
	}

	anyScheduled := false
	for _, podInfo := range queuedPodInfos {
		podResult, revertFn := p.podGroupPodSchedulingAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, podInfo)
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
		proceed, placementFeasibleStatus := podGroupPotentiallyFeasible(ctx, schedFwk, placementCycleState, podGroupInfo, placementProgress)
		result.status = placementFeasibleStatus
		if !proceed {
			// PlacementFeasible plugins said not to proceed to the subsequent pods.
			// We can stop the scheduling loop early.
			break
		}
		anyScheduled = anyScheduled || podResult.status.IsSuccess()
	}

	result.anyScheduled = anyScheduled
	return result, revertFns
}

// podGroupPotentiallyFeasible runs placement feasible plugins and returns a status indicating whether
// the (composite) pod group can meet its constraints based on the placement progress.
// It returns true when the scheduling should proceed, false otherwise.
// Returned status is modified to be fine to be copied directly to the PodGroupScheduledResult.
func podGroupPotentiallyFeasible(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, placementProgress framework.PlacementProgress) (bool, *fwk.Status) {
	status := schedFwk.RunPlacementFeasiblePlugins(ctx, placementCycleState, podGroupInfo, placementProgress)
	switch status.Code() {
	case fwk.Error:
		// Do not evaluate more children if PlacementFeasible plugins return error or unexpected status.
		return false, fwk.AsStatus(fmt.Errorf("failed to evaluate placement feasibility: %w", status.AsError()))
	case fwk.Unschedulable:
		// Unschedulable from PlacementFeasible plugins indicates that the (composite) pod group
		// cannot meet its constraints, even if we succeed in scheduling all the children.
		fitError := newPodGroupFitError(status)
		return false, fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
	case fwk.Wait:
		// Wait status indicates the scheduling should proceed,
		// while at the same time, if it's a terminal status, the outcome should be Unschedulable.
		fitError := newPodGroupFitError(status)
		return true, fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
	default:
		return true, status
	}
}

// podGroupPodSchedulingAlgorithm runs a scheduling algorithm for individual pod from a pod group.
// It returns the algorithm result together with the revert function.
// The returned revert function rolls back tentative node reservations for the pod if the overall
// pod group fails to schedule.
//
// Pods of a pod group are assumed into the snapshot rather than the cache: the group's placement
// stays tentative until the whole group is submitted, and a snapshot assume is dropped by the
// next UpdateSnapshot instead of outliving the cycle.
func (p *PodGroupSchedulingAlgorithm) podGroupPodSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework,
	placementCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo,
	podInfo *framework.QueuedPodInfo) (algorithmResult, func()) {

	pod := podInfo.Pod
	podCtx := initPodSchedulingContext(ctx, pod, placementCycleState)
	logger := podCtx.logger
	ctx = klog.NewContext(ctx, logger)
	start := time.Now()

	logger.V(4).Info("Attempting to schedule a pod belonging to a pod group",
		"podGroup", klog.KObj(podGroupInfo), "pod", klog.KObj(pod))

	scheduleResult, status := p.schedulePod(ctx, podCtx.state, schedFwk, podInfo)
	var revertFn func()
	if status.IsSuccess() {
		var assumeStatus *fwk.Status
		assumeStatus, revertFn = p.podScheduler.AssumeAndReserveInSnapshot(
			ctx, podCtx.state, schedFwk, podInfo, scheduleResult)
		if !assumeStatus.IsSuccess() {
			// The evaluation succeeded but the placement could not be held: drop the
			// result and clear the nomination, as the single-pod cycle does.
			scheduleResult = ScheduleResult{nominatingInfo: clearNominatedNode}
			status = assumeStatus
		}
	}

	return algorithmResult{
		podInfo:            podInfo,
		scheduleResult:     scheduleResult,
		podCtx:             podCtx,
		schedulingDuration: time.Since(start),
		status:             status,
	}, revertFn
}

// schedulePod runs filtering and scoring for an individual pod in a pod group cycle,
// translating SchedulePod results and errors into framework Status without running
// single-pod PostFilter (preemption).
func (p *PodGroupSchedulingAlgorithm) schedulePod(
	ctx context.Context,
	state fwk.CycleState,
	schedFramework framework.Framework,
	podInfo *framework.QueuedPodInfo,
) (ScheduleResult, *fwk.Status) {
	logger := klog.FromContext(ctx)
	scheduleResult, err := p.podScheduler.SchedulePod(ctx, schedFramework, state, podInfo)

	if err != nil {
		if err == ErrNoNodesAvailable {
			status := fwk.NewStatus(fwk.UnschedulableAndUnresolvable).WithError(err)
			return ScheduleResult{nominatingInfo: clearNominatedNode}, status
		}

		if _, ok := err.(*framework.FitError); !ok {
			logger.Error(err, "Error selecting node for pod", "pod", klog.KObj(podInfo.Pod))
			return ScheduleResult{nominatingInfo: clearNominatedNode}, fwk.AsStatus(err)
		}

		return ScheduleResult{nominatingInfo: nil}, fwk.NewStatus(fwk.Unschedulable).WithError(err)
	}

	return scheduleResult, nil
}

// podGroupSchedulingPlacementAlgorithm tries several different combinations for scheduling the pod group and selects the best one.
// First it runs placement generator plugins to create a list of placements.
// Placement is a set of nodes that will be considered when scheduling a pod group.
// Then for each placement it tries to schedule the pod group through podGroupSchedulingDefaultAlgorithm.
// Finally, it runs placement scorer plugins to select the best placement.
func (p *PodGroupSchedulingAlgorithm) podGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (finalResult *PodGroupScheduledResult, revertFns revertFns) {
	logger := klog.FromContext(ctx)
	allNodes, err := p.nodeInfoSnapshot.ListNodesInPlacement()
	if err != nil {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}, nil
	}

	// For now, always record plugin metrics until we understand its impact on performance.
	podGroupCycleState.SetRecordPluginMetrics(true)
	placements, status := schedFwk.RunPlacementGeneratePlugins(ctx, podGroupCycleState, podGroupInfo, allNodes)
	if !status.IsSuccess() {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}
	metrics.RecordGeneratedPlacements(schedFwk.ProfileName(), len(placements))

	var anyResult *PodGroupScheduledResult
	successfulResults := make(map[*fwk.Placement]*PodGroupScheduledResult)

	parentPlacement := p.nodeInfoSnapshot.GetPlacement()
	defer func() {
		p.nodeInfoSnapshot.ForgetPlacement()
		err := p.nodeInfoSnapshot.AssumePlacement(parentPlacement)
		if err != nil {
			finalResult.status = fwk.AsStatus(fmt.Errorf("failed to restore parent pod group placement: %w", err))
			revertFns.revert()
		}
	}()

	for _, placement := range placements {
		logger.V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
		evaluationStart := time.Now()
		err := p.nodeInfoSnapshot.AssumePlacement(placement)
		if err != nil {
			return &PodGroupScheduledResult{
				podGroupInfo: podGroupInfo,
				status:       fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
			}, nil
		}
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
		result, placementRevertFns := p.podGroupSchedulingDefaultAlgorithm(ctx, schedFwk, placementCycleState, podGroupInfo, queuedPodGroupInfo)
		placementRevertFns.revert()

		if result.status.IsError() {
			return result, nil
		}

		if anyResult == nil {
			anyResult = result
		}

		// Errors are excluded by the early return above since they are internal
		// faults, not feasibility results.
		evaluationResult := metrics.InfeasibleResult
		if result.status.IsSuccess() {
			evaluationResult = metrics.FeasibleResult
			successfulResults[placement] = result
		}
		metrics.ObservePlacementEvaluation(evaluationResult, schedFwk.ProfileName(), metrics.SinceInSeconds(evaluationStart))
	}

	if len(successfulResults) == 0 {
		// We need to send events and set the status for pods in case all simulations were infeasible.
		// The selection of which simulation we report is arbitrary for now, but may change in the future.
		fitError := newPodGroupPlacementFitError(anyResult.status, len(placements))
		anyResult.status = fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
		return anyResult, nil
	}

	bestPlacement, status := p.findBestPodGroupPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, successfulResults)
	if !status.IsSuccess() {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}
	bestResult := successfulResults[bestPlacement]

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		revertFns, err = p.assumeSubtreeWithRevert(ctx, schedFwk, podGroupInfo, map[fwk.EntityKey]*PodGroupScheduledResult{podGroupInfo.GetKey(): bestResult})
		if err != nil {
			return &PodGroupScheduledResult{
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
func (p *PodGroupSchedulingAlgorithm) compositePodGroupSchedulingPlacementAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) (finalResult *PodGroupScheduledResult, revertFns revertFns) {
	defer func() {
		results[podGroupInfo.GetKey()] = finalResult
	}()
	logger := klog.FromContext(ctx)
	allNodes, err := p.nodeInfoSnapshot.ListNodesInPlacement()
	if err != nil {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to list node infos: %w", err)),
		}, nil
	}

	// For now, always record plugin metrics until we understand its impact on performance.
	podGroupCycleState.SetRecordPluginMetrics(true)
	placements, status := schedFwk.RunPlacementGeneratePlugins(ctx, podGroupCycleState, podGroupInfo, allNodes)
	if !status.IsSuccess() {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}

	var anyResultSubtree map[fwk.EntityKey]*PodGroupScheduledResult
	successfulResults := make(map[*fwk.Placement]map[fwk.EntityKey]*PodGroupScheduledResult)

	parentPlacement := p.nodeInfoSnapshot.GetPlacement()
	defer func() {
		p.nodeInfoSnapshot.ForgetPlacement()
		err := p.nodeInfoSnapshot.AssumePlacement(parentPlacement)
		if err != nil {
			finalResult.status = fwk.AsStatus(fmt.Errorf("failed to restore parent pod group placement: %w", err))
			revertFns.revert()
		}
	}()

	for _, placement := range placements {
		logger.V(4).Info("Assuming placement in snapshot", "placement", placement.Name)
		err := p.nodeInfoSnapshot.AssumePlacement(placement)
		if err != nil {
			return &PodGroupScheduledResult{
				podGroupInfo: podGroupInfo,
				status:       fwk.AsStatus(fmt.Errorf("failed to assume pod group placement: %w", err)),
			}, nil
		}
		placementCycleState := framework.NewCycleState()
		placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
		subtreeResult := map[fwk.EntityKey]*PodGroupScheduledResult{}
		result, placementRevertFns := p.compositePodGroupSchedulingDefaultAlgorithm(ctx, schedFwk, placementCycleState, root, podGroupInfo, subtreeResult)
		placementRevertFns.revert()

		if result.status.IsError() {
			// It is critical to copy the entire subtreeResult into results.
			// If omitted, the pod results are reconstructed later using the generic parent error
			// (*podGroupFitError) rather than their original *framework.FitError.
			maps.Copy(results, subtreeResult)
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
		anyResultRoot := anyResultSubtree[podGroupInfo.GetKey()]
		fitError := newPodGroupPlacementFitError(anyResultRoot.status, len(placements))
		anyResultRoot.status = fwk.NewStatus(fwk.Unschedulable).WithError(fitError)
		// It is critical to copy the entire anyResultSubtree into results.
		// If omitted, the pod results are reconstructed later using the generic parent error
		// (*podGroupFitError) rather than their original *framework.FitError.
		// Losing the FitError means we lose the UnschedulablePlugins for each pod,
		// which breaks the QueueingHints.
		maps.Copy(results, anyResultSubtree)
		return anyResultRoot, nil
	}

	bestPlacement, status := p.findBestCompositePodGroupPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, successfulResults)
	if !status.IsSuccess() {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       status,
		}, nil
	}

	bestResult := successfulResults[bestPlacement]

	revertFns, err = p.assumeSubtreeWithRevert(ctx, schedFwk, podGroupInfo, bestResult)
	if err != nil {
		return &PodGroupScheduledResult{
			podGroupInfo: podGroupInfo,
			status:       fwk.AsStatus(fmt.Errorf("failed to assume the subtree: %w", err)),
		}, nil
	}
	maps.Copy(results, bestResult)

	return bestResult[podGroupInfo.GetKey()], revertFns
}

// findBestPodGroupPlacement selects the highest-scoring placement for a pod group from the feasible candidates.
// If exactly one candidate placement succeeded, it fast-paths to avoid scoring plugin overhead; otherwise, it converts
// the results into assignments and delegates to PlacementScore plugins via findBestPlacement.
func (p *PodGroupSchedulingAlgorithm) findBestPodGroupPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, successfulResults map[*fwk.Placement]*PodGroupScheduledResult) (*fwk.Placement, *fwk.Status) {
	if len(successfulResults) == 1 {
		for placement := range successfulResults {
			return placement, nil
		}
	}

	placementPodGroupAssignments, placementStates := makePodGroupAssignments(successfulResults)
	return p.findBestPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, placementPodGroupAssignments, placementStates)
}

// findBestCompositePodGroupPlacement selects the highest-scoring placement for a composite pod group from the feasible candidates.
// If exactly one candidate placement succeeded across the hierarchy, it fast-paths to avoid scoring plugin overhead; otherwise, it
// aggregates assignments across all leaf pod groups in the subtree and delegates to PlacementScore plugins via findBestPlacement.
func (p *PodGroupSchedulingAlgorithm) findBestCompositePodGroupPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, successfulResults map[*fwk.Placement]map[fwk.EntityKey]*PodGroupScheduledResult) (*fwk.Placement, *fwk.Status) {
	if len(successfulResults) == 1 {
		for placement := range successfulResults {
			return placement, nil
		}
	}

	placementPodGroupAssignments, placementStates := makeCompositePodGroupAssignments(podGroupInfo, successfulResults)
	return p.findBestPlacement(ctx, schedFwk, podGroupCycleState, podGroupInfo, placementPodGroupAssignments, placementStates)
}

// findBestPlacement uses PlacementScore plugins to determine the best placement based on the scheduling results.
func (p *PodGroupSchedulingAlgorithm) findBestPlacement(ctx context.Context, schedFwk framework.Framework, podGroupCycleState fwk.PodGroupCycleState, podGroupInfo *framework.PodGroupInfo, placementPodGroupAssignments []*fwk.PodGroupAssignments, placementStates []fwk.PlacementCycleState) (*fwk.Placement, *fwk.Status) {
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
func makePodGroupAssignments(successfulResults map[*fwk.Placement]*PodGroupScheduledResult) ([]*fwk.PodGroupAssignments, []fwk.PlacementCycleState) {
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
func makeCompositePodGroupAssignments(pgi *framework.PodGroupInfo, successfulResults map[*fwk.Placement]map[fwk.EntityKey]*PodGroupScheduledResult) ([]*fwk.PodGroupAssignments, []fwk.PlacementCycleState) {
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
		placementStates = append(placementStates, subtreeResults[pgi.GetKey()].placementCycleState)
	}
	return placementPodGroupAssignments, placementStates
}

// makeProposedAssignments builds a list of proposedAssignments from the result of a pod group scheduling attempt.
func makeProposedAssignments(res *PodGroupScheduledResult) []fwk.ProposedAssignment {
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
func (p *PodGroupSchedulingAlgorithm) podGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, podGroupInfo *framework.PodGroupInfo, queuedPodGroupInfo *framework.QueuedPodGroupInfo) (*PodGroupScheduledResult, revertFns) {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	if utilfeature.DefaultFeatureGate.Enabled(features.TopologyAwareWorkloadScheduling) {
		return p.podGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupCycleState, podGroupInfo, queuedPodGroupInfo)
	}

	// The non-TAS default algorithm does not evaluate placement candidates, but it
	// still runs in a single implicit placement context so placement-scoped
	// extension points can use the same state plumbing as TAS.
	placementCycleState := framework.NewCycleState()
	placementCycleState.SetPodGroupSchedulingCycle(podGroupCycleState)
	return p.podGroupSchedulingDefaultAlgorithm(podGroupCycleCtx, schedFwk, placementCycleState, podGroupInfo, queuedPodGroupInfo)
}

// podGroupSchedulingRecursiveAlgorithm runs a recursive pod group scheduling algorithm.
// If the pod group info wraps a composite pod group, it will recursively invoke the algorithm on its children.
// Otherwise, the pod group info wraps a leaf pod group for which we invoke the standard pod group scheduling algorithm.
// The returned revertFns propagates revert functions from all child pod group evaluations up to the root level.
func (p *PodGroupSchedulingAlgorithm) podGroupSchedulingRecursiveAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) (*PodGroupScheduledResult, revertFns) {
	logger := klog.FromContext(ctx)
	logger.V(5).Info("Running recursive podgroup scheduling algorithm", "rootType", podGroupInfo.GetType(), "root", klog.KObj(podGroupInfo))

	var algorithmResult *PodGroupScheduledResult
	var childRevertFns revertFns
	if podGroupInfo.GetType() == fwk.PodGroupKeyType {
		algorithmResult, childRevertFns = p.podGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, podGroupInfo, root)
		results[podGroupInfo.GetKey()] = algorithmResult
	} else {
		algorithmResult, childRevertFns = p.compositePodGroupSchedulingAlgorithm(ctx, schedFwk, podGroupCycleState, root, podGroupInfo, results)
	}
	return algorithmResult, childRevertFns
}

// compositePodGroupSchedulingAlgorithm executes the scheduling cycle for a composite pod group by evaluating candidate placements.
// Since composite pod groups require topology-aware placement, it runs exclusively via the placement algorithm within a scoped
// cancellable context. The returned revertFns accumulates revert functions across the child hierarchy, allowing the caller to roll
// back tentative reservations if the cycle fails.
func (p *PodGroupSchedulingAlgorithm) compositePodGroupSchedulingAlgorithm(ctx context.Context, schedFwk framework.Framework, podGroupCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) (result *PodGroupScheduledResult, revertFns revertFns) {
	podGroupCycleCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	// CPG requires TopologyAwareWorkloadScheduling feature to be enabled
	return p.compositePodGroupSchedulingPlacementAlgorithm(podGroupCycleCtx, schedFwk, podGroupCycleState, root, podGroupInfo, results)
}

// compositePodGroupSchedulingDefaultAlgorithm schedules a composite pod group by recursively scheduling
// its children. It uses PlacementFeasible plugins to verify if the composite group constraints
// remain satisfiable at each step of the recursion, aborting and reverting early if they cannot be met.
// The returned revertFns propagates revert functions from all child pod group evaluations up to the root level.
func (p *PodGroupSchedulingAlgorithm) compositePodGroupSchedulingDefaultAlgorithm(ctx context.Context, schedFwk framework.Framework, placementCycleState *framework.CycleState, root *framework.QueuedPodGroupInfo, podGroupInfo *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) (result *PodGroupScheduledResult, revertFns revertFns) {
	logger := klog.FromContext(ctx)
	defer func() {
		results[podGroupInfo.GetKey()] = result
		if result.status.IsSuccess() {
			logger.V(5).Info("Composite podgroup scheduling algorithm succeeded", "compositePodGroup", klog.KObj(podGroupInfo))
		} else {
			logger.V(5).Info("Composite podgroup scheduling algorithm failed", "compositePodGroup", klog.KObj(podGroupInfo), "status", result.status)
			revertFns.revert()
			result.anyScheduled = false
		}
	}()

	// Run PlacementFeasible plugins to check if the composite pod group can meet its constraints
	// before even attempting to schedule any children.
	placementProgress := framework.PlacementProgress{
		Remaining: len(podGroupInfo.Children),
	}
	proceed, placementFeasibleStatus := podGroupPotentiallyFeasible(ctx, schedFwk, placementCycleState, podGroupInfo, placementProgress)
	if !proceed {
		// PlacementFeasible plugins said not to proceed to the subsequent children.
		// Exit early from the pod group algorithm.
		return &PodGroupScheduledResult{
			podGroupInfo:        podGroupInfo,
			status:              placementFeasibleStatus,
			placementCycleState: placementCycleState,
		}, revertFns
	}

	anyScheduled := false
	for _, childPGInfo := range podGroupInfo.GetChildGroups() {
		childPodGroupState := framework.NewCycleState()
		childPodGroupState.SetPlacementCycleState(placementCycleState)
		childResult, childRevertFns := p.podGroupSchedulingRecursiveAlgorithm(ctx, schedFwk, childPodGroupState, root, childPGInfo, results)
		if childResult.status.IsError() {
			return &PodGroupScheduledResult{
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
		proceed, placementFeasibleStatus = podGroupPotentiallyFeasible(ctx, schedFwk, placementCycleState, podGroupInfo, placementProgress)
		if !proceed {
			// PlacementFeasible plugins said not to proceed to the subsequent children.
			// We can stop the scheduling loop early.
			break
		}
	}

	return &PodGroupScheduledResult{
		podGroupInfo:        podGroupInfo,
		status:              placementFeasibleStatus,
		placementCycleState: placementCycleState,
		anyScheduled:        anyScheduled,
	}, revertFns
}

// assumeSubtreeWithRevert runs assumeAndReserveWithRevert on all pods within the subtree.
// This is needed for placement-based algorithm, because after evaluating the results for all placements,
// the chosen result needs to be assumed for the other pods in the hierarchy to see the result.
func (p *PodGroupSchedulingAlgorithm) assumeSubtreeWithRevert(ctx context.Context, schedFwk framework.Framework, pgi *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) (_ revertFns, err error) {
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
			status, revert := p.podScheduler.AssumeAndReserveInSnapshot(ctx, podResult.podCtx.state, schedFwk, podResult.podInfo, podResult.scheduleResult)
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
func successfulLeafResults(root *framework.PodGroupInfo, results map[fwk.EntityKey]*PodGroupScheduledResult) iter.Seq[*PodGroupScheduledResult] {
	return func(yield func(*PodGroupScheduledResult) bool) {
		var walk func(pgi *framework.PodGroupInfo) bool
		walk = func(pgi *framework.PodGroupInfo) bool {
			result, ok := results[pgi.GetKey()]
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
