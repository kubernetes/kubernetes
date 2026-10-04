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
	"errors"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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

// podSchedulingContext holds the precomputed data needed to handle the pod scheduling.
// Each scheduling attempt in the same pod group scheduling cycle for the same pod
// should use a new podSchedulingContext.
type podSchedulingContext struct {
	logger         klog.Logger
	state          *framework.CycleState
	podsToActivate *framework.PodsToActivate
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

// podGroupFitError describes a fit error for a PodGroup or CompositePodGroup
type podGroupFitError struct {
	// status is a PodGroup rejection status.
	status *fwk.Status
	// numPlacements is the number of generated placements.
	numPlacements int
	// unschedulablePlugins are plugins that returned Wait, Unschedulable or UnschedulableAndUnresolvable status.
	unschedulablePlugins sets.Set[string]
	// pendingPlugins are plugins that returned Pending status.
	pendingPlugins sets.Set[string]
}

// newPodGroupFitError creates a new PodGroupFitError without placement context.
func newPodGroupFitError(s *fwk.Status) *podGroupFitError {
	return newPodGroupPlacementFitError(s, 0)
}

// newPodGroupPlacementFitError creates a new PodGroupFitError within placement algorithm.
func newPodGroupPlacementFitError(s *fwk.Status, numPlacements int) *podGroupFitError {
	fe := &podGroupFitError{
		status:        s,
		numPlacements: numPlacements,
	}
	fe.addPluginStatus(s)
	return fe
}

// addPluginStatus updates the fit error with plugin status.
func (fe *podGroupFitError) addPluginStatus(s *fwk.Status) {
	if fitError, ok := errors.AsType[*podGroupFitError](s.AsError()); ok {
		// If the status holds podGroupFitError, it means that it already contains
		// the unschedulablePlugins and pendingPlugins inside.
		fe.unschedulablePlugins = fe.unschedulablePlugins.Union(fitError.unschedulablePlugins)
		fe.pendingPlugins = fe.pendingPlugins.Union(fitError.pendingPlugins)
		return
	}
	if s.Plugin() == "" {
		return
	}
	if s.IsRejected() || s.IsWait() {
		if fe.unschedulablePlugins == nil {
			fe.unschedulablePlugins = sets.New[string]()
		}
		fe.unschedulablePlugins.Insert(s.Plugin())
	}
	if s.Code() == fwk.Pending {
		if fe.pendingPlugins == nil {
			fe.pendingPlugins = sets.New[string]()
		}
		fe.pendingPlugins.Insert(s.Plugin())
	}
}

// Error returns the error message for the fit error.
// In case of non-placement algorithm's error, it will not contain any placement context.
func (fe *podGroupFitError) Error() string {
	reason := fe.status.Message()
	if err := fe.status.AsError(); err != nil {
		reason = err.Error()
	}
	if fe.numPlacements == 0 {
		return reason
	}
	return fmt.Sprintf("0/%d placements are available, first placement status: %s", fe.numPlacements, reason)
}
