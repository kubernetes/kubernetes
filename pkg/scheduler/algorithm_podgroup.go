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

	fwk "k8s.io/kube-scheduler/framework"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// PodScheduler abstracts single-pod scheduling and tentative snapshot reservation
// for PodGroupSchedulingAlgorithm.
//
// Decoupling single-pod evaluation behind this interface allows the pod group algorithm
// to orchestrate gang placement, constraints, and rollbacks without depending directly
// on the concrete SchedulingAlgorithm or its live cache and batching state.
type PodScheduler interface {
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
	// group succeeds.
	//
	// On success, it returns a nil status and a non-nil revert closure that runs Unreserve
	// plugins, forgets the pod from the snapshot, and restores its prior nomination state if
	// gang constraints fail or alternate placements are explored. On failure, any partial
	// changes are rolled back internally and the returned revert closure is nil.
	AssumeAndReserveInSnapshot(ctx context.Context, state fwk.CycleState,
		schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
		scheduleResult ScheduleResult) (*fwk.Status, func())
}

var _ PodScheduler = (*SchedulingAlgorithm)(nil)

// PodGroupSchedulingAlgorithm schedules pod groups and composite pod group hierarchies in memory.
//
// For each group, it generates and scores candidate placements (when topology-aware scheduling is enabled),
// schedules member pods one by one, and checks group-level constraints via PlacementFeasible plugins.
// Pod assignments are tentatively assumed in nodeInfoSnapshot so later pods in the group see them,
// and are rolled back at the end of the run or when a placement fails.
// Like SchedulingAlgorithm, it does not interact with the scheduling queue, run binding, or handle failures.
//
// PodGroupSchedulingAlgorithm mutates the shared nodeInfoSnapshot and is not safe for concurrent use.
type PodGroupSchedulingAlgorithm struct {
	// nodeInfoSnapshot is the point-in-time view of cluster nodes. Placement algorithms
	// inspect it to generate candidate node sets, and tentative per-pod reservations mutate
	// it directly so subsequent group members observe allocations without touching the
	// live cache.
	nodeInfoSnapshot *internalcache.Snapshot

	// podScheduler executes single-pod filtering, scoring, and tentative snapshot
	// assumptions for members of the pod group.
	podScheduler PodScheduler

	// percentageOfPlacementsToScore is the default percentage of generated candidate placements that
	// need to be evaluated as feasible before the scheduler proceeds to placement scoring.
	// It is used when a scheduling profile does not configure its own threshold.
	percentageOfPlacementsToScore int32

	// shufflePlacements randomizes candidate placement order so the placement limit picks a fair
	// subset. Defaults to randShufflePlacements; nil disables shuffling to keep tests deterministic.
	shufflePlacements func(placements []*fwk.Placement)
}

type PodGroupAlgorithmOption func(*PodGroupSchedulingAlgorithm)

// WithAlgorithmPercentageOfPlacementsToScore sets the percentage of generated candidate placements
// that need to be evaluated as feasible before the algorithm stops searching for more placements
// and proceeds to placement scoring.
// If percentageOfPlacementsToScore is set on the scheduler profile level, the value in the profile
// will be used instead of the algorithm-level percentageOfPlacementsToScore value.
// If percentageOfPlacementsToScore is set to 0, the algorithm will use an adaptive percentage
// based on the total number of nodes across all generated placements.
func WithAlgorithmPercentageOfPlacementsToScore(percentage int32) PodGroupAlgorithmOption {
	return func(a *PodGroupSchedulingAlgorithm) {
		a.percentageOfPlacementsToScore = percentage
	}
}

// WithShufflePlacements sets the function used to randomize candidate placement order
// when only a subset of placements will be evaluated. Passing nil disables shuffling.
func WithShufflePlacements(shuffle func(placements []*fwk.Placement)) PodGroupAlgorithmOption {
	return func(a *PodGroupSchedulingAlgorithm) {
		a.shufflePlacements = shuffle
	}
}

// NewPodGroupAlgorithm creates a PodGroupSchedulingAlgorithm operating on the given snapshot and PodScheduler.
func NewPodGroupAlgorithm(snapshot *internalcache.Snapshot, podScheduler PodScheduler,
	opts ...PodGroupAlgorithmOption) *PodGroupSchedulingAlgorithm {
	a := &PodGroupSchedulingAlgorithm{
		nodeInfoSnapshot:              snapshot,
		podScheduler:                  podScheduler,
		percentageOfPlacementsToScore: schedulerapi.DefaultPercentageOfPlacementsToScore,
		shufflePlacements:             randShufflePlacements,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}
