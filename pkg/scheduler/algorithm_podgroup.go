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
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
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
