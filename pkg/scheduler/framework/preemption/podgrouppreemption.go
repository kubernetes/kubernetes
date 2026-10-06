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

package preemption

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/metrics"

	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodGroupEvaluator is a preemption evaluator that knows how to run
// preemption where a preemptor is a pod group across the whole cluster.
type PodGroupEvaluator struct {
	Handle fwk.Handle
}

func NewPodGroupEvaluator(fh fwk.Handle) *PodGroupEvaluator {
	evaluator := &PodGroupEvaluator{
		Handle: fh,
	}
	return evaluator
}

// evaluate determines the victims for preemption, without actuation.
func (ev *PodGroupEvaluator) evaluate(ctx context.Context, potentialVictims []fwk.PreemptionVictim, podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (res *selectVictimsResult, status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.PreemptionEvaluationDuration.WithLabelValues("podgroup", status.Code().String()).Observe(metrics.SinceInSeconds(startTime))
	}()

	return ev.selectVictimsOnDomain(ctx, potentialVictims, podGroupSchedulingFunc)
}

// Preempt implements the preemption logic where the preemptor is a pod group.
// in order to make enough room for the pod group to be scheduled.
// It returns PodGroupPreemptorResult which contains the mapping of nominated nodes
// for each pod in the pod group, and a status of the whole preemption process.
// The preemption logic modifies the NodeInfo provided by a Handle
// podGroupSchedulingFunc is a function that will be run to check feasibility of a pod group
// scheduling after modifying the node state.
// It is called only once, after all victims are removed from NodeInfos.
// Then the logic tries to reprieve as many victims as possible with preemptor
// pods assumed in their place.
// The caller is expected to backup the NodeInfo before calling this function
// And rollback the state to the backup after function is finished.
func (ev *PodGroupEvaluator) Preempt(ctx context.Context, pgInfo fwk.PodGroupInfo, podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	preemptionManager := ev.Handle.PreemptionManager()

	if preemptionManager.Executor().IsPodGroupWaitingForVictims(pgInfo) {
		return &fwk.PodGroupPostFilterResult{NominatingInfos: buildCurrentNominatingInfos(pgInfo)}, fwk.NewStatus(fwk.Success, "ongoing preemption on nominated nodes")
	}

	victims, status := preemptionManager.GenerateVictims(ctx, pgInfo)
	if !status.IsSuccess() {
		return nil, status
	}
	// No preemption victims found for incoming preemptor.
	if len(victims) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "No preemption victims found for incoming preemptor")
	}

	res, status := ev.evaluate(ctx, victims, podGroupSchedulingFunc)
	if !status.IsSuccess() {
		return nil, status
	}
	candidate := &candidate{
		victims:                res.victims,
		numPodGroupDisruptions: res.numPodGroupDisruptions,
		name:                   "cluster",
	}
	status = preemptionManager.Executor().ActuatePodGroupPreemption(ctx, candidate, pgInfo, names.DefaultPreemption)
	if status.IsSuccess() {
		status = fwk.NewStatus(fwk.Success, fmt.Sprintf("found a placement for podgroup, preempting %d victims", len(res.victims.Pods)))
	}
	return &fwk.PodGroupPostFilterResult{NominatingInfos: res.nominatedNodeNames}, status
}

type selectVictimsResult struct {
	nominatedNodeNames map[types.NamespacedName]*fwk.NominatingInfo
	// numPodGroupDisruptions should only be used for metrics.
	// See [isGroupVictim].
	numPodGroupDisruptions int
	victims                *extenderv1.Victims
}

// selectVictimsOnDomain selects a set of victims that can be removed
// in order to make enough room for the preemptor to be scheduled.
// It prioritizes victims that are not protected by a PDB.
func (ev *PodGroupEvaluator) selectVictimsOnDomain(
	ctx context.Context,
	potentialVictims []fwk.PreemptionVictim,
	podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (*selectVictimsResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	mutableLister := ev.Handle.MutableSnapshotSharedLister()

	// removePods removes all victims from the snapshot without updating CycleStates.
	// This is used both before podGroupSchedulingFunc (when CycleStates do not exist yet)
	// and when rolling back a victim rejected in Phase 1 (before PreFilterExtensions run).
	removePods := func(v fwk.PreemptionVictim) error {
		for _, pi := range v.Pods() {
			if err := mutableLister.RemovePod(logger, pi.GetPod(), pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
		}
		return nil
	}

	// addPods adds all of a victim's pods back to the snapshot without running PreFilterExtensions.
	// This is sufficient for Phase 1 because node-local Filter plugins do not implement
	// PreFilterExtensions and observe existing pods on a node solely via NodeInfo.
	addPods := func(v fwk.PreemptionVictim) error {
		for _, pi := range v.Pods() {
			if err := mutableLister.AddPod(pi, pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
		}
		return nil
	}

	// addVictimPreFilterExtensions calls PreFilterExtensionAddPod() for all preemptor assignments
	// that have active PreFilterExtensions.
	// It must be called after addPods(v) so that the NodeInfo passed to RunPreFilterExtensionAddPod
	// already has the victim pod added.
	addVictimPreFilterExtensions := func(v fwk.PreemptionVictim, preFilterAssignments []fwk.ProposedAssignment) error {
		if len(preFilterAssignments) == 0 {
			return nil
		}
		for _, pi := range v.Pods() {
			nodeInfo, err := mutableLister.NodeInfos().Get(pi.GetPod().Spec.NodeName)
			if err != nil {
				return err
			}
			for _, assignment := range preFilterAssignments {
				status := ev.Handle.RunPreFilterExtensionAddPod(ctx, assignment.GetCycleState(), assignment.GetPod(), pi, nodeInfo)
				if !status.IsSuccess() {
					return status.AsError()
				}
			}
		}
		return nil
	}

	// removeVictimPodsWithPreFilter removes all of a victim's pods from the snapshot
	// and calls PreFilterExtensionRemovePod(victim) for all preemptor assignments that
	// have active PreFilterExtensions.
	// The node passed to RunPreFilterExtensionRemovePod will have the victim pod removed.
	removeVictimPodsWithPreFilter := func(v fwk.PreemptionVictim, preFilterAssignments []fwk.ProposedAssignment) error {
		for _, pi := range v.Pods() {
			nodeInfo, err := mutableLister.NodeInfos().Get(pi.GetPod().Spec.NodeName)
			if err != nil {
				return err
			}
			if err := mutableLister.RemovePod(logger, pi.GetPod(), pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
			for _, assignment := range preFilterAssignments {
				status := ev.Handle.RunPreFilterExtensionRemovePod(ctx, assignment.GetCycleState(), assignment.GetPod(), pi, nodeInfo)
				if !status.IsSuccess() {
					return status.AsError()
				}
			}
		}
		return nil
	}

	for _, victim := range potentialVictims {
		if err := removePods(victim); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}

	// If the scheduling failed after removing all potential victims, return the status.
	podGroupAssignments, status := podGroupSchedulingFunc(ctx)
	if !status.IsSuccess() {
		return nil, status
	}

	numViolatingVictim := 0

	validAssignment := make([]fwk.ProposedAssignment, 0, len(podGroupAssignments.ProposedAssignments))
	preFilterAssignments := make([]fwk.ProposedAssignment, 0, len(podGroupAssignments.ProposedAssignments))
	assignmentsByNode := make(map[string][]fwk.ProposedAssignment)

	// Prepare podInfos for each of the assigned preemptor pods
	for _, assignment := range podGroupAssignments.ProposedAssignments {
		nodeName := assignment.GetNodeName()
		if nodeName != "" {
			validAssignment = append(validAssignment, assignment)
			assignmentsByNode[nodeName] = append(assignmentsByNode[nodeName], assignment)
			if !assignment.GetCycleState().ShouldSkipAllPreFilterExtensions() {
				preFilterAssignments = append(preFilterAssignments, assignment)
			}
		}
	}

	reprieveFilter := ev.Handle.PreemptionManager().NewReprieveFilter(ctx, potentialVictims)
	if reprieveFilter == nil {
		return nil, fwk.NewStatus(fwk.Error, "got nil reprieve filter")
	}
	// reprieveVictim tries to reprieve a victim as a single unit.
	// If reprieveFilter allows reprieving the victim, it adds all victim's pods back to snapshot
	// and to CycleStates of preemptor pods.
	//
	// Preemptor pods are evaluated against the CycleState returned for each preemptor pod from the
	// scheduling algorithm called on a cluster without victims. This means the CycleState for the
	// m-th preemptor pod was created with:
	// - all previous preemptor pods (0 .. m-1) assumed and reserved
	// - no knowledge of upcoming preemptor pods (m+1 .. P-1)
	//
	// Evaluation is split into two phases:
	// - Phase 1 adds the victim's pods to mutableLister (without running PreFilterExtensions) and
	//   runs only node-local Filter plugins (those implementing NodeLocalFilterPlugin with
	//   IsNodeLocal() == true) on the nodes directly affected by the victim (assignmentsByNode[nodeName])
	//   to fail fast when a victim causes a node-local conflict (e.g. CPU/memory/ports).
	// - Phase 2 runs PreFilterExtensionAddPod for assignments with active PreFilterExtensions
	//   (preFilterAssignments) and then replays all preemptor assignments in the original
	//   scheduling-cycle order (0 .. P-1), skipping node-local Filter plugins and running only
	//   cross-node Filter plugins.
	//
	// Why evaluating affected nodes out of global order in Phase 1 and skipping node-local filters
	// in Phase 2 is safe and preserves sequential scheduling invariants:
	//  1. Zero cross-node state in node-local Filter plugins:
	//     By the contract of NodeLocalFilterPlugin, a node-local Filter plugin evaluating an
	//     assignment a_m on target node N depends solely on (a_m.Pod, a_m.CycleState's node-local
	//     PreFilter state, NodeInfo(N)). Assuming or reserving a preemptor pod a_j assigned to a
	//     different node N' != N only mutates NodeInfo(N') and cross-node plugin state; it never
	//     modifies NodeInfo(N) or a_m's node-local PreFilter state.
	//  2. Intra-node relative ordering is preserved:
	//     assignmentsByNode[N] is constructed by iterating through ProposedAssignments in the exact
	//     scheduling-cycle order (0 .. P-1). If multiple preemptor pods a_{i_1}, a_{i_2}, ..., a_{i_k}
	//     (i_1 < i_2 < ... < i_k) are assigned to the same node N, assignmentsByNode[N] preserves
	//     that exact subsequence order. Because Phase 1 adds each a_{i_r} to NodeInfo(N) and runs
	//     Reserve before evaluating a_{i_{r+1}}, at step r in Phase 1 NodeInfo(N) contains:
	//       BasePods(N) U VictimPods(N) U {a_{i_1}, ..., a_{i_{r-1}}}
	//     which is bit-for-bit identical to the NodeInfo(N) state at step i_r during a full
	//     0 .. P-1 replay (since all intermediate pods a_j not in assignmentsByNode[N] target
	//     other nodes N' != N).
	//  3. Redundancy of node-local Filter plugins in Phase 2:
	//     - For every affected node N in affectedNodes: Phase 1 has already verified that all
	//       node-local Filter plugins succeed for every assignment in assignmentsByNode[N] against
	//       the exact NodeInfo(N) state they would observe in Phase 2.
	//     - For every unaffected node N not in affectedNodes: VictimPods(N) is empty, so adding
	//       victim v did not modify NodeInfo(N). During Phase 2's 0 .. P-1 replay, at step i_r
	//       NodeInfo(N) contains BasePods(N) U {a_{i_1}, ..., a_{i_{r-1}}}, which is the exact
	//       same NodeInfo(N) and node-local CycleState that already passed all node-local Filter
	//       plugins when podGroupSchedulingFunc produced validAssignment (plus any previously
	//       reprieved victims on N, which were already validated when those victims were reprieved).
	//  4. Cross-node sequential invariants in Phase 2:
	//     Phase 2 still iterates over all preemptorAssignments in the exact 0 .. P-1 order,
	//     adding each pod to mutableLister and running ReservePluginsReserve before evaluating
	//     subsequent pods, so all cross-node Filter plugins observe the exact sequential
	//     cluster-wide state transitions.
	//
	// Why deferring PreFilterExtensions to Phase 2 and restricting them to preFilterAssignments is safe:
	//  1. Node-local Filter plugins do not use PreFilterExtensions:
	//     By the contract of NodeLocalFilterPlugin, node-local Filter plugins never implement
	//     PreFilterExtensions (PreFilterExtensions() == nil), because their PreFilter state in
	//     CycleState depends strictly on the incoming preemptor pod itself (e.g. resource requests,
	//     container ports, node affinity terms, volume claims) and is invariant to adding or
	//     removing existing pods in the cluster. Instead, node-local Filter plugins observe the
	//     victim's pods on target node N exclusively through NodeInfo(N), which is updated by
	//     addPods(v) before Phase 1.
	//  2. Fast rollback on Phase 1 rejection without CycleState churn:
	//     If Phase 1 rejects the victim (!fits), no cross-node Filter plugin is evaluated for this
	//     victim. Because addVictimPreFilterExtensions has not been called yet, rolling back the
	//     victim only requires removePods(v) on mutableLister, completely avoiding O(P)
	//     RunPreFilterExtensionAddPod and RunPreFilterExtensionRemovePod calls per rejected victim.
	//  3. Exact NodeInfo and CycleState state at Phase 2 entry:
	//     When Phase 1 succeeds, evaluateAssignments has already cleaned up all temporary preemptor
	//     assumptions on affectedNodes while the victim's pods remain in mutableLister. Calling
	//     addVictimPreFilterExtensions immediately before Phase 2 therefore passes a NodeInfo that
	//     already contains the victim's pods (satisfying the RunPreFilterExtensionAddPod contract)
	//     and updates CycleState before any cross-node Filter plugin runs in Phase 2.
	//  4. Skipping assignments with ShouldSkipAllPreFilterExtensions() == true:
	//     During RunPreFilterPlugins, any PreFilterPlugin that returns Skip is recorded in
	//     CycleState.GetSkipFilterPlugins() and does not initialize plugin state in CycleState.
	//     Both RunPreFilterExtensionAddPod and RunPreFilterExtensionRemovePod already skip every
	//     plugin pl where pl.PreFilterExtensions() == nil || state.GetSkipFilterPlugins().Has(pl.Name()).
	//     ShouldSkipAllPreFilterExtensions() is set to true by RunPreFilterPlugins if and only if
	//     every configured PreFilterPlugin with non-nil PreFilterExtensions() returned Skip. Because
	//     GetSkipFilterPlugins() is not mutated after RunPreFilterPlugins completes, invoking
	//     RunPreFilterExtensionAddPod or RunPreFilterExtensionRemovePod on an assignment with
	//     ShouldSkipAllPreFilterExtensions() == true is guaranteed to be a no-op for every victim pod.
	//     Pre-filtering validAssignment into preFilterAssignments once before the victim loop avoids
	//     O(V * P) no-op calls while preserving the exact set of PreFilterExtensions invocations.
	reprieveVictim := func(v fwk.PreemptionVictim, preemptorAssignments []fwk.ProposedAssignment) (fits bool, err error) {
		ok, err := reprieveFilter.ShouldAttemptReprieval(ctx, v)
		if err != nil {
			return false, err
		}
		if !ok {
			return false, nil
		}
		if err = addPods(v); err != nil {
			return false, err
		}
		preFilterAdded := false
		defer func() {
			if !fits {
				var rmErr error
				if preFilterAdded {
					rmErr = removeVictimPodsWithPreFilter(v, preFilterAssignments)
				} else {
					rmErr = removePods(v)
				}
				if rmErr != nil {
					err = errors.Join(err, rmErr)
				}
				if err != nil {
					return
				}
				if l := logger.V(6); l.Enabled() {
					l.Info("Pods are potential preemption victims", "pods", toPodNames(v.Pods()))
				}
			}
		}()

		evaluateAssignments := func(assignments []fwk.ProposedAssignment, mode fwk.FilterPluginExecutionMode) (ok bool, evalErr error) {
			cleanupFns := []func() error{}
			defer func() {
				for i := len(cleanupFns) - 1; i >= 0; i-- {
					if cleanupErr := cleanupFns[i](); cleanupErr != nil {
						evalErr = errors.Join(evalErr, cleanupErr)
					}
				}
			}()
			for _, assignment := range assignments {
				nodeName := assignment.GetNodeName()
				nodeInfo, err := mutableLister.NodeInfos().Get(nodeName)
				if err != nil {
					return false, err
				}

				cs := assignment.GetCycleState()
				cs.SetFilterPluginExecutionMode(mode)
				s := ev.Handle.RunFilterPluginsWithNominatedPods(ctx, cs, assignment.GetPod(), nodeInfo)
				cs.SetFilterPluginExecutionMode(fwk.FilterPluginModeAll)
				if !s.IsSuccess() {
					return false, nil
				}
				// Simulate assuming a preemptor pod and reserving stateful plugins resources.
				// We do not need to add the preemptor pod to the cycle state of upcoming preemptor pods.
				// This is because the cycle state was created with them already assumed.
				if err = mutableLister.AddPod(assignment.GetPodInfo(), nodeName); err != nil {
					return false, err
				}
				ev.Handle.RunReservePluginsReserve(ctx, cs, assignment.GetPod(), nodeName)
				cleanupFns = append(cleanupFns, func() error {
					ev.Handle.RunReservePluginsUnreserve(ctx, cs, assignment.GetPod(), nodeName)
					return mutableLister.RemovePod(logger, assignment.GetPod(), nodeName)
				})
			}
			return true, nil
		}

		affectedNodes := sets.New[string]()
		for _, pi := range v.Pods() {
			affectedNodes.Insert(pi.GetPod().Spec.NodeName)
		}

		// Phase 1: Fast-fail on nodes directly affected by the victim using only node-local Filter plugins.
		for nodeName := range affectedNodes {
			if fits, err = evaluateAssignments(assignmentsByNode[nodeName], fwk.FilterPluginModeNodeLocalOnly); err != nil || !fits {
				return fits, err
			}
		}

		// Phase 2: Update PreFilterExtensions for assignments that require them, then replay all
		// assignments in scheduling-cycle order running only non-node-local (cross-node) Filter plugins.
		if err = addVictimPreFilterExtensions(v, preFilterAssignments); err != nil {
			return false, err
		}
		preFilterAdded = true
		if fits, err = evaluateAssignments(preemptorAssignments, fwk.FilterPluginModeNonNodeLocalOnly); err != nil || !fits {
			return fits, err
		}
		if err = reprieveFilter.OnVictimReprieved(ctx, v); err != nil {
			return false, err
		}
		return true, nil
	}

	// Try to reprieve as many pods as possible. The provided victims are ordered
	// from least to most important, so we iterate in reverse to reprieve the most
	// important victims first.
	var victimsToPreempt []fwk.PreemptionVictim
	for _, v := range slices.Backward(potentialVictims) {
		if fits, err := reprieveVictim(v, validAssignment); err != nil {
			return nil, fwk.AsStatus(err)
		} else if !fits {
			victimsToPreempt = append(victimsToPreempt, v)
			numViolatingVictim += v.NumPDBViolations()
		}
	}
	numPodGroupDisruptions := 0
	var podsToPreempt []*v1.Pod
	for _, v := range victimsToPreempt {
		if isGroupVictim(v) {
			numPodGroupDisruptions++
		}
		for _, pi := range v.Pods() {
			podsToPreempt = append(podsToPreempt, pi.GetPod())
		}
	}

	v := &extenderv1.Victims{
		Pods:             podsToPreempt,
		NumPDBViolations: int64(numViolatingVictim),
	}
	n := make(map[types.NamespacedName]*fwk.NominatingInfo)
	for _, p := range validAssignment {
		pod := p.GetPod()
		podKey := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		n[podKey] = &fwk.NominatingInfo{
			NominatingMode:    fwk.ModeOverride,
			NominatedNodeName: p.GetNodeName(),
		}
	}

	return &selectVictimsResult{
		nominatedNodeNames:     n,
		victims:                v,
		numPodGroupDisruptions: numPodGroupDisruptions,
	}, nil
}

func toPodNames(pods []fwk.PodInfo) string {
	names := make([]string, len(pods))
	for i, p := range pods {
		names[i] = p.GetPod().Namespace + "/" + p.GetPod().Name
	}
	return strings.Join(names, ",")
}

// buildCurrentNominatingInfos builds a map of nominating infos based on
// currently set NNN for the preemptor pods
func buildCurrentNominatingInfos(preemptor fwk.PodGroupInfo) map[types.NamespacedName]*fwk.NominatingInfo {
	n := make(map[types.NamespacedName]*fwk.NominatingInfo)
	for _, pod := range preemptor.GetAllUnscheduledPods() {
		podKey := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		n[podKey] = &fwk.NominatingInfo{
			NominatingMode:    fwk.ModeOverride,
			NominatedNodeName: pod.Status.NominatedNodeName,
		}
	}
	return n
}

// isGroupVictim returns true if the victim represents a PodGroup or CompositePodGroup.
// Should only be called with GenericWorkload gate enabled.
// This function should only be used for metrics.
func isGroupVictim(v fwk.PreemptionVictim) bool {
	pods := v.Pods()
	if len(pods) == 0 {
		return false
	}
	// Only a single pod is checked.
	// It is only valid for the default preemption manager which groups by PGs.
	// The assumption is that custom preemption managers will only be supplied in contexts where this metric isn't recorded.
	return pods[0].GetPod().Spec.SchedulingGroup != nil
}
