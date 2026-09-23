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

	// removePods removes all victims from the snapshot.
	// This is called before the podGroupSchedulingFunc so it does not have
	// to update any cycle states as podGroupSchedulingFunc creates empty CycleStates
	// and fills them by running PreFilter plugins for preemptor pods.
	removePods := func(v fwk.PreemptionVictim) error {
		for _, pi := range v.Pods() {
			if err := mutableLister.RemovePod(logger, pi.GetPod(), pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
		}
		return nil
	}

	// addVictimPodsWithPreFilter simulates adding back victim's pods to the snapshot
	// and calls PreFilterExtensionAddPod() for all preemptor pods's proposed valid assignments.
	// The node passed to the RunPreFilterExtensionAddPod will have the victim pod
	// added.
	addVictimPodsWithPreFilter := func(v fwk.PreemptionVictim, preemptorAssignments []fwk.ProposedAssignment) error {
		for _, pi := range v.Pods() {
			nodeInfo, err := mutableLister.NodeInfos().Get(pi.GetPod().Spec.NodeName)
			if err != nil {
				return err
			}
			if err := mutableLister.AddPod(pi, pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
			for _, assignment := range preemptorAssignments {
				status := ev.Handle.RunPreFilterExtensionAddPod(ctx, assignment.GetCycleState(), assignment.GetPod(), pi, nodeInfo)
				if !status.IsSuccess() {
					return status.AsError()
				}
			}
		}
		return nil
	}

	// removeVictimPodsWithPreFilter removes all victims from the snapshot
	// and calls PreFilterExtensionRemovePod(victim) for all preemptor pods.
	// The node passed to the RunPreFilterExtensionRemovePod will have the victim pod
	// removed.
	removeVictimPodsWithPreFilter := func(v fwk.PreemptionVictim, preemptorAssignments []fwk.ProposedAssignment) error {
		for _, pi := range v.Pods() {
			nodeInfo, err := mutableLister.NodeInfos().Get(pi.GetPod().Spec.NodeName)
			if err != nil {
				return err
			}
			if err := mutableLister.RemovePod(logger, pi.GetPod(), pi.GetPod().Spec.NodeName); err != nil {
				return err
			}
			for _, assignment := range preemptorAssignments {
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

	// Prepare podInfos for each of the assigned preemptor pods
	for _, assignment := range podGroupAssignments.ProposedAssignments {
		if assignment.GetNodeName() != "" {
			validAssignment = append(validAssignment, assignment)
		}
	}

	// reprieveVictim tries to reprieve a victim as a single unit.
	// It adds all victim's pods back to snapshot and to CycleStates of preemptor pods
	// It then goes through preemptor's proposed assignments and runs FilterPlugins for a given preemptor
	// pod on proposed node.
	// If all FilterPlugins succeed, it returns true.
	// Preemptor pods are evaluated in the same order as in the scheduling cycle.
	// This logic uses the CycleState returned for each of the preemptor pods from the
	// scheduling algorithm called on a cluster without victims.
	// This means that the CycleState for the Nth preemptor pod was created with:
	// - all previous preemptor pods assumed and reserved
	// - no knowledge of upcoming preemptor pods
	reprieveVictim := func(v fwk.PreemptionVictim, preemptorAssignments []fwk.ProposedAssignment) (fits bool, err error) {
		if err = addVictimPodsWithPreFilter(v, preemptorAssignments); err != nil {
			return false, err
		}
		cleanupFns := []func() error{}
		defer func() {
			for i := len(cleanupFns) - 1; i >= 0; i-- {
				if cleanupErr := cleanupFns[i](); cleanupErr != nil {
					err = errors.Join(err, cleanupErr)
				}
			}
		}()
		fits = true
		for _, assignment := range preemptorAssignments {
			nodeInfo, err := mutableLister.NodeInfos().Get(assignment.GetNodeName())
			if err != nil {
				return false, err
			}
			s := ev.Handle.RunFilterPluginsWithNominatedPods(ctx, assignment.GetCycleState(), assignment.GetPod(), nodeInfo)
			if !s.IsSuccess() {
				if err = removeVictimPodsWithPreFilter(v, preemptorAssignments); err != nil {
					return false, err
				}
				if l := logger.V(6); l.Enabled() {
					l.Info("Pods are potential preemption victims", "pods", toPodNames(v.Pods()))
				}
				return false, nil
			}
			// Simulate assuming a preemptor pod and reserving stateful plugins resources.
			// We do not need to add the preemptor pod to the cycle state of upcoming preemptor pods.
			// This is because the cycle state was created with them already assumed.
			if err = mutableLister.AddPod(assignment.GetPodInfo(), assignment.GetNodeName()); err != nil {
				return false, err
			}
			ev.Handle.RunReservePluginsReserve(ctx, assignment.GetCycleState(), assignment.GetPod(), nodeInfo.Node().GetName())
			cleanupFns = append(cleanupFns, func() error {
				if ev.Handle.RunReservePluginsUnreserve(ctx, assignment.GetCycleState(), assignment.GetPod(), nodeInfo.Node().GetName()); err != nil {
					return err
				}
				return mutableLister.RemovePod(logger, assignment.GetPod(), assignment.GetNodeName())
			})
		}
		return fits, nil
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
