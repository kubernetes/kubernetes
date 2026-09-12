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
	"sort"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	policylisters "k8s.io/client-go/listers/policy/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/metrics"

	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodGroupEvaluator is a preemption evaluator that knows how to run
// preemption where a preemptor is a pod group across the whole cluster.
type PodGroupEvaluator struct {
	Handle                         fwk.Handle
	pdbLister                      policylisters.PodDisruptionBudgetLister
	podGroupSnapshot               fwk.PodGroupLister
	compositePodGroupSnapshot      fwk.CompositePodGroupLister
	enablePodGroupPreemptionPolicy bool
}

func NewPodGroupEvaluator(fh fwk.Handle, fts feature.Features) *PodGroupEvaluator {
	evaluator := &PodGroupEvaluator{
		Handle:                         fh,
		pdbLister:                      fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		podGroupSnapshot:               fh.MutableSnapshotSharedLister().PodGroups(),
		enablePodGroupPreemptionPolicy: fts.EnablePodGroupPreemptionPolicy,
	}
	if fts.EnableCompositePodGroup {
		evaluator.compositePodGroupSnapshot = fh.MutableSnapshotSharedLister().CompositePodGroups()
	}
	return evaluator
}

type defaultPreemptionManager struct {
	snapshot  fwk.SharedLister
	executor  fwk.PreemptionExecutor
	pdbLister policylisters.PodDisruptionBudgetLister
	fts       feature.Features
}

var _ fwk.PreemptionManager = &defaultPreemptionManager{}

// NewDefaultPreemptionManager creates a PreemptionManager using the provided handle
// and features for victim discovery and actuation.
func NewDefaultPreemptionManager(fh fwk.Handle, fts feature.Features) fwk.PreemptionManager {
	return &defaultPreemptionManager{
		snapshot:  fh.SnapshotSharedLister(),
		executor:  NewExecutor(fh, fts),
		pdbLister: fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		fts:       fts,
	}
}

type domainVictimWithPDBViolations struct {
	*DomainVictim
	numPDBViolations int
}

var _ fwk.Victim = &domainVictimWithPDBViolations{}

func (v *domainVictimWithPDBViolations) NumPDBViolations() int {
	return v.numPDBViolations
}

// prepareDomainVictims filters domain victims that are eligible for preemption by preemptorPriority,
// computes PDB violations using the provided pdbs, and orders victims first by priority, and second such that violating victims
// are evaluated first, followed by non-violating victims.
func prepareDomainVictims(victims []*DomainVictim, preemptorPriority int32, pdbs []*policy.PodDisruptionBudget) []fwk.Victim {
	var potentialVictims []*DomainVictim
	for _, victim := range victims {
		if victim.Priority() < preemptorPriority {
			potentialVictims = append(potentialVictims, victim)
		}
	}

	sort.Slice(potentialVictims, func(i, j int) bool {
		return MoreImportantVictim(potentialVictims[i], potentialVictims[j])
	})

	violatingVictims, nonViolatingVictims := FilterVictimsWithPDBViolation(potentialVictims, pdbs)
	orderedVictims := make([]fwk.Victim, 0, len(potentialVictims))
	for _, vv := range violatingVictims {
		orderedVictims = append(orderedVictims, &domainVictimWithPDBViolations{
			DomainVictim:     vv.Victim,
			numPDBViolations: vv.ViolateCount,
		})
	}
	for _, nv := range nonViolatingVictims {
		orderedVictims = append(orderedVictims, &domainVictimWithPDBViolations{
			DomainVictim:     nv,
			numPDBViolations: 0,
		})
	}
	return orderedVictims
}

// GenerateVictims creates victims with grouping determined by the hierarchy's DisruptionPolicy and ordering determined by Priority and PDB violations.
func (m *defaultPreemptionManager) GenerateVictims(ctx context.Context, pgInfo fwk.PodGroupInfo) ([]fwk.Victim, error) {
	logger := klog.FromContext(ctx)
	podGroupSnapshot := m.snapshot.PodGroups()
	compositePodGroupSnapshot := m.snapshot.CompositePodGroups()
	victims, err := getWorkloadPreemptionVictims(logger, m.snapshot, podGroupSnapshot, compositePodGroupSnapshot)
	if err != nil {
		return nil, fmt.Errorf("failed to get victims: %w", err)
	}

	pdbs, err := getPodDisruptionBudgets(m.pdbLister)
	if err != nil {
		return nil, fmt.Errorf("failed to get pod disruption budgets: %w", err)
	}

	preemptor := newPodGroupPreemptor(pgInfo, m.fts.EnablePodGroupPreemptionPolicy)
	return prepareDomainVictims(victims, preemptor.priority, pdbs), nil
}

func (m *defaultPreemptionManager) Executor() fwk.PreemptionExecutor {
	return m.executor
}

// evaluate determines the victims for preemption, without actuation.
func (ev *PodGroupEvaluator) evaluate(ctx context.Context, preemptor *podGroupPreemptor, potentialVictims []fwk.Victim, podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (res *selectVictimsResult, status *fwk.Status) {
	startTime := time.Now()
	defer func() {
		metrics.PreemptionEvaluationDuration.WithLabelValues("podgroup", status.Code().String()).Observe(metrics.SinceInSeconds(startTime))
	}()

	return ev.selectVictimsOnDomain(ctx, preemptor, potentialVictims, podGroupSchedulingFunc)
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
func (ev *PodGroupEvaluator) Preempt(ctx context.Context, pgInfo fwk.PodGroupInfo, pgSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	preemptionManager := ev.Handle.PreemptionManager()
	if preemptionManager == nil {
		return nil, fwk.AsStatus(fmt.Errorf("preemption manager is nil"))
	}

	victims, err := preemptionManager.GenerateVictims(ctx, pgInfo)
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	preemptor := newPodGroupPreemptor(pgInfo, ev.enablePodGroupPreemptionPolicy)

	// Ensure the preemptor is eligible to preempt other pods.
	if ok, msg := ev.preemptorEligibleToPreemptOthers(ctx, preemptor); !ok {
		logger.V(5).Info("Preemptor is not eligible for preemption", "preemptor", preemptor.getObj(), "type", preemptor.getType(), "reason", msg)
		return nil, fwk.NewStatus(fwk.Unschedulable, msg)
	}

	if ev.isOngoingPreemption(ctx, preemptor) {
		return &fwk.PodGroupPostFilterResult{NominatingInfos: buildCurrentNominatingInfos(preemptor)}, fwk.NewStatus(fwk.Success, "ongoing preemption on nominated nodes")
	}

	res, status := ev.evaluate(ctx, preemptor, victims, pgSchedulingFunc)
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
	nominatedNodeNames     map[types.NamespacedName]*fwk.NominatingInfo
	numPodGroupDisruptions int
	victims                *extenderv1.Victims
}

// selectVictimsOnDomain selects a set of victims that can be removed
// in order to make enough room for the preemptor to be scheduled.
// It prioritizes victims that are not protected by a PDB.
func (ev *PodGroupEvaluator) selectVictimsOnDomain(
	ctx context.Context,
	preemptor *podGroupPreemptor,
	potentialVictims []fwk.Victim,
	podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (*selectVictimsResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	mutableLister := ev.Handle.MutableSnapshotSharedLister()

	// removePods removes all victims from the snapshot.
	// This is called before the podGroupSchedulingFunc so it does not have
	// to update any cycle states as podGroupSchedulingFunc creates empty CycleStates
	// and fills them by running PreFilter plugins for preemptor pods.
	removePods := func(v fwk.Victim) error {
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
	addVictimPodsWithPreFilter := func(v fwk.Victim, preemptorAssignments []fwk.ProposedAssignment) error {
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
	removeVictimPodsWithPreFilter := func(v fwk.Victim, preemptorAssignments []fwk.ProposedAssignment) error {
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

	// No preemption victims found for incoming preemptor.
	if len(potentialVictims) == 0 {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming preemptor")
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
	reprieveVictim := func(v fwk.Victim, preemptorAssignments []fwk.ProposedAssignment) (fits bool, err error) {
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

	// Try to reprieve as many pods as possible. The provided victims are already ordered.
	var victimsToPreempt []fwk.Victim
	for _, v := range potentialVictims {
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

// preemptorEligibleToPreemptOthers returns one bool and one string. The bool
// indicates whether this preemptor should be considered for preempting other pods or
// not. The string includes the reason if this preemptor isn't eligible.
func (ev *PodGroupEvaluator) preemptorEligibleToPreemptOthers(_ context.Context, preemptor *podGroupPreemptor) (bool, string) {
	if preemptor.PreemptionPolicy() == schedulingapi.PreemptNever {
		return false, "not eligible due to preemptionPolicy=Never."
	}

	return true, ""
}

// isOngoingPreemption checks whether there is an ongoing preemption on the
// nominated nodes for pods from this pod group.
func (ev *PodGroupEvaluator) isOngoingPreemption(_ context.Context, preemptor *podGroupPreemptor) bool {
	nominatedNodes := sets.New[string]()
	for _, pod := range preemptor.Members() {
		if len(pod.Status.NominatedNodeName) > 0 {
			nominatedNodes.Insert(pod.Status.NominatedNodeName)
		}
	}

	nodeLister := ev.Handle.SnapshotSharedLister().NodeInfos()
	for nomNodeName := range nominatedNodes {
		nodeInfo, err := nodeLister.Get(nomNodeName)
		if err != nil || nodeInfo == nil {
			continue
		}
		for _, p := range nodeInfo.GetPods() {
			if GetPodPriority(p.GetPod(), ev.podGroupSnapshot, ev.compositePodGroupSnapshot) < preemptor.Priority() && PodTerminatingByPreemption(p.GetPod()) {
				return true
			}
		}
	}

	return false
}

// buildCurrentNominatingInfos builds a map of nominating infos based on
// currently set NNN for the preemptor pods
func buildCurrentNominatingInfos(preemptor *podGroupPreemptor) map[types.NamespacedName]*fwk.NominatingInfo {
	n := make(map[types.NamespacedName]*fwk.NominatingInfo)
	for _, pod := range preemptor.Members() {
		podKey := types.NamespacedName{Namespace: pod.Namespace, Name: pod.Name}
		n[podKey] = &fwk.NominatingInfo{
			NominatingMode:    fwk.ModeOverride,
			NominatedNodeName: pod.Status.NominatedNodeName,
		}
	}
	return n
}

// isGroupVictim returns true if the victim represents a PodGroup or CompositePodGroup.
func isGroupVictim(v fwk.Victim) bool {
	pods := v.Pods()
	if len(pods) == 0 {
		return false
	}
	return len(pods) > 1 || pods[0].GetPod().Spec.SchedulingGroup != nil
}
