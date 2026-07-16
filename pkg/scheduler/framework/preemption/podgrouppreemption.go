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

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	policylisters "k8s.io/client-go/listers/policy/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"

	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

// PodGroupEvaluator is a preemption evaluator that knows how to run
// preemption where a preemptor is a pod group and the domain is the whole cluster.
type PodGroupEvaluator struct {
	Handle                         fwk.Handle
	pdbLister                      policylisters.PodDisruptionBudgetLister
	podGroupSnapshot               fwk.PodGroupLister
	enablePodGroupPreemptionPolicy bool

	Executor *Executor
}

func NewPodGroupEvaluator(fh fwk.Handle, executor *Executor, enablePodGroupPreemptionPolicy bool) *PodGroupEvaluator {
	return &PodGroupEvaluator{
		Handle:                         fh,
		pdbLister:                      fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		podGroupSnapshot:               fh.MutableSnapshotSharedLister().PodGroups(),
		enablePodGroupPreemptionPolicy: enablePodGroupPreemptionPolicy,
		Executor:                       executor,
	}
}

// Preempt implements the preemption logic where the preemptor is a pod group
// and the domain is the whole cluster. It preempts pod from the cluster
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
// and rollback the state to the backup after function is finished.
func (ev *PodGroupEvaluator) Preempt(ctx context.Context, pg *schedulingapi.PodGroup, pods []*v1.Pod, podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (*fwk.PodGroupPostFilterResult, *fwk.Status) {
	// In case of workload-aware preemption, the domain is whole cluster.
	// We do not make a snapshot of node info. Those nodes will be shared
	// with the PodGroup scheduling algorithm passed as podGroupSchedulingFunc.
	domain, err := newDomainForWorkloadPreemption(ev.Handle.MutableSnapshotSharedLister(), ev.podGroupSnapshot, "cluster-domain")
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to create domain: %w", err))
	}
	preemptor := newPodGroupPreemptor(pg, pods, ev.enablePodGroupPreemptionPolicy)
	pdbs, err := getPodDisruptionBudgets(ev.pdbLister)
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("failed to get pod disruption budgets: %w", err))
	}

	res, status := ev.selectVictimsOnDomain(ctx, preemptor, domain, pdbs, podGroupSchedulingFunc)
	if !status.IsSuccess() {
		return nil, status
	}
	status = ev.Executor.actuatePodGroupPreemption(ctx, res.victims, preemptor.pods, preemptor.podGroup, names.DefaultPreemption)
	return &fwk.PodGroupPostFilterResult{NominatingInfos: res.nominatedNodeNames}, status
}

type selectVictimsResult struct {
	nominatedNodeNames map[types.NamespacedName]*fwk.NominatingInfo
	victims            *extenderv1.Victims
}

// selectVictimsOnDomain selects a set of victims that can be removed from the
// domain in order to make enough room for the preemptor to be scheduled.
// It prioritizes victims that are not protected by a PDB.
func (ev *PodGroupEvaluator) selectVictimsOnDomain(
	ctx context.Context,
	preemptor *podGroupPreemptor,
	domain *domain,
	pdbs []*policy.PodDisruptionBudget,
	podGroupSchedulingFunc fwk.PodGroupSchedulingFunc) (*selectVictimsResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	nameToNode := make(map[string]fwk.NodeInfo)
	for _, nodeInfo := range domain.Nodes() {
		nameToNode[nodeInfo.Node().Name] = nodeInfo
	}

	mutableLister := ev.Handle.MutableSnapshotSharedLister()

	// Ensure the preemptor is eligible to preempt other pods.
	if ok, msg := ev.preemptorEligibleToPreemptOthers(ctx, preemptor, nameToNode); !ok {
		logger.V(5).Info("Preemptor is not eligible for preemption", "preemptor", klog.KObj(preemptor.podGroup), "reason", msg)
		return nil, fwk.NewStatus(fwk.Unschedulable, msg)
	}

	// removePods removes all victims from the snapshot.
	// This is called before the podGroupSchedulingFunc so it does not have
	// to update any cycle states as podGroupSchedulingFunc creates empty CycleStates
	// and fills them by running PreFilter plugins for preemptor pods.
	removePods := func(v *DomainVictim) error {
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
	addVictimPodsWithPreFilter := func(v *DomainVictim, preemptorAssignments []fwk.ProposedAssignment) error {
		for _, pi := range v.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
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
	removeVictimPodsWithPreFilter := func(v *DomainVictim, preemptorAssignments []fwk.ProposedAssignment) error {
		for _, pi := range v.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
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

	var potentialVictims []*DomainVictim
	allPossiblyAffectedVictims := domain.GetAllPossibleVictims()
	for _, victim := range allPossiblyAffectedVictims {
		if ev.isPreemptionAllowed(victim, preemptor) {
			potentialVictims = append(potentialVictims, victim)
		}
	}

	// No preemption victims found for incoming preemptor.
	if len(potentialVictims) == 0 {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming preemptor")
	}

	// PodGroup being a victim can have pods spread between current domain and other domains
	// We need to pull data about nodes from outside of the current domain
	// if a pod group has a pod on it.
	for _, victim := range potentialVictims {
		for name, nodeInfo := range victim.AffectedNodes() {
			_, ok := nameToNode[name]
			if !ok {
				nameToNode[name] = nodeInfo
			}
		}

		if err := removePods(victim); err != nil {
			return nil, fwk.AsStatus(err)
		}
	}

	// If the scheduling failed after removing all potential victims, return the status.
	podGroupAssignments, status := podGroupSchedulingFunc(ctx)
	if !status.IsSuccess() {
		return nil, status
	}

	sort.Slice(potentialVictims, func(i, j int) bool {
		return MoreImportantVictim(potentialVictims[i], potentialVictims[j], true)
	})

	violatingVictims, nonViolatingVictims := FilterVictimsWithPDBViolation(potentialVictims, pdbs)
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
	reprieveVictim := func(v *DomainVictim, preemptorAssignments []fwk.ProposedAssignment) (fits bool, err error) {
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
			nodeInfo := nameToNode[assignment.GetNodeName()]
			s := ev.Handle.RunFilterPluginsWithNominatedPods(ctx, assignment.GetCycleState(), assignment.GetPod(), nodeInfo)
			if !s.IsSuccess() {
				if err = removeVictimPodsWithPreFilter(v, preemptorAssignments); err != nil {
					return false, err
				}
				if l := logger.V(6); l.Enabled() {
					l.Info("Pods are potential preemption victims on domain", "pods", toPodNames(v.Pods()), "domain", domain.GetName())
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

	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest importance victims.
	var victimsToPreempt []*DomainVictim
	for _, violatingVictim := range violatingVictims {
		v := violatingVictim.Victim
		if fits, err := reprieveVictim(v, validAssignment); err != nil {
			return nil, fwk.AsStatus(err)
		} else if !fits {
			victimsToPreempt = append(victimsToPreempt, v)
			numViolatingVictim += violatingVictim.ViolateCount
		}
	}

	for _, v := range nonViolatingVictims {
		if fits, err := reprieveVictim(v, validAssignment); err != nil {
			return nil, fwk.AsStatus(err)
		} else if !fits {
			victimsToPreempt = append(victimsToPreempt, v)
		}
	}

	sort.Slice(victimsToPreempt, func(i, j int) bool {
		return MoreImportantVictim(victimsToPreempt[i], victimsToPreempt[j], true)
	})
	var podsToPreempt []*v1.Pod
	for _, v := range victimsToPreempt {
		for _, pi := range v.Pods() {
			podsToPreempt = append(podsToPreempt, pi.GetPod())
		}
	}

	v := &extenderv1.Victims{
		Pods: podsToPreempt,
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

	return &selectVictimsResult{nominatedNodeNames: n, victims: v}, nil
}

func toPodNames(pods []fwk.PodInfo) string {
	names := make([]string, len(pods))
	for i, p := range pods {
		names[i] = p.GetPod().Namespace + "/" + p.GetPod().Name
	}
	return strings.Join(names, ",")
}

// isPreemptionAllowed returns whether the victim residing on nodeInfo can be preempted by the preemptor
func (ev *PodGroupEvaluator) isPreemptionAllowed(victim Victim, preemptor *podGroupPreemptor) bool {
	return victim.Priority() < preemptor.priority
}

// preemptorEligibleToPreemptOthers returns one bool and one string. The bool
// indicates whether this preemptor should be considered for preempting other pods or
// not. The string includes the reason if this preemptor isn't eligible.
func (ev *PodGroupEvaluator) preemptorEligibleToPreemptOthers(_ context.Context, preemptor *podGroupPreemptor, nameToNode map[string]fwk.NodeInfo) (bool, string) {
	if preemptor.PreemptionPolicy() == schedulingapi.PreemptNever {
		return false, "not eligible due to preemptionPolicy=Never."
	}

	nominatedNodes := sets.New[string]()
	for _, pod := range preemptor.Members() {
		if len(pod.Status.NominatedNodeName) > 0 {
			nominatedNodes.Insert(pod.Status.NominatedNodeName)
		}
	}

	for nomNodeName := range nominatedNodes {
		if nodeInfo, exists := nameToNode[nomNodeName]; exists {
			for _, p := range nodeInfo.GetPods() {
				if GetPodPriority(p.GetPod(), ev.podGroupSnapshot) < preemptor.Priority() && PodTerminatingByPreemption(p.GetPod()) {
					return false, "not eligible due to a terminating pod on the nominated node."
				}
			}
		}
	}

	return true, ""
}
