/*
Copyright 2020 The Kubernetes Authors.

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

package defaultpreemption

import (
	"context"
	"fmt"
	"math/rand"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
)

// Name of the plugin used in the plugin registry and configurations.
const Name = names.DefaultPreemption

// IsEligiblePodFunc is a function which may be assigned to the DefaultPreemption plugin.
// This may implement rules/filtering around preemption eligibility, which is in addition to
// the internal requirement that the victim have lower priority than the preemptor pod.
// Any customizations should always allow system services to preempt normal pods, to avoid
// problems if system pods are unable to find space.
type IsEligiblePodFunc func(nodeInfo fwk.NodeInfo, victim preemption.Victim, preemptor *v1.Pod) bool

// MoreImportantVictimFunc is a function which may be assigned to the DefaultPreemption plugin.
// Implementations should return true if the first victim is more important than the second victim
// and the second one should be considered for preemption before the first one.
// For performance reasons, the search for nodes eligible for preemption is done by omitting all
// eligible victims from a node then checking whether the preemptor fits on the node without them,
// before adding back victims (starting from the most important) that still fit with the preemptor.
// The default behavior is to not consider pod affinity between the preemptor and the victims,
// as affinity between pods that are eligible to preempt each other isn't recommended.
type MoreImportantVictimFunc func(victim1, victim2 preemption.Victim) bool

// DefaultPreemption is a PostFilter plugin implements the preemption logic.
type DefaultPreemption struct {
	fh   fwk.Handle
	fts  feature.Features
	args config.DefaultPreemptionArgs

	Evaluator *preemption.Evaluator

	// IsEligiblePod returns whether a victim (single pod/pod group) is allowed to be preempted by a preemptor pod.
	// This filtering is in addition to the internal requirement that the victim pod have lower
	// priority than the preemptor pod. Any customizations should always allow system services
	// to preempt normal pods, to avoid problems if system pods are unable to find space.
	IsEligiblePod IsEligiblePodFunc

	MoreImportantVictim MoreImportantVictimFunc
}

var _ fwk.PostFilterPlugin = &DefaultPreemption{}
var _ fwk.PreEnqueuePlugin = &DefaultPreemption{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DefaultPreemption) Name() string {
	return Name
}

// New initializes a new plugin and returns it. The plugin type is retained to allow modification.
func New(_ context.Context, dpArgs runtime.Object, fh fwk.Handle, fts feature.Features) (*DefaultPreemption, error) {
	args, ok := dpArgs.(*config.DefaultPreemptionArgs)
	if !ok {
		return nil, fmt.Errorf("got args of type %T, want *DefaultPreemptionArgs", dpArgs)
	}
	if err := validation.ValidateDefaultPreemptionArgs(nil, args); err != nil {
		return nil, err
	}

	pl := DefaultPreemption{
		fh:   fh,
		fts:  fts,
		args: *args,
	}
	pl.Evaluator = preemption.NewEvaluator(Name, fh, &pl, fts)

	// Default behavior: No additional filtering, beyond the internal requirement that the victim
	// have lower priority than the preemptor.
	pl.IsEligiblePod = func(nodeInfo fwk.NodeInfo, victim preemption.Victim, preemptor *v1.Pod) bool {
		return true
	}

	// Default behavior, defines the order for victims sorting. The order is:
	// 1. Higher Priority.
	// 2. If WorkloadAwarePreemption is enabled: PodGroup > Single Pod.
	// 3. For single Pods: Older StartTime (longer runtime).
	// 4. For PodGroups: Larger Group Size, then Older StartTime.
	pl.MoreImportantVictim = pl.moreImportantVictim

	return &pl, nil
}

// PostFilter invoked at the postFilter extension point.
func (pl *DefaultPreemption) PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, m fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	defer func() {
		metrics.PreemptionAttempts.Inc()
	}()

	result, status := pl.Evaluator.Preempt(ctx, state, pod, m)
	msg := status.Message()
	if len(msg) > 0 {
		return result, fwk.NewStatus(status.Code(), "preemption: "+msg)
	}
	return result, status
}

func (pl *DefaultPreemption) PreEnqueue(ctx context.Context, p *v1.Pod) *fwk.Status {
	if !pl.fts.EnableAsyncPreemption {
		return nil
	}
	if pl.Evaluator.IsPodRunningPreemption(p.GetUID()) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for the preemption for this pod to be finished")
	}
	return nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *DefaultPreemption) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	if pl.fts.EnableAsyncPreemption {
		return []fwk.ClusterEventWithHint{
			// We need to register the event to tell the scheduling queue that the pod could be un-gated after some Pods' deletion.
			{Event: fwk.ClusterEvent{Resource: fwk.Pod, ActionType: fwk.Delete}, QueueingHintFn: pl.isPodSchedulableAfterPodDeletion},
		}, nil
	}

	// When the async preemption is disabled, PreEnqueue always returns nil, and hence pods never get rejected by this plugin.
	return nil, nil
}

// isPodSchedulableAfterPodDeletion returns the queueing hint for the pod after the pod deletion event,
// which always return Skip.
// The default preemption plugin is a bit tricky;
// the pods rejected by it are the ones that have run/are running the preemption asynchronously.
// And, those pods should always have the other plugins in pInfo.UnschedulablePlugins
// which failure will be resolved by the preemption.
// The reason why we return Skip here is that the preemption plugin should not make the decision of when to requeueing Pods,
// and rather, those plugins should be responsible for that.
func (pl *DefaultPreemption) isPodSchedulableAfterPodDeletion(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	return fwk.QueueSkip, nil
}

// calculateNumCandidates returns the number of candidates the FindCandidates
// method must produce from dry running based on the constraints given by
// <minCandidateNodesPercentage> and <minCandidateNodesAbsolute>. The number of
// candidates returned will never be greater than <numNodes>.
func (pl *DefaultPreemption) calculateNumCandidates(numNodes int32) int32 {
	n := (numNodes * pl.args.MinCandidateNodesPercentage) / 100
	if n < pl.args.MinCandidateNodesAbsolute {
		n = pl.args.MinCandidateNodesAbsolute
	}
	if n > numNodes {
		n = numNodes
	}
	return n
}

// getOffsetRand is a dedicated random source for GetOffsetAndNumCandidates calls.
// It defaults to rand.Int31n, but is a package variable so it can be overridden to make unit tests deterministic.
var getOffsetRand = rand.Int31n

// GetOffsetAndNumCandidates chooses a random offset and calculates the number
// of candidates that should be shortlisted for dry running preemption.
func (pl *DefaultPreemption) GetOffsetAndNumCandidates(numNodes int32) (int32, int32) {
	return getOffsetRand(numNodes), pl.calculateNumCandidates(numNodes)
}

// This function is not applicable for out-of-tree preemption plugins that exercise
// different preemption candidates on the same nominated node.
func (pl *DefaultPreemption) CandidatesToVictimsMap(candidates []preemption.Candidate) map[string]*extenderv1.Victims {
	m := make(map[string]*extenderv1.Victims, len(candidates))
	for _, c := range candidates {
		m[c.Name()] = c.Victims()
	}
	return m
}

func (pl *DefaultPreemption) runPreFilterExtensionAddPod(ctx context.Context, preemptor preemption.Preemptor, podInfoToAdd fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	for i, pod := range preemptor.Members() {
		status := pl.fh.RunPreFilterExtensionAddPod(ctx, preemptor.CycleStates()[i], pod, podInfoToAdd, nodeInfo)
		if !status.IsSuccess() {
			return status
		}
	}

	return nil
}

func (pl *DefaultPreemption) runPreFilterExtensionRemovePod(ctx context.Context, preemptor preemption.Preemptor, podInfoToRemove fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	for i, pod := range preemptor.Members() {
		status := pl.fh.RunPreFilterExtensionRemovePod(ctx, preemptor.CycleStates()[i], pod, podInfoToRemove, nodeInfo)
		if !status.IsSuccess() {
			return status
		}
	}

	return nil
}

// SelectVictimsOnDomain finds minimum set of pods on the given domain that should be preempted in order to make enough room
// for the preemptor to be scheduled.
func (pl *DefaultPreemption) SelectVictimsOnDomain(
	ctx context.Context,
	preemptor preemption.Preemptor,
	domain preemption.Domain,
	pdbs []*policy.PodDisruptionBudget) ([]*v1.Pod, int, *fwk.Status) {
	logger := klog.FromContext(ctx)
	nameToNode := make(map[string]fwk.NodeInfo)
	for _, nodeInfo := range domain.Nodes() {
		nameToNode[nodeInfo.Node().Name] = nodeInfo
	}

	removeVictim := func(pu preemption.Victim) error {
		for _, pi := range pu.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
			if err := nodeInfo.RemovePod(logger, pi.GetPod()); err != nil {
				return err
			}
			status := pl.runPreFilterExtensionRemovePod(ctx, preemptor, pi, nodeInfo)
			if !status.IsSuccess() {
				return status.AsError()
			}
		}

		return nil
	}
	addVictim := func(pu preemption.Victim) error {
		for _, pi := range pu.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
			nodeInfo.AddPodInfo(pi)
			status := pl.runPreFilterExtensionAddPod(ctx, preemptor, pi, nodeInfo)
			if !status.IsSuccess() {
				return status.AsError()
			}
		}

		return nil
	}

	var potentialVictims []preemption.Victim
	allPossiblyAffectedVictims := domain.GetAllPossibleVictims()
	for _, victim := range allPossiblyAffectedVictims {
		if pl.isPreemptionAllowedForDomain(domain, victim, preemptor) {
			potentialVictims = append(potentialVictims, victim)
		}
	}

	// No preemption victims found for incoming preemptor.
	if len(potentialVictims) == 0 {
		return nil, 0, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming preemptor")
	}

	// As the first step, remove all victims eligible for preemption from the domain.
	for _, victim := range potentialVictims {
		if err := removeVictim(victim); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// If the preemptor does not fit after removing all the eligible victims,
	// we are almost done and this domain is not suitable for preemption. The only
	// condition that we could check is if the preemptor is failing to schedule due to
	// inter-pod affinity to one or more victims, but we have decided not to
	// support this case for performance reasons. Having affinity to lower
	// importance (priority) pods is not a recommended configuration anyway.
	if status := pl.simulatePodScheduling(ctx, preemptor, domain); !status.IsSuccess() {
		return nil, 0, status
	}

	// Sort potentialVictims by descending importance, which ensures reprieve of
	// higher priority victims first.
	sort.Slice(potentialVictims, func(i, j int) bool {
		return pl.MoreImportantVictim(potentialVictims[i], potentialVictims[j])
	})

	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest importance victims.
	violatingVictims, nonViolatingVictims := filterVictimsWithPDBViolation(potentialVictims, pdbs)
	numViolatingVictim := 0
	var victims []preemption.Victim
	reprieveVictim := func(v preemption.Victim) (bool, error) {
		if err := addVictim(v); err != nil {
			return false, err
		}

		status := pl.simulatePodScheduling(ctx, preemptor, domain)
		fits := status.IsSuccess()
		if !fits {
			if err := removeVictim(v); err != nil {
				return false, err
			}
			victims = append(victims, v)
			if loggerV := logger.V(5); loggerV.Enabled() {
				var names []string
				for _, p := range v.Pods() {
					names = append(names, p.GetPod().Name)
				}
				pods := strings.Join(names, ",")
				loggerV.Info("Pods are potential preemption victims on domain", "pods", pods, "domain", domain.GetName())
			}
		}

		return fits, nil
	}

	for _, v := range violatingVictims {
		if fits, err := reprieveVictim(v); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			numViolatingVictim++
		}
	}

	// Now we try to reprieve non-violating victims.
	for _, v := range nonViolatingVictims {
		if _, err := reprieveVictim(v); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	if len(violatingVictims) != 0 && len(nonViolatingVictims) != 0 {
		sort.Slice(victims, func(i, j int) bool { return pl.MoreImportantVictim(victims[i], victims[j]) })
	}
	var victimPods []*v1.Pod
	for _, vi := range victims {
		for _, pi := range vi.Pods() {
			victimPods = append(victimPods, pi.GetPod())
		}
	}

	return victimPods, numViolatingVictim, nil
}

// PodEligibleToPreemptOthers returns one bool and one string. The bool
// indicates whether this pod should be considered for preempting other pods or
// not. The string includes the reason if this pod isn't eligible.
// There're several reasons:
//  1. The pod has a preemptionPolicy of Never.
//  2. The pod has already preempted other pods and the victims are in their graceful termination period.
//     Currently we check the node that is nominated for this pod, and as long as there are
//     terminating pods on this node, we don't attempt to preempt more pods.
func (pl *DefaultPreemption) PodEligibleToPreemptOthers(_ context.Context, pod *v1.Pod, nominatedNodeStatus *fwk.Status) (bool, string) {
	if pod.Spec.SchedulingGroup != nil && pl.fts.EnableTopologyAwareWorkloadScheduling {
		// When TAS is enabled, the default preemption logic needs to be disabled to avoid performing preemption multiple times for each topology option.
		// The TAS-compatible preemption logic will be implemented in Delayed Preemption KEP 4671 or Workload-aware preemption KEP 5710 features.
		return false, "not eligible due to placement-based pod group scheduling limitation."
	}
	if pod.Spec.PreemptionPolicy != nil && *pod.Spec.PreemptionPolicy == v1.PreemptNever {
		return false, "not eligible due to preemptionPolicy=Never."
	}

	nodeInfos := pl.fh.SnapshotSharedLister().NodeInfos()
	nomNodeName := pod.Status.NominatedNodeName
	if len(nomNodeName) > 0 {
		// If the pod's nominated node is considered as UnschedulableAndUnresolvable by the filters,
		// then the pod should be considered for preempting again.
		if nominatedNodeStatus.Code() == fwk.UnschedulableAndUnresolvable {
			return true, ""
		}

		if nodeInfo, _ := nodeInfos.Get(nomNodeName); nodeInfo != nil {
			for _, p := range nodeInfo.GetPods() {
				victim := preemption.NewVictim([]fwk.PodInfo{p}, corev1helpers.PodPriority(p.GetPod()), []fwk.NodeInfo{nodeInfo})
				if pl.isPreemptionAllowed(nodeInfo, victim, pod) && podTerminatingByPreemption(p.GetPod()) {
					// There is a terminating pod on the nominated node.
					return false, "not eligible due to a terminating pod on the nominated node."
				}
			}
		}
	}
	return true, ""
}

// isPreemptionAllowed returns whether the "victim" residing on "nodeInfo" can be preempted by the pod
func (pl *DefaultPreemption) isPreemptionAllowed(nodeInfo fwk.NodeInfo, victim preemption.Victim, pod *v1.Pod) bool {
	// The victim must have lower priority than the pod, in addition to any filtering implemented by IsEligiblePod
	return victim.Priority() < corev1helpers.PodPriority(pod) && pl.IsEligiblePod(nodeInfo, victim, pod)
}

// filterVictimsWithPDBViolation groups the given "victims" into two groups of "violatingVictims"
// and "nonViolatingVictims" based on whether their PDBs will be violated if they are
// preempted.
// This function is stable and does not change the order of received victims. So, if it
// receives a sorted list, grouping will preserve the order of the input list.
func filterVictimsWithPDBViolation(victims []preemption.Victim, pdbs []*policy.PodDisruptionBudget) (violatingVictims, nonViolatingVictims []preemption.Victim) {
	pdbsAllowed := make([]int32, len(pdbs))
	podIsViolating := func(pod *v1.Pod) bool {
		if len(pod.Labels) == 0 {
			return false
		}

		for i, pdb := range pdbs {
			if pdb.Namespace != pod.Namespace {
				continue
			}
			selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
			if err != nil {
				// This object has an invalid selector, it does not match the pod
				continue
			}
			// A PDB with a nil or empty selector matches nothing.
			if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
				continue
			}

			// Existing in DisruptedPods means it has been processed in API server,
			// we don't treat it as a violating case.
			if _, exist := pdb.Status.DisruptedPods[pod.Name]; exist {
				continue
			}
			// Only decrement the matched pdb when it's not in its <DisruptedPods>;
			// otherwise we may over-decrement the budget number.
			pdbsAllowed[i]--
			// We have found a matching PDB.
			if pdbsAllowed[i] < 0 {
				return true
			}
		}

		return false
	}

	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	for _, victim := range victims {
		isUnitViolating := false

		for _, pi := range victim.Pods() {
			if podIsViolating(pi.GetPod()) {
				isUnitViolating = true
			}
		}
		if isUnitViolating {
			violatingVictims = append(violatingVictims, victim)
		} else {
			nonViolatingVictims = append(nonViolatingVictims, victim)
		}
	}

	return violatingVictims, nonViolatingVictims
}

// OrderedScoreFuncs returns a list of ordered score functions to select preferable node where victims will be preempted.
func (pl *DefaultPreemption) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return nil
}

// isPreemptionAllowedForDomain returns whether the victim residing on domain can be preempted by the preemptor
// For now I moved the logic of checking if a victim is allowed to be preempted by a preemptor pod on domain
// to the separate method, to not leak preemptor type into SelectVictimsOnDomain method.
func (pl *DefaultPreemption) isPreemptionAllowedForDomain(domain preemption.Domain, victim preemption.Victim, preemptor preemption.Preemptor) bool {
	// TODO: Handle scenario when victim is PodGroup. In case if the algorithm will be extracted into share place
	// and used for Workload aware preemption, also need handle pod group as preemptor scenarios.
	return pl.isPreemptionAllowed(domain.Nodes()[0], victim, preemptor.Members()[0])
}

// podTerminatingByPreemption returns true if the pod is in the termination state caused by scheduler preemption.
func podTerminatingByPreemption(p *v1.Pod) bool {
	if p.DeletionTimestamp == nil {
		return false
	}

	for _, condition := range p.Status.Conditions {
		if condition.Type == v1.DisruptionTarget {
			return condition.Status == v1.ConditionTrue && condition.Reason == v1.PodReasonPreemptionByScheduler
		}
	}
	return false
}

// moreImportantVictim decides which of two preemption units is considered more critical.
//
// The comparison logic follows this strict hierarchy:
//
//  1. Priority: Higher priority units are always more important.
//
//  2. Workload Type (if WorkloadAwarePreemption is enabled):
//     Atomic workloads (PodGroups) are considered more important than individual Pods
//     of the same priority.
//
//  3. Start Time (for Single Pods):
//     If both units are single Pods, the one with the older StartTime is more important.
//     This honors "first-come, first-served".
//
//  4. Group Size (for PodGroups):
//     If both units are PodGroups, the one with more members (larger size) is considered
//     more important. This avoids the high cost of rescheduling massive jobs.
//
//  5. Start Time (Tie-breaker for PodGroups):
//     If sizes are equal, the group that started earlier (has the oldest pod)
//     is more important.
func (pl *DefaultPreemption) moreImportantVictim(vi1, vi2 preemption.Victim) bool {
	if vi1.Priority() != vi2.Priority() {
		return vi1.Priority() > vi2.Priority()
	}

	if pl.fts.EnableWorkloadAwarePreemption && vi1.IsPodGroup() != vi2.IsPodGroup() {
		return vi1.IsPodGroup()
	}

	if !vi1.IsPodGroup() {
		return vi1.EarliestStartTime().Before(vi2.EarliestStartTime())
	}

	if len(vi1.Pods()) != len(vi2.Pods()) {
		return len(vi1.Pods()) > len(vi2.Pods())
	}

	t1 := vi1.EarliestStartTime()
	t2 := vi2.EarliestStartTime()
	return t1.Before(t2)
}

// simulatePodScheduling is used to verify if the preemptor's pods (for now it is a single pod) can be successfully
// scheduled onto the target domain (for now it is a single node) given the current
// resource availability. This acts as the feasibility predicate during the preemption
// simulation, confirming whether a specific set of evictions actually creates enough space
// for the incoming workload.
func (pl *DefaultPreemption) simulatePodScheduling(ctx context.Context, preemptor preemption.Preemptor, domain preemption.Domain) *fwk.Status {
	nodes := domain.Nodes()
	pods := preemptor.Members()
	states := preemptor.CycleStates()
	// TODO: Handle scenario when preemptor is PodGroup. In case if the algorithm will be extracted into share place
	// and used for Workload aware preemption, also need handle pod group as preemptor scenarios.
	return pl.fh.RunFilterPluginsWithNominatedPods(ctx, states[0], pods[0], nodes[0])
}
