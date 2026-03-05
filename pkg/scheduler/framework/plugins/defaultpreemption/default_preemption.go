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
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Name of the plugin used in the plugin registry and configurations.
const Name = names.DefaultPreemption

// IsEligiblePreemptorFunc is a function which may be assigned to the DefaultPreemption plugin.
// It determines whether the incoming preemptor (whether a single Pod or a collective PodGroup)
// is allowed to initiate preemption against existing victims.
// Unlike IsEligiblePodFunc, this is invoked after domains are built and handles the
// more complex logic of workload-aware preemption.
type IsEligiblePreemptorFunc func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool

// MoreImportantVictimFunc is a function which may be assigned to the DefaultPreemption plugin.
// Implementations should return true if the first victim is more important than the second victim
// and the second one should be considered for preemption before the first one.
// For performance reasons, the search for nodes eligible for preemption is done by omitting all
// eligible victims from a node then checking whether the preemptor fits on the node without them,
// before adding back victims (starting from the most important) that still fit with the preemptor.
// The default behavior is to not consider pod affinity between the preemptor and the victims,
// as affinity between pods that are eligible to preempt each other isn't recommended.
type MoreImportantVictimFunc func(victim1, victim2 preemption.PreemptionUnit) bool

// SimulatePodSchedulingFunc is a function used to verify if the preemptor's pods can be successfully
// scheduled onto the target domain (a single Node or a set of Nodes) given the current
// resource availability. This acts as the feasibility predicate during the preemption
// simulation, confirming whether a specific set of evictions actually creates enough space
// for the incoming workload.
type SimulatePodSchedulingFunc func(ctx context.Context,
	preemptor preemption.Preemptor,
	domain preemption.Domain) *fwk.Status

// DefaultPreemption is a PostFilter plugin implements the preemption logic.
type DefaultPreemption struct {
	fh        fwk.Handle
	fts       feature.Features
	args      config.DefaultPreemptionArgs
	Evaluator *preemption.Evaluator

	IsEligiblePreemptor IsEligiblePreemptorFunc

	MoreImportantVictim MoreImportantVictimFunc

	SimulatePodScheduling SimulatePodSchedulingFunc
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
	evaluator := preemption.NewEvaluator(Name, fh, &pl, fts)
	pl.Evaluator = evaluator

	pl.IsEligiblePreemptor = func(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
		return true
	}

	pl.MoreImportantVictim = pl.moreImportantVictim

	pl.SimulatePodScheduling = func(
		ctx context.Context,
		preemptor preemption.Preemptor,
		domain preemption.Domain,
	) *fwk.Status {
		nodes := domain.Nodes()
		pods := preemptor.Members()
		states := preemptor.CycleStates()

		if len(pods) == 1 && len(nodes) == 1 {
			return pl.fh.RunFilterPluginsWithNominatedPods(ctx, states[0], pods[0], nodes[0])
		}

		// TODO: Adapt this logic to support PodGroups once Workload Scheduling is implemented. https://github.com/kubernetes/kubernetes/pull/136618
		return fwk.AsStatus(fmt.Errorf("workload scheduling is not implemented yet"))
	}

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

func (pl *DefaultPreemption) runPreFilterExtensionAddPod(ctx context.Context, preemptor preemption.Preemptor, piToAction fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	for i, pod := range preemptor.Members() {
		status := pl.fh.RunPreFilterExtensionAddPod(ctx, preemptor.CycleStates()[i], pod, piToAction, nodeInfo)
		if !status.IsSuccess() {
			return status
		}
	}

	return nil
}

func (pl *DefaultPreemption) runPreFilterExtensionRemovePod(ctx context.Context, preemptor preemption.Preemptor, piToAction fwk.PodInfo, nodeInfo fwk.NodeInfo) *fwk.Status {
	for i, pod := range preemptor.Members() {
		status := pl.fh.RunPreFilterExtensionRemovePod(ctx, preemptor.CycleStates()[i], pod, piToAction, nodeInfo)
		if !status.IsSuccess() {
			return status
		}
	}

	return nil
}

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

	removePods := func(pu preemption.PreemptionUnit) error {
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
	addPods := func(pu preemption.PreemptionUnit) error {
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

	var potentialVictims []preemption.PreemptionUnit
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

	for _, victim := range potentialVictims {
		// If a Victim is a PodGroup spanning multiple nodes, some affected nodes
		// might be outside of Domain. We add them to nameToNode map to ensure
		// they are considered during preemption.
		for name, nodeInfo := range victim.AffectedNodes() {
			if _, ok := nameToNode[name]; !ok {
				nameToNode[name] = nodeInfo
			}
		}

		if err := removePods(victim); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	if status := pl.SimulatePodScheduling(ctx, preemptor, domain); !status.IsSuccess() {
		return nil, 0, status
	}

	sort.Slice(potentialVictims, func(i, j int) bool {
		return pl.MoreImportantVictim(potentialVictims[i], potentialVictims[j])
	})

	violatingVictims, nonViolatingVictims := filterVictimsWithPDBViolation(potentialVictims, pdbs)
	numViolatingVictim := 0
	// Sort potentialVictims by descending importance, which ensures reprieve of
	// higher priority victims first.
	reprieveVictim := func(v preemption.PreemptionUnit) (bool, error) {
		if err := addPods(v); err != nil {
			return false, err
		}

		status := pl.SimulatePodScheduling(ctx, preemptor, domain)
		fits := status.IsSuccess()
		if !fits {
			if err := removePods(v); err != nil {
				return false, err
			}
			var names []string
			for _, p := range v.Pods() {
				names = append(names, p.GetPod().Name)
			}
			pods := strings.Join(names, ",")
			logger.V(5).Info("Pods are potential preemption victims on domain", "pods", pods, "domain", domain.GetName())
		}

		return fits, nil
	}

	var victimsToPreempt []preemption.PreemptionUnit
	for _, v := range violatingVictims {
		if fits, err := reprieveVictim(v); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			victimsToPreempt = append(victimsToPreempt, v)
			numViolatingVictim++
		}
	}

	victimsToPreempt = append(victimsToPreempt, nonViolatingVictims...)
	sort.Slice(victimsToPreempt, func(i, j int) bool {
		return pl.MoreImportantVictim(victimsToPreempt[i], victimsToPreempt[j])
	})
	var podsToPreempt []*v1.Pod
	for _, v := range victimsToPreempt {
		if fits, err := reprieveVictim(v); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			for _, pi := range v.Pods() {
				podsToPreempt = append(podsToPreempt, pi.GetPod())
			}
		}
	}

	return podsToPreempt, numViolatingVictim, nil
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
				preemptionUnit := preemption.NewPreemptionUnit([]fwk.PodInfo{p}, corev1helpers.PodPriority(p.GetPod()), []fwk.NodeInfo{nodeInfo})
				domain := preemption.NewDomainForPodByPodPreemption(nodeInfo, nodeInfo.Node().Name)
				if pl.isPreemptionAllowedForDomain(domain, preemptionUnit, preemption.NewPodPreemptor(pod, nil)) && podTerminatingByPreemption(p.GetPod()) {
					// There is a terminating pod on the nominated node.
					return false, "not eligible due to a terminating pod on the nominated node."
				}
			}
		}
	}
	return true, ""
}

func filterVictimsWithPDBViolation(victims []preemption.PreemptionUnit, pdbs []*policy.PodDisruptionBudget) (violatingVictims, nonViolatingVictims []preemption.PreemptionUnit) {
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

// isPreemptionAllowedForDomain returns whether the victim residing on nodeInfo can be preempted by the preemptor
func (pl *DefaultPreemption) isPreemptionAllowedForDomain(domain preemption.Domain, victim preemption.PreemptionUnit, preemptor preemption.Preemptor) bool {
	// The victim must have lower priority than the preemptor, in addition to any filtering implemented by IsEligiblePreemptor
	return victim.Priority() < preemptor.Priority() && pl.IsEligiblePreemptor(domain, victim, preemptor)
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

// MoreImportantVictim decides which of two preemption units is considered more critical.
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
func (pl *DefaultPreemption) moreImportantVictim(vi1, vi2 preemption.PreemptionUnit) bool {
	if vi1.Priority() != vi2.Priority() {
		return vi1.Priority() > vi2.Priority()
	}

	if pl.fts.EnableWorkloadAwarePreemption && vi1.IsPodGroup() != vi2.IsPodGroup() {
		return vi1.IsPodGroup()
	}

	if !vi1.IsPodGroup() {
		return util.GetPodStartTime(vi1.Pods()[0].GetPod()).Before(util.GetPodStartTime(vi2.Pods()[0].GetPod()))
	}

	if len(vi1.Pods()) != len(vi2.Pods()) {
		return len(vi1.Pods()) > len(vi2.Pods())
	}

	t1 := getEarliestPodStartTime(vi1.Pods())
	t2 := getEarliestPodStartTime(vi2.Pods())
	return t1.Before(t2)
}

// getEarliestPodStartTime finds the oldest StartTime among a list of Pods.
func getEarliestPodStartTime(pInfos []fwk.PodInfo) *metav1.Time {
	var earliest *metav1.Time
	for _, pInfo := range pInfos {
		t := util.GetPodStartTime(pInfo.GetPod())
		if earliest == nil || (t != nil && t.Before(earliest)) {
			earliest = t
		}
	}
	return earliest
}
