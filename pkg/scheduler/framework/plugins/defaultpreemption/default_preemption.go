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

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
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
// the internal requirement that the victim has lower priority than the preemptor pod.
// Any customizations should always allow system services to preempt normal pods, to avoid
// problems if system pods are unable to find space.
//
// For PodGroup victims, this function is called once per affected node: nodeInfo reflects
// that specific node, while victim always represents the entire PodGroup across all its nodes.
// The victim is eligible only if the function returns true for every affected node.
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

	Executor          *preemption.Executor
	Evaluator         *preemption.Evaluator
	pgLister          fwk.PodGroupLister
	podGroupEvaluator *preemption.PodGroupEvaluator

	// IsEligiblePod returns whether a victim (individual pod/pod group) is allowed to be preempted by a preemptor pod.
	// This filtering is in addition to the internal requirement that the victim pod have lower
	// priority than the preemptor pod. Any customizations should always allow system services
	// to preempt normal pods, to avoid problems if system pods are unable to find space.
	IsEligiblePod IsEligiblePodFunc

	// MoreImportantVictim is used to sort eligible victims in-place in descending order of highest to
	// lowest importance. Victims with higher importance are less likely to be preempted.
	//
	// By default, potential preemption victims are evaluated using a cascading set of rules. The system prioritizes based on
	// priority first, always deferring to higher-priority victims.
	// If GenericWorkload is disabled, it simply falls back to prioritizing older victims. However,
	// if WAP is enabled, it factors in the victim type by favoring PodGroups over standalone Pods. When comparing multiple PodGroups,
	// it prioritizes those with larger group sizes, ultimately using the start time as the final tiebreaker.
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
	pl.Executor = preemption.NewExecutor(fh, fts)
	pl.Evaluator = preemption.NewEvaluator(Name, fh, &pl, pl.Executor)

	if pl.fts.EnableGenericWorkload {
		pl.pgLister = fh.PodGroupManager().PodGroups()
		pl.podGroupEvaluator = preemption.NewPodGroupEvaluator(fh, pl.Executor, pl.fts.EnablePodGroupPreemptionPolicy)
	}

	// Default behavior: No additional filtering, beyond the internal requirement that the victim
	// has lower priority than the preemptor.
	pl.IsEligiblePod = func(nodeInfo fwk.NodeInfo, victim preemption.Victim, preemptor *v1.Pod) bool {
		return true
	}

	// Default behavior, defines the order for victims sorting. The order is:
	// 1. Higher Priority.
	// 2. If GenericWorkload is enabled: PodGroup > Individual Pod.
	// 3. For individual Pods: Older StartTime (longer runtime).
	// 4. For PodGroups: Larger Group Size, then Older StartTime.
	pl.MoreImportantVictim = func(vi1, vi2 preemption.Victim) bool {
		return preemption.MoreImportantVictim(vi1, vi2, pl.fts.EnableGenericWorkload)
	}

	return &pl, nil
}

// PostFilter invoked at the postFilter extension point.
func (pl *DefaultPreemption) PostFilter(ctx context.Context, state fwk.CycleState, pod *v1.Pod, m fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	defer func() {
		metrics.PreemptionAttempts.Inc()
	}()

	if pod.Spec.SchedulingGroup != nil && pl.fts.EnableGenericWorkload {
		// When GenericWorkload is enabled, the default preemption logic needs to be disabled for pod group scheduling to avoid performing preemption in pod by pod cycle
		// of pod group scheduling. Instead the WAP will be called to perform preemption for the entire pod group.
		return nil, fwk.NewStatus(fwk.Unschedulable, "preemption: not eligible due to workload aware preemption enabled")
	}

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
	if p.Spec.SchedulingGroup != nil && pl.fts.EnableGenericWorkload {
		pg, err := pl.pgLister.Get(p.Namespace, *p.Spec.SchedulingGroup.PodGroupName)
		// If the pg is not found do not block the pod. It's not a default preemption responsibility
		// to block pods from pod group without pg from entering the queue.
		if err != nil {
			return nil
		}
		if pl.Executor.IsPodGroupRunningPreemption(pg.GetUID()) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for the preemption for this pod group to be finished")
		}
		return nil
	}

	if pl.Executor.IsPodRunningPreemption(p.GetUID()) {
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
			{Event: fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Delete}, QueueingHintFn: pl.isPodSchedulableAfterAssignedPodDeletion},
		}, nil
	}

	// When the async preemption is disabled, PreEnqueue always returns nil, and hence pods never get rejected by this plugin.
	return nil, nil
}

// isPodSchedulableAfterAssignedPodDeletion returns the queueing hint for the pod after the assigned pod deletion event,
// which always return Skip.
// The default preemption plugin is a bit tricky;
// the pods rejected by it are the ones that have run/are running the preemption asynchronously.
// And, those pods should always have the other plugins in pInfo.UnschedulablePlugins
// which failure will be resolved by the preemption.
// The reason why we return Skip here is that the preemption plugin should not make the decision of when to requeueing Pods,
// and rather, those plugins should be responsible for that.
func (pl *DefaultPreemption) isPodSchedulableAfterAssignedPodDeletion(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
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

// SelectVictimsOnNode finds the minimum set of pods that should be preempted to make enough room for the preemptor to be
// scheduled on the node represented by nodeInfo (the "main node").
//
// On entry, nodeInfo represents only the main node. If the candidate victims include a PodGroup that spans additional nodes,
// SelectVictimsOnNode discovers those nodes from victim.AffectedNodes() during the removal pass and extends its internal nameToNode map
// accordingly, so PreFilterExtension hooks can run against them. See also the block comment inside the function body for the
// NodeInfo-mutation contract across main vs. remote nodes.
func (pl *DefaultPreemption) SelectVictimsOnNode(
	ctx context.Context,
	cycleState fwk.CycleState,
	preemptor *v1.Pod,
	nodeInfo fwk.NodeInfo,
	allPossibleVictims []*preemption.DomainVictim,
	pdbs []*policy.PodDisruptionBudget) ([]*v1.Pod, int, *fwk.Status) {
	logger := klog.FromContext(ctx)
	// Within a single SelectVictimsOnNode evaluation we mutate NodeInfo only for the main node (the node we are trying to fit the preemptor onto).
	// Filter plugins are re-run against mainNode below via RunFilterPluginsWithNominatedPods, so they must see the post-removal state
	// of mainNode reflected in NodeInfo.
	//
	// For remote nodes that are only affected because a PodGroup victim has members there, we deliberately do NOT mutate their NodeInfo.
	// Instead we invoke PreFilterExtension Remove/AddPod with the remote NodeInfo so that plugins can update any CycleState they keep
	// about those pods (e.g. topology spread skew, pod affinity counters). Plugins that read remote NodeInfo directly during Filter
	// (rather than CycleState) will observe stale state for those nodes; this is acceptable because we only run the Filter pass
	// against mainNode here.
	mainNodeName := nodeInfo.Node().Name
	nameToNode := map[string]fwk.NodeInfo{mainNodeName: nodeInfo}

	removeVictim := func(dv *preemption.DomainVictim) error {
		for _, pi := range dv.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
			// See the block comment in SelectVictimsOnNode: only mainNode's NodeInfo
			// is mutated; remote nodes go through PreFilterExtension only.
			if pi.GetPod().Spec.NodeName == mainNodeName {
				if err := nodeInfo.RemovePod(logger, pi.GetPod()); err != nil {
					return err
				}
			}
			status := pl.fh.RunPreFilterExtensionRemovePod(ctx, cycleState, preemptor, pi, nodeInfo)
			if !status.IsSuccess() {
				return status.AsError()
			}
		}

		return nil
	}
	addVictim := func(pu *preemption.DomainVictim) error {
		for _, pi := range pu.Pods() {
			nodeInfo := nameToNode[pi.GetPod().Spec.NodeName]
			// See the block comment in SelectVictimsOnNode: only mainNode's NodeInfo
			// is mutated; remote nodes go through PreFilterExtension only.
			if pi.GetPod().Spec.NodeName == mainNodeName {
				nodeInfo.AddPodInfo(pi)
			}
			status := pl.fh.RunPreFilterExtensionAddPod(ctx, cycleState, preemptor, pi, nodeInfo)
			if !status.IsSuccess() {
				return status.AsError()
			}
		}

		return nil
	}

	var potentialVictims []*preemption.DomainVictim
	for _, victim := range allPossibleVictims {
		if pl.isPreemptionAllowedAcrossAllVictimNodes(victim, preemptor) {
			potentialVictims = append(potentialVictims, victim)
		}
	}

	// No preemption victims found for incoming preemptor, and so we don't need to evaluate the node again since its state didn't change.
	if len(potentialVictims) == 0 {
		return nil, 0, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming pod")
	}

	// As the first step, remove all victims eligible for preemption from the node.
	for _, victim := range potentialVictims {
		// If a Victim is a PodGroup spanning multiple nodes, some affected nodes
		// might not be in the nameToNode map yet. We add them to ensure
		// they are considered during preemption.
		for name, nodeInfo := range victim.AffectedNodes() {
			if _, ok := nameToNode[name]; !ok {
				nameToNode[name] = nodeInfo
			}
		}
		if err := removeVictim(victim); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// If the preemptor does not fit after removing all the eligible victims,
	// we are almost done and this node is not suitable for preemption. The only
	// condition that we could check is if the preemptor is failing to schedule due to
	// inter-pod affinity to one or more victims, but we have decided not to
	// support this case for performance reasons. Having affinity to lower
	// importance (priority) pods is not a recommended configuration anyway.
	if status := pl.fh.RunFilterPluginsWithNominatedPods(ctx, cycleState, preemptor, nodeInfo); !status.IsSuccess() {
		return nil, 0, status
	}

	// Sort potentialVictims by descending importance, which ensures reprieve of
	// higher importance victims first.
	sort.Slice(potentialVictims, func(i, j int) bool {
		return pl.MoreImportantVictim(potentialVictims[i], potentialVictims[j])
	})

	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest importance victims.
	violatingVictims, nonViolatingVictims := preemption.FilterVictimsWithPDBViolation(potentialVictims, pdbs)
	var victims []*preemption.DomainVictim
	reprieveVictim := func(v *preemption.DomainVictim) (bool, error) {
		if err := addVictim(v); err != nil {
			return false, err
		}

		status := pl.fh.RunFilterPluginsWithNominatedPods(ctx, cycleState, preemptor, nodeInfo)
		fits := status.IsSuccess()
		if !fits {
			if err := removeVictim(v); err != nil {
				return false, err
			}
			victims = append(victims, v)
			if loggerV := logger.V(5); loggerV.Enabled() {
				var pods []klog.ObjectRef
				for _, p := range v.Pods() {
					pods = append(pods, klog.KObj(p.GetPod()))
				}
				loggerV.Info("Pods are potential preemption victims on node", "pods", pods, "node", mainNodeName)
			}
		}

		return fits, nil
	}

	numViolatingVictim := 0
	for _, violatingVictim := range violatingVictims {
		if fits, err := reprieveVictim(violatingVictim.Victim); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			numViolatingVictim += violatingVictim.ViolateCount
		}
	}

	// Now we try to reprieve non-violating victims.
	for _, v := range nonViolatingVictims {
		if _, err := reprieveVictim(v); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// Sort victims after reprieving pods to keep the victims sorted in order of importance from high to low.
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
	if pod.Spec.PreemptionPolicy != nil && *pod.Spec.PreemptionPolicy == v1.PreemptNever {
		return false, "not eligible due to preemptionPolicy=Never."
	}

	nodeInfos := pl.fh.MutableSnapshotSharedLister().NodeInfos()
	nomNodeName := pod.Status.NominatedNodeName
	if len(nomNodeName) > 0 {
		// If the pod's nominated node is considered as UnschedulableAndUnresolvable by the filters,
		// then the pod should be considered for preempting again.
		if nominatedNodeStatus.Code() == fwk.UnschedulableAndUnresolvable {
			return true, ""
		}

		if nodeInfo, _ := nodeInfos.Get(nomNodeName); nodeInfo != nil {
			for _, p := range nodeInfo.GetPods() {
				victim := preemption.NewPodVictim(p, pl.pgLister)

				if pl.isPreemptionAllowed(nodeInfo, victim, pod) && preemption.PodTerminatingByPreemption(p.GetPod()) {
					// There is a terminating pod on the nominated node.
					return false, "not eligible due to a terminating pod on the nominated node."
				}
			}
		}
	}
	return true, ""
}

// isPreemptionAllowed returns whether the "victim" residing on "nodeInfo" can be preempted by the preemptor pod
func (pl *DefaultPreemption) isPreemptionAllowed(nodeInfo fwk.NodeInfo, victim preemption.Victim, preemptor *v1.Pod) bool {
	// The victim must have lower priority than the pod, in addition to any filtering implemented by IsEligiblePod
	return victim.Priority() < corev1helpers.PodPriority(preemptor) && pl.IsEligiblePod(nodeInfo, victim, preemptor)
}

// OrderedScoreFuncs returns a list of ordered score functions to select preferable node where victims will be preempted.
func (pl *DefaultPreemption) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return nil
}

// isPreemptionAllowedAcrossAllVictimNodes returns whether the victim can be preempted on all its affected nodes.
func (pl *DefaultPreemption) isPreemptionAllowedAcrossAllVictimNodes(victim *preemption.DomainVictim, preemptor *v1.Pod) bool {
	if victim.Priority() >= corev1helpers.PodPriority(preemptor) {
		return false
	}

	for _, nodeInfo := range victim.AffectedNodes() {
		if !pl.IsEligiblePod(nodeInfo, victim, preemptor) {
			return false
		}
	}

	return true
}

// PodGroupPostFilter runs a default preemption for the pod group.
func (pl *DefaultPreemption) PodGroupPostFilter(ctx context.Context, state fwk.PodGroupCycleState, pgInfo fwk.PodGroupInfo, pgSchedulingFunc fwk.PodGroupSchedulingFunc) (postFilterResult *fwk.PodGroupPostFilterResult, status *fwk.Status) {
	pg := pgInfo.GetPodGroup()

	if pg.Spec.SchedulingConstraints != nil && len(pg.Spec.SchedulingConstraints.Topology) > 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "pod group preemption: not supported with topology constraints")
	}

	mutableLister := pl.fh.MutableSnapshotSharedLister()
	err := mutableLister.StartMutations()
	if err != nil {
		return nil, fwk.AsStatus(fmt.Errorf("pod group preemption: failed to start mutations: %w", err))
	}
	defer func() {
		if err := mutableLister.EndMutations(); err != nil {
			status = fwk.AsStatus(fmt.Errorf("pod group preemption: failed to end mutations: %w", err))
		}
	}()

	res, status := pl.podGroupEvaluator.Preempt(ctx, pg, pgInfo.GetUnscheduledPods(), pgSchedulingFunc)
	msg := status.Message()
	if len(msg) > 0 {
		return res, fwk.NewStatus(status.Code(), "pod group preemption: "+msg)
	}
	return res, status
}
