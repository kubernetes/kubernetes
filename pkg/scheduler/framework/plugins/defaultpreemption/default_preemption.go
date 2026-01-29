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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Name of the plugin used in the plugin registry and configurations.
const Name = names.DefaultPreemption

// IsEligiblePodFunc is a function which may be assigned to the DefaultPreemption plugin.
// This may implement rules/filtering around preemption eligibility, which is in addition to
// the internal requirement that the victim pod have lower priority than the preemptor pod.
// Any customizations should always allow system services to preempt normal pods, to avoid
// problems if system pods are unable to find space.
type IsEligiblePodFunc func(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool

// MoreImportantPodGroupFunc is a function which may be assigned to the DefaultPreemption plugin.
// Implementations should return true if the first victim pod group is more important than the second
// victim group (and thus the second should be considered for preemption before the first).
// For performance reasons, the search for nodes eligible for preemption is done by omitting all
// eligible victims from a node then checking whether the preemptor fits on the node without them,
// before adding back the victims (starting from the most important) that still fit with the preemptor.
// The default behavior is to not consider pod affinity between the preemptor and the victims,
// as affinity between pods that are eligible to preempt each other isn't recommended.
type MoreImportantPodGroupFunc func(g1, g2 *util.VictimGroup, workloadAwarePreemptionEnabled bool) bool

// DefaultPreemption is a PostFilter plugin implements the preemption logic.
type DefaultPreemption struct {
	fh        fwk.Handle
	fts       feature.Features
	args      config.DefaultPreemptionArgs
	Evaluator *preemption.Evaluator

	wm fwk.WorkloadManager
	wl schedulinglisters.WorkloadLister

	// IsEligiblePod returns whether a victim pod is allowed to be preempted by a preemptor pod.
	// This filtering is in addition to the internal requirement that the victim pod have lower
	// priority than the preemptor pod. Any customizations should always allow system services
	// to preempt normal pods, to avoid problems if system pods are unable to find space.
	IsEligiblePod IsEligiblePodFunc

	// MoreImportantPodGroup is used to sort eligible victim groups in-place in descending order of
	// highest to lowest importance. Victim groups with higher importance are less likely to be
	// preempted. The default behavior depends on whether or not workload-aware preemption is enabled:
	// - if workload-aware preemption is disabled, victim groups are simply wrappers for the victim pods
	//   and are sorted by descending priority, then descending runtime duration for pods with equal priority.
	// - if workload-aware preemption is enabled, victim groups are sorted by descending priority, then
	//   descending "importance" as calculated by the workload-aware preemption evaluator.
	MoreImportantPodGroup MoreImportantPodGroupFunc
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
	pl.Evaluator = preemption.NewEvaluator(Name, fh, &pl, fts.EnableAsyncPreemption)

	if fts.EnableWorkloadAwarePreemption {
		pl.wm = fh.WorkloadManager()
		pl.wl = fh.SharedInformerFactory().Scheduling().V1alpha1().Workloads().Lister()
	}

	// Default behavior: No additional filtering, beyond the internal requirement that the victim pod
	// have lower priority than the preemptor pod.
	pl.IsEligiblePod = func(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool {
		return true
	}

	// Default behavior: Sort by descending priority, then by descending runtime duration as secondary ordering.
	pl.MoreImportantPodGroup = util.MoreImportantPodGroup

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

// SelectVictimsOnNode finds minimum set of pods on the given node that should be preempted in order to make enough room
// for "pod" to be scheduled.
func (pl *DefaultPreemption) SelectVictimsOnNode(
	ctx context.Context,
	state fwk.CycleState,
	pod *v1.Pod,
	nodeInfo fwk.NodeInfo,
	pdbs []*policy.PodDisruptionBudget) ([]*v1.Pod, int, *fwk.Status) {
	logger := klog.FromContext(ctx)
	potentialVictimsOnNode := make(map[types.UID]fwk.PodInfo)
	removePod := func(rpi fwk.PodInfo) error {
		if err := nodeInfo.RemovePod(logger, rpi.GetPod()); err != nil {
			return err
		}
		status := pl.fh.RunPreFilterExtensionRemovePod(ctx, state, pod, rpi, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	removePodGroup := func(vg *util.VictimGroup) error {
		for _, p := range vg.Pods {
			pi, ok := potentialVictimsOnNode[p.UID]
			if !ok {
				continue
			}
			if err := removePod(pi); err != nil {
				return err
			}
		}
		return nil
	}
	addPodGroup := func(vg *util.VictimGroup) error {
		for _, p := range vg.Pods {
			pi, ok := potentialVictimsOnNode[p.UID]
			if !ok {
				continue
			}
			nodeInfo.AddPodInfo(pi)
			status := pl.fh.RunPreFilterExtensionAddPod(ctx, state, pod, pi, nodeInfo)
			if !status.IsSuccess() {
				return status.AsError()
			}
		}
		return nil
	}
	// As the first step, remove all pods eligible for preemption from the node and
	// check if the given pod can be scheduled without them present.
	for _, pi := range nodeInfo.GetPods() {
		if pl.isPreemptionAllowed(nodeInfo, pi, pod) {
			potentialVictimsOnNode[pi.GetPod().UID] = pi
		}
	}
	for _, pi := range potentialVictimsOnNode {
		if err := removePod(pi); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// No potential victims are found, and so we don't need to evaluate the node again since its state didn't change.
	if len(potentialVictimsOnNode) == 0 {
		return nil, 0, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "No preemption victims found for incoming pod")
	}

	// If the new pod does not fit after removing all the eligible pods,
	// we are almost done and this node is not suitable for preemption. The only
	// condition that we could check is if the "pod" is failing to schedule due to
	// inter-pod affinity to one or more victims, but we have decided not to
	// support this case for performance reasons. Having affinity to lower
	// importance (priority) pods is not a recommended configuration anyway.
	if status := pl.fh.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo); !status.IsSuccess() {
		return nil, 0, status
	}
	potentialVictimPodGroups := pl.preparePotentialVictimPodGroups(potentialVictimsOnNode, logger)
	// Sort potentialVictimGroups by descending importance, which ensures reprieve of
	// higher importance pod groups first.
	sort.Slice(potentialVictimPodGroups, func(i, j int) bool {
		return pl.MoreImportantPodGroup(potentialVictimPodGroups[i], potentialVictimPodGroups[j], pl.fts.EnableWorkloadAwarePreemption)
	})

	var victimPodGroups []*util.VictimGroup
	numViolatingVictim := 0
	// Try to reprieve as many victim pod groups as possible. We first try to reprieve
	// the PDB violating victim groups and then other non-violating ones. In both cases,
	// we start from the highest importance victim groups.
	violatingVictimGroups, nonViolatingVictimGroups := filterPodsGroupsWithPDBViolation(potentialVictimPodGroups, pdbs)
	reprievePodGroup := func(vg *util.VictimGroup) (bool, error) {
		if err := addPodGroup(vg); err != nil {
			return false, err
		}
		status := pl.fh.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo)
		fits := status.IsSuccess()
		if !fits {
			if err := removePodGroup(vg); err != nil {
				return false, err
			}
			victimPodGroups = append(victimPodGroups, vg)
			if !pl.fts.EnableWorkloadAwarePreemption || !vg.IsGang {
				logger.V(5).Info("Pod is a potential preemption victim on node", "pod", klog.KObj(vg.Pods[0]), "node", klog.KObj(nodeInfo.Node()))
			} else {
				podGroupKey := helper.GetPodGroupKey(vg.Pods[0])
				logger.V(5).Info("Pod group is a potential preemption victim on node", "pod group", *podGroupKey, "node", klog.KObj(nodeInfo.Node()))
			}
		}
		return fits, nil
	}
	for _, vg := range violatingVictimGroups {
		if fits, err := reprievePodGroup(vg); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			numViolatingVictim++
		}
	}

	// Now we try to reprieve non-violating victims.
	for _, vg := range nonViolatingVictimGroups {
		if _, err := reprievePodGroup(vg); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// Sort victims after reprieving pods to keep the pods in the victims sorted in order of importance from high to low.
	if len(violatingVictimGroups) != 0 && len(nonViolatingVictimGroups) != 0 {
		sort.Slice(victimPodGroups, func(i, j int) bool {
			return pl.MoreImportantPodGroup(victimPodGroups[i], victimPodGroups[j], pl.fts.EnableWorkloadAwarePreemption)
		})
	}
	var victims []*v1.Pod
	for _, vg := range victimPodGroups {
		victims = append(victims, vg.Pods...)
	}
	return victims, numViolatingVictim, fwk.NewStatus(fwk.Success)
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
				if pl.isPreemptionAllowed(nodeInfo, p, pod) && podTerminatingByPreemption(p.GetPod()) {
					// There is a terminating pod on the nominated node.
					return false, "not eligible due to a terminating pod on the nominated node."
				}
			}
		}
	}
	return true, ""
}

// OrderedScoreFuncs returns a list of ordered score functions to select preferable node where victims will be preempted.
func (pl *DefaultPreemption) OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64 {
	return nil
}

// isPreemptionAllowed returns whether the victim residing on nodeInfo can be preempted by the preemptor
func (pl *DefaultPreemption) isPreemptionAllowed(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool {
	// The victim must have lower priority than the preemptor, in addition to any filtering implemented by IsEligiblePod
	return corev1helpers.PodPriority(victim.GetPod()) < corev1helpers.PodPriority(preemptor) && pl.IsEligiblePod(nodeInfo, victim, preemptor)
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

// filterPodsGroupsWithPDBViolation groups the given "victim groups" into two
// separate sets of "violatingPodGroups" and "nonViolatingPodGroups" based on
// whether their PDBs will be violated if they are preempted.
// This function is stable and does not change the order of received groups.
// So, if it receives a sorted list, grouping will preserve the order of the
// input list.
func filterPodsGroupsWithPDBViolation(victimGroups []*util.VictimGroup, pdbs []*policy.PodDisruptionBudget) (violatingPodGroups, nonViolatingPodGroups []*util.VictimGroup) {
	pdbsAllowed := make([]int32, len(pdbs))
	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	for _, vg := range victimGroups {
		pdbForPodGroupIsViolated := false
		for _, pod := range vg.Pods {
			// A pod with no labels will not match any PDB. So, no need to check.
			if len(pod.Labels) == 0 {
				continue
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
					pdbForPodGroupIsViolated = true
				}
			}
		}
		if pdbForPodGroupIsViolated {
			violatingPodGroups = append(violatingPodGroups, vg)
		} else {
			nonViolatingPodGroups = append(nonViolatingPodGroups, vg)
		}
	}
	return violatingPodGroups, nonViolatingPodGroups
}

func (pl *DefaultPreemption) preparePotentialVictimPodGroups(potentialVictimsOnNode map[types.UID]fwk.PodInfo, logger klog.Logger) []*util.VictimGroup {
	var potentialVictimPodGroups []*util.VictimGroup
	if !pl.fts.EnableWorkloadAwarePreemption {
		for _, pi := range potentialVictimsOnNode {
			vg := util.WrapPodInVictimGroup(pi.GetPod())
			potentialVictimPodGroups = append(potentialVictimPodGroups, vg)
		}
		return potentialVictimPodGroups
	}

	// Iterate over all potential victims on the node and group them accordingly.
	// If the pod is part of a gang pod group with the PodGroup disruption mode,
	// then we take all pods in the gang and group them together, including gang
	// pods that are running on different nodes.
	// Otherwise, the victim group serves as a wrapper for the individual victim pod.
	podGroupKeys := sets.New[helper.PodGroupKey]()
	for _, pi := range potentialVictimsOnNode {
		if !helper.HasDisruptionModePodGroup(pi.GetPod(), pl.wl) {
			vg := util.WrapPodInVictimGroup(pi.GetPod())
			potentialVictimPodGroups = append(potentialVictimPodGroups, vg)
			continue
		}
		key := helper.GetPodGroupKey(pi.GetPod())
		if key == nil || podGroupKeys.Has(*key) {
			continue
		}
		podGroupKeys.Insert(*key)
		pgs, err := pl.wm.PodGroupState(key.GetNamespace(), key.GetWorkloadRef())
		if err != nil {
			// TODO: Should we fail the preemption instead of logging and treating as a non-gang pod?
			logger.Error(err, "Failed to get pod group state for the pod", "pod", klog.KObj(pi.GetPod()), "workloadRef", key.GetWorkloadRef())
			continue
		}
		// TODO: instead of using the priority of the individual pod, we should
		// use the priority of the pod group's Workload.
		// When https://github.com/kubernetes/kubernetes/pull/136426 is merged,
		// we will switch to using the priority of the pod group's Workload.
		vg := &util.VictimGroup{
			Pods:      pgs.ScheduledPods().UnsortedList(),
			Priority:  corev1helpers.PodPriority(pi.GetPod()),
			StartTime: util.GetPodGroupInitTimestamp(pgs),
			IsGang:    true,
		}
		// TODO: this assumes that the pods in the gang have the same priority which is not necessarily true.
		// Do we have any better way of sorting the victim pods of the gang itself?
		sort.Slice(vg.Pods, func(i, j int) bool {
			return vg.Pods[i].Status.StartTime.Before(vg.Pods[j].Status.StartTime)
		})
		potentialVictimPodGroups = append(potentialVictimPodGroups, vg)
	}
	return potentialVictimPodGroups
}
