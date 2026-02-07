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

// IsEligiblePodFunc is a function which may be assigned to the DefaultPreemption plugin.
// This may implement rules/filtering around preemption eligibility, which is in addition to
// the internal requirement that the victim pod have lower priority than the preemptor pod.
// Any customizations should always allow system services to preempt normal pods, to avoid
// problems if system pods are unable to find space.
type IsEligiblePodFunc func(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool

// MoreImportantPodFunc is a function which may be assigned to the DefaultPreemption plugin.
// Implementations should return true if the first pod is more important than the second pod
// and the second one should be considered for preemption before the first one.
// For performance reasons, the search for nodes eligible for preemption is done by omitting all
// eligible victims from a node then checking whether the preemptor fits on the node without them,
// before adding back victims (starting from the most important) that still fit with the preemptor.
// The default behavior is to not consider pod affinity between the preemptor and the victims,
// as affinity between pods that are eligible to preempt each other isn't recommended.
type MoreImportantPodFunc func(pod1, pod2 *v1.Pod) bool

// DefaultPreemption is a PostFilter plugin implements the preemption logic.
type DefaultPreemption struct {
	fh        fwk.Handle
	fts       feature.Features
	args      config.DefaultPreemptionArgs
	Evaluator *preemption.Evaluator

	// IsEligiblePod returns whether a victim pod is allowed to be preempted by a preemptor pod.
	// This filtering is in addition to the internal requirement that the victim pod have lower
	// priority than the preemptor pod. Any customizations should always allow system services
	// to preempt normal pods, to avoid problems if system pods are unable to find space.
	IsEligiblePod IsEligiblePodFunc

	// MoreImportantPod is used to sort eligible victims in-place in descending order of highest to
	// lowest importance. Pods with higher importance are less likely to be preempted.
	// The default behavior is to order pods by descending priority, then descending runtime duration
	// for pods with equal priority.
	MoreImportantPod MoreImportantPodFunc
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

	// Default behavior: No additional filtering, beyond the internal requirement that the victim pod
	// have lower priority than the preemptor pod.
	pl.IsEligiblePod = func(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool {
		return true
	}

	// Default behavior: Sort by descending priority, then by descending runtime duration as secondary ordering.
	pl.MoreImportantPod = util.MoreImportantPod

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
	var potentialVictims []fwk.PodInfo
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
	addPod := func(api fwk.PodInfo) error {
		nodeInfo.AddPodInfo(api)
		status := pl.fh.RunPreFilterExtensionAddPod(ctx, state, pod, api, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	// As the first step, remove all pods eligible for preemption from the node and
	// check if the given pod can be scheduled without them present.
	for _, pi := range nodeInfo.GetPods() {
		if pl.isPreemptionAllowed(nodeInfo, pi, pod) {
			potentialVictims = append(potentialVictims, pi)
		}
	}
	for _, pi := range potentialVictims {
		if err := removePod(pi); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// No potential victims are found, and so we don't need to evaluate the node again since its state didn't change.
	if len(potentialVictims) == 0 {
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
	var victims []fwk.PodInfo
	numViolatingVictim := 0
	// Sort potentialVictims by descending importance, which ensures reprieve of
	// higher importance pods first.
	sort.Slice(potentialVictims, func(i, j int) bool {
		return pl.MoreImportantPod(potentialVictims[i].GetPod(), potentialVictims[j].GetPod())
	})
	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest importance victims.
	violatingVictims, nonViolatingVictims := filterPodsWithPDBViolation(potentialVictims, pdbs)
	reprievePod := func(pi fwk.PodInfo) (bool, error) {
		if err := addPod(pi); err != nil {
			return false, err
		}
		status := pl.fh.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo)
		fits := status.IsSuccess()
		if !fits {
			if err := removePod(pi); err != nil {
				return false, err
			}
			victims = append(victims, pi)
			logger.V(5).Info("Pod is a potential preemption victim on node", "pod", klog.KObj(pi.GetPod()), "node", klog.KObj(nodeInfo.Node()))
		}
		return fits, nil
	}
	for _, p := range violatingVictims {
		if fits, err := reprievePod(p); err != nil {
			return nil, 0, fwk.AsStatus(err)
		} else if !fits {
			numViolatingVictim++
		}
	}
	// Now we try to reprieve non-violating victims.
	for _, p := range nonViolatingVictims {
		if _, err := reprievePod(p); err != nil {
			return nil, 0, fwk.AsStatus(err)
		}
	}

	// Sort victims after reprieving pods to keep the pods in the victims sorted in order of importance from high to low.
	if len(violatingVictims) != 0 && len(nonViolatingVictims) != 0 {
		sort.Slice(victims, func(i, j int) bool { return pl.MoreImportantPod(victims[i].GetPod(), victims[j].GetPod()) })
	}
	var victimPods []*v1.Pod
	for _, pi := range victims {
		victimPods = append(victimPods, pi.GetPod())
	}
	return victimPods, numViolatingVictim, fwk.NewStatus(fwk.Success)
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
// A victim can be preempted if it has strictly lower priority than the preemptor.
// Example: victim priority=0, preemptor priority=10 → preemption allowed
//
// ROLLING UPDATE PREEMPTION (Equal Priority Scenario):
// When priorities are equal, preemption is allowed ONLY if ALL of these conditions are met:
//  1. Same Namespace: Both pods must be in the same namespace
//  2. Deployment-Managed: Both pods must be managed by Deployments (via ReplicaSet owners)
//  3. Rolling Update: Pods must belong to different ReplicaSets of the same Deployment
func (pl *DefaultPreemption) isPreemptionAllowed(nodeInfo fwk.NodeInfo, victim fwk.PodInfo, preemptor *v1.Pod) bool {
	victimPod := victim.GetPod()
	victimPriority := corev1helpers.PodPriority(victimPod)
	preemptorPriority := corev1helpers.PodPriority(preemptor)

	// Standard: victim has lower priority
	if victimPriority < preemptorPriority {
		return pl.IsEligiblePod(nodeInfo, victim, preemptor)
	}

	// Equal priority: check rolling update scenario
	if victimPriority == preemptorPriority {
		// SAFEGUARD 1: Must be same namespace
		if victimPod.Namespace != preemptor.Namespace {
			return false
		}

		// SAFEGUARD 2: Must be Deployment-managed (not StatefulSet)
		if !isDeploymentManaged(victimPod) || !isDeploymentManaged(preemptor) {
			return false
		}

		// SAFEGUARD 3: Must be different ReplicaSets of same Deployment
		if isRollingUpdate(victimPod, preemptor) {
			return pl.IsEligiblePod(nodeInfo, victim, preemptor)
		}
	}

	return false
}

// isDeploymentManaged checks if a pod is managed by a Deployment (via ReplicaSet)
func isDeploymentManaged(pod *v1.Pod) bool {
	// Check if pod has ReplicaSet owner
	for _, owner := range pod.OwnerReferences {
		if owner.Kind == "ReplicaSet" {
			return true
		}
	}
	return false
}

// isRollingUpdate checks if victim and preemptor are from different ReplicaSets
// of the same Deployment
func isRollingUpdate(victim, preemptor *v1.Pod) bool {
	// Get ReplicaSet names from owner references
	victimRS := getReplicaSetName(victim)
	preemptorRS := getReplicaSetName(preemptor)

	// Both must have ReplicaSet owners
	if victimRS == "" || preemptorRS == "" {
		return false
	}

	// Must be different ReplicaSets
	if victimRS == preemptorRS {
		return false
	}

	// Must belong to the same Deployment
	return belongToSameDeployment(victim, preemptor)
}

// getReplicaSetName extracts the ReplicaSet name from pod's owner references
func getReplicaSetName(pod *v1.Pod) string {
	for _, owner := range pod.OwnerReferences {
		if owner.Kind == "ReplicaSet" {
			return owner.Name
		}
	}
	return ""
}

// belongToSameDeployment checks if two pods belong to the same Deployment
func belongToSameDeployment(pod1, pod2 *v1.Pod) bool {
	// ReplicaSets from the same Deployment share all labels except pod-template-hash
	// We'll use common selector labels to determine if they're from the same Deployment

	labels1 := pod1.Labels
	labels2 := pod2.Labels

	if labels1 == nil || labels2 == nil {
		return false
	}

	// Common Deployment selector labels to check
	commonLabels := []string{
		"app",
		"app.kubernetes.io/name",
		"app.kubernetes.io/instance",
		"app.kubernetes.io/component",
	}

	// Count matching labels
	matchCount := 0
	for _, label := range commonLabels {
		val1, exists1 := labels1[label]
		val2, exists2 := labels2[label]

		if exists1 && exists2 && val1 == val2 {
			matchCount++
		}
	}

	// If at least one common label matches, consider them from same Deployment
	if matchCount > 0 {
		return true
	}

	// Alternative: Check if all labels match except pod-template-hash
	// This is more strict but more accurate
	return labelsMatchExceptHash(labels1, labels2)
}

// labelsMatchExceptHash checks if all labels match except pod-template-hash
func labelsMatchExceptHash(labels1, labels2 map[string]string) bool {
	// Create copies without pod-template-hash
	filtered1 := make(map[string]string)
	filtered2 := make(map[string]string)

	for k, v := range labels1 {
		if k != "pod-template-hash" {
			filtered1[k] = v
		}
	}

	for k, v := range labels2 {
		if k != "pod-template-hash" {
			filtered2[k] = v
		}
	}

	// Must have same number of labels (excluding pod-template-hash)
	if len(filtered1) != len(filtered2) {
		return false
	}

	// All labels must match
	for k, v1 := range filtered1 {
		v2, exists := filtered2[k]
		if !exists || v1 != v2 {
			return false
		}
	}

	return len(filtered1) > 0 // At least one label must exist
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

// filterPodsWithPDBViolation groups the given "pods" into two groups of "violatingPods"
// and "nonViolatingPods" based on whether their PDBs will be violated if they are
// preempted.
// This function is stable and does not change the order of received pods. So, if it
// receives a sorted list, grouping will preserve the order of the input list.
func filterPodsWithPDBViolation(podInfos []fwk.PodInfo, pdbs []*policy.PodDisruptionBudget) (violatingPodInfos, nonViolatingPodInfos []fwk.PodInfo) {
	pdbsAllowed := make([]int32, len(pdbs))
	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	for _, podInfo := range podInfos {
		pod := podInfo.GetPod()
		pdbForPodIsViolated := false
		// A pod with no labels will not match any PDB. So, no need to check.
		if len(pod.Labels) != 0 {
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
					pdbForPodIsViolated = true
				}
			}
		}
		if pdbForPodIsViolated {
			violatingPodInfos = append(violatingPodInfos, podInfo)
		} else {
			nonViolatingPodInfos = append(nonViolatingPodInfos, podInfo)
		}
	}
	return violatingPodInfos, nonViolatingPodInfos
}
