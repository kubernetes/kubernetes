/*
Copyright 2021 The Kubernetes Authors.

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
	"math"
	"sync"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	"k8s.io/klog/v2"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

// Interface is expected to be implemented by different preemption plugins as all those member
// methods might have different behavior compared with the default preemption.
type Interface interface {
	// GetOffsetAndNumCandidates chooses a random offset and calculates the number of candidates that should be
	// shortlisted for dry running preemption.
	GetOffsetAndNumCandidates(nodes int32) (int32, int32)
	// CandidatesToVictimsMap builds a map from the target node to a list of to-be-preempted Pods and the number of PDB violation.
	CandidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims
	// PodEligibleToPreemptOthers returns one bool and one string. The bool indicates whether this pod should be considered for
	// preempting other pods or not. The string includes the reason if this pod isn't eligible.
	PodEligibleToPreemptOthers(ctx context.Context, pod *v1.Pod, nominatedNodeStatus *fwk.Status) (bool, string)
	// SelectVictimsOnDomain finds minimum set of pods on the given domain that should be preempted in order to make enough room
	// for "pods" from preemptor to be scheduled.
	// Note that both `state` and `nodeInfo` are deep copied.
	SelectVictimsOnDomain(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domain Domain, pdbs []*policy.PodDisruptionBudget) ([]*v1.Pod, int, *fwk.Status)
	// OrderedScoreFuncs returns a list of ordered score functions to select preferable node where victims will be preempted.
	// The ordered score functions will be processed one by one if we find more than one node with the highest score.
	// Default score functions will be processed if nil returned here for backwards-compatibility.
	OrderedScoreFuncs(ctx context.Context, nodesToVictims map[string]*extenderv1.Victims) []func(node string) int64
}

type Evaluator struct {
	PluginName string
	Handler    fwk.Handle
	PodLister  corelisters.PodLister
	PdbLister  policylisters.PodDisruptionBudgetLister

	enableAsyncPreemption bool

	*Executor
	Interface
}

func NewEvaluator(pluginName string, fh fwk.Handle, i Interface, enableAsyncPreemption bool) *Evaluator {
	return &Evaluator{
		PluginName:            pluginName,
		Handler:               fh,
		PodLister:             fh.SharedInformerFactory().Core().V1().Pods().Lister(),
		PdbLister:             fh.SharedInformerFactory().Policy().V1().PodDisruptionBudgets().Lister(),
		enableAsyncPreemption: enableAsyncPreemption,
		Executor:              newExecutor(fh),
		Interface:             i,
	}
}

// Preempt returns a PostFilterResult carrying suggested nominatedNodeName, along with a Status.
// The semantics of returned <PostFilterResult, Status> varies on different scenarios:
//
//   - <nil, Error>. This denotes it's a transient/rare error that may be self-healed in future cycles.
//
//   - <nil, Unschedulable>. This status is mostly as expected like the preemptor is waiting for the
//     victims to be fully terminated.
//
//   - In both cases above, a nil PostFilterResult is returned to keep the pod's nominatedNodeName unchanged.
//
//   - <non-nil PostFilterResult, Unschedulable>. It indicates the pod cannot be scheduled even with preemption.
//     In this case, a non-nil PostFilterResult is returned and result.NominatingMode instructs how to deal with
//     the nominatedNodeName.
//
//   - <non-nil PostFilterResult, Success>. It's the regular happy path
//     and the non-empty nominatedNodeName will be applied to the preemptor pod.
func (ev *Evaluator) Preempt(ctx context.Context, state fwk.CycleState, preemptor Preemptor, m fwk.NodeToStatusReader) (*fwk.PostFilterResult, *fwk.Status) {
	logger := klog.FromContext(ctx)

	if !preemptor.IsEligibleToPreemptOthers() {
		return nil, fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Preemptor couldn't preempt other victims because of preemptionPolicy: never")
	}

	for _, pod := range preemptor.Members() {
		// 0) Fetch the latest version of <pod>.
		// It's safe to directly fetch pod here. Because the informer cache has already been
		// initialized when creating the Scheduler obj.
		// However, tests may need to manually initialize the shared pod informer.
		podNamespace, podName := pod.Namespace, pod.Name
		pod, err := ev.PodLister.Pods(pod.Namespace).Get(pod.Name)
		if err != nil {
			logger.Error(err, "Could not get the updated preemptor pod object", "pod", klog.KRef(podNamespace, podName))
			return nil, fwk.AsStatus(err)
		}

		// 1) Ensure that pod related to the preemptor is eligible to preempt other pods.
		nominatedNodeStatus := m.Get(pod.Status.NominatedNodeName)
		if ok, msg := ev.PodEligibleToPreemptOthers(ctx, pod, nominatedNodeStatus); !ok {
			logger.V(5).Info("Pod is not eligible for preemption", "pod", klog.KObj(pod), "reason", msg)
			return nil, fwk.NewStatus(fwk.Unschedulable, msg)
		}
	}

	// 2) Find all preemption candidates.
	allNodes, err := ev.Handler.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, fwk.AsStatus(err)
	}
	candidates, nodeToStatusMap, err := ev.findCandidates(ctx, state, allNodes, preemptor, m)
	if err != nil && len(candidates) == 0 {
		return nil, fwk.AsStatus(err)
	}

	// Return a FitError only when there are no candidates that fit the pod.
	if len(candidates) == 0 {
		logger.V(2).Info("No preemption candidate is found; preemption is not helpful for scheduling", "pod", klog.KObj(preemptor))
		fitError := &framework.FitError{
			Pod:         preemptor.GetRepresentativePod(), //TODO: not sure, assuming we have single pod as preemptor for now
			NumAllNodes: len(allNodes),
			Diagnosis: framework.Diagnosis{
				NodeToStatus: nodeToStatusMap,
				// Leave UnschedulablePlugins or PendingPlugins as nil as it won't be used on moving Pods.
			},
		}
		fitError.Diagnosis.NodeToStatus.SetAbsentNodesStatus(fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "Preemption is not helpful for scheduling"))
		// Specify nominatedNodeName to clear the pod's nominatedNodeName status, if applicable.
		return framework.NewPostFilterResultWithNominatedNode(""), fwk.NewStatus(fwk.Unschedulable, fitError.Error())
	}

	// 3) Interact with registered Extenders to filter out some candidates if needed.
	candidates, status := ev.callExtenders(logger, preemptor, candidates)
	if !status.IsSuccess() {
		return nil, status
	}

	// 4) Find the best candidate.
	bestCandidate := ev.SelectCandidate(ctx, candidates)
	if bestCandidate == nil || len(bestCandidate.Name()) == 0 {
		return nil, fwk.NewStatus(fwk.Unschedulable, "no candidate node for preemption")
	}

	logger.V(2).Info("the best candidate for the preemption is determined", "domain", bestCandidate.Name(), "preemptor", klog.KObj(preemptor))

	// 5) Perform preparation work before nominating the selected candidate.
	if ev.enableAsyncPreemption {
		ev.Executor.prepareCandidateAsync(bestCandidate, preemptor, ev.PluginName)
	} else {
		if status := ev.Executor.prepareCandidate(ctx, bestCandidate, preemptor, ev.PluginName); !status.IsSuccess() {
			return nil, status
		}
	}

	return framework.NewPostFilterResultWithNominatedNode(bestCandidate.Name()), fwk.NewStatus(fwk.Success)
}

// FindCandidates calculates a slice of preemption candidates.
// Each candidate is executable to make the given <pod> schedulable.
func (ev *Evaluator) findCandidates(ctx context.Context, state fwk.CycleState, allNodes []fwk.NodeInfo, preemptor Preemptor, m fwk.NodeToStatusReader) ([]Candidate, *framework.NodeToStatus, error) {
	pod := preemptor.GetRepresentativePod()

	if len(allNodes) == 0 {
		return nil, nil, errors.New("no nodes available")
	}
	logger := klog.FromContext(ctx)

	var potentialNodes []fwk.NodeInfo
	workloadAwarePreemptionEnabeld := utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption)
	if workloadAwarePreemptionEnabeld && preemptor.IsPodGroup() {
		potentialNodes = allNodes
	} else {
		nodes, err := m.NodesForStatusCode(ev.Handler.SnapshotSharedLister().NodeInfos(), fwk.Unschedulable)
		if err != nil {
			return nil, nil, err
		}
		potentialNodes = nodes
	}

	if len(potentialNodes) == 0 {
		logger.V(3).Info("Preemption will not help schedule pod on any node", "pod", klog.KObj(pod))
		return nil, framework.NewDefaultNodeToStatus(), nil
	}

	pdbs, err := getPodDisruptionBudgets(ev.PdbLister)
	if err != nil {
		return nil, nil, err
	}

	domains := ev.NewDomains(preemptor, potentialNodes)

	offset, candidatesNum := ev.GetOffsetAndNumCandidates(int32(len(domains)))
	return ev.DryRunPreemption(ctx, state, preemptor, domains, pdbs, offset, candidatesNum)
}

// callExtenders calls given <extenders> to select the list of feasible candidates.
// We will only check <candidates> with extenders that support preemption.
// Extenders which do not support preemption may later prevent preemptor from being scheduled on the nominated
// node. In that case, scheduler will find a different host for the preemptor in subsequent scheduling cycles.
func (ev *Evaluator) callExtenders(logger klog.Logger, preemptor Preemptor, candidates []Candidate) ([]Candidate, *fwk.Status) {
	if !preemptor.SupportExtenders() {
		return candidates, nil
	}

	extenders := ev.Handler.Extenders()
	nodeLister := ev.Handler.SnapshotSharedLister().NodeInfos()
	if len(extenders) == 0 {
		return candidates, nil
	}

	// Migrate candidate slice to victimsMap to adapt to the Extender interface.
	// It's only applicable for candidate slice that have unique nominated node name.
	victimsMap := ev.CandidatesToVictimsMap(candidates)
	if len(victimsMap) == 0 {
		return candidates, nil
	}
	for _, extender := range extenders {
		pod := preemptor.GetRepresentativePod()
		if !extender.SupportsPreemption() || !extender.IsInterested(pod) {
			continue
		}
		nodeNameToVictims, err := extender.ProcessPreemption(pod, victimsMap, nodeLister)
		if err != nil {
			if extender.IsIgnorable() {
				logger.V(2).Info("Skipped extender as it returned error and has ignorable flag set",
					"extender", extender.Name(), "err", err)
				continue
			}
			return nil, fwk.AsStatus(err)
		}
		// Check if the returned victims are valid.
		for nodeName, victims := range nodeNameToVictims {
			if victims == nil || len(victims.Pods) == 0 {
				if extender.IsIgnorable() {
					delete(nodeNameToVictims, nodeName)
					logger.V(2).Info("Ignored node for which the extender didn't report victims", "node", klog.KRef("", nodeName), "extender", extender.Name())
					continue
				}
				return nil, fwk.AsStatus(fmt.Errorf("expected at least one victim pod on node %q", nodeName))
			}
		}
		// Replace victimsMap with new result after preemption. So the
		// rest of extenders can continue use it as parameter.
		victimsMap = nodeNameToVictims

		// If node list becomes empty, no preemption can happen regardless of other extenders.
		if len(victimsMap) == 0 {
			break
		}
	}

	var newCandidates []Candidate
	for nodeName := range victimsMap {
		newCandidates = append(newCandidates, &candidate{
			victims: victimsMap[nodeName],
			name:    nodeName,
		})
	}
	return newCandidates, nil
}

// SelectCandidate chooses the best-fit candidate from given <candidates> and return it.
// NOTE: This method is exported for easier testing in default preemption.
func (ev *Evaluator) SelectCandidate(ctx context.Context, candidates []Candidate) Candidate {
	logger := klog.FromContext(ctx)

	if len(candidates) == 0 {
		return nil
	}
	if len(candidates) == 1 {
		return candidates[0]
	}

	victimsMap := ev.CandidatesToVictimsMap(candidates)
	scoreFuncs := ev.OrderedScoreFuncs(ctx, victimsMap)
	candidateNode := pickOneNodeForPreemption(logger, victimsMap, scoreFuncs)

	// Same as candidatesToVictimsMap, this logic is not applicable for out-of-tree
	// preemption plugins that exercise different candidates on the same nominated node.
	if victims := victimsMap[candidateNode]; victims != nil {
		return &candidate{
			victims: victims,
			name:    candidateNode,
		}
	}

	// We shouldn't reach here.
	utilruntime.HandleErrorWithContext(ctx, nil, "Unexpected case no candidate was selected", "candidates", candidates)
	// To not break the whole flow, return the first candidate.
	return candidates[0]
}

func getPodDisruptionBudgets(pdbLister policylisters.PodDisruptionBudgetLister) ([]*policy.PodDisruptionBudget, error) {
	if pdbLister != nil {
		return pdbLister.List(labels.Everything())
	}
	return nil, nil
}

// pickOneNodeForPreemption chooses one node among the given nodes.
// It assumes pods in each map entry are ordered by decreasing priority.
// If the scoreFuns is not empty, It picks a node based on score scoreFuns returns.
// If the scoreFuns is empty,
// It picks a node based on the following criteria:
// 1. A node with minimum number of PDB violations.
// 2. A node with minimum highest priority victim is picked.
// 3. Ties are broken by sum of priorities of all victims.
// 4. If there are still ties, node with the minimum number of victims is picked.
// 5. If there are still ties, node with the latest start time of all highest priority victims is picked.
// 6. If there are still ties, the first such node is picked (sort of randomly).
// The 'minNodes1' and 'minNodes2' are being reused here to save the memory
// allocation and garbage collection time.
func pickOneNodeForPreemption(logger klog.Logger, nodesToVictims map[string]*extenderv1.Victims, scoreFuncs []func(node string) int64) string {
	if len(nodesToVictims) == 0 {
		return ""
	}

	allCandidates := make([]string, 0, len(nodesToVictims))
	for node := range nodesToVictims {
		allCandidates = append(allCandidates, node)
	}

	if len(scoreFuncs) == 0 {
		minNumPDBViolatingScoreFunc := func(node string) int64 {
			// The smaller the NumPDBViolations, the higher the score.
			return -nodesToVictims[node].NumPDBViolations
		}
		minHighestPriorityScoreFunc := func(node string) int64 {
			// highestPodPriority is the highest priority among the victims on this node.
			highestPodPriority := corev1helpers.PodPriority(nodesToVictims[node].Pods[0])
			// The smaller the highestPodPriority, the higher the score.
			return -int64(highestPodPriority)
		}
		minSumPrioritiesScoreFunc := func(node string) int64 {
			var sumPriorities int64
			for _, pod := range nodesToVictims[node].Pods {
				// We add MaxInt32+1 to all priorities to make all of them >= 0. This is
				// needed so that a node with a few pods with negative priority is not
				// picked over a node with a smaller number of pods with the same negative
				// priority (and similar scenarios).
				sumPriorities += int64(corev1helpers.PodPriority(pod)) + int64(math.MaxInt32+1)
			}
			// The smaller the sumPriorities, the higher the score.
			return -sumPriorities
		}
		minNumPodsScoreFunc := func(node string) int64 {
			// The smaller the length of pods, the higher the score.
			return -int64(len(nodesToVictims[node].Pods))
		}
		latestStartTimeScoreFunc := func(node string) int64 {
			// Get the earliest start time of all pods on the current node.
			earliestStartTimeOnNode := util.GetEarliestPodStartTime(nodesToVictims[node])
			if earliestStartTimeOnNode == nil {
				utilruntime.HandleErrorWithLogger(logger, nil, "Unexpected nil earliestStartTime for node", "node", node)
				return int64(math.MinInt64)
			}
			// The bigger the earliestStartTimeOnNode, the higher the score.
			return earliestStartTimeOnNode.UnixNano()
		}

		// Each scoreFunc scores the nodes according to specific rules and keeps the name of the node
		// with the highest score. If and only if the scoreFunc has more than one node with the highest
		// score, we will execute the other scoreFunc in order of precedence.
		scoreFuncs = []func(string) int64{
			// A node with a minimum number of PDB is preferable.
			minNumPDBViolatingScoreFunc,
			// A node with a minimum highest priority victim is preferable.
			minHighestPriorityScoreFunc,
			// A node with the smallest sum of priorities is preferable.
			minSumPrioritiesScoreFunc,
			// A node with the minimum number of pods is preferable.
			minNumPodsScoreFunc,
			// A node with the latest start time of all highest priority victims is preferable.
			latestStartTimeScoreFunc,
			// If there are still ties, then the first Node in the list is selected.
		}
	}

	for _, f := range scoreFuncs {
		selectedNodes := []string{}
		maxScore := int64(math.MinInt64)
		for _, node := range allCandidates {
			score := f(node)
			if score > maxScore {
				maxScore = score
				selectedNodes = []string{}
			}
			if score == maxScore {
				selectedNodes = append(selectedNodes, node)
			}
		}
		if len(selectedNodes) == 1 {
			return selectedNodes[0]
		}
		allCandidates = selectedNodes
	}

	return allCandidates[0]
}

// DryRunPreemption simulates Preemption logic on <potentialNodes> in parallel,
// returns preemption candidates and a map indicating filtered nodes statuses.
// The number of candidates depends on the constraints defined in the plugin's args. In the returned list of
// candidates, ones that do not violate PDB are preferred over ones that do.
// NOTE: This method is exported for easier testing in default preemption.
func (ev *Evaluator) DryRunPreemption(ctx context.Context, state fwk.CycleState, preemptor Preemptor, domains []Domain,
	pdbs []*policy.PodDisruptionBudget, offset int32, candidatesNum int32) ([]Candidate, *framework.NodeToStatus, error) {

	fh := ev.Handler
	nonViolatingCandidates := newCandidateList(candidatesNum)
	violatingCandidates := newCandidateList(candidatesNum)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	nodeStatuses := framework.NewDefaultNodeToStatus()

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Dry run the preemption", "domainsNumber", len(domains), "pdbsNumber", len(pdbs), "offset", offset, "candidatesNumber", candidatesNum)

	var statusesLock sync.Mutex
	var errs []error
	checkDomain := func(i int) {
		domain := domains[(int(offset)+i)%len(domains)].Snapshot()
		logger.V(5).Info("Check the potential domain for preemption", "domain", domain)

		stateCopy := state.Clone()
		pods, numPDBViolations, status := ev.SelectVictimsOnDomain(ctx, stateCopy, preemptor, domain, pdbs)

		if status.IsSuccess() && len(pods) != 0 {
			victims := extenderv1.Victims{
				Pods:             pods,
				NumPDBViolations: int64(numPDBViolations),
			}
			c := &candidate{
				victims: &victims,
				name:    domain.GetName(),
			}
			if numPDBViolations == 0 {
				nonViolatingCandidates.add(c)
			} else {
				violatingCandidates.add(c)
			}
			nvcSize, vcSize := nonViolatingCandidates.size(), violatingCandidates.size()
			if nvcSize > 0 && nvcSize+vcSize >= candidatesNum {
				cancel()
			}
			return
		}
		if status.IsSuccess() && len(pods) == 0 {
			status = fwk.AsStatus(fmt.Errorf("expected at least one victim pod on domain %q", domain.GetName()))
		}
		statusesLock.Lock()
		if status.Code() == fwk.Error {
			errs = append(errs, status.AsError())
		}
		for _, node := range domain.Nodes() {
			nodeStatuses.Set(node.Node().Name, status)
		}
		statusesLock.Unlock()
	}
	fh.Parallelizer().Until(ctx, len(domains), checkDomain, ev.PluginName)
	return append(nonViolatingCandidates.get(), violatingCandidates.get()...), nodeStatuses, utilerrors.NewAggregate(errs)
}

func (ev *Evaluator) NewDomains(preemptor Preemptor, potentialNodes []fwk.NodeInfo) []Domain {
	// Get the global lister once
	nodeInfoLister := ev.Handler.SnapshotSharedLister().NodeInfos()

	workloadAwarePreemptionEnabeld := utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption)
	if workloadAwarePreemptionEnabeld && preemptor.IsPodGroup() {
		if len(potentialNodes) == 0 {
			return []Domain{}
		}

		podGroupIndex := ev.buildPodGroupIndex(nodeInfoLister)

		representative := preemptor.GetRepresentativePod().Spec.WorkloadRef.PodGroup
		domainName := fmt.Sprintf("Cluster-Scope-%s", representative)

		d := &domain{
			nodes:         potentialNodes,
			name:          domainName,
			podGroupIndex: podGroupIndex, // Pass the cache!
		}
		return []Domain{d}
	}

	// Single Pod Case: Standard Node-by-Node domains
	domains := make([]Domain, 0, len(potentialNodes))

	podGroupIndex := ev.buildPodGroupIndex(nodeInfoLister)

	for _, node := range potentialNodes {
		domains = append(domains, &domain{
			nodes:         []fwk.NodeInfo{node},
			name:          node.Node().Name,
			podGroupIndex: podGroupIndex,
		})
	}

	return domains
}

// Helper to build the cache
func (ev *Evaluator) buildPodGroupIndex(lister fwk.NodeInfoLister) map[podGroupKey][]fwk.PodInfo {
	index := make(map[podGroupKey][]fwk.PodInfo)

	workloadAwarePreemptionEnabled := utilfeature.DefaultFeatureGate.Enabled(features.WorkloadAwarePreemption)
	if !workloadAwarePreemptionEnabled {
		return index
	}

	allNodes, err := lister.List()
	if err != nil {
		return index
	}

	for _, node := range allNodes {
		for _, pi := range node.GetPods() {
			if ref := pi.GetPod().Spec.WorkloadRef; ref != nil {
				key := newPodGroupKey(pi.GetPod().Namespace, pi.GetPod().Spec.WorkloadRef)
				index[key] = append(index[key], pi)
			}
		}
	}
	return index
}

type podGroupKey struct {
	namespace    string
	workloadName string
	podGroupName string
	replicaKey   string
}

func newPodGroupKey(namespace string, workloadRef *v1.WorkloadReference) podGroupKey {
	return podGroupKey{
		namespace:    namespace,
		workloadName: workloadRef.Name,
		podGroupName: workloadRef.PodGroup,
		replicaKey:   workloadRef.PodGroupReplicaKey,
	}
}
