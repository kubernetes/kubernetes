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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/validation"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Name of the plugin used in the plugin registry and configurations.
	Name = "DefaultPreemption"
)

// DefaultPreemption is a PostFilter plugin implements the preemption logic.
type DefaultPreemption struct {
	fh        framework.Handle
	args      config.DefaultPreemptionArgs
	podLister corelisters.PodLister
	pdbLister policylisters.PodDisruptionBudgetLister
}

var _ framework.PostFilterPlugin = &DefaultPreemption{}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *DefaultPreemption) Name() string {
	return Name
}

// New initializes a new plugin and returns it.
func New(dpArgs runtime.Object, fh framework.Handle) (framework.Plugin, error) {
	args, ok := dpArgs.(*config.DefaultPreemptionArgs)
	if !ok {
		return nil, fmt.Errorf("got args of type %T, want *DefaultPreemptionArgs", dpArgs)
	}
	if err := validation.ValidateDefaultPreemptionArgs(*args); err != nil {
		return nil, err
	}
	pl := DefaultPreemption{
		fh:        fh,
		args:      *args,
		podLister: fh.SharedInformerFactory().Core().V1().Pods().Lister(),
		pdbLister: getPDBLister(fh.SharedInformerFactory()),
	}
	return &pl, nil
}

// PostFilter invoked at the postFilter extension point.
func (pl *DefaultPreemption) PostFilter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, m framework.NodeToStatusMap) (*framework.PostFilterResult, *framework.Status) {
	defer func() {
		metrics.PreemptionAttempts.Inc()
	}()

	nnn, status := pl.preempt(ctx, state, pod, m)
	if !status.IsSuccess() {
		return nil, status
	}
	// This happens when the pod is not eligible for preemption or extenders filtered all candidates.
	if nnn == "" {
		return nil, framework.NewStatus(framework.Unschedulable)
	}
	return &framework.PostFilterResult{NominatedNodeName: nnn}, framework.NewStatus(framework.Success)
}

// preempt finds nodes with pods that can be preempted to make room for "pod" to
// schedule. It chooses one of the nodes and preempts the pods on the node and
// returns 1) the node name which is picked up for preemption, 2) any possible error.
// preempt does not update its snapshot. It uses the same snapshot used in the
// scheduling cycle. This is to avoid a scenario where preempt finds feasible
// nodes without preempting any pod. When there are many pending pods in the
// scheduling queue a nominated pod will go back to the queue and behind
// other pods with the same priority. The nominated pod prevents other pods from
// using the nominated resources and the nominated pod could take a long time
// before it is retried after many other pending pods.
func (pl *DefaultPreemption) preempt(ctx context.Context, state *framework.CycleState, pod *v1.Pod, m framework.NodeToStatusMap) (string, *framework.Status) {
	cs := pl.fh.ClientSet()
	nodeLister := pl.fh.SnapshotSharedLister().NodeInfos()

	// 0) Fetch the latest version of <pod>.
	// It's safe to directly fetch pod here. Because the informer cache has already been
	// initialized when creating the Scheduler obj, i.e., factory.go#MakeDefaultErrorFunc().
	// However, tests may need to manually initialize the shared pod informer.
	podNamespace, podName := pod.Namespace, pod.Name
	pod, err := pl.podLister.Pods(pod.Namespace).Get(pod.Name)
	if err != nil {
		klog.ErrorS(err, "getting the updated preemptor pod object", "pod", klog.KRef(podNamespace, podName))
		return "", framework.AsStatus(err)
	}

	// 1) Ensure the preemptor is eligible to preempt other pods.
	if !PodEligibleToPreemptOthers(pod, nodeLister, m[pod.Status.NominatedNodeName]) {
		klog.V(5).InfoS("Pod is not eligible for more preemption", "pod", klog.KObj(pod))
		return "", nil
	}

	// 2) Find all preemption candidates.
	candidates, nodeToStatusMap, status := pl.FindCandidates(ctx, state, pod, m)
	if !status.IsSuccess() {
		return "", status
	}

	// Return a FitError only when there are no candidates that fit the pod.
	if len(candidates) == 0 {
		fitError := &framework.FitError{
			Pod:         pod,
			NumAllNodes: len(nodeToStatusMap),
			Diagnosis: framework.Diagnosis{
				NodeToStatusMap: nodeToStatusMap,
				// Leave FailedPlugins as nil as it won't be used on moving Pods.
			},
		}
		return "", framework.NewStatus(framework.Unschedulable, fitError.Error())
	}

	// 3) Interact with registered Extenders to filter out some candidates if needed.
	candidates, status = CallExtenders(pl.fh.Extenders(), pod, nodeLister, candidates)
	if !status.IsSuccess() {
		return "", status
	}

	// 4) Find the best candidate.
	bestCandidate := SelectCandidate(candidates)
	if bestCandidate == nil || len(bestCandidate.Name()) == 0 {
		return "", nil
	}

	// 5) Perform preparation work before nominating the selected candidate.
	if status := PrepareCandidate(bestCandidate, pl.fh, cs, pod, pl.Name()); !status.IsSuccess() {
		return "", status
	}

	return bestCandidate.Name(), nil
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

// getOffsetAndNumCandidates chooses a random offset and calculates the number
// of candidates that should be shortlisted for dry running preemption.
func (pl *DefaultPreemption) getOffsetAndNumCandidates(numNodes int32) (int32, int32) {
	return rand.Int31n(numNodes), pl.calculateNumCandidates(numNodes)
}

// FindCandidates calculates a slice of preemption candidates.
// Each candidate is executable to make the given <pod> schedulable.
func (pl *DefaultPreemption) FindCandidates(ctx context.Context, state *framework.CycleState, pod *v1.Pod, m framework.NodeToStatusMap) ([]Candidate, framework.NodeToStatusMap, *framework.Status) {
	allNodes, err := pl.fh.SnapshotSharedLister().NodeInfos().List()
	if err != nil {
		return nil, nil, framework.AsStatus(err)
	}
	if len(allNodes) == 0 {
		return nil, nil, framework.NewStatus(framework.Error, "no nodes available")
	}
	potentialNodes, unschedulableNodeStatus := nodesWherePreemptionMightHelp(allNodes, m)
	if len(potentialNodes) == 0 {
		klog.V(3).InfoS("Preemption will not help schedule pod on any node", "pod", klog.KObj(pod))
		// In this case, we should clean-up any existing nominated node name of the pod.
		if err := util.ClearNominatedNodeName(pl.fh.ClientSet(), pod); err != nil {
			klog.ErrorS(err, "cannot clear 'NominatedNodeName' field of pod", "pod", klog.KObj(pod))
			// We do not return as this error is not critical.
		}
		return nil, unschedulableNodeStatus, nil
	}

	pdbs, err := getPodDisruptionBudgets(pl.pdbLister)
	if err != nil {
		return nil, nil, framework.AsStatus(err)
	}

	offset, numCandidates := pl.getOffsetAndNumCandidates(int32(len(potentialNodes)))
	if klog.V(5).Enabled() {
		var sample []string
		for i := offset; i < offset+10 && i < int32(len(potentialNodes)); i++ {
			sample = append(sample, potentialNodes[i].Node().Name)
		}
		klog.Infof("from a pool of %d nodes (offset: %d, sample %d nodes: %v), ~%d candidates will be chosen", len(potentialNodes), offset, len(sample), sample, numCandidates)
	}
	candidates, nodeStatuses := dryRunPreemption(ctx, pl.fh, state, pod, potentialNodes, pdbs, offset, numCandidates)
	for node, status := range unschedulableNodeStatus {
		nodeStatuses[node] = status
	}
	return candidates, nodeStatuses, nil
}

// PodEligibleToPreemptOthers determines whether this pod should be considered
// for preempting other pods or not. If this pod has already preempted other
// pods and those are in their graceful termination period, it shouldn't be
// considered for preemption.
// We look at the node that is nominated for this pod and as long as there are
// terminating pods on the node, we don't consider this for preempting more pods.
func PodEligibleToPreemptOthers(pod *v1.Pod, nodeInfos framework.NodeInfoLister, nominatedNodeStatus *framework.Status) bool {
	if pod.Spec.PreemptionPolicy != nil && *pod.Spec.PreemptionPolicy == v1.PreemptNever {
		klog.V(5).InfoS("Pod is not eligible for preemption because it has a preemptionPolicy of Never", "pod", klog.KObj(pod))
		return false
	}
	nomNodeName := pod.Status.NominatedNodeName
	if len(nomNodeName) > 0 {
		// If the pod's nominated node is considered as UnschedulableAndUnresolvable by the filters,
		// then the pod should be considered for preempting again.
		if nominatedNodeStatus.Code() == framework.UnschedulableAndUnresolvable {
			return true
		}

		if nodeInfo, _ := nodeInfos.Get(nomNodeName); nodeInfo != nil {
			podPriority := corev1helpers.PodPriority(pod)
			for _, p := range nodeInfo.Pods {
				if p.Pod.DeletionTimestamp != nil && corev1helpers.PodPriority(p.Pod) < podPriority {
					// There is a terminating pod on the nominated node.
					return false
				}
			}
		}
	}
	return true
}

// nodesWherePreemptionMightHelp returns a list of nodes with failed predicates
// that may be satisfied by removing pods from the node.
func nodesWherePreemptionMightHelp(nodes []*framework.NodeInfo, m framework.NodeToStatusMap) ([]*framework.NodeInfo, framework.NodeToStatusMap) {
	var potentialNodes []*framework.NodeInfo
	nodeStatuses := make(framework.NodeToStatusMap)
	for _, node := range nodes {
		name := node.Node().Name
		// We rely on the status by each plugin - 'Unschedulable' or 'UnschedulableAndUnresolvable'
		// to determine whether preemption may help or not on the node.
		if m[name].Code() == framework.UnschedulableAndUnresolvable {
			nodeStatuses[node.Node().Name] = framework.NewStatus(framework.UnschedulableAndUnresolvable, "Preemption is not helpful for scheduling")
			continue
		}
		potentialNodes = append(potentialNodes, node)
	}
	return potentialNodes, nodeStatuses
}

type candidateList struct {
	idx   int32
	items []Candidate
}

func newCandidateList(size int32) *candidateList {
	return &candidateList{idx: -1, items: make([]Candidate, size)}
}

// add adds a new candidate to the internal array atomically.
func (cl *candidateList) add(c *candidate) {
	if idx := atomic.AddInt32(&cl.idx, 1); idx < int32(len(cl.items)) {
		cl.items[idx] = c
	}
}

// size returns the number of candidate stored. Note that some add() operations
// might still be executing when this is called, so care must be taken to
// ensure that all add() operations complete before accessing the elements of
// the list.
func (cl *candidateList) size() int32 {
	n := atomic.LoadInt32(&cl.idx) + 1
	if n >= int32(len(cl.items)) {
		n = int32(len(cl.items))
	}
	return n
}

// get returns the internal candidate array. This function is NOT atomic and
// assumes that all add() operations have been completed.
func (cl *candidateList) get() []Candidate {
	return cl.items[:cl.size()]
}

// dryRunPreemption simulates Preemption logic on <potentialNodes> in parallel,
// returns preemption candidates and a map indicating filtered nodes statuses.
// The number of candidates depends on the constraints defined in the plugin's args. In the returned list of
// candidates, ones that do not violate PDB are preferred over ones that do.
func dryRunPreemption(ctx context.Context, fh framework.Handle,
	state *framework.CycleState, pod *v1.Pod, potentialNodes []*framework.NodeInfo,
	pdbs []*policy.PodDisruptionBudget, offset int32, numCandidates int32) ([]Candidate, framework.NodeToStatusMap) {
	nonViolatingCandidates := newCandidateList(numCandidates)
	violatingCandidates := newCandidateList(numCandidates)
	parallelCtx, cancel := context.WithCancel(ctx)
	nodeStatuses := make(framework.NodeToStatusMap)
	var statusesLock sync.Mutex
	checkNode := func(i int) {
		nodeInfoCopy := potentialNodes[(int(offset)+i)%len(potentialNodes)].Clone()
		stateCopy := state.Clone()
		pods, numPDBViolations, status := selectVictimsOnNode(ctx, fh, stateCopy, pod, nodeInfoCopy, pdbs)
		if status.IsSuccess() {
			victims := extenderv1.Victims{
				Pods:             pods,
				NumPDBViolations: int64(numPDBViolations),
			}
			c := &candidate{
				victims: &victims,
				name:    nodeInfoCopy.Node().Name,
			}
			if numPDBViolations == 0 {
				nonViolatingCandidates.add(c)
			} else {
				violatingCandidates.add(c)
			}
			nvcSize, vcSize := nonViolatingCandidates.size(), violatingCandidates.size()
			if nvcSize > 0 && nvcSize+vcSize >= numCandidates {
				cancel()
			}
		} else {
			statusesLock.Lock()
			nodeStatuses[nodeInfoCopy.Node().Name] = status
			statusesLock.Unlock()
		}
	}
	fh.Parallelizer().Until(parallelCtx, len(potentialNodes), checkNode)
	return append(nonViolatingCandidates.get(), violatingCandidates.get()...), nodeStatuses
}

// CallExtenders calls given <extenders> to select the list of feasible candidates.
// We will only check <candidates> with extenders that support preemption.
// Extenders which do not support preemption may later prevent preemptor from being scheduled on the nominated
// node. In that case, scheduler will find a different host for the preemptor in subsequent scheduling cycles.
func CallExtenders(extenders []framework.Extender, pod *v1.Pod, nodeLister framework.NodeInfoLister,
	candidates []Candidate) ([]Candidate, *framework.Status) {
	if len(extenders) == 0 {
		return candidates, nil
	}

	// Migrate candidate slice to victimsMap to adapt to the Extender interface.
	// It's only applicable for candidate slice that have unique nominated node name.
	victimsMap := candidatesToVictimsMap(candidates)
	if len(victimsMap) == 0 {
		return candidates, nil
	}
	for _, extender := range extenders {
		if !extender.SupportsPreemption() || !extender.IsInterested(pod) {
			continue
		}
		nodeNameToVictims, err := extender.ProcessPreemption(pod, victimsMap, nodeLister)
		if err != nil {
			if extender.IsIgnorable() {
				klog.InfoS("Skipping extender as it returned error and has ignorable flag set",
					"extender", extender, "err", err)
				continue
			}
			return nil, framework.AsStatus(err)
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

// This function is not applicable for out-of-tree preemption plugins that exercise
// different preemption candidates on the same nominated node.
func candidatesToVictimsMap(candidates []Candidate) map[string]*extenderv1.Victims {
	m := make(map[string]*extenderv1.Victims)
	for _, c := range candidates {
		m[c.Name()] = c.Victims()
	}
	return m
}

// SelectCandidate chooses the best-fit candidate from given <candidates> and return it.
func SelectCandidate(candidates []Candidate) Candidate {
	if len(candidates) == 0 {
		return nil
	}
	if len(candidates) == 1 {
		return candidates[0]
	}

	victimsMap := candidatesToVictimsMap(candidates)
	candidateNode := pickOneNodeForPreemption(victimsMap)

	// Same as candidatesToVictimsMap, this logic is not applicable for out-of-tree
	// preemption plugins that exercise different candidates on the same nominated node.
	if victims := victimsMap[candidateNode]; victims != nil {
		return &candidate{
			victims: victims,
			name:    candidateNode,
		}
	}

	// We shouldn't reach here.
	klog.ErrorS(errors.New("no candidate selected"), "should not reach here", "candidates", candidates)
	// To not break the whole flow, return the first candidate.
	return candidates[0]
}

// pickOneNodeForPreemption chooses one node among the given nodes. It assumes
// pods in each map entry are ordered by decreasing priority.
// It picks a node based on the following criteria:
// 1. A node with minimum number of PDB violations.
// 2. A node with minimum highest priority victim is picked.
// 3. Ties are broken by sum of priorities of all victims.
// 4. If there are still ties, node with the minimum number of victims is picked.
// 5. If there are still ties, node with the latest start time of all highest priority victims is picked.
// 6. If there are still ties, the first such node is picked (sort of randomly).
// The 'minNodes1' and 'minNodes2' are being reused here to save the memory
// allocation and garbage collection time.
func pickOneNodeForPreemption(nodesToVictims map[string]*extenderv1.Victims) string {
	if len(nodesToVictims) == 0 {
		return ""
	}
	minNumPDBViolatingPods := int64(math.MaxInt32)
	var minNodes1 []string
	lenNodes1 := 0
	for node, victims := range nodesToVictims {
		numPDBViolatingPods := victims.NumPDBViolations
		if numPDBViolatingPods < minNumPDBViolatingPods {
			minNumPDBViolatingPods = numPDBViolatingPods
			minNodes1 = nil
			lenNodes1 = 0
		}
		if numPDBViolatingPods == minNumPDBViolatingPods {
			minNodes1 = append(minNodes1, node)
			lenNodes1++
		}
	}
	if lenNodes1 == 1 {
		return minNodes1[0]
	}

	// There are more than one node with minimum number PDB violating pods. Find
	// the one with minimum highest priority victim.
	minHighestPriority := int32(math.MaxInt32)
	var minNodes2 = make([]string, lenNodes1)
	lenNodes2 := 0
	for i := 0; i < lenNodes1; i++ {
		node := minNodes1[i]
		victims := nodesToVictims[node]
		// highestPodPriority is the highest priority among the victims on this node.
		highestPodPriority := corev1helpers.PodPriority(victims.Pods[0])
		if highestPodPriority < minHighestPriority {
			minHighestPriority = highestPodPriority
			lenNodes2 = 0
		}
		if highestPodPriority == minHighestPriority {
			minNodes2[lenNodes2] = node
			lenNodes2++
		}
	}
	if lenNodes2 == 1 {
		return minNodes2[0]
	}

	// There are a few nodes with minimum highest priority victim. Find the
	// smallest sum of priorities.
	minSumPriorities := int64(math.MaxInt64)
	lenNodes1 = 0
	for i := 0; i < lenNodes2; i++ {
		var sumPriorities int64
		node := minNodes2[i]
		for _, pod := range nodesToVictims[node].Pods {
			// We add MaxInt32+1 to all priorities to make all of them >= 0. This is
			// needed so that a node with a few pods with negative priority is not
			// picked over a node with a smaller number of pods with the same negative
			// priority (and similar scenarios).
			sumPriorities += int64(corev1helpers.PodPriority(pod)) + int64(math.MaxInt32+1)
		}
		if sumPriorities < minSumPriorities {
			minSumPriorities = sumPriorities
			lenNodes1 = 0
		}
		if sumPriorities == minSumPriorities {
			minNodes1[lenNodes1] = node
			lenNodes1++
		}
	}
	if lenNodes1 == 1 {
		return minNodes1[0]
	}

	// There are a few nodes with minimum highest priority victim and sum of priorities.
	// Find one with the minimum number of pods.
	minNumPods := math.MaxInt32
	lenNodes2 = 0
	for i := 0; i < lenNodes1; i++ {
		node := minNodes1[i]
		numPods := len(nodesToVictims[node].Pods)
		if numPods < minNumPods {
			minNumPods = numPods
			lenNodes2 = 0
		}
		if numPods == minNumPods {
			minNodes2[lenNodes2] = node
			lenNodes2++
		}
	}
	if lenNodes2 == 1 {
		return minNodes2[0]
	}

	// There are a few nodes with same number of pods.
	// Find the node that satisfies latest(earliestStartTime(all highest-priority pods on node))
	latestStartTime := util.GetEarliestPodStartTime(nodesToVictims[minNodes2[0]])
	if latestStartTime == nil {
		// If the earliest start time of all pods on the 1st node is nil, just return it,
		// which is not expected to happen.
		klog.ErrorS(errors.New("earliestStartTime is nil for node"), "should not reach here", "node", minNodes2[0])
		return minNodes2[0]
	}
	nodeToReturn := minNodes2[0]
	for i := 1; i < lenNodes2; i++ {
		node := minNodes2[i]
		// Get earliest start time of all pods on the current node.
		earliestStartTimeOnNode := util.GetEarliestPodStartTime(nodesToVictims[node])
		if earliestStartTimeOnNode == nil {
			klog.ErrorS(errors.New("earliestStartTime is nil for node"), "should not reach here", "node", node)
			continue
		}
		if earliestStartTimeOnNode.After(latestStartTime.Time) {
			latestStartTime = earliestStartTimeOnNode
			nodeToReturn = node
		}
	}

	return nodeToReturn
}

// selectVictimsOnNode finds minimum set of pods on the given node that should
// be preempted in order to make enough room for "pod" to be scheduled. The
// minimum set selected is subject to the constraint that a higher-priority pod
// is never preempted when a lower-priority pod could be (higher/lower relative
// to one another, not relative to the preemptor "pod").
// The algorithm first checks if the pod can be scheduled on the node when all the
// lower priority pods are gone. If so, it sorts all the lower priority pods by
// their priority and then puts them into two groups of those whose PodDisruptionBudget
// will be violated if preempted and other non-violating pods. Both groups are
// sorted by priority. It first tries to reprieve as many PDB violating pods as
// possible and then does them same for non-PDB-violating pods while checking
// that the "pod" can still fit on the node.
// NOTE: This function assumes that it is never called if "pod" cannot be scheduled
// due to pod affinity, node affinity, or node anti-affinity reasons. None of
// these predicates can be satisfied by removing more pods from the node.
func selectVictimsOnNode(
	ctx context.Context,
	fh framework.Handle,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeInfo *framework.NodeInfo,
	pdbs []*policy.PodDisruptionBudget,
) ([]*v1.Pod, int, *framework.Status) {
	var potentialVictims []*framework.PodInfo
	removePod := func(rpi *framework.PodInfo) error {
		if err := nodeInfo.RemovePod(rpi.Pod); err != nil {
			return err
		}
		status := fh.RunPreFilterExtensionRemovePod(ctx, state, pod, rpi, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	addPod := func(api *framework.PodInfo) error {
		nodeInfo.AddPodInfo(api)
		status := fh.RunPreFilterExtensionAddPod(ctx, state, pod, api, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := corev1helpers.PodPriority(pod)
	for _, pi := range nodeInfo.Pods {
		if corev1helpers.PodPriority(pi.Pod) < podPriority {
			potentialVictims = append(potentialVictims, pi)
			if err := removePod(pi); err != nil {
				return nil, 0, framework.AsStatus(err)
			}
		}
	}

	// No potential victims are found, and so we don't need to evaluate the node again since its state didn't change.
	if len(potentialVictims) == 0 {
		message := fmt.Sprintf("No victims found on node %v for preemptor pod %v", nodeInfo.Node().Name, pod.Name)
		return nil, 0, framework.NewStatus(framework.UnschedulableAndUnresolvable, message)
	}

	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption. The only
	// condition that we could check is if the "pod" is failing to schedule due to
	// inter-pod affinity to one or more victims, but we have decided not to
	// support this case for performance reasons. Having affinity to lower
	// priority pods is not a recommended configuration anyway.
	if status := fh.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo); !status.IsSuccess() {
		return nil, 0, status
	}
	var victims []*v1.Pod
	numViolatingVictim := 0
	sort.Slice(potentialVictims, func(i, j int) bool { return util.MoreImportantPod(potentialVictims[i].Pod, potentialVictims[j].Pod) })
	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest priority victims.
	violatingVictims, nonViolatingVictims := filterPodsWithPDBViolation(potentialVictims, pdbs)
	reprievePod := func(pi *framework.PodInfo) (bool, error) {
		if err := addPod(pi); err != nil {
			return false, err
		}
		status := fh.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo)
		fits := status.IsSuccess()
		if !fits {
			if err := removePod(pi); err != nil {
				return false, err
			}
			rpi := pi.Pod
			victims = append(victims, rpi)
			klog.V(5).InfoS("Pod is a potential preemption victim on node", "pod", klog.KObj(rpi), "node", klog.KObj(nodeInfo.Node()))
		}
		return fits, nil
	}
	for _, p := range violatingVictims {
		if fits, err := reprievePod(p); err != nil {
			return nil, 0, framework.AsStatus(err)
		} else if !fits {
			numViolatingVictim++
		}
	}
	// Now we try to reprieve non-violating victims.
	for _, p := range nonViolatingVictims {
		if _, err := reprievePod(p); err != nil {
			return nil, 0, framework.AsStatus(err)
		}
	}
	return victims, numViolatingVictim, framework.NewStatus(framework.Success)
}

// PrepareCandidate does some preparation work before nominating the selected candidate:
// - Evict the victim pods
// - Reject the victim pods if they are in waitingPod map
// - Clear the low-priority pods' nominatedNodeName status if needed
func PrepareCandidate(c Candidate, fh framework.Handle, cs kubernetes.Interface, pod *v1.Pod, pluginName string) *framework.Status {
	for _, victim := range c.Victims().Pods {
		if err := util.DeletePod(cs, victim); err != nil {
			klog.ErrorS(err, "preempting pod", "pod", klog.KObj(victim))
			return framework.AsStatus(err)
		}
		// If the victim is a WaitingPod, send a reject message to the PermitPlugin
		if waitingPod := fh.GetWaitingPod(victim.UID); waitingPod != nil {
			waitingPod.Reject(pluginName, "preempted")
		}
		fh.EventRecorder().Eventf(victim, pod, v1.EventTypeNormal, "Preempted", "Preempting", "Preempted by %v/%v on node %v",
			pod.Namespace, pod.Name, c.Name())
	}
	metrics.PreemptionVictims.Observe(float64(len(c.Victims().Pods)))

	// Lower priority pods nominated to run on this node, may no longer fit on
	// this node. So, we should remove their nomination. Removing their
	// nomination updates these pods and moves them to the active queue. It
	// lets scheduler find another place for them.
	nominatedPods := getLowerPriorityNominatedPods(fh, pod, c.Name())
	if err := util.ClearNominatedNodeName(cs, nominatedPods...); err != nil {
		klog.ErrorS(err, "cannot clear 'NominatedNodeName' field")
		// We do not return as this error is not critical.
	}

	return nil
}

// getLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given node.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PreFilter state per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func getLowerPriorityNominatedPods(pn framework.PodNominator, pod *v1.Pod, nodeName string) []*v1.Pod {
	podInfos := pn.NominatedPodsForNode(nodeName)

	if len(podInfos) == 0 {
		return nil
	}

	var lowerPriorityPods []*v1.Pod
	podPriority := corev1helpers.PodPriority(pod)
	for _, pi := range podInfos {
		if corev1helpers.PodPriority(pi.Pod) < podPriority {
			lowerPriorityPods = append(lowerPriorityPods, pi.Pod)
		}
	}
	return lowerPriorityPods
}

// filterPodsWithPDBViolation groups the given "pods" into two groups of "violatingPods"
// and "nonViolatingPods" based on whether their PDBs will be violated if they are
// preempted.
// This function is stable and does not change the order of received pods. So, if it
// receives a sorted list, grouping will preserve the order of the input list.
func filterPodsWithPDBViolation(podInfos []*framework.PodInfo, pdbs []*policy.PodDisruptionBudget) (violatingPodInfos, nonViolatingPodInfos []*framework.PodInfo) {
	pdbsAllowed := make([]int32, len(pdbs))
	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	for _, podInfo := range podInfos {
		pod := podInfo.Pod
		pdbForPodIsViolated := false
		// A pod with no labels will not match any PDB. So, no need to check.
		if len(pod.Labels) != 0 {
			for i, pdb := range pdbs {
				if pdb.Namespace != pod.Namespace {
					continue
				}
				selector, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
				if err != nil {
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

func getPDBLister(informerFactory informers.SharedInformerFactory) policylisters.PodDisruptionBudgetLister {
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.PodDisruptionBudget) {
		return informerFactory.Policy().V1().PodDisruptionBudgets().Lister()
	}
	return nil
}

func getPodDisruptionBudgets(pdbLister policylisters.PodDisruptionBudgetLister) ([]*policy.PodDisruptionBudget, error) {
	if pdbLister != nil {
		return pdbLister.List(labels.Everything())
	}
	return nil, nil
}
