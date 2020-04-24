/*
Copyright 2014 The Kubernetes Authors.

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

package core

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	"k8s.io/kubernetes/pkg/scheduler/internal/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	"k8s.io/kubernetes/pkg/scheduler/util"
	utiltrace "k8s.io/utils/trace"
)

const (
	// minFeasibleNodesToFind is the minimum number of nodes that would be scored
	// in each scheduling cycle. This is a semi-arbitrary value to ensure that a
	// certain minimum of nodes are checked for feasibility. This in turn helps
	// ensure a minimum level of spreading.
	minFeasibleNodesToFind = 100
	// minFeasibleNodesPercentageToFind is the minimum percentage of nodes that
	// would be scored in each scheduling cycle. This is a semi-arbitrary value
	// to ensure that a certain minimum of nodes are checked for feasibility.
	// This in turn helps ensure a minimum level of spreading.
	minFeasibleNodesPercentageToFind = 5
)

// FitError describes a fit error of a pod.
type FitError struct {
	Pod                   *v1.Pod
	NumAllNodes           int
	FilteredNodesStatuses framework.NodeToStatusMap
}

// ErrNoNodesAvailable is used to describe the error that no nodes available to schedule pods.
var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

const (
	// NoNodeAvailableMsg is used to format message when no nodes available.
	NoNodeAvailableMsg = "0/%v nodes are available"
)

// Error returns detailed information of why the pod failed to fit on each node
func (f *FitError) Error() string {
	reasons := make(map[string]int)
	for _, status := range f.FilteredNodesStatuses {
		for _, reason := range status.Reasons() {
			reasons[reason]++
		}
	}

	sortReasonsHistogram := func() []string {
		var reasonStrings []string
		for k, v := range reasons {
			reasonStrings = append(reasonStrings, fmt.Sprintf("%v %v", v, k))
		}
		sort.Strings(reasonStrings)
		return reasonStrings
	}
	reasonMsg := fmt.Sprintf(NoNodeAvailableMsg+": %v.", f.NumAllNodes, strings.Join(sortReasonsHistogram(), ", "))
	return reasonMsg
}

// ScheduleAlgorithm is an interface implemented by things that know how to schedule pods
// onto machines.
// TODO: Rename this type.
type ScheduleAlgorithm interface {
	Schedule(context.Context, *profile.Profile, *framework.CycleState, *v1.Pod) (scheduleResult ScheduleResult, err error)
	// Preempt receives scheduling errors for a pod and tries to create room for
	// the pod by preempting lower priority pods if possible.
	// It returns the node where preemption happened, a list of preempted pods, a
	// list of pods whose nominated node name should be removed, and error if any.
	Preempt(context.Context, *profile.Profile, *framework.CycleState, *v1.Pod, error) (selectedNode string, preemptedPods []*v1.Pod, cleanupNominatedPods []*v1.Pod, err error)
	// Extenders returns a slice of extender config. This is exposed for
	// testing.
	Extenders() []framework.Extender
}

// ScheduleResult represents the result of one pod scheduled. It will contain
// the final selected Node, along with the selected intermediate information.
type ScheduleResult struct {
	// Name of the scheduler suggest host
	SuggestedHost string
	// Number of nodes scheduler evaluated on one pod scheduled
	EvaluatedNodes int
	// Number of feasible nodes on one pod scheduled
	FeasibleNodes int
}

type genericScheduler struct {
	cache                    internalcache.Cache
	podNominator             framework.PodNominator
	extenders                []framework.Extender
	nodeInfoSnapshot         *internalcache.Snapshot
	pvcLister                corelisters.PersistentVolumeClaimLister
	pdbLister                policylisters.PodDisruptionBudgetLister
	disablePreemption        bool
	percentageOfNodesToScore int32
	nextStartNodeIndex       int
}

// snapshot snapshots scheduler cache and node infos for all fit and priority
// functions.
func (g *genericScheduler) snapshot() error {
	// Used for all fit and priority funcs.
	return g.cache.UpdateSnapshot(g.nodeInfoSnapshot)
}

// Schedule tries to schedule the given pod to one of the nodes in the node list.
// If it succeeds, it will return the name of the node.
// If it fails, it will return a FitError error with reasons.
func (g *genericScheduler) Schedule(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod) (result ScheduleResult, err error) {
	trace := utiltrace.New("Scheduling", utiltrace.Field{Key: "namespace", Value: pod.Namespace}, utiltrace.Field{Key: "name", Value: pod.Name})
	defer trace.LogIfLong(100 * time.Millisecond)

	if err := podPassesBasicChecks(pod, g.pvcLister); err != nil {
		return result, err
	}
	trace.Step("Basic checks done")

	if err := g.snapshot(); err != nil {
		return result, err
	}
	trace.Step("Snapshotting scheduler cache and node infos done")

	if g.nodeInfoSnapshot.NumNodes() == 0 {
		return result, ErrNoNodesAvailable
	}

	startPredicateEvalTime := time.Now()
	filteredNodes, filteredNodesStatuses, err := g.findNodesThatFitPod(ctx, prof, state, pod)
	if err != nil {
		return result, err
	}
	trace.Step("Computing predicates done")

	if len(filteredNodes) == 0 {
		return result, &FitError{
			Pod:                   pod,
			NumAllNodes:           g.nodeInfoSnapshot.NumNodes(),
			FilteredNodesStatuses: filteredNodesStatuses,
		}
	}

	metrics.DeprecatedSchedulingAlgorithmPredicateEvaluationSecondsDuration.Observe(metrics.SinceInSeconds(startPredicateEvalTime))
	metrics.DeprecatedSchedulingDuration.WithLabelValues(metrics.PredicateEvaluation).Observe(metrics.SinceInSeconds(startPredicateEvalTime))

	startPriorityEvalTime := time.Now()
	// When only one node after predicate, just use it.
	if len(filteredNodes) == 1 {
		metrics.DeprecatedSchedulingAlgorithmPriorityEvaluationSecondsDuration.Observe(metrics.SinceInSeconds(startPriorityEvalTime))
		return ScheduleResult{
			SuggestedHost:  filteredNodes[0].Name,
			EvaluatedNodes: 1 + len(filteredNodesStatuses),
			FeasibleNodes:  1,
		}, nil
	}

	priorityList, err := g.prioritizeNodes(ctx, prof, state, pod, filteredNodes)
	if err != nil {
		return result, err
	}

	metrics.DeprecatedSchedulingAlgorithmPriorityEvaluationSecondsDuration.Observe(metrics.SinceInSeconds(startPriorityEvalTime))
	metrics.DeprecatedSchedulingDuration.WithLabelValues(metrics.PriorityEvaluation).Observe(metrics.SinceInSeconds(startPriorityEvalTime))

	host, err := g.selectHost(priorityList)
	trace.Step("Prioritizing done")

	return ScheduleResult{
		SuggestedHost:  host,
		EvaluatedNodes: len(filteredNodes) + len(filteredNodesStatuses),
		FeasibleNodes:  len(filteredNodes),
	}, err
}

func (g *genericScheduler) Extenders() []framework.Extender {
	return g.extenders
}

// selectHost takes a prioritized list of nodes and then picks one
// in a reservoir sampling manner from the nodes that had the highest score.
func (g *genericScheduler) selectHost(nodeScoreList framework.NodeScoreList) (string, error) {
	if len(nodeScoreList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}
	maxScore := nodeScoreList[0].Score
	selected := nodeScoreList[0].Name
	cntOfMaxScore := 1
	for _, ns := range nodeScoreList[1:] {
		if ns.Score > maxScore {
			maxScore = ns.Score
			selected = ns.Name
			cntOfMaxScore = 1
		} else if ns.Score == maxScore {
			cntOfMaxScore++
			if rand.Intn(cntOfMaxScore) == 0 {
				// Replace the candidate with probability of 1/cntOfMaxScore
				selected = ns.Name
			}
		}
	}
	return selected, nil
}

// preempt finds nodes with pods that can be preempted to make room for "pod" to
// schedule. It chooses one of the nodes and preempts the pods on the node and
// returns 1) the node, 2) the list of preempted pods if such a node is found,
// 3) A list of pods whose nominated node name should be cleared, and 4) any
// possible error.
// Preempt does not update its snapshot. It uses the same snapshot used in the
// scheduling cycle. This is to avoid a scenario where preempt finds feasible
// nodes without preempting any pod. When there are many pending pods in the
// scheduling queue a nominated pod will go back to the queue and behind
// other pods with the same priority. The nominated pod prevents other pods from
// using the nominated resources and the nominated pod could take a long time
// before it is retried after many other pending pods.
func (g *genericScheduler) Preempt(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod, scheduleErr error) (string, []*v1.Pod, []*v1.Pod, error) {
	// Scheduler may return various types of errors. Consider preemption only if
	// the error is of type FitError.
	fitError, ok := scheduleErr.(*FitError)
	if !ok || fitError == nil {
		return "", nil, nil, nil
	}
	if !podEligibleToPreemptOthers(pod, g.nodeInfoSnapshot.NodeInfos()) {
		klog.V(5).Infof("Pod %v/%v is not eligible for more preemption.", pod.Namespace, pod.Name)
		return "", nil, nil, nil
	}
	allNodes, err := g.nodeInfoSnapshot.NodeInfos().List()
	if err != nil {
		return "", nil, nil, err
	}
	if len(allNodes) == 0 {
		return "", nil, nil, ErrNoNodesAvailable
	}
	potentialNodes := nodesWherePreemptionMightHelp(allNodes, fitError)
	if len(potentialNodes) == 0 {
		klog.V(3).Infof("Preemption will not help schedule pod %v/%v on any node.", pod.Namespace, pod.Name)
		// In this case, we should clean-up any existing nominated node name of the pod.
		return "", nil, []*v1.Pod{pod}, nil
	}
	var pdbs []*policy.PodDisruptionBudget
	if g.pdbLister != nil {
		pdbs, err = g.pdbLister.List(labels.Everything())
		if err != nil {
			return "", nil, nil, err
		}
	}
	nodeNameToVictims, err := selectNodesForPreemption(ctx, prof, g.podNominator, state, pod, potentialNodes, pdbs)
	if err != nil {
		return "", nil, nil, err
	}

	// We will only check nodeNameToVictims with extenders that support preemption.
	// Extenders which do not support preemption may later prevent preemptor from being scheduled on the nominated
	// node. In that case, scheduler will find a different host for the preemptor in subsequent scheduling cycles.
	nodeNameToVictims, err = g.processPreemptionWithExtenders(pod, nodeNameToVictims)
	if err != nil {
		return "", nil, nil, err
	}

	candidateNode := pickOneNodeForPreemption(nodeNameToVictims)
	if len(candidateNode) == 0 {
		return "", nil, nil, nil
	}

	// Lower priority pods nominated to run on this node, may no longer fit on
	// this node. So, we should remove their nomination. Removing their
	// nomination updates these pods and moves them to the active queue. It
	// lets scheduler find another place for them.
	nominatedPods := g.getLowerPriorityNominatedPods(pod, candidateNode)
	return candidateNode, nodeNameToVictims[candidateNode].Pods, nominatedPods, nil
}

// processPreemptionWithExtenders processes preemption with extenders
func (g *genericScheduler) processPreemptionWithExtenders(
	pod *v1.Pod,
	nodeNameToVictims map[string]*extenderv1.Victims,
) (map[string]*extenderv1.Victims, error) {
	if len(nodeNameToVictims) > 0 {
		for _, extender := range g.extenders {
			if extender.SupportsPreemption() && extender.IsInterested(pod) {
				newNodeNameToVictims, err := extender.ProcessPreemption(
					pod,
					nodeNameToVictims,
					g.nodeInfoSnapshot.NodeInfos(),
				)
				if err != nil {
					if extender.IsIgnorable() {
						klog.Warningf("Skipping extender %v as it returned error %v and has ignorable flag set",
							extender, err)
						continue
					}
					return nil, err
				}

				// Replace nodeNameToVictims with new result after preemption. So the
				// rest of extenders can continue use it as parameter.
				nodeNameToVictims = newNodeNameToVictims

				// If node list becomes empty, no preemption can happen regardless of other extenders.
				if len(nodeNameToVictims) == 0 {
					break
				}
			}
		}
	}

	return nodeNameToVictims, nil
}

// getLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given node.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PreFilter state per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func (g *genericScheduler) getLowerPriorityNominatedPods(pod *v1.Pod, nodeName string) []*v1.Pod {
	pods := g.podNominator.NominatedPodsForNode(nodeName)

	if len(pods) == 0 {
		return nil
	}

	var lowerPriorityPods []*v1.Pod
	podPriority := podutil.GetPodPriority(pod)
	for _, p := range pods {
		if podutil.GetPodPriority(p) < podPriority {
			lowerPriorityPods = append(lowerPriorityPods, p)
		}
	}
	return lowerPriorityPods
}

// numFeasibleNodesToFind returns the number of feasible nodes that once found, the scheduler stops
// its search for more feasible nodes.
func (g *genericScheduler) numFeasibleNodesToFind(numAllNodes int32) (numNodes int32) {
	if numAllNodes < minFeasibleNodesToFind || g.percentageOfNodesToScore >= 100 {
		return numAllNodes
	}

	adaptivePercentage := g.percentageOfNodesToScore
	if adaptivePercentage <= 0 {
		basePercentageOfNodesToScore := int32(50)
		adaptivePercentage = basePercentageOfNodesToScore - numAllNodes/125
		if adaptivePercentage < minFeasibleNodesPercentageToFind {
			adaptivePercentage = minFeasibleNodesPercentageToFind
		}
	}

	numNodes = numAllNodes * adaptivePercentage / 100
	if numNodes < minFeasibleNodesToFind {
		return minFeasibleNodesToFind
	}

	return numNodes
}

// Filters the nodes to find the ones that fit the pod based on the framework
// filter plugins and filter extenders.
func (g *genericScheduler) findNodesThatFitPod(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod) ([]*v1.Node, framework.NodeToStatusMap, error) {
	// Run "prefilter" plugins.
	s := prof.RunPreFilterPlugins(ctx, state, pod)
	if !s.IsSuccess() {
		return nil, nil, s.AsError()
	}

	filteredNodesStatuses := make(framework.NodeToStatusMap)
	filtered, err := g.findNodesThatPassFilters(ctx, prof, state, pod, filteredNodesStatuses)
	if err != nil {
		return nil, nil, err
	}

	filtered, err = g.findNodesThatPassExtenders(pod, filtered, filteredNodesStatuses)
	if err != nil {
		return nil, nil, err
	}
	return filtered, filteredNodesStatuses, nil
}

// findNodesThatPassFilters finds the nodes that fit the filter plugins.
func (g *genericScheduler) findNodesThatPassFilters(ctx context.Context, prof *profile.Profile, state *framework.CycleState, pod *v1.Pod, statuses framework.NodeToStatusMap) ([]*v1.Node, error) {
	allNodes, err := g.nodeInfoSnapshot.NodeInfos().List()
	if err != nil {
		return nil, err
	}

	numNodesToFind := g.numFeasibleNodesToFind(int32(len(allNodes)))

	// Create filtered list with enough space to avoid growing it
	// and allow assigning.
	filtered := make([]*v1.Node, numNodesToFind)

	if !prof.HasFilterPlugins() {
		for i := range filtered {
			filtered[i] = allNodes[i].Node()
		}
		g.nextStartNodeIndex = (g.nextStartNodeIndex + len(filtered)) % len(allNodes)
		return filtered, nil
	}

	errCh := parallelize.NewErrorChannel()
	var statusesLock sync.Mutex
	var filteredLen int32
	ctx, cancel := context.WithCancel(ctx)
	checkNode := func(i int) {
		// We check the nodes starting from where we left off in the previous scheduling cycle,
		// this is to make sure all nodes have the same chance of being examined across pods.
		nodeInfo := allNodes[(g.nextStartNodeIndex+i)%len(allNodes)]
		fits, status, err := podPassesFiltersOnNode(ctx, prof, g.podNominator, state, pod, nodeInfo)
		if err != nil {
			errCh.SendErrorWithCancel(err, cancel)
			return
		}
		if fits {
			length := atomic.AddInt32(&filteredLen, 1)
			if length > numNodesToFind {
				cancel()
				atomic.AddInt32(&filteredLen, -1)
			} else {
				filtered[length-1] = nodeInfo.Node()
			}
		} else {
			statusesLock.Lock()
			if !status.IsSuccess() {
				statuses[nodeInfo.Node().Name] = status
			}
			statusesLock.Unlock()
		}
	}

	beginCheckNode := time.Now()
	statusCode := framework.Success
	defer func() {
		// We record Filter extension point latency here instead of in framework.go because framework.RunFilterPlugins
		// function is called for each node, whereas we want to have an overall latency for all nodes per scheduling cycle.
		// Note that this latency also includes latency for `addNominatedPods`, which calls framework.RunPreFilterAddPod.
		metrics.FrameworkExtensionPointDuration.WithLabelValues(framework.Filter, statusCode.String()).Observe(metrics.SinceInSeconds(beginCheckNode))
	}()

	// Stops searching for more nodes once the configured number of feasible nodes
	// are found.
	parallelize.Until(ctx, len(allNodes), checkNode)
	processedNodes := int(filteredLen) + len(statuses)
	g.nextStartNodeIndex = (g.nextStartNodeIndex + processedNodes) % len(allNodes)

	filtered = filtered[:filteredLen]
	if err := errCh.ReceiveError(); err != nil {
		statusCode = framework.Error
		return nil, err
	}
	return filtered, nil
}

func (g *genericScheduler) findNodesThatPassExtenders(pod *v1.Pod, filtered []*v1.Node, statuses framework.NodeToStatusMap) ([]*v1.Node, error) {
	for _, extender := range g.extenders {
		if len(filtered) == 0 {
			break
		}
		if !extender.IsInterested(pod) {
			continue
		}
		filteredList, failedMap, err := extender.Filter(pod, filtered)
		if err != nil {
			if extender.IsIgnorable() {
				klog.Warningf("Skipping extender %v as it returned error %v and has ignorable flag set",
					extender, err)
				continue
			}
			return nil, err
		}

		for failedNodeName, failedMsg := range failedMap {
			if _, found := statuses[failedNodeName]; !found {
				statuses[failedNodeName] = framework.NewStatus(framework.Unschedulable, failedMsg)
			} else {
				statuses[failedNodeName].AppendReason(failedMsg)
			}
		}
		filtered = filteredList
	}
	return filtered, nil
}

// addNominatedPods adds pods with equal or greater priority which are nominated
// to run on the node. It returns 1) whether any pod was added, 2) augmented cycleState,
// 3) augmented nodeInfo.
func addNominatedPods(ctx context.Context, pr framework.PluginsRunner, nominator framework.PodNominator, pod *v1.Pod, state *framework.CycleState, nodeInfo *framework.NodeInfo) (bool, *framework.CycleState, *framework.NodeInfo, error) {
	if nominator == nil || nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen only in tests.
		return false, state, nodeInfo, nil
	}
	nominatedPods := nominator.NominatedPodsForNode(nodeInfo.Node().Name)
	if len(nominatedPods) == 0 {
		return false, state, nodeInfo, nil
	}
	nodeInfoOut := nodeInfo.Clone()
	stateOut := state.Clone()
	podsAdded := false
	for _, p := range nominatedPods {
		if podutil.GetPodPriority(p) >= podutil.GetPodPriority(pod) && p.UID != pod.UID {
			nodeInfoOut.AddPod(p)
			status := pr.RunPreFilterExtensionAddPod(ctx, stateOut, pod, p, nodeInfoOut)
			if !status.IsSuccess() {
				return false, state, nodeInfo, status.AsError()
			}
			podsAdded = true
		}
	}
	return podsAdded, stateOut, nodeInfoOut, nil
}

// podPassesFiltersOnNode checks whether a node given by NodeInfo satisfies the
// filter plugins.
// This function is called from two different places: Schedule and Preempt.
// When it is called from Schedule, we want to test whether the pod is
// schedulable on the node with all the existing pods on the node plus higher
// and equal priority pods nominated to run on the node.
// When it is called from Preempt, we should remove the victims of preemption
// and add the nominated pods. Removal of the victims is done by
// SelectVictimsOnNode(). Preempt removes victims from PreFilter state and
// NodeInfo before calling this function.
func podPassesFiltersOnNode(
	ctx context.Context,
	pr framework.PluginsRunner,
	nominator framework.PodNominator,
	state *framework.CycleState,
	pod *v1.Pod,
	info *framework.NodeInfo,
) (bool, *framework.Status, error) {
	var status *framework.Status

	podsAdded := false
	// We run filters twice in some cases. If the node has greater or equal priority
	// nominated pods, we run them when those pods are added to PreFilter state and nodeInfo.
	// If all filters succeed in this pass, we run them again when these
	// nominated pods are not added. This second pass is necessary because some
	// filters such as inter-pod affinity may not pass without the nominated pods.
	// If there are no nominated pods for the node or if the first run of the
	// filters fail, we don't run the second pass.
	// We consider only equal or higher priority pods in the first pass, because
	// those are the current "pod" must yield to them and not take a space opened
	// for running them. It is ok if the current "pod" take resources freed for
	// lower priority pods.
	// Requiring that the new pod is schedulable in both circumstances ensures that
	// we are making a conservative decision: filters like resources and inter-pod
	// anti-affinity are more likely to fail when the nominated pods are treated
	// as running, while filters like pod affinity are more likely to fail when
	// the nominated pods are treated as not running. We can't just assume the
	// nominated pods are running because they are not running right now and in fact,
	// they may end up getting scheduled to a different node.
	for i := 0; i < 2; i++ {
		stateToUse := state
		nodeInfoToUse := info
		if i == 0 {
			var err error
			podsAdded, stateToUse, nodeInfoToUse, err = addNominatedPods(ctx, pr, nominator, pod, state, info)
			if err != nil {
				return false, nil, err
			}
		} else if !podsAdded || !status.IsSuccess() {
			break
		}

		statusMap := pr.RunFilterPlugins(ctx, stateToUse, pod, nodeInfoToUse)
		status = statusMap.Merge()
		if !status.IsSuccess() && !status.IsUnschedulable() {
			return false, status, status.AsError()
		}
	}

	return status.IsSuccess(), status, nil
}

// prioritizeNodes prioritizes the nodes by running the score plugins,
// which return a score for each node from the call to RunScorePlugins().
// The scores from each plugin are added together to make the score for that node, then
// any extenders are run as well.
// All scores are finally combined (added) to get the total weighted scores of all nodes
func (g *genericScheduler) prioritizeNodes(
	ctx context.Context,
	prof *profile.Profile,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) (framework.NodeScoreList, error) {
	// If no priority configs are provided, then all nodes will have a score of one.
	// This is required to generate the priority list in the required format
	if len(g.extenders) == 0 && !prof.HasScorePlugins() {
		result := make(framework.NodeScoreList, 0, len(nodes))
		for i := range nodes {
			result = append(result, framework.NodeScore{
				Name:  nodes[i].Name,
				Score: 1,
			})
		}
		return result, nil
	}

	// Run PreScore plugins.
	preScoreStatus := prof.RunPreScorePlugins(ctx, state, pod, nodes)
	if !preScoreStatus.IsSuccess() {
		return nil, preScoreStatus.AsError()
	}

	// Run the Score plugins.
	scoresMap, scoreStatus := prof.RunScorePlugins(ctx, state, pod, nodes)
	if !scoreStatus.IsSuccess() {
		return nil, scoreStatus.AsError()
	}

	if klog.V(10).Enabled() {
		for plugin, nodeScoreList := range scoresMap {
			klog.Infof("Plugin %s scores on %v/%v => %v", plugin, pod.Namespace, pod.Name, nodeScoreList)
		}
	}

	// Summarize all scores.
	result := make(framework.NodeScoreList, 0, len(nodes))

	for i := range nodes {
		result = append(result, framework.NodeScore{Name: nodes[i].Name, Score: 0})
		for j := range scoresMap {
			result[i].Score += scoresMap[j][i].Score
		}
	}

	if len(g.extenders) != 0 && nodes != nil {
		var mu sync.Mutex
		var wg sync.WaitGroup
		combinedScores := make(map[string]int64, len(nodes))
		for i := range g.extenders {
			if !g.extenders[i].IsInterested(pod) {
				continue
			}
			wg.Add(1)
			go func(extIndex int) {
				metrics.SchedulerGoroutines.WithLabelValues("prioritizing_extender").Inc()
				defer func() {
					metrics.SchedulerGoroutines.WithLabelValues("prioritizing_extender").Dec()
					wg.Done()
				}()
				prioritizedList, weight, err := g.extenders[extIndex].Prioritize(pod, nodes)
				if err != nil {
					// Prioritization errors from extender can be ignored, let k8s/other extenders determine the priorities
					return
				}
				mu.Lock()
				for i := range *prioritizedList {
					host, score := (*prioritizedList)[i].Host, (*prioritizedList)[i].Score
					if klog.V(10).Enabled() {
						klog.Infof("%v -> %v: %v, Score: (%d)", util.GetPodFullName(pod), host, g.extenders[extIndex].Name(), score)
					}
					combinedScores[host] += score * weight
				}
				mu.Unlock()
			}(i)
		}
		// wait for all go routines to finish
		wg.Wait()
		for i := range result {
			// MaxExtenderPriority may diverge from the max priority used in the scheduler and defined by MaxNodeScore,
			// therefore we need to scale the score returned by extenders to the score range used by the scheduler.
			result[i].Score += combinedScores[result[i].Name] * (framework.MaxNodeScore / extenderv1.MaxExtenderPriority)
		}
	}

	if klog.V(10).Enabled() {
		for i := range result {
			klog.Infof("Host %s => Score %d", result[i].Name, result[i].Score)
		}
	}
	return result, nil
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
		if len(victims.Pods) == 0 {
			// We found a node that doesn't need any preemption. Return it!
			// This should happen rarely when one or more pods are terminated between
			// the time that scheduler tries to schedule the pod and the time that
			// preemption logic tries to find nodes for preemption.
			return node
		}
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
		highestPodPriority := podutil.GetPodPriority(victims.Pods[0])
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
			sumPriorities += int64(podutil.GetPodPriority(pod)) + int64(math.MaxInt32+1)
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
		klog.Errorf("earliestStartTime is nil for node %s. Should not reach here.", minNodes2[0])
		return minNodes2[0]
	}
	nodeToReturn := minNodes2[0]
	for i := 1; i < lenNodes2; i++ {
		node := minNodes2[i]
		// Get earliest start time of all pods on the current node.
		earliestStartTimeOnNode := util.GetEarliestPodStartTime(nodesToVictims[node])
		if earliestStartTimeOnNode == nil {
			klog.Errorf("earliestStartTime is nil for node %s. Should not reach here.", node)
			continue
		}
		if earliestStartTimeOnNode.After(latestStartTime.Time) {
			latestStartTime = earliestStartTimeOnNode
			nodeToReturn = node
		}
	}

	return nodeToReturn
}

// selectNodesForPreemption finds all the nodes with possible victims for
// preemption in parallel.
func selectNodesForPreemption(
	ctx context.Context,
	pr framework.PluginsRunner,
	nominator framework.PodNominator,
	state *framework.CycleState,
	pod *v1.Pod,
	potentialNodes []*framework.NodeInfo,
	pdbs []*policy.PodDisruptionBudget,
) (map[string]*extenderv1.Victims, error) {
	nodeNameToVictims := map[string]*extenderv1.Victims{}
	var resultLock sync.Mutex

	checkNode := func(i int) {
		nodeInfoCopy := potentialNodes[i].Clone()
		stateCopy := state.Clone()
		pods, numPDBViolations, fits := selectVictimsOnNode(ctx, pr, nominator, stateCopy, pod, nodeInfoCopy, pdbs)
		if fits {
			resultLock.Lock()
			victims := extenderv1.Victims{
				Pods:             pods,
				NumPDBViolations: int64(numPDBViolations),
			}
			nodeNameToVictims[potentialNodes[i].Node().Name] = &victims
			resultLock.Unlock()
		}
	}
	parallelize.Until(ctx, len(potentialNodes), checkNode)
	return nodeNameToVictims, nil
}

// filterPodsWithPDBViolation groups the given "pods" into two groups of "violatingPods"
// and "nonViolatingPods" based on whether their PDBs will be violated if they are
// preempted.
// This function is stable and does not change the order of received pods. So, if it
// receives a sorted list, grouping will preserve the order of the input list.
func filterPodsWithPDBViolation(pods []*v1.Pod, pdbs []*policy.PodDisruptionBudget) (violatingPods, nonViolatingPods []*v1.Pod) {
	pdbsAllowed := make([]int32, len(pdbs))
	for i, pdb := range pdbs {
		pdbsAllowed[i] = pdb.Status.DisruptionsAllowed
	}

	for _, obj := range pods {
		pod := obj
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
				// We have found a matching PDB.
				if pdbsAllowed[i] <= 0 {
					pdbForPodIsViolated = true
					break
				} else {
					pdbsAllowed[i]--
				}
			}
		}
		if pdbForPodIsViolated {
			violatingPods = append(violatingPods, pod)
		} else {
			nonViolatingPods = append(nonViolatingPods, pod)
		}
	}
	return violatingPods, nonViolatingPods
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
	pr framework.PluginsRunner,
	nominator framework.PodNominator,
	state *framework.CycleState,
	pod *v1.Pod,
	nodeInfo *framework.NodeInfo,
	pdbs []*policy.PodDisruptionBudget,
) ([]*v1.Pod, int, bool) {
	var potentialVictims []*v1.Pod

	removePod := func(rp *v1.Pod) error {
		if err := nodeInfo.RemovePod(rp); err != nil {
			return err
		}
		status := pr.RunPreFilterExtensionRemovePod(ctx, state, pod, rp, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	addPod := func(ap *v1.Pod) error {
		nodeInfo.AddPod(ap)
		status := pr.RunPreFilterExtensionAddPod(ctx, state, pod, ap, nodeInfo)
		if !status.IsSuccess() {
			return status.AsError()
		}
		return nil
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := podutil.GetPodPriority(pod)
	for _, p := range nodeInfo.Pods {
		if podutil.GetPodPriority(p.Pod) < podPriority {
			potentialVictims = append(potentialVictims, p.Pod)
			if err := removePod(p.Pod); err != nil {
				return nil, 0, false
			}
		}
	}
	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption. The only
	// condition that we could check is if the "pod" is failing to schedule due to
	// inter-pod affinity to one or more victims, but we have decided not to
	// support this case for performance reasons. Having affinity to lower
	// priority pods is not a recommended configuration anyway.
	if fits, _, err := podPassesFiltersOnNode(ctx, pr, nominator, state, pod, nodeInfo); !fits {
		if err != nil {
			klog.Warningf("Encountered error while selecting victims on node %v: %v", nodeInfo.Node().Name, err)
		}

		return nil, 0, false
	}
	var victims []*v1.Pod
	numViolatingVictim := 0
	sort.Slice(potentialVictims, func(i, j int) bool { return util.MoreImportantPod(potentialVictims[i], potentialVictims[j]) })
	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest priority victims.
	violatingVictims, nonViolatingVictims := filterPodsWithPDBViolation(potentialVictims, pdbs)
	reprievePod := func(p *v1.Pod) (bool, error) {
		if err := addPod(p); err != nil {
			return false, err
		}
		fits, _, _ := podPassesFiltersOnNode(ctx, pr, nominator, state, pod, nodeInfo)
		if !fits {
			if err := removePod(p); err != nil {
				return false, err
			}
			victims = append(victims, p)
			klog.V(5).Infof("Pod %v/%v is a potential preemption victim on node %v.", p.Namespace, p.Name, nodeInfo.Node().Name)
		}
		return fits, nil
	}
	for _, p := range violatingVictims {
		if fits, err := reprievePod(p); err != nil {
			klog.Warningf("Failed to reprieve pod %q: %v", p.Name, err)
			return nil, 0, false
		} else if !fits {
			numViolatingVictim++
		}
	}
	// Now we try to reprieve non-violating victims.
	for _, p := range nonViolatingVictims {
		if _, err := reprievePod(p); err != nil {
			klog.Warningf("Failed to reprieve pod %q: %v", p.Name, err)
			return nil, 0, false
		}
	}
	return victims, numViolatingVictim, true
}

// nodesWherePreemptionMightHelp returns a list of nodes with failed predicates
// that may be satisfied by removing pods from the node.
func nodesWherePreemptionMightHelp(nodes []*framework.NodeInfo, fitErr *FitError) []*framework.NodeInfo {
	var potentialNodes []*framework.NodeInfo
	for _, node := range nodes {
		name := node.Node().Name
		// We reply on the status by each plugin - 'Unschedulable' or 'UnschedulableAndUnresolvable'
		// to determine whether preemption may help or not on the node.
		if fitErr.FilteredNodesStatuses[name].Code() == framework.UnschedulableAndUnresolvable {
			continue
		}
		klog.V(3).Infof("Node %v is a potential node for preemption.", name)
		potentialNodes = append(potentialNodes, node)
	}
	return potentialNodes
}

// podEligibleToPreemptOthers determines whether this pod should be considered
// for preempting other pods or not. If this pod has already preempted other
// pods and those are in their graceful termination period, it shouldn't be
// considered for preemption.
// We look at the node that is nominated for this pod and as long as there are
// terminating pods on the node, we don't consider this for preempting more pods.
func podEligibleToPreemptOthers(pod *v1.Pod, nodeInfos framework.NodeInfoLister) bool {
	if pod.Spec.PreemptionPolicy != nil && *pod.Spec.PreemptionPolicy == v1.PreemptNever {
		klog.V(5).Infof("Pod %v/%v is not eligible for preemption because it has a preemptionPolicy of %v", pod.Namespace, pod.Name, v1.PreemptNever)
		return false
	}
	nomNodeName := pod.Status.NominatedNodeName
	if len(nomNodeName) > 0 {
		if nodeInfo, _ := nodeInfos.Get(nomNodeName); nodeInfo != nil {
			podPriority := podutil.GetPodPriority(pod)
			for _, p := range nodeInfo.Pods {
				if p.Pod.DeletionTimestamp != nil && podutil.GetPodPriority(p.Pod) < podPriority {
					// There is a terminating pod on the nominated node.
					return false
				}
			}
		}
	}
	return true
}

// podPassesBasicChecks makes sanity checks on the pod if it can be scheduled.
func podPassesBasicChecks(pod *v1.Pod, pvcLister corelisters.PersistentVolumeClaimLister) error {
	// Check PVCs used by the pod
	namespace := pod.Namespace
	manifest := &(pod.Spec)
	for i := range manifest.Volumes {
		volume := &manifest.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			// Volume is not a PVC, ignore
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		pvc, err := pvcLister.PersistentVolumeClaims(namespace).Get(pvcName)
		if err != nil {
			// The error has already enough context ("persistentvolumeclaim "myclaim" not found")
			return err
		}

		if pvc.DeletionTimestamp != nil {
			return fmt.Errorf("persistentvolumeclaim %q is being deleted", pvc.Name)
		}
	}

	return nil
}

// NewGenericScheduler creates a genericScheduler object.
func NewGenericScheduler(
	cache internalcache.Cache,
	podNominator framework.PodNominator,
	nodeInfoSnapshot *internalcache.Snapshot,
	extenders []framework.Extender,
	pvcLister corelisters.PersistentVolumeClaimLister,
	pdbLister policylisters.PodDisruptionBudgetLister,
	disablePreemption bool,
	percentageOfNodesToScore int32) ScheduleAlgorithm {
	return &genericScheduler{
		cache:                    cache,
		podNominator:             podNominator,
		extenders:                extenders,
		nodeInfoSnapshot:         nodeInfoSnapshot,
		pvcLister:                pvcLister,
		pdbLister:                pdbLister,
		disablePreemption:        disablePreemption,
		percentageOfNodesToScore: percentageOfNodesToScore,
	}
}
