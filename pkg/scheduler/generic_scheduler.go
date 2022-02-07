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

package scheduler

import (
	"container/heap"
	"context"
	"fmt"
	"math/rand"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	extenderv1 "k8s.io/kube-scheduler/extender/v1"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
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
	// numberOfNodeScoresToIncludeInScheduleResult is the number of node scores
	// to be included in ScheduleResult.
	numberOfNodeScoresToIncludeInScheduleResult = 3
)

// ErrNoNodesAvailable is used to describe the error that no nodes available to schedule pods.
var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

// ScheduleAlgorithm is an interface implemented by things that know how to schedule pods
// onto machines.
// TODO: Rename this type.
type ScheduleAlgorithm interface {
	Schedule(context.Context, []framework.Extender, framework.Framework, *framework.CycleState, *v1.Pod) (scheduleResult ScheduleResult, err error)
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
	// Only the top numberOfNodeScoresToIncludeInScheduleResult scores will be recorded.
	NodeScoreList NodeScoreResultList
}

type NodeScoreResultList []NodeScoreResult

// String implements fmt.Stringer interface.
// It will be like this:
// [NodeName: node3 FinalScore: 675 Detailed Scores: [PodTopologySpread: 200 InterPodAffinity: 0 NodeResourcesBalancedAllocation: 100 ImageLocality: 0 TaintToleration: 300 NodeAffinity: 0 NodeResourcesFit: 75 VolumeBinding: 0]] [NodeName: node2 FinalScore: 425 Detailed Scores: [NodeAffinity: 0 NodeResourcesFit: 25 VolumeBinding: 0 PodTopologySpread: 0 InterPodAffinity: 0 NodeResourcesBalancedAllocation: 100 ImageLocality: 0 TaintToleration: 300]]]
func (n NodeScoreResultList) String() string {
	var str string
	for i := range n {
		str += fmt.Sprintf("[NodeName: %s FinalScore: %d Detailed Scores: [", n[i].Name, n[i].FinalScore)
		for k, v := range n[i].Scores {
			str += fmt.Sprintf("%s: %d ", k, v)
		}
		str = strings.TrimSpace(str)
		str += fmt.Sprintf("]] ")
	}
	str = strings.TrimSpace(str)
	str += fmt.Sprintf("]")
	return str
}

// NodeScoreResult has the score result for the Node.
type NodeScoreResult struct {
	// Name indicates the name of the Node.
	Name string
	// FinalScore indicates the final score for the Node.
	FinalScore int64
	// Scores indicates the scores which each plugin returned for the Node.
	// plugin name → score
	Scores map[string]int64
}

func NewNodeScoreResult(nodeName string, finalScore int64) NodeScoreResult {
	return NodeScoreResult{
		Name:       nodeName,
		FinalScore: finalScore,
		Scores:     map[string]int64{},
	}
}

func (result *NodeScoreResult) AddScores(scoreMap framework.PluginToNodeScores) {
	if result.Scores == nil {
		result.Scores = map[string]int64{}
	}

	for plugin, nodeScoreList := range scoreMap {
		for _, nodeScore := range nodeScoreList {
			if result.Name == nodeScore.Name {
				result.Scores[plugin] = nodeScore.Score
			}
		}
	}
}

type genericScheduler struct {
	cache                    internalcache.Cache
	nodeInfoSnapshot         *internalcache.Snapshot
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
func (g *genericScheduler) Schedule(ctx context.Context, extenders []framework.Extender, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) (result ScheduleResult, err error) {
	trace := utiltrace.New("Scheduling", utiltrace.Field{Key: "namespace", Value: pod.Namespace}, utiltrace.Field{Key: "name", Value: pod.Name})
	defer trace.LogIfLong(100 * time.Millisecond)

	if err := g.snapshot(); err != nil {
		return result, err
	}
	trace.Step("Snapshotting scheduler cache and node infos done")

	if g.nodeInfoSnapshot.NumNodes() == 0 {
		return result, ErrNoNodesAvailable
	}

	feasibleNodes, diagnosis, err := g.findNodesThatFitPod(ctx, extenders, fwk, state, pod)
	if err != nil {
		return result, err
	}
	trace.Step("Computing predicates done")

	if len(feasibleNodes) == 0 {
		return result, &framework.FitError{
			Pod:         pod,
			NumAllNodes: g.nodeInfoSnapshot.NumNodes(),
			Diagnosis:   diagnosis,
		}
	}

	// When only one node after predicate, just use it.
	if len(feasibleNodes) == 1 {
		return ScheduleResult{
			SuggestedHost:  feasibleNodes[0].Name,
			EvaluatedNodes: 1 + len(diagnosis.NodeToStatusMap),
			FeasibleNodes:  1,
		}, nil
	}

	priorityList, scoreMap, err := prioritizeNodes(ctx, extenders, fwk, state, pod, feasibleNodes)
	if err != nil {
		return result, err
	}

	host, nodeScores, err := g.selectHost(priorityList, numberOfNodeScoresToIncludeInScheduleResult)
	trace.Step("Prioritizing done")

	for _, score := range nodeScores {
		score.AddScores(scoreMap)
	}

	return ScheduleResult{
		SuggestedHost:  host,
		EvaluatedNodes: len(feasibleNodes) + len(diagnosis.NodeToStatusMap),
		FeasibleNodes:  len(feasibleNodes),
		NodeScoreList:  nodeScores,
	}, err
}

// selectHost takes a prioritized list of nodes and then picks one
// in a reservoir sampling manner from the nodes that had the highest score.
// It also returns the top {numberOfNodeScoresToReturn} scores which sorted in descending order.
// The top of the list is always the score of the selected host.
func (g *genericScheduler) selectHost(nodeScoreList []framework.NodeScore, numberOfNodeScoresToReturn int) (string, []NodeScoreResult, error) {
	if len(nodeScoreList) == 0 {
		return "", nil, fmt.Errorf("empty priorityList")
	}

	var h nodeScoreHeap = nodeScoreList
	heap.Init(&h)
	cntOfMaxScore := 1
	// The top of the heap is the NodeScoreResult with the highest score.
	selected := heap.Pop(&h).(framework.NodeScore)
	sortedNodeScoreList := make([]NodeScoreResult, 0, numberOfNodeScoresToReturn)

	// For efficient computation, the following for-loop does two things:
	// - There may be more than one NodeScoreResult with the highest score,
	//   and all Nodes should be taken into account for a reservoir sampling.
	// - Take {numberOfNodeScoresToReturn - 1} NodeScoreResult from the top of the heap and put them in sortedNodeScoreList.
	for ns := heap.Pop(&h).(framework.NodeScore);
	// All Nodes with the highest scores must be checked for a reservoir sampling,
	// even if the next condition for the length of sortedNodeScoreList is violated.
	ns.Score == selected.Score ||
		// selectedNode will be inserted into sortedNodeScoreList later
		// to ensure to make selectedNode the top of sortedNodeScoreList.
		len(sortedNodeScoreList) < numberOfNodeScoresToReturn-1; ns = heap.Pop(&h).(framework.NodeScore) {
		if ns.Score == selected.Score {
			cntOfMaxScore++
			if rand.Intn(cntOfMaxScore) == 0 {
				// Replace the candidate with probability of 1/cntOfMaxScore
				previousSelected := selected
				selected = ns
				ns = previousSelected
			}
		}

		if len(sortedNodeScoreList)+1 <= numberOfNodeScoresToReturn {
			sortedNodeScoreList = append(sortedNodeScoreList, NewNodeScoreResult(ns.Name, ns.Score))
		}
		if h.Len() == 0 {
			break
		}
	}

	sortedNodeScoreList = append([]NodeScoreResult{
		NewNodeScoreResult(selected.Name, selected.Score),
	}, sortedNodeScoreList...)

	return selected.Name, sortedNodeScoreList, nil
}

// An nodeScoreHeap is a max-heap of framework.NodeScore.
type nodeScoreHeap []framework.NodeScore

// nodeScoreHeap implements heap.Interface.
var _ heap.Interface = &nodeScoreHeap{}

func (h nodeScoreHeap) Len() int           { return len(h) }
func (h nodeScoreHeap) Less(i, j int) bool { return h[i].Score > h[j].Score }
func (h nodeScoreHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *nodeScoreHeap) Push(x interface{}) {
	*h = append(*h, x.(framework.NodeScore))
}

func (h *nodeScoreHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
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

func (g *genericScheduler) evaluateNominatedNode(ctx context.Context, extenders []framework.Extender, pod *v1.Pod, fwk framework.Framework, state *framework.CycleState, diagnosis framework.Diagnosis) ([]*v1.Node, error) {
	nnn := pod.Status.NominatedNodeName
	nodeInfo, err := g.nodeInfoSnapshot.Get(nnn)
	if err != nil {
		return nil, err
	}
	node := []*framework.NodeInfo{nodeInfo}
	feasibleNodes, err := g.findNodesThatPassFilters(ctx, fwk, state, pod, diagnosis, node)
	if err != nil {
		return nil, err
	}

	feasibleNodes, err = findNodesThatPassExtenders(extenders, pod, feasibleNodes, diagnosis.NodeToStatusMap)
	if err != nil {
		return nil, err
	}

	return feasibleNodes, nil
}

// Filters the nodes to find the ones that fit the pod based on the framework
// filter plugins and filter extenders.
func (g *genericScheduler) findNodesThatFitPod(ctx context.Context, extenders []framework.Extender, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) ([]*v1.Node, framework.Diagnosis, error) {
	diagnosis := framework.Diagnosis{
		NodeToStatusMap:      make(framework.NodeToStatusMap),
		UnschedulablePlugins: sets.NewString(),
	}

	// Run "prefilter" plugins.
	s := fwk.RunPreFilterPlugins(ctx, state, pod)
	allNodes, err := g.nodeInfoSnapshot.NodeInfos().List()
	if err != nil {
		return nil, diagnosis, err
	}
	if !s.IsSuccess() {
		if !s.IsUnschedulable() {
			return nil, diagnosis, s.AsError()
		}
		// All nodes will have the same status. Some non trivial refactoring is
		// needed to avoid this copy.
		for _, n := range allNodes {
			diagnosis.NodeToStatusMap[n.Node().Name] = s
		}
		// Status satisfying IsUnschedulable() gets injected into diagnosis.UnschedulablePlugins.
		diagnosis.UnschedulablePlugins.Insert(s.FailedPlugin())
		return nil, diagnosis, nil
	}

	// "NominatedNodeName" can potentially be set in a previous scheduling cycle as a result of preemption.
	// This node is likely the only candidate that will fit the pod, and hence we try it first before iterating over all nodes.
	if len(pod.Status.NominatedNodeName) > 0 {
		feasibleNodes, err := g.evaluateNominatedNode(ctx, extenders, pod, fwk, state, diagnosis)
		if err != nil {
			klog.ErrorS(err, "Evaluation failed on nominated node", "pod", klog.KObj(pod), "node", pod.Status.NominatedNodeName)
		}
		// Nominated node passes all the filters, scheduler is good to assign this node to the pod.
		if len(feasibleNodes) != 0 {
			return feasibleNodes, diagnosis, nil
		}
	}
	feasibleNodes, err := g.findNodesThatPassFilters(ctx, fwk, state, pod, diagnosis, allNodes)
	if err != nil {
		return nil, diagnosis, err
	}

	feasibleNodes, err = findNodesThatPassExtenders(extenders, pod, feasibleNodes, diagnosis.NodeToStatusMap)
	if err != nil {
		return nil, diagnosis, err
	}
	return feasibleNodes, diagnosis, nil
}

// findNodesThatPassFilters finds the nodes that fit the filter plugins.
func (g *genericScheduler) findNodesThatPassFilters(
	ctx context.Context,
	fwk framework.Framework,
	state *framework.CycleState,
	pod *v1.Pod,
	diagnosis framework.Diagnosis,
	nodes []*framework.NodeInfo) ([]*v1.Node, error) {
	numNodesToFind := g.numFeasibleNodesToFind(int32(len(nodes)))

	// Create feasible list with enough space to avoid growing it
	// and allow assigning.
	feasibleNodes := make([]*v1.Node, numNodesToFind)

	if !fwk.HasFilterPlugins() {
		length := len(nodes)
		for i := range feasibleNodes {
			feasibleNodes[i] = nodes[(g.nextStartNodeIndex+i)%length].Node()
		}
		g.nextStartNodeIndex = (g.nextStartNodeIndex + len(feasibleNodes)) % length
		return feasibleNodes, nil
	}

	errCh := parallelize.NewErrorChannel()
	var statusesLock sync.Mutex
	var feasibleNodesLen int32
	ctx, cancel := context.WithCancel(ctx)
	checkNode := func(i int) {
		// We check the nodes starting from where we left off in the previous scheduling cycle,
		// this is to make sure all nodes have the same chance of being examined across pods.
		nodeInfo := nodes[(g.nextStartNodeIndex+i)%len(nodes)]
		status := fwk.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo)
		if status.Code() == framework.Error {
			errCh.SendErrorWithCancel(status.AsError(), cancel)
			return
		}
		if status.IsSuccess() {
			length := atomic.AddInt32(&feasibleNodesLen, 1)
			if length > numNodesToFind {
				cancel()
				atomic.AddInt32(&feasibleNodesLen, -1)
			} else {
				feasibleNodes[length-1] = nodeInfo.Node()
			}
		} else {
			statusesLock.Lock()
			diagnosis.NodeToStatusMap[nodeInfo.Node().Name] = status
			diagnosis.UnschedulablePlugins.Insert(status.FailedPlugin())
			statusesLock.Unlock()
		}
	}

	beginCheckNode := time.Now()
	statusCode := framework.Success
	defer func() {
		// We record Filter extension point latency here instead of in framework.go because framework.RunFilterPlugins
		// function is called for each node, whereas we want to have an overall latency for all nodes per scheduling cycle.
		// Note that this latency also includes latency for `addNominatedPods`, which calls framework.RunPreFilterAddPod.
		metrics.FrameworkExtensionPointDuration.WithLabelValues(runtime.Filter, statusCode.String(), fwk.ProfileName()).Observe(metrics.SinceInSeconds(beginCheckNode))
	}()

	// Stops searching for more nodes once the configured number of feasible nodes
	// are found.
	fwk.Parallelizer().Until(ctx, len(nodes), checkNode)
	processedNodes := int(feasibleNodesLen) + len(diagnosis.NodeToStatusMap)
	g.nextStartNodeIndex = (g.nextStartNodeIndex + processedNodes) % len(nodes)

	feasibleNodes = feasibleNodes[:feasibleNodesLen]
	if err := errCh.ReceiveError(); err != nil {
		statusCode = framework.Error
		return nil, err
	}
	return feasibleNodes, nil
}

func findNodesThatPassExtenders(extenders []framework.Extender, pod *v1.Pod, feasibleNodes []*v1.Node, statuses framework.NodeToStatusMap) ([]*v1.Node, error) {
	// Extenders are called sequentially.
	// Nodes in original feasibleNodes can be excluded in one extender, and pass on to the next
	// extender in a decreasing manner.
	for _, extender := range extenders {
		if len(feasibleNodes) == 0 {
			break
		}
		if !extender.IsInterested(pod) {
			continue
		}

		// Status of failed nodes in failedAndUnresolvableMap will be added or overwritten in <statuses>,
		// so that the scheduler framework can respect the UnschedulableAndUnresolvable status for
		// particular nodes, and this may eventually improve preemption efficiency.
		// Note: users are recommended to configure the extenders that may return UnschedulableAndUnresolvable
		// status ahead of others.
		feasibleList, failedMap, failedAndUnresolvableMap, err := extender.Filter(pod, feasibleNodes)
		if err != nil {
			if extender.IsIgnorable() {
				klog.InfoS("Skipping extender as it returned error and has ignorable flag set", "extender", extender, "err", err)
				continue
			}
			return nil, err
		}

		for failedNodeName, failedMsg := range failedAndUnresolvableMap {
			var aggregatedReasons []string
			if _, found := statuses[failedNodeName]; found {
				aggregatedReasons = statuses[failedNodeName].Reasons()
			}
			aggregatedReasons = append(aggregatedReasons, failedMsg)
			statuses[failedNodeName] = framework.NewStatus(framework.UnschedulableAndUnresolvable, aggregatedReasons...)
		}

		for failedNodeName, failedMsg := range failedMap {
			if _, found := failedAndUnresolvableMap[failedNodeName]; found {
				// failedAndUnresolvableMap takes precedence over failedMap
				// note that this only happens if the extender returns the node in both maps
				continue
			}
			if _, found := statuses[failedNodeName]; !found {
				statuses[failedNodeName] = framework.NewStatus(framework.Unschedulable, failedMsg)
			} else {
				statuses[failedNodeName].AppendReason(failedMsg)
			}
		}

		feasibleNodes = feasibleList
	}
	return feasibleNodes, nil
}

// prioritizeNodes prioritizes the nodes by running the score plugins,
// which return a score for each node from the call to RunScorePlugins().
// The scores from each plugin are added together to make the score for that node, then
// any extenders are run as well.
// All scores are finally combined (added) to get the total weighted scores of all nodes.
// And it also returns PluginToNodeScores which contains the result from each plugin and extender.
func prioritizeNodes(
	ctx context.Context,
	extenders []framework.Extender,
	fwk framework.Framework,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) (framework.NodeScoreList, framework.PluginToNodeScores, error) {
	// If no priority configs are provided, then all nodes will have a score of one.
	// This is required to generate the priority list in the required format
	if len(extenders) == 0 && !fwk.HasScorePlugins() {
		result := make(framework.NodeScoreList, 0, len(nodes))
		for i := range nodes {
			result = append(result, framework.NodeScore{
				Name:  nodes[i].Name,
				Score: 1,
			})
		}
		return result, nil, nil
	}

	// Run PreScore plugins.
	preScoreStatus := fwk.RunPreScorePlugins(ctx, state, pod, nodes)
	if !preScoreStatus.IsSuccess() {
		return nil, nil, preScoreStatus.AsError()
	}

	// Run the Score plugins.
	scoresMap, scoreStatus := fwk.RunScorePlugins(ctx, state, pod, nodes)
	if !scoreStatus.IsSuccess() {
		return nil, nil, scoreStatus.AsError()
	}

	// Additional details logged at level 10 if enabled.
	klogV := klog.V(10)
	if klogV.Enabled() {
		for plugin, nodeScoreList := range scoresMap {
			for _, nodeScore := range nodeScoreList {
				klogV.InfoS("Plugin scored node for pod", "pod", klog.KObj(pod), "plugin", plugin, "node", nodeScore.Name, "score", nodeScore.Score)
			}
		}
	}

	if len(extenders) != 0 && nodes != nil {
		// extenderPrefix is used to avoid name collision
		// if there is an extender with the same name as the plugin's name.
		const extenderPrefix = "extenders/"
		var mu sync.Mutex
		var wg sync.WaitGroup
		for _, ex := range extenders {
			scoresMap[extenderPrefix+ex.Name()] = make(framework.NodeScoreList, len(nodes))
		}

		for i := range extenders {
			if !extenders[i].IsInterested(pod) {
				continue
			}
			wg.Add(1)
			go func(extIndex int) {
				metrics.SchedulerGoroutines.WithLabelValues(metrics.PrioritizingExtender).Inc()
				defer func() {
					metrics.SchedulerGoroutines.WithLabelValues(metrics.PrioritizingExtender).Dec()
					wg.Done()
				}()
				nodeScoreList := scoresMap[extenderPrefix+extenders[extIndex].Name()]
				prioritizedList, weight, err := extenders[extIndex].Prioritize(pod, nodes)
				if err != nil {
					// Prioritization errors from extender can be ignored, let k8s/other extenders determine the priorities
					return
				}
				for i := range *prioritizedList {
					score := (*prioritizedList)[i].Score
					if klogV.Enabled() {
						klogV.InfoS("Extender scored node for pod", "pod", klog.KObj(pod), "extender", extenders[extIndex].Name(), "node", nodeScoreList[i].Name, "score", nodeScoreList[i].Score)
					}
					mu.Lock()
					nodeScoreList[i].Name = (*prioritizedList)[i].Host
					// MaxExtenderPriority may diverge from the max priority used in the scheduler and defined by MaxNodeScore,
					// therefore we need to scale the score returned by extenders to the score range used by the scheduler.
					nodeScoreList[i].Score = score * weight * (framework.MaxNodeScore / extenderv1.MaxExtenderPriority)
					mu.Unlock()
				}
			}(i)
		}
		// wait for all go routines to finish
		wg.Wait()
	}

	// Summarize all scores.
	result := make(framework.NodeScoreList, 0, len(nodes))
	indexMap := map[string]int{}
	for i := range nodes {
		result = append(result, framework.NodeScore{Name: nodes[i].Name, Score: 0})
		indexMap[nodes[i].Name] = len(result) - 1
	}

	for j := range scoresMap {
		for _, s := range scoresMap[j] {
			result[indexMap[s.Name]].Score += s.Score
		}
	}

	if klogV.Enabled() {
		for i := range result {
			klogV.InfoS("Calculated node's final score for pod", "pod", klog.KObj(pod), "node", result[i].Name, "score", result[i].Score)
		}
	}
	return result, scoresMap, nil
}

// NewGenericScheduler creates a genericScheduler object.
func NewGenericScheduler(
	cache internalcache.Cache,
	nodeInfoSnapshot *internalcache.Snapshot,
	percentageOfNodesToScore int32) ScheduleAlgorithm {
	return &genericScheduler{
		cache:                    cache,
		nodeInfoSnapshot:         nodeInfoSnapshot,
		percentageOfNodesToScore: percentageOfNodesToScore,
	}
}
