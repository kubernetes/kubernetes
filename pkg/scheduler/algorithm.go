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

package scheduler

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	v1 "k8s.io/api/core/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	utiltrace "k8s.io/utils/trace"
)

// SchedulingAlgorithm encapsulates the state and logic required for the in-memory
// part of a scheduling attempt. It provides methods to select a node for a pod
// (filter, score, extenders) and assume the resulting placement in the scheduler's
// cache and snapshot. It does not interact with the scheduling queue, run binding,
// or handle scheduling failures - those responsibilities remain with the Scheduler.
//
// A SchedulingAlgorithm instance is stateful: it advances nextStartNodeIndex
// between calls and mutates the shared node snapshot when assuming and
// forgetting pods. It is not safe for concurrent use, so scheduling cycles
// sharing an instance have to be serialized by the caller.
type SchedulingAlgorithm struct {
	// nodeInfoSnapshot is the point-in-time view of cluster nodes used during
	// filtering and scoring. During pod group scheduling, tentative placements
	// mutate this snapshot directly so subsequent pods in the group observe them
	// without modifying the live cache before placement is found for the whole pod group.
	nodeInfoSnapshot *internalcache.Snapshot

	// cache is the scheduler's live cache. Successful placements are assumed here
	// immediately so subsequent scheduling cycles account for allocated resources
	// without waiting for asynchronous binding to complete against the apiserver.
	cache internalcache.Cache

	// percentageOfNodesToScore is the default percentage of total cluster nodes to
	// find during filtering before proceeding to scoring. It acts as an algorithm-level
	// fallback when a scheduling profile does not configure its own threshold,
	// preventing exhaustive scans in large clusters.
	percentageOfNodesToScore int32

	// cycleProvider reports the current scheduling cycle counter. It is used by
	// opportunistic batching to key and reuse scoring state and node hints strictly
	// across consecutive cycles.
	cycleProvider func() int64

	// nextStartNodeIndex is the starting index in the node list for the next filter
	// pass. Advancing this across scheduling cycles distributes evaluations uniformly
	// across all nodes when percentageOfNodesToScore halts the search early.
	nextStartNodeIndex int
}

type AlgorithmOption func(*SchedulingAlgorithm)

// WithAlgorithmPercentageOfNodesToScore sets how many nodes the algorithm should evaluate
// while filtering, as a percentage of all nodes: once the number of evaluated nodes reaches the set value and
// at least N feasible nodes are found (where N is calculated by numFeasibleNodesToFind),
// the search stops, and those are the nodes that go on to scoring.
// If percentageOfNodesToScore is set on scheduler profile level, the value in profile will be used
// instead of the global percentageOfNodesToScore value.
// If percentageOfNodesToScore is set to 0, scheduler will use an adaptive algorithm to calculate a percentage
// value to use; the larger the cluster, the lower the result value.
// If the limit of nodes to score is reached but the number of feasible nodes found is lower than the value
// from numFeasibleNodesToFind, the filtering of subsequent nodes will continue.
func WithAlgorithmPercentageOfNodesToScore(percentage int32) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.percentageOfNodesToScore = percentage
	}
}

// WithCurrentCycleProvider supplies the function reporting the scheduling cycle the
// algorithm is running in. Only opportunistic batching consumes it: the batch keys
// its cached scoring state by cycle number and reuses that state only across
// consecutive cycles. Without this option the algorithm does not batch at all,
// rather than batching against a synthetic counter that would make unrelated pods
// look consecutive.
func WithCurrentCycleProvider(cycleProvider func() int64) AlgorithmOption {
	return func(a *SchedulingAlgorithm) {
		a.cycleProvider = cycleProvider
	}
}

// NewSchedulingAlgorithm creates a scheduling algorithm operating on the given snapshot and cache.
func NewSchedulingAlgorithm(snapshot *internalcache.Snapshot, cache internalcache.Cache,
	opts ...AlgorithmOption) *SchedulingAlgorithm {
	a := &SchedulingAlgorithm{
		nodeInfoSnapshot:         snapshot,
		cache:                    cache,
		percentageOfNodesToScore: schedulerapi.DefaultPercentageOfNodesToScore,
	}
	for _, opt := range opts {
		opt(a)
	}
	return a
}

// opportunisticBatchingEnabled reports whether opportunistic batching can run. It needs both the
// feature gate and a cycle provider: batch state is keyed by scheduling cycle, so
// without a real counter the batch cannot tell consecutive cycles apart.
func (a *SchedulingAlgorithm) opportunisticBatchingEnabled() bool {
	return a.cycleProvider != nil && utilfeature.DefaultFeatureGate.Enabled(features.OpportunisticBatching)
}

// SchedulePod runs PreFilter, Filter, filter extenders and Score, and returns the
// selected node. It returns a *framework.FitError when no node fits — running
// PostFilter (preemption) and requeueing in response is the caller's job:
// preemption mutates cluster state and stays with Scheduler.
func (a *SchedulingAlgorithm) SchedulePod(ctx context.Context, schedFramework framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) (result ScheduleResult, err error) {
	pod := podInfo.Pod
	trace := utiltrace.New("Scheduling", utiltrace.Field{Key: "namespace", Value: pod.Namespace}, utiltrace.Field{Key: "name", Value: pod.Name})
	defer trace.LogIfLong(100 * time.Millisecond)

	if a.nodeInfoSnapshot.NumNodesInPlacement() == 0 {
		return result, ErrNoNodesAvailable
	}

	feasibleNodes, diagnosis, nodeHint, err := a.findNodesThatFitPod(ctx, schedFramework, state, podInfo)
	if err != nil {
		return result, err
	}
	trace.Step("Computing predicates done")

	if len(feasibleNodes) == 0 {
		return result, &framework.FitError{
			Pod:         pod,
			NumAllNodes: a.nodeInfoSnapshot.NumNodesInPlacement(),
			Diagnosis:   diagnosis,
		}
	}

	// When only one node after predicate, just use it.
	if len(feasibleNodes) == 1 {
		node := feasibleNodes[0].Node().Name
		if a.opportunisticBatchingEnabled() {
			schedFramework.StoreScheduleResults(ctx, podInfo.PodSignature, nodeHint, node, nil, a.cycleProvider())
		}
		return ScheduleResult{
			SuggestedHost:  node,
			EvaluatedNodes: 1 + diagnosis.NodeToStatus.Len(),
			FeasibleNodes:  1,
		}, nil
	}

	priorityList, err := prioritizeNodes(ctx, schedFramework, state, pod, feasibleNodes)
	if err != nil {
		return result, err
	}

	sortedPrioritizedNodes := framework.NewSortedScoredNodes(priorityList)
	node := sortedPrioritizedNodes.Pop().Name
	trace.Step("Prioritizing done")

	if a.opportunisticBatchingEnabled() {
		schedFramework.StoreScheduleResults(ctx, podInfo.PodSignature, nodeHint, node, sortedPrioritizedNodes, a.cycleProvider())
	}

	return ScheduleResult{
		SuggestedHost:  node,
		EvaluatedNodes: len(feasibleNodes) + diagnosis.NodeToStatus.Len(),
		FeasibleNodes:  len(feasibleNodes),
	}, err
}

// preFilterOutcome captures the state and intermediate results produced while running
// PreFilter plugins for a pod. It aggregates node candidates and plugin diagnoses so
// downstream candidate selection and filtering can proceed without re-evaluating cluster
// snapshots or dropping diagnostic context needed for preemption.
type preFilterOutcome struct {
	// result is the optional node subset returned by PreFilter plugins. When non-nil
	// and containing a specific node set, getCandidateNodes restricts filter evaluations
	// to only these nodes instead of scanning the full cluster snapshot.
	result *fwk.PreFilterResult

	// allNodes holds the complete list of nodes in the placement snapshot prior to
	// plugin narrowing. The algorithm retains this full set so it can accurately
	// advance nextStartNodeIndex across cycles to prevent node starvation.
	allNodes []fwk.NodeInfo

	// diagnosis accumulates rejection reasons, messages, and plugin failure statuses
	// encountered during PreFilter. It is propagated to the caller and preemption
	// logic if no candidate nodes fit the pod.
	diagnosis *framework.Diagnosis
}

// prefilterNodes runs PreFilter plugins against the current cluster placement snapshot and captures
// the outcome for downstream filtering.
//
// It retains the complete snapshot node list so nextStartNodeIndex can advance uniformly across
// cycles even if PreFilter restricts candidate nodes. When a plugin explicitly rejects the pod,
// the rejection status is broadcast across all absent nodes in Diagnosis.NodeToStatus so preemption
// (PostFilter) has the necessary diagnostic context without triggering a full filter pass.
func (a *SchedulingAlgorithm) prefilterNodes(ctx context.Context, schedFramework framework.Framework, state fwk.CycleState, pod *v1.Pod) (preFilterOutcome, *fwk.Status) {
	logger := klog.FromContext(ctx)
	diagnosis := framework.Diagnosis{
		NodeToStatus: framework.NewDefaultNodeToStatus(),
	}
	prefilterOutcome := preFilterOutcome{
		diagnosis: &diagnosis,
	}
	allNodes, err := a.nodeInfoSnapshot.ListNodesInPlacement()
	prefilterOutcome.allNodes = allNodes
	if err != nil {
		return prefilterOutcome, fwk.AsStatus(err)
	}

	// Run "prefilter" plugins.
	preRes, s, unscheduledPlugins := schedFramework.RunPreFilterPlugins(ctx, state, pod)
	prefilterOutcome.result = preRes
	diagnosis.UnschedulablePlugins = unscheduledPlugins
	if !s.IsSuccess() {
		if s.IsRejected() {
			// All nodes in NodeToStatus will have the same status so that they can be handled in the preemption.
			diagnosis.NodeToStatus.SetAbsentNodesStatus(s)

			// Record the messages from PreFilter in Diagnosis.PreFilterMsg.
			msg := s.Message()
			diagnosis.PreFilterMsg = msg
			logger.V(5).Info("Status after running PreFilter plugins for pod", "pod", klog.KObj(pod), "status", msg)
			diagnosis.AddPluginStatus(s)
		}
		return prefilterOutcome, s
	}

	return prefilterOutcome, nil
}

// getCandidateNodes derives the subset of nodes to evaluate in Filter from the PreFilter outcome.
//
// When PreFilter plugins return a restricted set of node names, each name is validated against
// the placement snapshot to prevent invalid or out-of-placement references from entering parallel
// filter workers. Any cluster nodes omitted by PreFilter are marked UnschedulableAndUnresolvable
// in Diagnosis.NodeToStatus so preemption avoids attempting unresolvable evictions on them.
func (a *SchedulingAlgorithm) getCandidateNodes(preFilterOutcome preFilterOutcome) []fwk.NodeInfo {
	nodes := preFilterOutcome.allNodes
	diagnosis := preFilterOutcome.diagnosis
	preRes := preFilterOutcome.result
	if !preRes.AllNodes() {
		nodes = make([]fwk.NodeInfo, 0, len(preRes.NodeNames))
		for nodeName := range preRes.NodeNames {
			// PreRes may return nodeName(s) which do not exist; we verify
			// node exists in the Snapshot within the selected placement.
			if nodeInfo, err := a.nodeInfoSnapshot.GetNodeInPlacement(nodeName); err == nil {
				nodes = append(nodes, nodeInfo)
			}
		}
		diagnosis.NodeToStatus.SetAbsentNodesStatus(fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("node(s) didn't satisfy plugin(s) %v", sets.List(diagnosis.UnschedulablePlugins))))
	}

	return nodes
}

// filterWithExtenders filters candidate nodes through configured scheduling extenders and updates
// diagnostic accounting for queue requeueing.
//
// Because external extenders lack event-driven requeueing hooks (EnqueueExtensions), any extender
// rejection that reduces the feasible set adds framework.ExtenderName to Diagnosis.UnschedulablePlugins.
// This ensures unschedulable pods are retried on any cluster event rather than waiting for plugin-specific
// triggers.
func (a *SchedulingAlgorithm) filterWithExtenders(ctx context.Context, schedFramework framework.Framework, pod *v1.Pod, feasibleNodes []fwk.NodeInfo, diagnosis *framework.Diagnosis) ([]fwk.NodeInfo, error) {
	feasibleNodesAfterExtender, err := findNodesThatPassExtenders(ctx, schedFramework.Extenders(), pod, feasibleNodes, diagnosis.NodeToStatus)
	if err != nil {
		return nil, err
	}
	if len(feasibleNodesAfterExtender) != len(feasibleNodes) {
		// Extenders filtered out some nodes.
		//
		// Extender doesn't support any kind of requeueing feature like EnqueueExtensions in the scheduling framework.
		// When Extenders reject some Nodes and the pod ends up being unschedulable,
		// we put fwk.ExtenderName to pInfo.UnschedulablePlugins.
		// This Pod will be requeued from unschedulable pod pool to activeQ/backoffQ
		// by any kind of cluster events.
		// https://github.com/kubernetes/kubernetes/issues/122019
		if diagnosis.UnschedulablePlugins == nil {
			diagnosis.UnschedulablePlugins = sets.New[string]()
		}
		diagnosis.UnschedulablePlugins.Insert(framework.ExtenderName)
	}

	return feasibleNodesAfterExtender, nil
}

// findNodesThatFitPod filters the nodes to find the ones that fit the pod based on the framework
// filter plugins and filter extenders.
func (a *SchedulingAlgorithm) findNodesThatFitPod(ctx context.Context, schedFramework framework.Framework, state fwk.CycleState, podInfo *framework.QueuedPodInfo) ([]fwk.NodeInfo, framework.Diagnosis, string, error) {
	pod := podInfo.Pod
	preFilterOut, status := a.prefilterNodes(ctx, schedFramework, state, pod)
	diagnosis := preFilterOut.diagnosis
	if !status.IsSuccess() {
		if status.IsRejected() {
			return nil, *diagnosis, "", nil
		}
		return nil, *diagnosis, "", status.AsError()
	}

	var nodeHint string
	if a.opportunisticBatchingEnabled() {
		// We get the node hint even if we have a nominated name for simplicity, but we could potentially avoid it
		// in this scenario in the future.
		nodeHint = schedFramework.GetNodeHint(ctx, pod, podInfo.PodSignature, state, a.cycleProvider())
	}

	// "NominatedNodeName" can potentially be set in a previous scheduling cycle as a result of preemption.
	// This node is likely the only candidate that will fit the pod, and hence we try it first before iterating over all nodes.
	// We take the same tack for hinted nodes from the batch module.
	if len(pod.Status.NominatedNodeName) > 0 || len(nodeHint) > 0 {
		feasibleNodes, err := a.evaluateNominatedNode(ctx, pod, schedFramework, state, nodeHint, *diagnosis)
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Evaluation failed on nominated node", "pod", klog.KObj(pod), "node", pod.Status.NominatedNodeName)
		}
		// Nominated node passes all the filters, scheduler is good to assign this node to the pod.
		if len(feasibleNodes) != 0 {
			return feasibleNodes, *diagnosis, nodeHint, nil
		}
	}

	nodes := a.getCandidateNodes(preFilterOut)

	// The budget is computed from the candidate list, which PreFilter may have narrowed —
	// not from allNodes. findAll mode takes every candidate and skips the budget entirely.
	numNodesToFind := a.numNodesToFind(schedFramework, int32(len(nodes)))
	feasibleNodes, err := a.findNodesThatPassFilters(ctx, schedFramework, state, pod, diagnosis, nodes, numNodesToFind)
	// always try to update the a.nextStartNodeIndex regardless of whether an error has occurred
	// this is helpful to make sure that all the nodes have a chance to be searched
	processedNodes := len(feasibleNodes) + diagnosis.NodeToStatus.Len()
	if len(preFilterOut.allNodes) > 0 {
		a.nextStartNodeIndex = (a.nextStartNodeIndex + processedNodes) % len(preFilterOut.allNodes)
	}
	if err != nil {
		return nil, *diagnosis, nodeHint, err
	}

	feasibleNodes, err = a.filterWithExtenders(ctx, schedFramework, pod, feasibleNodes, diagnosis)
	if err != nil {
		return nil, *diagnosis, nodeHint, err
	}
	return feasibleNodes, *diagnosis, nodeHint, nil
}

func (a *SchedulingAlgorithm) evaluateNominatedNode(ctx context.Context, pod *v1.Pod, schedFramework framework.Framework, state fwk.CycleState, nodeHint string, diagnosis framework.Diagnosis) ([]fwk.NodeInfo, error) {
	// In the future we could potentially use the hint if the nominated node failed.
	// https://github.com/kubernetes/kubernetes/issues/135163
	nnn := pod.Status.NominatedNodeName
	if len(nnn) == 0 {
		nnn = nodeHint
	}

	nodeInfo, err := a.nodeInfoSnapshot.GetNodeInPlacement(nnn)
	if err != nil {
		if _, err := a.nodeInfoSnapshot.Get(nnn); err != nil {
			return nil, err
		}
		// It's not an error if NNN is in the cluster but not in the placement.
		// This can happen during the pod group placement scheduling cycle, where we simulate multiple potential placements.
		logger := klog.FromContext(ctx)
		logger.V(4).Info("Pod's nominated node is present in the cluster but not available in the current placement", "pod", klog.KObj(pod), "node", pod.Status.NominatedNodeName)
		return nil, nil
	}
	node := []fwk.NodeInfo{nodeInfo}
	feasibleNodes, err := a.findNodesThatPassFilters(ctx, schedFramework, state, pod, &diagnosis, node, int32(len(node)))
	if err != nil {
		return nil, err
	}

	feasibleNodes, err = findNodesThatPassExtenders(ctx, schedFramework.Extenders(), pod, feasibleNodes, diagnosis.NodeToStatus)
	if err != nil {
		return nil, err
	}

	return feasibleNodes, nil
}

// findNodesThatPassFilters finds the nodes that fit the filter plugins.
func (a *SchedulingAlgorithm) findNodesThatPassFilters(
	ctx context.Context,
	schedFramework framework.Framework,
	state fwk.CycleState,
	pod *v1.Pod,
	diagnosis *framework.Diagnosis,
	nodes []fwk.NodeInfo,
	numNodesToFind int32) ([]fwk.NodeInfo, error) {
	numAllNodes := len(nodes)
	// The budget can exceed the candidate list: numNodesToFind applies the n = 1
	// shortcut for profiles with neither extender filters nor scoring, which would
	// otherwise ask for one node out of an empty candidate list.
	numNodesToFind = min(numNodesToFind, int32(numAllNodes))

	// Create feasible list with enough space to avoid growing it
	// and allow assigning.
	feasibleNodes := make([]fwk.NodeInfo, numNodesToFind)

	if !schedFramework.HasFilterPlugins() {
		for i := range feasibleNodes {
			feasibleNodes[i] = nodes[(a.nextStartNodeIndex+i)%numAllNodes]
		}
		return feasibleNodes, nil
	}

	errCh := parallelize.NewResultChannel[error]()
	var feasibleNodesLen int32
	ctx, cancel := context.WithCancelCause(ctx)
	defer cancel(errors.New("findNodesThatPassFilters has completed"))

	type nodeStatus struct {
		node   string
		status *fwk.Status
	}
	result := make([]*nodeStatus, numAllNodes)
	checkNode := func(i int) {
		// We check the nodes starting from where we left off in the previous scheduling cycle,
		// this is to make sure all nodes have the same chance of being examined across pods.
		nodeInfo := nodes[(a.nextStartNodeIndex+i)%numAllNodes]
		status := schedFramework.RunFilterPluginsWithNominatedPods(ctx, state, pod, nodeInfo)
		if status.Code() == fwk.Error {
			errCh.SendWithCancel(status.AsError(), func() {
				cancel(errors.New("some other Filter operation failed"))
			})
			return
		}
		if status.IsSuccess() {
			length := atomic.AddInt32(&feasibleNodesLen, 1)
			if length > numNodesToFind {
				cancel(errors.New("findNodesThatPassFilters has found enough nodes"))
				atomic.AddInt32(&feasibleNodesLen, -1)
			} else {
				feasibleNodes[length-1] = nodeInfo
			}
		} else {
			result[i] = &nodeStatus{node: nodeInfo.Node().Name, status: status}
		}
	}

	beginCheckNode := time.Now()
	statusCode := fwk.Success
	defer func() {
		// We record Filter extension point latency here instead of in framework.go because framework.RunFilterPlugins
		// function is called for each node, whereas we want to have an overall latency for all nodes per scheduling cycle.
		// Note that this latency also includes latency for `addNominatedPods`, which calls framework.RunPreFilterAddPod.
		metrics.FrameworkExtensionPointDuration.WithLabelValues(metrics.Filter, statusCode.String(), schedFramework.ProfileName()).Observe(metrics.SinceInSeconds(beginCheckNode))
	}()

	// Stops searching for more nodes once the configured number of feasible nodes
	// are found.
	schedFramework.Parallelizer().Until(ctx, numAllNodes, checkNode, metrics.Filter)
	feasibleNodes = feasibleNodes[:feasibleNodesLen]
	for _, item := range result {
		if item == nil {
			continue
		}
		diagnosis.NodeToStatus.Set(item.node, item.status)
		diagnosis.AddPluginStatus(item.status)
	}
	if err := errCh.Receive(); err != nil {
		statusCode = fwk.Error
		return feasibleNodes, err
	}
	return feasibleNodes, nil
}

// numNodesToFind decides how many feasible nodes Filter should collect, following
// kube-scheduler's percentageOfNodesToScore policy. A profile that neither filters
// through extenders nor scores has no use for a second candidate, so one is enough.
func (a *SchedulingAlgorithm) numNodesToFind(schedFramework framework.Framework, numAllNodes int32) int32 {
	n := a.numFeasibleNodesToFind(schedFramework.PercentageOfNodesToScore(), numAllNodes)
	if !hasExtenderFilters(schedFramework) && !hasScoring(schedFramework) {
		n = 1
	}
	return n
}

// numFeasibleNodesToFind returns the number of feasible nodes that once found, the scheduler stops
// its search for more feasible nodes.
func (a *SchedulingAlgorithm) numFeasibleNodesToFind(percentageOfNodesToScore *int32, numAllNodes int32) (numNodes int32) {
	if numAllNodes < minFeasibleNodesToFind {
		return numAllNodes
	}

	// Use profile percentageOfNodesToScore if it's set. Otherwise, use global percentageOfNodesToScore.
	var percentage int32
	if percentageOfNodesToScore != nil {
		percentage = *percentageOfNodesToScore
	} else {
		percentage = a.percentageOfNodesToScore
	}

	if percentage == 0 {
		percentage = max(int32(50)-numAllNodes/125, minFeasibleNodesPercentageToFind)
	}

	numNodes = numAllNodes * percentage / 100
	if numNodes < minFeasibleNodesToFind {
		return minFeasibleNodesToFind
	}

	return numNodes
}

// prepareAssumedPod returns the copy of podInfo that will be recorded as assumed:
// bound to host and carrying the DRA allocations computed for this cycle. It
// touches no store — the caller decides where the result is recorded.
func (a *SchedulingAlgorithm) prepareAssumedPod(logger klog.Logger, state fwk.CycleState,
	podInfo *framework.QueuedPodInfo, host string) *framework.QueuedPodInfo {

	assumedPodInfo := podInfo.DeepCopy()
	assumedPodInfo.Pod.Spec.NodeName = host
	if utilfeature.DefaultFeatureGate.Enabled(features.DRANodeAllocatableResources) {
		// If DRANodeAllocatableResources is enabled, copy the calculated node allocatable resource claim status
		// from the cycle state to the assumed pod's status. This ensures that the scheduler's
		// cached version of the pod reflects the node allocatable resources allocated by the DRA plugin
		// for this scheduling cycle, making this information available for NodeInfo cache update.
		// Any potential NodeAllocatableResourceClaimStatuses from a previously failed scheduling attempt is overwritten.
		// This field is not explicitly cleared as the Pod object is reconstructed in handleSchedulingFailure()
		// before re-queueing.
		assumedPodInfo.Pod.Status.NodeAllocatableResourceClaimStatuses =
			dynamicresources.ExtractPodNodeAllocatableResourceClaimStatus(logger, state, host)
	}
	return assumedPodInfo
}

// reserve runs Reserve plugins for the assumed pod and converts plugin rejections
// into a single-node FitError.
func (a *SchedulingAlgorithm) reserve(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, assumedPodInfo *framework.QueuedPodInfo,
	origPod *v1.Pod, host string) *fwk.Status {
	if status := schedFramework.RunReservePluginsReserve(ctx, state, assumedPodInfo.Pod, host); !status.IsSuccess() {
		if status.IsRejected() {
			fitErr := &framework.FitError{
				NumAllNodes: 1,
				Pod:         origPod,
				Diagnosis: framework.Diagnosis{
					NodeToStatus: framework.NewDefaultNodeToStatus(),
				},
			}
			fitErr.Diagnosis.NodeToStatus.Set(host, status)
			fitErr.Diagnosis.AddPluginStatus(status)
			return fwk.NewStatus(status.Code()).WithError(fitErr)
		}
		return status
	}
	return nil
}

// FindAllNodesThatFitPod evaluates all placement nodes without early-exit shortcuts
// so callers inspecting cluster capacity or running custom batching receive exhaustive results.
func (a *SchedulingAlgorithm) FindAllNodesThatFitPod(ctx context.Context, state fwk.CycleState, schedFramework framework.Framework, podInfo *framework.QueuedPodInfo) ([]fwk.NodeInfo, framework.Diagnosis, error) {
	pod := podInfo.Pod
	preFilterOutcome, status := a.prefilterNodes(ctx, schedFramework, state, pod)
	diagnosis := preFilterOutcome.diagnosis
	if !status.IsSuccess() {
		if status.IsRejected() {
			return nil, *diagnosis, nil
		}
		return nil, *diagnosis, status.AsError()
	}

	nodes := a.getCandidateNodes(preFilterOutcome)
	feasibleNodes, err := a.findNodesThatPassFilters(ctx, schedFramework, state, pod, diagnosis, nodes, int32(len(nodes)))
	if err != nil {
		return nil, *diagnosis, err
	}
	feasibleNodes, err = a.filterWithExtenders(ctx, schedFramework, pod, feasibleNodes, diagnosis)

	return feasibleNodes, *diagnosis, err
}

// AssumeAndReserveInCache assumes the pod into the scheduler cache and runs Reserve plugins.
// If Reserve plugins fail, the assumption is rolled back from the cache.
func (a *SchedulingAlgorithm) AssumeAndReserveInCache(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
	scheduleResult ScheduleResult) (*framework.QueuedPodInfo, *fwk.Status) {

	logger := klog.FromContext(ctx)
	host := scheduleResult.SuggestedHost
	if a.cache == nil {
		return podInfo, fwk.AsStatus(errors.New("the SchedulingAlgorithm was built without a cache: " +
			"pass one to NewSchedulingAlgorithm, or assume into the snapshot instead"))
	}
	assumedPodInfo := a.prepareAssumedPod(logger, state, podInfo, host)
	if err := a.cache.AssumePod(logger, assumedPodInfo.Pod); err != nil {
		logger.Error(err, "Scheduler cache AssumePod failed")
		return assumedPodInfo, fwk.AsStatus(err)
	}
	schedFramework.DeleteNominatedPodIfExists(assumedPodInfo.Pod)

	if status := a.reserve(ctx, state, schedFramework, assumedPodInfo, podInfo.Pod, host); status != nil {
		if err := a.UnreserveAndForgetFromCache(ctx, state, schedFramework, assumedPodInfo, host); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "UnreserveAndForgetFromCache failed")
		}
		return assumedPodInfo, status
	}
	return assumedPodInfo, nil
}

// UnreserveAndForgetFromCache runs Unreserve plugins and forgets the pod from the scheduler cache.
func (a *SchedulingAlgorithm) UnreserveAndForgetFromCache(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, assumedPodInfo *framework.QueuedPodInfo, nodeName string) error {

	logger := klog.FromContext(ctx)
	schedFramework.RunReservePluginsUnreserve(ctx, state, assumedPodInfo.Pod, nodeName)
	// No nomination restore here: a pod that fails after a cache assume goes back
	// through handleSchedulingFailure, which re-adds the nomination itself.
	return a.cache.ForgetPod(logger, assumedPodInfo.Pod)
}

// AssumeAndReserveInSnapshot assumes the pod into the transient node snapshot and runs Reserve plugins.
// Returns a revert function to roll back the assumption and reservation from the snapshot if needed.
func (a *SchedulingAlgorithm) AssumeAndReserveInSnapshot(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, podInfo *framework.QueuedPodInfo,
	scheduleResult ScheduleResult) (*fwk.Status, func()) {

	logger := klog.FromContext(ctx)
	host := scheduleResult.SuggestedHost

	assumedPodInfo := a.prepareAssumedPod(logger, state, podInfo, host)
	if err := a.nodeInfoSnapshot.AssumePod(assumedPodInfo.PodInfo); err != nil {
		logger.Error(err, "Scheduler snapshot AssumePod failed")
		return fwk.AsStatus(err), nil
	}
	schedFramework.DeleteNominatedPodIfExists(assumedPodInfo.Pod)

	revert := func() {
		if err := a.unreserveAndForgetFromSnapshot(ctx, state, schedFramework, assumedPodInfo, host); err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "ForgetPod failed")
		}
	}
	if status := a.reserve(ctx, state, schedFramework, assumedPodInfo, podInfo.Pod, host); status != nil {
		revert()
		return status, nil
	}
	return nil, revert
}

// unreserveAndForgetFromSnapshot runs Unreserve plugins, forgets the pod from the node snapshot,
// and restores any existing pod nomination.
func (a *SchedulingAlgorithm) unreserveAndForgetFromSnapshot(ctx context.Context, state fwk.CycleState,
	schedFramework framework.Framework, assumedPodInfo *framework.QueuedPodInfo, nodeName string) error {

	logger := klog.FromContext(ctx)
	schedFramework.RunReservePluginsUnreserve(ctx, state, assumedPodInfo.Pod, nodeName)
	if err := a.nodeInfoSnapshot.ForgetPod(logger, assumedPodInfo.Pod); err != nil {
		return err
	}
	if assumedPodInfo.Pod.Status.NominatedNodeName != "" {
		// The assume removed the nomination; reverting a tentative assume restores it.
		schedFramework.AddNominatedPod(logger, assumedPodInfo.PodInfo, &fwk.NominatingInfo{
			NominatedNodeName: assumedPodInfo.Pod.Status.NominatedNodeName,
			NominatingMode:    fwk.ModeOverride,
		})
	}
	return nil
}
