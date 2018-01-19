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
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/api/core/v1"
	policy "k8s.io/api/policy/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/errors"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	"k8s.io/kubernetes/pkg/scheduler/util"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

type FailedPredicateMap map[string][]algorithm.PredicateFailureReason

type FitError struct {
	Pod              *v1.Pod
	NumAllNodes      int
	FailedPredicates FailedPredicateMap
}

type Victims struct {
	pods             []*v1.Pod
	numPDBViolations int
}

var ErrNoNodesAvailable = fmt.Errorf("no nodes available to schedule pods")

const (
	NoNodeAvailableMsg = "0/%v nodes are available"
	// NominatedNodeAnnotationKey is used to annotate a pod that has preempted other pods.
	// The scheduler uses the annotation to find that the pod shouldn't preempt more pods
	// when it gets to the head of scheduling queue again.
	// See podEligibleToPreemptOthers() for more information.
	NominatedNodeAnnotationKey = "scheduler.kubernetes.io/nominated-node-name"
)

// Error returns detailed information of why the pod failed to fit on each node
func (f *FitError) Error() string {
	reasons := make(map[string]int)
	for _, predicates := range f.FailedPredicates {
		for _, pred := range predicates {
			reasons[pred.GetReason()] += 1
		}
	}

	sortReasonsHistogram := func() []string {
		reasonStrings := []string{}
		for k, v := range reasons {
			reasonStrings = append(reasonStrings, fmt.Sprintf("%v %v", v, k))
		}
		sort.Strings(reasonStrings)
		return reasonStrings
	}
	reasonMsg := fmt.Sprintf(NoNodeAvailableMsg+": %v.", f.NumAllNodes, strings.Join(sortReasonsHistogram(), ", "))
	return reasonMsg
}

type genericScheduler struct {
	cache                    schedulercache.Cache
	equivalenceCache         *EquivalenceCache
	schedulingQueue          SchedulingQueue
	predicates               map[string]algorithm.FitPredicate
	priorityMetaProducer     algorithm.PriorityMetadataProducer
	predicateMetaProducer    algorithm.PredicateMetadataProducer
	prioritizers             []algorithm.PriorityConfig
	extenders                []algorithm.SchedulerExtender
	lastNodeIndexLock        sync.Mutex
	lastNodeIndex            uint64
	alwaysCheckAllPredicates bool
	cachedNodeInfoMap        map[string]*schedulercache.NodeInfo
	volumeBinder             *volumebinder.VolumeBinder
	pvcLister                corelisters.PersistentVolumeClaimLister
}

// Schedule tries to schedule the given pod to one of node in the node list.
// If it succeeds, it will return the name of the node.
// If it fails, it will return a Fiterror error with reasons.
func (g *genericScheduler) Schedule(pod *v1.Pod, nodeLister algorithm.NodeLister) (string, error) {
	trace := utiltrace.New(fmt.Sprintf("Scheduling %s/%s", pod.Namespace, pod.Name))
	defer trace.LogIfLong(100 * time.Millisecond)

	if err := podPassesBasicChecks(pod, g.pvcLister); err != nil {
		return "", err
	}

	nodes, err := nodeLister.List()
	if err != nil {
		return "", err
	}
	if len(nodes) == 0 {
		return "", ErrNoNodesAvailable
	}

	// Used for all fit and priority funcs.
	err = g.cache.UpdateNodeNameToInfoMap(g.cachedNodeInfoMap)
	if err != nil {
		return "", err
	}

	trace.Step("Computing predicates")
	startPredicateEvalTime := time.Now()
	filteredNodes, failedPredicateMap, err := findNodesThatFit(pod, g.cachedNodeInfoMap, nodes, g.predicates, g.extenders, g.predicateMetaProducer, g.equivalenceCache, g.schedulingQueue, g.alwaysCheckAllPredicates)
	if err != nil {
		return "", err
	}

	if len(filteredNodes) == 0 {
		return "", &FitError{
			Pod:              pod,
			NumAllNodes:      len(nodes),
			FailedPredicates: failedPredicateMap,
		}
	}
	metrics.SchedulingAlgorithmPredicateEvaluationDuration.Observe(metrics.SinceInMicroseconds(startPredicateEvalTime))

	trace.Step("Prioritizing")
	startPriorityEvalTime := time.Now()
	// When only one node after predicate, just use it.
	if len(filteredNodes) == 1 {
		metrics.SchedulingAlgorithmPriorityEvaluationDuration.Observe(metrics.SinceInMicroseconds(startPriorityEvalTime))
		return filteredNodes[0].Name, nil
	}

	metaPrioritiesInterface := g.priorityMetaProducer(pod, g.cachedNodeInfoMap)
	priorityList, err := PrioritizeNodes(pod, g.cachedNodeInfoMap, metaPrioritiesInterface, g.prioritizers, filteredNodes, g.extenders)
	if err != nil {
		return "", err
	}
	metrics.SchedulingAlgorithmPriorityEvaluationDuration.Observe(metrics.SinceInMicroseconds(startPriorityEvalTime))

	trace.Step("Selecting host")
	return g.selectHost(priorityList)
}

// Prioritizers returns a slice containing all the scheduler's priority
// functions and their config. It is exposed for testing only.
func (g *genericScheduler) Prioritizers() []algorithm.PriorityConfig {
	return g.prioritizers
}

// Predicates returns a map containing all the scheduler's predicate
// functions. It is exposed for testing only.
func (g *genericScheduler) Predicates() map[string]algorithm.FitPredicate {
	return g.predicates
}

// selectHost takes a prioritized list of nodes and then picks one
// in a round-robin manner from the nodes that had the highest score.
func (g *genericScheduler) selectHost(priorityList schedulerapi.HostPriorityList) (string, error) {
	if len(priorityList) == 0 {
		return "", fmt.Errorf("empty priorityList")
	}

	sort.Sort(sort.Reverse(priorityList))
	maxScore := priorityList[0].Score
	firstAfterMaxScore := sort.Search(len(priorityList), func(i int) bool { return priorityList[i].Score < maxScore })

	g.lastNodeIndexLock.Lock()
	ix := int(g.lastNodeIndex % uint64(firstAfterMaxScore))
	g.lastNodeIndex++
	g.lastNodeIndexLock.Unlock()

	return priorityList[ix].Host, nil
}

// preempt finds nodes with pods that can be preempted to make room for "pod" to
// schedule. It chooses one of the nodes and preempts the pods on the node and
// returns 1) the node, 2) the list of preempted pods if such a node is found,
// 3) A list of pods whose nominated node name should be cleared, and 4) any
// possible error.
func (g *genericScheduler) Preempt(pod *v1.Pod, nodeLister algorithm.NodeLister, scheduleErr error) (*v1.Node, []*v1.Pod, []*v1.Pod, error) {
	// Scheduler may return various types of errors. Consider preemption only if
	// the error is of type FitError.
	fitError, ok := scheduleErr.(*FitError)
	if !ok || fitError == nil {
		return nil, nil, nil, nil
	}
	err := g.cache.UpdateNodeNameToInfoMap(g.cachedNodeInfoMap)
	if err != nil {
		return nil, nil, nil, err
	}
	if !podEligibleToPreemptOthers(pod, g.cachedNodeInfoMap) {
		glog.V(5).Infof("Pod %v is not eligible for more preemption.", pod.Name)
		return nil, nil, nil, nil
	}
	allNodes, err := nodeLister.List()
	if err != nil {
		return nil, nil, nil, err
	}
	if len(allNodes) == 0 {
		return nil, nil, nil, ErrNoNodesAvailable
	}
	potentialNodes := nodesWherePreemptionMightHelp(pod, allNodes, fitError.FailedPredicates)
	if len(potentialNodes) == 0 {
		glog.V(3).Infof("Preemption will not help schedule pod %v on any node.", pod.Name)
		// In this case, we should clean-up any existing nominated node name of the pod.
		return nil, nil, []*v1.Pod{pod}, nil
	}
	pdbs, err := g.cache.ListPDBs(labels.Everything())
	if err != nil {
		return nil, nil, nil, err
	}
	nodeToVictims, err := selectNodesForPreemption(pod, g.cachedNodeInfoMap, potentialNodes, g.predicates, g.predicateMetaProducer, g.schedulingQueue, pdbs)
	if err != nil {
		return nil, nil, nil, err
	}
	for len(nodeToVictims) > 0 {
		node := pickOneNodeForPreemption(nodeToVictims)
		if node == nil {
			return nil, nil, nil, err
		}
		passes, pErr := nodePassesExtendersForPreemption(pod, node.Name, nodeToVictims[node].pods, g.cachedNodeInfoMap, g.extenders)
		if passes && pErr == nil {
			// Lower priority pods nominated to run on this node, may no longer fit on
			// this node. So, we should remove their nomination. Removing their
			// nomination updates these pods and moves them to the active queue. It
			// lets scheduler find another place for them.
			nominatedPods := g.getLowerPriorityNominatedPods(pod, node.Name)
			return node, nodeToVictims[node].pods, nominatedPods, err
		}
		if pErr != nil {
			glog.Errorf("Error occurred while checking extenders for preemption on node %v: %v", node, pErr)
		}
		// Remove the node from the map and try to pick a different node.
		delete(nodeToVictims, node)
	}
	return nil, nil, nil, err
}

// GetLowerPriorityNominatedPods returns pods whose priority is smaller than the
// priority of the given "pod" and are nominated to run on the given node.
// Note: We could possibly check if the nominated lower priority pods still fit
// and return those that no longer fit, but that would require lots of
// manipulation of NodeInfo and PredicateMeta per nominated pod. It may not be
// worth the complexity, especially because we generally expect to have a very
// small number of nominated pods per node.
func (g *genericScheduler) getLowerPriorityNominatedPods(pod *v1.Pod, nodeName string) []*v1.Pod {
	pods := g.schedulingQueue.WaitingPodsForNode(nodeName)
	if len(pods) == 0 {
		return nil
	}

	var lowerPriorityPods []*v1.Pod
	podPriority := util.GetPodPriority(pod)
	for _, p := range pods {
		if util.GetPodPriority(p) < podPriority {
			lowerPriorityPods = append(lowerPriorityPods, p)
		}
	}
	return lowerPriorityPods
}

// Filters the nodes to find the ones that fit based on the given predicate functions
// Each node is passed through the predicate functions to determine if it is a fit
func findNodesThatFit(
	pod *v1.Pod,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
	nodes []*v1.Node,
	predicateFuncs map[string]algorithm.FitPredicate,
	extenders []algorithm.SchedulerExtender,
	metadataProducer algorithm.PredicateMetadataProducer,
	ecache *EquivalenceCache,
	schedulingQueue SchedulingQueue,
	alwaysCheckAllPredicates bool,
) ([]*v1.Node, FailedPredicateMap, error) {
	var filtered []*v1.Node
	failedPredicateMap := FailedPredicateMap{}

	if len(predicateFuncs) == 0 {
		filtered = nodes
	} else {
		// Create filtered list with enough space to avoid growing it
		// and allow assigning.
		filtered = make([]*v1.Node, len(nodes))
		errs := errors.MessageCountMap{}
		var predicateResultLock sync.Mutex
		var filteredLen int32

		// We can use the same metadata producer for all nodes.
		meta := metadataProducer(pod, nodeNameToInfo)
		checkNode := func(i int) {
			nodeName := nodes[i].Name
			fits, failedPredicates, err := podFitsOnNode(pod, meta, nodeNameToInfo[nodeName], predicateFuncs, ecache, schedulingQueue, alwaysCheckAllPredicates)
			if err != nil {
				predicateResultLock.Lock()
				errs[err.Error()]++
				predicateResultLock.Unlock()
				return
			}
			if fits {
				filtered[atomic.AddInt32(&filteredLen, 1)-1] = nodes[i]
			} else {
				predicateResultLock.Lock()
				failedPredicateMap[nodeName] = failedPredicates
				predicateResultLock.Unlock()
			}
		}
		workqueue.Parallelize(16, len(nodes), checkNode)
		filtered = filtered[:filteredLen]
		if len(errs) > 0 {
			return []*v1.Node{}, FailedPredicateMap{}, errors.CreateAggregateFromMessageCountMap(errs)
		}
	}

	if len(filtered) > 0 && len(extenders) != 0 {
		for _, extender := range extenders {
			filteredList, failedMap, err := extender.Filter(pod, filtered, nodeNameToInfo)
			if err != nil {
				return []*v1.Node{}, FailedPredicateMap{}, err
			}

			for failedNodeName, failedMsg := range failedMap {
				if _, found := failedPredicateMap[failedNodeName]; !found {
					failedPredicateMap[failedNodeName] = []algorithm.PredicateFailureReason{}
				}
				failedPredicateMap[failedNodeName] = append(failedPredicateMap[failedNodeName], predicates.NewFailureReason(failedMsg))
			}
			filtered = filteredList
			if len(filtered) == 0 {
				break
			}
		}
	}
	return filtered, failedPredicateMap, nil
}

// addNominatedPods adds pods with equal or greater priority which are nominated
// to run on the node given in nodeInfo to meta and nodeInfo. It returns 1) whether
// any pod was found, 2) augmented meta data, 3) augmented nodeInfo.
func addNominatedPods(podPriority int32, meta algorithm.PredicateMetadata,
	nodeInfo *schedulercache.NodeInfo, queue SchedulingQueue) (bool, algorithm.PredicateMetadata,
	*schedulercache.NodeInfo) {
	if queue == nil || nodeInfo == nil || nodeInfo.Node() == nil {
		// This may happen only in tests.
		return false, meta, nodeInfo
	}
	nominatedPods := queue.WaitingPodsForNode(nodeInfo.Node().Name)
	if nominatedPods == nil || len(nominatedPods) == 0 {
		return false, meta, nodeInfo
	}
	var metaOut algorithm.PredicateMetadata = nil
	if meta != nil {
		metaOut = meta.ShallowCopy()
	}
	nodeInfoOut := nodeInfo.Clone()
	for _, p := range nominatedPods {
		if util.GetPodPriority(p) >= podPriority {
			nodeInfoOut.AddPod(p)
			if metaOut != nil {
				metaOut.AddPod(p, nodeInfoOut)
			}
		}
	}
	return true, metaOut, nodeInfoOut
}

// podFitsOnNode checks whether a node given by NodeInfo satisfies the given predicate functions.
// This function is called from two different places: Schedule and Preempt.
// When it is called from Schedule, we want to test whether the pod is schedulable
// on the node with all the existing pods on the node plus higher and equal priority
// pods nominated to run on the node.
// When it is called from Preempt, we should remove the victims of preemption and
// add the nominated pods. Removal of the victims is done by SelectVictimsOnNode().
// It removes victims from meta and NodeInfo before calling this function.
func podFitsOnNode(
	pod *v1.Pod,
	meta algorithm.PredicateMetadata,
	info *schedulercache.NodeInfo,
	predicateFuncs map[string]algorithm.FitPredicate,
	ecache *EquivalenceCache,
	queue SchedulingQueue,
	alwaysCheckAllPredicates bool,
) (bool, []algorithm.PredicateFailureReason, error) {
	var (
		equivalenceHash  uint64
		failedPredicates []algorithm.PredicateFailureReason
		eCacheAvailable  bool
		invalid          bool
		fit              bool
		reasons          []algorithm.PredicateFailureReason
		err              error
	)
	predicateResults := make(map[string]HostPredicate)

	if ecache != nil {
		// getHashEquivalencePod will return immediately if no equivalence pod found
		equivalenceHash, eCacheAvailable = ecache.getHashEquivalencePod(pod)
	}
	podsAdded := false
	// We run predicates twice in some cases. If the node has greater or equal priority
	// nominated pods, we run them when those pods are added to meta and nodeInfo.
	// If all predicates succeed in this pass, we run them again when these
	// nominated pods are not added. This second pass is necessary because some
	// predicates such as inter-pod affinity may not pass without the nominated pods.
	// If there are no nominated pods for the node or if the first run of the
	// predicates fail, we don't run the second pass.
	// We consider only equal or higher priority pods in the first pass, because
	// those are the current "pod" must yield to them and not take a space opened
	// for running them. It is ok if the current "pod" take resources freed for
	// lower priority pods.
	// Requiring that the new pod is schedulable in both circumstances ensures that
	// we are making a conservative decision: predicates like resources and inter-pod
	// anti-affinity are more likely to fail when the nominated pods are treated
	// as running, while predicates like pod affinity are more likely to fail when
	// the nominated pods are treated as not running. We can't just assume the
	// nominated pods are running because they are not running right now and in fact,
	// they may end up getting scheduled to a different node.
	for i := 0; i < 2; i++ {
		metaToUse := meta
		nodeInfoToUse := info
		if i == 0 {
			podsAdded, metaToUse, nodeInfoToUse = addNominatedPods(util.GetPodPriority(pod), meta, info, queue)
		} else if !podsAdded || len(failedPredicates) != 0 {
			break
		}
		// Bypass eCache if node has any nominated pods.
		// TODO(bsalamat): consider using eCache and adding proper eCache invalidations
		// when pods are nominated or their nominations change.
		eCacheAvailable = eCacheAvailable && !podsAdded
		for _, predicateKey := range predicates.PredicatesOrdering() {
			//TODO (yastij) : compute average predicate restrictiveness to export it as promethus metric
			if predicate, exist := predicateFuncs[predicateKey]; exist {
				if eCacheAvailable {
					// PredicateWithECache will return its cached predicate results.
					fit, reasons, invalid = ecache.PredicateWithECache(pod.GetName(), info.Node().GetName(), predicateKey, equivalenceHash)
				}

				if !eCacheAvailable || invalid {
					// we need to execute predicate functions since equivalence cache does not work
					fit, reasons, err = predicate(pod, metaToUse, nodeInfoToUse)
					if err != nil {
						return false, []algorithm.PredicateFailureReason{}, err
					}
					if eCacheAvailable {
						// Store data to update eCache after this loop.
						if res, exists := predicateResults[predicateKey]; exists {
							res.Fit = res.Fit && fit
							res.FailReasons = append(res.FailReasons, reasons...)
							predicateResults[predicateKey] = res
						} else {
							predicateResults[predicateKey] = HostPredicate{Fit: fit, FailReasons: reasons}
						}
					}
				}
				if !fit {
					// eCache is available and valid, and predicates result is unfit, record the fail reasons
					failedPredicates = append(failedPredicates, reasons...)
					// if alwaysCheckAllPredicates is false, short circuit all predicates when one predicate fails.
					if !alwaysCheckAllPredicates {
						glog.V(5).Infoln("since alwaysCheckAllPredicates has not been set, the predicate evaluation is short circuited and there are chances of other predicates failing as well.")
						break
					}
				}
			}
		}
	}

	// TODO(bsalamat): This way of updating equiv. cache has a race condition against
	// cache invalidations invoked in event handlers. This race has existed despite locks
	// in eCache implementation. If cache is invalidated after a predicate is executed
	// and before we update the cache, the updates should not be written to the cache.
	if eCacheAvailable {
		nodeName := info.Node().GetName()
		for predKey, result := range predicateResults {
			// update equivalence cache with newly computed fit & reasons
			// TODO(resouer) should we do this in another thread? any race?
			ecache.UpdateCachedPredicateItem(pod.GetName(), nodeName, predKey, result.Fit, result.FailReasons, equivalenceHash)
		}
	}
	return len(failedPredicates) == 0, failedPredicates, nil
}

// Prioritizes the nodes by running the individual priority functions in parallel.
// Each priority function is expected to set a score of 0-10
// 0 is the lowest priority score (least preferred node) and 10 is the highest
// Each priority function can also have its own weight
// The node scores returned by the priority function are multiplied by the weights to get weighted scores
// All scores are finally combined (added) to get the total weighted scores of all nodes
func PrioritizeNodes(
	pod *v1.Pod,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
	meta interface{},
	priorityConfigs []algorithm.PriorityConfig,
	nodes []*v1.Node,
	extenders []algorithm.SchedulerExtender,
) (schedulerapi.HostPriorityList, error) {
	// If no priority configs are provided, then the EqualPriority function is applied
	// This is required to generate the priority list in the required format
	if len(priorityConfigs) == 0 && len(extenders) == 0 {
		result := make(schedulerapi.HostPriorityList, 0, len(nodes))
		for i := range nodes {
			hostPriority, err := EqualPriorityMap(pod, meta, nodeNameToInfo[nodes[i].Name])
			if err != nil {
				return nil, err
			}
			result = append(result, hostPriority)
		}
		return result, nil
	}

	var (
		mu   = sync.Mutex{}
		wg   = sync.WaitGroup{}
		errs []error
	)
	appendError := func(err error) {
		mu.Lock()
		defer mu.Unlock()
		errs = append(errs, err)
	}

	results := make([]schedulerapi.HostPriorityList, len(priorityConfigs), len(priorityConfigs))

	for i, priorityConfig := range priorityConfigs {
		if priorityConfig.Function != nil {
			// DEPRECATED
			wg.Add(1)
			go func(index int, config algorithm.PriorityConfig) {
				defer wg.Done()
				var err error
				results[index], err = config.Function(pod, nodeNameToInfo, nodes)
				if err != nil {
					appendError(err)
				}
			}(i, priorityConfig)
		} else {
			results[i] = make(schedulerapi.HostPriorityList, len(nodes))
		}
	}
	processNode := func(index int) {
		nodeInfo := nodeNameToInfo[nodes[index].Name]
		var err error
		for i := range priorityConfigs {
			if priorityConfigs[i].Function != nil {
				continue
			}
			results[i][index], err = priorityConfigs[i].Map(pod, meta, nodeInfo)
			if err != nil {
				appendError(err)
				return
			}
		}
	}
	workqueue.Parallelize(16, len(nodes), processNode)
	for i, priorityConfig := range priorityConfigs {
		if priorityConfig.Reduce == nil {
			continue
		}
		wg.Add(1)
		go func(index int, config algorithm.PriorityConfig) {
			defer wg.Done()
			if err := config.Reduce(pod, meta, nodeNameToInfo, results[index]); err != nil {
				appendError(err)
			}
			if glog.V(10) {
				for _, hostPriority := range results[index] {
					glog.Infof("%v -> %v: %v, Score: (%d)", pod.Name, hostPriority.Host, config.Name, hostPriority.Score)
				}
			}
		}(i, priorityConfig)
	}
	// Wait for all computations to be finished.
	wg.Wait()
	if len(errs) != 0 {
		return schedulerapi.HostPriorityList{}, errors.NewAggregate(errs)
	}

	// Summarize all scores.
	result := make(schedulerapi.HostPriorityList, 0, len(nodes))

	for i := range nodes {
		result = append(result, schedulerapi.HostPriority{Host: nodes[i].Name, Score: 0})
		for j := range priorityConfigs {
			result[i].Score += results[j][i].Score * priorityConfigs[j].Weight
		}
	}

	if len(extenders) != 0 && nodes != nil {
		combinedScores := make(map[string]int, len(nodeNameToInfo))
		for _, extender := range extenders {
			wg.Add(1)
			go func(ext algorithm.SchedulerExtender) {
				defer wg.Done()
				prioritizedList, weight, err := ext.Prioritize(pod, nodes)
				if err != nil {
					// Prioritization errors from extender can be ignored, let k8s/other extenders determine the priorities
					return
				}
				mu.Lock()
				for i := range *prioritizedList {
					host, score := (*prioritizedList)[i].Host, (*prioritizedList)[i].Score
					combinedScores[host] += score * weight
				}
				mu.Unlock()
			}(extender)
		}
		// wait for all go routines to finish
		wg.Wait()
		for i := range result {
			result[i].Score += combinedScores[result[i].Host]
		}
	}

	if glog.V(10) {
		for i := range result {
			glog.V(10).Infof("Host %s => Score %d", result[i].Host, result[i].Score)
		}
	}
	return result, nil
}

// EqualPriority is a prioritizer function that gives an equal weight of one to all nodes
func EqualPriorityMap(_ *v1.Pod, _ interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}
	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: 1,
	}, nil
}

// pickOneNodeForPreemption chooses one node among the given nodes. It assumes
// pods in each map entry are ordered by decreasing priority.
// It picks a node based on the following criteria:
// 1. A node with minimum number of PDB violations.
// 2. A node with minimum highest priority victim is picked.
// 3. Ties are broken by sum of priorities of all victims.
// 4. If there are still ties, node with the minimum number of victims is picked.
// 5. If there are still ties, the first such node is picked (sort of randomly).
//TODO(bsalamat): Try to reuse the "min*Nodes" slices in order to save GC time.
func pickOneNodeForPreemption(nodesToVictims map[*v1.Node]*Victims) *v1.Node {
	if len(nodesToVictims) == 0 {
		return nil
	}
	minNumPDBViolatingPods := math.MaxInt32
	var minPDBViolatingNodes []*v1.Node
	for node, victims := range nodesToVictims {
		if len(victims.pods) == 0 {
			// We found a node that doesn't need any preemption. Return it!
			// This should happen rarely when one or more pods are terminated between
			// the time that scheduler tries to schedule the pod and the time that
			// preemption logic tries to find nodes for preemption.
			return node
		}
		numPDBViolatingPods := victims.numPDBViolations
		if numPDBViolatingPods < minNumPDBViolatingPods {
			minNumPDBViolatingPods = numPDBViolatingPods
			minPDBViolatingNodes = nil
		}
		if numPDBViolatingPods == minNumPDBViolatingPods {
			minPDBViolatingNodes = append(minPDBViolatingNodes, node)
		}
	}
	if len(minPDBViolatingNodes) == 1 {
		return minPDBViolatingNodes[0]
	}

	// There are more than one node with minimum number PDB violating pods. Find
	// the one with minimum highest priority victim.
	minHighestPriority := int32(math.MaxInt32)
	var minPriorityNodes []*v1.Node
	for _, node := range minPDBViolatingNodes {
		victims := nodesToVictims[node]
		// highestPodPriority is the highest priority among the victims on this node.
		highestPodPriority := util.GetPodPriority(victims.pods[0])
		if highestPodPriority < minHighestPriority {
			minHighestPriority = highestPodPriority
			minPriorityNodes = nil
		}
		if highestPodPriority == minHighestPriority {
			minPriorityNodes = append(minPriorityNodes, node)
		}
	}
	if len(minPriorityNodes) == 1 {
		return minPriorityNodes[0]
	}

	// There are a few nodes with minimum highest priority victim. Find the
	// smallest sum of priorities.
	minSumPriorities := int64(math.MaxInt64)
	var minSumPriorityNodes []*v1.Node
	for _, node := range minPriorityNodes {
		var sumPriorities int64
		for _, pod := range nodesToVictims[node].pods {
			// We add MaxInt32+1 to all priorities to make all of them >= 0. This is
			// needed so that a node with a few pods with negative priority is not
			// picked over a node with a smaller number of pods with the same negative
			// priority (and similar scenarios).
			sumPriorities += int64(util.GetPodPriority(pod)) + int64(math.MaxInt32+1)
		}
		if sumPriorities < minSumPriorities {
			minSumPriorities = sumPriorities
			minSumPriorityNodes = nil
		}
		if sumPriorities == minSumPriorities {
			minSumPriorityNodes = append(minSumPriorityNodes, node)
		}
	}
	if len(minSumPriorityNodes) == 1 {
		return minSumPriorityNodes[0]
	}

	// There are a few nodes with minimum highest priority victim and sum of priorities.
	// Find one with the minimum number of pods.
	minNumPods := math.MaxInt32
	var minNumPodNodes []*v1.Node
	for _, node := range minSumPriorityNodes {
		numPods := len(nodesToVictims[node].pods)
		if numPods < minNumPods {
			minNumPods = numPods
			minNumPodNodes = nil
		}
		if numPods == minNumPods {
			minNumPodNodes = append(minNumPodNodes, node)
		}
	}
	// At this point, even if there are more than one node with the same score,
	// return the first one.
	if len(minNumPodNodes) > 0 {
		return minNumPodNodes[0]
	}
	glog.Errorf("Error in logic of node scoring for preemption. We should never reach here!")
	return nil
}

// selectNodesForPreemption finds all the nodes with possible victims for
// preemption in parallel.
func selectNodesForPreemption(pod *v1.Pod,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
	potentialNodes []*v1.Node,
	predicates map[string]algorithm.FitPredicate,
	metadataProducer algorithm.PredicateMetadataProducer,
	queue SchedulingQueue,
	pdbs []*policy.PodDisruptionBudget,
) (map[*v1.Node]*Victims, error) {

	nodeNameToVictims := map[*v1.Node]*Victims{}
	var resultLock sync.Mutex

	// We can use the same metadata producer for all nodes.
	meta := metadataProducer(pod, nodeNameToInfo)
	checkNode := func(i int) {
		nodeName := potentialNodes[i].Name
		var metaCopy algorithm.PredicateMetadata
		if meta != nil {
			metaCopy = meta.ShallowCopy()
		}
		pods, numPDBViolations, fits := selectVictimsOnNode(pod, metaCopy, nodeNameToInfo[nodeName], predicates, queue, pdbs)
		if fits {
			resultLock.Lock()
			victims := Victims{
				pods:             pods,
				numPDBViolations: numPDBViolations,
			}
			nodeNameToVictims[potentialNodes[i]] = &victims
			resultLock.Unlock()
		}
	}
	workqueue.Parallelize(16, len(potentialNodes), checkNode)
	return nodeNameToVictims, nil
}

func nodePassesExtendersForPreemption(
	pod *v1.Pod,
	nodeName string,
	victims []*v1.Pod,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
	extenders []algorithm.SchedulerExtender) (bool, error) {
	// If there are any extenders, run them and filter the list of candidate nodes.
	if len(extenders) == 0 {
		return true, nil
	}
	// Remove the victims from the corresponding nodeInfo and send nodes to the
	// extenders for filtering.
	originalNodeInfo := nodeNameToInfo[nodeName]
	nodeInfoCopy := nodeNameToInfo[nodeName].Clone()
	for _, victim := range victims {
		nodeInfoCopy.RemovePod(victim)
	}
	nodeNameToInfo[nodeName] = nodeInfoCopy
	defer func() { nodeNameToInfo[nodeName] = originalNodeInfo }()
	filteredNodes := []*v1.Node{nodeInfoCopy.Node()}
	for _, extender := range extenders {
		var err error
		var failedNodesMap map[string]string
		filteredNodes, failedNodesMap, err = extender.Filter(pod, filteredNodes, nodeNameToInfo)
		if err != nil {
			return false, err
		}
		if _, found := failedNodesMap[nodeName]; found || len(filteredNodes) == 0 {
			return false, nil
		}
	}
	return true, nil
}

// filterPodsWithPDBViolation groups the given "pods" into two groups of "violatingPods"
// and "nonViolatingPods" based on whether their PDBs will be violated if they are
// preempted.
// This function is stable and does not change the order of received pods. So, if it
// receives a sorted list, grouping will preserve the order of the input list.
func filterPodsWithPDBViolation(pods []interface{}, pdbs []*policy.PodDisruptionBudget) (violatingPods, nonViolatingPods []*v1.Pod) {
	for _, obj := range pods {
		pod := obj.(*v1.Pod)
		pdbForPodIsViolated := false
		// A pod with no labels will not match any PDB. So, no need to check.
		if len(pod.Labels) != 0 {
			for _, pdb := range pdbs {
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
				if pdb.Status.PodDisruptionsAllowed <= 0 {
					pdbForPodIsViolated = true
					break
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
	pod *v1.Pod,
	meta algorithm.PredicateMetadata,
	nodeInfo *schedulercache.NodeInfo,
	fitPredicates map[string]algorithm.FitPredicate,
	queue SchedulingQueue,
	pdbs []*policy.PodDisruptionBudget,
) ([]*v1.Pod, int, bool) {
	potentialVictims := util.SortableList{CompFunc: util.HigherPriorityPod}
	nodeInfoCopy := nodeInfo.Clone()

	removePod := func(rp *v1.Pod) {
		nodeInfoCopy.RemovePod(rp)
		if meta != nil {
			meta.RemovePod(rp)
		}
	}
	addPod := func(ap *v1.Pod) {
		nodeInfoCopy.AddPod(ap)
		if meta != nil {
			meta.AddPod(ap, nodeInfoCopy)
		}
	}
	// As the first step, remove all the lower priority pods from the node and
	// check if the given pod can be scheduled.
	podPriority := util.GetPodPriority(pod)
	for _, p := range nodeInfoCopy.Pods() {
		if util.GetPodPriority(p) < podPriority {
			potentialVictims.Items = append(potentialVictims.Items, p)
			removePod(p)
		}
	}
	potentialVictims.Sort()
	// If the new pod does not fit after removing all the lower priority pods,
	// we are almost done and this node is not suitable for preemption. The only condition
	// that we should check is if the "pod" is failing to schedule due to pod affinity
	// failure.
	// TODO(bsalamat): Consider checking affinity to lower priority pods if feasible with reasonable performance.
	if fits, _, err := podFitsOnNode(pod, meta, nodeInfoCopy, fitPredicates, nil, queue, false); !fits {
		if err != nil {
			glog.Warningf("Encountered error while selecting victims on node %v: %v", nodeInfo.Node().Name, err)
		}
		return nil, 0, false
	}
	var victims []*v1.Pod
	numViolatingVictim := 0
	// Try to reprieve as many pods as possible. We first try to reprieve the PDB
	// violating victims and then other non-violating ones. In both cases, we start
	// from the highest priority victims.
	violatingVictims, nonViolatingVictims := filterPodsWithPDBViolation(potentialVictims.Items, pdbs)
	reprievePod := func(p *v1.Pod) bool {
		addPod(p)
		fits, _, _ := podFitsOnNode(pod, meta, nodeInfoCopy, fitPredicates, nil, queue, false)
		if !fits {
			removePod(p)
			victims = append(victims, p)
			glog.V(5).Infof("Pod %v is a potential preemption victim on node %v.", p.Name, nodeInfo.Node().Name)
		}
		return fits
	}
	for _, p := range violatingVictims {
		if !reprievePod(p) {
			numViolatingVictim++
		}
	}
	// Now we try to reprieve non-violating victims.
	for _, p := range nonViolatingVictims {
		reprievePod(p)
	}
	return victims, numViolatingVictim, true
}

// nodesWherePreemptionMightHelp returns a list of nodes with failed predicates
// that may be satisfied by removing pods from the node.
func nodesWherePreemptionMightHelp(pod *v1.Pod, nodes []*v1.Node, failedPredicatesMap FailedPredicateMap) []*v1.Node {
	potentialNodes := []*v1.Node{}
	for _, node := range nodes {
		unresolvableReasonExist := false
		failedPredicates, found := failedPredicatesMap[node.Name]
		// If we assume that scheduler looks at all nodes and populates the failedPredicateMap
		// (which is the case today), the !found case should never happen, but we'd prefer
		// to rely less on such assumptions in the code when checking does not impose
		// significant overhead.
		for _, failedPredicate := range failedPredicates {
			switch failedPredicate {
			case
				predicates.ErrNodeSelectorNotMatch,
				predicates.ErrPodNotMatchHostName,
				predicates.ErrTaintsTolerationsNotMatch,
				predicates.ErrNodeLabelPresenceViolated,
				predicates.ErrNodeNotReady,
				predicates.ErrNodeNetworkUnavailable,
				predicates.ErrNodeUnschedulable,
				predicates.ErrNodeUnknownCondition,
				predicates.ErrVolumeZoneConflict,
				predicates.ErrVolumeNodeConflict,
				predicates.ErrVolumeBindConflict:
				unresolvableReasonExist = true
				break
				// TODO(bsalamat): Please add affinity failure cases once we have specific affinity failure errors.
			}
		}
		if !found || !unresolvableReasonExist {
			glog.V(3).Infof("Node %v is a potential node for preemption.", node.Name)
			potentialNodes = append(potentialNodes, node)
		}
	}
	return potentialNodes
}

// podEligibleToPreemptOthers determines whether this pod should be considered
// for preempting other pods or not. If this pod has already preempted other
// pods and those are in their graceful termination period, it shouldn't be
// considered for preemption.
// We look at the node that is nominated for this pod and as long as there are
// terminating pods on the node, we don't consider this for preempting more pods.
func podEligibleToPreemptOthers(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo) bool {
	if nodeName, found := pod.Annotations[NominatedNodeAnnotationKey]; found {
		if nodeInfo, found := nodeNameToInfo[nodeName]; found {
			for _, p := range nodeInfo.Pods() {
				if p.DeletionTimestamp != nil && util.GetPodPriority(p) < util.GetPodPriority(pod) {
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

func NewGenericScheduler(
	cache schedulercache.Cache,
	eCache *EquivalenceCache,
	podQueue SchedulingQueue,
	predicates map[string]algorithm.FitPredicate,
	predicateMetaProducer algorithm.PredicateMetadataProducer,
	prioritizers []algorithm.PriorityConfig,
	priorityMetaProducer algorithm.PriorityMetadataProducer,
	extenders []algorithm.SchedulerExtender,
	volumeBinder *volumebinder.VolumeBinder,
	pvcLister corelisters.PersistentVolumeClaimLister,
	alwaysCheckAllPredicates bool) algorithm.ScheduleAlgorithm {
	return &genericScheduler{
		cache:                    cache,
		equivalenceCache:         eCache,
		schedulingQueue:          podQueue,
		predicates:               predicates,
		predicateMetaProducer:    predicateMetaProducer,
		prioritizers:             prioritizers,
		priorityMetaProducer:     priorityMetaProducer,
		extenders:                extenders,
		cachedNodeInfoMap:        make(map[string]*schedulercache.NodeInfo),
		volumeBinder:             volumeBinder,
		pvcLister:                pvcLister,
		alwaysCheckAllPredicates: alwaysCheckAllPredicates,
	}
}
