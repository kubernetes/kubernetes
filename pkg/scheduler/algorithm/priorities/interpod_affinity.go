/*
Copyright 2016 The Kubernetes Authors.

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

package priorities

import (
	"fmt"
	"sync"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

type InterPodAffinity struct {
	info                  predicates.NodeInfo
	nodeLister            algorithm.NodeLister
	podLister             algorithm.PodLister
	hardPodAffinityWeight int32
}

func NewInterPodAffinityPriority(
	info predicates.NodeInfo,
	nodeLister algorithm.NodeLister,
	podLister algorithm.PodLister,
	hardPodAffinityWeight int32) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
	interPodAffinity := &InterPodAffinity{
		info:                  info,
		nodeLister:            nodeLister,
		podLister:             podLister,
		hardPodAffinityWeight: hardPodAffinityWeight,
	}
	return interPodAffinity.CalculateInterPodAffinityPriorityMap, interPodAffinity.CalculateInterPodAffinityPriorityReduce
}

type podAffinityPriorityMap struct {
	sync.Mutex

	// nodes contain all nodes that should be considered
	nodes []*v1.Node
	// counts store the mapping from node name to so-far computed score of
	// the node.
	nameToIndex map[string]int
	counts      []schedulerapi.HostPriority
	// The first error that we faced.
	firstError error
}

var countsDataInitLock sync.Mutex
var podAffinityPriorityMapPointer *podAffinityPriorityMap

func deletePodAffinityCountsData() {
	podAffinityPriorityMapPointer = nil
}

func newPodAffinityPriorityMap(nodes []*v1.Node) *podAffinityPriorityMap {
	countsDataInitLock.Lock()
	defer countsDataInitLock.Unlock()
	if podAffinityPriorityMapPointer == nil {
		podAffinityPriorityMapPointer = &podAffinityPriorityMap{
			nodes:       nodes,
			nameToIndex: make(map[string]int, len(nodes)),
			counts:      make([]schedulerapi.HostPriority, len(nodes)),
		}
		//map node name to index for finding node data place quickly in invocation processTerm
		for index, node := range nodes {
			podAffinityPriorityMapPointer.nameToIndex[node.Name] = index
		}
	}
	return podAffinityPriorityMapPointer
}

func (p *podAffinityPriorityMap) setError(err error) {
	p.Lock()
	defer p.Unlock()
	if p.firstError == nil {
		p.firstError = err
	}
}

func (p *podAffinityPriorityMap) processTerm(term *v1.PodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, weight float64) {
	namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(podDefiningAffinityTerm, term)
	selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
	if err != nil {
		p.setError(err)
		return
	}
	match := priorityutil.PodMatchesTermsNamespaceAndSelector(podToCheck, namespaces, selector)
	if match {
		func() {
			p.Lock()
			defer p.Unlock()
			var index int
			for _, node := range p.nodes {
				if priorityutil.NodesHaveSameTopologyKey(node, fixedNode, term.TopologyKey) {
					index = p.nameToIndex[node.Name]
					p.counts[index].Host = node.Name
					p.counts[index].Score = int(float64(p.counts[index].Score) + weight)
				}
			}
		}()
	}
}

func (p *podAffinityPriorityMap) processTerms(terms []v1.WeightedPodAffinityTerm, podDefiningAffinityTerm, podToCheck *v1.Pod, fixedNode *v1.Node, multiplier int) {
	for i := range terms {
		term := &terms[i]
		p.processTerm(&term.PodAffinityTerm, podDefiningAffinityTerm, podToCheck, fixedNode, float64(term.Weight*int32(multiplier)))
	}
}

// CalculateInterPodAffinityPriorityMap compute a sum by iterating through the elements of weightedPodAffinityTerm and adding
// "weight" to the sum if the corresponding PodAffinityTerm is satisfied for
// that node;
// Symmetry need to be considered for preferredDuringSchedulingIgnoredDuringExecution from podAffinity & podAntiAffinity,
// symmetry need to be considered for hard requirements from podAffinity
func (ipa *InterPodAffinity) CalculateInterPodAffinityPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (*schedulerapi.HostPriority, error) {

	var affinity *v1.Affinity
	var nodes []*v1.Node
	node := nodeInfo.Node()
	if node == nil {
		return &schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	priorityMeta, ok := meta.(*priorityMetadata)
	if ok {
		affinity = priorityMeta.affinity
		nodes = priorityMeta.nodes
	} else {
		//TODO (guangxuli) seems that we can not get filtered nodes directly
		return &schedulerapi.HostPriority{}, fmt.Errorf("can not get priority meta data")
	}

	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	// priorityMap stores the mapping from node name to so-far computed score of
	// the node.
	pm := newPodAffinityPriorityMap(nodes)

	processPod := func(existingPod *v1.Pod) error {
		existingPodNode, err := ipa.info.GetNodeInfo(existingPod.Spec.NodeName)
		if err != nil {
			if apierrors.IsNotFound(err) {
				glog.Errorf("Node not found, %v", existingPod.Spec.NodeName)
				return nil
			}
			return err
		}
		existingPodAffinity := existingPod.Spec.Affinity
		existingHasAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAffinity != nil
		existingHasAntiAffinityConstraints := existingPodAffinity != nil && existingPodAffinity.PodAntiAffinity != nil

		if hasAffinityConstraints {
			// For every soft pod affinity term of <pod>, if <existingPod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPods>`s node by the term`s weight.
			terms := affinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			pm.processTerms(terms, pod, existingPod, existingPodNode, 1)
		}
		if hasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <pod>, if <existingPod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>`s node by the term`s weight.
			terms := affinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			pm.processTerms(terms, pod, existingPod, existingPodNode, -1)
		}

		if existingHasAffinityConstraints {
			// For every hard pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the constant <ipa.hardPodAffinityWeight>
			if ipa.hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					pm.processTerm(&term, existingPod, pod, existingPodNode, float64(ipa.hardPodAffinityWeight))
				}
			}
			// For every soft pod affinity term of <existingPod>, if <pod> matches the term,
			// increment <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			pm.processTerms(terms, existingPod, pod, existingPodNode, 1)
		}
		if existingHasAntiAffinityConstraints {
			// For every soft pod anti-affinity term of <existingPod>, if <pod> matches the term,
			// decrement <pm.counts> for every node in the cluster with the same <term.TopologyKey>
			// value as that of <existingPod>'s node by the term's weight.
			terms := existingPodAffinity.PodAntiAffinity.PreferredDuringSchedulingIgnoredDuringExecution
			pm.processTerms(terms, existingPod, pod, existingPodNode, -1)
		}
		return nil
	}

	if hasAffinityConstraints || hasAntiAffinityConstraints {
		// We need to process all the nodes.
		for _, existingPod := range nodeInfo.Pods() {
			if err := processPod(existingPod); err != nil {
				pm.setError(err)
			}
		}
	} else {
		// The pod doesn't have any constraints - we need to check only existing
		// ones that have some.
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			if err := processPod(existingPod); err != nil {
				pm.setError(err)
			}
		}
	}

	if len(pm.counts[pm.nameToIndex[node.Name]].Host) <= 0 {
		pm.counts[pm.nameToIndex[node.Name]].Host = node.Name
	}

	return &(pm.counts[pm.nameToIndex[node.Name]]), nil
}

//CalculateInterPodAffinityPriorityReduce calculate each node final score
func (ipa *InterPodAffinity) CalculateInterPodAffinityPriorityReduce(pod *v1.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {

	// convert the topology key based weights to the node name based weights
	var maxCount float64
	var minCount float64

	// make sure scheduling process use different data place every time
	defer deletePodAffinityCountsData()

	for i := range result {
		if float64(result[i].Score) > maxCount {
			maxCount = float64(result[i].Score)
		}
		if float64(result[i].Score) < minCount {
			minCount = float64(result[i].Score)
		}
	}

	maxPriorityFloat64 := float64(schedulerapi.MaxPriority)
	// calculate final priority score for each node
	for i := range result {
		fScore := float64(0)
		if (maxCount - minCount) > 0 {
			fScore = maxPriorityFloat64 * ((float64(result[i].Score) - minCount) / (maxCount - minCount))
		}

		result[i].Score = int(fScore)
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.Infof("%v -> %v: InterPodAffinityPriority, Score: (%d)", pod.Name, result[i].Host, int(fScore))
		}
	}

	return nil
}
