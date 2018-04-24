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
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
)

// PriorityMetadataFactory is a factory to produce PriorityMetadata.
type PriorityMetadataFactory struct {
	serviceLister         algorithm.ServiceLister
	controllerLister      algorithm.ControllerLister
	replicaSetLister      algorithm.ReplicaSetLister
	statefulSetLister     algorithm.StatefulSetLister
	info                  predicates.NodeInfo
	nodeLister            algorithm.NodeLister
	podLister             algorithm.PodLister
	hardPodAffinityWeight int32
}

// NewPriorityMetadataFactory creates a PriorityMetadataFactory.
func NewPriorityMetadataFactory(serviceLister algorithm.ServiceLister, controllerLister algorithm.ControllerLister, replicaSetLister algorithm.ReplicaSetLister, statefulSetLister algorithm.StatefulSetLister,
	info predicates.NodeInfo,
	nodeLister algorithm.NodeLister,
	podLister algorithm.PodLister,
	hardPodAffinityWeight int32) algorithm.PriorityMetadataProducer {
	factory := &PriorityMetadataFactory{
		serviceLister:         serviceLister,
		controllerLister:      controllerLister,
		replicaSetLister:      replicaSetLister,
		statefulSetLister:     statefulSetLister,
		info:                  info,
		nodeLister:            nodeLister,
		podLister:             podLister,
		hardPodAffinityWeight: hardPodAffinityWeight,
	}
	return factory.PriorityMetadata
}

// priorityMetadata is a type that is passed as metadata for priority functions
type priorityMetadata struct {
	nonZeroRequest                  *schedulercache.Resource
	podTolerations                  []v1.Toleration
	affinity                        *v1.Affinity
	podSelectors                    []labels.Selector
	controllerRef                   *metav1.OwnerReference
	podFirstServiceSelector         labels.Selector
	interpodAffinityHostPriorityMap map[string]schedulerapi.HostPriority
}

// PriorityMetadata is a PriorityMetadataProducer.  Node info can be nil.
func (pmf *PriorityMetadataFactory) PriorityMetadata(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) interface{} {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	hpMap, err := pmf.calculateInterPodAffinityPriority(pod, nodeNameToInfo, nodes)
	if err != nil {
		return nil
	}

	return &priorityMetadata{
		nonZeroRequest:                  getNonZeroRequests(pod),
		podTolerations:                  getAllTolerationPreferNoSchedule(pod.Spec.Tolerations),
		affinity:                        pod.Spec.Affinity,
		podSelectors:                    getSelectors(pod, pmf.serviceLister, pmf.controllerLister, pmf.replicaSetLister, pmf.statefulSetLister),
		controllerRef:                   priorityutil.GetControllerRef(pod),
		podFirstServiceSelector:         getFirstServiceSelector(pod, pmf.serviceLister),
		interpodAffinityHostPriorityMap: hpMap,
	}
}

// CalculateInterPodAffinityPriority compute a sum by iterating through the elements of weightedPodAffinityTerm and adding
// "weight" to the sum if the corresponding PodAffinityTerm is satisfied for
// that node; the node(s) with the highest sum are the most preferred.
// Symmetry need to be considered for preferredDuringSchedulingIgnoredDuringExecution from podAffinity & podAntiAffinity,
// symmetry need to be considered for hard requirements from podAffinity
func (pmf *PriorityMetadataFactory) calculateInterPodAffinityPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (map[string]schedulerapi.HostPriority, error) {
	if nodes == nil || len(nodes) == 0 {
		return nil, nil
	}
	affinity := pod.Spec.Affinity
	hasAffinityConstraints := affinity != nil && affinity.PodAffinity != nil
	hasAntiAffinityConstraints := affinity != nil && affinity.PodAntiAffinity != nil

	allNodeNames := make([]string, 0, len(nodeNameToInfo))
	for name := range nodeNameToInfo {
		allNodeNames = append(allNodeNames, name)
	}

	// convert the topology key based weights to the node name based weights
	var maxCount float64
	var minCount float64
	// priorityMap stores the mapping from node name to so-far computed score of
	// the node.
	pm := newPodAffinityPriorityMap(nodes)

	processPod := func(existingPod *v1.Pod) error {
		existingPodNode, err := pmf.info.GetNodeInfo(existingPod.Spec.NodeName)
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
			if pmf.hardPodAffinityWeight > 0 {
				terms := existingPodAffinity.PodAffinity.RequiredDuringSchedulingIgnoredDuringExecution
				// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
				//if len(existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
				//	terms = append(terms, existingPodAffinity.PodAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
				//}
				for _, term := range terms {
					pm.processTerm(&term, existingPod, pod, existingPodNode, float64(pmf.hardPodAffinityWeight))
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
	processNode := func(i int) {
		nodeInfo := nodeNameToInfo[allNodeNames[i]]
		if nodeInfo.Node() != nil {
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
		}
	}
	workqueue.Parallelize(16, len(allNodeNames), processNode)
	if pm.firstError != nil {
		return nil, pm.firstError
	}

	for _, node := range nodes {
		if pm.counts[node.Name] > maxCount {
			maxCount = pm.counts[node.Name]
		}
		if pm.counts[node.Name] < minCount {
			minCount = pm.counts[node.Name]
		}
	}

	// calculate final priority score for each node
	result := make(map[string]schedulerapi.HostPriority)
	for _, node := range nodes {
		fScore := float64(0)
		if (maxCount - minCount) > 0 {
			fScore = float64(schedulerapi.MaxPriority) * ((pm.counts[node.Name] - minCount) / (maxCount - minCount))
		}

		result[node.Name] = schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)}
		if glog.V(10) {
			glog.Infof("%v -> %v: InterPodAffinityPriority, Score: (%d)", pod.Name, node.Name, int(fScore))
		}
	}
	return result, nil
}

type podAffinityPriorityMap struct {
	sync.Mutex

	// nodes contain all nodes that should be considered
	nodes []*v1.Node
	// counts store the mapping from node name to so-far computed score of
	// the node.
	counts map[string]float64
	// The first error that we faced.
	firstError error
}

func newPodAffinityPriorityMap(nodes []*v1.Node) *podAffinityPriorityMap {
	return &podAffinityPriorityMap{
		nodes:  nodes,
		counts: make(map[string]float64, len(nodes)),
	}
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
			for _, node := range p.nodes {
				if priorityutil.NodesHaveSameTopologyKey(node, fixedNode, term.TopologyKey) {
					p.counts[node.Name] += weight
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

// getFirstServiceSelector returns one selector of services the given pod.
func getFirstServiceSelector(pod *v1.Pod, sl algorithm.ServiceLister) (firstServiceSelector labels.Selector) {
	if services, err := sl.GetPodServices(pod); err == nil && len(services) > 0 {
		return labels.SelectorFromSet(services[0].Spec.Selector)
	}
	return nil
}

// getSelectors returns selectors of services, RCs and RSs matching the given pod.
func getSelectors(pod *v1.Pod, sl algorithm.ServiceLister, cl algorithm.ControllerLister, rsl algorithm.ReplicaSetLister, ssl algorithm.StatefulSetLister) []labels.Selector {
	var selectors []labels.Selector

	if services, err := sl.GetPodServices(pod); err == nil {
		for _, service := range services {
			selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
		}
	}

	if rcs, err := cl.GetPodControllers(pod); err == nil {
		for _, rc := range rcs {
			selectors = append(selectors, labels.SelectorFromSet(rc.Spec.Selector))
		}
	}

	if rss, err := rsl.GetPodReplicaSets(pod); err == nil {
		for _, rs := range rss {
			if selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
				selectors = append(selectors, selector)
			}
		}
	}

	if sss, err := ssl.GetPodStatefulSets(pod); err == nil {
		for _, ss := range sss {
			if selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector); err == nil {
				selectors = append(selectors, selector)
			}
		}
	}

	return selectors
}
