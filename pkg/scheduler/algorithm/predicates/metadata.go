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

package predicates

import (
	"fmt"
	"sync"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// PredicateMetadataFactory defines a factory of predicate metadata.
type PredicateMetadataFactory struct {
	podLister algorithm.PodLister
}

// AntiAffinityTerm's topology key value used in predicate metadata
type topologyPair struct {
	key   string
	value string
}

//  Note that predicateMetadata and matchingPodAntiAffinityTerm need to be declared in the same file
//  due to the way declarations are processed in predicate declaration unit tests.
type matchingPodAntiAffinityTerm struct {
	term *v1.PodAffinityTerm
	node *v1.Node
}

type podSet map[*v1.Pod]struct{}

type topologyPairSet map[topologyPair]struct{}

// topologyPairsMaps keeps topologyPairToAntiAffinityPods and antiAffinityPodToTopologyPairs in sync
// as they are the inverse of each others.
type topologyPairsMaps struct {
	topologyPairToPods map[topologyPair]podSet
	podToTopologyPairs map[string]topologyPairSet
}

// NOTE: When new fields are added/removed or logic is changed, please make sure that
// RemovePod, AddPod, and ShallowCopy functions are updated to work with the new changes.
type predicateMetadata struct {
	pod           *v1.Pod
	podBestEffort bool
	podRequest    *schedulercache.Resource
	podPorts      []*v1.ContainerPort

	topologyPairsAntiAffinityPodsMap *topologyPairsMaps
	// A map of topology pairs to a list of Pods that can potentially match
	// the affinity terms of the "pod" and its inverse.
	topologyPairsPotentialAffinityPods *topologyPairsMaps
	// A map of topology pairs to a list of Pods that can potentially match
	// the anti-affinity terms of the "pod" and its inverse.
	topologyPairsPotentialAntiAffinityPods *topologyPairsMaps
	serviceAffinityInUse                   bool
	serviceAffinityMatchingPodList         []*v1.Pod
	serviceAffinityMatchingPodServices     []*v1.Service
	// ignoredExtendedResources is a set of extended resource names that will
	// be ignored in the PodFitsResources predicate.
	//
	// They can be scheduler extender managed resources, the consumption of
	// which should be accounted only by the extenders. This set is synthesized
	// from scheduler extender configuration and does not change per pod.
	ignoredExtendedResources sets.String
}

// Ensure that predicateMetadata implements algorithm.PredicateMetadata.
var _ algorithm.PredicateMetadata = &predicateMetadata{}

// PredicateMetadataProducer function produces predicate metadata.
type PredicateMetadataProducer func(pm *predicateMetadata)

var predicateMetaProducerRegisterLock sync.Mutex
var predicateMetadataProducers = make(map[string]PredicateMetadataProducer)

// RegisterPredicateMetadataProducer registers a PredicateMetadataProducer.
func RegisterPredicateMetadataProducer(predicateName string, precomp PredicateMetadataProducer) {
	predicateMetaProducerRegisterLock.Lock()
	defer predicateMetaProducerRegisterLock.Unlock()
	predicateMetadataProducers[predicateName] = precomp
}

// RegisterPredicateMetadataProducerWithExtendedResourceOptions registers a
// PredicateMetadataProducer that creates predicate metadata with the provided
// options for extended resources.
//
// See the comments in "predicateMetadata" for the explanation of the options.
func RegisterPredicateMetadataProducerWithExtendedResourceOptions(ignoredExtendedResources sets.String) {
	RegisterPredicateMetadataProducer("PredicateWithExtendedResourceOptions", func(pm *predicateMetadata) {
		pm.ignoredExtendedResources = ignoredExtendedResources
	})
}

// NewPredicateMetadataFactory creates a PredicateMetadataFactory.
func NewPredicateMetadataFactory(podLister algorithm.PodLister) algorithm.PredicateMetadataProducer {
	factory := &PredicateMetadataFactory{
		podLister,
	}
	return factory.GetMetadata
}

// GetMetadata returns the predicateMetadata used which will be used by various predicates.
func (pfactory *PredicateMetadataFactory) GetMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulercache.NodeInfo) algorithm.PredicateMetadata {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap, err := getTPMapMatchingExistingAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		return nil
	}
	// incomingPodAffinityMap will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	incomingPodAffinityMap, incomingPodAntiAffinityMap, err := getTPMapMatchingIncomingAffinityAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		glog.Errorf("[predicate meta data generation] error finding pods that match affinity terms: %v", err)
		return nil
	}
	predicateMetadata := &predicateMetadata{
		pod:           pod,
		podBestEffort: isPodBestEffort(pod),
		podRequest:    GetResourceRequest(pod),
		podPorts:      schedutil.GetContainerPorts(pod),
		topologyPairsPotentialAffinityPods:     incomingPodAffinityMap,
		topologyPairsPotentialAntiAffinityPods: incomingPodAntiAffinityMap,
		topologyPairsAntiAffinityPodsMap:       existingPodAntiAffinityMap,
	}
	for predicateName, precomputeFunc := range predicateMetadataProducers {
		glog.V(10).Infof("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

// returns a pointer to a new topologyPairsMaps
func newTopologyPairsMaps() *topologyPairsMaps {
	return &topologyPairsMaps{topologyPairToPods: make(map[topologyPair]podSet),
		podToTopologyPairs: make(map[string]topologyPairSet)}
}

func (topologyPairsMaps *topologyPairsMaps) addTopologyPair(pair topologyPair, pod *v1.Pod) {
	podFullName := schedutil.GetPodFullName(pod)
	if topologyPairsMaps.topologyPairToPods[pair] == nil {
		topologyPairsMaps.topologyPairToPods[pair] = make(map[*v1.Pod]struct{})
	}
	topologyPairsMaps.topologyPairToPods[pair][pod] = struct{}{}
	if topologyPairsMaps.podToTopologyPairs[podFullName] == nil {
		topologyPairsMaps.podToTopologyPairs[podFullName] = make(map[topologyPair]struct{})
	}
	topologyPairsMaps.podToTopologyPairs[podFullName][pair] = struct{}{}
}

func (topologyPairsMaps *topologyPairsMaps) removePod(deletedPod *v1.Pod) {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	for pair := range topologyPairsMaps.podToTopologyPairs[deletedPodFullName] {
		delete(topologyPairsMaps.topologyPairToPods[pair], deletedPod)
		if len(topologyPairsMaps.topologyPairToPods[pair]) == 0 {
			delete(topologyPairsMaps.topologyPairToPods, pair)
		}
	}
	delete(topologyPairsMaps.podToTopologyPairs, deletedPodFullName)
}

func (topologyPairsMaps *topologyPairsMaps) appendMaps(toAppend *topologyPairsMaps) {
	if toAppend == nil {
		return
	}
	for pair := range toAppend.topologyPairToPods {
		for pod := range toAppend.topologyPairToPods[pair] {
			topologyPairsMaps.addTopologyPair(pair, pod)
		}
	}
}

// RemovePod changes predicateMetadata assuming that the given `deletedPod` is
// deleted from the system.
func (meta *predicateMetadata) RemovePod(deletedPod *v1.Pod) error {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	if deletedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("deletedPod and meta.pod must not be the same")
	}
	meta.topologyPairsAntiAffinityPodsMap.removePod(deletedPod)
	// Delete pod from the matching affinity or anti-affinity topology pairs maps.
	meta.topologyPairsPotentialAffinityPods.removePod(deletedPod)
	meta.topologyPairsPotentialAntiAffinityPods.removePod(deletedPod)
	// All pods in the serviceAffinityMatchingPodList are in the same namespace.
	// So, if the namespace of the first one is not the same as the namespace of the
	// deletedPod, we don't need to check the list, as deletedPod isn't in the list.
	if meta.serviceAffinityInUse &&
		len(meta.serviceAffinityMatchingPodList) > 0 &&
		deletedPod.Namespace == meta.serviceAffinityMatchingPodList[0].Namespace {
		for i, pod := range meta.serviceAffinityMatchingPodList {
			if schedutil.GetPodFullName(pod) == deletedPodFullName {
				meta.serviceAffinityMatchingPodList = append(
					meta.serviceAffinityMatchingPodList[:i],
					meta.serviceAffinityMatchingPodList[i+1:]...)
				break
			}
		}
	}
	return nil
}

// AddPod changes predicateMetadata assuming that `newPod` is added to the
// system.
func (meta *predicateMetadata) AddPod(addedPod *v1.Pod, nodeInfo *schedulercache.NodeInfo) error {
	addedPodFullName := schedutil.GetPodFullName(addedPod)
	if addedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("addedPod and meta.pod must not be the same")
	}
	if nodeInfo.Node() == nil {
		return fmt.Errorf("invalid node in nodeInfo")
	}
	// Add matching anti-affinity terms of the addedPod to the map.
	topologyPairsMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(meta.pod, addedPod, nodeInfo.Node())
	if err != nil {
		return err
	}
	meta.topologyPairsAntiAffinityPodsMap.appendMaps(topologyPairsMaps)
	// Add the pod to nodeNameToMatchingAffinityPods and nodeNameToMatchingAntiAffinityPods if needed.
	affinity := meta.pod.Spec.Affinity
	podNodeName := addedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		podNode := nodeInfo.Node()
		// It is assumed that when the added pod matches affinity of the meta.pod, all the terms must match,
		// this should be changed when the implementation of targetPodMatchesAffinityOfPod/podMatchesAffinityTermProperties
		// is changed
		if targetPodMatchesAffinityOfPod(meta.pod, addedPod) {
			affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
			for _, term := range affinityTerms {
				if topologyValue, ok := podNode.Labels[term.TopologyKey]; ok {
					pair := topologyPair{key: term.TopologyKey, value: topologyValue}
					meta.topologyPairsPotentialAffinityPods.addTopologyPair(pair, addedPod)
				}
			}
		}
		if targetPodMatchesAntiAffinityOfPod(meta.pod, addedPod) {
			antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)
			for _, term := range antiAffinityTerms {
				if topologyValue, ok := podNode.Labels[term.TopologyKey]; ok {
					pair := topologyPair{key: term.TopologyKey, value: topologyValue}
					meta.topologyPairsPotentialAntiAffinityPods.addTopologyPair(pair, addedPod)
				}
			}
		}
	}
	// If addedPod is in the same namespace as the meta.pod, update the list
	// of matching pods if applicable.
	if meta.serviceAffinityInUse && addedPod.Namespace == meta.pod.Namespace {
		selector := CreateSelectorFromLabels(meta.pod.Labels)
		if selector.Matches(labels.Set(addedPod.Labels)) {
			meta.serviceAffinityMatchingPodList = append(meta.serviceAffinityMatchingPodList,
				addedPod)
		}
	}
	return nil
}

// ShallowCopy copies a metadata struct into a new struct and creates a copy of
// its maps and slices, but it does not copy the contents of pointer values.
func (meta *predicateMetadata) ShallowCopy() algorithm.PredicateMetadata {
	newPredMeta := &predicateMetadata{
		pod:                      meta.pod,
		podBestEffort:            meta.podBestEffort,
		podRequest:               meta.podRequest,
		serviceAffinityInUse:     meta.serviceAffinityInUse,
		ignoredExtendedResources: meta.ignoredExtendedResources,
	}
	newPredMeta.podPorts = append([]*v1.ContainerPort(nil), meta.podPorts...)
	newPredMeta.topologyPairsPotentialAffinityPods = newTopologyPairsMaps()
	newPredMeta.topologyPairsPotentialAffinityPods.appendMaps(meta.topologyPairsPotentialAffinityPods)
	newPredMeta.topologyPairsPotentialAntiAffinityPods = newTopologyPairsMaps()
	newPredMeta.topologyPairsPotentialAntiAffinityPods.appendMaps(meta.topologyPairsPotentialAntiAffinityPods)
	newPredMeta.topologyPairsAntiAffinityPodsMap = newTopologyPairsMaps()
	newPredMeta.topologyPairsAntiAffinityPodsMap.appendMaps(meta.topologyPairsAntiAffinityPodsMap)
	newPredMeta.serviceAffinityMatchingPodServices = append([]*v1.Service(nil),
		meta.serviceAffinityMatchingPodServices...)
	newPredMeta.serviceAffinityMatchingPodList = append([]*v1.Pod(nil),
		meta.serviceAffinityMatchingPodList...)
	return (algorithm.PredicateMetadata)(newPredMeta)
}

type affinityTermProperties struct {
	namespaces sets.String
	selector   labels.Selector
}

// getAffinityTermProperties receives a Pod and affinity terms and returns the namespaces and
// selectors of the terms.
func getAffinityTermProperties(pod *v1.Pod, terms []v1.PodAffinityTerm) (properties []*affinityTermProperties, err error) {
	if terms == nil {
		return properties, nil
	}

	for _, term := range terms {
		namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(pod, &term)
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return nil, err
		}
		properties = append(properties, &affinityTermProperties{namespaces: namespaces, selector: selector})
	}
	return properties, nil
}

// podMatchesAllAffinityTermProperties returns true IFF the given pod matches all the given properties.
func podMatchesAllAffinityTermProperties(pod *v1.Pod, properties []*affinityTermProperties) bool {
	if len(properties) == 0 {
		return false
	}
	for _, property := range properties {
		if !priorityutil.PodMatchesTermsNamespaceAndSelector(pod, property.namespaces, property.selector) {
			return false
		}
	}
	return true
}

// podMatchesAnyAffinityTermProperties returns true if the given pod matches any given property.
func podMatchesAnyAffinityTermProperties(pod *v1.Pod, properties []*affinityTermProperties) bool {
	if len(properties) == 0 {
		return false
	}
	for _, property := range properties {
		if priorityutil.PodMatchesTermsNamespaceAndSelector(pod, property.namespaces, property.selector) {
			return true
		}
	}
	return false
}

// getTPMapMatchingExistingAntiAffinity calculates the following for each existing pod on each node:
// (1) Whether it has PodAntiAffinity
// (2) Whether any AffinityTerm matches the incoming pod
func getTPMapMatchingExistingAntiAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulercache.NodeInfo) (*topologyPairsMaps, error) {
	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	var lock sync.Mutex
	var firstError error

	topologyMaps := newTopologyPairsMaps()

	appendTopologyPairsMaps := func(toAppend *topologyPairsMaps) {
		lock.Lock()
		defer lock.Unlock()
		topologyMaps.appendMaps(toAppend)
	}
	catchError := func(err error) {
		lock.Lock()
		defer lock.Unlock()
		if firstError == nil {
			firstError = err
		}
	}

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			catchError(fmt.Errorf("node not found"))
			return
		}
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			existingPodTopologyMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, existingPod, node)
			if err != nil {
				catchError(err)
				return
			}
			appendTopologyPairsMaps(existingPodTopologyMaps)
		}
	}
	workqueue.Parallelize(16, len(allNodeNames), processNode)
	return topologyMaps, firstError
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns a topologyPairsMaps that are checked later by the affinity
// predicate. With this topologyPairsMaps available, the affinity predicate does not
// need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulercache.NodeInfo) (topologyPairsAffinityPodsMaps *topologyPairsMaps, topologyPairsAntiAffinityPodsMaps *topologyPairsMaps, err error) {
	allNodeNames := make([]string, 0, len(nodeInfoMap))

	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return newTopologyPairsMaps(), newTopologyPairsMaps(), nil
	}

	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	var lock sync.Mutex
	var firstError error
	topologyPairsAffinityPodsMaps = newTopologyPairsMaps()
	topologyPairsAntiAffinityPodsMaps = newTopologyPairsMaps()
	appendResult := func(nodeName string, nodeTopologyPairsAffinityPodsMaps, nodeTopologyPairsAntiAffinityPodsMaps *topologyPairsMaps) {
		lock.Lock()
		defer lock.Unlock()
		if len(nodeTopologyPairsAffinityPodsMaps.topologyPairToPods) > 0 {
			topologyPairsAffinityPodsMaps.appendMaps(nodeTopologyPairsAffinityPodsMaps)
		}
		if len(nodeTopologyPairsAntiAffinityPodsMaps.topologyPairToPods) > 0 {
			topologyPairsAntiAffinityPodsMaps.appendMaps(nodeTopologyPairsAntiAffinityPodsMaps)
		}
	}

	catchError := func(err error) {
		lock.Lock()
		defer lock.Unlock()
		if firstError == nil {
			firstError = err
		}
	}

	affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
	affinityProperties, err := getAffinityTermProperties(pod, affinityTerms)
	if err != nil {
		return nil, nil, err
	}
	antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			catchError(fmt.Errorf("nodeInfo.Node is nil"))
			return
		}
		nodeTopologyPairsAffinityPodsMaps := newTopologyPairsMaps()
		nodeTopologyPairsAntiAffinityPodsMaps := newTopologyPairsMaps()
		for _, existingPod := range nodeInfo.Pods() {
			// Check affinity properties.
			if podMatchesAllAffinityTermProperties(existingPod, affinityProperties) {
				for _, term := range affinityTerms {
					if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
						pair := topologyPair{key: term.TopologyKey, value: topologyValue}
						nodeTopologyPairsAffinityPodsMaps.addTopologyPair(pair, existingPod)
					}
				}
			}
			// Check anti-affinity properties.
			for _, term := range antiAffinityTerms {
				namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(pod, &term)
				selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
				if err != nil {
					catchError(err)
					return
				}
				if priorityutil.PodMatchesTermsNamespaceAndSelector(existingPod, namespaces, selector) {
					if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
						pair := topologyPair{key: term.TopologyKey, value: topologyValue}
						nodeTopologyPairsAntiAffinityPodsMaps.addTopologyPair(pair, existingPod)
					}
				}
			}
		}
		if len(nodeTopologyPairsAffinityPodsMaps.topologyPairToPods) > 0 || len(nodeTopologyPairsAntiAffinityPodsMaps.topologyPairToPods) > 0 {
			appendResult(node.Name, nodeTopologyPairsAffinityPodsMaps, nodeTopologyPairsAntiAffinityPodsMaps)
		}
	}
	workqueue.Parallelize(16, len(allNodeNames), processNode)
	return topologyPairsAffinityPodsMaps, topologyPairsAntiAffinityPodsMaps, firstError
}

// targetPodMatchesAffinityOfPod returns true if "targetPod" matches ALL affinity terms of
// "pod". Similar to getPodsMatchingAffinity, this function does not check topology.
// So, whether the targetPod actually matches or not needs further checks for a specific
// node.
func targetPodMatchesAffinityOfPod(pod, targetPod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAffinity == nil {
		return false
	}
	affinityProperties, err := getAffinityTermProperties(pod, GetPodAffinityTerms(affinity.PodAffinity))
	if err != nil {
		glog.Errorf("error in getting affinity properties of Pod %v", pod.Name)
		return false
	}
	return podMatchesAllAffinityTermProperties(targetPod, affinityProperties)
}

// targetPodMatchesAntiAffinityOfPod returns true if "targetPod" matches ANY anti-affinity
// term of "pod". Similar to getPodsMatchingAffinity, this function does not check topology.
// So, whether the targetPod actually matches or not needs further checks for a specific
// node.
func targetPodMatchesAntiAffinityOfPod(pod, targetPod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAntiAffinity == nil {
		return false
	}
	properties, err := getAffinityTermProperties(pod, GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
	if err != nil {
		glog.Errorf("error in getting anti-affinity properties of Pod %v", pod.Name)
		return false
	}
	return podMatchesAnyAffinityTermProperties(targetPod, properties)
}
