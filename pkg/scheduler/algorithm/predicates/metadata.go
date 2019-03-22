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
	"context"
	"fmt"
	"sync"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// PredicateMetadata interface represents anything that can access a predicate metadata.
type PredicateMetadata interface {
	ShallowCopy() PredicateMetadata
	AddPod(addedPod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) error
	RemovePod(deletedPod *v1.Pod) error
}

// PredicateMetadataProducer is a function that computes predicate metadata for a given pod.
type PredicateMetadataProducer func(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo) PredicateMetadata

// PredicateMetadataFactory defines a factory of predicate metadata.
type PredicateMetadataFactory struct {
	podLister    algorithm.PodLister
	topologyInfo internalcache.TopologyInfo
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

// affinityQuery represents how an incoming pod's affinity terms are matched:
// - [region/region1 => matched on which pods, region/region2 => matched on which pods, ...]
// - [zone/zone1 => matched on which pods, zone/zone2 => matched on which pods, ...]
type affinityQuery []internalcache.TopologyInfo

// NOTE: When new fields are added/removed or logic is changed, please make sure that
// RemovePod, AddPod, and ShallowCopy functions are updated to work with the new changes.
type predicateMetadata struct {
	pod           *v1.Pod
	podBestEffort bool
	podRequest    *schedulernodeinfo.Resource
	podPorts      []*v1.ContainerPort

	// Used to represent podAffinity rules
	podAffinityQuery affinityQuery
	// Used to get matching node names by giving a topologyPair
	topologyInfo internalcache.TopologyInfo
	// Node names which represent which nodes are fits for podAffinity
	podAffinityFits sets.String
	// Indicates whether "podAffinityFits" is up-to-dated;
	podAffinityFitsUpdated     bool
	podAffinityFitsUpdatedLock sync.Mutex
	// A map of topology pairs to a list of Pods that can potentially match
	// the anti-affinity terms of the "pod" and its inverse.
	topologyPairsPotentialAntiAffinityPods *topologyPairsMaps
	topologyPairsAntiAffinityPodsMap       *topologyPairsMaps
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
var _ PredicateMetadata = &predicateMetadata{}

// predicateMetadataProducer function produces predicate metadata. It is stored in a global variable below
// and used to modify the return values of PredicateMetadataProducer
type predicateMetadataProducer func(pm *predicateMetadata)

var predicateMetaProducerRegisterLock sync.Mutex
var predicateMetadataProducers = make(map[string]predicateMetadataProducer)

// RegisterPredicateMetadataProducer registers a PredicateMetadataProducer.
func RegisterPredicateMetadataProducer(predicateName string, precomp predicateMetadataProducer) {
	predicateMetaProducerRegisterLock.Lock()
	defer predicateMetaProducerRegisterLock.Unlock()
	predicateMetadataProducers[predicateName] = precomp
}

// EmptyPredicateMetadataProducer returns a no-op MetadataProducer type.
func EmptyPredicateMetadataProducer(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo) PredicateMetadata {
	return nil
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
func NewPredicateMetadataFactory(podLister algorithm.PodLister, topologyInfo internalcache.TopologyInfo) PredicateMetadataProducer {
	factory := &PredicateMetadataFactory{
		podLister,
		topologyInfo,
	}
	return factory.GetMetadata
}

// GetMetadata returns the predicateMetadata used which will be used by various predicates.
func (pfactory *PredicateMetadataFactory) GetMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulernodeinfo.NodeInfo) PredicateMetadata {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap, err := getTPMapMatchingExistingAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		return nil
	}
	// podAffinityQuery will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	podAffinityQuery, incomingPodAntiAffinityMap, err := getTPMapMatchingIncomingAffinityAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		klog.Errorf("[predicate meta data generation] error finding pods that match affinity terms: %v", err)
		return nil
	}
	predicateMetadata := &predicateMetadata{
		pod:                                    pod,
		podBestEffort:                          isPodBestEffort(pod),
		podRequest:                             GetResourceRequest(pod),
		podPorts:                               schedutil.GetContainerPorts(pod),
		podAffinityQuery:                       podAffinityQuery,
		topologyInfo:                           pfactory.topologyInfo,
		topologyPairsPotentialAntiAffinityPods: incomingPodAntiAffinityMap,
		topologyPairsAntiAffinityPodsMap:       existingPodAntiAffinityMap,
	}
	// initialize calculation of podAffinityFits
	predicateMetadata.podAffinityFits = predicateMetadata.getPodAffinityFits()
	for predicateName, precomputeFunc := range predicateMetadataProducers {
		klog.V(10).Infof("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

func newAffinityQuery(length int) affinityQuery {
	query := make(affinityQuery, length)
	for i := 0; i < length; i++ {
		query[i] = make(internalcache.TopologyInfo)
	}
	return query
}

// returns a pointer to a new topologyPairsMaps
func newTopologyPairsMaps() *topologyPairsMaps {
	return &topologyPairsMaps{
		topologyPairToPods: make(map[topologyPair]podSet),
		podToTopologyPairs: make(map[string]topologyPairSet),
	}
}

func (topologyPairsMaps *topologyPairsMaps) addTopologyPair(pair topologyPair, pod *v1.Pod) {
	podFullName := schedutil.GetPodFullName(pod)
	if topologyPairsMaps.topologyPairToPods[pair] == nil {
		topologyPairsMaps.topologyPairToPods[pair] = make(podSet)
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

func (aq affinityQuery) clone() affinityQuery {
	copy := newAffinityQuery(len(aq))
	for i, queryTerm := range aq {
		for pair, podNames := range queryTerm {
			copy[i][pair] = podNames
		}
	}
	return copy
}

func (aq affinityQuery) append(toAppend affinityQuery) {
	for i := range toAppend {
		if len(aq[i]) == 0 {
			aq[i] = toAppend[i]
		} else {
			for topologyPair, podNames := range toAppend[i] {
				if existingPodNames, ok := aq[i][topologyPair]; ok {
					for podName := range podNames {
						existingPodNames[podName] = sets.Empty{}
					}
				} else {
					aq[i][topologyPair] = podNames
				}
			}
		}
	}
}

func (aq affinityQuery) removePod(delPod *v1.Pod) bool {
	var updated bool
	delPodFullName := podName(delPod)
	for _, querySet := range aq {
		for pair, nodeNames := range querySet {
			if _, ok := nodeNames[delPodFullName]; ok {
				updated = true
				delete(nodeNames, delPodFullName)
				if len(nodeNames) == 0 {
					delete(querySet, pair)
				}
			}
		}
	}
	return updated
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
	isQueryUpdated := meta.podAffinityQuery.removePod(deletedPod)
	if isQueryUpdated && len(meta.podAffinityFits) > 0 {
		// set flag instead of re-calculating immediately
		// since preemption dry-run usually deletes a series of pods
		meta.setPodAffinityFitsUpdated(false)
	}
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

func updateAffinityQuery(query affinityQuery, terms []v1.PodAffinityTerm, node *v1.Node,
	incomingPod, existingPod *v1.Pod) (bool, error) {
	var updated bool
	for i, term := range terms {
		namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(incomingPod, &term)
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return updated, err
		}
		if priorityutil.PodMatchesTermsNamespaceAndSelector(existingPod, namespaces, selector) {
			topologyValue, ok := node.Labels[term.TopologyKey]
			if !ok {
				continue
			}
			updated = true
			pair := internalcache.TopologyPair{Key: term.TopologyKey, Value: topologyValue}
			if query[i][pair] == nil {
				query[i][pair] = make(sets.String)
			}
			query[i][pair][podName(existingPod)] = sets.Empty{}
		}
	}
	return updated, nil
}

func updateAntiAffinityInTPM(tpm *topologyPairsMaps, terms []v1.PodAffinityTerm, node *v1.Node,
	incomingPod, existingPod *v1.Pod) error {
	for _, term := range terms {
		namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(incomingPod, &term)
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return err
		}
		if priorityutil.PodMatchesTermsNamespaceAndSelector(existingPod, namespaces, selector) {
			if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
				pair := topologyPair{key: term.TopologyKey, value: topologyValue}
				tpm.addTopologyPair(pair, existingPod)
			}
		}
	}
	return nil
}

func (meta *predicateMetadata) setPodAffinityFitsUpdated(val bool) {
	meta.podAffinityFitsUpdatedLock.Lock()
	meta.podAffinityFitsUpdated = val
	meta.podAffinityFitsUpdatedLock.Unlock()
}

// NOTE: this is an expensive operation. Use it only when it's necessary.
func (meta *predicateMetadata) getPodAffinityFits() sets.String {
	meta.podAffinityFitsUpdatedLock.Lock()
	defer meta.podAffinityFitsUpdatedLock.Unlock()

	if meta.podAffinityFitsUpdated {
		return meta.podAffinityFits
	}

	meta.podAffinityFitsUpdated = true
	pod := meta.pod
	if pod.Spec.Affinity == nil || pod.Spec.Affinity.PodAffinity == nil {
		return nil
	}
	queryLen := len(meta.podAffinityQuery)
	if queryLen == 0 || meta.topologyInfo == nil {
		return nil
	}
	nodeNames := make([]sets.String, queryLen)
	for i := 0; i < queryLen; i++ {
		nodeSets := make([]sets.String, len(meta.podAffinityQuery[i]))
		j := 0
		for pair := range meta.podAffinityQuery[i] {
			if nodeSet, ok := meta.topologyInfo[pair]; ok {
				nodeSets[j] = nodeSet
			} else {
				nodeSets[j] = sets.String{}
			}
			j++
		}
		// nodeNames[i] represents nodes which satisfy on the ith term
		nodeNames[i] = Union(nodeSets)
	}
	// return nodes which satisfy on all terms
	return Intersect(nodeNames)
}

// AddPod changes predicateMetadata assuming that `newPod` is added to the
// system.
func (meta *predicateMetadata) AddPod(addedPod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) error {
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
	// Update meta.podAffinityQuery and meta.topologyPairsPotentialAntiAffinityPods if needed.
	affinity := meta.pod.Spec.Affinity
	podNodeName := addedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		podNode := nodeInfo.Node()

		// Update podAffinityQuery according to affinity properties.
		isQueryUpdated, err := updateAffinityQuery(meta.podAffinityQuery,
			GetPodAffinityTerms(affinity.PodAffinity), podNode, meta.pod, addedPod)
		if err != nil {
			return err
		}
		// set flag instead of do recalculation immediately, as it's halfway of preemption
		if isQueryUpdated {
			meta.setPodAffinityFitsUpdated(false)
		}

		// Update topologyPairsPotentialAntiAffinityPods according to anti-affinity properties.
		if err := updateAntiAffinityInTPM(meta.topologyPairsPotentialAntiAffinityPods,
			GetPodAntiAffinityTerms(affinity.PodAntiAffinity), podNode, meta.pod, addedPod); err != nil {
			return err
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
func (meta *predicateMetadata) ShallowCopy() PredicateMetadata {
	newPredMeta := &predicateMetadata{
		pod:                      meta.pod,
		podBestEffort:            meta.podBestEffort,
		podRequest:               meta.podRequest,
		serviceAffinityInUse:     meta.serviceAffinityInUse,
		ignoredExtendedResources: meta.ignoredExtendedResources,
		topologyInfo:             meta.topologyInfo,
		podAffinityFitsUpdated:   meta.podAffinityFitsUpdated,
	}
	newPredMeta.podPorts = append([]*v1.ContainerPort(nil), meta.podPorts...)
	// deep-copy podAffinityQuery & podAffinityFits as it's used in preemption
	newPredMeta.podAffinityQuery = meta.podAffinityQuery.clone()
	newPredMeta.podAffinityFits = sets.String{}.Union(meta.podAffinityFits)
	newPredMeta.topologyPairsPotentialAntiAffinityPods = newTopologyPairsMaps()
	newPredMeta.topologyPairsPotentialAntiAffinityPods.appendMaps(meta.topologyPairsPotentialAntiAffinityPods)
	newPredMeta.topologyPairsAntiAffinityPodsMap = newTopologyPairsMaps()
	newPredMeta.topologyPairsAntiAffinityPodsMap.appendMaps(meta.topologyPairsAntiAffinityPodsMap)
	newPredMeta.serviceAffinityMatchingPodServices = append([]*v1.Service(nil),
		meta.serviceAffinityMatchingPodServices...)
	newPredMeta.serviceAffinityMatchingPodList = append([]*v1.Pod(nil),
		meta.serviceAffinityMatchingPodList...)
	return (PredicateMetadata)(newPredMeta)
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
func getTPMapMatchingExistingAntiAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulernodeinfo.NodeInfo) (*topologyPairsMaps, error) {
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
	workqueue.ParallelizeUntil(context.TODO(), 16, len(allNodeNames), processNode)
	return topologyMaps, firstError
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns an affinityQuery and topologyPairsMaps that are checked later by the affinity
// predicate. With this affinityQuery and topologyPairsMaps available, the affinity predicate
// does not need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulernodeinfo.NodeInfo) (affinityQuery, *topologyPairsMaps, error) {
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return affinityQuery(nil), newTopologyPairsMaps(), nil
	}

	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}
	affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
	antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)

	var lock sync.Mutex
	var firstError error
	topologyPairsAntiAffinityPodsMaps := newTopologyPairsMaps()
	query := newAffinityQuery(len(affinityTerms))
	appendResult := func(q affinityQuery, nodeName string, nodeTopologyPairsAntiAffinityPodsMaps *topologyPairsMaps) {
		lock.Lock()
		query.append(q)
		if len(nodeTopologyPairsAntiAffinityPodsMaps.topologyPairToPods) > 0 {
			topologyPairsAntiAffinityPodsMaps.appendMaps(nodeTopologyPairsAntiAffinityPodsMaps)
		}
		lock.Unlock()
	}

	catchError := func(err error) {
		lock.Lock()
		if firstError == nil {
			firstError = err
		}
		lock.Unlock()
	}

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			catchError(fmt.Errorf("nodeInfo.Node is nil"))
			return
		}
		nodeTopologyPairsAntiAffinityPodsMaps := newTopologyPairsMaps()
		q := newAffinityQuery(len(affinityTerms))
		updated := false
		for _, existingPod := range nodeInfo.Pods() {
			// Update podAffinityQuery according to affinity properties.
			u, err := updateAffinityQuery(q, affinityTerms, node, pod, existingPod)
			if err != nil {
				catchError(err)
				return
			}
			updated = updated || u

			// Update nodeTopologyPairsAntiAffinityPodsMaps according to anti-affinity properties.
			if err := updateAntiAffinityInTPM(nodeTopologyPairsAntiAffinityPodsMaps,
				antiAffinityTerms, node, pod, existingPod); err != nil {
				catchError(err)
				return
			}
		}
		if updated || len(nodeTopologyPairsAntiAffinityPodsMaps.topologyPairToPods) > 0 {
			appendResult(q, node.Name, nodeTopologyPairsAntiAffinityPodsMaps)
		}
	}
	workqueue.ParallelizeUntil(context.TODO(), 16, len(allNodeNames), processNode)
	return query, topologyPairsAntiAffinityPodsMaps, firstError
}

func podMatchesItsOwnAffinityOnNode(pod *v1.Pod, node *v1.Node) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAffinity == nil {
		return false
	}
	affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
	affinityProperties, err := getAffinityTermProperties(pod, affinityTerms)
	if err != nil {
		klog.Errorf("error in getting affinity properties of Pod %v", pod.Name)
		return false
	}
	podMatchesItself := podMatchesAllAffinityTermProperties(pod, affinityProperties)
	if !podMatchesItself {
		return false
	}
	for _, term := range affinityTerms {
		if _, ok := node.Labels[term.TopologyKey]; !ok {
			return false
		}
	}
	return true
}

// DEPRECATED: use podMatchesItsOwnAffinityOnNode instead
// targetPodMatchesAffinityOfPod returns true if "targetPod" matches ALL affinity terms of
// "pod". This function does not check topology.
// So, whether the targetPod actually matches or not needs further checks for a specific
// node.
func targetPodMatchesAffinityOfPod(pod, targetPod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAffinity == nil {
		return false
	}
	affinityProperties, err := getAffinityTermProperties(pod, GetPodAffinityTerms(affinity.PodAffinity))
	if err != nil {
		klog.Errorf("error in getting affinity properties of Pod %v", pod.Name)
		return false
	}
	return podMatchesAllAffinityTermProperties(targetPod, affinityProperties)
}

// targetPodMatchesAntiAffinityOfPod returns true if "targetPod" matches ANY anti-affinity
// term of "pod". This function does not check topology.
// So, whether the targetPod actually matches or not needs further checks for a specific
// node.
func targetPodMatchesAntiAffinityOfPod(pod, targetPod *v1.Pod) bool {
	affinity := pod.Spec.Affinity
	if affinity == nil || affinity.PodAntiAffinity == nil {
		return false
	}
	properties, err := getAffinityTermProperties(pod, GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
	if err != nil {
		klog.Errorf("error in getting anti-affinity properties of Pod %v", pod.Name)
		return false
	}
	return podMatchesAnyAffinityTermProperties(targetPod, properties)
}
