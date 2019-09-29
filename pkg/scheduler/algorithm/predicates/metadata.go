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
	"math"
	"sync"

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/workqueue"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// PredicateMetadata interface represents anything that can access a predicate metadata.
type PredicateMetadata interface {
	ShallowCopy() PredicateMetadata
	AddPod(addedPod *v1.Pod, node *v1.Node) error
	RemovePod(deletedPod *v1.Pod, node *v1.Node) error
}

// PredicateMetadataProducer is a function that computes predicate metadata for a given pod.
type PredicateMetadataProducer func(pod *v1.Pod, nodeNameToInfo map[string]*schedulernodeinfo.NodeInfo) PredicateMetadata

// AntiAffinityTerm's topology key value used in predicate metadata
type topologyPair struct {
	key   string
	value string
}

type podSet map[*v1.Pod]struct{}

type topologyPairSet map[topologyPair]struct{}

// topologyPairsMaps keeps topologyPairToAntiAffinityPods and antiAffinityPodToTopologyPairs in sync
// as they are the inverse of each others.
type topologyPairsMaps struct {
	topologyPairToPods map[topologyPair]podSet
	podToTopologyPairs map[string]topologyPairSet
}

type criticalPath struct {
	// topologyValue denotes the topology value mapping to topology key.
	topologyValue string
	// matchNum denotes the number of matching pods.
	matchNum int32
}

// CAVEAT: the reason that `[2]criticalPath` can work is based on the implementation of current
// preemption algorithm, in particular the following 2 facts:
// Fact 1: we only preempt pods on the same node, instead of pods on multiple nodes.
// Fact 2: each node is evaluated on a separate copy of the metadata during its preemption cycle.
// If we plan to turn to a more complex algorithm like "arbitrary pods on multiple nodes", this
// structure needs to be revisited.
type criticalPaths [2]criticalPath

func newCriticalPaths() *criticalPaths {
	return &criticalPaths{{matchNum: math.MaxInt32}, {matchNum: math.MaxInt32}}
}

func (paths *criticalPaths) update(tpVal string, num int32) {
	// first verify if `tpVal` exists or not
	i := -1
	if tpVal == paths[0].topologyValue {
		i = 0
	} else if tpVal == paths[1].topologyValue {
		i = 1
	}

	if i >= 0 {
		// `tpVal` exists
		paths[i].matchNum = num
		if paths[0].matchNum > paths[1].matchNum {
			// swap paths[0] and paths[1]
			paths[0], paths[1] = paths[1], paths[0]
		}
	} else {
		// `tpVal` doesn't exist
		if num < paths[0].matchNum {
			// update paths[1] with paths[0]
			paths[1] = paths[0]
			// update paths[0]
			paths[0].topologyValue, paths[0].matchNum = tpVal, num
		} else if num < paths[1].matchNum {
			// update paths[1]
			paths[1].topologyValue, paths[1].matchNum = tpVal, num
		}
	}
}

// evenPodsSpreadMetadata combines tpKeyToCriticalPaths and tpPairToMatchNum
// to represent:
// (1) critical paths where the least pods are matched on each spread constraint.
// (2) number of pods matched on each spread constraint.
type evenPodsSpreadMetadata struct {
	// We record 2 critical paths instead of all critical paths here.
	// criticalPaths[0].matchNum always holds the minimum matching number.
	// criticalPaths[1].matchNum is always greater or equal to criticalPaths[0].matchNum, but
	// it's not guaranteed to be the 2nd minimum match number.
	tpKeyToCriticalPaths map[string]*criticalPaths
	// tpPairToMatchNum is keyed with topologyPair, and valued with the number of matching pods.
	tpPairToMatchNum map[topologyPair]int32
}

type serviceAffinityMetadata struct {
	matchingPodList     []*v1.Pod
	matchingPodServices []*v1.Service
}

func (m *serviceAffinityMetadata) addPod(addedPod *v1.Pod, pod *v1.Pod, node *v1.Node) {
	// If addedPod is in the same namespace as the pod, update the list
	// of matching pods if applicable.
	if m == nil || addedPod.Namespace != pod.Namespace {
		return
	}

	selector := CreateSelectorFromLabels(pod.Labels)
	if selector.Matches(labels.Set(addedPod.Labels)) {
		m.matchingPodList = append(m.matchingPodList, addedPod)
	}
}

func (m *serviceAffinityMetadata) removePod(deletedPod *v1.Pod, node *v1.Node) {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)

	if m == nil ||
		len(m.matchingPodList) == 0 ||
		deletedPod.Namespace != m.matchingPodList[0].Namespace {
		return
	}

	for i, pod := range m.matchingPodList {
		if schedutil.GetPodFullName(pod) == deletedPodFullName {
			m.matchingPodList = append(m.matchingPodList[:i], m.matchingPodList[i+1:]...)
			break
		}
	}
}

func (m *serviceAffinityMetadata) clone() *serviceAffinityMetadata {
	if m == nil {
		return nil
	}

	copy := serviceAffinityMetadata{}

	copy.matchingPodServices = append([]*v1.Service(nil),
		m.matchingPodServices...)
	copy.matchingPodList = append([]*v1.Pod(nil),
		m.matchingPodList...)

	return &copy
}

type podAffinityMetadata struct {
	topologyPairsAntiAffinityPodsMap *topologyPairsMaps
	// A map of topology pairs to a list of Pods that can potentially match
	// the affinity terms of the "pod" and its inverse.
	topologyPairsPotentialAffinityPods *topologyPairsMaps
	// A map of topology pairs to a list of Pods that can potentially match
	// the anti-affinity terms of the "pod" and its inverse.
	topologyPairsPotentialAntiAffinityPods *topologyPairsMaps
}

func (m *podAffinityMetadata) addPod(addedPod *v1.Pod, pod *v1.Pod, node *v1.Node) error {
	// Add matching anti-affinity terms of the addedPod to the map.
	topologyPairsMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, addedPod, node)
	if err != nil {
		return err
	}
	m.topologyPairsAntiAffinityPodsMap.appendMaps(topologyPairsMaps)
	// Add the pod to nodeNameToMatchingAffinityPods and nodeNameToMatchingAntiAffinityPods if needed.
	affinity := pod.Spec.Affinity
	podNodeName := addedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		// It is assumed that when the added pod matches affinity of the pod, all the terms must match,
		// this should be changed when the implementation of targetPodMatchesAffinityOfPod/podMatchesAffinityTermProperties
		// is changed
		if targetPodMatchesAffinityOfPod(pod, addedPod) {
			affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
			for _, term := range affinityTerms {
				if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
					pair := topologyPair{key: term.TopologyKey, value: topologyValue}
					m.topologyPairsPotentialAffinityPods.addTopologyPair(pair, addedPod)
				}
			}
		}
		if targetPodMatchesAntiAffinityOfPod(pod, addedPod) {
			antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)
			for _, term := range antiAffinityTerms {
				if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
					pair := topologyPair{key: term.TopologyKey, value: topologyValue}
					m.topologyPairsPotentialAntiAffinityPods.addTopologyPair(pair, addedPod)
				}
			}
		}
	}

	return nil
}

func (m *podAffinityMetadata) removePod(deletedPod *v1.Pod) {
	if m == nil {
		return
	}

	m.topologyPairsAntiAffinityPodsMap.removePod(deletedPod)
	// Delete pod from the matching affinity or anti-affinity topology pairs maps.
	m.topologyPairsPotentialAffinityPods.removePod(deletedPod)
	m.topologyPairsPotentialAntiAffinityPods.removePod(deletedPod)
}

func (m *podAffinityMetadata) clone() *podAffinityMetadata {
	if m == nil {
		return nil
	}

	copy := podAffinityMetadata{}
	copy.topologyPairsPotentialAffinityPods = m.topologyPairsPotentialAffinityPods.clone()
	copy.topologyPairsPotentialAntiAffinityPods = m.topologyPairsPotentialAntiAffinityPods.clone()
	copy.topologyPairsAntiAffinityPodsMap = m.topologyPairsAntiAffinityPodsMap.clone()

	return &copy
}

type podFitsResourcesMetadata struct {
	// ignoredExtendedResources is a set of extended resource names that will
	// be ignored in the PodFitsResources predicate.
	//
	// They can be scheduler extender managed resources, the consumption of
	// which should be accounted only by the extenders. This set is synthesized
	// from scheduler extender configuration and does not change per pod.
	ignoredExtendedResources sets.String
	podRequest               *schedulernodeinfo.Resource
}

func (m *podFitsResourcesMetadata) clone() *podFitsResourcesMetadata {
	if m == nil {
		return nil
	}

	copy := podFitsResourcesMetadata{}
	copy.ignoredExtendedResources = m.ignoredExtendedResources
	copy.podRequest = m.podRequest

	return &copy
}

type podFitsHostPortsMetadata struct {
	podPorts []*v1.ContainerPort
}

func (m *podFitsHostPortsMetadata) clone() *podFitsHostPortsMetadata {
	if m == nil {
		return nil
	}

	copy := podFitsHostPortsMetadata{}
	copy.podPorts = append([]*v1.ContainerPort(nil), m.podPorts...)

	return &copy
}

// NOTE: When new fields are added/removed or logic is changed, please make sure that
// RemovePod, AddPod, and ShallowCopy functions are updated to work with the new changes.
type predicateMetadata struct {
	pod           *v1.Pod
	podBestEffort bool

	// evenPodsSpreadMetadata holds info of the minimum match number on each topology spread constraint,
	// and the match number of all valid topology pairs.
	evenPodsSpreadMetadata *evenPodsSpreadMetadata

	serviceAffinityMetadata  *serviceAffinityMetadata
	podAffinityMetadata      *podAffinityMetadata
	podFitsResourcesMetadata *podFitsResourcesMetadata
	podFitsHostPortsMetadata *podFitsHostPortsMetadata
}

// Ensure that predicateMetadata implements algorithm.PredicateMetadata.
var _ PredicateMetadata = &predicateMetadata{}

// predicateMetadataProducer function produces predicate metadata. It is stored in a global variable below
// and used to modify the return values of PredicateMetadataProducer
type predicateMetadataProducer func(pm *predicateMetadata)

var predicateMetadataProducers = make(map[string]predicateMetadataProducer)

// RegisterPredicateMetadataProducer registers a PredicateMetadataProducer.
func RegisterPredicateMetadataProducer(predicateName string, precomp predicateMetadataProducer) {
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
		pm.podFitsResourcesMetadata.ignoredExtendedResources = ignoredExtendedResources
	})
}

// GetPredicateMetadata returns the predicateMetadata which will be used by various predicates.
func GetPredicateMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulernodeinfo.NodeInfo) PredicateMetadata {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	// evenPodsSpreadMetadata represents how existing pods match "pod"
	// on its spread constraints
	evenPodsSpreadMetadata, err := getEvenPodsSpreadMetadata(pod, nodeNameToInfoMap)
	if err != nil {
		klog.Errorf("Error calculating spreadConstraintsMap: %v", err)
		return nil
	}

	podAffinityMetadata, err := getPodAffinityMetadata(pod, nodeNameToInfoMap)
	if err != nil {
		klog.Errorf("Error calculating podAffinityMetadata: %v", err)
		return nil
	}

	predicateMetadata := &predicateMetadata{
		pod:                      pod,
		podBestEffort:            isPodBestEffort(pod),
		evenPodsSpreadMetadata:   evenPodsSpreadMetadata,
		podAffinityMetadata:      podAffinityMetadata,
		podFitsResourcesMetadata: getPodFitsResourcesMetedata(pod),
		podFitsHostPortsMetadata: getPodFitsHostPortsMetadata(pod),
	}
	for predicateName, precomputeFunc := range predicateMetadataProducers {
		klog.V(10).Infof("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

func getPodFitsHostPortsMetadata(pod *v1.Pod) *podFitsHostPortsMetadata {
	return &podFitsHostPortsMetadata{
		podPorts: schedutil.GetContainerPorts(pod),
	}
}

func getPodFitsResourcesMetedata(pod *v1.Pod) *podFitsResourcesMetadata {
	return &podFitsResourcesMetadata{
		podRequest: GetResourceRequest(pod),
	}
}

func getPodAffinityMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulernodeinfo.NodeInfo) (*podAffinityMetadata, error) {
	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap, err := getTPMapMatchingExistingAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		return nil, err
	}
	// incomingPodAffinityMap will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	incomingPodAffinityMap, incomingPodAntiAffinityMap, err := getTPMapMatchingIncomingAffinityAntiAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		return nil, err
	}

	return &podAffinityMetadata{
		topologyPairsPotentialAffinityPods:     incomingPodAffinityMap,
		topologyPairsPotentialAntiAffinityPods: incomingPodAntiAffinityMap,
		topologyPairsAntiAffinityPodsMap:       existingPodAntiAffinityMap,
	}, nil
}

func getEvenPodsSpreadMetadata(pod *v1.Pod, nodeInfoMap map[string]*schedulernodeinfo.NodeInfo) (*evenPodsSpreadMetadata, error) {
	// We have feature gating in APIServer to strip the spec
	// so don't need to re-check feature gate, just check length of constraints.
	constraints := getHardTopologySpreadConstraints(pod)
	if len(constraints) == 0 {
		return nil, nil
	}

	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	errCh := schedutil.NewErrorChannel()
	var lock sync.Mutex

	// TODO(Huang-Wei): It might be possible to use "make(map[topologyPair]*int32)".
	// In that case, need to consider how to init each tpPairToCount[pair] in an atomic fashion.
	m := evenPodsSpreadMetadata{
		tpKeyToCriticalPaths: make(map[string]*criticalPaths, len(constraints)),
		tpPairToMatchNum:     make(map[topologyPair]int32),
	}
	addTopologyPairMatchNum := func(pair topologyPair, num int32) {
		lock.Lock()
		m.tpPairToMatchNum[pair] += num
		lock.Unlock()
	}

	ctx, cancel := context.WithCancel(context.Background())

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			klog.Errorf("node %q not found", allNodeNames[i])
			return
		}
		// In accordance to design, if NodeAffinity or NodeSelector is defined,
		// spreading is applied to nodes that pass those filters.
		if !PodMatchesNodeSelectorAndAffinityTerms(pod, node) {
			return
		}

		// Ensure current node's labels contains all topologyKeys in 'constraints'.
		if !NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
			return
		}
		for _, constraint := range constraints {
			matchTotal := int32(0)
			// nodeInfo.Pods() can be empty; or all pods don't fit
			for _, existingPod := range nodeInfo.Pods() {
				if existingPod.Namespace != pod.Namespace {
					continue
				}
				ok, err := PodMatchesSpreadConstraint(existingPod.Labels, constraint)
				if err != nil {
					errCh.SendErrorWithCancel(err, cancel)
					return
				}
				if ok {
					matchTotal++
				}
			}
			pair := topologyPair{key: constraint.TopologyKey, value: node.Labels[constraint.TopologyKey]}
			addTopologyPairMatchNum(pair, matchTotal)
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processNode)

	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	// calculate min match for each topology pair
	for i := 0; i < len(constraints); i++ {
		key := constraints[i].TopologyKey
		m.tpKeyToCriticalPaths[key] = newCriticalPaths()
	}
	for pair, num := range m.tpPairToMatchNum {
		m.tpKeyToCriticalPaths[pair.key].update(pair.value, num)
	}

	return &m, nil
}

func getHardTopologySpreadConstraints(pod *v1.Pod) (constraints []v1.TopologySpreadConstraint) {
	if pod != nil {
		for _, constraint := range pod.Spec.TopologySpreadConstraints {
			if constraint.WhenUnsatisfiable == v1.DoNotSchedule {
				constraints = append(constraints, constraint)
			}
		}
	}
	return
}

// PodMatchesSpreadConstraint verifies if <constraint.LabelSelector> matches <podLabelSet>.
// Some corner cases:
// 1. podLabelSet = nil => returns (false, nil)
// 2. constraint.LabelSelector = nil => returns (false, nil)
func PodMatchesSpreadConstraint(podLabelSet labels.Set, constraint v1.TopologySpreadConstraint) (bool, error) {
	selector, err := metav1.LabelSelectorAsSelector(constraint.LabelSelector)
	if err != nil {
		return false, err
	}
	if !selector.Matches(podLabelSet) {
		return false, nil
	}
	return true, nil
}

// NodeLabelsMatchSpreadConstraints checks if ALL topology keys in spread constraints are present in node labels.
func NodeLabelsMatchSpreadConstraints(nodeLabels map[string]string, constraints []v1.TopologySpreadConstraint) bool {
	for _, constraint := range constraints {
		if _, ok := nodeLabels[constraint.TopologyKey]; !ok {
			return false
		}
	}
	return true
}

// returns a pointer to a new topologyPairsMaps
func newTopologyPairsMaps() *topologyPairsMaps {
	return &topologyPairsMaps{
		topologyPairToPods: make(map[topologyPair]podSet),
		podToTopologyPairs: make(map[string]topologyPairSet),
	}
}

func (m *topologyPairsMaps) addTopologyPair(pair topologyPair, pod *v1.Pod) {
	podFullName := schedutil.GetPodFullName(pod)
	if m.topologyPairToPods[pair] == nil {
		m.topologyPairToPods[pair] = make(map[*v1.Pod]struct{})
	}
	m.topologyPairToPods[pair][pod] = struct{}{}
	if m.podToTopologyPairs[podFullName] == nil {
		m.podToTopologyPairs[podFullName] = make(map[topologyPair]struct{})
	}
	m.podToTopologyPairs[podFullName][pair] = struct{}{}
}

func (m *topologyPairsMaps) removePod(deletedPod *v1.Pod) {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	for pair := range m.podToTopologyPairs[deletedPodFullName] {
		delete(m.topologyPairToPods[pair], deletedPod)
		if len(m.topologyPairToPods[pair]) == 0 {
			delete(m.topologyPairToPods, pair)
		}
	}
	delete(m.podToTopologyPairs, deletedPodFullName)
}

func (m *topologyPairsMaps) appendMaps(toAppend *topologyPairsMaps) {
	if toAppend == nil {
		return
	}
	for pair := range toAppend.topologyPairToPods {
		for pod := range toAppend.topologyPairToPods[pair] {
			m.addTopologyPair(pair, pod)
		}
	}
}

func (m *topologyPairsMaps) clone() *topologyPairsMaps {
	copy := newTopologyPairsMaps()
	copy.appendMaps(m)
	return copy
}

func (c *evenPodsSpreadMetadata) addPod(addedPod, preemptorPod *v1.Pod, node *v1.Node) error {
	return c.updatePod(addedPod, preemptorPod, node, 1)
}

func (c *evenPodsSpreadMetadata) removePod(deletedPod, preemptorPod *v1.Pod, node *v1.Node) error {
	return c.updatePod(deletedPod, preemptorPod, node, -1)
}

func (c *evenPodsSpreadMetadata) updatePod(updatedPod, preemptorPod *v1.Pod, node *v1.Node, delta int32) error {
	if updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return nil
	}
	constraints := getHardTopologySpreadConstraints(preemptorPod)
	if !NodeLabelsMatchSpreadConstraints(node.Labels, constraints) {
		return nil
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range constraints {
		if match, err := PodMatchesSpreadConstraint(podLabelSet, constraint); err != nil {
			return err
		} else if !match {
			continue
		}

		k, v := constraint.TopologyKey, node.Labels[constraint.TopologyKey]
		pair := topologyPair{key: k, value: v}
		c.tpPairToMatchNum[pair] = c.tpPairToMatchNum[pair] + delta

		c.tpKeyToCriticalPaths[k].update(v, c.tpPairToMatchNum[pair])
	}
	return nil
}

func (c *evenPodsSpreadMetadata) clone() *evenPodsSpreadMetadata {
	// c could be nil when EvenPodsSpread feature is disabled
	if c == nil {
		return nil
	}
	copy := evenPodsSpreadMetadata{
		tpKeyToCriticalPaths: make(map[string]*criticalPaths),
		tpPairToMatchNum:     make(map[topologyPair]int32),
	}
	for tpKey, paths := range c.tpKeyToCriticalPaths {
		copy.tpKeyToCriticalPaths[tpKey] = &criticalPaths{paths[0], paths[1]}
	}
	for tpPair, matchNum := range c.tpPairToMatchNum {
		copyPair := topologyPair{key: tpPair.key, value: tpPair.value}
		copy.tpPairToMatchNum[copyPair] = matchNum
	}
	return &copy
}

// RemovePod changes predicateMetadata assuming that the given `deletedPod` is
// deleted from the system.
func (meta *predicateMetadata) RemovePod(deletedPod *v1.Pod, node *v1.Node) error {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	if deletedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("deletedPod and meta.pod must not be the same")
	}
	meta.podAffinityMetadata.removePod(deletedPod)
	// Delete pod from the pod spread topology maps.
	if err := meta.evenPodsSpreadMetadata.removePod(deletedPod, meta.pod, node); err != nil {
		return err
	}
	meta.serviceAffinityMetadata.removePod(deletedPod, node)

	return nil
}

// AddPod changes predicateMetadata assuming that the given `addedPod` is added to the
// system.
func (meta *predicateMetadata) AddPod(addedPod *v1.Pod, node *v1.Node) error {
	addedPodFullName := schedutil.GetPodFullName(addedPod)
	if addedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("addedPod and meta.pod must not be the same")
	}
	if node == nil {
		return fmt.Errorf("node not found")
	}

	if err := meta.podAffinityMetadata.addPod(addedPod, meta.pod, node); err != nil {
		return err
	}
	// Update meta.evenPodsSpreadMetadata if meta.pod has hard spread constraints
	// and addedPod matches that
	if err := meta.evenPodsSpreadMetadata.addPod(addedPod, meta.pod, node); err != nil {
		return err
	}

	meta.serviceAffinityMetadata.addPod(addedPod, meta.pod, node)

	return nil
}

// ShallowCopy copies a metadata struct into a new struct and creates a copy of
// its maps and slices, but it does not copy the contents of pointer values.
func (meta *predicateMetadata) ShallowCopy() PredicateMetadata {
	newPredMeta := &predicateMetadata{
		pod:           meta.pod,
		podBestEffort: meta.podBestEffort,
	}
	newPredMeta.podFitsHostPortsMetadata = meta.podFitsHostPortsMetadata.clone()
	newPredMeta.podAffinityMetadata = meta.podAffinityMetadata.clone()
	newPredMeta.evenPodsSpreadMetadata = meta.evenPodsSpreadMetadata.clone()
	newPredMeta.serviceAffinityMetadata = meta.serviceAffinityMetadata.clone()
	newPredMeta.podFitsResourcesMetadata = meta.podFitsResourcesMetadata.clone()
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

	errCh := schedutil.NewErrorChannel()
	var lock sync.Mutex
	topologyMaps := newTopologyPairsMaps()

	appendTopologyPairsMaps := func(toAppend *topologyPairsMaps) {
		lock.Lock()
		defer lock.Unlock()
		topologyMaps.appendMaps(toAppend)
	}

	ctx, cancel := context.WithCancel(context.Background())

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			klog.Errorf("node %q not found", allNodeNames[i])
			return
		}
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			existingPodTopologyMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, existingPod, node)
			if err != nil {
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
			appendTopologyPairsMaps(existingPodTopologyMaps)
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processNode)

	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	return topologyMaps, nil
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns a topologyPairsMaps that are checked later by the affinity
// predicate. With this topologyPairsMaps available, the affinity predicate does not
// need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulernodeinfo.NodeInfo) (topologyPairsAffinityPodsMaps *topologyPairsMaps, topologyPairsAntiAffinityPodsMaps *topologyPairsMaps, err error) {
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return newTopologyPairsMaps(), newTopologyPairsMaps(), nil
	}

	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	errCh := schedutil.NewErrorChannel()

	var lock sync.Mutex
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

	affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
	affinityProperties, err := getAffinityTermProperties(pod, affinityTerms)
	if err != nil {
		return nil, nil, err
	}

	antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)

	ctx, cancel := context.WithCancel(context.Background())

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			klog.Errorf("node %q not found", allNodeNames[i])
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
					errCh.SendErrorWithCancel(err, cancel)
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
	workqueue.ParallelizeUntil(ctx, 16, len(allNodeNames), processNode)

	if err := errCh.ReceiveError(); err != nil {
		return nil, nil, err
	}

	return topologyPairsAffinityPodsMaps, topologyPairsAntiAffinityPodsMaps, nil
}

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
