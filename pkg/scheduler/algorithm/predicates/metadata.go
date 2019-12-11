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
	schedulerlisters "k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
)

// Metadata interface represents anything that can access a predicate metadata.
type Metadata interface {
	ShallowCopy() Metadata
	AddPod(addedPod *v1.Pod, node *v1.Node) error
	RemovePod(deletedPod *v1.Pod, node *v1.Node) error
}

// MetadataProducer is a function that computes predicate metadata for a given pod.
type MetadataProducer func(pod *v1.Pod, sharedLister schedulerlisters.SharedLister) Metadata

// AntiAffinityTerm's topology key value used in predicate metadata
type topologyPair struct {
	key   string
	value string
}

// TODO(Huang-Wei): It might be possible to use "make(map[topologyPair]*int64)" so that
// we can do atomic additions instead of using a global mutext, however we need to consider
// how to init each topologyToMatchedTermCount.
type topologyToMatchedTermCount map[topologyPair]int64

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
	constraints []topologySpreadConstraint
	// We record 2 critical paths instead of all critical paths here.
	// criticalPaths[0].matchNum always holds the minimum matching number.
	// criticalPaths[1].matchNum is always greater or equal to criticalPaths[0].matchNum, but
	// it's not guaranteed to be the 2nd minimum match number.
	tpKeyToCriticalPaths map[string]*criticalPaths
	// tpPairToMatchNum is keyed with topologyPair, and valued with the number of matching pods.
	tpPairToMatchNum map[topologyPair]int32
}

// topologySpreadConstraint is an internal version for a hard (DoNotSchedule
// unsatisfiable constraint action) v1.TopologySpreadConstraint and where the
// selector is parsed.
type topologySpreadConstraint struct {
	maxSkew     int32
	topologyKey string
	selector    labels.Selector
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

// PodAffinityMetadata pre-computed state for inter-pod affinity predicate.
type PodAffinityMetadata struct {
	// A map of topology pairs to the number of existing pods that has anti-affinity terms that match the "pod".
	topologyToMatchedExistingAntiAffinityTerms topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the affinity terms of the "pod".
	topologyToMatchedAffinityTerms topologyToMatchedTermCount
	// A map of topology pairs to the number of existing pods that match the anti-affinity terms of the "pod".
	topologyToMatchedAntiAffinityTerms topologyToMatchedTermCount
}

// updateWithAffinityTerms updates the topologyToMatchedTermCount map with the specified value
// for each affinity term if "targetPod" matches ALL terms.
func (m topologyToMatchedTermCount) updateWithAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, affinityTerms []*affinityTermProperties, value int64) {
	if podMatchesAllAffinityTermProperties(targetPod, affinityTerms) {
		for _, t := range affinityTerms {
			if topologyValue, ok := targetPodNode.Labels[t.topologyKey]; ok {
				pair := topologyPair{key: t.topologyKey, value: topologyValue}
				m[pair] += value
				// value could be a negative value, hence we delete the entry if
				// the entry is down to zero.
				if m[pair] == 0 {
					delete(m, pair)
				}
			}
		}
	}
}

// updateAntiAffinityTerms updates the topologyToMatchedTermCount map with the specified value
// for each anti-affinity term matched the target pod.
func (m topologyToMatchedTermCount) updateWithAntiAffinityTerms(targetPod *v1.Pod, targetPodNode *v1.Node, antiAffinityTerms []*affinityTermProperties, value int64) {
	// Check anti-affinity properties.
	for _, a := range antiAffinityTerms {
		if priorityutil.PodMatchesTermsNamespaceAndSelector(targetPod, a.namespaces, a.selector) {
			if topologyValue, ok := targetPodNode.Labels[a.topologyKey]; ok {
				pair := topologyPair{key: a.topologyKey, value: topologyValue}
				m[pair] += value
				// value could be a negative value, hence we delete the entry if
				// the entry is down to zero.
				if m[pair] == 0 {
					delete(m, pair)
				}
			}
		}
	}
}

// UpdateWithPod updates the metadata counters with the (anti)affinity matches for the given pod.
func (m *PodAffinityMetadata) UpdateWithPod(updatedPod, pod *v1.Pod, node *v1.Node, multiplier int64) error {
	if m == nil {
		return nil
	}

	// Update matching existing anti-affinity terms.
	updatedPodAffinity := updatedPod.Spec.Affinity
	if updatedPodAffinity != nil && updatedPodAffinity.PodAntiAffinity != nil {
		antiAffinityProperties, err := getAffinityTermProperties(pod, GetPodAntiAffinityTerms(updatedPodAffinity.PodAntiAffinity))
		if err != nil {
			klog.Errorf("error in getting anti-affinity properties of Pod %v", updatedPod.Name)
			return err
		}
		m.topologyToMatchedExistingAntiAffinityTerms.updateWithAntiAffinityTerms(pod, node, antiAffinityProperties, multiplier)
	}

	// Update matching incoming pod (anti)affinity terms.
	affinity := pod.Spec.Affinity
	podNodeName := updatedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		if affinity.PodAffinity == nil {
			affinityProperties, err := getAffinityTermProperties(pod, GetPodAffinityTerms(affinity.PodAffinity))
			if err != nil {
				klog.Errorf("error in getting affinity properties of Pod %v", pod.Name)
				return err
			}
			m.topologyToMatchedAffinityTerms.updateWithAffinityTerms(updatedPod, node, affinityProperties, multiplier)
		}
		if affinity.PodAntiAffinity != nil {
			antiAffinityProperties, err := getAffinityTermProperties(pod, GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
			if err != nil {
				klog.Errorf("error in getting anti-affinity properties of Pod %v", pod.Name)
				return err
			}
			m.topologyToMatchedAntiAffinityTerms.updateWithAntiAffinityTerms(updatedPod, node, antiAffinityProperties, multiplier)
		}
	}
	return nil
}

// Clone makes a deep copy of PodAffinityMetadata.
func (m *PodAffinityMetadata) Clone() *PodAffinityMetadata {
	if m == nil {
		return nil
	}

	copy := PodAffinityMetadata{}
	copy.topologyToMatchedAffinityTerms = m.topologyToMatchedAffinityTerms.clone()
	copy.topologyToMatchedAntiAffinityTerms = m.topologyToMatchedAntiAffinityTerms.clone()
	copy.topologyToMatchedExistingAntiAffinityTerms = m.topologyToMatchedExistingAntiAffinityTerms.clone()

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

// NOTE: When new fields are added/removed or logic is changed, please make sure that
// RemovePod, AddPod, and ShallowCopy functions are updated to work with the new changes.
type predicateMetadata struct {
	pod *v1.Pod

	// evenPodsSpreadMetadata holds info of the minimum match number on each topology spread constraint,
	// and the match number of all valid topology pairs.
	evenPodsSpreadMetadata *evenPodsSpreadMetadata

	serviceAffinityMetadata  *serviceAffinityMetadata
	podFitsResourcesMetadata *podFitsResourcesMetadata
}

// Ensure that predicateMetadata implements algorithm.Metadata.
var _ Metadata = &predicateMetadata{}

// predicateMetadataProducer function produces predicate metadata. It is stored in a global variable below
// and used to modify the return values of MetadataProducer
type predicateMetadataProducer func(pm *predicateMetadata)

var predicateMetadataProducers = make(map[string]predicateMetadataProducer)

// RegisterPredicateMetadataProducer registers a MetadataProducer.
func RegisterPredicateMetadataProducer(predicateName string, precomp predicateMetadataProducer) {
	predicateMetadataProducers[predicateName] = precomp
}

// EmptyMetadataProducer returns a no-op MetadataProducer type.
func EmptyMetadataProducer(pod *v1.Pod, sharedLister schedulerlisters.SharedLister) Metadata {
	return nil
}

// RegisterPredicateMetadataProducerWithExtendedResourceOptions registers a
// MetadataProducer that creates predicate metadata with the provided
// options for extended resources.
//
// See the comments in "predicateMetadata" for the explanation of the options.
func RegisterPredicateMetadataProducerWithExtendedResourceOptions(ignoredExtendedResources sets.String) {
	RegisterPredicateMetadataProducer("PredicateWithExtendedResourceOptions", func(pm *predicateMetadata) {
		pm.podFitsResourcesMetadata.ignoredExtendedResources = ignoredExtendedResources
	})
}

// MetadataProducerFactory is a factory to produce Metadata.
type MetadataProducerFactory struct{}

// GetPredicateMetadata returns the predicateMetadata which will be used by various predicates.
func (f *MetadataProducerFactory) GetPredicateMetadata(pod *v1.Pod, sharedLister schedulerlisters.SharedLister) Metadata {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}

	var allNodes []*schedulernodeinfo.NodeInfo
	if sharedLister != nil {
		var err error
		allNodes, err = sharedLister.NodeInfos().List()
		if err != nil {
			klog.Errorf("failed to list NodeInfos: %v", err)
			return nil
		}
	}

	// evenPodsSpreadMetadata represents how existing pods match "pod"
	// on its spread constraints
	evenPodsSpreadMetadata, err := getEvenPodsSpreadMetadata(pod, allNodes)
	if err != nil {
		klog.Errorf("Error calculating spreadConstraintsMap: %v", err)
		return nil
	}

	predicateMetadata := &predicateMetadata{
		pod:                      pod,
		evenPodsSpreadMetadata:   evenPodsSpreadMetadata,
		podFitsResourcesMetadata: getPodFitsResourcesMetedata(pod),
	}
	for predicateName, precomputeFunc := range predicateMetadataProducers {
		klog.V(10).Infof("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

func getPodFitsResourcesMetedata(pod *v1.Pod) *podFitsResourcesMetadata {
	return &podFitsResourcesMetadata{
		podRequest: GetResourceRequest(pod),
	}
}

// GetPodAffinityMetadata computes inter-pod affinity metadata.
func GetPodAffinityMetadata(pod *v1.Pod, allNodes []*schedulernodeinfo.NodeInfo, havePodsWithAffinityNodes []*schedulernodeinfo.NodeInfo) (*PodAffinityMetadata, error) {
	// existingPodAntiAffinityMap will be used later for efficient check on existing pods' anti-affinity
	existingPodAntiAffinityMap, err := getTPMapMatchingExistingAntiAffinity(pod, havePodsWithAffinityNodes)
	if err != nil {
		return nil, err
	}
	// incomingPodAffinityMap will be used later for efficient check on incoming pod's affinity
	// incomingPodAntiAffinityMap will be used later for efficient check on incoming pod's anti-affinity
	incomingPodAffinityMap, incomingPodAntiAffinityMap, err := getTPMapMatchingIncomingAffinityAntiAffinity(pod, allNodes)
	if err != nil {
		return nil, err
	}

	return &PodAffinityMetadata{
		topologyToMatchedAffinityTerms:             incomingPodAffinityMap,
		topologyToMatchedAntiAffinityTerms:         incomingPodAntiAffinityMap,
		topologyToMatchedExistingAntiAffinityTerms: existingPodAntiAffinityMap,
	}, nil
}

func getEvenPodsSpreadMetadata(pod *v1.Pod, allNodes []*schedulernodeinfo.NodeInfo) (*evenPodsSpreadMetadata, error) {
	// We have feature gating in APIServer to strip the spec
	// so don't need to re-check feature gate, just check length of constraints.
	constraints, err := filterHardTopologySpreadConstraints(pod.Spec.TopologySpreadConstraints)
	if err != nil {
		return nil, err
	}
	if len(constraints) == 0 {
		return nil, nil
	}

	var lock sync.Mutex

	// TODO(Huang-Wei): It might be possible to use "make(map[topologyPair]*int32)".
	// In that case, need to consider how to init each tpPairToCount[pair] in an atomic fashion.
	m := evenPodsSpreadMetadata{
		constraints:          constraints,
		tpKeyToCriticalPaths: make(map[string]*criticalPaths, len(constraints)),
		tpPairToMatchNum:     make(map[topologyPair]int32),
	}
	addTopologyPairMatchNum := func(pair topologyPair, num int32) {
		lock.Lock()
		m.tpPairToMatchNum[pair] += num
		lock.Unlock()
	}

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
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
				if constraint.selector.Matches(labels.Set(existingPod.Labels)) {
					matchTotal++
				}
			}
			pair := topologyPair{key: constraint.topologyKey, value: node.Labels[constraint.topologyKey]}
			addTopologyPairMatchNum(pair, matchTotal)
		}
	}
	workqueue.ParallelizeUntil(context.Background(), 16, len(allNodes), processNode)

	// calculate min match for each topology pair
	for i := 0; i < len(constraints); i++ {
		key := constraints[i].topologyKey
		m.tpKeyToCriticalPaths[key] = newCriticalPaths()
	}
	for pair, num := range m.tpPairToMatchNum {
		m.tpKeyToCriticalPaths[pair.key].update(pair.value, num)
	}

	return &m, nil
}

func filterHardTopologySpreadConstraints(constraints []v1.TopologySpreadConstraint) ([]topologySpreadConstraint, error) {
	var result []topologySpreadConstraint
	for _, c := range constraints {
		if c.WhenUnsatisfiable == v1.DoNotSchedule {
			selector, err := metav1.LabelSelectorAsSelector(c.LabelSelector)
			if err != nil {
				return nil, err
			}
			result = append(result, topologySpreadConstraint{
				maxSkew:     c.MaxSkew,
				topologyKey: c.TopologyKey,
				selector:    selector,
			})
		}
	}
	return result, nil
}

// NodeLabelsMatchSpreadConstraints checks if ALL topology keys in spread constraints are present in node labels.
func NodeLabelsMatchSpreadConstraints(nodeLabels map[string]string, constraints []topologySpreadConstraint) bool {
	for _, c := range constraints {
		if _, ok := nodeLabels[c.topologyKey]; !ok {
			return false
		}
	}
	return true
}

func (m topologyToMatchedTermCount) appendMaps(toAppend topologyToMatchedTermCount) {
	for pair := range toAppend {
		m[pair] += toAppend[pair]
	}
}

func (m topologyToMatchedTermCount) clone() topologyToMatchedTermCount {
	copy := make(topologyToMatchedTermCount, len(m))
	copy.appendMaps(m)
	return copy
}

func (m *evenPodsSpreadMetadata) addPod(addedPod, preemptorPod *v1.Pod, node *v1.Node) {
	m.updatePod(addedPod, preemptorPod, node, 1)
}

func (m *evenPodsSpreadMetadata) removePod(deletedPod, preemptorPod *v1.Pod, node *v1.Node) {
	m.updatePod(deletedPod, preemptorPod, node, -1)
}

func (m *evenPodsSpreadMetadata) updatePod(updatedPod, preemptorPod *v1.Pod, node *v1.Node, delta int32) {
	if m == nil || updatedPod.Namespace != preemptorPod.Namespace || node == nil {
		return
	}
	if !NodeLabelsMatchSpreadConstraints(node.Labels, m.constraints) {
		return
	}

	podLabelSet := labels.Set(updatedPod.Labels)
	for _, constraint := range m.constraints {
		if !constraint.selector.Matches(podLabelSet) {
			continue
		}

		k, v := constraint.topologyKey, node.Labels[constraint.topologyKey]
		pair := topologyPair{key: k, value: v}
		m.tpPairToMatchNum[pair] = m.tpPairToMatchNum[pair] + delta

		m.tpKeyToCriticalPaths[k].update(v, m.tpPairToMatchNum[pair])
	}
}

func (m *evenPodsSpreadMetadata) clone() *evenPodsSpreadMetadata {
	// c could be nil when EvenPodsSpread feature is disabled
	if m == nil {
		return nil
	}
	cp := evenPodsSpreadMetadata{
		// constraints are shared because they don't change.
		constraints:          m.constraints,
		tpKeyToCriticalPaths: make(map[string]*criticalPaths, len(m.tpKeyToCriticalPaths)),
		tpPairToMatchNum:     make(map[topologyPair]int32, len(m.tpPairToMatchNum)),
	}
	for tpKey, paths := range m.tpKeyToCriticalPaths {
		cp.tpKeyToCriticalPaths[tpKey] = &criticalPaths{paths[0], paths[1]}
	}
	for tpPair, matchNum := range m.tpPairToMatchNum {
		copyPair := topologyPair{key: tpPair.key, value: tpPair.value}
		cp.tpPairToMatchNum[copyPair] = matchNum
	}
	return &cp
}

// RemovePod changes predicateMetadata assuming that the given `deletedPod` is
// deleted from the system.
func (meta *predicateMetadata) RemovePod(deletedPod *v1.Pod, node *v1.Node) error {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	if deletedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("deletedPod and meta.pod must not be the same")
	}
	meta.evenPodsSpreadMetadata.removePod(deletedPod, meta.pod, node)
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

	// Update meta.evenPodsSpreadMetadata if meta.pod has hard spread constraints
	// and addedPod matches that
	meta.evenPodsSpreadMetadata.addPod(addedPod, meta.pod, node)

	meta.serviceAffinityMetadata.addPod(addedPod, meta.pod, node)

	return nil
}

// ShallowCopy copies a metadata struct into a new struct and creates a copy of
// its maps and slices, but it does not copy the contents of pointer values.
func (meta *predicateMetadata) ShallowCopy() Metadata {
	newPredMeta := &predicateMetadata{
		pod: meta.pod,
	}
	newPredMeta.evenPodsSpreadMetadata = meta.evenPodsSpreadMetadata.clone()
	newPredMeta.serviceAffinityMetadata = meta.serviceAffinityMetadata.clone()
	newPredMeta.podFitsResourcesMetadata = meta.podFitsResourcesMetadata.clone()
	return (Metadata)(newPredMeta)
}

// A processed version of v1.PodAffinityTerm.
type affinityTermProperties struct {
	namespaces  sets.String
	selector    labels.Selector
	topologyKey string
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
		properties = append(properties, &affinityTermProperties{namespaces: namespaces, selector: selector, topologyKey: term.TopologyKey})
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

// getTPMapMatchingExistingAntiAffinity calculates the following for each existing pod on each node:
// (1) Whether it has PodAntiAffinity
// (2) Whether any AffinityTerm matches the incoming pod
func getTPMapMatchingExistingAntiAffinity(pod *v1.Pod, allNodes []*schedulernodeinfo.NodeInfo) (topologyToMatchedTermCount, error) {
	errCh := schedutil.NewErrorChannel()
	var lock sync.Mutex
	topologyMap := make(topologyToMatchedTermCount)

	appendResult := func(toAppend topologyToMatchedTermCount) {
		lock.Lock()
		defer lock.Unlock()
		topologyMap.appendMaps(toAppend)
	}

	ctx, cancel := context.WithCancel(context.Background())

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			existingPodTopologyMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, existingPod, node)
			if err != nil {
				errCh.SendErrorWithCancel(err, cancel)
				return
			}
			if existingPodTopologyMaps != nil {
				appendResult(existingPodTopologyMaps)
			}
		}
	}
	workqueue.ParallelizeUntil(ctx, 16, len(allNodes), processNode)

	if err := errCh.ReceiveError(); err != nil {
		return nil, err
	}

	return topologyMap, nil
}

// getTPMapMatchingIncomingAffinityAntiAffinity finds existing Pods that match affinity terms of the given "pod".
// It returns a topologyToMatchedTermCount that are checked later by the affinity
// predicate. With this topologyToMatchedTermCount available, the affinity predicate does not
// need to check all the pods in the cluster.
func getTPMapMatchingIncomingAffinityAntiAffinity(pod *v1.Pod, allNodes []*schedulernodeinfo.NodeInfo) (topologyToMatchedTermCount, topologyToMatchedTermCount, error) {
	topologyPairsAffinityPodsMap := make(topologyToMatchedTermCount)
	topologyToMatchedExistingAntiAffinityTerms := make(topologyToMatchedTermCount)
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return topologyPairsAffinityPodsMap, topologyToMatchedExistingAntiAffinityTerms, nil
	}

	var lock sync.Mutex
	appendResult := func(nodeName string, nodeTopologyPairsAffinityPodsMap, nodeTopologyPairsAntiAffinityPodsMap topologyToMatchedTermCount) {
		lock.Lock()
		defer lock.Unlock()
		if len(nodeTopologyPairsAffinityPodsMap) > 0 {
			topologyPairsAffinityPodsMap.appendMaps(nodeTopologyPairsAffinityPodsMap)
		}
		if len(nodeTopologyPairsAntiAffinityPodsMap) > 0 {
			topologyToMatchedExistingAntiAffinityTerms.appendMaps(nodeTopologyPairsAntiAffinityPodsMap)
		}
	}

	affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
	affinityProperties, err := getAffinityTermProperties(pod, affinityTerms)
	if err != nil {
		return nil, nil, err
	}

	antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)
	antiAffinityProperties, err := getAffinityTermProperties(pod, antiAffinityTerms)
	if err != nil {
		return nil, nil, err
	}

	processNode := func(i int) {
		nodeInfo := allNodes[i]
		node := nodeInfo.Node()
		if node == nil {
			klog.Error("node not found")
			return
		}
		nodeTopologyPairsAffinityPodsMap := make(topologyToMatchedTermCount)
		nodeTopologyPairsAntiAffinityPodsMap := make(topologyToMatchedTermCount)
		for _, existingPod := range nodeInfo.Pods() {
			// Check affinity properties.
			nodeTopologyPairsAffinityPodsMap.updateWithAffinityTerms(existingPod, node, affinityProperties, 1)

			// Check anti-affinity properties.
			nodeTopologyPairsAntiAffinityPodsMap.updateWithAntiAffinityTerms(existingPod, node, antiAffinityProperties, 1)
		}

		if len(nodeTopologyPairsAffinityPodsMap) > 0 || len(nodeTopologyPairsAntiAffinityPodsMap) > 0 {
			appendResult(node.Name, nodeTopologyPairsAffinityPodsMap, nodeTopologyPairsAntiAffinityPodsMap)
		}
	}
	workqueue.ParallelizeUntil(context.Background(), 16, len(allNodes), processNode)

	return topologyPairsAffinityPodsMap, topologyToMatchedExistingAntiAffinityTerms, nil
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
