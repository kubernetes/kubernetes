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

//  Note that predicateMetadata and matchingPodAntiAffinityTerm need to be declared in the same file
//  due to the way declarations are processed in predicate declaration unit tests.
type matchingPodAntiAffinityTerm struct {
	term *v1.PodAffinityTerm
	node *v1.Node
}

// NOTE: When new fields are added/removed or logic is changed, please make sure that
// RemovePod, AddPod, and ShallowCopy functions are updated to work with the new changes.
type predicateMetadata struct {
	pod           *v1.Pod
	podBestEffort bool
	podRequest    *schedulercache.Resource
	podPorts      []*v1.ContainerPort
	//key is a pod full name with the anti-affinity rules.
	matchingAntiAffinityTerms map[string][]matchingPodAntiAffinityTerm
	// A map of node name to a list of Pods on the node that can potentially match
	// the affinity rules of the "pod".
	nodeNameToMatchingAffinityPods map[string][]*v1.Pod
	// A map of node name to a list of Pods on the node that can potentially match
	// the anti-affinity rules of the "pod".
	nodeNameToMatchingAntiAffinityPods map[string][]*v1.Pod
	serviceAffinityInUse               bool
	serviceAffinityMatchingPodList     []*v1.Pod
	serviceAffinityMatchingPodServices []*v1.Service
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
	matchingTerms, err := getMatchingAntiAffinityTerms(pod, nodeNameToInfoMap)
	if err != nil {
		return nil
	}
	affinityPods, antiAffinityPods, err := getPodsMatchingAffinity(pod, nodeNameToInfoMap)
	if err != nil {
		glog.Errorf("[predicate meta data generation] error finding pods that match affinity terms: %v", err)
		return nil
	}
	predicateMetadata := &predicateMetadata{
		pod:                                pod,
		podBestEffort:                      isPodBestEffort(pod),
		podRequest:                         GetResourceRequest(pod),
		podPorts:                           schedutil.GetContainerPorts(pod),
		matchingAntiAffinityTerms:          matchingTerms,
		nodeNameToMatchingAffinityPods:     affinityPods,
		nodeNameToMatchingAntiAffinityPods: antiAffinityPods,
	}
	for predicateName, precomputeFunc := range predicateMetadataProducers {
		glog.V(10).Infof("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

// RemovePod changes predicateMetadata assuming that the given `deletedPod` is
// deleted from the system.
func (meta *predicateMetadata) RemovePod(deletedPod *v1.Pod) error {
	deletedPodFullName := schedutil.GetPodFullName(deletedPod)
	if deletedPodFullName == schedutil.GetPodFullName(meta.pod) {
		return fmt.Errorf("deletedPod and meta.pod must not be the same")
	}
	// Delete any anti-affinity rule from the deletedPod.
	delete(meta.matchingAntiAffinityTerms, deletedPodFullName)
	// Delete pod from the matching affinity or anti-affinity pods if exists.
	affinity := meta.pod.Spec.Affinity
	podNodeName := deletedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		if affinity.PodAffinity != nil {
			for i, p := range meta.nodeNameToMatchingAffinityPods[podNodeName] {
				if p == deletedPod {
					s := meta.nodeNameToMatchingAffinityPods[podNodeName]
					s[i] = s[len(s)-1]
					s = s[:len(s)-1]
					meta.nodeNameToMatchingAffinityPods[podNodeName] = s
					break
				}
			}
		}
		if affinity.PodAntiAffinity != nil {
			for i, p := range meta.nodeNameToMatchingAntiAffinityPods[podNodeName] {
				if p == deletedPod {
					s := meta.nodeNameToMatchingAntiAffinityPods[podNodeName]
					s[i] = s[len(s)-1]
					s = s[:len(s)-1]
					meta.nodeNameToMatchingAntiAffinityPods[podNodeName] = s
					break
				}
			}
		}
	}
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
	podMatchingTerms, err := getMatchingAntiAffinityTermsOfExistingPod(meta.pod, addedPod, nodeInfo.Node())
	if err != nil {
		return err
	}
	if len(podMatchingTerms) > 0 {
		existingTerms, found := meta.matchingAntiAffinityTerms[addedPodFullName]
		if found {
			meta.matchingAntiAffinityTerms[addedPodFullName] = append(existingTerms,
				podMatchingTerms...)
		} else {
			meta.matchingAntiAffinityTerms[addedPodFullName] = podMatchingTerms
		}
	}
	// Add the pod to nodeNameToMatchingAffinityPods and nodeNameToMatchingAntiAffinityPods if needed.
	affinity := meta.pod.Spec.Affinity
	podNodeName := addedPod.Spec.NodeName
	if affinity != nil && len(podNodeName) > 0 {
		if targetPodMatchesAffinityOfPod(meta.pod, addedPod) {
			found := false
			for _, p := range meta.nodeNameToMatchingAffinityPods[podNodeName] {
				if p == addedPod {
					found = true
					break
				}
			}
			if !found {
				meta.nodeNameToMatchingAffinityPods[podNodeName] = append(meta.nodeNameToMatchingAffinityPods[podNodeName], addedPod)
			}
		}
		if targetPodMatchesAntiAffinityOfPod(meta.pod, addedPod) {
			found := false
			for _, p := range meta.nodeNameToMatchingAntiAffinityPods[podNodeName] {
				if p == addedPod {
					found = true
					break
				}
			}
			if !found {
				meta.nodeNameToMatchingAntiAffinityPods[podNodeName] = append(meta.nodeNameToMatchingAntiAffinityPods[podNodeName], addedPod)
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
	newPredMeta.matchingAntiAffinityTerms = map[string][]matchingPodAntiAffinityTerm{}
	for k, v := range meta.matchingAntiAffinityTerms {
		newPredMeta.matchingAntiAffinityTerms[k] = append([]matchingPodAntiAffinityTerm(nil), v...)
	}
	newPredMeta.nodeNameToMatchingAffinityPods = make(map[string][]*v1.Pod)
	for k, v := range meta.nodeNameToMatchingAffinityPods {
		newPredMeta.nodeNameToMatchingAffinityPods[k] = append([]*v1.Pod(nil), v...)
	}
	newPredMeta.nodeNameToMatchingAntiAffinityPods = make(map[string][]*v1.Pod)
	for k, v := range meta.nodeNameToMatchingAntiAffinityPods {
		newPredMeta.nodeNameToMatchingAntiAffinityPods[k] = append([]*v1.Pod(nil), v...)
	}
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

// podMatchesAffinityTermProperties return true IFF the given pod matches all the given properties.
func podMatchesAffinityTermProperties(pod *v1.Pod, properties []*affinityTermProperties) bool {
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

// getPodsMatchingAffinity finds existing Pods that match affinity terms of the given "pod".
// It ignores topology. It returns a set of Pods that are checked later by the affinity
// predicate. With this set of pods available, the affinity predicate does not
// need to check all the pods in the cluster.
func getPodsMatchingAffinity(pod *v1.Pod, nodeInfoMap map[string]*schedulercache.NodeInfo) (affinityPods map[string][]*v1.Pod, antiAffinityPods map[string][]*v1.Pod, err error) {
	allNodeNames := make([]string, 0, len(nodeInfoMap))

	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return nil, nil, nil
	}

	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	var lock sync.Mutex
	var firstError error
	affinityPods = make(map[string][]*v1.Pod)
	antiAffinityPods = make(map[string][]*v1.Pod)
	appendResult := func(nodeName string, affPods, antiAffPods []*v1.Pod) {
		lock.Lock()
		defer lock.Unlock()
		if len(affPods) > 0 {
			affinityPods[nodeName] = affPods
		}
		if len(antiAffPods) > 0 {
			antiAffinityPods[nodeName] = antiAffPods
		}
	}

	catchError := func(err error) {
		lock.Lock()
		defer lock.Unlock()
		if firstError == nil {
			firstError = err
		}
	}

	affinityProperties, err := getAffinityTermProperties(pod, GetPodAffinityTerms(affinity.PodAffinity))
	if err != nil {
		return nil, nil, err
	}
	antiAffinityProperties, err := getAffinityTermProperties(pod, GetPodAntiAffinityTerms(affinity.PodAntiAffinity))
	if err != nil {
		return nil, nil, err
	}

	processNode := func(i int) {
		nodeInfo := nodeInfoMap[allNodeNames[i]]
		node := nodeInfo.Node()
		if node == nil {
			catchError(fmt.Errorf("nodeInfo.Node is nil"))
			return
		}
		affPods := make([]*v1.Pod, 0, len(nodeInfo.Pods()))
		antiAffPods := make([]*v1.Pod, 0, len(nodeInfo.Pods()))
		for _, existingPod := range nodeInfo.Pods() {
			// Check affinity properties.
			if podMatchesAffinityTermProperties(existingPod, affinityProperties) {
				affPods = append(affPods, existingPod)
			}
			// Check anti-affinity properties.
			if podMatchesAffinityTermProperties(existingPod, antiAffinityProperties) {
				antiAffPods = append(antiAffPods, existingPod)
			}
		}
		if len(antiAffPods) > 0 || len(affPods) > 0 {
			appendResult(node.Name, affPods, antiAffPods)
		}
	}
	workqueue.Parallelize(16, len(allNodeNames), processNode)
	return affinityPods, antiAffinityPods, firstError
}

// podMatchesAffinity returns true if "targetPod" matches any affinity rule of
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
	return podMatchesAffinityTermProperties(targetPod, affinityProperties)
}

// targetPodMatchesAntiAffinityOfPod returns true if "targetPod" matches any anti-affinity
// rule of "pod". Similar to getPodsMatchingAffinity, this function does not check topology.
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
	return podMatchesAffinityTermProperties(targetPod, properties)
}
