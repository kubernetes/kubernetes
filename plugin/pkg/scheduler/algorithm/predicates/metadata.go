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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedutil "k8s.io/kubernetes/plugin/pkg/scheduler/util"

	"github.com/golang/glog"
)

type PredicateMetadataFactory struct {
	podLister       algorithm.PodLister
	namespaceLister algorithm.NamespaceLister
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
	podPorts      map[int]bool
	//key is a pod full name with the anti-affinity rules.
	matchingAntiAffinityTerms          map[string][]matchingPodAntiAffinityTerm
	serviceAffinityInUse               bool
	serviceAffinityMatchingPodList     []*v1.Pod
	serviceAffinityMatchingPodServices []*v1.Service
	namespaceLister                    algorithm.NamespaceLister
}

// Ensure that predicateMetadata implements algorithm.PredicateMetadata.
var _ algorithm.PredicateMetadata = &predicateMetadata{}

// PredicateMetadataProducer: Helper types/variables...
type PredicateMetadataProducer func(pm *predicateMetadata)

var predicateMetaProducerRegisterLock sync.Mutex
var predicateMetadataProducers map[string]PredicateMetadataProducer = make(map[string]PredicateMetadataProducer)

func RegisterPredicateMetadataProducer(predicateName string, precomp PredicateMetadataProducer) {
	predicateMetaProducerRegisterLock.Lock()
	defer predicateMetaProducerRegisterLock.Unlock()
	predicateMetadataProducers[predicateName] = precomp
}

func NewPredicateMetadataFactory(podLister algorithm.PodLister, namespaceLister algorithm.NamespaceLister) algorithm.PredicateMetadataProducer {
	factory := &PredicateMetadataFactory{
		podLister:       podLister,
		namespaceLister: namespaceLister,
	}
	return factory.GetMetadata
}

// GetMetadata returns the predicateMetadata used which will be used by various predicates.
func (pfactory *PredicateMetadataFactory) GetMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulercache.NodeInfo) algorithm.PredicateMetadata {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	matchingTerms, err := pfactory.getMatchingAntiAffinityTerms(pod, nodeNameToInfoMap)
	if err != nil {
		return nil
	}
	predicateMetadata := &predicateMetadata{
		pod:                       pod,
		podBestEffort:             isPodBestEffort(pod),
		podRequest:                GetResourceRequest(pod),
		podPorts:                  schedutil.GetUsedPorts(pod),
		matchingAntiAffinityTerms: matchingTerms,
		namespaceLister:           pfactory.namespaceLister,
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
		return fmt.Errorf("deletedPod and meta.pod must not be the same.")
	}
	// Delete any anti-affinity rule from the deletedPod.
	delete(meta.matchingAntiAffinityTerms, deletedPodFullName)
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
		return fmt.Errorf("addedPod and meta.pod must not be the same.")
	}
	if nodeInfo.Node() == nil {
		return fmt.Errorf("Invalid node in nodeInfo.")
	}
	// Add matching anti-affinity terms of the addedPod to the map.
	podMatchingTerms, err := getMatchingAntiAffinityTermsOfExistingPod(meta.pod, addedPod, nodeInfo.Node(), meta.namespaceLister)
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
		pod:                  meta.pod,
		podBestEffort:        meta.podBestEffort,
		podRequest:           meta.podRequest,
		serviceAffinityInUse: meta.serviceAffinityInUse,
	}
	newPredMeta.podPorts = map[int]bool{}
	for k, v := range meta.podPorts {
		newPredMeta.podPorts[k] = v
	}
	newPredMeta.matchingAntiAffinityTerms = map[string][]matchingPodAntiAffinityTerm{}
	for k, v := range meta.matchingAntiAffinityTerms {
		newPredMeta.matchingAntiAffinityTerms[k] = append([]matchingPodAntiAffinityTerm(nil), v...)
	}
	newPredMeta.serviceAffinityMatchingPodServices = append([]*v1.Service(nil),
		meta.serviceAffinityMatchingPodServices...)
	newPredMeta.serviceAffinityMatchingPodList = append([]*v1.Pod(nil),
		meta.serviceAffinityMatchingPodList...)
	return (algorithm.PredicateMetadata)(newPredMeta)
}

func (pfactory *PredicateMetadataFactory) getMatchingAntiAffinityTerms(pod *v1.Pod, nodeInfoMap map[string]*schedulercache.NodeInfo) (map[string][]matchingPodAntiAffinityTerm, error) {
	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	var lock sync.Mutex
	var firstError error
	result := make(map[string][]matchingPodAntiAffinityTerm)
	appendResult := func(toAppend map[string][]matchingPodAntiAffinityTerm) {
		lock.Lock()
		defer lock.Unlock()
		for uid, terms := range toAppend {
			result[uid] = append(result[uid], terms...)
		}
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
		nodeResult := make(map[string][]matchingPodAntiAffinityTerm)
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			affinity := existingPod.Spec.Affinity
			if affinity == nil {
				continue
			}
			for _, term := range getPodAntiAffinityTerms(affinity.PodAntiAffinity) {
				namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(pfactory.namespaceLister, existingPod, &term)
				selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
				if err != nil {
					catchError(err)
					return
				}
				if priorityutil.PodMatchesTermsNamespaceAndSelector(pod, namespaces, selector) {
					existingPodFullName := schedutil.GetPodFullName(existingPod)
					nodeResult[existingPodFullName] = append(
						nodeResult[existingPodFullName],
						matchingPodAntiAffinityTerm{term: &term, node: node})
				}
			}
		}
		if len(nodeResult) > 0 {
			appendResult(nodeResult)
		}
	}
	workqueue.Parallelize(16, len(allNodeNames), processNode)
	return result, firstError
}
