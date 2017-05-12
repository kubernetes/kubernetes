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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type PredicateMetadataFactory struct {
	podLister       algorithm.PodLister
	namespaceLister algorithm.NamespaceLister
}

func NewPredicateMetadataFactory(podLister algorithm.PodLister, namespaceLister algorithm.NamespaceLister) algorithm.MetadataProducer {
	factory := &PredicateMetadataFactory{
		podLister:       podLister,
		namespaceLister: namespaceLister,
	}
	return factory.GetMetadata
}

// GetMetadata returns the predicateMetadata used which will be used by various predicates.
func (pfactory *PredicateMetadataFactory) GetMetadata(pod *v1.Pod, nodeNameToInfoMap map[string]*schedulercache.NodeInfo) interface{} {
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
		podPorts:                  GetUsedPorts(pod),
		matchingAntiAffinityTerms: matchingTerms,
	}
	for predicateName, precomputeFunc := range predicatePrecomputations {
		glog.V(10).Info("Precompute: %v", predicateName)
		precomputeFunc(predicateMetadata)
	}
	return predicateMetadata
}

func (pfactory *PredicateMetadataFactory) getMatchingAntiAffinityTerms(pod *v1.Pod, nodeInfoMap map[string]*schedulercache.NodeInfo) ([]matchingPodAntiAffinityTerm, error) {
	allNodeNames := make([]string, 0, len(nodeInfoMap))
	for name := range nodeInfoMap {
		allNodeNames = append(allNodeNames, name)
	}

	var lock sync.Mutex
	var result []matchingPodAntiAffinityTerm
	var firstError error
	appendResult := func(toAppend []matchingPodAntiAffinityTerm) {
		lock.Lock()
		defer lock.Unlock()
		result = append(result, toAppend...)
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
		var nodeResult []matchingPodAntiAffinityTerm
		for _, existingPod := range nodeInfo.PodsWithAffinity() {
			affinity := schedulercache.ReconcileAffinity(existingPod)
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
				match := priorityutil.PodMatchesTermsNamespaceAndSelector(pod, namespaces, selector)
				if match {
					nodeResult = append(nodeResult, matchingPodAntiAffinityTerm{term: &term, node: node})
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
