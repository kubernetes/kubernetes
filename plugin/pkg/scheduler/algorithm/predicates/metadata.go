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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type PredicateMetadataFactory struct {
	podLister algorithm.PodLister
}

func NewPredicateMetadataFactory(podLister algorithm.PodLister) algorithm.MetadataProducer {
	factory := &PredicateMetadataFactory{
		podLister,
	}
	return factory.GetMetadata
}

// PredicateMetadata generates whatever metadata will be used over time to process predicate logic.
// Optional Vararg: PredicateMetadata is given access to all predicate information, so that it precompute data as necessary as an optimization.
func (pfactory *PredicateMetadataFactory) GetMetadata(pod *api.Pod, nodeNameToInfoMap map[string]*schedulercache.NodeInfo) interface{} {

	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	matchingTerms, err := getMatchingAntiAffinityTerms(pod, nodeNameToInfoMap)
	if err != nil {
		return nil
	}

	predicateMetadata := &predicateMetadata{
		pod:                       pod,
		podBestEffort:             isPodBestEffort(pod),
		podRequest:                getResourceRequest(pod),
		podPorts:                  getUsedPorts(pod),
		matchingAntiAffinityTerms: matchingTerms,
		servicePods:               make(map[string]([]*api.Pod)),
		nodeNameToInfo:            nodeNameToInfoMap,
	}

	// predicates is optional: its just a mechanism for us to iterate and possible precompute some things.

	if predicatePrecomputations != nil {
		for predicateName, preComputeFunction := range predicatePrecomputations {
			glog.V(4).Info("Precompute: %v", predicateName)
			preComputeFunction(predicateMetadata)
		}
	}
	return predicateMetadata
}
