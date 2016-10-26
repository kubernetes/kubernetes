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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

type PriorityMetadataFactory struct {
	serviceLister    algorithm.ServiceLister
	controllerLister algorithm.ControllerLister
	replicaSetLister algorithm.ReplicaSetLister
}

func NewPriorityMetadataFactory(
	sl algorithm.ServiceLister,
	cl algorithm.ControllerLister,
	rsl algorithm.ReplicaSetLister,
) algorithm.MetadataProducer {
	factory := &PriorityMetadataFactory{
		serviceLister:    sl,
		controllerLister: cl,
		replicaSetLister: rsl,
	}
	return factory.GetMetadata
}

// priorityMetadata is a type that is passed as metadata for priority functions
type priorityMetadata struct {
	nonZeroRequest      *schedulercache.Resource
	podTolerations      []api.Toleration
	affinity            *api.Affinity
	controllerSelectors []labels.Selector
}

// Returns selectors of services, RCs and RSs matching the given pod.
func getSelectors(
	pod *api.Pod,
	serviceLister algorithm.ServiceLister,
	controllerLister algorithm.ControllerLister,
	replicaSetLister algorithm.ReplicaSetLister,
) []labels.Selector {
	selectors := make([]labels.Selector, 0, 3)
	if serviceLister != nil {
		if services, err := serviceLister.GetPodServices(pod); err == nil {
			for _, service := range services {
				selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
			}
		}
	}
	if controllerLister != nil {
		if rcs, err := controllerLister.GetPodControllers(pod); err == nil {
			for _, rc := range rcs {
				selectors = append(selectors, labels.SelectorFromSet(rc.Spec.Selector))
			}
		}
	}
	if replicaSetLister != nil {
		if rss, err := replicaSetLister.GetPodReplicaSets(pod); err == nil {
			for _, rs := range rss {
				if selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
					selectors = append(selectors, selector)
				}
			}
		}
	}
	return selectors
}

func (pmf *PriorityMetadataFactory) getSelectors(pod *api.Pod) []labels.Selector {
	return getSelectors(pod, pmf.serviceLister, pmf.controllerLister, pmf.replicaSetLister)
}

// GetMetadata returns the priorityMetadata used which will be used by various predicates.
func (pmf *PriorityMetadataFactory) GetMetadata(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo) interface{} {
	// If we cannot compute metadata, just return nil
	if pod == nil {
		return nil
	}
	tolerations, err := getTolerationListFromPod(pod)
	if err != nil {
		return nil
	}
	affinity, err := api.GetAffinityFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil
	}
	return &priorityMetadata{
		nonZeroRequest:      getNonZeroRequests(pod),
		podTolerations:      tolerations,
		affinity:            affinity,
		controllerSelectors: pmf.getSelectors(pod),
	}
}
