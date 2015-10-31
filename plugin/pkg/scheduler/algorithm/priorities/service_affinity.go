/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
)

type ServiceAffinity struct {
	serviceLister    algorithm.ServiceLister
	controllerLister algorithm.ControllerLister
}

func NewServiceAffinityPriority(serviceLister algorithm.ServiceLister, controllerLister algorithm.ControllerLister) algorithm.PriorityFunction {
	serviceAffinity := &ServiceAffinity{
		serviceLister:    serviceLister,
		controllerLister: controllerLister,
	}
	return serviceAffinity.CalculateAffinityPriority
}

// CalculateAffinityPriority affiliates a pod to the existing services' pods
// that matches the labels indicated in affinitySelector.
// The more existing pods to affiliate deployed on, the higher priority the node gets.
func (s *ServiceAffinity) CalculateAffinityPriority(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (algorithm.HostPriorityList, error) {
	var maxCount int
	counts := map[string]int{}
	affinitySelector := labels.SelectorFromSet(labels.Set(pod.Spec.AffinitySelector))

	// One direction: affiliate to pods of services
	// 1) get services in the namespace same with pod
	// 2) filter out services that matches affinitySelector
	// 3) list services' pods to affiliate.
	selectors := make([]labels.Selector, 0)
	serviceList, err := s.serviceLister.List()
	if err == nil {
		for _, service := range serviceList.Items {
			// consider only services that are in the same namespace as the pod
			if service.Namespace != pod.Namespace {
				continue
			}
			if affinitySelector.Matches(labels.Set(service.Labels)) {
				selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
			}
		}
	}
	controllers, err := s.controllerLister.List()
	if err == nil {
		for _, controller := range controllers {
			// consider only ReplicationController that are in the same namespace as the pod
			if controller.Namespace != pod.Namespace {
				continue
			}
			if affinitySelector.Matches(labels.Set(controller.Labels)) {
				selectors = append(selectors, labels.SelectorFromSet(controller.Spec.Selector))
			}
		}
	}

	for _, selector := range selectors {
		selectorPods, err := podLister.List(selector)
		if err != nil {
			glog.V(10).Infof("PodLister Error")
			return nil, err
		}

		if len(selectorPods) > 0 {
			for _, selectorPod := range selectorPods {
				// Only match pods in the same namespace
				if selectorPod.Namespace != pod.Namespace {
					continue
				}

				counts[selectorPod.Spec.NodeName]++
				if counts[selectorPod.Spec.NodeName] > maxCount {
					maxCount = counts[selectorPod.Spec.NodeName]
				}
			}
		} else {
			glog.V(10).Infof("No Pods")
		}
	}

	// Another direction: affiliated by pods whose affinitySelectors match pod's service labels.
	// 1) get services of pod
	// 2) get existingPods in namespace same with pod
	// 3) filter existingPods if has affinitySelector matches pod's service labels.
	existingPods, err := podLister.List(labels.Everything())
	if err != nil {
		glog.V(10).Infof("PodLister Error")
		return nil, err
	}
	podServices, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
		for _, existingPod := range existingPods {
			if existingPod.Namespace != pod.Namespace {
				continue
			}

			for _, podService := range podServices {
				affiliatedBySelector := labels.SelectorFromSet(labels.Set(existingPod.Spec.AffinitySelector))
				if affiliatedBySelector.Matches(labels.Set(podService.Labels)) {
					counts[existingPod.Spec.NodeName]++
					if counts[existingPod.Spec.NodeName] > maxCount {
						maxCount = counts[existingPod.Spec.NodeName]
					}
				}
			}
		}
	}

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}
	result := []algorithm.HostPriority{}
	// score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for _, node := range nodes.Items {
		// initializing to the default/min node score of 0
		fScore := float32(0)
		if maxCount > 0 {
			fScore = 10 * (float32(counts[node.Name]) / float32(maxCount))
		}
		result = append(result, algorithm.HostPriority{Host: node.Name, Score: int(fScore)})
		glog.V(10).Infof(
			"%v -> %v: ServiceAffinityPriority, Score: (%d)", pod.Name, node.Name, int(fScore),
		)
	}
	return result, nil
}
