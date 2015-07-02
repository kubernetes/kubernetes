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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/scheduler/algorithm"
)

type ServiceSpread struct {
	serviceLister algorithm.ServiceLister
}

func NewServiceSpreadPriority(serviceLister algorithm.ServiceLister) algorithm.PriorityFunction {
	serviceSpread := &ServiceSpread{
		serviceLister: serviceLister,
	}
	return serviceSpread.CalculateSpreadPriority
}

// CalculateSpreadPriority spreads pods by minimizing the number of pods belonging to the same service
// on the same machine.
func (s *ServiceSpread) CalculateSpreadPriority(pod *api.Pod, podLister algorithm.PodLister, minionLister algorithm.MinionLister) (algorithm.HostPriorityList, error) {
	var maxCount int
	var nsServicePods []*api.Pod

	services, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
		// just use the first service and get the other pods within the service
		// TODO: a separate predicate can be created that tries to handle all services for the pod
		selector := labels.SelectorFromSet(services[0].Spec.Selector)
		pods, err := podLister.List(selector)
		if err != nil {
			return nil, err
		}
		// consider only the pods that belong to the same namespace
		for _, nsPod := range pods {
			if nsPod.Namespace == pod.Namespace {
				nsServicePods = append(nsServicePods, nsPod)
			}
		}
	}

	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	counts := map[string]int{}
	if len(nsServicePods) > 0 {
		for _, pod := range nsServicePods {
			counts[pod.Spec.NodeName]++
			// Compute the maximum number of pods hosted on any minion
			if counts[pod.Spec.NodeName] > maxCount {
				maxCount = counts[pod.Spec.NodeName]
			}
		}
	}

	result := []algorithm.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for _, minion := range minions.Items {
		// initializing to the default/max minion score of 10
		fScore := float32(10)
		if maxCount > 0 {
			fScore = 10 * (float32(maxCount-counts[minion.Name]) / float32(maxCount))
		}
		result = append(result, algorithm.HostPriority{Host: minion.Name, Score: int(fScore)})
	}
	return result, nil
}

type ServiceAntiAffinity struct {
	serviceLister algorithm.ServiceLister
	label         string
}

func NewServiceAntiAffinityPriority(serviceLister algorithm.ServiceLister, label string) algorithm.PriorityFunction {
	antiAffinity := &ServiceAntiAffinity{
		serviceLister: serviceLister,
		label:         label,
	}
	return antiAffinity.CalculateAntiAffinityPriority
}

// CalculateAntiAffinityPriority spreads pods by minimizing the number of pods belonging to the same service
// on machines with the same value for a particular label.
// The label to be considered is provided to the struct (ServiceAntiAffinity).
func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod *api.Pod, podLister algorithm.PodLister, minionLister algorithm.MinionLister) (algorithm.HostPriorityList, error) {
	var nsServicePods []*api.Pod

	services, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
		// just use the first service and get the other pods within the service
		// TODO: a separate predicate can be created that tries to handle all services for the pod
		selector := labels.SelectorFromSet(services[0].Spec.Selector)
		pods, err := podLister.List(selector)
		if err != nil {
			return nil, err
		}
		// consider only the pods that belong to the same namespace
		for _, nsPod := range pods {
			if nsPod.Namespace == pod.Namespace {
				nsServicePods = append(nsServicePods, nsPod)
			}
		}
	}

	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	// separate out the minions that have the label from the ones that don't
	otherMinions := []string{}
	labeledMinions := map[string]string{}
	for _, minion := range minions.Items {
		if labels.Set(minion.Labels).Has(s.label) {
			label := labels.Set(minion.Labels).Get(s.label)
			labeledMinions[minion.Name] = label
		} else {
			otherMinions = append(otherMinions, minion.Name)
		}
	}

	podCounts := map[string]int{}
	for _, pod := range nsServicePods {
		label, exists := labeledMinions[pod.Spec.NodeName]
		if !exists {
			continue
		}
		podCounts[label]++
	}

	numServicePods := len(nsServicePods)
	result := []algorithm.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for minion := range labeledMinions {
		// initializing to the default/max minion score of 10
		fScore := float32(10)
		if numServicePods > 0 {
			fScore = 10 * (float32(numServicePods-podCounts[labeledMinions[minion]]) / float32(numServicePods))
		}
		result = append(result, algorithm.HostPriority{Host: minion, Score: int(fScore)})
	}
	// add the open minions with a score of 0
	for _, minion := range otherMinions {
		result = append(result, algorithm.HostPriority{Host: minion, Score: 0})
	}

	return result, nil
}
