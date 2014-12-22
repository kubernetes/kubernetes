/*
Copyright 2014 Google Inc. All rights reserved.

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

package scheduler

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type ServiceSpread struct {
	serviceLister ServiceLister
}

func NewServiceSpreadPriority(serviceLister ServiceLister) PriorityFunction {
	serviceSpread := &ServiceSpread{
		serviceLister: serviceLister,
	}
	return serviceSpread.CalculateSpreadPriority
}

// CalculateSpreadPriority spreads pods by minimizing the number of pods on the same machine with the same labels.
// Importantly, if there are services in the system that span multiple heterogenous sets of pods, this spreading priority
// may not provide optimal spreading for the members of that Service.
// TODO: consider if we want to include Service label sets in the scheduling priority.
func (s *ServiceSpread) CalculateSpreadPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	var maxCount int
	var pods []api.Pod
	var err error

	service, err := s.serviceLister.GetPodService(pod)
	if err == nil {
		selector := labels.SelectorFromSet(service.Spec.Selector)
		pods, err = podLister.ListPods(selector)
		if err != nil {
			return nil, err
		}
	}

	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	counts := map[string]int{}
	if len(pods) > 0 {
		for _, pod := range pods {
			counts[pod.Status.Host]++
			// Compute the maximum number of pods hosted on any minion
			if counts[pod.Status.Host] > maxCount {
				maxCount = counts[pod.Status.Host]
			}
		}
	}

	result := []HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for _, minion := range minions.Items {
		// initializing to the default/max minion score of 10
		fScore := float32(10)
		if maxCount > 0 {
			fScore = 10 * (float32(maxCount-counts[minion.Name]) / float32(maxCount))
		}
		result = append(result, HostPriority{host: minion.Name, score: int(fScore)})
	}
	return result, nil
}

type ServiceAntiAffinity struct {
	serviceLister ServiceLister
	label         string
}

func NewServiceAntiAffinityPriority(serviceLister ServiceLister, label string) PriorityFunction {
	antiAffinity := &ServiceAntiAffinity{
		serviceLister: serviceLister,
		label:         label,
	}
	return antiAffinity.CalculateAntiAffinityPriority
}

func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod api.Pod, podLister PodLister, minionLister MinionLister) (HostPriorityList, error) {
	var service api.Service
	var pods []api.Pod
	var err error

	service, err = s.serviceLister.GetPodService(pod)
	if err == nil {
		selector := labels.SelectorFromSet(service.Spec.Selector)
		pods, err = podLister.ListPods(selector)
		if err != nil {
			return nil, err
		}
	}

	minions, err := minionLister.List()
	if err != nil {
		return nil, err
	}

	// find the zones that the minions belong to
	openMinions := []string{}
	zonedMinions := map[string]string{}
	for _, minion := range minions.Items {
		if labels.Set(minion.Labels).Has(s.label) {
			zone := labels.Set(minion.Labels).Get(s.label)
			zonedMinions[minion.Name] = zone
		} else {
			openMinions = append(openMinions, minion.Name)
		}
	}

	podCounts := map[string]int{}
	numServicePods := len(pods)
	if numServicePods > 0 {
		for _, pod := range pods {
			zone, exists := zonedMinions[pod.Status.Host]
			if !exists {
				continue
			}
			podCounts[zone]++
		}
	}

	result := []HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for minion := range zonedMinions {
		// initializing to the default/max minion score of 10
		fScore := float32(10)
		if numServicePods > 0 {
			fScore = 10 * (float32(numServicePods-podCounts[zonedMinions[minion]]) / float32(numServicePods))
		}
		result = append(result, HostPriority{host: minion, score: int(fScore)})
	}
	// add the open minions with a score of 0
	for _, minion := range openMinions {
		result = append(result, HostPriority{host: minion, score: 0})
	}

	return result, nil
}
