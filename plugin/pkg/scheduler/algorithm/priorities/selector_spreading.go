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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
)

type SelectorSpread struct {
	serviceLister    algorithm.ServiceLister
	controllerLister algorithm.ControllerLister
}

func NewSelectorSpreadPriority(serviceLister algorithm.ServiceLister, controllerLister algorithm.ControllerLister) algorithm.PriorityFunction {
	selectorSpread := &SelectorSpread{
		serviceLister:    serviceLister,
		controllerLister: controllerLister,
	}
	return selectorSpread.CalculateSpreadPriority
}

// Helper function that builds a string identifier that is unique per failure-zone
// Returns empty-string for no zone
func getZoneKey(node *api.Node) string {
	labels := node.Labels
	if labels == nil {
		return ""
	}

	region, _ := labels[unversioned.LabelZoneRegion]
	failureDomain, _ := labels[unversioned.LabelZoneFailureDomain]

	if region == "" && failureDomain == "" {
		return ""
	}

	return region + ":" + failureDomain
}

// CalculateSpreadPriority spreads pods by minimizing the number of pods belonging to the same service or replication controller. It counts number of pods that run under
// Services or RCs as the pod being scheduled and tries to minimize the number of conflicts. I.e. pushes scheduler towards a Node where there's a smallest number of
// pods which match the same selectors of Services and RCs as current pod.
func (s *SelectorSpread) CalculateSpreadPriority(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	var nsPods []*api.Pod

	selectors := make([]labels.Selector, 0)
	services, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
		for _, service := range services {
			selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
		}
	}
	controllers, err := s.controllerLister.GetPodControllers(pod)
	if err == nil {
		for _, controller := range controllers {
			selectors = append(selectors, labels.SelectorFromSet(controller.Spec.Selector))
		}
	}

	if len(selectors) > 0 {
		pods, err := podLister.List(labels.Everything())
		if err != nil {
			return nil, err
		}
		// consider only the pods that belong to the same namespace
		for _, nsPod := range pods {
			if nsPod.Namespace == pod.Namespace {
				nsPods = append(nsPods, nsPod)
			}
		}
	}

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	maxCountByNodeName := 0
	countsByNodeName := map[string]int{}
	if len(nsPods) > 0 {
		for _, pod := range nsPods {
			// When we are replacing a failed pod, we often see the previous deleted version
			// while scheduling the replacement.  Ignore the previous deleted version for spreading
			// purposes (it can still be considered for resource restrictions etc.)
			if pod.DeletionTimestamp != nil {
				glog.V(2).Infof("skipping pending-deleted pod: %s/%s", pod.Namespace, pod.Name)
				continue
			}
			matches := false
			for _, selector := range selectors {
				if selector.Matches(labels.Set(pod.ObjectMeta.Labels)) {
					matches = true
					break
				}
			}
			if matches {
				countsByNodeName[pod.Spec.NodeName]++
				// Compute the maximum number of pods hosted on any node
				if countsByNodeName[pod.Spec.NodeName] > maxCountByNodeName {
					maxCountByNodeName = countsByNodeName[pod.Spec.NodeName]
				}
			}
		}
	}

	maxCountByZone := 0
	haveZones := false
	countsByZone := map[string]int{}
	for i := range nodes.Items {
		node := &nodes.Items[i]

		count, found := countsByNodeName[node.Name]
		if !found {
			continue
		}

		zoneId := getZoneKey(node)
		if zoneId == "" {
			continue
		}

		haveZones = true
		countsByZone[zoneId] += count
		// Compute the maximum number of pods hosted in any zone
		if countsByZone[zoneId] > maxCountByZone {
			maxCountByZone = countsByZone[zoneId]
		}
	}

	result := []schedulerapi.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for i := range nodes.Items {
		node := &nodes.Items[i]
		// initializing to the default/max node score of 10
		fScore := float32(10)
		if maxCountByNodeName > 0 {
			fScore = 10 * (float32(maxCountByNodeName-countsByNodeName[node.Name]) / float32(maxCountByNodeName))
		}

		// If there is zone information present, incorporate it
		if haveZones {
			zoneId := getZoneKey(node)
			if zoneId != "" {
				fScore += 20 * (float32(maxCountByZone-countsByZone[zoneId]) / float32(maxCountByZone))
			}

			// Give 2/3 of the weighting to zone spreading, 1/3 to node spreading
			// TODO: Any way to justify this weighting?
			fScore /= 3.0
		}

		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
		glog.V(10).Infof(
			"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, node.Name, int(fScore),
		)
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
func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod *api.Pod, podLister algorithm.PodLister, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
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

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	// separate out the nodes that have the label from the ones that don't
	otherNodes := []string{}
	labeledNodes := map[string]string{}
	for _, node := range nodes.Items {
		if labels.Set(node.Labels).Has(s.label) {
			label := labels.Set(node.Labels).Get(s.label)
			labeledNodes[node.Name] = label
		} else {
			otherNodes = append(otherNodes, node.Name)
		}
	}

	podCounts := map[string]int{}
	for _, pod := range nsServicePods {
		label, exists := labeledNodes[pod.Spec.NodeName]
		if !exists {
			continue
		}
		podCounts[label]++
	}

	numServicePods := len(nsServicePods)
	result := []schedulerapi.HostPriority{}
	//score int - scale of 0-10
	// 0 being the lowest priority and 10 being the highest
	for node := range labeledNodes {
		// initializing to the default/max node score of 10
		fScore := float32(10)
		if numServicePods > 0 {
			fScore = 10 * (float32(numServicePods-podCounts[labeledNodes[node]]) / float32(numServicePods))
		}
		result = append(result, schedulerapi.HostPriority{Host: node, Score: int(fScore)})
	}
	// add the open nodes with a score of 0
	for _, node := range otherNodes {
		result = append(result, schedulerapi.HostPriority{Host: node, Score: 0})
	}

	return result, nil
}
