/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"

	"github.com/golang/glog"
)

// The maximum priority value to give to a node
// Prioritiy values range from 0-maxPriority
const maxPriority float32 = 10

// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
// TODO: Any way to justify this weighting?
const zoneWeighting = 2.0 / 3.0

type SelectorSpread struct {
	serviceLister    algorithm.ServiceLister
	controllerLister algorithm.ControllerLister
	replicaSetLister algorithm.ReplicaSetLister
}

func NewSelectorSpreadPriority(
	serviceLister algorithm.ServiceLister,
	controllerLister algorithm.ControllerLister,
	replicaSetLister algorithm.ReplicaSetLister) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
	selectorSpread := &SelectorSpread{
		serviceLister:    serviceLister,
		controllerLister: controllerLister,
		replicaSetLister: replicaSetLister,
	}
	return selectorSpread.CalculateSpreadPriorityMap, selectorSpread.CalculateSpreadPriorityReduce
}

func (s *SelectorSpread) CalculateSpreadPriorityMap(pod *api.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	nodeName := nodeInfo.Node().Name
	var selectors []labels.Selector
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		selectors = priorityMeta.controllerSelectors
	} else {
		// We couldn't parse metadata - fallback to computing it.
		selectors = getSelectors(pod, s.serviceLister, s.controllerLister, s.replicaSetLister)
	}

	if len(selectors) == 0 {
		return schedulerapi.HostPriority{Host: nodeName, Score: 10}, nil
	}

	count := 0
	for _, nodePod := range nodeInfo.Pods() {
		if pod.Namespace != nodePod.Namespace {
			continue
		}
		// When we are replacing a failed pod, we often see the previous
		// deleted version while scheduling the replacement.
		// Ignore the previous deleted version for spreading purposes
		// (it can still be considered for resource restrictions etc.)
		if nodePod.DeletionTimestamp != nil {
			glog.V(4).Infof("skipping pending-deleted pod: %s/%s", nodePod.Namespace, nodePod.Name)
			continue
		}
		matches := false
		for _, selector := range selectors {
			if selector.Matches(labels.Set(nodePod.ObjectMeta.Labels)) {
				matches = true
				break
			}
		}
		if matches {
			count++
		}
	}

	return schedulerapi.HostPriority{Host: nodeName, Score: count}, nil
}

func (s *SelectorSpread) CalculateSpreadPriorityReduce(pod *api.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {
	var selectors []labels.Selector
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		selectors = priorityMeta.controllerSelectors
	} else {
		// We couldn't parse metadata - fallback to computing it.
		selectors = getSelectors(pod, s.serviceLister, s.controllerLister, s.replicaSetLister)
	}

	if len(selectors) == 0 {
		return nil
	}

	countsByNodeName := make(map[string]float32, len(result))
	maxCountByNodeName := float32(0)
	countsByZone := make(map[string]float32, 10)

	for i := range result {
		nodeName := result[i].Host
		count := float32(result[i].Score)
		zoneId := nodeNameToInfo[nodeName].Zone()
		countsByNodeName[nodeName] = count

		if count > maxCountByNodeName {
			maxCountByNodeName = count
		}
		if zoneId != "" {
			countsByZone[zoneId] += count
		}
	}

	// Aggregate by-zone information
	// Compute the maximum number of pods hosted in any zone
	haveZones := len(countsByZone) != 0
	maxCountByZone := float32(0)
	for _, count := range countsByZone {
		if count > maxCountByZone {
			maxCountByZone = count
		}
	}

	//score int - scale of 0-maxPriority
	// 0 being the lowest priority and maxPriority being the highest
	for i := range result {
		nodeName := result[i].Host
		// initializing to the default/max node score of maxPriority
		fScore := maxPriority
		if maxCountByNodeName > 0 {
			fScore = maxPriority * ((maxCountByNodeName - countsByNodeName[nodeName]) / maxCountByNodeName)
		}

		// If there is zone information present, incorporate it
		if haveZones {
			zoneId := nodeNameToInfo[nodeName].Zone()
			if zoneId != "" {
				zoneScore := maxPriority * ((maxCountByZone - countsByZone[zoneId]) / maxCountByZone)
				fScore = (fScore * (1.0 - zoneWeighting)) + (zoneWeighting * zoneScore)
			}
		}

		result[i].Score = int(fScore)
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.V(10).Infof(
				"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, nodeName, int(fScore),
			)
		}
	}

	return nil
}

type ServiceAntiAffinity struct {
	podLister     algorithm.PodLister
	serviceLister algorithm.ServiceLister
	label         string
}

func NewServiceAntiAffinityPriority(podLister algorithm.PodLister, serviceLister algorithm.ServiceLister, label string) algorithm.PriorityFunction {
	antiAffinity := &ServiceAntiAffinity{
		podLister:     podLister,
		serviceLister: serviceLister,
		label:         label,
	}
	return antiAffinity.CalculateAntiAffinityPriority
}

// CalculateAntiAffinityPriority spreads pods by minimizing the number of pods belonging to the same service
// on machines with the same value for a particular label.
// The label to be considered is provided to the struct (ServiceAntiAffinity).
func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*api.Node) (schedulerapi.HostPriorityList, error) {
	var nsServicePods []*api.Pod
	if services, err := s.serviceLister.GetPodServices(pod); err == nil && len(services) > 0 {
		// just use the first service and get the other pods within the service
		// TODO: a separate predicate can be created that tries to handle all services for the pod
		selector := labels.SelectorFromSet(services[0].Spec.Selector)
		pods, err := s.podLister.List(selector)
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

	// separate out the nodes that have the label from the ones that don't
	otherNodes := []string{}
	labeledNodes := map[string]string{}
	for _, node := range nodes {
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
	//score int - scale of 0-maxPriority
	// 0 being the lowest priority and maxPriority being the highest
	for node := range labeledNodes {
		// initializing to the default/max node score of maxPriority
		fScore := float32(maxPriority)
		if numServicePods > 0 {
			fScore = maxPriority * (float32(numServicePods-podCounts[labeledNodes[node]]) / float32(numServicePods))
		}
		result = append(result, schedulerapi.HostPriority{Host: node, Score: int(fScore)})
	}
	// add the open nodes with a score of 0
	for _, node := range otherNodes {
		result = append(result, schedulerapi.HostPriority{Host: node, Score: 0})
	}

	return result, nil
}
