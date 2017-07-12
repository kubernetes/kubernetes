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
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/workqueue"
	utilnode "k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
// TODO: Any way to justify this weighting?
const zoneWeighting float64 = 2.0 / 3.0

type SelectorSpread struct {
	serviceLister     algorithm.ServiceLister
	controllerLister  algorithm.ControllerLister
	replicaSetLister  algorithm.ReplicaSetLister
	statefulSetLister algorithm.StatefulSetLister
}

func NewSelectorSpreadPriority(
	serviceLister algorithm.ServiceLister,
	controllerLister algorithm.ControllerLister,
	replicaSetLister algorithm.ReplicaSetLister,
	statefulSetLister algorithm.StatefulSetLister) algorithm.PriorityFunction {
	selectorSpread := &SelectorSpread{
		serviceLister:     serviceLister,
		controllerLister:  controllerLister,
		replicaSetLister:  replicaSetLister,
		statefulSetLister: statefulSetLister,
	}
	return selectorSpread.CalculateSpreadPriority
}

// Returns selectors of services, RCs and RSs matching the given pod.
func getSelectors(pod *v1.Pod, sl algorithm.ServiceLister, cl algorithm.ControllerLister, rsl algorithm.ReplicaSetLister, ssl algorithm.StatefulSetLister) []labels.Selector {
	var selectors []labels.Selector
	if services, err := sl.GetPodServices(pod); err == nil {
		for _, service := range services {
			selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
		}
	}
	if rcs, err := cl.GetPodControllers(pod); err == nil {
		for _, rc := range rcs {
			selectors = append(selectors, labels.SelectorFromSet(rc.Spec.Selector))
		}
	}
	if rss, err := rsl.GetPodReplicaSets(pod); err == nil {
		for _, rs := range rss {
			if selector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
				selectors = append(selectors, selector)
			}
		}
	}
	if sss, err := ssl.GetPodStatefulSets(pod); err == nil {
		for _, ss := range sss {
			if selector, err := metav1.LabelSelectorAsSelector(ss.Spec.Selector); err == nil {
				selectors = append(selectors, selector)
			}
		}
	}
	return selectors
}

func (s *SelectorSpread) getSelectors(pod *v1.Pod) []labels.Selector {
	return getSelectors(pod, s.serviceLister, s.controllerLister, s.replicaSetLister, s.statefulSetLister)
}

// CalculateSpreadPriority spreads pods across hosts and zones, considering pods belonging to the same service or replication controller.
// When a pod is scheduled, it looks for services, RCs or RSs that match the pod, then finds existing pods that match those selectors.
// It favors nodes that have fewer existing matching pods.
// i.e. it pushes the scheduler towards a node where there's the smallest number of
// pods which match the same service, RC or RS selectors as the pod being scheduled.
// Where zone information is included on the nodes, it favors nodes in zones with fewer existing matching pods.
func (s *SelectorSpread) CalculateSpreadPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	selectors := s.getSelectors(pod)

	// Count similar pods by node
	countsByNodeName := make(map[string]float64, len(nodes))
	countsByZone := make(map[string]float64, 10)
	maxCountByNodeName := float64(0)
	countsByNodeNameLock := sync.Mutex{}

	if len(selectors) > 0 {
		processNodeFunc := func(i int) {
			nodeName := nodes[i].Name
			count := float64(0)
			for _, nodePod := range nodeNameToInfo[nodeName].Pods() {
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
			zoneId := utilnode.GetZoneKey(nodes[i])

			countsByNodeNameLock.Lock()
			defer countsByNodeNameLock.Unlock()
			countsByNodeName[nodeName] = count
			if count > maxCountByNodeName {
				maxCountByNodeName = count
			}
			if zoneId != "" {
				countsByZone[zoneId] += count
			}
		}
		workqueue.Parallelize(16, len(nodes), processNodeFunc)
	}

	// Aggregate by-zone information
	// Compute the maximum number of pods hosted in any zone
	haveZones := len(countsByZone) != 0
	maxCountByZone := float64(0)
	for _, count := range countsByZone {
		if count > maxCountByZone {
			maxCountByZone = count
		}
	}

	result := make(schedulerapi.HostPriorityList, 0, len(nodes))
	//score int - scale of 0-maxPriority
	// 0 being the lowest priority and maxPriority being the highest
	for _, node := range nodes {
		// initializing to the default/max node score of maxPriority
		fScore := float64(schedulerapi.MaxPriority)
		if maxCountByNodeName > 0 {
			fScore = float64(schedulerapi.MaxPriority) * ((maxCountByNodeName - countsByNodeName[node.Name]) / maxCountByNodeName)
		}

		// If there is zone information present, incorporate it
		if haveZones {
			zoneId := utilnode.GetZoneKey(node)
			if zoneId != "" {
				zoneScore := float64(schedulerapi.MaxPriority) * ((maxCountByZone - countsByZone[zoneId]) / maxCountByZone)
				fScore = (fScore * (1.0 - zoneWeighting)) + (zoneWeighting * zoneScore)
			}
		}

		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.V(10).Infof(
				"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, node.Name, int(fScore),
			)
		}
	}
	return result, nil
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

// Classifies nodes into ones with labels and without labels.
func (s *ServiceAntiAffinity) getNodeClassificationByLabels(nodes []*v1.Node) (map[string]string, []string) {
	labeledNodes := map[string]string{}
	nonLabeledNodes := []string{}
	for _, node := range nodes {
		if labels.Set(node.Labels).Has(s.label) {
			label := labels.Set(node.Labels).Get(s.label)
			labeledNodes[node.Name] = label
		} else {
			nonLabeledNodes = append(nonLabeledNodes, node.Name)
		}
	}
	return labeledNodes, nonLabeledNodes
}

// CalculateAntiAffinityPriority spreads pods by minimizing the number of pods belonging to the same service
// on machines with the same value for a particular label.
// The label to be considered is provided to the struct (ServiceAntiAffinity).
func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	var nsServicePods []*v1.Pod
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
	labeledNodes, nonLabeledNodes := s.getNodeClassificationByLabels(nodes)
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
		fScore := float64(schedulerapi.MaxPriority)
		if numServicePods > 0 {
			fScore = float64(schedulerapi.MaxPriority) * (float64(numServicePods-podCounts[labeledNodes[node]]) / float64(numServicePods))
		}
		result = append(result, schedulerapi.HostPriority{Host: node, Score: int(fScore)})
	}
	// add the open nodes with a score of 0
	for _, node := range nonLabeledNodes {
		result = append(result, schedulerapi.HostPriority{Host: node, Score: 0})
	}
	return result, nil
}
