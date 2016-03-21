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
	"sync"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// The maximum priority value to give to a node
// Prioritiy values range from 0-maxPriority
const maxPriority = 10

// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
// TODO: Any way to justify this weighting?
const zoneWeighting = 2.0 / 3.0

type SelectorSpread struct {
	podLister        algorithm.PodLister
	serviceLister    algorithm.ServiceLister
	controllerLister algorithm.ControllerLister
	replicaSetLister algorithm.ReplicaSetLister
}

func NewSelectorSpreadPriority(podLister algorithm.PodLister, serviceLister algorithm.ServiceLister, controllerLister algorithm.ControllerLister, replicaSetLister algorithm.ReplicaSetLister) algorithm.PriorityFunction {
	selectorSpread := &SelectorSpread{
		podLister:        podLister,
		serviceLister:    serviceLister,
		controllerLister: controllerLister,
		replicaSetLister: replicaSetLister,
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

	// We include the null character just in case region or failureDomain has a colon
	// (We do assume there's no null characters in a region or failureDomain)
	// As a nice side-benefit, the null character is not printed by fmt.Print or glog
	return region + ":\x00:" + failureDomain
}

// CalculateSpreadPriority spreads pods across hosts and zones, considering pods belonging to the same service or replication controller.
// When a pod is scheduled, it looks for services or RCs that match the pod, then finds existing pods that match those selectors.
// It favors nodes that have fewer existing matching pods.
// i.e. it pushes the scheduler towards a node where there's the smallest number of
// pods which match the same service selectors or RC selectors as the pod being scheduled.
// Where zone information is included on the nodes, it favors nodes in zones with fewer existing matching pods.
func (s *SelectorSpread) CalculateSpreadPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	selectors := make([]labels.Selector, 0)
	services, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
		for _, service := range services {
			selectors = append(selectors, labels.SelectorFromSet(service.Spec.Selector))
		}
	}
	rcs, err := s.controllerLister.GetPodControllers(pod)
	if err == nil {
		for _, rc := range rcs {
			selectors = append(selectors, labels.SelectorFromSet(rc.Spec.Selector))
		}
	}
	rss, err := s.replicaSetLister.GetPodReplicaSets(pod)
	if err == nil {
		for _, rs := range rss {
			if selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector); err == nil {
				selectors = append(selectors, selector)
			}
		}
	}

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	// Count similar pods by node
	countsByNodeName := map[string]int{}
	countsByNodeNameLock := sync.Mutex{}

	if len(selectors) > 0 {
		// Create a number of go-routines that will be computing number
		// of "similar" pods for given nodes.
		workers := 16
		toProcess := make(chan string, len(nodes.Items))
		for i := range nodes.Items {
			toProcess <- nodes.Items[i].Name
		}
		close(toProcess)

		wg := sync.WaitGroup{}
		wg.Add(workers)
		for i := 0; i < workers; i++ {
			go func() {
				defer utilruntime.HandleCrash()
				defer wg.Done()
				for {
					nodeName, ok := <-toProcess
					if !ok {
						return
					}
					count := 0
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

					func() {
						countsByNodeNameLock.Lock()
						defer countsByNodeNameLock.Unlock()
						countsByNodeName[nodeName] = count
					}()
				}
			}()
		}
		wg.Wait()
	}

	// Aggregate by-node information
	// Compute the maximum number of pods hosted on any node
	maxCountByNodeName := 0
	for _, count := range countsByNodeName {
		if count > maxCountByNodeName {
			maxCountByNodeName = count
		}
	}

	// Count similar pods by zone, if zone information is present
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

		countsByZone[zoneId] += count
	}

	// Aggregate by-zone information
	// Compute the maximum number of pods hosted in any zone
	haveZones := len(countsByZone) != 0
	maxCountByZone := 0
	for _, count := range countsByZone {
		if count > maxCountByZone {
			maxCountByZone = count
		}
	}

	result := []schedulerapi.HostPriority{}
	//score int - scale of 0-maxPriority
	// 0 being the lowest priority and maxPriority being the highest
	for i := range nodes.Items {
		node := &nodes.Items[i]
		// initializing to the default/max node score of maxPriority
		fScore := float32(maxPriority)
		if maxCountByNodeName > 0 {
			fScore = maxPriority * (float32(maxCountByNodeName-countsByNodeName[node.Name]) / float32(maxCountByNodeName))
		}

		// If there is zone information present, incorporate it
		if haveZones {
			zoneId := getZoneKey(node)
			if zoneId != "" {
				zoneScore := maxPriority * (float32(maxCountByZone-countsByZone[zoneId]) / float32(maxCountByZone))
				fScore = (fScore * (1.0 - zoneWeighting)) + (zoneWeighting * zoneScore)
			}
		}

		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
		glog.V(10).Infof(
			"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, node.Name, int(fScore),
		)
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

// CalculateAntiAffinityPriority spreads pods by minimizing the number of pods belonging to the same service
// on machines with the same value for a particular label.
// The label to be considered is provided to the struct (ServiceAntiAffinity).
func (s *ServiceAntiAffinity) CalculateAntiAffinityPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	var nsServicePods []*api.Pod

	services, err := s.serviceLister.GetPodServices(pod)
	if err == nil {
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
