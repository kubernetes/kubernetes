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
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	"k8s.io/kubernetes/pkg/scheduler/schedulercache"
	utilnode "k8s.io/kubernetes/pkg/util/node"

	"github.com/golang/glog"
)

// When zone information is present, give 2/3 of the weighting to zone spreading, 1/3 to node spreading
// TODO: Any way to justify this weighting?
const zoneWeighting float64 = 2.0 / 3.0

// SelectorSpread contains information to calculate selector spread priority.
type SelectorSpread struct {
	serviceLister     algorithm.ServiceLister
	controllerLister  algorithm.ControllerLister
	replicaSetLister  algorithm.ReplicaSetLister
	statefulSetLister algorithm.StatefulSetLister
	// ServiceAntiAffinity works on custom labels
	label string
}

// NewSelectorSpreadPriority creates a SelectorSpread.
func NewSelectorSpreadPriority(
	serviceLister algorithm.ServiceLister,
	controllerLister algorithm.ControllerLister,
	replicaSetLister algorithm.ReplicaSetLister,
	statefulSetLister algorithm.StatefulSetLister,
	label string) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {
	selectorSpread := &SelectorSpread{
		serviceLister:     serviceLister,
		controllerLister:  controllerLister,
		replicaSetLister:  replicaSetLister,
		statefulSetLister: statefulSetLister,
		label:             label,
	}
	return selectorSpread.CalculateSpreadPriorityMap, selectorSpread.CalculateSpreadPriorityReduce
}

// CalculateSpreadPriorityMap spreads pods across hosts, considering pods
// belonging to the same service,RC,RS or StatefulSet.
// When a pod is scheduled, it looks for services, RCs,RSs and StatefulSets that match the pod,
// then finds existing pods that match those selectors.
// It favors nodes that have fewer existing matching pods.
// i.e. it pushes the scheduler towards a node where there's the smallest number of
// pods which match the same service, RC,RSs or StatefulSets selectors as the pod being scheduled.
func (s *SelectorSpread) CalculateSpreadPriorityMap(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	var selectors []labels.Selector
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	priorityMeta, ok := meta.(*priorityMetadata)
	if ok {
		selectors = priorityMeta.podSelectors
	} else {
		selectors = getSelectors(pod, s.serviceLister, s.controllerLister, s.replicaSetLister, s.statefulSetLister)
	}

	if len(selectors) == 0 {
		return schedulerapi.HostPriority{
			Host:  node.Name,
			Score: int(0),
		}, nil
	}

	count := int(0)
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
	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: int(count),
	}, nil
}

// CalculateSpreadPriorityReduce calculates the source of each node
// based on the number of existing matching pods on the node
// where zone information is included on the nodes, it favors nodes
// in zones with fewer existing matching pods.
func (s *SelectorSpread) CalculateSpreadPriorityReduce(pod *v1.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {
	countsByZone := make(map[string]int, 10)
	maxCountByZone := int(0)
	maxCountByNodeName := int(0)

	for i := range result {
		if result[i].Score > maxCountByNodeName {
			maxCountByNodeName = result[i].Score
		}
		var zoneID string
		if len(s.label) > 0 {
			// ServiceAntiAffinity works on custom labels
			zoneID = utilnode.GetNodeLabelValue(nodeNameToInfo[result[i].Host].Node(), s.label)
		} else {
			zoneID = utilnode.GetZoneKey(nodeNameToInfo[result[i].Host].Node())
		}

		if zoneID == "" {
			continue
		}
		countsByZone[zoneID] += result[i].Score
	}

	for zoneID := range countsByZone {
		if countsByZone[zoneID] > maxCountByZone {
			maxCountByZone = countsByZone[zoneID]
		}
	}

	haveZones := len(countsByZone) != 0

	maxCountByNodeNameFloat64 := float64(maxCountByNodeName)
	maxCountByZoneFloat64 := float64(maxCountByZone)
	MaxPriorityFloat64 := float64(schedulerapi.MaxPriority)

	for i := range result {
		// initializing to the default/max node score of maxPriority
		fScore := MaxPriorityFloat64
		if maxCountByNodeName > 0 {
			fScore = MaxPriorityFloat64 * (float64(maxCountByNodeName-result[i].Score) / maxCountByNodeNameFloat64)
		}
		// If there is zone information present, incorporate it
		if haveZones {
			var zoneID string
			if len(s.label) > 0 {
				// ServiceAntiAffinity works on custom labels
				zoneID = utilnode.GetNodeLabelValue(nodeNameToInfo[result[i].Host].Node(), s.label)
			} else {
				zoneID = utilnode.GetZoneKey(nodeNameToInfo[result[i].Host].Node())
			}
			if zoneID != "" {
				zoneScore := MaxPriorityFloat64
				if maxCountByZone > 0 {
					zoneScore = MaxPriorityFloat64 * (float64(maxCountByZone-countsByZone[zoneID]) / maxCountByZoneFloat64)
				}
				fScore = (fScore * (1.0 - zoneWeighting)) + (zoneWeighting * zoneScore)
			}
		}
		result[i].Score = int(fScore)
		if glog.V(10) {
			glog.Infof(
				"%v -> %v: SelectorSpreadPriority, Score: (%d)", pod.Name, result[i].Host, int(fScore),
			)
		}
	}
	return nil
}

// NewServiceAntiAffinityPriority creates a ServiceAntiAffinity.
func NewServiceAntiAffinityPriority(
	serviceLister algorithm.ServiceLister,
	controllerLister algorithm.ControllerLister,
	replicaSetLister algorithm.ReplicaSetLister,
	statefulSetLister algorithm.StatefulSetLister,
	label string) (algorithm.PriorityMapFunction, algorithm.PriorityReduceFunction) {

	return NewSelectorSpreadPriority(serviceLister, controllerLister, replicaSetLister, statefulSetLister, label)
}
