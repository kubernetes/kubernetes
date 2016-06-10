/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// NodeTaints hold the node lister
type TaintToleration struct {
	nodeLister algorithm.NodeLister
}

// NewTaintTolerationPriority
func NewTaintTolerationPriority(nodeLister algorithm.NodeLister) algorithm.PriorityFunction {
	taintToleration := &TaintToleration{
		nodeLister: nodeLister,
	}
	return taintToleration.ComputeTaintTolerationPriority
}

// CountIntolerableTaintsPreferNoSchedule gives the count of intolerable taints of a pod with effect PreferNoSchedule
func countIntolerableTaintsPreferNoSchedule(taints []api.Taint, tolerations []api.Toleration) (intolerableTaints int) {
	for _, taint := range taints {
		// check only on taints that have effect PreferNoSchedule
		if taint.Effect != api.TaintEffectPreferNoSchedule {
			continue
		}

		if !api.TaintToleratedByTolerations(taint, tolerations) {
			intolerableTaints++
		}
	}
	return
}

// getAllTolerationEffectPreferNoSchedule gets the list of all Toleration with Effect PreferNoSchedule
func getAllTolerationPreferNoSchedule(tolerations []api.Toleration) (tolerationList []api.Toleration) {
	for _, toleration := range tolerations {
		if len(toleration.Effect) == 0 || toleration.Effect == api.TaintEffectPreferNoSchedule {
			tolerationList = append(tolerationList, toleration)
		}
	}
	return
}

// ComputeTaintTolerationPriority prepares the priority list for all the nodes based on the number of intolerable taints on the node
func (s *TaintToleration) ComputeTaintTolerationPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodeLister algorithm.NodeLister) (schedulerapi.HostPriorityList, error) {
	// counts hold the count of intolerable taints of a pod for a given node
	counts := make(map[string]int)

	// the max value of counts
	var maxCount int

	nodes, err := nodeLister.List()
	if err != nil {
		return nil, err
	}

	tolerations, err := api.GetTolerationsFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}
	// Fetch a list of all toleration with effect PreferNoSchedule
	tolerationList := getAllTolerationPreferNoSchedule(tolerations)

	// calculate the intolerable taints for all the nodes
	for _, node := range nodes.Items {
		taints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
		if err != nil {
			return nil, err
		}

		count := countIntolerableTaintsPreferNoSchedule(taints, tolerationList)
		counts[node.Name] = count
		if count > maxCount {
			maxCount = count
		}
	}

	// The maximum priority value to give to a node
	// Priority values range from 0 - maxPriority
	const maxPriority = 10
	result := []schedulerapi.HostPriority{}
	for _, node := range nodes.Items {
		fScore := float64(maxPriority)
		if maxCount > 0 {
			fScore = (1.0 - float64(counts[node.Name])/float64(maxCount)) * 10
		}
		glog.V(10).Infof("%v -> %v: Taint Toleration Priority, Score: (%d)", pod.Name, node.Name, int(fScore))

		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
	}
	return result, nil
}
