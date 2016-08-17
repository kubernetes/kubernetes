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
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// CountIntolerableTaintsPreferNoSchedule gives the count of intolerable taints of a pod with effect PreferNoSchedule
func countIntolerableTaintsPreferNoSchedule(taints []api.Taint, tolerations []api.Toleration) (intolerableTaints float64) {
	for i := range taints {
		taint := &taints[i]
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
	for i := range tolerations {
		toleration := &tolerations[i]
		if len(toleration.Effect) == 0 || toleration.Effect == api.TaintEffectPreferNoSchedule {
			tolerationList = append(tolerationList, *toleration)
		}
	}
	return
}

// ComputeTaintTolerationPriority prepares the priority list for all the nodes based on the number of intolerable taints on the node
func ComputeTaintTolerationPriority(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*api.Node) (schedulerapi.HostPriorityList, error) {
	// the max value of counts
	var maxCount float64
	// counts hold the count of intolerable taints of a pod for a given node
	counts := make(map[string]float64, len(nodes))

	tolerations, err := api.GetTolerationsFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}
	// Fetch a list of all toleration with effect PreferNoSchedule
	tolerationList := getAllTolerationPreferNoSchedule(tolerations)

	// calculate the intolerable taints for all the nodes
	for _, node := range nodes {
		taints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
		if err != nil {
			return nil, err
		}

		count := countIntolerableTaintsPreferNoSchedule(taints, tolerationList)
		if count > 0 {
			// 0 is default value, so avoid unnecessary map operations.
			counts[node.Name] = count
			if count > maxCount {
				maxCount = count
			}
		}
	}

	// The maximum priority value to give to a node
	// Priority values range from 0 - maxPriority
	const maxPriority = float64(10)
	result := make(schedulerapi.HostPriorityList, 0, len(nodes))
	for _, node := range nodes {
		fScore := maxPriority
		if maxCount > 0 {
			fScore = (1.0 - counts[node.Name]/maxCount) * 10
		}
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.Infof("%v -> %v: Taint Toleration Priority, Score: (%d)", pod.Name, node.Name, int(fScore))
		}

		result = append(result, schedulerapi.HostPriority{Host: node.Name, Score: int(fScore)})
	}
	return result, nil
}
