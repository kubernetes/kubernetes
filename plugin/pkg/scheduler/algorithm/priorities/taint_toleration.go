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
	"fmt"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

// CountIntolerableTaintsPreferNoSchedule gives the count of intolerable taints of a pod with effect PreferNoSchedule
func countIntolerableTaintsPreferNoSchedule(taints []api.Taint, tolerations []api.Toleration) (intolerableTaints int) {
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

func getTolerationListFromPod(pod *api.Pod) ([]api.Toleration, error) {
	tolerations, err := api.GetTolerationsFromPodAnnotations(pod.Annotations)
	if err != nil {
		return nil, err
	}
	return getAllTolerationPreferNoSchedule(tolerations), nil
}

// ComputeTaintTolerationPriority prepares the priority list for all the nodes based on the number of intolerable taints on the node
func ComputeTaintTolerationPriorityMap(pod *api.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (schedulerapi.HostPriority, error) {
	node := nodeInfo.Node()
	if node == nil {
		return schedulerapi.HostPriority{}, fmt.Errorf("node not found")
	}

	var tolerationList []api.Toleration
	if priorityMeta, ok := meta.(*priorityMetadata); ok {
		tolerationList = priorityMeta.podTolerations
	} else {
		var err error
		tolerationList, err = getTolerationListFromPod(pod)
		if err != nil {
			return schedulerapi.HostPriority{}, err
		}
	}

	taints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
	if err != nil {
		return schedulerapi.HostPriority{}, err
	}
	return schedulerapi.HostPriority{
		Host:  node.Name,
		Score: countIntolerableTaintsPreferNoSchedule(taints, tolerationList),
	}, nil
}

func ComputeTaintTolerationPriorityReduce(pod *api.Pod, meta interface{}, nodeNameToInfo map[string]*schedulercache.NodeInfo, result schedulerapi.HostPriorityList) error {
	var maxCount int
	for i := range result {
		if result[i].Score > maxCount {
			maxCount = result[i].Score
		}
	}
	maxCountFloat := float64(maxCount)

	// The maximum priority value to give to a node
	// Priority values range from 0 - maxPriority
	const maxPriority = float64(10)
	for i := range result {
		fScore := maxPriority
		if maxCountFloat > 0 {
			fScore = (1.0 - float64(result[i].Score)/maxCountFloat) * 10
		}
		if glog.V(10) {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.Infof("%v -> %v: Taint Toleration Priority, Score: (%d)", pod.Name, result[i].Host, int(fScore))
		}
		result[i].Score = int(fScore)
	}
	return nil
}
