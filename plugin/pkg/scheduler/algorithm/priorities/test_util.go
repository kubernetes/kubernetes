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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func makeNode(node string, milliCPU, memory int64) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: node},
		Status: api.NodeStatus{
			Capacity: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
			Allocatable: api.ResourceList{
				"cpu":    *resource.NewMilliQuantity(milliCPU, resource.DecimalSI),
				"memory": *resource.NewQuantity(memory, resource.BinarySI),
			},
		},
	}
}

func priorityFunction(mapFn algorithm.PriorityMapFunction, reduceFn algorithm.PriorityReduceFunction) algorithm.PriorityFunction {
	return func(pod *api.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*api.Node) (schedulerapi.HostPriorityList, error) {
		result := make(schedulerapi.HostPriorityList, 0, len(nodes))
		for i := range nodes {
			hostResult, err := mapFn(pod, nil, nodeNameToInfo[nodes[i].Name])
			if err != nil {
				return nil, err
			}
			result = append(result, hostResult)
		}
		if reduceFn != nil {
			if err := reduceFn(pod, nil, nodeNameToInfo, result); err != nil {
				return nil, err
			}
		}
		return result, nil
	}
}
