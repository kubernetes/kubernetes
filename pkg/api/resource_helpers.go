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

package api

import (
	"k8s.io/kubernetes/pkg/api/resource"
)

// Returns string version of ResourceName.
func (self ResourceName) String() string {
	return string(self)
}

// Returns the CPU limit if specified.
func (self *ResourceList) Cpu() *resource.Quantity {
	if val, ok := (*self)[ResourceCPU]; ok {
		return &val
	}
	return &resource.Quantity{Format: resource.DecimalSI}
}

// Returns the Memory limit if specified.
func (self *ResourceList) Memory() *resource.Quantity {
	if val, ok := (*self)[ResourceMemory]; ok {
		return &val
	}
	return &resource.Quantity{Format: resource.BinarySI}
}

func (self *ResourceList) Pods() *resource.Quantity {
	if val, ok := (*self)[ResourcePods]; ok {
		return &val
	}
	return &resource.Quantity{}
}

func GetContainerStatus(statuses []ContainerStatus, name string) (ContainerStatus, bool) {
	for i := range statuses {
		if statuses[i].Name == name {
			return statuses[i], true
		}
	}
	return ContainerStatus{}, false
}

func GetExistingContainerStatus(statuses []ContainerStatus, name string) ContainerStatus {
	for i := range statuses {
		if statuses[i].Name == name {
			return statuses[i]
		}
	}
	return ContainerStatus{}
}

// IsPodReady returns true if a pod is ready; false otherwise.
func IsPodReady(pod *Pod) bool {
	return IsPodReadyConditionTrue(pod.Status)
}

// IsPodReady retruns true if a pod is ready; false otherwise.
func IsPodReadyConditionTrue(status PodStatus) bool {
	condition := GetPodReadyCondition(status)
	return condition != nil && condition.Status == ConditionTrue
}

// Extracts the pod ready condition from the given status and returns that.
// Returns nil if the condition is not present.
func GetPodReadyCondition(status PodStatus) *PodCondition {
	for i, c := range status.Conditions {
		if c.Type == PodReady {
			return &status.Conditions[i]
		}
	}
	return nil
}

// IsNodeReady returns true if a node is ready; false otherwise.
func IsNodeReady(node *Node) bool {
	for _, c := range node.Status.Conditions {
		if c.Type == NodeReady {
			return c.Status == ConditionTrue
		}
	}
	return false
}

// PodRequestsAndLimits returns a dictionary of all defined resources summed up for all
// containers of the pod.
func PodRequestsAndLimits(pod *Pod) (reqs map[ResourceName]resource.Quantity, limits map[ResourceName]resource.Quantity, err error) {
	reqs, limits = map[ResourceName]resource.Quantity{}, map[ResourceName]resource.Quantity{}
	for _, container := range pod.Spec.Containers {
		for name, quantity := range container.Resources.Requests {
			if value, ok := reqs[name]; !ok {
				reqs[name] = *quantity.Copy()
			} else if err = value.Add(quantity); err != nil {
				return nil, nil, err
			}
		}
		for name, quantity := range container.Resources.Limits {
			if value, ok := limits[name]; !ok {
				limits[name] = *quantity.Copy()
			} else if err = value.Add(quantity); err != nil {
				return nil, nil, err
			}
		}
	}
	return
}
