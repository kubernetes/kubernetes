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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
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
	return &resource.Quantity{}
}

// Returns the Memory limit if specified.
func (self *ResourceList) Memory() *resource.Quantity {
	if val, ok := (*self)[ResourceMemory]; ok {
		return &val
	}
	return &resource.Quantity{}
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

// IsPodReady retruns true if a pod is ready; false otherwise.
func IsPodReady(pod *Pod) bool {
	for _, c := range pod.Status.Conditions {
		if c.Type == PodReady && c.Status == ConditionTrue {
			return true
		}
	}
	return false
}
