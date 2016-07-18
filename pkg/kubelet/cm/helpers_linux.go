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

package cm

import (
	"fmt"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/quota"
)

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*cgroupSubsystems, error) {
	// get all cgroup mounts.
	allCgroups, err := libcontainercgroups.GetCgroupMounts()
	if err != nil {
		return &cgroupSubsystems{}, err
	}
	if len(allCgroups) == 0 {
		return &cgroupSubsystems{}, fmt.Errorf("failed to find cgroup mounts")
	}

	mountPoints := make(map[string]string, len(allCgroups))
	for _, mount := range allCgroups {
		for _, subsystem := range mount.Subsystems {
			mountPoints[subsystem] = mount.Mountpoint
		}
	}
	return &cgroupSubsystems{
		mounts:      allCgroups,
		mountPoints: mountPoints,
	}, nil
}

// GetPodResourceRequest returns the pod requests for the supported resources.
// Pod request is the summation of resource requests of all containers in the pod.
func GetPodResourceRequests(pod *api.Pod) api.ResourceList {
	requests := api.ResourceList{}
	for _, container := range pod.Spec.Containers {
		requests = quota.Add(requests, container.Resources.Requests)
	}
	return requests
}

// GetPodResourceLimits returns the pod limits for the supported resources
// Pod limit is the summation of resource limits of all containers
// in the pod. If limit for a particular resource is not specified for
// even a single container then we return the node resource Allocatable
// as the pod limit for the particular resource.
func GetPodResourceLimits(pod *api.Pod, nodeInfo *api.Node) api.ResourceList {
	allocatable := nodeInfo.Status.Allocatable
	limits := api.ResourceList{}
	for _, resource := range []api.ResourceName{api.ResourceCPU, api.ResourceMemory} {
		for _, container := range pod.Spec.Containers {
			quantity, exists := container.Resources.Limits[resource]
			if exists && !quantity.IsZero() {
				delta := quantity.Copy()
				if _, exists := limits[resource]; !exists {
					limits[resource] = *delta
				} else {
					delta.Add(limits[resource])
					limits[resource] = *delta
				}
			} else {
				// if limit not specified for a particular resource in a container
				// we default the pod resource limit to the resource allocatable of the node
				if alo, exists := allocatable[resource]; exists {
					limits[resource] = *alo.Copy()
					break
				}
			}
		}
	}
	return limits
}
