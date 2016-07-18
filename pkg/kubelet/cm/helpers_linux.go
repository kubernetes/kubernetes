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
	"bufio"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/quota"
	"k8s.io/kubernetes/pkg/types"
)

// ReduceCpuLimits reduces the cgroup's cpu shares to the lowest possible value
func ReduceCpuLimits(cgroupName string, subsystems *CgroupSubsystems) error {
	// Set lowest possible CpuShares value for the cgroup
	minimumCPUShares := int64(2)
	resources := &ResourceConfig{
		CpuShares: &minimumCPUShares,
	}
	containerConfig := &CgroupConfig{
		Name:               cgroupName,
		ResourceParameters: resources,
	}
	cgroupManager := NewCgroupManager(subsystems)
	err := cgroupManager.Update(containerConfig)
	if err != nil {
		return fmt.Errorf("failed to update %v cgroup: %v", cgroupName, err)
	}
	return nil
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*CgroupSubsystems, error) {
	// get all cgroup mounts.
	allCgroups, err := libcontainercgroups.GetCgroupMounts()
	if err != nil {
		return &CgroupSubsystems{}, err
	}
	if len(allCgroups) == 0 {
		return &CgroupSubsystems{}, fmt.Errorf("failed to find cgroup mounts")
	}
	mountPoints := make(map[string]string, len(allCgroups))
	for _, mount := range allCgroups {
		for _, subsystem := range mount.Subsystems {
			mountPoints[subsystem] = mount.Mountpoint
		}
	}
	return &CgroupSubsystems{
		Mounts:      allCgroups,
		MountPoints: mountPoints,
	}, nil
}

// getCgroupProcs takes a cgroup directory name as an argument
// reads through the cgroup's procs file and returns a list of tgid's.
// It returns an empty list if a procs file doesn't exists
func getCgroupProcs(dir string) ([]int, error) {
	procsFile := filepath.Join(dir, "cgroup.procs")
	_, err := os.Stat(procsFile)
	if os.IsNotExist(err) {
		// The procsFile does not exist, So no pids attached to this directory
		return []int{}, nil
	}
	f, err := os.Open(procsFile)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var (
		s   = bufio.NewScanner(f)
		out = []int{}
	)

	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, err
			}
			out = append(out, pid)
		}
	}
	return out, nil
}

// GetAllPodsFromCgroups scans through all the subsytems of pod cgroups
// Get list of pods whose cgroup still exist on the cgroup mounts
func GetAllPodsFromCgroups(subsystems *CgroupSubsystems, qosContainers QOSContainersInfo) (map[types.UID]string, error) {
	// Map for storing all the found pods on the disk
	foundPods := make(map[types.UID]string)
	qosContainersList := [3]string{qosContainers.Guaranteed, qosContainers.Burstable, qosContainers.BestEffort}
	// Scan through all the subsystem mounts
	// and through each QoS cgroup directory for each subsystem mount
	// If a pod cgroup exists in even a single subsystem mount
	// we will attempt to delete it
	for _, val := range subsystems.MountPoints {
		for _, name := range qosContainersList {
			// get the subsystems QoS cgroup absolute name
			qc := path.Join(val, name)
			dirInfo, err := ioutil.ReadDir(qc)
			if err != nil {
				return nil, fmt.Errorf("failed to read the cgroup directory %v : %v", qc, err)
			}
			for i := range dirInfo {
				if dirInfo[i].IsDir() && strings.HasPrefix(dirInfo[i].Name(), podCgroupNamePrefix) {
					podUID := strings.TrimPrefix(dirInfo[i].Name(), podCgroupNamePrefix)
					foundPods[types.UID(podUID)] = path.Join(name, dirInfo[i].Name())
				}
			}
		}
	}
	return foundPods, nil
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
