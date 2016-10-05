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
	"os"
	"path/filepath"
	"strconv"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	minShares     = 2
	sharesPerCPU  = 1024
	milliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	quotaPeriod    = 100000
	minQuotaPeriod = 1000
)

// milliCPUToQuota converts milliCPU to CFS quota and period values
func milliCPUToQuota(milliCPU int64) (quota int64, period int64) {
	// CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts

	if milliCPU == 0 {
		// take the default behavior from docker
		return
	}

	// we set the period to 100ms by default
	period = quotaPeriod

	// we then convert your milliCPU to a value normalized over a period
	quota = (milliCPU * quotaPeriod) / milliCPUToCPU

	// quota needs to be a minimum of 1ms.
	if quota < minQuotaPeriod {
		quota = minQuotaPeriod
	}

	return
}

func milliCPUToShares(milliCPU int64) int64 {
	if milliCPU == 0 {
		// Docker converts zero milliCPU to unset, which maps to kernel default
		// for unset: 1024. Return 2 here to really match kernel default for
		// zero milliCPU.
		return minShares
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * sharesPerCPU) / milliCPUToCPU
	if shares < minShares {
		return minShares
	}
	return shares
}

func ResourceConfigForPod(pod *api.Pod) *ResourceConfig {
	qosClass := qos.GetPodQOS(pod)

	cpuRequests := int64(0)
	cpuLimits := int64(0)
	memoryLimits := int64(0)
	memoryDeclared := true
	cpuDeclared := true

	for _, container := range pod.Spec.Containers {
		cpuRequests += container.Resources.Requests.Cpu().MilliValue()
		cpuLimits += container.Resources.Limits.Cpu().MilliValue()
		if container.Resources.Limits.Cpu().IsZero() {
			cpuDeclared = false
		}
		memoryLimits += container.Resources.Limits.Memory().Value()
		if container.Resources.Limits.Memory().IsZero() {
			memoryDeclared = false
		}
	}

	cpuShares := milliCPUToShares(cpuRequests)
	// TODO: this needs to expose period!!!
	cpuQuota, _ := milliCPUToQuota(cpuLimits)

	result := &ResourceConfig{}
	switch qosClass {
	case qos.Guaranteed:
		result.CpuQuota = &cpuQuota
		result.CpuShares = &cpuShares
		result.Memory = &memoryLimits
	case qos.Burstable:
		result.CpuShares = &cpuShares
		if cpuDeclared {
			result.CpuQuota = &cpuQuota
		}
		if memoryDeclared {
			result.Memory = &memoryLimits
		}
	case qos.BestEffort:
		shares := int64(minShares)
		result.CpuShares = &shares
	}

	fmt.Printf("cgroup manager: ResourceConfigForPod result: shares=%v, quota: %v, memory: %v\n", result.CpuShares, result.CpuQuota, result.Memory)
	return result
}

// ReduceCpuLimits reduces the cgroup's cpu shares to the lowest possible value
func ReduceCpuLimits(cgroupName CgroupName, subsystems *CgroupSubsystems) error {
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
