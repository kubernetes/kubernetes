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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
)

const (
	// Taken from lmctfy https://github.com/google/lmctfy/blob/master/lmctfy/controllers/cpu_controller.cc
	MinShares     = 2
	SharesPerCPU  = 1024
	MilliCPUToCPU = 1000

	// 100000 is equivalent to 100ms
	QuotaPeriod    = 100000
	MinQuotaPeriod = 1000
)

// MilliCPUToQuota converts milliCPU to CFS quota and period values.
func MilliCPUToQuota(milliCPU int64) (quota int64, period uint64) {
	// CFS quota is measured in two values:
	//  - cfs_period_us=100ms (the amount of time to measure usage across)
	//  - cfs_quota=20ms (the amount of cpu time allowed to be used across a period)
	// so in the above example, you are limited to 20% of a single CPU
	// for multi-cpu environments, you just scale equivalent amounts

	if milliCPU == 0 {
		return
	}

	// we set the period to 100ms by default
	period = QuotaPeriod

	// we then convert your milliCPU to a value normalized over a period
	quota = (milliCPU * QuotaPeriod) / MilliCPUToCPU

	// quota needs to be a minimum of 1ms.
	if quota < MinQuotaPeriod {
		quota = MinQuotaPeriod
	}

	return
}

// MilliCPUToShares converts the milliCPU to CFS shares.
func MilliCPUToShares(milliCPU int64) uint64 {
	if milliCPU == 0 {
		// Docker converts zero milliCPU to unset, which maps to kernel default
		// for unset: 1024. Return 2 here to really match kernel default for
		// zero milliCPU.
		return MinShares
	}
	// Conceptually (milliCPU / milliCPUToCPU) * sharesPerCPU, but factored to improve rounding.
	shares := (milliCPU * SharesPerCPU) / MilliCPUToCPU
	if shares < MinShares {
		return MinShares
	}
	return uint64(shares)
}

// HugePageLimits converts the API representation to a map
// from huge page size (in bytes) to huge page limit (in bytes).
func HugePageLimits(resourceList v1.ResourceList) map[int64]int64 {
	hugePageLimits := map[int64]int64{}
	for k, v := range resourceList {
		if v1helper.IsHugePageResourceName(k) {
			pageSize, _ := v1helper.HugePageSizeFromResourceName(k)
			if value, exists := hugePageLimits[pageSize.Value()]; exists {
				hugePageLimits[pageSize.Value()] = value + v.Value()
			} else {
				hugePageLimits[pageSize.Value()] = v.Value()
			}
		}
	}
	return hugePageLimits
}

func CPUSharesAfterCPUOvercommited(shares uint64, ratio float64) uint64 {
	shares = uint64(float64(shares) / ratio)
	if shares < MinShares {
		return MinShares
	}
	return shares
}

// ResourceConfigForPod takes the input pod and outputs the cgroup resource config.
func ResourceConfigForPod(pod *v1.Pod, enforceCPULimits bool, cpuOvercommitRatio float64) *ResourceConfig {
	// sum requests and limits.
	reqs, limits := resource.PodRequestsAndLimits(pod)

	cpuRequests := int64(0)
	cpuLimits := int64(0)
	memoryLimits := int64(0)
	if request, found := reqs[v1.ResourceCPU]; found {
		cpuRequests = request.MilliValue()
	}
	if limit, found := limits[v1.ResourceCPU]; found {
		cpuLimits = limit.MilliValue()
	}
	if limit, found := limits[v1.ResourceMemory]; found {
		memoryLimits = limit.Value()
	}

	// convert to CFS values
	cpuShares := MilliCPUToShares(cpuRequests)
	cpuShares = CPUSharesAfterCPUOvercommited(cpuShares, cpuOvercommitRatio)
	cpuQuota, cpuPeriod := MilliCPUToQuota(cpuLimits)

	// track if limits were applied for each resource.
	memoryLimitsDeclared := true
	cpuLimitsDeclared := true
	// map hugepage pagesize (bytes) to limits (bytes)
	hugePageLimits := map[int64]int64{}
	for _, container := range pod.Spec.Containers {
		if container.Resources.Limits.Cpu().IsZero() {
			cpuLimitsDeclared = false
		}
		if container.Resources.Limits.Memory().IsZero() {
			memoryLimitsDeclared = false
		}
		containerHugePageLimits := HugePageLimits(container.Resources.Requests)
		for k, v := range containerHugePageLimits {
			if value, exists := hugePageLimits[k]; exists {
				hugePageLimits[k] = value + v
			} else {
				hugePageLimits[k] = v
			}
		}
	}

	// quota is not capped when cfs quota is disabled
	if !enforceCPULimits {
		cpuQuota = int64(-1)
	}

	// determine the qos class
	qosClass := v1qos.GetPodQOS(pod)

	// build the result
	result := &ResourceConfig{}
	if qosClass == v1.PodQOSGuaranteed {
		result.CpuShares = &cpuShares
		result.CpuQuota = &cpuQuota
		result.CpuPeriod = &cpuPeriod
		result.Memory = &memoryLimits
	} else if qosClass == v1.PodQOSBurstable {
		result.CpuShares = &cpuShares
		if cpuLimitsDeclared {
			result.CpuQuota = &cpuQuota
			result.CpuPeriod = &cpuPeriod
		}
		if memoryLimitsDeclared {
			result.Memory = &memoryLimits
		}
	} else {
		shares := uint64(MinShares)
		result.CpuShares = &shares
	}
	result.HugePageLimit = hugePageLimits
	return result
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*CgroupSubsystems, error) {
	// get all cgroup mounts.
	allCgroups, err := libcontainercgroups.GetCgroupMounts(true)
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
	f, err := os.Open(procsFile)
	if err != nil {
		if os.IsNotExist(err) {
			// The procsFile does not exist, So no pids attached to this directory
			return []int{}, nil
		}
		return nil, err
	}
	defer f.Close()

	s := bufio.NewScanner(f)
	out := []int{}
	for s.Scan() {
		if t := s.Text(); t != "" {
			pid, err := strconv.Atoi(t)
			if err != nil {
				return nil, fmt.Errorf("unexpected line in %v; could not convert to pid: %v", procsFile, err)
			}
			out = append(out, pid)
		}
	}
	return out, nil
}

// GetPodCgroupNameSuffix returns the last element of the pod CgroupName identifier
func GetPodCgroupNameSuffix(podUID types.UID) string {
	return podCgroupNamePrefix + string(podUID)
}
