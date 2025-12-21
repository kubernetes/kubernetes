//go:build !linux

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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/klog/v2"
)

const (
	MinShares = 0
	MaxShares = 0

	SharesPerCPU  = 0
	MilliCPUToCPU = 0

	QuotaPeriod      = 0
	MinQuotaPeriod   = 0
	MinMilliCPULimit = 0
)

// MilliCPUToQuota converts milliCPU and period to CFS quota values.
func MilliCPUToQuota(milliCPU, period int64) int64 {
	return 0
}

// MilliCPUToShares converts the milliCPU to CFS shares.
func MilliCPUToShares(milliCPU int64) uint64 {
	return 0
}

// ResourceConfigForPod takes the input pod and outputs the cgroup resource config.
func ResourceConfigForPod(pod *v1.Pod, enforceCPULimit bool, cpuPeriod uint64, enforceMemoryQoS bool) *ResourceConfig {
	return nil
}

// GetCgroupSubsystems returns information about the mounted cgroup subsystems
func GetCgroupSubsystems() (*CgroupSubsystems, error) {
	return nil, nil
}

func getCgroupProcs(dir string) ([]int, error) {
	return nil, nil
}

// GetPodCgroupNameSuffix returns the last element of the pod CgroupName identifier
func GetPodCgroupNameSuffix(podUID types.UID) string {
	return ""
}

// NodeAllocatableRoot returns the literal cgroup path for the node allocatable cgroup
func NodeAllocatableRoot(cgroupRoot string, cgroupsPerQOS bool, cgroupDriver string) string {
	return ""
}

// GetKubeletContainer returns the cgroup the kubelet will use
func GetKubeletContainer(logger klog.Logger, kubeletCgroups string) (string, error) {
	return "", nil
}

func CPURequestsFromConfig(podConfig *ResourceConfig) *resource.Quantity {
	return nil
}

func CPULimitsFromConfig(podConfig *ResourceConfig) *resource.Quantity {
	return nil
}

func MemoryLimitsFromConfig(podConfig *ResourceConfig) *resource.Quantity {
	return nil
}
