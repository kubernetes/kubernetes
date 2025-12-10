//go:build linux
// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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

package kuberuntime

import (
	"math"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

// sharesToMilliCPU converts CpuShares (cpu.shares) to milli-CPU value
func sharesToMilliCPU(shares int64) int64 {
	milliCPU := int64(0)
	if shares >= int64(cm.MinShares) {
		milliCPU = int64(math.Ceil(float64(shares*cm.MilliCPUToCPU) / float64(cm.SharesPerCPU)))
	}
	return milliCPU
}

// quotaToMilliCPU converts cpu.cfs_quota_us and cpu.cfs_period_us to milli-CPU value
func quotaToMilliCPU(quota int64, period int64) int64 {
	if quota == -1 {
		return int64(0)
	}
	return (quota * cm.MilliCPUToCPU) / period
}

func subtractOverheadFromResourceConfig(resCfg *cm.ResourceConfig, pod *v1.Pod) *cm.ResourceConfig {
	if resCfg == nil {
		return nil
	}

	rc := *resCfg

	if pod.Spec.Overhead != nil {
		if cpu, found := pod.Spec.Overhead[v1.ResourceCPU]; found {
			if rc.CPUPeriod != nil {
				cpuPeriod := int64(*rc.CPUPeriod)
				cpuQuota := *rc.CPUQuota - cm.MilliCPUToQuota(cpu.MilliValue(), cpuPeriod)
				rc.CPUQuota = &cpuQuota
			}

			if rc.CPUShares != nil {
				totalCPUMilli := sharesToMilliCPU(int64(*rc.CPUShares))
				cpuShares := cm.MilliCPUToShares(totalCPUMilli - cpu.MilliValue())
				rc.CPUShares = &cpuShares
			}
		}

		if memory, found := pod.Spec.Overhead[v1.ResourceMemory]; found {
			if rc.Memory != nil {
				currMemory := *rc.Memory

				if mem, ok := memory.AsInt64(); ok {
					currMemory -= mem
				}

				rc.Memory = &currMemory
			}
		}
	}
	return &rc
}
