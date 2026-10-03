//go:build linux

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
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

func subtractOverheadFromResourceConfig(resCfg *cm.ResourceConfig, pod *v1.Pod) *cm.ResourceConfig {
	if resCfg == nil {
		return nil
	}

	rc := *resCfg

	if pod.Spec.Overhead != nil {
		if cpu, found := pod.Spec.Overhead[v1.ResourceCPU]; found {
			// An unlimited quota, -1, stays unlimited; nothing is subtracted from it,
			// and an overhead that itself converts to -1 subtracts nothing.
			if rc.CPUPeriod != nil && rc.CPUQuota != nil && *rc.CPUQuota != -1 {
				cpuPeriod := int64(*rc.CPUPeriod)
				if overheadQuota := cm.MilliCPUToQuota(cpu.MilliValue(), cpuPeriod); overheadQuota != -1 {
					cpuQuota := *rc.CPUQuota - overheadQuota
					rc.CPUQuota = &cpuQuota
				}
			}

			if rc.CPUShares != nil {
				totalCPUMilli := cm.SharesToMilliCPU(int64(*rc.CPUShares))
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
