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
	"strconv"
	"time"

	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/qos"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

// applyPlatformSpecificContainerConfig applies platform specific configurations to runtimeapi.ContainerConfig.
func (m *kubeGenericRuntimeManager) applyPlatformSpecificContainerConfig(config *runtimeapi.ContainerConfig, container *v1.Container, pod *v1.Pod, uid *int64, username string, nsTarget *kubecontainer.ContainerID) error {
	enforceMemoryQoS := false
	// Set memory.min and memory.high if MemoryQoS enabled with cgroups v2
	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.MemoryQoS) &&
		libcontainercgroups.IsCgroup2UnifiedMode() {
		enforceMemoryQoS = true
	}
	config.Linux = m.generateLinuxContainerConfig(container, pod, uid, username, nsTarget, enforceMemoryQoS)
	return nil
}

// generateLinuxContainerConfig generates linux container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateLinuxContainerConfig(container *v1.Container, pod *v1.Pod, uid *int64, username string, nsTarget *kubecontainer.ContainerID, enforceMemoryQoS bool) *runtimeapi.LinuxContainerConfig {
	lc := &runtimeapi.LinuxContainerConfig{
		Resources:       &runtimeapi.LinuxContainerResources{},
		SecurityContext: m.determineEffectiveSecurityContext(pod, container, uid, username),
	}

	if nsTarget != nil && lc.SecurityContext.NamespaceOptions.Pid == runtimeapi.NamespaceMode_CONTAINER {
		lc.SecurityContext.NamespaceOptions.Pid = runtimeapi.NamespaceMode_TARGET
		lc.SecurityContext.NamespaceOptions.TargetId = nsTarget.ID
	}

	// set linux container resources
	var cpuShares int64
	cpuRequest := container.Resources.Requests.Cpu()
	cpuLimit := container.Resources.Limits.Cpu()
	memoryLimit := container.Resources.Limits.Memory().Value()
	memoryRequest := container.Resources.Requests.Memory().Value()
	oomScoreAdj := int64(qos.GetContainerOOMScoreAdjust(pod, container,
		int64(m.machineInfo.MemoryCapacity)))
	// If request is not specified, but limit is, we want request to default to limit.
	// API server does this for new containers, but we repeat this logic in Kubelet
	// for containers running on existing Kubernetes clusters.
	if cpuRequest.IsZero() && !cpuLimit.IsZero() {
		cpuShares = milliCPUToShares(cpuLimit.MilliValue())
	} else {
		// if cpuRequest.Amount is nil, then milliCPUToShares will return the minimal number
		// of CPU shares.
		cpuShares = milliCPUToShares(cpuRequest.MilliValue())
	}
	lc.Resources.CpuShares = cpuShares
	if memoryLimit != 0 {
		lc.Resources.MemoryLimitInBytes = memoryLimit
	}
	// Set OOM score of the container based on qos policy. Processes in lower-priority pods should
	// be killed first if the system runs out of memory.
	lc.Resources.OomScoreAdj = oomScoreAdj

	if m.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned
		// to allow full usage of cpu resource.
		cpuPeriod := int64(quotaPeriod)
		if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CPUCFSQuotaPeriod) {
			cpuPeriod = int64(m.cpuCFSQuotaPeriod.Duration / time.Microsecond)
		}
		cpuQuota := milliCPUToQuota(cpuLimit.MilliValue(), cpuPeriod)
		lc.Resources.CpuQuota = cpuQuota
		lc.Resources.CpuPeriod = cpuPeriod
	}

	lc.Resources.HugepageLimits = GetHugepageLimitsFromResources(container.Resources)

	if utilfeature.DefaultFeatureGate.Enabled(kubefeatures.NodeSwap) {
		// NOTE(ehashman): Behaviour is defined in the opencontainers runtime spec:
		// https://github.com/opencontainers/runtime-spec/blob/1c3f411f041711bbeecf35ff7e93461ea6789220/config-linux.md#memory
		switch m.memorySwapBehavior {
		case kubelettypes.UnlimitedSwap:
			// -1 = unlimited swap
			lc.Resources.MemorySwapLimitInBytes = -1
		case kubelettypes.LimitedSwap:
			fallthrough
		default:
			// memorySwapLimit = total permitted memory+swap; if equal to memory limit, => 0 swap above memory limit
			// Some swapping is still possible.
			// Note that if memory limit is 0, memory swap limit is ignored.
			lc.Resources.MemorySwapLimitInBytes = lc.Resources.MemoryLimitInBytes
		}
	}

	// Set memory.min and memory.high to enforce MemoryQoS
	if enforceMemoryQoS {
		unified := map[string]string{}

		if memoryRequest != 0 {
			unified[cm.MemoryMin] = strconv.FormatInt(memoryRequest, 10)
		}

		// If container sets limits.memory, we set memory.high=pod.spec.containers[i].resources.limits[memory] * memory_throttling_factor
		// for container level cgroup if memory.high>memory.min.
		// If container doesn't set limits.memory, we set memory.high=node_allocatable_memory * memory_throttling_factor
		// for container level cgroup.
		memoryHigh := int64(0)
		if memoryLimit != 0 {
			memoryHigh = int64(float64(memoryLimit) * m.memoryThrottlingFactor)
		} else {
			allocatable := m.getNodeAllocatable()
			allocatableMemory, ok := allocatable[v1.ResourceMemory]
			if ok && allocatableMemory.Value() > 0 {
				memoryHigh = int64(float64(allocatableMemory.Value()) * m.memoryThrottlingFactor)
			}
		}
		if memoryHigh > memoryRequest {
			unified[cm.MemoryHigh] = strconv.FormatInt(memoryHigh, 10)
		}
		if len(unified) > 0 {
			if lc.Resources.Unified == nil {
				lc.Resources.Unified = unified
			} else {
				for k, v := range unified {
					lc.Resources.Unified[k] = v
				}
			}
			klog.V(4).InfoS("MemoryQoS config for container", "pod", klog.KObj(pod), "containerName", container.Name, "unified", unified)
		}
	}

	return lc
}

// GetHugepageLimitsFromResources returns limits of each hugepages from resources.
func GetHugepageLimitsFromResources(resources v1.ResourceRequirements) []*runtimeapi.HugepageLimit {
	var hugepageLimits []*runtimeapi.HugepageLimit

	// For each page size, limit to 0.
	for _, pageSize := range cgroupfs.HugePageSizes {
		hugepageLimits = append(hugepageLimits, &runtimeapi.HugepageLimit{
			PageSize: pageSize,
			Limit:    uint64(0),
		})
	}

	requiredHugepageLimits := map[string]uint64{}
	for resourceObj, amountObj := range resources.Limits {
		if !v1helper.IsHugePageResourceName(resourceObj) {
			continue
		}

		pageSize, err := v1helper.HugePageSizeFromResourceName(resourceObj)
		if err != nil {
			klog.InfoS("Failed to get hugepage size from resource", "object", resourceObj, "err", err)
			continue
		}

		sizeString, err := v1helper.HugePageUnitSizeFromByteSize(pageSize.Value())
		if err != nil {
			klog.InfoS("Size is invalid", "object", resourceObj, "err", err)
			continue
		}
		requiredHugepageLimits[sizeString] = uint64(amountObj.Value())
	}

	for _, hugepageLimit := range hugepageLimits {
		if limit, exists := requiredHugepageLimits[hugepageLimit.PageSize]; exists {
			hugepageLimit.Limit = limit
		}
	}

	return hugepageLimits
}
