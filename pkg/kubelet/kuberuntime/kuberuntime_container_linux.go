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
	"time"

	cgroupfs "github.com/opencontainers/runc/libcontainer/cgroups/fs"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

// applyPlatformSpecificContainerConfig applies platform specific configurations to runtimeapi.ContainerConfig.
func (m *kubeGenericRuntimeManager) applyPlatformSpecificContainerConfig(config *runtimeapi.ContainerConfig, container *v1.Container, pod *v1.Pod, uid *int64, username string, nsTarget *kubecontainer.ContainerID) error {
	config.Linux = m.generateLinuxContainerConfig(container, pod, uid, username, nsTarget)
	return nil
}

// generateLinuxContainerConfig generates linux container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateLinuxContainerConfig(container *v1.Container, pod *v1.Pod, uid *int64, username string, nsTarget *kubecontainer.ContainerID) *runtimeapi.LinuxContainerConfig {
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
			klog.Warningf("Failed to get hugepage size from resource name: %v", err)
			continue
		}

		sizeString, err := v1helper.HugePageUnitSizeFromByteSize(pageSize.Value())
		if err != nil {
			klog.Warningf("pageSize is invalid: %v", err)
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
