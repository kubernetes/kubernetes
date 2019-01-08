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
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/qos"
)

// applyPlatformSpecificContainerConfig applies platform specific configurations to runtimeapi.ContainerConfig.
func (m *kubeGenericRuntimeManager) applyPlatformSpecificContainerConfig(config *runtimeapi.ContainerConfig, container *v1.Container, pod *v1.Pod, uid *int64, username string) error {
	config.Linux = m.generateLinuxContainerConfig(container, pod, uid, username)
	return nil
}

// generateLinuxContainerConfig generates linux container config for kubelet runtime v1.
func (m *kubeGenericRuntimeManager) generateLinuxContainerConfig(container *v1.Container, pod *v1.Pod, uid *int64, username string) *runtimeapi.LinuxContainerConfig {
	lc := &runtimeapi.LinuxContainerConfig{
		Resources:       &runtimeapi.LinuxContainerResources{},
		SecurityContext: m.determineEffectiveSecurityContext(pod, container, uid, username),
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
	qosClass := v1qos.GetPodQOS(pod)
	if qosClass == v1.PodQOSGuaranteed || qosClass == v1.PodQOSBurstable {
		cpuOvercommitRatio := m.cpuOvercommitRatioGetter()
		lc.Resources.CpuShares = cpuSharesAfterCPUOvercommited(cpuShares, cpuOvercommitRatio)
		glog.V(4).Infof("[k8s.qiniu.com/cpu-overcommit-ratio]: container %s/%s/%s cpu.shares old: %v, actual: %v, ratio: %v", pod.Namespace, pod.Name, container.Name, cpuShares, lc.Resources.CpuShares, cpuOvercommitRatio)
	} else {
		glog.V(4).Infof("[k8s.qiniu.com/cpu-overcommit-ratio]: container %s/%s/%s we only adjust cpu.shares for qos Guaranteed or Burstable", pod.Namespace, pod.Name, container.Name)
		lc.Resources.CpuShares = cpuShares
	}
	if memoryLimit != 0 {
		lc.Resources.MemoryLimitInBytes = memoryLimit
	}
	// Set OOM score of the container based on qos policy. Processes in lower-priority pods should
	// be killed first if the system runs out of memory.
	lc.Resources.OomScoreAdj = oomScoreAdj

	if m.cpuCFSQuota {
		// if cpuLimit.Amount is nil, then the appropriate default value is returned
		// to allow full usage of cpu resource.
		cpuQuota, cpuPeriod := milliCPUToQuota(cpuLimit.MilliValue())
		lc.Resources.CpuQuota = cpuQuota
		lc.Resources.CpuPeriod = cpuPeriod
	}

	return lc
}
