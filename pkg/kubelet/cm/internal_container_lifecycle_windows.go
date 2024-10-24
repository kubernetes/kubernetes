//go:build windows
// +build windows

/*
Copyright 2020 The Kubernetes Authors.

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
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

func (i *internalContainerLifecycleImpl) PreCreateContainer(pod *v1.Pod, container *v1.Container, containerConfig *runtimeapi.ContainerConfig) error {
	if i.cpuManager != nil && utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsCPUAndMemoryAffinity) {
		klog.Info("PreCreateContainer for Windows")
		allocatedCPUs := i.cpuManager.GetCPUAffinity(string(pod.UID), container.Name)
		if !allocatedCPUs.IsEmpty() {
			klog.Infof("Setting CPU affinity for container %q cpus %v", container.Name, allocatedCPUs.String())
			var cpuGroupAffinities []*runtimeapi.WindowsCpuGroupAffinity
			affinities := winstats.CpusToGroupAffinity(allocatedCPUs.List())
			for _, affinity := range affinities {
				klog.Infof("Setting CPU affinity for container %q in group %v with mask %v (processors %v)", container.Name, affinity.Group, affinity.Mask, affinity.Processors())
				cpuGroupAffinities = append(cpuGroupAffinities, &runtimeapi.WindowsCpuGroupAffinity{
					CpuGroup: uint32(affinity.Group),
					CpuMask:  uint64(affinity.Mask),
				})
			}

			containerConfig.Windows.Resources.AffinityCpus = cpuGroupAffinities
		}
	}
	return nil
}
