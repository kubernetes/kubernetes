//go:build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"context"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
	"k8s.io/utils/cpuset"
)

func (m *manager) updateContainerCPUSet(ctx context.Context, containerID string, cpus cpuset.CPUSet) error {
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.WindowsCPUAndMemoryAffinity) {
		return nil
	}

	affinities := winstats.CpusToGroupAffinity(cpus.List())
	var cpuGroupAffinities []*runtimeapi.WindowsCpuGroupAffinity
	for _, affinity := range affinities {
		cpuGroupAffinities = append(cpuGroupAffinities, &runtimeapi.WindowsCpuGroupAffinity{
			CpuGroup: uint32(affinity.Group),
			CpuMask:  uint64(affinity.Mask),
		})
	}
	return m.containerRuntime.UpdateContainerResources(ctx, containerID, &runtimeapi.ContainerResources{
		Windows: &runtimeapi.WindowsContainerResources{
			AffinityCpus: cpuGroupAffinities,
		},
	})
}
