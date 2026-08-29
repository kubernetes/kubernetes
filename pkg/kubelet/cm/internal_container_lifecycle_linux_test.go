//go:build linux

/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/utils/cpuset"
)

type mockCPUManagerAffinity struct {
	cpumanager.Manager
	affinity cpuset.CPUSet
}

func (m *mockCPUManagerAffinity) GetCPUAffinity(podUID, containerName string) cpuset.CPUSet {
	return m.affinity
}

type mockMemoryManagerNUMA struct {
	memorymanager.Manager
	nodes sets.Set[int]
}

func (m *mockMemoryManagerNUMA) GetMemoryNUMANodes(logger klog.Logger, pod *v1.Pod, container *v1.Container) sets.Set[int] {
	return m.nodes
}

func TestPreCreateContainer(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	pod := &v1.Pod{}
	container := &v1.Container{}

	newContainerConfig := func() *runtimeapi.ContainerConfig {
		return &runtimeapi.ContainerConfig{
			Linux: &runtimeapi.LinuxContainerConfig{
				Resources: &runtimeapi.LinuxContainerResources{},
			},
		}
	}

	tests := []struct {
		name           string
		cpuManager     cpumanager.Manager
		memoryManager  memorymanager.Manager
		wantCpusetCpus string
		wantCpusetMems string
	}{
		{
			name:           "no managers configured leaves resources untouched",
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "empty CPU affinity leaves CpusetCpus empty",
			cpuManager:     &mockCPUManagerAffinity{affinity: cpuset.New()},
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "CPU affinity is applied to the container config",
			cpuManager:     &mockCPUManagerAffinity{affinity: cpuset.New(0, 1)},
			wantCpusetCpus: cpuset.New(0, 1).String(),
			wantCpusetMems: "",
		},
		{
			name:           "empty NUMA node set leaves CpusetMems empty",
			memoryManager:  &mockMemoryManagerNUMA{nodes: sets.New[int]()},
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "NUMA nodes are applied sorted to the container config",
			memoryManager:  &mockMemoryManagerNUMA{nodes: sets.New(1, 0)},
			wantCpusetCpus: "",
			wantCpusetMems: "0,1",
		},
		{
			name:           "both managers apply their affinities",
			cpuManager:     &mockCPUManagerAffinity{affinity: cpuset.New(2)},
			memoryManager:  &mockMemoryManagerNUMA{nodes: sets.New(3)},
			wantCpusetCpus: cpuset.New(2).String(),
			wantCpusetMems: "3",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lifecycle := internalContainerLifecycleImpl{
				cpuManager:    test.cpuManager,
				memoryManager: test.memoryManager,
			}
			containerConfig := newContainerConfig()

			if err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig); err != nil {
				t.Fatalf("PreCreateContainer returned error: %v", err)
			}

			if got := containerConfig.Linux.Resources.CpusetCpus; got != test.wantCpusetCpus {
				t.Errorf("CpusetCpus = %q, want %q", got, test.wantCpusetCpus)
			}
			if got := containerConfig.Linux.Resources.CpusetMems; got != test.wantCpusetMems {
				t.Errorf("CpusetMems = %q, want %q", got, test.wantCpusetMems)
			}
		})
	}
}
