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

type mockCPUManagerWithAffinity struct {
	cpumanager.Manager
	affinity cpuset.CPUSet
}

func (m *mockCPUManagerWithAffinity) GetCPUAffinity(podUID, containerName string) cpuset.CPUSet {
	return m.affinity
}

type mockMemoryManagerWithNUMANodes struct {
	memorymanager.Manager
	numaNodes sets.Set[int]
}

func (m *mockMemoryManagerWithNUMANodes) GetMemoryNUMANodes(logger klog.Logger, pod *v1.Pod, container *v1.Container) sets.Set[int] {
	return m.numaNodes
}

func TestPreCreateContainer(t *testing.T) {
	pod := &v1.Pod{}
	container := &v1.Container{Name: "test-container"}

	tests := []struct {
		name           string
		cpuManager     cpumanager.Manager
		memoryManager  memorymanager.Manager
		wantCpusetCpus string
		wantCpusetMems string
	}{
		{
			name:           "no cpu manager, no memory manager",
			cpuManager:     nil,
			memoryManager:  nil,
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "cpu manager with empty affinity is a no-op",
			cpuManager:     &mockCPUManagerWithAffinity{affinity: cpuset.New()},
			memoryManager:  nil,
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "cpu manager sets CpusetCpus from allocated affinity",
			cpuManager:     &mockCPUManagerWithAffinity{affinity: cpuset.New(0, 1, 2)},
			memoryManager:  nil,
			wantCpusetCpus: "0-2",
			wantCpusetMems: "",
		},
		{
			name:           "memory manager with no NUMA nodes is a no-op",
			cpuManager:     nil,
			memoryManager:  &mockMemoryManagerWithNUMANodes{numaNodes: sets.New[int]()},
			wantCpusetCpus: "",
			wantCpusetMems: "",
		},
		{
			name:           "memory manager sets CpusetMems from NUMA node affinity",
			cpuManager:     nil,
			memoryManager:  &mockMemoryManagerWithNUMANodes{numaNodes: sets.New(0, 1)},
			wantCpusetCpus: "",
			wantCpusetMems: "0,1",
		},
		{
			name:           "both cpu and memory manager set their respective cpuset fields",
			cpuManager:     &mockCPUManagerWithAffinity{affinity: cpuset.New(4, 5)},
			memoryManager:  &mockMemoryManagerWithNUMANodes{numaNodes: sets.New(1)},
			wantCpusetCpus: "4-5",
			wantCpusetMems: "1",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			lifecycle := &internalContainerLifecycleImpl{
				cpuManager:    test.cpuManager,
				memoryManager: test.memoryManager,
			}
			containerConfig := &runtimeapi.ContainerConfig{
				Linux: &runtimeapi.LinuxContainerConfig{
					Resources: &runtimeapi.LinuxContainerResources{},
				},
			}

			if err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig); err != nil {
				t.Fatalf("PreCreateContainer() returned unexpected error: %v", err)
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
