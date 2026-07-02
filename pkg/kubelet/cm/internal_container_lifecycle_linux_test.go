//go:build linux

/*
Copyright 2025 The Kubernetes Authors.

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

	"github.com/go-logr/logr"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"

	"k8s.io/utils/cpuset"
)

type mockCPUManagerForPreCreate struct {
	cpumanager.Manager
	cpuAffinity cpuset.CPUSet
}

func (m *mockCPUManagerForPreCreate) GetCPUAffinity(podUID, containerName string) cpuset.CPUSet {
	return m.cpuAffinity
}

type mockMemoryManagerForPreCreate struct {
	memorymanager.Manager
	numaNodes sets.Set[int]
}

func (m *mockMemoryManagerForPreCreate) GetMemoryNUMANodes(logger logr.Logger, pod *v1.Pod, container *v1.Container) sets.Set[int] {
	return m.numaNodes
}

func TestPreCreateContainer(t *testing.T) {
	pod := &v1.Pod{}
	container := &v1.Container{}

	tests := []struct {
		name           string
		lifecycle      internalContainerLifecycleImpl
		expectedCpuset string
		expectedMems   string
	}{
		{
			name: "When both CPU and Memory managers are provided with non-empty values",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    &mockCPUManagerForPreCreate{cpuAffinity: cpuset.New(0, 1, 2, 3)},
				memoryManager: &mockMemoryManagerForPreCreate{numaNodes: sets.New(0, 1)},
			},
			expectedCpuset: "0-3",
			expectedMems:   "0,1",
		},
		{
			name: "When only CPU manager is provided with non-empty CPUs",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    &mockCPUManagerForPreCreate{cpuAffinity: cpuset.New(0, 1, 2)},
				memoryManager: nil,
			},
			expectedCpuset: "0-2",
			expectedMems:   "",
		},
		{
			name: "When only Memory manager is provided with non-empty NUMA nodes",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    nil,
				memoryManager: &mockMemoryManagerForPreCreate{numaNodes: sets.New(0)},
			},
			expectedCpuset: "",
			expectedMems:   "0",
		},
		{
			name: "When both managers are nil",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    nil,
				memoryManager: nil,
			},
			expectedCpuset: "",
			expectedMems:   "",
		},
		{
			name: "When CPU manager returns empty CPUSet",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    &mockCPUManagerForPreCreate{cpuAffinity: cpuset.New()},
				memoryManager: &mockMemoryManagerForPreCreate{numaNodes: sets.New(0, 1)},
			},
			expectedCpuset: "",
			expectedMems:   "0,1",
		},
		{
			name: "When Memory manager returns empty NUMA nodes",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    &mockCPUManagerForPreCreate{cpuAffinity: cpuset.New(0, 1, 2, 3)},
				memoryManager: &mockMemoryManagerForPreCreate{numaNodes: sets.New[int]()},
			},
			expectedCpuset: "0-3",
			expectedMems:   "",
		},
		{
			name: "With multiple NUMA nodes",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:    nil,
				memoryManager: &mockMemoryManagerForPreCreate{numaNodes: sets.New(0, 1, 2, 3)},
			},
			expectedCpuset: "",
			expectedMems:   "0,1,2,3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			containerConfig := &runtimeapi.ContainerConfig{
				Linux: &runtimeapi.LinuxContainerConfig{
					Resources: &runtimeapi.LinuxContainerResources{},
				},
			}

			err := tt.lifecycle.PreCreateContainer(logger, pod, container, containerConfig)

			if err != nil {
				t.Errorf("PreCreateContainer returned unexpected error: %v", err)
			}

			if containerConfig.Linux.Resources.CpusetCpus != tt.expectedCpuset {
				t.Errorf("CpusetCpus = %q, want %q", containerConfig.Linux.Resources.CpusetCpus, tt.expectedCpuset)
			}

			if containerConfig.Linux.Resources.CpusetMems != tt.expectedMems {
				t.Errorf("CpusetMems = %q, want %q", containerConfig.Linux.Resources.CpusetMems, tt.expectedMems)
			}
		})
	}
}

func TestPreCreateContainerWithPodInfo(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "pod-12345",
			Name:      "my-pod",
			Namespace: "default",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "main-container"},
			},
		},
	}
	container := &v1.Container{
		Name: "main-container",
	}

	cpuManager := &mockCPUManagerForPreCreate{cpuAffinity: cpuset.New(1, 2, 3, 4)}
	memoryManager := &mockMemoryManagerForPreCreate{numaNodes: sets.New(2)}

	lifecycle := internalContainerLifecycleImpl{
		cpuManager:    cpuManager,
		memoryManager: memoryManager,
	}

	logger, _ := ktesting.NewTestContext(t)
	containerConfig := &runtimeapi.ContainerConfig{
		Linux: &runtimeapi.LinuxContainerConfig{
			Resources: &runtimeapi.LinuxContainerResources{},
		},
	}

	err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig)
	if err != nil {
		t.Fatalf("PreCreateContainer failed: %v", err)
	}

	if containerConfig.Linux.Resources.CpusetCpus != "1-4" {
		t.Errorf("CpusetCpus = %q, want %q", containerConfig.Linux.Resources.CpusetCpus, "1-4")
	}

	if containerConfig.Linux.Resources.CpusetMems != "2" {
		t.Errorf("CpusetMems = %q, want %q", containerConfig.Linux.Resources.CpusetMems, "2")
	}
}
