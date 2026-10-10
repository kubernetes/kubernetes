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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/utils/cpuset"
)

type mockPreCreateCPUManager struct {
	cpumanager.Manager
	allocatedCPUs cpuset.CPUSet
}

func (m *mockPreCreateCPUManager) GetCPUAffinity(string, string) cpuset.CPUSet {
	return m.allocatedCPUs
}

type mockPreCreateMemoryManager struct {
	memorymanager.Manager
	numaNodes sets.Set[int]
}

func (m *mockPreCreateMemoryManager) GetMemoryNUMANodes(klog.Logger, *v1.Pod, *v1.Container) sets.Set[int] {
	return m.numaNodes
}

func TestPreCreateContainer(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: types.UID("pod-uid")}}
	container := &v1.Container{Name: "container-name"}

	tests := []struct {
		name          string
		cpuManager    cpumanager.Manager
		memoryManager memorymanager.Manager
		wantCPUs      string
		wantMems      string
	}{
		{
			name: "no resource managers",
		},
		{
			name:       "CPU manager allocation",
			cpuManager: &mockPreCreateCPUManager{allocatedCPUs: cpuset.New(1, 3)},
			wantCPUs:   "1,3",
		},
		{
			name:          "memory manager allocation",
			memoryManager: &mockPreCreateMemoryManager{numaNodes: sets.New(0, 2)},
			wantMems:      "0,2",
		},
		{
			name:          "CPU and memory manager allocations",
			cpuManager:    &mockPreCreateCPUManager{allocatedCPUs: cpuset.New(2, 4)},
			memoryManager: &mockPreCreateMemoryManager{numaNodes: sets.New(1, 3)},
			wantCPUs:      "2,4",
			wantMems:      "1,3",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lifecycle := internalContainerLifecycleImpl{
				cpuManager:    test.cpuManager,
				memoryManager: test.memoryManager,
			}
			containerConfig := &runtimeapi.ContainerConfig{
				Linux: &runtimeapi.LinuxContainerConfig{
					Resources: &runtimeapi.LinuxContainerResources{},
				},
			}

			if err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig); err != nil {
				t.Fatalf("PreCreateContainer() error = %v", err)
			}
			if got := containerConfig.Linux.Resources.CpusetCpus; got != test.wantCPUs {
				t.Errorf("CpusetCpus = %q, want %q", got, test.wantCPUs)
			}
			if got := containerConfig.Linux.Resources.CpusetMems; got != test.wantMems {
				t.Errorf("CpusetMems = %q, want %q", got, test.wantMems)
			}
		})
	}
}
