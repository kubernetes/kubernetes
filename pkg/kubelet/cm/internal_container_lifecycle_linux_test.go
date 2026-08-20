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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/utils/cpuset"
)

type mockCPUManagerWithAffinity struct {
	cpuAffinity cpuset.CPUSet
	cpumanager.Manager
}

func (m *mockCPUManagerWithAffinity) GetCPUAffinity(string, string) cpuset.CPUSet {
	return m.cpuAffinity
}

type mockMemoryManagerWithNUMA struct {
	numaNodes sets.Set[int]
	memorymanager.Manager
}

func (m *mockMemoryManagerWithNUMA) GetMemoryNUMANodes(klog.Logger, *v1.Pod, *v1.Container) sets.Set[int] {
	return m.numaNodes
}

func TestPreCreateContainer(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "test-pod-uid",
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	tests := []struct {
		name                string
		lifecycle           internalContainerLifecycleImpl
		expectedCpusetCpus  string
		expectedCpusetMems  string
		expectCpusetCpusSet bool
		expectCpusetMemsSet bool
	}{
		{
			name: "CPU manager with allocated CPUs sets CpusetCpus",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: &mockCPUManagerWithAffinity{
					cpuAffinity: cpuset.New(0, 1, 2),
				},
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: true,
			expectedCpusetCpus:  "0-2",
			expectCpusetMemsSet: false,
		},
		{
			name: "CPU manager with empty CPU affinity does not set CpusetCpus",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: &mockCPUManagerWithAffinity{
					cpuAffinity: cpuset.New(),
				},
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: false,
			expectedCpusetCpus:  "",
			expectCpusetMemsSet: false,
		},
		{
			name: "CPU manager with non-contiguous CPUs sets CpusetCpus with comma-separated values",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: &mockCPUManagerWithAffinity{
					cpuAffinity: cpuset.New(0, 2, 4),
				},
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: true,
			expectedCpusetCpus:  "0,2,4",
			expectCpusetMemsSet: false,
		},
		{
			name: "Memory manager with NUMA nodes sets CpusetMems",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: nil,
				memoryManager: &mockMemoryManagerWithNUMA{
					numaNodes: sets.New[int](0, 1),
				},
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: false,
			expectedCpusetCpus:  "",
			expectCpusetMemsSet: true,
			expectedCpusetMems:  "0,1",
		},
		{
			name: "Memory manager with single NUMA node sets CpusetMems without comma",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: nil,
				memoryManager: &mockMemoryManagerWithNUMA{
					numaNodes: sets.New[int](2),
				},
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: false,
			expectedCpusetCpus:  "",
			expectCpusetMemsSet: true,
			expectedCpusetMems:  "2",
		},
		{
			name: "Memory manager with empty NUMA nodes does not set CpusetMems",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: nil,
				memoryManager: &mockMemoryManagerWithNUMA{
					numaNodes: nil,
				},
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: false,
			expectedCpusetCpus:  "",
			expectCpusetMemsSet: false,
		},
		{
			name: "Both CPU manager and Memory manager set respective fields",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager: &mockCPUManagerWithAffinity{
					cpuAffinity: cpuset.New(0, 1),
				},
				memoryManager: &mockMemoryManagerWithNUMA{
					numaNodes: sets.New[int](0),
				},
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: true,
			expectedCpusetCpus:  "0-1",
			expectCpusetMemsSet: true,
			expectedCpusetMems:  "0",
		},
		{
			name: "Neither CPU manager nor Memory manager is set",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
			expectCpusetCpusSet: false,
			expectedCpusetCpus:  "",
			expectCpusetMemsSet: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			containerConfig := &runtimeapi.ContainerConfig{
				Linux: &runtimeapi.LinuxContainerConfig{
					Resources: &runtimeapi.LinuxContainerResources{},
				},
			}

			err := test.lifecycle.PreCreateContainer(logger, pod, container, containerConfig)
			if err != nil {
				t.Errorf("PreCreateContainer should not return an error, got: %v", err)
			}

			if test.expectCpusetCpusSet {
				if containerConfig.Linux.Resources.CpusetCpus != test.expectedCpusetCpus {
					t.Errorf("expected CpusetCpus=%q, got %q", test.expectedCpusetCpus, containerConfig.Linux.Resources.CpusetCpus)
				}
			} else {
				if containerConfig.Linux.Resources.CpusetCpus != "" {
					t.Errorf("expected CpusetCpus to be empty, got %q", containerConfig.Linux.Resources.CpusetCpus)
				}
			}

			if test.expectCpusetMemsSet {
				if containerConfig.Linux.Resources.CpusetMems != test.expectedCpusetMems {
					t.Errorf("expected CpusetMems=%q, got %q", test.expectedCpusetMems, containerConfig.Linux.Resources.CpusetMems)
				}
			} else {
				if containerConfig.Linux.Resources.CpusetMems != "" {
					t.Errorf("expected CpusetMems to be empty, got %q", containerConfig.Linux.Resources.CpusetMems)
				}
			}
		})
	}
}

func TestPreCreateContainerDoesNotOverwriteExistingLinuxConfig(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "test-pod-uid",
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	containerConfig := &runtimeapi.ContainerConfig{
		Linux: &runtimeapi.LinuxContainerConfig{
			Resources: &runtimeapi.LinuxContainerResources{
				CpusetCpus: "existing-value",
				CpusetMems: "existing-mems",
			},
		},
	}

	lifecycle := internalContainerLifecycleImpl{
		cpuManager:      nil,
		memoryManager:   nil,
		topologyManager: &mockTopologyManager{},
	}

	err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig)
	if err != nil {
		t.Errorf("PreCreateContainer should not return an error, got: %v", err)
	}

	if containerConfig.Linux.Resources.CpusetCpus != "existing-value" {
		t.Errorf("expected existing CpusetCpus to be preserved when no managers set new values, got %q", containerConfig.Linux.Resources.CpusetCpus)
	}
	if containerConfig.Linux.Resources.CpusetMems != "existing-mems" {
		t.Errorf("expected existing CpusetMems to be preserved when no managers set new values, got %q", containerConfig.Linux.Resources.CpusetMems)
	}
}

func TestPreCreateContainerOverwritesExistingValuesWhenManagersProvideThem(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "test-pod-uid",
		},
	}
	container := &v1.Container{
		Name: "test-container",
	}

	containerConfig := &runtimeapi.ContainerConfig{
		Linux: &runtimeapi.LinuxContainerConfig{
			Resources: &runtimeapi.LinuxContainerResources{
				CpusetCpus: "existing-value",
				CpusetMems: "existing-mems",
			},
		},
	}

	lifecycle := internalContainerLifecycleImpl{
		cpuManager: &mockCPUManagerWithAffinity{
			cpuAffinity: cpuset.New(0),
		},
		memoryManager: &mockMemoryManagerWithNUMA{
			numaNodes: sets.New[int](1),
		},
		topologyManager: &mockTopologyManager{},
	}

	err := lifecycle.PreCreateContainer(logger, pod, container, containerConfig)
	if err != nil {
		t.Errorf("PreCreateContainer should not return an error, got: %v", err)
	}

	if containerConfig.Linux.Resources.CpusetCpus != "0" {
		t.Errorf("expected CpusetCpus to be overwritten to %q, got %q", "0", containerConfig.Linux.Resources.CpusetCpus)
	}
	if containerConfig.Linux.Resources.CpusetMems != "1" {
		t.Errorf("expected CpusetMems to be overwritten to %q, got %q", "1", containerConfig.Linux.Resources.CpusetMems)
	}
}

func TestPreCreateContainerPodUIDPassedCorrectly(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	type trackedCall struct {
		podUID         string
		containerName  string
	}
	track := &trackedCall{}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "my-pod-uid",
		},
	}
	container := &v1.Container{
		Name: "my-container",
	}

	containerConfig := &runtimeapi.ContainerConfig{
		Linux: &runtimeapi.LinuxContainerConfig{
			Resources: &runtimeapi.LinuxContainerResources{},
		},
	}

	lifecycle := internalContainerLifecycleImpl{
		cpuManager: &mockCPUManagerTrackingArgs{
			cpuAffinity: cpuset.New(0),
			track:       track,
		},
		memoryManager:   nil,
		topologyManager: &mockTopologyManager{},
	}

	_ = lifecycle.PreCreateContainer(logger, pod, container, containerConfig)

	if track.podUID != "my-pod-uid" {
		t.Errorf("expected GetCPUAffinity called with podUID=%q, got %q", "my-pod-uid", track.podUID)
	}
	if track.containerName != "my-container" {
		t.Errorf("expected GetCPUAffinity called with containerName=%q, got %q", "my-container", track.containerName)
	}
}

type mockCPUManagerTrackingArgs struct {
	cpuAffinity cpuset.CPUSet
	track       *struct {
		podUID        string
		containerName string
	}
	cpumanager.Manager
}

func (m *mockCPUManagerTrackingArgs) GetCPUAffinity(podUID string, containerName string) cpuset.CPUSet {
	m.track.podUID = podUID
	m.track.containerName = containerName
	return m.cpuAffinity
}