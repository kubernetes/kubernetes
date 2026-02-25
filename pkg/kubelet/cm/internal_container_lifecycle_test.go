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
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

type mockCPUManager struct {
	called bool
	cpumanager.Manager
}

func (cpuManager *mockCPUManager) AddContainer(logr.Logger, *v1.Pod, *v1.Container, string) {
	cpuManager.called = true
}

type mockMemoryManager struct {
	called bool
	memorymanager.Manager
}

func (memoryManager *mockMemoryManager) AddContainer(logr.Logger, *v1.Pod, *v1.Container, string) {
	memoryManager.called = true
}

type mockTopologyManager struct {
	called bool
	topologymanager.Manager
}

func (topologyManager *mockTopologyManager) AddContainer(*v1.Pod, *v1.Container, string) {
	topologyManager.called = true
}

func TestPreStartContainer(t *testing.T) {
	pod := &v1.Pod{}
	container := &v1.Container{}

	tests := []struct {
		name      string
		lifecycle internalContainerLifecycleImpl
	}{
		{
			name: "When a CPU manager is provided it has AddContainer called",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:      &mockCPUManager{},
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
		}, {
			name: "When a Memory manager is provided it has AddContainer called",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   &mockMemoryManager{},
				topologyManager: &mockTopologyManager{},
			},
		}, {
			name: "When a CPU manager/Memory manager is not provided, it's ok",
			lifecycle: internalContainerLifecycleImpl{
				cpuManager:      nil,
				memoryManager:   nil,
				topologyManager: &mockTopologyManager{},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			_ = test.lifecycle.PreStartContainer(logger, pod, container, "42")
		})

		cManager := test.lifecycle.cpuManager
		mManager := test.lifecycle.memoryManager
		tManager := test.lifecycle.topologyManager
		if cManager != nil && !cManager.(*mockCPUManager).called {
			t.Errorf("When a CPU manager is provided it must have AddContainer called")
		}
		if mManager != nil && !mManager.(*mockMemoryManager).called {
			t.Errorf("When a Memory manager is provided it must have AddContainer called")
		}
		if !tManager.(*mockTopologyManager).called {
			t.Errorf("TopologyManager's AddContainer method must be called during container startup")
		}
	}
}
