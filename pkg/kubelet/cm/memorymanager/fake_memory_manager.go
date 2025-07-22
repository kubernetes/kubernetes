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

package memorymanager

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

type fakeManager struct {
	state state.State
}

func (m *fakeManager) Start(ctx context.Context, activePods ActivePodsFunc, sourcesReady config.SourcesReady, podStatusProvider status.PodStatusProvider, containerRuntime runtimeService, initialContainers containermap.ContainerMap) error {
	logger := klog.FromContext(ctx)
	logger.Info("Start()")
	return nil
}

func (m *fakeManager) Policy(ctx context.Context) Policy {
	logger := klog.FromContext(ctx)
	logger.Info("Policy()")
	return NewPolicyNone(ctx)
}

func (m *fakeManager) Allocate(pod *v1.Pod, container *v1.Container) error {
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("Allocate", "pod", klog.KObj(pod), "containerName", container.Name)
	return nil
}

func (m *fakeManager) AddContainer(ctx context.Context, pod *v1.Pod, container *v1.Container, containerID string) {
	logger := klog.FromContext(ctx)
	logger.Info("Add container", "pod", klog.KObj(pod), "containerName", container.Name, "containerID", containerID)
}

func (m *fakeManager) GetMemoryNUMANodes(ctx context.Context, pod *v1.Pod, container *v1.Container) sets.Set[int] {
	logger := klog.FromContext(ctx)
	logger.Info("Get MemoryNUMANodes", "pod", klog.KObj(pod), "containerName", container.Name)
	return nil
}

func (m *fakeManager) RemoveContainer(ctx context.Context, containerID string) error {
	logger := klog.FromContext(ctx)
	logger.Info("RemoveContainer", "containerID", containerID)
	return nil
}

func (m *fakeManager) GetTopologyHints(pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("Get Topology Hints", "pod", klog.KObj(pod), "containerName", container.Name)
	return map[string][]topologymanager.TopologyHint{}
}

func (m *fakeManager) GetPodTopologyHints(pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	ctx := context.TODO()
	logger := klog.FromContext(ctx)
	logger.Info("Get Pod Topology Hints", "pod", klog.KObj(pod))
	return map[string][]topologymanager.TopologyHint{}
}

func (m *fakeManager) State() state.Reader {
	return m.state
}

// GetAllocatableMemory returns the amount of allocatable memory for each NUMA node
func (m *fakeManager) GetAllocatableMemory(ctx context.Context) []state.Block {
	logger := klog.FromContext(ctx)
	logger.Info("Get Allocatable Memory")
	return []state.Block{}
}

// GetMemory returns the memory allocated by a container from NUMA nodes
func (m *fakeManager) GetMemory(ctx context.Context, podUID, containerName string) []state.Block {
	logger := klog.LoggerWithValues(klog.FromContext(ctx), "podUID", podUID, "containerName", containerName)
	logger.Info("Get Memory")
	return []state.Block{}
}

// NewFakeManager creates empty/fake memory manager
func NewFakeManager(ctx context.Context) Manager {
	logger := klog.LoggerWithName(klog.FromContext(ctx), "memory-mgr.fake")
	return &fakeManager{
		// logger: logger,
		state: state.NewMemoryState(logger),
	}
}
