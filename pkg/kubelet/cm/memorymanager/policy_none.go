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
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

const policyTypeNone policyType = "None"

// none is implementation of the policy interface for the none policy, using none
// policy is the same as disable memory management
type none struct{}

var _ Policy = &none{}

// NewPolicyNone returns new none policy instance
func NewPolicyNone(ctx context.Context) Policy {
	return &none{}
}

func (p *none) Name() string {
	return string(policyTypeNone)
}

func (p *none) Start(ctx context.Context, s state.State) error {
	logger := klog.FromContext(ctx)
	logger.Info("Start")
	return nil
}

// Allocate call is idempotent
func (p *none) Allocate(_ context.Context, s state.State, pod *v1.Pod, container *v1.Container) error {
	return nil
}

// RemoveContainer call is idempotent
func (p *none) RemoveContainer(_ context.Context, s state.State, podUID string, containerName string) {
}

// GetTopologyHints implements the topologymanager.HintProvider Interface
// and is consulted to achieve NUMA aware resource alignment among this
// and other resource controllers.
func (p *none) GetTopologyHints(_ context.Context, s state.State, pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	return nil
}

// GetPodTopologyHints implements the topologymanager.HintProvider Interface
// and is consulted to achieve NUMA aware resource alignment among this
// and other resource controllers.
func (p *none) GetPodTopologyHints(_ context.Context, s state.State, pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	return nil
}

// GetAllocatableMemory returns the amount of allocatable memory for each NUMA node
func (p *none) GetAllocatableMemory(_ context.Context, s state.State) []state.Block {
	return []state.Block{}
}
