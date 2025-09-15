/*
Copyright 2024 The Kubernetes Authors.

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

	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"

	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

// On Windows we want to use the same logic as the StaticPolicy to compute the memory topology hints
// but unlike linux based systems, on Windows systems numa nodes cannot be directly assigned or guaranteed via Windows APIs
// (windows scheduler will use the numa node that is closest to the cpu assigned therefor respecting the numa node assignment as a best effort). Because of this we don't want to have users specify "StaticPolicy" for the memory manager
// policy via kubelet configuration. Instead we want to use the "BestEffort" policy which will use the same logic as the StaticPolicy
// and doing so will reduce code duplication.
const policyTypeBestEffort policyType = "BestEffort"

// bestEffortPolicy is implementation of the policy interface for the BestEffort policy
type bestEffortPolicy struct {
	static *staticPolicy
}

var _ Policy = &bestEffortPolicy{}

func NewPolicyBestEffort(ctx context.Context, machineInfo *cadvisorapi.MachineInfo, reserved systemReservedMemory, affinity topologymanager.Store) (Policy, error) {
	p, err := NewPolicyStatic(ctx, machineInfo, reserved, affinity)

	if err != nil {
		return nil, err
	}

	return &bestEffortPolicy{
		static: p.(*staticPolicy),
	}, nil
}

func (p *bestEffortPolicy) Name() string {
	return string(policyTypeBestEffort)
}

func (p *bestEffortPolicy) Start(ctx context.Context, s state.State) error {
	return p.static.Start(ctx, s)
}

func (p *bestEffortPolicy) Allocate(ctx context.Context, s state.State, pod *v1.Pod, container *v1.Container) (rerr error) {
	return p.static.Allocate(ctx, s, pod, container)
}

func (p *bestEffortPolicy) RemoveContainer(ctx context.Context, s state.State, podUID string, containerName string) {
	p.static.RemoveContainer(ctx, s, podUID, containerName)
}

func (p *bestEffortPolicy) GetPodTopologyHints(ctx context.Context, s state.State, pod *v1.Pod) map[string][]topologymanager.TopologyHint {
	return p.static.GetPodTopologyHints(ctx, s, pod)
}

func (p *bestEffortPolicy) GetTopologyHints(ctx context.Context, s state.State, pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	return p.static.GetTopologyHints(ctx, s, pod, container)
}

func (p *bestEffortPolicy) GetAllocatableMemory(ctx context.Context, s state.State) []state.Block {
	return p.static.GetAllocatableMemory(ctx, s)
}
