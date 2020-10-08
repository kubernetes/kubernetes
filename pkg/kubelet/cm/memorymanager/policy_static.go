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
	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

const policyTypeStatic policyType = "static"

type systemReservedMemory map[int]map[v1.ResourceName]uint64

// staticPolicy is implementation of the policy interface for the single NUMA policy
type staticPolicy struct {
	// machineInfo contains machine memory related information
	machineInfo *cadvisorapi.MachineInfo
	// reserved contains memory that reserved for kube
	systemReserved systemReservedMemory
	// topology manager reference to get container Topology affinity
	affinity topologymanager.Store
}

var _ Policy = &staticPolicy{}

// NewPolicyStatic returns new single NUMA policy instance
func NewPolicyStatic() Policy {
	return &staticPolicy{}
}

func (p *staticPolicy) Name() string {
	return string(policyTypeStatic)
}

func (p *staticPolicy) Start(s state.State) error {
	return nil
}

// Allocate call is idempotent
func (p *staticPolicy) Allocate(s state.State, pod *v1.Pod, container *v1.Container) error {
	return nil
}

// RemoveContainer call is idempotent
func (p *staticPolicy) RemoveContainer(s state.State, podUID string, containerName string) error {
	return nil
}

// GetTopologyHints implements the topologymanager.HintProvider Interface
// and is consulted to achieve NUMA aware resource alignment among this
// and other resource controllers.
func (p *staticPolicy) GetTopologyHints(s state.State, pod *v1.Pod, container *v1.Container) map[string][]topologymanager.TopologyHint {
	return nil
}
