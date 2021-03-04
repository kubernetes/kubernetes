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
	"fmt"
	"reflect"
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

const (
	mb           = 1024 * 1024
	gb           = mb * 1024
	pageSize1Gb  = 1048576
	hugepages1Gi = v1.ResourceName(v1.ResourceHugePagesPrefix + "1Gi")
)

var (
	requirementsGuaranteed = &v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1000Mi"),
			v1.ResourceMemory: resource.MustParse("1Gi"),
			hugepages1Gi:      resource.MustParse("1Gi"),
		},
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1000Mi"),
			v1.ResourceMemory: resource.MustParse("1Gi"),
			hugepages1Gi:      resource.MustParse("1Gi"),
		},
	}
	requirementsBurstable = &v1.ResourceRequirements{
		Limits: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1000Mi"),
			v1.ResourceMemory: resource.MustParse("2Gi"),
			hugepages1Gi:      resource.MustParse("2Gi"),
		},
		Requests: v1.ResourceList{
			v1.ResourceCPU:    resource.MustParse("1000Mi"),
			v1.ResourceMemory: resource.MustParse("1Gi"),
			hugepages1Gi:      resource.MustParse("1Gi"),
		},
	}
)

type fakeTopologyManagerWithHint struct {
	hint *topologymanager.TopologyHint
}

// NewFakeTopologyManagerWithHint returns an instance of fake topology manager with specified topology hints
func NewFakeTopologyManagerWithHint(hint *topologymanager.TopologyHint) topologymanager.Manager {
	return &fakeTopologyManagerWithHint{
		hint: hint,
	}
}

func (m *fakeTopologyManagerWithHint) AddHintProvider(h topologymanager.HintProvider) {
	klog.Infof("[fake topologymanager] AddHintProvider HintProvider:  %v", h)
}

func (m *fakeTopologyManagerWithHint) AddContainer(pod *v1.Pod, containerID string) error {
	klog.Infof("[fake topologymanager] AddContainer  pod: %v container id:  %v", format.Pod(pod), containerID)
	return nil
}

func (m *fakeTopologyManagerWithHint) RemoveContainer(containerID string) error {
	klog.Infof("[fake topologymanager] RemoveContainer container id:  %v", containerID)
	return nil
}

func (m *fakeTopologyManagerWithHint) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	klog.Infof("[fake topologymanager] Topology Admit Handler")
	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}

func (m *fakeTopologyManagerWithHint) GetAffinity(podUID string, containerName string) topologymanager.TopologyHint {
	klog.Infof("[fake topologymanager] GetAffinity podUID: %v container name:  %v", podUID, containerName)
	return *m.hint
}

func areMemoryBlocksEqual(mb1, mb2 []state.Block) bool {
	if len(mb1) != len(mb2) {
		return false
	}

	copyMemoryBlocks := make([]state.Block, len(mb2))
	copy(copyMemoryBlocks, mb2)
	for _, block := range mb1 {
		for i, copyBlock := range copyMemoryBlocks {
			if reflect.DeepEqual(block, copyBlock) {
				// move the element that equals to the block to the end of the slice
				copyMemoryBlocks[i] = copyMemoryBlocks[len(copyMemoryBlocks)-1]

				// remove the last element from our slice
				copyMemoryBlocks = copyMemoryBlocks[:len(copyMemoryBlocks)-1]

				break
			}
		}
	}

	return len(copyMemoryBlocks) == 0
}

func areContainerMemoryAssignmentsEqual(cma1, cma2 state.ContainerMemoryAssignments) bool {
	if len(cma1) != len(cma2) {
		return false
	}

	for podUID, container := range cma1 {
		if _, ok := cma2[podUID]; !ok {
			klog.Errorf("[memorymanager_tests] the assignment does not have pod UID %s", podUID)
			return false
		}

		for containerName, memoryBlocks := range container {
			if _, ok := cma2[podUID][containerName]; !ok {
				klog.Errorf("[memorymanager_tests] the assignment does not have container name %s", containerName)
				return false
			}

			if !areMemoryBlocksEqual(memoryBlocks, cma2[podUID][containerName]) {
				klog.Errorf("[memorymanager_tests] assignments memory blocks are different: %v != %v", memoryBlocks, cma2[podUID][containerName])
				return false
			}
		}
	}
	return true
}

type testStaticPolicy struct {
	description           string
	assignments           state.ContainerMemoryAssignments
	expectedAssignments   state.ContainerMemoryAssignments
	machineState          state.NUMANodeMap
	expectedMachineState  state.NUMANodeMap
	systemReserved        systemReservedMemory
	expectedError         error
	machineInfo           *cadvisorapi.MachineInfo
	pod                   *v1.Pod
	topologyHint          *topologymanager.TopologyHint
	expectedTopologyHints map[string][]topologymanager.TopologyHint
}

func initTests(testCase *testStaticPolicy, hint *topologymanager.TopologyHint) (Policy, state.State, error) {
	manager := topologymanager.NewFakeManager()
	if hint != nil {
		manager = NewFakeTopologyManagerWithHint(hint)
	}

	p, err := NewPolicyStatic(testCase.machineInfo, testCase.systemReserved, manager)
	if err != nil {
		return nil, nil, err
	}
	s := state.NewMemoryState()
	s.SetMachineState(testCase.machineState)
	s.SetMemoryAssignments(testCase.assignments)
	return p, s, nil
}

func newNUMAAffinity(bits ...int) bitmask.BitMask {
	affinity, err := bitmask.NewBitMask(bits...)
	if err != nil {
		panic(err)
	}
	return affinity
}

func TestStaticPolicyNew(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description:   "should fail, when machine does not have reserved memory for the system workloads",
			expectedError: fmt.Errorf("[memorymanager] you should specify the system reserved memory"),
		},
		{
			description: "should succeed, when at least one NUMA node has reserved memory",
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			_, _, err := initTests(&testCase, nil)
			if !reflect.DeepEqual(err, testCase.expectedError) {
				t.Fatalf("The actual error: %v is different from the expected one: %v", err, testCase.expectedError)
			}
		})
	}
}

func TestStaticPolicyName(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description: "should return the correct policy name",
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, _, err := initTests(&testCase, nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if p.Name() != string(policyTypeStatic) {
				t.Errorf("policy name is different, expected: %q, actual: %q", p.Name(), policyTypeStatic)
			}
		})
	}
}

func TestStaticPolicyStart(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description: "should fail, if machine state is empty, but it has memory assignments",
			assignments: state.ContainerMemoryAssignments{
				"pod": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512 * mb,
						},
					},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedError: fmt.Errorf("[memorymanager] machine state can not be empty when it has memory assignments"),
		},
		{
			description:         "should fill the state with default values, when the state is empty",
			expectedAssignments: state.ContainerMemoryAssignments{},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   3 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 0,
					Cells:               []int{0},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 3 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
		},
		{
			description: "should fail when machine state does not have all NUMA nodes",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
					{
						Id:     1,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state does not have memory resource",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state has wrong size of total memory",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state has wrong size of system reserved memory",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 1024,
							TotalMemSize:   2 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state reserved memory is different from the memory of all containers memory assignments",
			assignments: state.ContainerMemoryAssignments{
				"pod": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512 * mb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 1,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state has wrong size of hugepages",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 2,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state has wrong size of system reserved hugepages",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: gb,
							TotalMemSize:   2 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 2,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when the hugepages reserved machine state is different from the hugepages of all containers memory assignments",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
				"pod2": map[string][]state.Block{
					"container2": {
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    4 * gb,
							Free:           gb,
							Reserved:       3 * gb,
							SystemReserved: 0,
							TotalMemSize:   4 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 2,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 4,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should fail when machine state does not have NUMA node that used under the memory assignment",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{1},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] (pod: pod1, container: container1) the memory assignment uses the NUMA that does not exist"),
		},
		{
			description: "should fail when machine state does not have resource that used under the memory assignment",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages2M,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 2,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] (pod: pod1, container: container1) the memory assignment uses memory resource that does not exist"),
		},
		{
			description: "should fail when machine state number of assignments is different from the expected one",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 1,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2 * gb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
			expectedError: fmt.Errorf("[memorymanager] the expected machine state is different from the real one"),
		},
		{
			description: "should validate cross NUMA reserved memory vs container assignments",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         768 * mb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
				"pod2": map[string][]state.Block{
					"container2": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         256 * mb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         768 * mb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
				"pod2": map[string][]state.Block{
					"container2": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         256 * mb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    640 * mb,
							Free:           0,
							Reserved:       640 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2176 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 4,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    640 * mb,
							Free:           256 * mb,
							Reserved:       384 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2176 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 4,
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    640 * mb,
							Free:           0,
							Reserved:       640 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2176 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 4,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    640 * mb,
							Free:           256 * mb,
							Reserved:       384 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2176 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 4,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 2176 * mb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
					{
						Id:     1,
						Memory: 2176 * mb,
						HugePages: []cadvisorapi.HugePagesInfo{
							{
								// size in KB
								PageSize: pageSize1Gb,
								NumPages: 1,
							},
						},
					},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			klog.Infof("[Start] %s", testCase.description)
			p, s, err := initTests(&testCase, nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			err = p.Start(s)
			if !reflect.DeepEqual(err, testCase.expectedError) {
				t.Fatalf("The actual error: %v is different from the expected one: %v", err, testCase.expectedError)
			}

			if err != nil {
				return
			}

			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments: %v is different from the expected one: %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state: %v is different from the expected one: %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyAllocate(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description:         "should do nothing for non-guaranteed pods",
			expectedAssignments: state.ContainerMemoryAssignments{},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:                   getPod("pod1", "container1", requirementsBurstable),
			expectedTopologyHints: nil,
			topologyHint:          &topologymanager.TopologyHint{},
		},
		{
			description: "should do nothing once container already exists under the state file",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
				},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:                   getPod("pod1", "container1", requirementsGuaranteed),
			expectedTopologyHints: nil,
			topologyHint:          &topologymanager.TopologyHint{},
		},
		{
			description: "should calculate a default topology hint when no NUMA affinity was provided by the topology manager hint",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 2,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:          getPod("pod1", "container1", requirementsGuaranteed),
			topologyHint: &topologymanager.TopologyHint{},
		},
		{
			description: "should fail when no NUMA affinity was provided under the topology manager hint and calculation of the default hint failed",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 2,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:           getPod("pod2", "container2", requirementsGuaranteed),
			expectedError: fmt.Errorf("[memorymanager] failed to get the default NUMA affinity, no NUMA nodes with enough memory is available"),
			topologyHint:  &topologymanager.TopologyHint{},
		},
		{
			description: "should fail when no NUMA affinity was provided under the topology manager preferred hint and default hint has preferred false",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512 * mb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    gb,
							Free:           512 * mb,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 1,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{1},
					NumberOfAssignments: 0,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{2},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				2: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:           getPod("pod2", "container2", requirementsGuaranteed),
			expectedError: fmt.Errorf("[memorymanager] failed to find the default preferred hint"),
			topologyHint:  &topologymanager.TopologyHint{Preferred: true},
		},
		{
			description: "should fail when NUMA affinity provided under the topology manager hint did not satisfy container requirements and extended hint generation failed",
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{1, 2},
					NumberOfAssignments: 1,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{1, 2},
					NumberOfAssignments: 1,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				2: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:           getPod("pod1", "container1", requirementsGuaranteed),
			expectedError: fmt.Errorf("[memorymanager] failed to find NUMA nodes to extend the current topology hint"),
			topologyHint:  &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: false},
		},
		{
			description: "should fail when the topology manager provided the preferred hint and extended hint has preferred false",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512 * mb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    gb,
							Free:           512 * mb,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 1,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{1},
					NumberOfAssignments: 0,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   1536 * mb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{2},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				2: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:           getPod("pod2", "container2", requirementsGuaranteed),
			expectedError: fmt.Errorf("[memorymanager] failed to find the extended preferred hint"),
			topologyHint:  &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true},
		},
		{
			description: "should succeed to allocate memory from multiple NUMA nodes",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{1},
					NumberOfAssignments: 0,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{2},
					NumberOfAssignments: 0,
				},
				3: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{3},
					NumberOfAssignments: 0,
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           0,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           0,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{2},
					NumberOfAssignments: 0,
				},
				3: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{3},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				2: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				3: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:          getPod("pod1", "container1", requirementsGuaranteed),
			topologyHint: &topologymanager.TopologyHint{Preferred: true},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			klog.Infof("TestStaticPolicyAllocate %s", testCase.description)
			p, s, err := initTests(&testCase, testCase.topologyHint)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			err = p.Allocate(s, testCase.pod, &testCase.pod.Spec.Containers[0])
			if !reflect.DeepEqual(err, testCase.expectedError) {
				t.Fatalf("The actual error %v is different from the expected one %v", err, testCase.expectedError)
			}

			if err != nil {
				return
			}

			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyRemoveContainer(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description:         "should do nothing when the container does not exist under the state",
			expectedAssignments: state.ContainerMemoryAssignments{},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells: []int{},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
		},
		{
			description: "should delete the container assignment and update the machine state",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			expectedAssignments: state.ContainerMemoryAssignments{},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 2,
					Cells:               []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 0,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
		},
		{
			description: "should delete the cross NUMA container assignment and update the machine state",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			expectedAssignments: state.ContainerMemoryAssignments{},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           0,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 2,
					Cells:               []int{0, 1},
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           0,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 2,
					Cells:               []int{0, 1},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 0,
					Cells:               []int{0},
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    512 * mb,
							Free:           512 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					NumberOfAssignments: 0,
					Cells:               []int{1},
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, s, err := initTests(&testCase, nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			err = p.RemoveContainer(s, "pod1", "container1")
			if !reflect.DeepEqual(err, testCase.expectedError) {
				t.Fatalf("The actual error %v is different from the expected one %v", err, testCase.expectedError)
			}

			if err != nil {
				return
			}

			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyGetTopologyHints(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description: "should not provide topology hints for non-guaranteed pods",
			pod:         getPod("pod1", "container1", requirementsBurstable),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: nil,
		},
		{
			description: "should provide topology hints based on the existent memory assignment",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			pod: getPod("pod1", "container1", requirementsGuaranteed),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{
				string(v1.ResourceMemory): {
					{
						NUMANodeAffinity: newNUMAAffinity(0),
						Preferred:        true,
					},
				},
				string(hugepages1Gi): {
					{
						NUMANodeAffinity: newNUMAAffinity(0),
						Preferred:        true,
					},
				},
			},
		},
		{
			description: "should calculate new topology hints, when the container does not exist under assignments",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         2 * gb,
						},
						{
							NUMAAffinity: []int{0, 1},
							Type:         hugepages1Gi,
							Size:         2 * gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           0,
							Reserved:       1536 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           gb,
							Reserved:       512 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           0,
							Reserved:       gb,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				2: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{2},
					NumberOfAssignments: 0,
				},
				3: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           1536 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   2 * gb,
						},
						hugepages1Gi: {
							Allocatable:    gb,
							Free:           gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   gb,
						},
					},
					Cells:               []int{3},
					NumberOfAssignments: 0,
				},
			},
			pod: getPod("pod2", "container2", requirementsGuaranteed),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				2: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				3: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{
				string(v1.ResourceMemory): {
					{
						NUMANodeAffinity: newNUMAAffinity(2),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: newNUMAAffinity(3),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: newNUMAAffinity(2, 3),
						Preferred:        false,
					},
				},
				string(hugepages1Gi): {
					{
						NUMANodeAffinity: newNUMAAffinity(2),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: newNUMAAffinity(3),
						Preferred:        true,
					},
					{
						NUMANodeAffinity: newNUMAAffinity(2, 3),
						Preferred:        false,
					},
				},
			},
		},
		{
			description: "should fail when number of existing memory assignment resources are different from resources requested by container",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
				},
			},
			pod: getPod("pod1", "container1", requirementsGuaranteed),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: nil,
		},
		{
			description: "should fail when existing memory assignment resources are different from resources requested by container",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages2M,
							Size:         gb,
						},
					},
				},
			},
			pod: getPod("pod1", "container1", requirementsGuaranteed),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: nil,
		},
		{
			description: "should fail when existing memory assignment size is different from one requested by the container",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         512 * mb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         gb,
						},
					},
				},
			},
			pod: getPod("pod1", "container1", requirementsGuaranteed),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, s, err := initTests(&testCase, nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			topologyHints := p.GetTopologyHints(s, testCase.pod, &testCase.pod.Spec.Containers[0])
			if !reflect.DeepEqual(topologyHints, testCase.expectedTopologyHints) {
				t.Fatalf("The actual topology hints: '%+v' are different from the expected one: '%+v'", topologyHints, testCase.expectedTopologyHints)
			}
		})
	}
}
