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

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

const (
	mb           = 1024 * 1024
	gb           = mb * 1024
	pageSize1Gb  = 1048576
	hugepages1Gi = v1.ResourceName(v1.ResourceHugePagesPrefix + "1Gi")
)

var (
	containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways

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

func areContainerMemoryAssignmentsEqual(t *testing.T, cma1, cma2 state.ContainerMemoryAssignments) bool {
	if len(cma1) != len(cma2) {
		return false
	}

	for podUID, container := range cma1 {
		if _, ok := cma2[podUID]; !ok {
			t.Logf("[memorymanager_tests] the assignment does not have pod UID %s", podUID)
			return false
		}

		for containerName, memoryBlocks := range container {
			if _, ok := cma2[podUID][containerName]; !ok {
				t.Logf("[memorymanager_tests] the assignment does not have container name %s", containerName)
				return false
			}

			if !areMemoryBlocksEqual(memoryBlocks, cma2[podUID][containerName]) {
				t.Logf("[memorymanager_tests] assignments memory blocks are different: %v != %v", memoryBlocks, cma2[podUID][containerName])
				return false
			}
		}
	}
	return true
}

func getPodWithAllocatedResources(podUID string, containerName string, requirements *v1.ResourceRequirements, allocatedResources v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: types.UID(podUID),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      containerName,
					Resources: *requirements,
				},
			},
		},
		Status: v1.PodStatus{
			ContainerStatuses: []v1.ContainerStatus{
				{
					Name:               containerName,
					AllocatedResources: allocatedResources,
				},
			},
		},
	}
}

type resizeRequest struct {
	pod       string
	container string
	resource  v1.ResourceName
	newSize   resource.Quantity
	decrease  bool
}

type testStaticPolicy struct {
	description                  string
	assignments                  state.ContainerMemoryAssignments
	expectedAssignments          state.ContainerMemoryAssignments
	machineState                 state.NUMANodeMap
	expectedMachineState         state.NUMANodeMap
	systemReserved               systemReservedMemory
	expectedError                error
	machineInfo                  *cadvisorapi.MachineInfo
	pod                          *v1.Pod
	topologyHint                 *topologymanager.TopologyHint
	expectedTopologyHints        map[string][]topologymanager.TopologyHint
	initContainersReusableMemory reusableMemory
	resizeRequests               []resizeRequest
}

func initTests(t *testing.T, testCase *testStaticPolicy, hint *topologymanager.TopologyHint, initContainersReusableMemory reusableMemory) (Policy, state.State, error) {
	manager := topologymanager.NewFakeManager()
	if hint != nil {
		manager = topologymanager.NewFakeManagerWithHint(hint)
	}

	p, err := NewPolicyStatic(testCase.machineInfo, testCase.systemReserved, manager)
	if err != nil {
		return nil, nil, err
	}
	if initContainersReusableMemory != nil {
		p.(*staticPolicy).initContainersReusableMemory = initContainersReusableMemory
	}
	s := state.NewMemoryState()
	s.SetMachineState(testCase.machineState)
	s.SetMemoryAssignments(testCase.assignments)
	return p, s, nil
}

func initTestsPodResize(t *testing.T, testCase *testStaticPolicy, hint *topologymanager.TopologyHint, initContainersReusableMemory reusableMemory) (Policy, state.State, error) {

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScalingStaticMemoryPolicy, true)

	manager := topologymanager.NewFakeManager()
	if hint != nil {
		manager = topologymanager.NewFakeManagerWithHint(hint)
	}

	p, err := NewPolicyStatic(testCase.machineInfo, testCase.systemReserved, manager)
	if err != nil {
		return nil, nil, err
	}
	if initContainersReusableMemory != nil {
		p.(*staticPolicy).initContainersReusableMemory = initContainersReusableMemory
	}
	s := state.NewMemoryState()
	s.SetMachineState(testCase.machineState)
	s.SetMemoryAssignments(testCase.assignments)

	for pod, assignments := range testCase.assignments {
		for container, containerBlocks := range assignments {
			s.SetMemoryBlocks(pod, container, containerBlocks)
			s.SetPromisedMemoryBlocks(pod, container, containerBlocks)
		}
	}

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
			_, _, err := initTests(t, &testCase, nil, nil)
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
			p, _, err := initTests(t, &testCase, nil, nil)
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
		{
			description: "should validate the totalFree and totalReserved size for a resource within a group",
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         240 * mb,
						},
					},
				},
				"pod2": map[string][]state.Block{
					"container2": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         10 * mb,
						},
					},
				},
			},
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         240 * mb,
						},
					},
				},
				"pod2": map[string][]state.Block{
					"container2": {
						{
							NUMAAffinity: []int{0, 1},
							Type:         v1.ResourceMemory,
							Size:         10 * mb,
						},
					},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    220 * mb,
							Free:           30 * mb,
							Reserved:       190 * mb,
							SystemReserved: 20 * mb,
							TotalMemSize:   240 * mb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    220 * mb,
							Free:           160 * mb,
							Reserved:       60 * mb,
							SystemReserved: 20 * mb,
							TotalMemSize:   240 * mb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    220 * mb,
							Free:           10 * mb,
							Reserved:       210 * mb,
							SystemReserved: 20 * mb,
							TotalMemSize:   240 * mb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
				1: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    220 * mb,
							Free:           180 * mb,
							Reserved:       40 * mb,
							SystemReserved: 20 * mb,
							TotalMemSize:   240 * mb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 20 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 20 * mb,
				},
			},
			machineInfo: &cadvisorapi.MachineInfo{
				Topology: []cadvisorapi.Node{
					{
						Id:     0,
						Memory: 240 * mb,
					},
					{
						Id:     1,
						Memory: 240 * mb,
					},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			t.Logf("[Start] %s", testCase.description)
			p, s, err := initTests(t, &testCase, nil, nil)
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
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
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
		{
			description: "should validate NUMA node can not have both single and cross NUMA node memory allocations",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         1024 * mb,
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
							Size:         1024 * mb,
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
							TotalMemSize:   2176 * mb,
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
							TotalMemSize:   2176 * mb,
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
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    1536 * mb,
							Free:           512 * mb,
							Reserved:       1024 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   2176 * mb,
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
							TotalMemSize:   2176 * mb,
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
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod:           getPod("pod2", "container1", requirementsGuaranteed),
			topologyHint:  &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedError: fmt.Errorf("[memorymanager] preferred hint violates NUMA node allocation"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			t.Logf("TestStaticPolicyAllocate %s", testCase.description)
			p, s, err := initTests(t, &testCase, testCase.topologyHint, nil)
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
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyAllocateWithInitContainers(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description: "should re-use init containers memory, init containers requests 1Gi and 2Gi, apps containers 3Gi and 4Gi",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"initContainer2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         3 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         3 * gb,
						},
					},
					"container2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         4 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         4 * gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7680 * mb,
							Free:           7680 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           8 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7680 * mb,
							Free:           512 * mb,
							Reserved:       7 * gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           1 * gb,
							Reserved:       7 * gb,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 8,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
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
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
				},
			),
			topologyHint: &topologymanager.TopologyHint{},
		},
		{
			description: "should re-use init containers memory, init containers requests 4Gi and 3Gi, apps containers 2Gi and 1Gi",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"initContainer2": {
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
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         2 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         2 * gb,
						},
					},
					"container2": {
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
							Allocatable:    7680 * mb,
							Free:           7680 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           8 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7680 * mb,
							Free:           3584 * mb,
							Reserved:       4 * gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           4 * gb,
							Reserved:       4 * gb,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 8,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
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
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
				},
			),
			topologyHint: &topologymanager.TopologyHint{},
		},
		{
			description: "should re-use init containers memory, init containers requests 7Gi and 4Gi, apps containers 4Gi and 3Gi",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"initContainer2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         4 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         4 * gb,
						},
					},
					"container2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         3 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         3 * gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7680 * mb,
							Free:           7680 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           8 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7680 * mb,
							Free:           512 * mb,
							Reserved:       7 * gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   8 * gb,
						},
						hugepages1Gi: {
							Allocatable:    8 * gb,
							Free:           1 * gb,
							Reserved:       7 * gb,
							SystemReserved: 0,
							TotalMemSize:   8 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 8,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("7Gi"),
								hugepages1Gi:      resource.MustParse("7Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("7Gi"),
								hugepages1Gi:      resource.MustParse("7Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			topologyHint: &topologymanager.TopologyHint{},
		},
		{
			description:                  "should re-use init containers memory, init containers requests 7Gi and 4Gi, apps containers 5Gi and 2Gi",
			assignments:                  state.ContainerMemoryAssignments{},
			initContainersReusableMemory: reusableMemory{"pod0": map[string]map[v1.ResourceName]uint64{}},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"initContainer2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         5 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         5 * gb,
						},
					},
					"container2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         2 * gb,
						},
						{
							NUMAAffinity: []int{0},
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
							Allocatable:    10240 * mb,
							Free:           10240 * mb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   10 * gb,
						},
						hugepages1Gi: {
							Allocatable:    10 * gb,
							Free:           10 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   10 * gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    10240 * mb,
							Free:           3072 * mb,
							Reserved:       7 * gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   10 * gb,
						},
						hugepages1Gi: {
							Allocatable:    10 * gb,
							Free:           3 * gb,
							Reserved:       7 * gb,
							SystemReserved: 0,
							TotalMemSize:   10 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 9,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("5Gi"),
								hugepages1Gi:      resource.MustParse("5Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("5Gi"),
								hugepages1Gi:      resource.MustParse("5Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("7Gi"),
								hugepages1Gi:      resource.MustParse("7Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("7Gi"),
								hugepages1Gi:      resource.MustParse("7Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			topologyHint: &topologymanager.TopologyHint{},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			klog.InfoS("TestStaticPolicyAllocateWithInitContainers", "name", testCase.description)
			p, s, err := initTests(t, &testCase, testCase.topologyHint, testCase.initContainersReusableMemory)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			for i := range testCase.pod.Spec.InitContainers {
				err = p.Allocate(s, testCase.pod, &testCase.pod.Spec.InitContainers[i])
				if !reflect.DeepEqual(err, testCase.expectedError) {
					t.Fatalf("The actual error %v is different from the expected one %v", err, testCase.expectedError)
				}
			}

			for i := range testCase.pod.Spec.Containers {
				err = p.Allocate(s, testCase.pod, &testCase.pod.Spec.Containers[i])
				if !reflect.DeepEqual(err, testCase.expectedError) {
					t.Fatalf("The actual error %v is different from the expected one %v", err, testCase.expectedError)
				}
			}

			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyAllocateWithRestartableInitContainers(t *testing.T) {
	testCases := []testStaticPolicy{
		{
			description: "should do nothing once containers already exist under the state file",
			assignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
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
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         gb,
						},
					},
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
							Allocatable:    2560 * mb,
							Free:           512 * mb,
							Reserved:       2048 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   3 * gb,
						},
						hugepages1Gi: {
							Allocatable:    2 * gb,
							Free:           2 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   2 * gb,
						},
					},
					Cells: []int{},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    2560 * mb,
							Free:           512 * mb,
							Reserved:       2048 * mb,
							SystemReserved: 512 * mb,
							TotalMemSize:   3 * gb,
						},
						hugepages1Gi: {
							Allocatable:    2 * gb,
							Free:           2 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   2 * gb,
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
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name:      "container1",
						Resources: *requirementsGuaranteed,
					},
				},
				[]v1.Container{
					{
						Name:          "initContainer1",
						Resources:     *requirementsGuaranteed,
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
			),
			expectedTopologyHints: nil,
			topologyHint:          &topologymanager.TopologyHint{},
		},
		{
			description: "should not re-use restartable init containers memory",
			assignments: state.ContainerMemoryAssignments{},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": map[string][]state.Block{
					"initContainer1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"restartableInitContainer2": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         2 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         2 * gb,
						},
					},
					"initContainer3": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         0,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         0,
						},
					},
					"restartableInitContainer4": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         4 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         4 * gb,
						},
					},
					"container1": {
						{
							NUMAAffinity: []int{0},
							Type:         v1.ResourceMemory,
							Size:         1 * gb,
						},
						{
							NUMAAffinity: []int{0},
							Type:         hugepages1Gi,
							Size:         1 * gb,
						},
					},
				},
			},
			machineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7 * gb,
							Free:           7 * gb,
							Reserved:       0,
							SystemReserved: 512 * mb,
							TotalMemSize:   7680 * mb,
						},
						hugepages1Gi: {
							Allocatable:    7 * gb,
							Free:           7 * gb,
							Reserved:       0,
							SystemReserved: 0,
							TotalMemSize:   7 * gb,
						},
					},
					Cells: []int{0},
				},
			},
			expectedMachineState: state.NUMANodeMap{
				0: &state.NUMANodeState{
					MemoryMap: map[v1.ResourceName]*state.MemoryTable{
						v1.ResourceMemory: {
							Allocatable:    7 * gb,
							Free:           0,
							Reserved:       7 * gb,
							SystemReserved: 512 * mb,
							TotalMemSize:   7680 * mb,
						},
						hugepages1Gi: {
							Allocatable:    7 * gb,
							Free:           0,
							Reserved:       7 * gb,
							SystemReserved: 0,
							TotalMemSize:   7 * gb,
						},
					},
					Cells:               []int{0},
					NumberOfAssignments: 10,
				},
			},
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			pod: getPodWithInitContainers(
				"pod1",
				[]v1.Container{
					{
						Name:      "container1",
						Resources: *requirementsGuaranteed,
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
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
						},
					},
					{
						Name: "restartableInitContainer2",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name: "initContainer3",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
					{
						Name: "restartableInitContainer4",
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("1000Mi"),
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
			),
			topologyHint: &topologymanager.TopologyHint{},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			klog.InfoS("TestStaticPolicyAllocateWithRestartableInitContainers", "name", testCase.description)
			p, s, err := initTests(t, &testCase, testCase.topologyHint, testCase.initContainersReusableMemory)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			for i := range testCase.pod.Spec.InitContainers {
				err = p.Allocate(s, testCase.pod, &testCase.pod.Spec.InitContainers[i])
				if !reflect.DeepEqual(err, testCase.expectedError) {
					t.Fatalf("The actual error %v is different from the expected one %v", err, testCase.expectedError)
				}
			}

			if err != nil {
				return
			}

			for i := range testCase.pod.Spec.Containers {
				err = p.Allocate(s, testCase.pod, &testCase.pod.Spec.Containers[i])
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
			}

			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
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
			p, s, err := initTests(t, &testCase, nil, nil)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}

			p.RemoveContainer(s, "pod1", "container1")
			assignments := s.GetMemoryAssignments()
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
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
		{
			description: "should not return preferred hints with multiple NUMA nodes for the pod with resources satisfied by a single NUMA node",
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
							Type:         hugepages2M,
							Size:         24 * mb,
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
						hugepages2M: {
							Allocatable:    20 * mb,
							Free:           0,
							Reserved:       20 * mb,
							SystemReserved: 0,
							TotalMemSize:   20 * mb,
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
						hugepages2M: {
							Allocatable:    20 * mb,
							Free:           16 * mb,
							Reserved:       4 * mb,
							SystemReserved: 0,
							TotalMemSize:   20 * mb,
						},
					},
					Cells:               []int{0, 1},
					NumberOfAssignments: 2,
				},
			},
			pod: getPod("pod2",
				"container2",
				&v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000Mi"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
						hugepages2M:       resource.MustParse("16Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("1000Mi"),
						v1.ResourceMemory: resource.MustParse("1Gi"),
						hugepages2M:       resource.MustParse("16Mi"),
					},
				},
			),
			systemReserved: systemReservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 512 * mb,
				},
			},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{
				string(v1.ResourceMemory): {
					{
						NUMANodeAffinity: newNUMAAffinity(0, 1),
						Preferred:        false,
					},
				},
				hugepages2M: {
					{
						NUMANodeAffinity: newNUMAAffinity(0, 1),
						Preferred:        false,
					},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, s, err := initTests(t, &testCase, nil, nil)
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

func TestStaticPolicyGetTopologyHintsPodResize(t *testing.T) {

	cpu1000m := resource.MustParse("1000Mi")
	mem1Gi := resource.MustParse("1Gi")
	mem2Gi := resource.MustParse("2Gi")
	mem2048Mi := resource.MustParse("2048Mi")
	mem2560Mi := resource.MustParse("2560Mi")
	mem4096i := resource.MustParse("4096Mi")
	hugepages1GiResName := v1.ResourceName("hugepages-1Gi")
	memZero := resource.Quantity{}

	systemReserved := systemReservedMemory{
		0: {v1.ResourceMemory: 512 * mb},
		1: {v1.ResourceMemory: 512 * mb},
	}

	newMemTable := func(free, reserved uint64) map[v1.ResourceName]*state.MemoryTable {
		return map[v1.ResourceName]*state.MemoryTable{
			v1.ResourceMemory: {
				Allocatable:    1536 * mb,
				Free:           free,
				Reserved:       reserved,
				SystemReserved: 512 * mb,
				TotalMemSize:   2 * gb,
			},
		}
	}
	newMemTableWithHugePages := func(memFree, memReserved, hugeFree, hugeReserved uint64) map[v1.ResourceName]*state.MemoryTable {
		return map[v1.ResourceName]*state.MemoryTable{
			v1.ResourceMemory:   {Allocatable: 1536 * mb, Free: memFree, Reserved: memReserved, SystemReserved: 512 * mb, TotalMemSize: 2 * gb},
			hugepages1GiResName: {Allocatable: 2 * gb, Free: hugeFree, Reserved: hugeReserved, SystemReserved: 0, TotalMemSize: 2 * gb},
		}
	}

	newResourceRequirements := func(mem, hugepages resource.Quantity) *v1.ResourceRequirements {
		limits := v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem}
		if !hugepages.IsZero() {
			limits[hugepages1GiResName] = hugepages
		}
		return &v1.ResourceRequirements{
			Limits:   limits,
			Requests: limits,
		}
	}

	testCases := []testStaticPolicy{
		{
			description: "memory increase - existing NUMA assignment can accomodate",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2 * gb}}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(gb, 1536*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
			},
			pod:                   getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2560Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem2Gi}),
			systemReserved:        systemReserved,
			topologyHint:          &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{string(v1.ResourceMemory): {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}}},
		},
		{
			description: "memory increase - extending NUMA assignment to accomodate",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1024*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{1}, NumberOfAssignments: 0},
			},
			pod:                   getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2048Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			systemReserved:        systemReserved,
			topologyHint:          &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{string(v1.ResourceMemory): {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}}},
		},
		{
			description: "memory increase - no valid assignment due to insufficient memory",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1024*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{1}, NumberOfAssignments: 0},
			},
			pod:                   getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem4096i, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			systemReserved:        systemReserved,
			topologyHint:          &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{},
		},
		{
			description: "memory increase - no valid assignment due to existing assignments on other node",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
				"pod2": {"container2": {{NUMAAffinity: []int{1}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1024*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(512*mb, 1024*mb), Cells: []int{1}, NumberOfAssignments: 1},
			},
			pod:                   getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2048Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			systemReserved:        systemReserved,
			topologyHint:          &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{},
		},
		{
			description: "memory increase - existing NUMA assignment can accomodate with container having memory and hugepages",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {
				{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2 * gb},
				{NUMAAffinity: []int{0, 1}, Type: hugepages1GiResName, Size: 2 * gb},
			}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTableWithHugePages(gb, 1536*mb, 2*gb, 2*gb), Cells: []int{0, 1}, NumberOfAssignments: 2},
				1: {MemoryMap: newMemTableWithHugePages(1024*mb, 512*mb, 4*gb, 0), Cells: []int{0, 1}, NumberOfAssignments: 2},
			},
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2560Mi, mem2048Mi), v1.ResourceList{v1.ResourceMemory: mem2Gi, hugepages1GiResName: mem2Gi}),
			systemReserved: systemReserved,
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{
				string(v1.ResourceMemory):   {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}},
				string(hugepages1GiResName): {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}},
			},
		},
		{
			description: "memory increase - extending NUMA assignment to accomodate with container having memory and hugepages",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {
				{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb},
				{NUMAAffinity: []int{0}, Type: hugepages1GiResName, Size: 1 * gb},
			}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTableWithHugePages(512*mb, 1024*mb, 1*gb, 1*gb), Cells: []int{0}, NumberOfAssignments: 2},
				1: {MemoryMap: newMemTableWithHugePages(1536*mb, 0, 2*gb, 0), Cells: []int{1}, NumberOfAssignments: 0},
			},
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2048Mi, mem1Gi), v1.ResourceList{v1.ResourceMemory: mem1Gi, hugepages1GiResName: mem1Gi}),
			systemReserved: systemReserved,
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedTopologyHints: map[string][]topologymanager.TopologyHint{
				string(v1.ResourceMemory):   {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}},
				string(hugepages1GiResName): {{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true}},
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, s, err := initTestsPodResize(t, &testCase, testCase.topologyHint, nil)
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

func TestStaticPolicyAllocateForResize(t *testing.T) {

	cpu1000m := resource.MustParse("1000Mi")
	mem1Gi := resource.MustParse("1Gi")
	mem1536Mi := resource.MustParse("1536Mi")
	mem2560Mi := resource.MustParse("2560Mi")
	mem2Gi := resource.MustParse("2Gi")
	hugepages1GiResName := v1.ResourceName("hugepages-1Gi")
	memZero := resource.Quantity{}
	mem512Mi := resource.MustParse("512Mi")

	systemReserved := systemReservedMemory{
		0: {v1.ResourceMemory: 512 * mb},
		1: {v1.ResourceMemory: 512 * mb},
	}

	newMemTable := func(free, reserved uint64) map[v1.ResourceName]*state.MemoryTable {
		return map[v1.ResourceName]*state.MemoryTable{
			v1.ResourceMemory: {
				Allocatable:    1536 * mb,
				Free:           free,
				Reserved:       reserved,
				SystemReserved: 512 * mb,
				TotalMemSize:   2 * gb,
			},
		}
	}

	newMemTableWithHugePages := func(memFree, memReserved, hugeFree, hugeReserved uint64) map[v1.ResourceName]*state.MemoryTable {
		return map[v1.ResourceName]*state.MemoryTable{
			v1.ResourceMemory:   {Allocatable: 1536 * mb, Free: memFree, Reserved: memReserved, SystemReserved: 512 * mb, TotalMemSize: 2 * gb},
			hugepages1GiResName: {Allocatable: 2 * gb, Free: hugeFree, Reserved: hugeReserved, SystemReserved: 0, TotalMemSize: 2 * gb},
		}
	}

	newResourceRequirements := func(mem, hugepages resource.Quantity) *v1.ResourceRequirements {
		limits := v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem}
		if !hugepages.IsZero() {
			limits[hugepages1GiResName] = hugepages
		}
		return &v1.ResourceRequirements{
			Limits:   limits,
			Requests: limits,
		}
	}

	testCases := []testStaticPolicy{
		{
			description: "memory increase - existing single NUMA assignment can accomodate",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem1536Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 1536*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}}},
			},
			expectedError: nil,
		},
		{
			description: "memory increase - existing multi NUMA assignment can accomodate",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 1536*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2560Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem2Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 1536*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(512*mb, 1024*mb), Cells: []int{0, 1}, NumberOfAssignments: 1},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0, 1}, Type: v1.ResourceMemory, Size: 2560 * mb}}},
			},
			expectedError: nil,
		},
		{
			description: "memory increase - error when requested memory is less than promised",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem512Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			// The machine state and assignments should not change because an error is expected
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			expectedError: fmt.Errorf("[memorymanager] requested memory cannot be less than original allocation"),
		},
		{
			description: "huge page resize is ignored, no change in machine state and assignments",
			assignments: state.ContainerMemoryAssignments{"pod1": {"container1": {
				{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb},
				{NUMAAffinity: []int{0}, Type: hugepages1GiResName, Size: 1 * gb},
			}}},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTableWithHugePages(512*mb, 1*gb, 1*gb, 1*gb), Cells: []int{0}, NumberOfAssignments: 2},
				1: {MemoryMap: newMemTableWithHugePages(1536*mb, 0, 2*gb, 0), Cells: []int{1}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem1Gi, mem2Gi), v1.ResourceList{v1.ResourceMemory: mem1Gi, hugepages1GiResName: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTableWithHugePages(512*mb, 1*gb, 1*gb, 1*gb), Cells: []int{0}, NumberOfAssignments: 2},
				1: {MemoryMap: newMemTableWithHugePages(1536*mb, 0, 2*gb, 0), Cells: []int{1}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{"pod1": {"container1": {
				{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb},
				{NUMAAffinity: []int{0}, Type: hugepages1GiResName, Size: 1 * gb},
			}}},
			expectedError: nil,
		},
		{
			description: "should calculate hint for resize when no NUMA affinity was provided by the topology manager hint",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem1536Mi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 1536*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}}},
			},
			expectedError: nil,
		},
		{
			description: "should fail when hint calculation fails and no NUMA affinity was provided by the topology manager hint",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(resource.MustParse("4Gi"), memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 1536*mb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}}},
			},
			expectedError: fmt.Errorf("[memorymanager] failed to get NUMA affinity for resize, no NUMA nodes with enough memory is available"),
		},
		{
			description: "topology manager hint violates NUMA affinity and memory manager cannot extend hint",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{}, NumberOfAssignments: 1},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(resource.MustParse("3Gi"), memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{}, NumberOfAssignments: 1},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}}},
			},
			expectedError: fmt.Errorf("[memorymanager] failed to find NUMA nodes to extend the current topology hint"),
		},
		{
			description: "topology manager hint iolates existing NUMA allocation",
			assignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}}},
			},
			machineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{}, NumberOfAssignments: 1},
			},
			systemReserved: systemReserved,
			pod:            getPodWithAllocatedResources("pod1", "container1", newResourceRequirements(mem2Gi, memZero), v1.ResourceList{v1.ResourceMemory: mem1Gi}),
			topologyHint:   &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 1), Preferred: true},
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(512*mb, 1*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1024*mb, 512*mb), Cells: []int{}, NumberOfAssignments: 1},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {"container1": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}}},
			},
			expectedError: fmt.Errorf("[memorymanager] preferred hint violates NUMA node allocation"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			t.Logf("TestStaticPolicyAllocate %s", testCase.description)
			p, s, err := initTestsPodResize(t, &testCase, testCase.topologyHint, nil)
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
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func TestStaticPolicyAllocateForResizeWithInitContainers(t *testing.T) {

	mem1536Mi := resource.MustParse("1536Mi")
	mem1Gi := resource.MustParse("1Gi")
	cpu1000m := resource.MustParse("1000Mi")
	hugepages1GiResName := v1.ResourceName("hugepages-1Gi")
	memZero := resource.Quantity{}

	systemReserved := systemReservedMemory{
		0: {v1.ResourceMemory: 512 * mb},
		1: {v1.ResourceMemory: 512 * mb},
	}

	newMemTable := func(free, reserved uint64) map[v1.ResourceName]*state.MemoryTable {
		return map[v1.ResourceName]*state.MemoryTable{
			v1.ResourceMemory: {
				Allocatable:    1536 * mb,
				Free:           free,
				Reserved:       reserved,
				SystemReserved: 512 * mb,
				TotalMemSize:   2 * gb,
			},
		}
	}

	newResourceRequirements := func(mem, hugepages resource.Quantity) *v1.ResourceRequirements {
		limits := v1.ResourceList{v1.ResourceCPU: cpu1000m, v1.ResourceMemory: mem}
		if !hugepages.IsZero() {
			limits[hugepages1GiResName] = hugepages
		}
		return &v1.ResourceRequirements{
			Limits:   limits,
			Requests: limits,
		}
	}

	testCases := []testStaticPolicy{
		{
			description: "account for pod reusable memory during resize of regular container",
			// init container requests 2GB and regular container requests 1GB. podreusable memory is 1GB after actual container is allocated.
			// The machine state should ot change when the regualr contianer is resized from 1GB to 1.5GB.
			assignments: state.ContainerMemoryAssignments{
				"pod1": {
					"initcontainer": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 2 * gb}},
					"container1":    {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1 * gb}},
				},
			},
			initContainersReusableMemory: reusableMemory{
				"pod1": {"01": {v1.ResourceMemory: 1 * gb}},
			},
			machineState: state.NUMANodeMap{
				// NUMA 0 is fully reserved by the init container initially.
				0: {MemoryMap: newMemTable(0, 2*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			systemReserved: systemReserved,
			pod: getPodWithAllocatedResources(
				"pod1", "container1",
				newResourceRequirements(mem1536Mi, memZero), // New request is 1.5GiB
				v1.ResourceList{v1.ResourceMemory: mem1Gi},  // Original allocation was 1GiB
			),
			topologyHint: &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			// Because 1GiB is reusable, the resize from 1GiB to 1.5GiB only needs 0.5GiB of *new*
			// memory. The machine state's reserved memory should not change from the initial 2GiB.
			expectedMachineState: state.NUMANodeMap{
				0: {MemoryMap: newMemTable(0, 2*gb), Cells: []int{0}, NumberOfAssignments: 1},
				1: {MemoryMap: newMemTable(1536*mb, 0), Cells: []int{}, NumberOfAssignments: 0},
			},
			expectedAssignments: state.ContainerMemoryAssignments{
				"pod1": {
					"initcontainer": {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 2 * gb}},
					"container1":    {{NUMAAffinity: []int{0}, Type: v1.ResourceMemory, Size: 1536 * mb}},
				},
			},
			expectedError: nil,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			t.Logf("TestStaticPolicyAllocate %s", testCase.description)
			p, s, err := initTestsPodResize(t, &testCase, testCase.topologyHint, testCase.initContainersReusableMemory)
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
			if !areContainerMemoryAssignmentsEqual(t, assignments, testCase.expectedAssignments) {
				t.Fatalf("Actual assignments %v are different from the expected %v", assignments, testCase.expectedAssignments)
			}

			machineState := s.GetMachineState()
			if !areMachineStatesEqual(machineState, testCase.expectedMachineState) {
				t.Fatalf("The actual machine state %v is different from the expected %v", machineState, testCase.expectedMachineState)
			}
		})
	}
}

func Test_getPodRequestedResources(t *testing.T) {
	testCases := []struct {
		description string
		pod         *v1.Pod
		expected    map[v1.ResourceName]uint64
	}{
		{
			description: "maximum resources of init containers > total resources of containers",
			pod: getPodWithInitContainers(
				"",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("1Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("1Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			expected: map[v1.ResourceName]uint64{
				v1.ResourceMemory: 4 * gb,
				hugepages1Gi:      4 * gb,
			},
		},
		{
			description: "maximum resources of init containers < total resources of containers",
			pod: getPodWithInitContainers(
				"",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("1Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
				},
			),
			expected: map[v1.ResourceName]uint64{
				v1.ResourceMemory: 5 * gb,
				hugepages1Gi:      5 * gb,
			},
		},
		{
			description: "calculate different resources independently",
			pod: getPodWithInitContainers(
				"",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "initContainer1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("1Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			expected: map[v1.ResourceName]uint64{
				v1.ResourceMemory: 5 * gb,
				hugepages1Gi:      4 * gb,
			},
		},
		{
			description: "maximum resources of init containers > total resources of long running containers, including restartable init containers",
			pod: getPodWithInitContainers(
				"",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("1Gi"),
								hugepages1Gi:      resource.MustParse("1Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "restartableInit1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			expected: map[v1.ResourceName]uint64{
				v1.ResourceMemory: 6 * gb,
				hugepages1Gi:      6 * gb,
			},
		},
		{
			description: "maximum resources of init containers < total resources of long running containers, including restartable init containers",
			pod: getPodWithInitContainers(
				"",
				[]v1.Container{
					{
						Name: "container1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
					},
					{
						Name: "container2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("3Gi"),
								hugepages1Gi:      resource.MustParse("3Gi"),
							},
						},
					},
				},
				[]v1.Container{
					{
						Name: "restartableInit1",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("2Gi"),
								hugepages1Gi:      resource.MustParse("2Gi"),
							},
						},
						RestartPolicy: &containerRestartPolicyAlways,
					},
					{
						Name: "initContainer2",
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceMemory: resource.MustParse("4Gi"),
								hugepages1Gi:      resource.MustParse("4Gi"),
							},
						},
					},
				},
			),
			expected: map[v1.ResourceName]uint64{
				v1.ResourceMemory: 7 * gb,
				hugepages1Gi:      7 * gb,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			actual, err := getPodRequestedResources(tc.pod)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			if diff := cmp.Diff(actual, tc.expected); diff != "" {
				t.Errorf("getPodRequestedResources() mismatch (-want +got):\n%s", diff)
			}
		})
	}
}

func Test_isAffinityViolatingNUMAAllocations(t *testing.T) {
	testsCases := []struct {
		description         string
		machineState        map[int]*state.NUMANodeState
		topologyHint        *topologymanager.TopologyHint
		isViolationExpected bool
	}{
		{
			description: "violating NUMA allocations because given affinity asks for NUMA ID 1 which is on different cells group",
			machineState: map[int]*state.NUMANodeState{
				0: {
					NumberOfAssignments: 1,
					Cells:               []int{0, 1},
				},
				1: {
					NumberOfAssignments: 1,
					Cells:               []int{0, 1},
				},
				2: {
					NumberOfAssignments: 1,
					Cells:               []int{2},
				},
				3: {
					NumberOfAssignments: 0,
					Cells:               []int{3},
				},
			},
			topologyHint: &topologymanager.TopologyHint{
				NUMANodeAffinity: newNUMAAffinity(1, 2),
			},
			isViolationExpected: true,
		},
		{
			description: "violating NUMA allocations because given affinity with multiple nodes asks for NUMA ID 1 which is used for a single NUMA node memory allocation",
			machineState: map[int]*state.NUMANodeState{
				0: {
					NumberOfAssignments: 0,
					Cells:               []int{0, 1},
				},
				1: {
					NumberOfAssignments: 1,
					Cells:               []int{1},
				},
			},
			topologyHint: &topologymanager.TopologyHint{
				NUMANodeAffinity: newNUMAAffinity(0, 1),
			},
			isViolationExpected: true,
		},
		{
			description: "valid affinity, no prior assignments",
			machineState: map[int]*state.NUMANodeState{
				0: {
					NumberOfAssignments: 0,
					Cells:               []int{0},
				},
				1: {
					NumberOfAssignments: 0,
					Cells:               []int{1},
				},
			},
			topologyHint: &topologymanager.TopologyHint{
				NUMANodeAffinity: newNUMAAffinity(0, 1),
			},
			isViolationExpected: false,
		},
	}
	for _, tc := range testsCases {
		t.Run(tc.description, func(t *testing.T) {
			if isAffinityViolatingNUMAAllocations(tc.machineState, tc.topologyHint.NUMANodeAffinity) != tc.isViolationExpected {
				t.Errorf("isAffinityViolatingNUMAAllocations with affinity %v expected to return %t, got %t", tc.topologyHint.NUMANodeAffinity.GetBits(), tc.isViolationExpected, !tc.isViolationExpected)
			}
		})
	}
}
