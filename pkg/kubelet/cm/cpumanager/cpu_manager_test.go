/*
Copyright 2017 The Kubernetes Authors.

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

package cpumanager

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"io/ioutil"
	"os"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
)

type mockState struct {
	assignments   state.ContainerCPUAssignments
	defaultCPUSet cpuset.CPUSet
}

func (s *mockState) GetCPUSet(containerID string) (cpuset.CPUSet, bool) {
	res, ok := s.assignments[containerID]
	return res.Clone(), ok
}

func (s *mockState) GetDefaultCPUSet() cpuset.CPUSet {
	return s.defaultCPUSet.Clone()
}

func (s *mockState) GetCPUSetOrDefault(containerID string) cpuset.CPUSet {
	if res, ok := s.GetCPUSet(containerID); ok {
		return res
	}
	return s.GetDefaultCPUSet()
}

func (s *mockState) SetCPUSet(containerID string, cset cpuset.CPUSet) {
	s.assignments[containerID] = cset
}

func (s *mockState) SetDefaultCPUSet(cset cpuset.CPUSet) {
	s.defaultCPUSet = cset
}

func (s *mockState) Delete(containerID string) {
	delete(s.assignments, containerID)
}

func (s *mockState) ClearState() {
	s.defaultCPUSet = cpuset.CPUSet{}
	s.assignments = make(state.ContainerCPUAssignments)
}

func (s *mockState) SetCPUAssignments(a state.ContainerCPUAssignments) {
	s.assignments = a.Clone()
}

func (s *mockState) GetCPUAssignments() state.ContainerCPUAssignments {
	return s.assignments.Clone()
}

type mockPolicy struct {
	err error
}

func (p *mockPolicy) Name() string {
	return "mock"
}

func (p *mockPolicy) Start(s state.State) {
}

func (p *mockPolicy) AddContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error {
	return p.err
}

func (p *mockPolicy) RemoveContainer(s state.State, containerID string) error {
	return p.err
}

type mockRuntimeService struct {
	err error
}

func (rt mockRuntimeService) UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error {
	return rt.err
}

type mockPodStatusProvider struct {
	podStatus v1.PodStatus
	found     bool
}

func (psp mockPodStatusProvider) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	return psp.podStatus, psp.found
}

func makePod(cpuRequest, cpuLimit string) *v1.Pod {
	return &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpuRequest),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpuLimit),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}
}

func makeMultiContainerPod(initCPUs, appCPUs []struct{ request, limit string }) *v1.Pod {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{},
			Containers:     []v1.Container{},
		},
	}

	for i, cpu := range initCPUs {
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, v1.Container{
			Name: "initContainer-" + string(i),
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpu.request),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpu.limit),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
				},
			},
		})
	}

	for i, cpu := range appCPUs {
		pod.Spec.Containers = append(pod.Spec.Containers, v1.Container{
			Name: "appContainer-" + string(i),
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpu.request),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(cpu.limit),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
				},
			},
		})
	}

	return pod
}

func TestCPUManagerAdd(t *testing.T) {
	testPolicy := NewStaticPolicy(
		&topology.CPUTopology{
			NumCPUs:    4,
			NumSockets: 1,
			NumCores:   4,
			CPUDetails: map[int]topology.CPUInfo{
				0: {CoreID: 0, SocketID: 0},
				1: {CoreID: 1, SocketID: 0},
				2: {CoreID: 2, SocketID: 0},
				3: {CoreID: 3, SocketID: 0},
			},
		}, 0, topologymanager.NewFakeManager())
	testCases := []struct {
		description string
		updateErr   error
		policy      Policy
		expCPUSet   cpuset.CPUSet
		expErr      error
	}{
		{
			description: "cpu manager add - no error",
			updateErr:   nil,
			policy:      testPolicy,
			expCPUSet:   cpuset.NewCPUSet(3, 4),
			expErr:      nil,
		},
		{
			description: "cpu manager add - policy add container error",
			updateErr:   nil,
			policy: &mockPolicy{
				err: fmt.Errorf("fake reg error"),
			},
			expCPUSet: cpuset.NewCPUSet(1, 2, 3, 4),
			expErr:    fmt.Errorf("fake reg error"),
		},
		{
			description: "cpu manager add - container update error",
			updateErr:   fmt.Errorf("fake update error"),
			policy:      testPolicy,
			expCPUSet:   cpuset.NewCPUSet(1, 2, 3, 4),
			expErr:      fmt.Errorf("fake update error"),
		},
	}

	for _, testCase := range testCases {
		mgr := &manager{
			policy: testCase.policy,
			state: &mockState{
				assignments:   state.ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(1, 2, 3, 4),
			},
			containerRuntime: mockRuntimeService{
				err: testCase.updateErr,
			},
			activePods:        func() []*v1.Pod { return nil },
			podStatusProvider: mockPodStatusProvider{},
		}

		pod := makePod("2", "2")
		container := &pod.Spec.Containers[0]
		err := mgr.AddContainer(pod, container, "fakeID")
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("CPU Manager AddContainer() error (%v). expected error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
		if !testCase.expCPUSet.Equals(mgr.state.GetDefaultCPUSet()) {
			t.Errorf("CPU Manager AddContainer() error (%v). expected cpuset: %v but got: %v",
				testCase.description, testCase.expCPUSet, mgr.state.GetDefaultCPUSet())
		}
	}
}

func TestCPUManagerGenerate(t *testing.T) {
	testCases := []struct {
		description                string
		cpuPolicyName              string
		nodeAllocatableReservation v1.ResourceList
		isTopologyBroken           bool
		expectedPolicy             string
		expectedError              error
	}{
		{
			description:                "set none policy",
			cpuPolicyName:              "none",
			nodeAllocatableReservation: nil,
			expectedPolicy:             "none",
		},
		{
			description:                "invalid policy name",
			cpuPolicyName:              "invalid",
			nodeAllocatableReservation: nil,
			expectedError:              fmt.Errorf("unknown policy: \"invalid\""),
		},
		{
			description:                "static policy",
			cpuPolicyName:              "static",
			nodeAllocatableReservation: v1.ResourceList{v1.ResourceCPU: *resource.NewQuantity(3, resource.DecimalSI)},
			expectedPolicy:             "static",
		},
		{
			description:                "static policy - broken topology",
			cpuPolicyName:              "static",
			nodeAllocatableReservation: v1.ResourceList{},
			isTopologyBroken:           true,
			expectedError:              fmt.Errorf("could not detect number of cpus"),
		},
		{
			description:                "static policy - broken reservation",
			cpuPolicyName:              "static",
			nodeAllocatableReservation: v1.ResourceList{},
			expectedError:              fmt.Errorf("unable to determine reserved CPU resources for static policy"),
		},
		{
			description:                "static policy - no CPU resources",
			cpuPolicyName:              "static",
			nodeAllocatableReservation: v1.ResourceList{v1.ResourceCPU: *resource.NewQuantity(0, resource.DecimalSI)},
			expectedError:              fmt.Errorf("the static policy requires systemreserved.cpu + kubereserved.cpu to be greater than zero"),
		},
	}

	mockedMachineInfo := cadvisorapi.MachineInfo{
		NumCores: 4,
		Topology: []cadvisorapi.Node{
			{
				Cores: []cadvisorapi.Core{
					{
						Id:      0,
						Threads: []int{0},
					},
					{
						Id:      1,
						Threads: []int{1},
					},
					{
						Id:      2,
						Threads: []int{2},
					},
					{
						Id:      3,
						Threads: []int{3},
					},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			machineInfo := &mockedMachineInfo
			if testCase.isTopologyBroken {
				machineInfo = &cadvisorapi.MachineInfo{}
			}
			sDir, err := ioutil.TempDir("/tmp/", "cpu_manager_test")
			if err != nil {
				t.Errorf("cannot create state file: %s", err.Error())
			}
			defer os.RemoveAll(sDir)

			mgr, err := NewManager(testCase.cpuPolicyName, 5*time.Second, machineInfo, testCase.nodeAllocatableReservation, sDir, topologymanager.NewFakeManager())
			if testCase.expectedError != nil {
				if !strings.Contains(err.Error(), testCase.expectedError.Error()) {
					t.Errorf("Unexpected error message. Have: %s wants %s", err.Error(), testCase.expectedError.Error())
				}
			} else {
				rawMgr := mgr.(*manager)
				if rawMgr.policy.Name() != testCase.expectedPolicy {
					t.Errorf("Unexpected policy name. Have: %q wants %q", rawMgr.policy.Name(), testCase.expectedPolicy)
				}
			}
		})

	}
}

func TestCPUManagerRemove(t *testing.T) {
	mgr := &manager{
		policy: &mockPolicy{
			err: nil,
		},
		state: &mockState{
			assignments:   state.ContainerCPUAssignments{},
			defaultCPUSet: cpuset.NewCPUSet(),
		},
		containerRuntime:  mockRuntimeService{},
		activePods:        func() []*v1.Pod { return nil },
		podStatusProvider: mockPodStatusProvider{},
	}

	err := mgr.RemoveContainer("fakeID")
	if err != nil {
		t.Errorf("CPU Manager RemoveContainer() error. expected error to be nil but got: %v", err)
	}

	mgr = &manager{
		policy: &mockPolicy{
			err: fmt.Errorf("fake error"),
		},
		state:             state.NewMemoryState(),
		containerRuntime:  mockRuntimeService{},
		activePods:        func() []*v1.Pod { return nil },
		podStatusProvider: mockPodStatusProvider{},
	}

	err = mgr.RemoveContainer("fakeID")
	if !reflect.DeepEqual(err, fmt.Errorf("fake error")) {
		t.Errorf("CPU Manager RemoveContainer() error. expected error: fake error but got: %v", err)
	}
}

func TestReconcileState(t *testing.T) {
	testCases := []struct {
		description                  string
		activePods                   []*v1.Pod
		pspPS                        v1.PodStatus
		pspFound                     bool
		stAssignments                state.ContainerCPUAssignments
		stDefaultCPUSet              cpuset.CPUSet
		updateErr                    error
		expectSucceededContainerName string
		expectFailedContainerName    string
	}{
		{
			description: "cpu manager reconclie - no error",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					{
						Name:        "fakeName",
						ContainerID: "docker://fakeID",
					},
				},
			},
			pspFound: true,
			stAssignments: state.ContainerCPUAssignments{
				"fakeID": cpuset.NewCPUSet(1, 2),
			},
			stDefaultCPUSet:              cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:                    nil,
			expectSucceededContainerName: "fakeName",
			expectFailedContainerName:    "",
		},
		{
			description: "cpu manager reconcile init container - no error",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS: v1.PodStatus{
				InitContainerStatuses: []v1.ContainerStatus{
					{
						Name:        "fakeName",
						ContainerID: "docker://fakeID",
					},
				},
			},
			pspFound: true,
			stAssignments: state.ContainerCPUAssignments{
				"fakeID": cpuset.NewCPUSet(1, 2),
			},
			stDefaultCPUSet:              cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:                    nil,
			expectSucceededContainerName: "fakeName",
			expectFailedContainerName:    "",
		},
		{
			description: "cpu manager reconclie - pod status not found",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS:                        v1.PodStatus{},
			pspFound:                     false,
			stAssignments:                state.ContainerCPUAssignments{},
			stDefaultCPUSet:              cpuset.NewCPUSet(),
			updateErr:                    nil,
			expectSucceededContainerName: "",
			expectFailedContainerName:    "fakeName",
		},
		{
			description: "cpu manager reconclie - container id not found",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					{
						Name:        "fakeName1",
						ContainerID: "docker://fakeID",
					},
				},
			},
			pspFound:                     true,
			stAssignments:                state.ContainerCPUAssignments{},
			stDefaultCPUSet:              cpuset.NewCPUSet(),
			updateErr:                    nil,
			expectSucceededContainerName: "",
			expectFailedContainerName:    "fakeName",
		},
		{
			description: "cpu manager reconclie - cpuset is empty",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					{
						Name:        "fakeName",
						ContainerID: "docker://fakeID",
					},
				},
			},
			pspFound: true,
			stAssignments: state.ContainerCPUAssignments{
				"fakeID": cpuset.NewCPUSet(),
			},
			stDefaultCPUSet:              cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7),
			updateErr:                    nil,
			expectSucceededContainerName: "",
			expectFailedContainerName:    "fakeName",
		},
		{
			description: "cpu manager reconclie - container update error",
			activePods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: "fakePodName",
						UID:  "fakeUID",
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{
							{
								Name: "fakeName",
							},
						},
					},
				},
			},
			pspPS: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					{
						Name:        "fakeName",
						ContainerID: "docker://fakeID",
					},
				},
			},
			pspFound: true,
			stAssignments: state.ContainerCPUAssignments{
				"fakeID": cpuset.NewCPUSet(1, 2),
			},
			stDefaultCPUSet:              cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:                    fmt.Errorf("fake container update error"),
			expectSucceededContainerName: "",
			expectFailedContainerName:    "fakeName",
		},
	}

	for _, testCase := range testCases {
		mgr := &manager{
			policy: &mockPolicy{
				err: nil,
			},
			state: &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet,
			},
			containerRuntime: mockRuntimeService{
				err: testCase.updateErr,
			},
			activePods: func() []*v1.Pod {
				return testCase.activePods
			},
			podStatusProvider: mockPodStatusProvider{
				podStatus: testCase.pspPS,
				found:     testCase.pspFound,
			},
		}

		success, failure := mgr.reconcileState()

		if testCase.expectSucceededContainerName != "" {
			// Search succeeded reconciled containers for the supplied name.
			foundSucceededContainer := false
			for _, reconciled := range success {
				if reconciled.containerName == testCase.expectSucceededContainerName {
					foundSucceededContainer = true
					break
				}
			}
			if !foundSucceededContainer {
				t.Errorf("%v", testCase.description)
				t.Errorf("Expected reconciliation success for container: %s", testCase.expectSucceededContainerName)
			}
		}

		if testCase.expectFailedContainerName != "" {
			// Search failed reconciled containers for the supplied name.
			foundFailedContainer := false
			for _, reconciled := range failure {
				if reconciled.containerName == testCase.expectFailedContainerName {
					foundFailedContainer = true
					break
				}
			}
			if !foundFailedContainer {
				t.Errorf("%v", testCase.description)
				t.Errorf("Expected reconciliation failure for container: %s", testCase.expectFailedContainerName)
			}
		}
	}
}
