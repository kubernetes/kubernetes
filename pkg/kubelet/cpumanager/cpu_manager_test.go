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
	"testing"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
)

type mockState struct {
	assignments   map[string]cpuset.CPUSet
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

type mockPolicy struct {
	err           error
	underPressure bool
}

func (p *mockPolicy) Name() string {
	return "mock"
}

func (p *mockPolicy) Start(s state.State) {
}

func (p *mockPolicy) RegisterContainer(s state.State, pod *v1.Pod, container *v1.Container, containerID string) error {
	return p.err
}

func (p *mockPolicy) UnregisterContainer(s state.State, containerID string) error {
	return p.err
}

func (p *mockPolicy) IsUnderPressure() bool {
	return p.underPressure
}

type mockRuntimeService struct {
	err error
}

func (rt mockRuntimeService) UpdateContainerResources(id string, resources *runtimeapi.LinuxContainerResources) error {
	return rt.err
}

type mockKletGetter struct {
	pods        []*v1.Pod
	machineInfo *cadvisorapi.MachineInfo
	node        *v1.Node
	err         error
}

func (klg mockKletGetter) GetPods() []*v1.Pod {
	return klg.pods
}

func (klg mockKletGetter) GetCachedMachineInfo() (*cadvisorapi.MachineInfo, error) {
	return klg.machineInfo, klg.err
}

func (klg mockKletGetter) GetNode() (*v1.Node, error) {
	return klg.node, klg.err
}

type mockPodStatusProvider struct {
	podStatus v1.PodStatus
	found     bool
}

func (psp mockPodStatusProvider) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	return psp.podStatus, psp.found
}

type mockPodKiller struct {
	killedPods []*v1.Pod
}

func (f *mockPodKiller) killPodNow(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error {
	f.killedPods = append(f.killedPods, pod)
	return nil
}

type mockPodProvider struct {
	pods []*v1.Pod
}

func (f *mockPodProvider) getPods() []*v1.Pod {
	return f.pods
}

type mockRecorder struct{}

func (r *mockRecorder) Eventf(object runtime.Object, eventtype, reason, messageFmt string, args ...interface{}) {
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

// CpuAllocatable must be <= CpuCapacity
func prepareCPUNodeStatus(CPUCapacity, CPUAllocatable string) v1.NodeStatus {
	nodestatus := v1.NodeStatus{
		Capacity:    make(v1.ResourceList, 1),
		Allocatable: make(v1.ResourceList, 1),
	}
	cpucap, _ := resource.ParseQuantity(CPUCapacity)
	cpuall, _ := resource.ParseQuantity(CPUAllocatable)

	nodestatus.Capacity[v1.ResourceCPU] = cpucap
	nodestatus.Allocatable[v1.ResourceCPU] = cpuall
	return nodestatus
}

func TestCPUManagerRegister(t *testing.T) {
	testCases := []struct {
		description string
		regErr      error
		updateErr   error
		expErr      error
	}{
		{
			description: "cpu manager register - no error",
			regErr:      nil,
			updateErr:   nil,
			expErr:      nil,
		},
		{
			description: "cpu manager register - policy register container error",
			regErr:      fmt.Errorf("fake reg error"),
			updateErr:   nil,
			expErr:      fmt.Errorf("fake reg error"),
		},
		{
			description: "cpu manager register - container update error",
			regErr:      nil,
			updateErr:   fmt.Errorf("fake update error"),
			expErr:      fmt.Errorf("fake update error"),
		},
	}

	for _, testCase := range testCases {
		mgr := &manager{
			policy: &mockPolicy{
				err: testCase.regErr,
			},
			state: &mockState{
				assignments:   map[string]cpuset.CPUSet{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
			containerRuntime: mockRuntimeService{
				err: testCase.updateErr,
			},
			kletGetter:        mockKletGetter{},
			podStatusProvider: mockPodStatusProvider{},
		}

		pod := makePod("1000", "1000")
		container := &pod.Spec.Containers[0]
		err := mgr.RegisterContainer(pod, container, "fakeID")
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("CPU Manager Register() error (%v). expected register error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}
	}
}

func TestCPUManagerUnRegister(t *testing.T) {
	mgr := &manager{
		policy: &mockPolicy{
			err: nil,
		},
		state: &mockState{
			assignments:   map[string]cpuset.CPUSet{},
			defaultCPUSet: cpuset.NewCPUSet(),
		},
		containerRuntime:  mockRuntimeService{},
		kletGetter:        mockKletGetter{},
		podStatusProvider: mockPodStatusProvider{},
	}

	err := mgr.UnregisterContainer("fakeID")
	if err != nil {
		t.Errorf("CPU Manager UnRegister() error. expected unregister error to be nil but got: %v", err)
	}

	mgr = &manager{
		policy: &mockPolicy{
			err: fmt.Errorf("fake error"),
		},
		state:             state.NewMemoryState(),
		containerRuntime:  mockRuntimeService{},
		kletGetter:        mockKletGetter{},
		podStatusProvider: mockPodStatusProvider{},
	}

	err = mgr.UnregisterContainer("fakeID")
	if !reflect.DeepEqual(err, fmt.Errorf("fake error")) {
		t.Errorf("CPU Manager UnRegister() error. expected unregister error: fake error but got: %v", err)
	}
}

func TestReconcileState(t *testing.T) {
	testCases := []struct {
		description               string
		klgPods                   []*v1.Pod
		pspPS                     v1.PodStatus
		pspFound                  bool
		stAssignments             map[string]cpuset.CPUSet
		stDefaultCPUSet           cpuset.CPUSet
		updateErr                 error
		expectFailedContainerName string
	}{
		{
			description: "cpu manager reconclie - no error",
			klgPods: []*v1.Pod{
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
			stAssignments: map[string]cpuset.CPUSet{
				"fakeID": cpuset.NewCPUSet(1, 2),
			},
			stDefaultCPUSet:           cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:                 nil,
			expectFailedContainerName: "",
		},
		{
			description: "cpu manager reconclie - pod status not found",
			klgPods: []*v1.Pod{
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
			pspPS:                     v1.PodStatus{},
			pspFound:                  false,
			stAssignments:             map[string]cpuset.CPUSet{},
			stDefaultCPUSet:           cpuset.NewCPUSet(),
			updateErr:                 nil,
			expectFailedContainerName: "fakeName",
		},
		{
			description: "cpu manager reconclie - container id not found",
			klgPods: []*v1.Pod{
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
			pspFound:                  true,
			stAssignments:             map[string]cpuset.CPUSet{},
			stDefaultCPUSet:           cpuset.NewCPUSet(),
			updateErr:                 nil,
			expectFailedContainerName: "fakeName",
		},
		{
			description: "cpu manager reconclie - cpuset is empty",
			klgPods: []*v1.Pod{
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
			stAssignments: map[string]cpuset.CPUSet{
				"fakeID": cpuset.NewCPUSet(),
			},
			stDefaultCPUSet:           cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7),
			updateErr:                 nil,
			expectFailedContainerName: "fakeName",
		},
		{
			description: "cpu manager reconclie - container update error",
			klgPods: []*v1.Pod{
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
			stAssignments: map[string]cpuset.CPUSet{
				"fakeID": cpuset.NewCPUSet(1, 2),
			},
			stDefaultCPUSet:           cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:                 fmt.Errorf("fake container update error"),
			expectFailedContainerName: "fakeName",
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
			kletGetter: mockKletGetter{
				pods: testCase.klgPods,
			},
			podStatusProvider: mockPodStatusProvider{
				podStatus: testCase.pspPS,
				found:     testCase.pspFound,
			},
		}

		_, failure := mgr.reconcileState()

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
				t.Errorf("Expected reconciliation failure for container: %s", testCase.expectFailedContainerName)
			}
		}
	}
}

func TestEvictNoCPUPods(t *testing.T) {
	bestEffortPod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{},
					},
				},
			},
		},
	}
	burstablePodNoCPU := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}
	burstablePodWithCPU := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}
	burstablePodWithBoth := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}
	guaranteedPod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
						Limits: v1.ResourceList{
							v1.ResourceName(v1.ResourceCPU):    resource.MustParse("1"),
							v1.ResourceName(v1.ResourceMemory): resource.MustParse("1G"),
						},
					},
				},
			},
		},
	}

	allPods := []*v1.Pod{bestEffortPod, burstablePodNoCPU, burstablePodWithCPU, burstablePodWithBoth, guaranteedPod}
	expectedEvictedPods := []*v1.Pod{bestEffortPod, burstablePodNoCPU, burstablePodWithBoth}
	podProvider := &mockPodProvider{pods: allPods}
	podKiller := &mockPodKiller{}
	recorder := &record.FakeRecorder{}
	mgr := &manager{
		getPodsFunc: podProvider.getPods,
		killPodFunc: podKiller.killPodNow,
		recorder:    recorder,
	}
	mgr.evictNoCPUPods()
	if !reflect.DeepEqual(podKiller.killedPods, expectedEvictedPods) {
		t.Errorf("killed pods did not match expected result - killed: %v, expected: %v", podKiller.killedPods, expectedEvictedPods)
	}
}
