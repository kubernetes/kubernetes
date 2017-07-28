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
	"bytes"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cpuset"
	"testing"
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
	err error
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

func getStderr(f func()) (string, error) {
	origStderr := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		return "", err
	}

	os.Stderr = w
	outChan := make(chan string)
	go func() {
		var buff bytes.Buffer
		io.Copy(&buff, r)
		outChan <- buff.String()
	}()

	f()

	w.Close()
	os.Stderr = origStderr
	return <-outChan, nil
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
		description     string
		klgPods         []*v1.Pod
		pspPS           v1.PodStatus
		pspFound        bool
		stAssignments   map[string]cpuset.CPUSet
		stDefaultCPUSet cpuset.CPUSet
		updateErr       error
		expOut          string
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
			stDefaultCPUSet: cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:       nil,
			expOut:          "",
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
			pspPS:           v1.PodStatus{},
			pspFound:        false,
			stAssignments:   map[string]cpuset.CPUSet{},
			stDefaultCPUSet: cpuset.NewCPUSet(),
			updateErr:       nil,
			expOut:          fmt.Sprintf("[cpumanager] reconcileState: skipping pod; status not found (pod: fakePodName, container: fakeName)"),
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
			pspFound:        true,
			stAssignments:   map[string]cpuset.CPUSet{},
			stDefaultCPUSet: cpuset.NewCPUSet(),
			updateErr:       nil,
			expOut: fmt.Sprintf("[cpumanager] reconcileState: skipping container; ID not found in status (pod: fakePodName, container: fakeName, error: %v)",
				fmt.Sprintf("[cpumanager] unable to find ID for container with name fakeName in pod status (it may not be running)")),
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
			stDefaultCPUSet: cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7),
			updateErr:       nil,
			expOut:          fmt.Sprintf("[cpumanager] reconcileState: skipping container; assigned cpuset is empty (pod: fakePodName, container: fakeName)"),
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
			stDefaultCPUSet: cpuset.NewCPUSet(3, 4, 5, 6, 7),
			updateErr:       fmt.Errorf("fake container update error"),
			expOut: fmt.Sprintf("[cpumanager] reconcileState: failed to update container (pod: fakePodName, container: fakeName, container id: fakeID, cpuset: \"1-2\", error: %v)",
				fmt.Sprintf("fake container update error")),
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

		out, err := getStderr(func() { mgr.reconcileState() })
		if err != nil {
			t.Errorf("CPU Manger Reconcile() error. output capture error")
		}

		if !strings.HasSuffix(strings.TrimSpace(out), testCase.expOut) {
			t.Errorf("CPU Manager Reconcile() error (%v). expected out: %v but got: %v",
				testCase.description, testCase.expOut, out)
		}
	}
}

func TestGetReservedCpus(t *testing.T) {
	var reservedCPUTests = []struct {
		cpuCapacity     string
		cpuAllocatable  string
		expReservedCpus int
	}{
		{cpuCapacity: "1000m", cpuAllocatable: "1000m", expReservedCpus: 0},
		{cpuCapacity: "8000m", cpuAllocatable: "7500m", expReservedCpus: 0},
		{cpuCapacity: "16000m", cpuAllocatable: "14100m", expReservedCpus: 1},
		{cpuCapacity: "8000m", cpuAllocatable: "5500m", expReservedCpus: 2},
		{cpuCapacity: "8000m", cpuAllocatable: "900m", expReservedCpus: 7},
	}

	for idx, test := range reservedCPUTests {
		tmpNodeStaus := prepareCPUNodeStatus(test.cpuCapacity, test.cpuAllocatable)
		gotReservedCpus := getReservedCPUs(tmpNodeStaus)
		if test.expReservedCpus != gotReservedCpus {
			t.Errorf("(Case %d) Expected reserved cpus %d, got %d",
				idx, test.expReservedCpus, gotReservedCpus)
		}
	}
}
