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

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
)

type staticPolicyTest struct {
	description     string
	topo            *topology.CPUTopology
	numReservedCPUs int
	podUID          string
	containerName   string
	stAssignments   state.ContainerCPUAssignments
	stDefaultCPUSet cpuset.CPUSet
	pod             *v1.Pod
	expErr          error
	expCPUAlloc     bool
	expCSet         cpuset.CPUSet
}

func TestStaticPolicyName(t *testing.T) {
	policy, _ := NewStaticPolicy(topoSingleSocketHT, 1, cpuset.NewCPUSet(), topologymanager.NewFakeManager())

	policyName := policy.Name()
	if policyName != "static" {
		t.Errorf("StaticPolicy Name() error. expected: static, returned: %v",
			policyName)
	}
}

func TestStaticPolicyStart(t *testing.T) {
	testCases := []staticPolicyTest{
		{
			description: "non-corrupted state",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.NewCPUSet(0),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expCSet:         cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "empty cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(),
			expCSet:         cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "reserved cores 0 & 6 are not present in available cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1),
			expErr:          fmt.Errorf("not all reserved cpus: \"0,6\" are present in defaultCpuSet: \"0-1\""),
		},
		{
			description: "assigned core 2 is still present in available cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.NewCPUSet(0, 1, 2),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expErr:          fmt.Errorf("pod: fakePod, container: 0 cpuset: \"0-2\" overlaps with default cpuset \"2-11\""),
		},
		{
			description: "core 12 is not present in topology but is in state cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.NewCPUSet(0, 1, 2),
					"1": cpuset.NewCPUSet(3, 4),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(5, 6, 7, 8, 9, 10, 11, 12),
			expErr:          fmt.Errorf("current set of available CPUs \"0-11\" doesn't match with CPUs in state \"0-12\""),
		},
		{
			description: "core 11 is present in topology but is not in state cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.NewCPUSet(0, 1, 2),
					"1": cpuset.NewCPUSet(3, 4),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(5, 6, 7, 8, 9, 10),
			expErr:          fmt.Errorf("current set of available CPUs \"0-11\" doesn't match with CPUs in state \"0-10\""),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, _ := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, cpuset.NewCPUSet(), topologymanager.NewFakeManager())
			policy := p.(*staticPolicy)
			st := &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}
			err := policy.Start(st)
			if !reflect.DeepEqual(err, testCase.expErr) {
				t.Errorf("StaticPolicy Start() error (%v). expected error: %v but got: %v",
					testCase.description, testCase.expErr, err)
			}
			if err != nil {
				return
			}

			if !testCase.stDefaultCPUSet.IsEmpty() {
				for cpuid := 1; cpuid < policy.topology.NumCPUs; cpuid++ {
					if !st.defaultCPUSet.Contains(cpuid) {
						t.Errorf("StaticPolicy Start() error. expected cpuid %d to be present in defaultCPUSet", cpuid)
					}
				}
			}
			if !st.GetDefaultCPUSet().Equals(testCase.expCSet) {
				t.Errorf("State CPUSet is different than expected. Have %q wants: %q", st.GetDefaultCPUSet(),
					testCase.expCSet)
			}

		})
	}
}

func TestStaticPolicyAdd(t *testing.T) {
	largeTopoBuilder := cpuset.NewBuilder()
	largeTopoSock0Builder := cpuset.NewBuilder()
	largeTopoSock1Builder := cpuset.NewBuilder()
	largeTopo := *topoQuadSocketFourWayHT
	for cpuid, val := range largeTopo.CPUDetails {
		largeTopoBuilder.Add(cpuid)
		if val.SocketID == 0 {
			largeTopoSock0Builder.Add(cpuid)
		} else if val.SocketID == 1 {
			largeTopoSock1Builder.Add(cpuid)
		}
	}
	largeTopoCPUSet := largeTopoBuilder.Result()
	largeTopoSock0CPUSet := largeTopoSock0Builder.Result()
	largeTopoSock1CPUSet := largeTopoSock1Builder.Result()

	testCases := []staticPolicyTest{
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request"),
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "1000m", "1000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(4), // expect sibling of partial core
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectAllocOneCore",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "2000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(1, 5),
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectSameAllocation",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.NewCPUSet(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(2, 3, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocOneSocket",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(2),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "6000m", "6000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(1, 3, 5, 7, 9, 11),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocThreeCores",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(1, 5),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 2, 3, 4, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "6000m", "6000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(2, 3, 4, 8, 9, 10),
		},
		{
			description:     "GuPodMultipleCores, DualSocketNoHT, ExpectAllocOneSocket",
			topo:            topoDualSocketNoHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(4, 5, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketNoHT, ExpectAllocFourCores",
			topo:            topoDualSocketNoHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(4, 5),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 3, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(1, 3, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocOneSocketOneCore",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(2),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "8000m", "8000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(1, 3, 4, 5, 7, 9, 10, 11),
		},
		{
			description:     "NonGuPod, SingleSocketHT, NoAlloc",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "1000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			description:     "GuPodNonIntegerCore, SingleSocketHT, NoAlloc",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer4", "977m", "977m"),
			expErr:          nil,
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, NoAllocExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(1, 2, 3, 4, 5, 6),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 7),
			pod:             makePod("fakePod", "fakeContainer5", "2000m", "2000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request"),
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, NoAllocExpectError",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(1, 2, 3),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer5", "10000m", "10000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request"),
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			// All the CPUs from Socket 0 are available. Some CPUs from each
			// Socket have been already assigned.
			// Expect all CPUs from Socket 0.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocSock0",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(3, 11, 4, 5, 6, 7),
				},
			},
			stDefaultCPUSet: largeTopoCPUSet.Difference(cpuset.NewCPUSet(3, 11, 4, 5, 6, 7)),
			pod:             makePod("fakePod", "fakeContainer5", "72000m", "72000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         largeTopoSock0CPUSet,
		},
		{
			// Only 2 full cores from three Sockets and some partial cores are available.
			// Expect CPUs from the 2 full cores available from the three Sockets.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocAllFullCoresFromThreeSockets",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.NewCPUSet(1, 25, 13, 38, 2, 9, 11, 35, 23, 48, 12, 51,
						53, 173, 113, 233, 54, 61)),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(1, 25, 13, 38, 2, 9, 11, 35, 23, 48, 12, 51, 53, 173, 113, 233, 54, 61),
			pod:             makePod("fakePod", "fakeCcontainer5", "12000m", "12000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(1, 25, 13, 38, 11, 35, 23, 48, 53, 173, 113, 233),
		},
		{
			// All CPUs from Socket 1, 1 full core and some partial cores are available.
			// Expect all CPUs from Socket 1 and the hyper-threads from the full core.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocAllSock1+FullCore",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(largeTopoSock1CPUSet.Union(cpuset.NewCPUSet(10, 34, 22, 47, 53,
						173, 61, 181, 108, 228, 115, 235))),
				},
			},
			stDefaultCPUSet: largeTopoSock1CPUSet.Union(cpuset.NewCPUSet(10, 34, 22, 47, 53, 173, 61, 181, 108, 228,
				115, 235)),
			pod:         makePod("fakePod", "fakeContainer5", "76000m", "76000m"),
			expErr:      nil,
			expCPUAlloc: true,
			expCSet:     largeTopoSock1CPUSet.Union(cpuset.NewCPUSet(10, 34, 22, 47)),
		},
		{
			// Only partial cores are available in the entire system.
			// Expect allocation of all the CPUs from the partial cores.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocCPUs",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.NewCPUSet(10, 11, 53, 37, 55, 67, 52)),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(10, 11, 53, 67, 52),
			pod:             makePod("fakePod", "fakeContainer5", "5000m", "5000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(10, 11, 53, 67, 52),
		},
		{
			// Only 7 CPUs are available.
			// Pod requests 76 cores.
			// Error is expect since available CPUs are less than the request.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, NoAlloc",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.NewCPUSet(10, 11, 53, 37, 55, 67, 52)),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(10, 11, 53, 37, 55, 67, 52),
			pod:             makePod("fakePod", "fakeContainer5", "76000m", "76000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request"),
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
	}

	for _, testCase := range testCases {
		policy, _ := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, cpuset.NewCPUSet(), topologymanager.NewFakeManager())

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		container := &testCase.pod.Spec.Containers[0]
		err := policy.Allocate(st, testCase.pod, container)
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected add error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}

		if testCase.expCPUAlloc {
			cset, found := st.assignments[string(testCase.pod.UID)][container.Name]
			if !found {
				t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
					testCase.description, container.Name, st.assignments)
			}

			if !reflect.DeepEqual(cset, testCase.expCSet) {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %v but got %v",
					testCase.description, testCase.expCSet, cset)
			}

			if !cset.Intersection(st.defaultCPUSet).IsEmpty() {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %v to be disoint from the shared cpuset %v",
					testCase.description, cset, st.defaultCPUSet)
			}
		}

		if !testCase.expCPUAlloc {
			_, found := st.assignments[string(testCase.pod.UID)][container.Name]
			if found {
				t.Errorf("StaticPolicy Allocate() error (%v). Did not expect container %v to be present in assignments %v",
					testCase.description, container.Name, st.assignments)
			}
		}
	}
}

func TestStaticPolicyRemove(t *testing.T) {
	testCases := []staticPolicyTest{
		{
			description:   "SingleSocketHT, DeAllocOneContainer",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer1",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.NewCPUSet(1, 2, 3),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(4, 5, 6, 7),
			expCSet:         cpuset.NewCPUSet(1, 2, 3, 4, 5, 6, 7),
		},
		{
			description:   "SingleSocketHT, DeAllocOneContainer, BeginEmpty",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer1",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.NewCPUSet(1, 2, 3),
					"fakeContainer2": cpuset.NewCPUSet(4, 5, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(),
			expCSet:         cpuset.NewCPUSet(1, 2, 3),
		},
		{
			description:   "SingleSocketHT, DeAllocTwoContainer",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer1",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.NewCPUSet(1, 3, 5),
					"fakeContainer2": cpuset.NewCPUSet(2, 4),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(6, 7),
			expCSet:         cpuset.NewCPUSet(1, 3, 5, 6, 7),
		},
		{
			description:   "SingleSocketHT, NoDeAlloc",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer2",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.NewCPUSet(1, 3, 5),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(2, 4, 6, 7),
			expCSet:         cpuset.NewCPUSet(2, 4, 6, 7),
		},
	}

	for _, testCase := range testCases {
		policy, _ := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, cpuset.NewCPUSet(), topologymanager.NewFakeManager())

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		policy.RemoveContainer(st, testCase.podUID, testCase.containerName)

		if !reflect.DeepEqual(st.defaultCPUSet, testCase.expCSet) {
			t.Errorf("StaticPolicy RemoveContainer() error (%v). expected default cpuset %v but got %v",
				testCase.description, testCase.expCSet, st.defaultCPUSet)
		}

		if _, found := st.assignments[testCase.podUID][testCase.containerName]; found {
			t.Errorf("StaticPolicy RemoveContainer() error (%v). expected (pod %v, container %v) not be in assignments %v",
				testCase.description, testCase.podUID, testCase.containerName, st.assignments)
		}
	}
}

func TestTopologyAwareAllocateCPUs(t *testing.T) {
	testCases := []struct {
		description     string
		topo            *topology.CPUTopology
		stAssignments   state.ContainerCPUAssignments
		stDefaultCPUSet cpuset.CPUSet
		numRequested    int
		socketMask      bitmask.BitMask
		expCSet         cpuset.CPUSet
	}{
		{
			description:     "Request 2 CPUs, No BitMask",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask:      nil,
			expCSet:         cpuset.NewCPUSet(0, 6),
		},
		{
			description:     "Request 2 CPUs, BitMask on Socket 0",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(0)
				return mask
			}(),
			expCSet: cpuset.NewCPUSet(0, 6),
		},
		{
			description:     "Request 2 CPUs, BitMask on Socket 1",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(1)
				return mask
			}(),
			expCSet: cpuset.NewCPUSet(1, 7),
		},
		{
			description:     "Request 8 CPUs, BitMask on Socket 0",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    8,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(0)
				return mask
			}(),
			expCSet: cpuset.NewCPUSet(0, 6, 2, 8, 4, 10, 1, 7),
		},
		{
			description:     "Request 8 CPUs, BitMask on Socket 1",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    8,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(1)
				return mask
			}(),
			expCSet: cpuset.NewCPUSet(1, 7, 3, 9, 5, 11, 0, 6),
		},
	}
	for _, tc := range testCases {
		p, _ := NewStaticPolicy(tc.topo, 0, cpuset.NewCPUSet(), topologymanager.NewFakeManager())
		policy := p.(*staticPolicy)
		st := &mockState{
			assignments:   tc.stAssignments,
			defaultCPUSet: tc.stDefaultCPUSet,
		}
		err := policy.Start(st)
		if err != nil {
			t.Errorf("StaticPolicy Start() error (%v)", err)
			continue
		}

		cset, err := policy.allocateCPUs(st, tc.numRequested, tc.socketMask, cpuset.NewCPUSet())
		if err != nil {
			t.Errorf("StaticPolicy allocateCPUs() error (%v). expected CPUSet %v not error %v",
				tc.description, tc.expCSet, err)
			continue
		}

		if !reflect.DeepEqual(tc.expCSet, cset) {
			t.Errorf("StaticPolicy allocateCPUs() error (%v). expected CPUSet %v but got %v",
				tc.description, tc.expCSet, cset)
		}
	}
}

// above test cases are without kubelet --reserved-cpus cmd option
// the following tests are with --reserved-cpus configured
type staticPolicyTestWithResvList struct {
	description     string
	topo            *topology.CPUTopology
	numReservedCPUs int
	reserved        cpuset.CPUSet
	stAssignments   state.ContainerCPUAssignments
	stDefaultCPUSet cpuset.CPUSet
	pod             *v1.Pod
	expErr          error
	expNewErr       error
	expCPUAlloc     bool
	expCSet         cpuset.CPUSet
}

func TestStaticPolicyStartWithResvList(t *testing.T) {
	testCases := []staticPolicyTestWithResvList{
		{
			description:     "empty cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.NewCPUSet(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(),
			expCSet:         cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "reserved cores 0 & 1 are not present in available cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.NewCPUSet(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(2, 3, 4, 5),
			expErr:          fmt.Errorf("not all reserved cpus: \"0-1\" are present in defaultCpuSet: \"2-5\""),
		},
		{
			description:     "inconsistency between numReservedCPUs and reserved",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reserved:        cpuset.NewCPUSet(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expNewErr:       fmt.Errorf("[cpumanager] unable to reserve the required amount of CPUs (size of 0-1 did not equal 1)"),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			p, err := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager())
			if !reflect.DeepEqual(err, testCase.expNewErr) {
				t.Errorf("StaticPolicy Start() error (%v). expected error: %v but got: %v",
					testCase.description, testCase.expNewErr, err)
			}
			if err != nil {
				return
			}
			policy := p.(*staticPolicy)
			st := &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}
			err = policy.Start(st)
			if !reflect.DeepEqual(err, testCase.expErr) {
				t.Errorf("StaticPolicy Start() error (%v). expected error: %v but got: %v",
					testCase.description, testCase.expErr, err)
			}
			if err != nil {
				return
			}

			if !st.GetDefaultCPUSet().Equals(testCase.expCSet) {
				t.Errorf("State CPUSet is different than expected. Have %q wants: %q", st.GetDefaultCPUSet(),
					testCase.expCSet)
			}

		})
	}
}

func TestStaticPolicyAddWithResvList(t *testing.T) {

	testCases := []staticPolicyTestWithResvList{
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			reserved:        cpuset.NewCPUSet(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request"),
			expCPUAlloc:     false,
			expCSet:         cpuset.NewCPUSet(),
		},
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.NewCPUSet(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "1000m", "1000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(4), // expect sibling of partial core
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectAllocOneCore",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.NewCPUSet(0, 1),
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.NewCPUSet(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.NewCPUSet(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "2000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.NewCPUSet(4, 5),
		},
	}

	for _, testCase := range testCases {
		policy, _ := NewStaticPolicy(testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager())

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		container := &testCase.pod.Spec.Containers[0]
		err := policy.Allocate(st, testCase.pod, container)
		if !reflect.DeepEqual(err, testCase.expErr) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected add error: %v but got: %v",
				testCase.description, testCase.expErr, err)
		}

		if testCase.expCPUAlloc {
			cset, found := st.assignments[string(testCase.pod.UID)][container.Name]
			if !found {
				t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
					testCase.description, container.Name, st.assignments)
			}

			if !reflect.DeepEqual(cset, testCase.expCSet) {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %v but got %v",
					testCase.description, testCase.expCSet, cset)
			}

			if !cset.Intersection(st.defaultCPUSet).IsEmpty() {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %v to be disoint from the shared cpuset %v",
					testCase.description, cset, st.defaultCPUSet)
			}
		}

		if !testCase.expCPUAlloc {
			_, found := st.assignments[string(testCase.pod.UID)][container.Name]
			if found {
				t.Errorf("StaticPolicy Allocate() error (%v). Did not expect container %v to be present in assignments %v",
					testCase.description, container.Name, st.assignments)
			}
		}
	}
}
