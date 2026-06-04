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

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/cpuset"
)

var (
	containerRestartPolicyAlways = v1.ContainerRestartPolicyAlways
)

func newNUMAAffinity(bits ...int) bitmask.BitMask {
	affinity, err := bitmask.NewBitMask(bits...)
	if err != nil {
		panic(err)
	}
	return affinity
}

type staticPolicyTest struct {
	description     string
	topo            *topology.CPUTopology
	numReservedCPUs int
	reservedCPUs    *cpuset.CPUSet
	podUID          string
	options         map[string]string
	containerName   string
	stAssignments   state.ContainerCPUAssignments
	stDefaultCPUSet cpuset.CPUSet
	pod             *v1.Pod
	topologyHint    *topologymanager.TopologyHint
	expErr          error
	expCPUAlloc     bool
	expCSet         cpuset.CPUSet
}

// this is not a real Clone() - hence Pseudo- - because we don't clone some
// objects which are accessed read-only
func (spt staticPolicyTest) PseudoClone() staticPolicyTest {
	return staticPolicyTest{
		description:     spt.description,
		topo:            spt.topo, // accessed in read-only
		numReservedCPUs: spt.numReservedCPUs,
		podUID:          spt.podUID,
		options:         spt.options, // accessed in read-only
		containerName:   spt.containerName,
		stAssignments:   spt.stAssignments.Clone(),
		stDefaultCPUSet: spt.stDefaultCPUSet.Clone(),
		pod:             spt.pod, // accessed in read-only
		expErr:          spt.expErr,
		expCPUAlloc:     spt.expCPUAlloc,
		expCSet:         spt.expCSet.Clone(),
	}
}

func TestStaticPolicyName(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	policy, err := NewStaticPolicy(logger, topoSingleSocketHT, 1, cpuset.New(), topologymanager.NewFakeManager(), nil)
	if err != nil {
		t.Fatalf("NewStaticPolicy() failed: %v", err)
	}

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
					"0": cpuset.New(0),
				},
			},
			stDefaultCPUSet: cpuset.New(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expCSet:         cpuset.New(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "empty cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(),
			expCSet:         cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "reserved cores 0 & 6 are not present in available cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1),
			expErr:          fmt.Errorf("not all reserved cpus: \"0,6\" are present in defaultCpuSet: \"0-1\""),
		},
		{
			description:     "some of reserved cores are present in available cpuset (StrictCPUReservationOption)",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			options:         map[string]string{StrictCPUReservationOption: "true"},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1),
			expErr:          fmt.Errorf("some of strictly reserved cpus: \"0\" are present in defaultCpuSet: \"0-1\""),
		},
		{
			description: "assigned core 2 is still present in available cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.New(0, 1, 2),
				},
			},
			stDefaultCPUSet: cpuset.New(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expErr:          fmt.Errorf("pod: fakePod, container: 0 cpuset: \"0-2\" overlaps with default cpuset \"2-11\""),
		},
		{
			description: "assigned core 2 is still present in available cpuset (StrictCPUReservationOption)",
			topo:        topoDualSocketHT,
			options:     map[string]string{StrictCPUReservationOption: "true"},
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.New(0, 1, 2),
				},
			},
			stDefaultCPUSet: cpuset.New(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expErr:          fmt.Errorf("pod: fakePod, container: 0 cpuset: \"0-2\" overlaps with default cpuset \"2-11\""),
		},
		{
			description: "core 12 is not present in topology but is in state cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.New(0, 1, 2),
					"1": cpuset.New(3, 4),
				},
			},
			stDefaultCPUSet: cpuset.New(5, 6, 7, 8, 9, 10, 11, 12),
			expErr:          fmt.Errorf("current set of available CPUs \"0-11\" doesn't match with CPUs in state \"0-12\""),
		},
		{
			description: "core 11 is present in topology but is not in state cpuset",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.New(0, 1, 2),
					"1": cpuset.New(3, 4),
				},
			},
			stDefaultCPUSet: cpuset.New(5, 6, 7, 8, 9, 10),
			expErr:          fmt.Errorf("current set of available CPUs \"0-11\" doesn't match with CPUs in state \"0-10\""),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)
			p, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(), testCase.options)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}
			policy := p.(*staticPolicy)
			st := &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}
			err = policy.Start(logger, st)
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
	var largeTopoCPUids []int
	var largeTopoSock0CPUids []int
	var largeTopoSock1CPUids []int
	largeTopo := *topoQuadSocketFourWayHT
	for cpuid, val := range largeTopo.CPUDetails {
		largeTopoCPUids = append(largeTopoCPUids, cpuid)
		if val.SocketID == 0 {
			largeTopoSock0CPUids = append(largeTopoSock0CPUids, cpuid)
		} else if val.SocketID == 1 {
			largeTopoSock1CPUids = append(largeTopoSock1CPUids, cpuid)
		}
	}
	largeTopoCPUSet := cpuset.New(largeTopoCPUids...)
	largeTopoSock0CPUSet := cpuset.New(largeTopoSock0CPUids...)
	largeTopoSock1CPUSet := cpuset.New(largeTopoSock1CPUids...)

	// these are the cases which must behave the same regardless the policy options.
	// So we will permutate the options to ensure this holds true.

	optionsInsensitiveTestCases := []staticPolicyTest{
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectAllocOneCore",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "2000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 5),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocOneSocket",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(2),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "6000m", "6000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 3, 5, 7, 9, 11),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocThreeCores",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(1, 5),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "6000m", "6000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(2, 3, 4, 8, 9, 10),
		},
		{
			description:     "GuPodMultipleCores, DualSocketNoHT, ExpectAllocOneSocket",
			topo:            topoDualSocketNoHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(4, 5, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketNoHT, ExpectAllocFourCores",
			topo:            topoDualSocketNoHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(4, 5),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 3, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, ExpectAllocOneSocketOneCore",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(2),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer3", "8000m", "8000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 3, 4, 5, 7, 9, 10, 11),
		},
		{
			description:     "NonGuPod, SingleSocketHT, NoAlloc",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer1", "1000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodNonIntegerCore, SingleSocketHT, NoAlloc",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer4", "977m", "977m"),
			expErr:          nil,
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			// All the CPUs from Socket 0 are available. Some CPUs from each
			// Socket have been already assigned.
			// Expect all CPUs from Socket 0.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocSock0",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(3, 11, 4, 5, 6, 7),
				},
			},
			stDefaultCPUSet: largeTopoCPUSet.Difference(cpuset.New(3, 11, 4, 5, 6, 7)),
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
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.New(1, 25, 13, 38, 2, 9, 11, 35, 23, 48, 12, 51,
						53, 173, 113, 233, 54, 61)),
				},
			},
			stDefaultCPUSet: cpuset.New(1, 25, 13, 38, 2, 9, 11, 35, 23, 48, 12, 51, 53, 173, 113, 233, 54, 61),
			pod:             makePod("fakePod", "fakeCcontainer5", "12000m", "12000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 25, 13, 38, 11, 35, 23, 48, 53, 173, 113, 233),
		},
		{
			// All CPUs from Socket 1, 1 full core and some partial cores are available.
			// Expect all CPUs from Socket 1 and the hyper-threads from the full core.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocAllSock1+FullCore",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(largeTopoSock1CPUSet.Union(cpuset.New(10, 34, 22, 47, 53,
						173, 61, 181, 108, 228, 115, 235))),
				},
			},
			stDefaultCPUSet: largeTopoSock1CPUSet.Union(cpuset.New(10, 34, 22, 47, 53, 173, 61, 181, 108, 228,
				115, 235)),
			pod:         makePod("fakePod", "fakeContainer5", "76000m", "76000m"),
			expErr:      nil,
			expCPUAlloc: true,
			expCSet:     largeTopoSock1CPUSet.Union(cpuset.New(10, 34, 22, 47)),
		},
	}

	// testcases for the default behaviour of the policy.
	defaultOptionsTestCases := []staticPolicyTest{
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "1000m", "1000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(4), // expect sibling of partial core
		},
		{
			// Only partial cores are available in the entire system.
			// Expect allocation of all the CPUs from the partial cores.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocCPUs",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.New(10, 11, 53, 37, 55, 67, 52)),
				},
			},
			stDefaultCPUSet: cpuset.New(10, 11, 53, 67, 52),
			pod:             makePod("fakePod", "fakeContainer5", "5000m", "5000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(10, 11, 53, 67, 52),
		},
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request: requested=8, available=7"),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectSameAllocation",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(2, 3, 6, 7),
		},
		{
			description:     "GuPodMultipleCores, DualSocketHT, NoAllocExpectError",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(1, 2, 3),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer5", "10000m", "10000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request: requested=10, available=8"),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, NoAllocExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(1, 2, 3, 4, 5, 6),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 7),
			pod:             makePod("fakePod", "fakeContainer5", "2000m", "2000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request: requested=2, available=1"),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			// Only 7 CPUs are available.
			// Pod requests 76 cores.
			// Error is expected since available CPUs are less than the request.
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, NoAlloc",
			topo:        topoQuadSocketFourWayHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": largeTopoCPUSet.Difference(cpuset.New(10, 11, 53, 37, 55, 67, 52)),
				},
			},
			stDefaultCPUSet: cpuset.New(10, 11, 53, 37, 55, 67, 52),
			pod:             makePod("fakePod", "fakeContainer5", "76000m", "76000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request: requested=76, available=7"),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
	}

	// testcases for the FullPCPUsOnlyOption
	smtalignOptionTestCases := []staticPolicyTest{
		{
			description: "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "1000m", "1000m"),
			expErr:          SMTAlignmentError{RequestedCPUs: 1, CpusPerCore: 2},
			expCPUAlloc:     false,
			expCSet:         cpuset.New(), // reject allocation of sibling of partial core
		},
		{
			// test SMT-level != 2 - which is the default on x86_64
			description: "GuPodMultipleCores, topoQuadSocketFourWayHT, ExpectAllocOneCPUs",
			topo:        topoQuadSocketFourWayHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 8,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: largeTopoCPUSet,
			pod:             makePod("fakePod", "fakeContainer15", "15000m", "15000m"),
			expErr:          SMTAlignmentError{RequestedCPUs: 15, CpusPerCore: 4},
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description: "GuPodManyCores, topoDualSocketHT, ExpectDoNotAllocPartialCPU",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 2,
			reservedCPUs:    newCPUSetPtr(1, 6),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainerBug113537_1", "10000m", "10000m"),
			expErr:          SMTAlignmentError{RequestedCPUs: 10, CpusPerCore: 2, AvailablePhysicalCPUs: 8, CausedByPhysicalCPUs: true},
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description: "GuPodManyCores, topoDualSocketHT, AutoReserve, ExpectAllocAllCPUs",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 2,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainerBug113537_2", "10000m", "10000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
		},
		{
			description: "GuPodManyCores, topoDualSocketHT, ExpectAllocAllCPUs",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 2,
			reservedCPUs:    newCPUSetPtr(0, 6),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainerBug113537_2", "10000m", "10000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
		},
	}
	alignBySocketOptionTestCases := []staticPolicyTest{
		{
			description: "Align by socket: true, cpu's within same socket of numa in hint are part of allocation",
			topo:        topoDualSocketMultiNumaPerSocketHT,
			options: map[string]string{
				AlignBySocketOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(2, 11, 21, 22),
			pod:             makePod("fakePod", "fakeContainer2", "2000m", "2000m"),
			topologyHint:    &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 2), Preferred: true},
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(2, 11),
		},
		{
			description: "Align by socket: false, cpu's are taken strictly from NUMA nodes in hint",
			topo:        topoDualSocketMultiNumaPerSocketHT,
			options: map[string]string{
				AlignBySocketOption: "false",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(2, 11, 21, 22),
			pod:             makePod("fakePod", "fakeContainer2", "2000m", "2000m"),
			topologyHint:    &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0, 2), Preferred: true},
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(2, 21),
		},
	}

	for _, testCase := range optionsInsensitiveTestCases {
		for _, options := range []map[string]string{
			nil,
			{
				FullPCPUsOnlyOption: "true",
			},
		} {
			tCase := testCase.PseudoClone()
			tCase.description = fmt.Sprintf("options=%v %s", options, testCase.description)
			tCase.options = options
			runStaticPolicyTestCase(t, tCase)
		}
	}

	for _, testCase := range defaultOptionsTestCases {
		runStaticPolicyTestCase(t, testCase)
	}
	for _, testCase := range smtalignOptionTestCases {
		runStaticPolicyTestCase(t, testCase)
	}
	for _, testCase := range alignBySocketOptionTestCases {
		runStaticPolicyTestCaseWithFeatureGate(t, testCase)
	}
}

func runStaticPolicyTestCase(t *testing.T, testCase staticPolicyTest) {
	tm := topologymanager.NewFakeManager()
	if testCase.topologyHint != nil {
		tm = topologymanager.NewFakeManagerWithHint(testCase.topologyHint)
	}
	cpus := cpuset.New()
	if testCase.reservedCPUs != nil {
		cpus = testCase.reservedCPUs.Clone()
	}
	logger, _ := ktesting.NewTestContext(t)
	policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpus, tm, testCase.options)
	if err != nil {
		t.Fatalf("NewStaticPolicy() failed: %v", err)
	}

	st := &mockState{
		assignments:   testCase.stAssignments,
		defaultCPUSet: testCase.stDefaultCPUSet,
	}

	container := &testCase.pod.Spec.Containers[0]
	err = policy.Allocate(logger, st, testCase.pod, container)
	if !reflect.DeepEqual(err, testCase.expErr) {
		t.Errorf("StaticPolicy Allocate() error (%v). expected add error: %q but got: %q",
			testCase.description, testCase.expErr, err)
	}

	if testCase.expCPUAlloc {
		cset, found := st.assignments[string(testCase.pod.UID)][container.Name]
		if !found {
			t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
				testCase.description, container.Name, st.assignments)
		}

		if !cset.Equals(testCase.expCSet) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s but got %s",
				testCase.description, testCase.expCSet, cset)
		}

		if !cset.Intersection(st.defaultCPUSet).IsEmpty() {
			t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s to be disoint from the shared cpuset %s",
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

func runStaticPolicyTestCaseWithFeatureGate(t *testing.T, testCase staticPolicyTest) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)
	runStaticPolicyTestCase(t, testCase)
}

func TestStaticPolicyReuseCPUs(t *testing.T) {
	testCases := []struct {
		staticPolicyTest
		expCSetAfterAlloc  cpuset.CPUSet
		expCSetAfterRemove cpuset.CPUSet
	}{
		{
			staticPolicyTest: staticPolicyTest{
				description: "SingleSocketHT, DeAllocOneInitContainer",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPod(
					[]struct{ request, limit string }{
						{"4000m", "4000m"}}, // 0, 1, 4, 5
					[]struct{ request, limit string }{
						{"2000m", "2000m"}}), // 0, 4
				containerName:   "initContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expCSetAfterAlloc:  cpuset.New(2, 3, 6, 7),
			expCSetAfterRemove: cpuset.New(1, 2, 3, 5, 6, 7),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}
		pod := testCase.pod

		// allocate
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
			_ = policy.Allocate(logger, st, pod, &container)
		}
		if !st.defaultCPUSet.Equals(testCase.expCSetAfterAlloc) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected default cpuset %s but got %s",
				testCase.description, testCase.expCSetAfterAlloc, st.defaultCPUSet)
		}

		// remove
		_ = policy.RemoveContainer(logger, st, string(pod.UID), testCase.containerName)

		if !st.defaultCPUSet.Equals(testCase.expCSetAfterRemove) {
			t.Errorf("StaticPolicy RemoveContainer() error (%v). expected default cpuset %sv but got %s",
				testCase.description, testCase.expCSetAfterRemove, st.defaultCPUSet)
		}
		if _, found := st.assignments[string(pod.UID)][testCase.containerName]; found {
			t.Errorf("StaticPolicy RemoveContainer() error (%v). expected (pod %v, container %v) not be in assignments %v",
				testCase.description, testCase.podUID, testCase.containerName, st.assignments)
		}
	}
}

func TestStaticPolicyDoNotReuseCPUs(t *testing.T) {
	testCases := []struct {
		staticPolicyTest
		expCSetAfterAlloc cpuset.CPUSet
	}{
		{
			staticPolicyTest: staticPolicyTest{
				description: "SingleSocketHT, Don't reuse CPUs of a restartable init container",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicyAlways}}, // 0, 1, 4, 5
					[]*containerOptions{
						{request: "2000m", limit: "2000m"}}), // 2, 6
				stAssignments:   state.ContainerCPUAssignments{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expCSetAfterAlloc: cpuset.New(3, 7),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}
		pod := testCase.pod

		// allocate
		for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
			err := policy.Allocate(logger, st, pod, &container)
			if err != nil {
				t.Errorf("StaticPolicy Allocate() error (%v). expected no error but got %v",
					testCase.description, err)
			}
		}
		if !st.defaultCPUSet.Equals(testCase.expCSetAfterAlloc) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected default cpuset %s but got %s",
				testCase.description, testCase.expCSetAfterAlloc, st.defaultCPUSet)
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
					"fakeContainer1": cpuset.New(1, 2, 3),
				},
			},
			stDefaultCPUSet: cpuset.New(4, 5, 6, 7),
			expCSet:         cpuset.New(1, 2, 3, 4, 5, 6, 7),
		},
		{
			description:   "SingleSocketHT, DeAllocOneContainer, BeginEmpty",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer1",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.New(1, 2, 3),
					"fakeContainer2": cpuset.New(4, 5, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.New(),
			expCSet:         cpuset.New(1, 2, 3),
		},
		{
			description:   "SingleSocketHT, DeAllocTwoContainer",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer1",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.New(1, 3, 5),
					"fakeContainer2": cpuset.New(2, 4),
				},
			},
			stDefaultCPUSet: cpuset.New(6, 7),
			expCSet:         cpuset.New(1, 3, 5, 6, 7),
		},
		{
			description:   "SingleSocketHT, NoDeAlloc",
			topo:          topoSingleSocketHT,
			podUID:        "fakePod",
			containerName: "fakeContainer2",
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer1": cpuset.New(1, 3, 5),
				},
			},
			stDefaultCPUSet: cpuset.New(2, 4, 6, 7),
			expCSet:         cpuset.New(2, 4, 6, 7),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		_ = policy.RemoveContainer(logger, st, testCase.podUID, testCase.containerName)

		if !st.defaultCPUSet.Equals(testCase.expCSet) {
			t.Errorf("StaticPolicy RemoveContainer() error (%v). expected default cpuset %s but got %s",
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
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask:      nil,
			expCSet:         cpuset.New(0, 6),
		},
		{
			description:     "Request 2 CPUs, BitMask on Socket 0",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(0)
				return mask
			}(),
			expCSet: cpuset.New(0, 6),
		},
		{
			description:     "Request 2 CPUs, BitMask on Socket 1",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(1)
				return mask
			}(),
			expCSet: cpuset.New(1, 7),
		},
		{
			description:     "Request 8 CPUs, BitMask on Socket 0",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    8,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(0)
				return mask
			}(),
			expCSet: cpuset.New(0, 6, 2, 8, 4, 10, 1, 7),
		},
		{
			description:     "Request 8 CPUs, BitMask on Socket 1",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    8,
			socketMask: func() bitmask.BitMask {
				mask, _ := bitmask.NewBitMask(1)
				return mask
			}(),
			expCSet: cpuset.New(1, 7, 3, 9, 5, 11, 0, 6),
		},
	}
	for _, tc := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		p, err := NewStaticPolicy(logger, tc.topo, 0, cpuset.New(), topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}
		policy := p.(*staticPolicy)
		st := &mockState{
			assignments:   tc.stAssignments,
			defaultCPUSet: tc.stDefaultCPUSet,
		}
		err = policy.Start(logger, st)
		if err != nil {
			t.Errorf("StaticPolicy Start() error (%v)", err)
			continue
		}

		cpuAlloc, err := policy.allocateCPUs(klog.Background(), st, tc.numRequested, tc.socketMask, cpuset.New())
		if err != nil {
			t.Errorf("StaticPolicy allocateCPUs() error (%v). expected CPUSet %v not error %v",
				tc.description, tc.expCSet, err)
			continue
		}

		if !tc.expCSet.Equals(cpuAlloc.CPUs) {
			t.Errorf("StaticPolicy allocateCPUs() error (%v). expected CPUSet %v but got %v",
				tc.description, tc.expCSet, cpuAlloc.CPUs)
		}
	}
}

// above test cases are without kubelet --reserved-cpus cmd option
// the following tests are with --reserved-cpus configured
type staticPolicyTestWithResvList struct {
	description      string
	topo             *topology.CPUTopology
	numReservedCPUs  int
	reserved         cpuset.CPUSet
	cpuPolicyOptions map[string]string
	stAssignments    state.ContainerCPUAssignments
	stDefaultCPUSet  cpuset.CPUSet
	pod              *v1.Pod
	expErr           error
	expNewErr        error
	expCPUAlloc      bool
	expCSet          cpuset.CPUSet
	expUncoreCache   cpuset.CPUSet // represents the expected UncoreCacheIDs
}

func TestStaticPolicyStartWithResvList(t *testing.T) {
	testCases := []staticPolicyTestWithResvList{
		{
			description:     "empty cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(),
			expCSet:         cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:      "empty cpuset with StrictCPUReservationOption enabled",
			topo:             topoDualSocketHT,
			numReservedCPUs:  2,
			reserved:         cpuset.New(0, 1),
			cpuPolicyOptions: map[string]string{StrictCPUReservationOption: "true"},
			stAssignments:    state.ContainerCPUAssignments{},
			stDefaultCPUSet:  cpuset.New(),
			expCSet:          cpuset.New(2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "reserved cores 0 & 1 are not present in available cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(2, 3, 4, 5),
			expErr:          fmt.Errorf("not all reserved cpus: \"0-1\" are present in defaultCpuSet: \"2-5\""),
		},
		{
			description:      "reserved cores 0 & 1 are present in available cpuset with StrictCPUReservationOption enabled",
			topo:             topoDualSocketHT,
			numReservedCPUs:  2,
			reserved:         cpuset.New(0, 1),
			cpuPolicyOptions: map[string]string{StrictCPUReservationOption: "true"},
			stAssignments:    state.ContainerCPUAssignments{},
			stDefaultCPUSet:  cpuset.New(0, 1, 2, 3, 4, 5),
			expErr:           fmt.Errorf("some of strictly reserved cpus: \"0-1\" are present in defaultCpuSet: \"0-5\""),
		},
		{
			description:     "inconsistency between numReservedCPUs and reserved",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reserved:        cpuset.New(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expNewErr:       fmt.Errorf("[cpumanager] unable to reserve the required amount of CPUs (size of 0-1 did not equal 1)"),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)
			p, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager(), testCase.cpuPolicyOptions)
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
			err = policy.Start(logger, st)
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
			reserved:        cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          fmt.Errorf("not enough cpus available to satisfy request: requested=8, available=7"),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "1000m", "1000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(4), // expect sibling of partial core
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectAllocOneCore",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(2, 3, 6, 7),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 4, 5),
			pod:             makePod("fakePod", "fakeContainer3", "2000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(4, 5),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager(), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}

		st := &mockState{
			assignments:   testCase.stAssignments,
			defaultCPUSet: testCase.stDefaultCPUSet,
		}

		container := &testCase.pod.Spec.Containers[0]
		err = policy.Allocate(logger, st, testCase.pod, container)
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

			if !cset.Equals(testCase.expCSet) {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s but got %s",
					testCase.description, testCase.expCSet, cset)
			}

			if !cset.Intersection(st.defaultCPUSet).IsEmpty() {
				t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s to be disoint from the shared cpuset %s",
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

func WithPodUID(pod *v1.Pod, podUID string) *v1.Pod {
	ret := pod.DeepCopy()
	ret.UID = types.UID(podUID)
	return ret
}

func TestStaticPolicyAddWithUncoreAlignment(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)

	testCases := []staticPolicyTestWithResvList{
		{
			description:     "GuPodSingleContainerSaturating, DualSocketHTUncore, ExpectAllocOneUncore, FullUncoreAvail",
			topo:            topoDualSocketSingleNumaPerSocketSMTUncore,
			numReservedCPUs: 8,
			reserved:        cpuset.New(0, 1, 96, 97, 192, 193, 288, 289), // note 4 cpus taken from uncore 0, 4 from uncore 12
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// remove partially used uncores from the available CPUs to simulate fully clean slate
			stDefaultCPUSet: topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(0),
				).Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(12),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"16000m", "16000m"}, // CpusPerUncore=16 with this topology
					},
				),
				"with-app-container-saturating",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			description:     "GuPodMainAndSidecarContainer, DualSocketHTUncore, ExpectAllocOneUncore, FullUncoreAvail",
			topo:            topoDualSocketSingleNumaPerSocketSMTUncore,
			numReservedCPUs: 8,
			reserved:        cpuset.New(0, 1, 96, 97, 192, 193, 288, 289), // note 4 cpus taken from uncore 0, 4 from uncore 12
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// remove partially used uncores from the available CPUs to simulate fully clean slate
			stDefaultCPUSet: topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(0),
				).Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(12),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"},
						{"2000m", "2000m"},
					},
				),
				"with-app-container-and-sidecar",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			description:     "GuPodSidecarAndMainContainer, DualSocketHTUncore, ExpectAllocOneUncore, FullUncoreAvail",
			topo:            topoDualSocketSingleNumaPerSocketSMTUncore,
			numReservedCPUs: 8,
			reserved:        cpuset.New(0, 1, 96, 97, 192, 193, 288, 289), // note 4 cpus taken from uncore 0, 4 from uncore 12
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// remove partially used uncores from the available CPUs to simulate fully clean slate
			stDefaultCPUSet: topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(0),
				).Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(12),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"2000m", "2000m"},
						{"12000m", "12000m"},
					},
				),
				"with-sidecar-and-app-container",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			description:     "GuPodMainAndManySidecarContainer, DualSocketHTUncore, ExpectAllocOneUncore, FullUncoreAvail",
			topo:            topoDualSocketSingleNumaPerSocketSMTUncore,
			numReservedCPUs: 8,
			reserved:        cpuset.New(0, 1, 96, 97, 192, 193, 288, 289), // note 4 cpus taken from uncore 0, 4 from uncore 12
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// remove partially used uncores from the available CPUs to simulate fully clean slate
			stDefaultCPUSet: topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(0),
				).Union(
					topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUsInUncoreCaches(12),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"10000m", "10000m"},
						{"2000m", "2000m"},
						{"2000m", "2000m"},
						{"2000m", "2000m"},
					},
				),
				"with-app-container-and-multi-sidecar",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			description:     "GuPodMainAndSidecarContainer, DualSocketHTUncore, ExpectAllocTwoUncore",
			topo:            topoDualSocketSingleNumaPerSocketSMTUncore,
			numReservedCPUs: 8,
			reserved:        cpuset.New(0, 1, 96, 97, 192, 193, 288, 289), // note 4 cpus taken from uncore 0, 4 from uncore 12
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoDualSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"},
						{"2000m", "2000m"},
					},
				),
				"with-app-container-and-sidecar",
			),
			expUncoreCache: cpuset.New(0, 1), // expected CPU alignment to UncoreCacheIDs 0-1
		},
		{
			description:     "GuPodSingleContainer, SingleSocketSMTSmallUncore, ExpectAllocOneUncore",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore,
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 64, 65), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"8000m", "8000m"},
					},
				),
				"with-app-container-saturating",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			// Best-effort policy allows larger containers to be scheduled using a packed method
			description:     "GuPodSingleContainer, SingleSocketSMTSmallUncore, ExpectAllocTwoUncore",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 64, 65), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// Uncore 1 fully allocated
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs().Difference(
				topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUsInUncoreCaches(1),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"}, // larger than topology's uncore cache
					},
				),
				"with-app-container-saturating",
			),
			expUncoreCache: cpuset.New(0, 2),
		},
		{
			// Best-effort policy allows larger containers to be scheduled using a packed method
			description:     "GuPodSingleContainer, SingleSocketNoSMTSmallUncore, FragmentedUncore, ExpectAllocThreeUncore",
			topo:            topoSingleSocketSingleNumaPerSocketNoSMTSmallUncore, // 4 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 2, 3), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// Uncore 2, 3, and 5 fully allocated
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketNoSMTSmallUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					topoSingleSocketSingleNumaPerSocketNoSMTSmallUncore.CPUDetails.CPUsInUncoreCaches(2),
				).Union(
					topoSingleSocketSingleNumaPerSocketNoSMTSmallUncore.CPUDetails.CPUsInUncoreCaches(3),
				).Union(
					topoSingleSocketSingleNumaPerSocketNoSMTSmallUncore.CPUDetails.CPUsInUncoreCaches(5),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"}, // 3 uncore cache's worth of CPUs
					},
				),
				"with-app-container-saturating",
			),
			expUncoreCache: cpuset.New(1, 4, 6),
		},
		{
			// Uncore cache alignment following a packed methodology
			description:     "GuPodMultiContainer, DualSocketSMTUncore, FragmentedUncore, ExpectAllocOneUncore",
			topo:            topoSmallDualSocketSingleNumaPerSocketNoSMTUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 32, 33), // note 2 cpus taken from uncore 0, 2 from uncore 4
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// uncore 1 fully allocated
			stDefaultCPUSet: topoSmallDualSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUs().Difference(
				topoSmallDualSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUsInUncoreCaches(1),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"4000m", "4000m"},
						{"2000m", "2000m"},
					},
				),
				"with-multiple-container",
			),
			expUncoreCache: cpuset.New(0),
		},
		{
			// Uncore cache alignment following a packed methodology
			description:     "GuPodMultiContainer, DualSocketSMTUncore, FragmentedUncore, ExpectAllocTwoUncore",
			topo:            topoSmallDualSocketSingleNumaPerSocketNoSMTUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 32, 33), // note 2 cpus taken from uncore 0, 2 from uncore 4
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// uncore 1 fully allocated
			stDefaultCPUSet: topoSmallDualSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUs().Difference(
				topoSmallDualSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUsInUncoreCaches(1),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"4000m", "4000m"},
						{"4000m", "4000m"},
					},
				),
				"with-multiple-container",
			),
			expUncoreCache: cpuset.New(0, 2),
		},
		{
			// CPU assignments able to fit on partially available uncore cache
			description:     "GuPodMultiContainer, LargeSingleSocketSMTUncore, PartialUncoreFit, ExpectAllocTwoUncore",
			topo:            topoLargeSingleSocketSingleNumaPerSocketSMTUncore, // 16 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 128, 129), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// 4 cpus allocated from uncore 1
			stDefaultCPUSet: topoLargeSingleSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New(8, 9, 136, 137),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"},
						{"12000m", "12000m"},
					},
				),
				"with-multiple-container",
			),
			expUncoreCache: cpuset.New(0, 1),
		},
		{
			// CPU assignments unable to fit on partially available uncore cache
			description:     "GuPodMultiContainer, LargeSingleSocketSMTUncore, PartialUncoreNoFit, ExpectAllocTwoUncore",
			topo:            topoLargeSingleSocketSingleNumaPerSocketSMTUncore, // 16 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 128, 129), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// 4 cpus allocated from uncore 1
			stDefaultCPUSet: topoLargeSingleSocketSingleNumaPerSocketSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New(8, 9, 136, 137),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"14000m", "14000m"},
						{"14000m", "14000m"},
					},
				),
				"with-multiple-container",
			),
			expUncoreCache: cpuset.New(2, 3),
		},
		{
			// Full NUMA allocation on split-cache architecture with NPS=2
			description:     "GuPodLargeSingleContainer, DualSocketNoSMTUncore, FullNUMAsAvail, ExpectAllocFullNUMA",
			topo:            topoDualSocketMultiNumaPerSocketUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 2, 3), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoDualSocketMultiNumaPerSocketUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"48000m", "48000m"}, // NUMA's worth of CPUs
					},
				),
				"with-large-single-container",
			),
			expUncoreCache: cpuset.New(6, 7, 8, 9, 10, 11), // uncore caches of NUMA Node 1
		},
		{
			// PreferAlignByUnCoreCacheOption will not impact monolithic x86 architectures
			description:     "GuPodSingleContainer, MonoUncoreCacheHT, ExpectAllocCPUSet",
			topo:            topoDualSocketSubNumaPerSocketHTMonolithicUncore, // Uncore cache CPUs = Socket CPUs
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 120, 121), // note 4 cpus taken from first 2 cores
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoDualSocketSubNumaPerSocketHTMonolithicUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"6000m", "6000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc:    true,
			expCSet:        cpuset.New(2, 3, 4, 122, 123, 124),
			expUncoreCache: cpuset.New(0),
		},
		{
			// PreferAlignByUnCoreCacheOption on fragmented monolithic cache x86 architectures
			description:     "GuPodSingleContainer, MonoUncoreCacheHT, ExpectAllocCPUSet",
			topo:            topoSingleSocketSingleNumaPerSocketPCoreHTMonolithicUncore, // Uncore cache CPUs = Socket CPUs
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// CPUs 4-7 allocated
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketPCoreHTMonolithicUncore.CPUDetails.CPUs().Difference(
				cpuset.New(4, 5, 6, 7),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"6000m", "6000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc:    true,
			expCSet:        cpuset.New(2, 3, 8, 9, 16, 17), // identical to default packed assignment
			expUncoreCache: cpuset.New(0),
		},
		{
			// Compatibility with ARM-based split cache architectures
			description:     "GuPodSingleContainer, LargeSingleSocketUncore, ExpectAllocOneUncore",
			topo:            topoLargeSingleSocketSingleNumaPerSocketUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 2, 3), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoLargeSingleSocketSingleNumaPerSocketUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"8000m", "8000m"},
					},
				),
				"with-single-container",
			),
			expUncoreCache: cpuset.New(1),
		},
		{
			// PreferAlignByUnCoreCacheOption on fragmented monolithic cache ARM architectures
			description:     "GuPodSingleContainer, MonoUncoreCacheHT, ExpectFragmentedAllocCPUSet",
			topo:            topoSingleSocketSingleNumaPerSocketUncore, // Uncore cache CPUs = Socket CPUs
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// CPUs 6-9, 12-15, 18-19 allocated
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					cpuset.New(6, 7, 8, 9),
				).Union(
					cpuset.New(12, 13, 14, 15),
				).Union(
					cpuset.New(18, 19),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"12000m", "12000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc:    true,
			expCSet:        cpuset.New(2, 3, 4, 5, 10, 11, 16, 17, 20, 21, 22, 23), // identical to default packed assignment
			expUncoreCache: cpuset.New(0),
		},
		{
			// Best-effort policy can result in multiple uncore caches
			// Every uncore cache is partially allocated
			description:     "GuPodSingleContainer, SingleSocketUncore, PartialUncore, ExpectBestEffortAllocTwoUncore",
			topo:            topoSmallSingleSocketSingleNumaPerSocketNoSMTUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 2, 3), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments: state.ContainerCPUAssignments{},
			// Every uncore has partially allocated 4 CPUs
			stDefaultCPUSet: topoSmallSingleSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUs().Difference(
				cpuset.New().Union(
					cpuset.New(8, 9, 10, 11),
				).Union(
					cpuset.New(16, 17, 18, 19),
				).Union(
					cpuset.New(24, 25, 26, 27),
				),
			),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"8000m", "8000m"}, // full uncore cache worth of cpus
					},
				),
				"with-single-container",
			),
			expUncoreCache: cpuset.New(0, 1), // best-effort across uncore cache 0 and 1
		},
		{
			// odd integer cpu required on smt-disabled processor
			description:     "odd integer cpu required on smt-disabled",
			topo:            topoSmallSingleSocketSingleNumaPerSocketNoSMTUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 2, 3), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				FullPCPUsOnlyOption:            "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSmallSingleSocketSingleNumaPerSocketNoSMTUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"5000m", "5000m"}, // full uncore cache worth of cpus
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(8, 9, 10, 11, 12),
		},
		{
			// odd integer cpu requirement on smt-enabled
			description:     "odd integer required on smt-enabled",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore, // 8 cpus per uncore
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 64, 65), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"3000m", "3000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 3, 66),
		},
		{
			// odd integer cpu required on smt-enabled and odd integer free cpus available on uncore
			description:     "odd integer required on odd integer partial uncore",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore, // 8 cpus per uncore
			numReservedCPUs: 3,
			reserved:        cpuset.New(0, 1, 64), // note 3 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"3000m", "3000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 65, 66),
		},
		{
			// even integer requested on smt-enabled processor with odd integer available cpus on uncore
			// even integer cpu containers will not be placed on uncore caches with odd integer free cpus
			description:     "even integer required on odd integer partial uncore",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore, // 8 cpus per uncore
			numReservedCPUs: 3,
			reserved:        cpuset.New(0, 1, 64), // note 3 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"4000m", "4000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(4, 5, 68, 69),
		},
		{
			// large odd integer cpu required on smt-enabled
			description:     "large odd integer required on smt-enabled",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore, // 8 cpus per uncore
			numReservedCPUs: 3,
			reserved:        cpuset.New(0, 1, 64), // note 3 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"11000m", "11000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 65, 66, 4, 5, 6, 7, 68, 69, 70, 71), // full uncore 1 and partial uncore 0
		},
		{
			// odd integer cpu required on hyperthread-enabled and monolithic uncore cache
			description:     "odd integer required on HT monolithic uncore",
			topo:            topoDualSocketSubNumaPerSocketHTMonolithicUncore,
			numReservedCPUs: 3,
			reserved:        cpuset.New(0, 1, 120), // note 3 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"5000m", "5000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 3, 121, 122, 123),
		},
		{
			// even integer cpu required on hyperthread-enabled and monolithic uncore cache
			description:     "even integer required on HT monolithic uncore",
			topo:            topoDualSocketSubNumaPerSocketHTMonolithicUncore,
			numReservedCPUs: 3,
			reserved:        cpuset.New(0, 1, 120), // note 3 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"4000m", "4000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 3, 122, 123), // takeFullCores
		},
		{
			// test feature compatibility with strict-cpu-reservation on split cache architecture
			description:     "GuPodSingleContainer, SingleSocketSMTSmallUncore, StrictReserveCompatability",
			topo:            topoSingleSocketSingleNumaPerSocketSMTSmallUncore,
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 64, 65), // note 4 cpus taken from uncore 0
			cpuPolicyOptions: map[string]string{
				StrictCPUReservationOption:     "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoSingleSocketSingleNumaPerSocketSMTSmallUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"2000m", "2000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 66), // should avoid reserved cpuset
		},
		{
			// test feature compatibility with strict-cpu-reservation on monolithic uncore architecture
			description:     "GuPodSingleContainer, DualSocketHTMonoUncore, StrictReserveCompatability",
			topo:            topoDualSocketSubNumaPerSocketHTMonolithicUncore,
			numReservedCPUs: 4,
			reserved:        cpuset.New(0, 1, 120, 121), // first two cores reserved
			cpuPolicyOptions: map[string]string{
				StrictCPUReservationOption:     "true",
				PreferAlignByUnCoreCacheOption: "true",
			},
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: topoDualSocketSubNumaPerSocketHTMonolithicUncore.CPUDetails.CPUs(),
			pod: WithPodUID(
				makeMultiContainerPod(
					[]struct{ request, limit string }{}, // init container
					[]struct{ request, limit string }{ // app container
						{"4000m", "4000m"},
					},
				),
				"with-single-container",
			),
			expCPUAlloc: true,
			expCSet:     cpuset.New(2, 3, 122, 123), // packed assignment avoid reserved cpuset
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager(), testCase.cpuPolicyOptions)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed with %v", err)
			}

			st := &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet.Difference(testCase.reserved), // ensure the cpumanager invariant
			}

			for idx := range testCase.pod.Spec.Containers {
				container := &testCase.pod.Spec.Containers[idx]
				err := policy.Allocate(logger, st, testCase.pod, container)
				if err != nil {
					t.Fatalf("Allocate failed: pod=%q container=%q", testCase.pod.UID, container.Name)
				}
			}

			if testCase.expCPUAlloc {
				container := &testCase.pod.Spec.Containers[0]
				cset, found := st.assignments[string(testCase.pod.UID)][container.Name]
				if !found {
					t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
						testCase.description, container.Name, st.assignments)
				}
				if !testCase.expCSet.Equals(cset) {
					t.Errorf("StaticPolicy Allocate() error (%v). expected CPUSet %v but got %v",
						testCase.description, testCase.expCSet, cset)
				}
				return
			}

			uncoreCacheIDs, err := getPodUncoreCacheIDs(st, testCase.topo, testCase.pod)
			if err != nil {
				t.Fatalf("uncore cache check: %v", err.Error())
			}
			ids := cpuset.New(uncoreCacheIDs...)

			if !ids.Equals(testCase.expUncoreCache) {
				t.Errorf("StaticPolicy Allocate() error (%v). expected UncoreCacheIDs %v but got %v",
					testCase.description, testCase.expUncoreCache, ids)
			}
		})
	}
}

type staticPolicyOptionTestCase struct {
	description   string
	policyOptions map[string]string
	expectedError bool
	expectedValue StaticPolicyOptions
}

func TestStaticPolicyOptions(t *testing.T) {
	testCases := []staticPolicyOptionTestCase{
		{
			description:   "nil args",
			policyOptions: nil,
			expectedError: false,
			expectedValue: StaticPolicyOptions{},
		},
		{
			description:   "empty args",
			policyOptions: map[string]string{},
			expectedError: false,
			expectedValue: StaticPolicyOptions{},
		},
		{
			description: "bad single arg",
			policyOptions: map[string]string{
				"badValue1": "",
			},
			expectedError: true,
		},
		{
			description: "bad multiple arg",
			policyOptions: map[string]string{
				"badValue1": "",
				"badvalue2": "aaaa",
			},
			expectedError: true,
		},
		{
			description: "good arg",
			policyOptions: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			expectedError: false,
			expectedValue: StaticPolicyOptions{
				FullPhysicalCPUsOnly: true,
			},
		},
		{
			description: "good arg, bad value",
			policyOptions: map[string]string{
				FullPCPUsOnlyOption: "enabled!",
			},
			expectedError: true,
		},

		{
			description: "bad arg intermixed",
			policyOptions: map[string]string{
				FullPCPUsOnlyOption: "1",
				"badvalue2":         "lorem ipsum",
			},
			expectedError: true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			opts, err := NewStaticPolicyOptions(testCase.policyOptions)
			gotError := (err != nil)
			if gotError != testCase.expectedError {
				t.Fatalf("error with args %v expected error %v got %v: %v",
					testCase.policyOptions, testCase.expectedError, gotError, err)
			}

			if testCase.expectedError {
				return
			}

			if !reflect.DeepEqual(opts, testCase.expectedValue) {
				t.Fatalf("value mismatch with args %v expected value %v got %v",
					testCase.policyOptions, testCase.expectedValue, opts)
			}
		})
	}
}

type staticPolicyAllocatePodTest struct {
	description                     string
	topo                            *topology.CPUTopology
	numReservedCPUs                 int
	reservedCPUs                    cpuset.CPUSet
	options                         map[string]string
	stAssignments                   state.ContainerCPUAssignments
	stDefaultCPUSet                 cpuset.CPUSet
	pod                             *v1.Pod
	topologyHint                    topologymanager.TopologyHint
	expErr                          error
	expPodAssignments               state.ContainerCPUAssignments
	expDefaultCPUSet                cpuset.CPUSet
	podLevelResourcesEnabled        bool
	podLevelResourceManagersEnabled bool
	requiredMetrics                 requiredMetrics
}

type requiredMetrics struct {
	expTotalAllocs              int
	expTotalErrors              int
	expExclusiveAssignments     int
	expPodSharedPoolAssignments int
}

type containerSpec struct {
	name          string
	request       string
	limit         string
	restartPolicy *v1.ContainerRestartPolicy
}

func makePodWithContainersAndPodLevelResources(podName, podRequest, podLimit string, initContainers, appContainers []containerSpec) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
			UID:  types.UID(podName),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{},
			Containers:     []v1.Container{},
			RestartPolicy:  v1.RestartPolicyNever,
		},
	}

	for _, c := range initContainers {
		container := v1.Container{
			Name:  c.name,
			Image: "image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		}
		if c.request != "" {
			container.Resources.Requests[v1.ResourceCPU] = resource.MustParse(c.request)
		}
		if c.limit != "" {
			container.Resources.Limits[v1.ResourceCPU] = resource.MustParse(c.limit)
		}
		if c.restartPolicy != nil {
			container.RestartPolicy = c.restartPolicy
		}
		pod.Spec.InitContainers = append(pod.Spec.InitContainers, container)
	}

	for _, c := range appContainers {
		container := v1.Container{
			Name:  c.name,
			Image: "image",
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
		}
		if c.request != "" {
			container.Resources.Requests[v1.ResourceCPU] = resource.MustParse(c.request)
		}
		if c.limit != "" {
			container.Resources.Limits[v1.ResourceCPU] = resource.MustParse(c.limit)
		}
		pod.Spec.Containers = append(pod.Spec.Containers, container)
	}

	if podRequest != "" || podLimit != "" {
		pod.Spec.Resources = &v1.ResourceRequirements{
			Requests: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse("300Mi"),
			},
		}
		if podRequest != "" {
			pod.Spec.Resources.Requests[v1.ResourceCPU] = resource.MustParse(podRequest)
		}
		if podLimit != "" {
			pod.Spec.Resources.Limits[v1.ResourceCPU] = resource.MustParse(podLimit)
		}
	} else {
		pod.Spec.Resources = nil
	}
	return pod
}

func TestStaticPolicyAllocatePod(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	testCases := []staticPolicyAllocatePodTest{
		{
			description:     "should successfully allocate CPUs for a guaranteed pod with pod-level resources",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePodWithPodLevelResources("pod1", "2", "2", "container1", "1", "1"),
			topologyHint:    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:          nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"pod1": map[string]cpuset.CPUSet{
					"container1": cpuset.New(2),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 6, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              1,
				expExclusiveAssignments:     1,
				expPodSharedPoolAssignments: 0,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and guaranteed container, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePodWithPodLevelResources("gu-pod-level-resources", "2", "2", "gu-container", "1", "1"),
			topologyHint:    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:          nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-resources": map[string]cpuset.CPUSet{
					"gu-container": cpuset.New(2),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 6, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              1,
				expExclusiveAssignments:     1,
				expPodSharedPoolAssignments: 0,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and non-guaranteed container, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-resources", "1", "1", []containerSpec{}, []containerSpec{
				{name: "ngu-container"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-resources": map[string]cpuset.CPUSet{
					"ngu-container": cpuset.New(6),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              1,
				expExclusiveAssignments:     0,
				expPodSharedPoolAssignments: 1,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and mix of guaranteed and non-guaranteed containers, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-mix-ctn", "3", "3", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "gu-container-2", request: "1", limit: "1"},
				{name: "ngu-container"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-mix-ctn": {
					"gu-container-1": cpuset.New(6),
					"gu-container-2": cpuset.New(2),
					"ngu-container":  cpuset.New(8),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              3,
				expExclusiveAssignments:     2,
				expPodSharedPoolAssignments: 1,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and mix of guaranteed standard and init containers, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-gu-init-ctn", "2", "2", []containerSpec{
				{name: "gu-init-container-1", request: "1", limit: "1"},
				{name: "gu-init-container-2", request: "1", limit: "1"},
			}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "gu-container-2", request: "1", limit: "1"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-gu-init-ctn": {
					"gu-init-container-1": cpuset.New(2),
					"gu-init-container-2": cpuset.New(2),
					"gu-container-1":      cpuset.New(2),
					"gu-container-2":      cpuset.New(8),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 6, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              4,
				expExclusiveAssignments:     4,
				expPodSharedPoolAssignments: 0,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and mix of guaranteed standard and guaranteed restartable and non-guaranteed standard init containers, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-gu-init-ctn", "3", "3", []containerSpec{
				{name: "ngu-init-container-1"},
				{name: "gu-init-restartable-2", request: "1", limit: "1", restartPolicy: &containerRestartPolicyAlways},
				{name: "ngu-init-container-3"},
			}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "ngu-container-2"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-gu-init-ctn": {
					"ngu-init-container-1":  cpuset.New(2, 6, 8),
					"gu-init-restartable-2": cpuset.New(6),
					"ngu-init-container-3":  cpuset.New(2, 8),
					"gu-container-1":        cpuset.New(2),
					"ngu-container-2":       cpuset.New(8),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              5,
				expExclusiveAssignments:     2,
				expPodSharedPoolAssignments: 3,
			},
		},
		{
			description:     "scope: pod, should allocate exclusive CPUs to a guaranteed pod with pod-level resources and mix of guaranteed standard and non-guaranteed restartable init containers, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-gu-init-ctn", "3", "3", []containerSpec{
				{name: "ngu-init-container-1"},
				{name: "ngu-init-restartable-2", restartPolicy: &containerRestartPolicyAlways},
				{name: "ngu-init-container-3"},
			}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "ngu-container-2"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodAssignments: state.ContainerCPUAssignments{
				"gu-pod-level-gu-init-ctn": {
					"ngu-init-container-1":   cpuset.New(2, 6, 8),
					"ngu-init-restartable-2": cpuset.New(2, 8),
					"ngu-init-container-3":   cpuset.New(2, 6, 8),
					"gu-container-1":         cpuset.New(6),
					"ngu-container-2":        cpuset.New(2, 8),
				},
			},
			expDefaultCPUSet:                cpuset.New(0, 1, 3, 4, 5, 7, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalAllocs:              5,
				expExclusiveAssignments:     1,
				expPodSharedPoolAssignments: 4,
			},
		},
		{
			description:     "scope: pod, should reject a pod that would result in an empty pod shared pool",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-empty-pod-shared-pool", "2", "2", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "gu-container-2", request: "1", limit: "1"},
				{name: "ngu-container"},
			}),
			topologyHint:                    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:                          admission.NewEmptyPodSharedPoolError(fmt.Errorf("pod rejected, sum of exclusive container cpu requests equals pod budget, leaving no cpus for shared containers")),
			expPodAssignments:               state.ContainerCPUAssignments{},
			expDefaultCPUSet:                cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalErrors: 1,
			},
		},
		{
			description:     "scope: pod, should not allocate exclusive CPUs to a non-guaranteed pod with pod-level resources and guaranteed containers, PodLevelResourceManagers enabled",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("ngu-pod-level-empty-pod-shared-pool", "2", "1", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
			}),
			topologyHint:                    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:                          nil,
			expPodAssignments:               state.ContainerCPUAssignments{},
			expDefaultCPUSet:                cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			podLevelResourcesEnabled:        true,
			podLevelResourceManagersEnabled: true,
			requiredMetrics: requiredMetrics{
				expTotalErrors: 0,
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, testCase.podLevelResourcesEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, testCase.podLevelResourceManagersEnabled)

			metrics.Register()
			metrics.ResourceManagerAllocationsTotal.Reset()
			metrics.ResourceManagerAllocationErrorsTotal.Reset()
			metrics.ResourceManagerContainerAssignments.Reset()

			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reservedCPUs, topologymanager.NewFakeManager(), testCase.options)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			st := &mockState{
				assignments:   testCase.stAssignments,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}

			err = policy.AllocatePod(logger, st, testCase.pod)
			if testCase.expErr != nil {
				require.Error(t, err)

				errors, err := testutil.GetCounterMetricValue(metrics.ResourceManagerAllocationErrorsTotal.WithLabelValues(metrics.ResourceManagerCPU, metrics.ResourceManagerPod))
				require.NoError(t, err)
				require.InDelta(t, float64(testCase.requiredMetrics.expTotalErrors), errors, 0.001, "expected allocation errors to be incremented")

				return
			}
			require.NoError(t, err)

			if !reflect.DeepEqual(st.GetCPUAssignments(), testCase.expPodAssignments) {
				t.Errorf("StaticPolicy AllocatePod() error (%v). expected assignments: %v but got: %v",
					testCase.description, testCase.expPodAssignments, st.GetCPUAssignments())
			}

			if !st.GetDefaultCPUSet().Equals(testCase.expDefaultCPUSet) {
				t.Errorf("StaticPolicy AllocatePod() error (%v). expected default cpuset: %v but got: %v",
					testCase.description, testCase.expDefaultCPUSet, st.GetDefaultCPUSet())
			}

			allocations, err := testutil.GetCounterMetricValue(metrics.ResourceManagerAllocationsTotal.WithLabelValues(metrics.ResourceManagerCPU, metrics.ResourceManagerPod))
			require.NoError(t, err)
			require.InDelta(t, float64(testCase.requiredMetrics.expTotalAllocs), allocations, 0.001, "unexpected number of allocations")

			exclusiveAssignments, err := testutil.GetCounterMetricValue(metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerCPU, metrics.ResourceManagerExclusivePod))
			require.NoError(t, err)
			require.InDelta(t, float64(testCase.requiredMetrics.expExclusiveAssignments), exclusiveAssignments, 0.001, "unexpected number of assignments")

			podSharedPoolAssignments, err := testutil.GetCounterMetricValue(metrics.ResourceManagerContainerAssignments.WithLabelValues(metrics.ResourceManagerCPU, metrics.ResourceManagerSharedPod))
			require.NoError(t, err)
			require.InDelta(t, float64(testCase.requiredMetrics.expPodSharedPoolAssignments), podSharedPoolAssignments, 0.001, "unexpected number of assignments")
		})
	}
}

func TestSMTAlignmentErrorText(t *testing.T) {
	type smtErrTestCase struct {
		name     string
		err      SMTAlignmentError
		expected string
	}

	testCases := []smtErrTestCase{
		{
			name:     "base SMT alignment error",
			err:      SMTAlignmentError{RequestedCPUs: 15, CpusPerCore: 4},
			expected: `SMT Alignment Error: requested 15 cpus not multiple cpus per core = 4`,
		},
		{
			// Note the explicit 0. The intent is to signal the lack of physical CPUs, but
			// in the corner case of no available physical CPUs at all, without the explicit
			// flag we cannot distinguish the case, and before PR#127959 we printed the old message
			name:     "base SMT alignment error, no physical CPUs, missing flag",
			err:      SMTAlignmentError{RequestedCPUs: 4, CpusPerCore: 2, AvailablePhysicalCPUs: 0},
			expected: `SMT Alignment Error: requested 4 cpus not multiple cpus per core = 2`,
		},
		{
			name:     "base SMT alignment error, no physical CPUs, explicit flag",
			err:      SMTAlignmentError{RequestedCPUs: 4, CpusPerCore: 2, AvailablePhysicalCPUs: 0, CausedByPhysicalCPUs: true},
			expected: `SMT Alignment Error: not enough free physical CPUs: available physical CPUs = 0, requested CPUs = 4, CPUs per core = 2`,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			got := testCase.err.Error()
			if got != testCase.expected {
				t.Errorf("got=%v expected=%v", got, testCase.expected)
			}
		})
	}
}

func TestValidatePodScopeResources(t *testing.T) {
	testCases := []struct {
		name          string
		pod           *v1.Pod
		scope         string
		features      map[featuregate.Feature]bool
		expectErr     bool
		topology      *topology.CPUTopology
		numReserved   int
		reservedCPUs  cpuset.CPUSet
		policyOptions map[string]string
	}{
		{
			name: "Valid: Pod-level != Guaranteed containers, has podSharedPool containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"4",
				[]*containerOptions{},
				[]*containerOptions{
					{name: "c1", request: "2", limit: "2"},
					{name: "c2"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Only Guaranteed containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"4",
				[]*containerOptions{},
				[]*containerOptions{
					{name: "c1", request: "2", limit: "2"},
					{name: "c2", request: "2", limit: "2"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Only podSharedPool containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"4",
				[]*containerOptions{},
				[]*containerOptions{
					{name: "c1"},
					{name: "c2"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Pod-level == Guaranteed containers, has podSharedPool containers with request, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"6",
				[]*containerOptions{},
				[]*containerOptions{
					{name: "c1", request: "3", limit: "3"},
					{name: "c2", request: "2", limit: "2"},
					{name: "c3", request: "500m", limit: "1000m"},
					{name: "c4"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Failure: Pod-level == Guaranteed containers, has podSharedPool containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"5",
				[]*containerOptions{},
				[]*containerOptions{
					{name: "c1", request: "3", limit: "3"},
					{name: "c2", request: "2", limit: "2"},
					{name: "c3"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: true,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Pod-level resources with standard and restartable init containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "init-container1", request: "1", limit: "1"},
					{name: "restartable-init-container1", request: "1", limit: "1", restartPolicy: v1.ContainerRestartPolicyAlways},
				},
				[]*containerOptions{
					{name: "container1"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Pod-level resources equal init and standard container resources, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "init-container1", request: "2", limit: "2"},
				},
				[]*containerOptions{
					{name: "container1", request: "2", limit: "2"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Failure: Pod-level resources with shared standard init container and no available pool, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "restartable-init-container1", request: "2", limit: "2", restartPolicy: v1.ContainerRestartPolicyAlways},
					{name: "init-container1"},
				},
				[]*containerOptions{},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: true,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Pod-level resources with shared standard init container and no available pool, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "init-container1"},
					{name: "restartable-init-container1", request: "2", limit: "2", restartPolicy: v1.ContainerRestartPolicyAlways},
				},
				[]*containerOptions{},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Failure: Pod-level == Restartable guaranteed init containers, has podSharedPool containers, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "restartable-init-container1", request: "2", limit: "2", restartPolicy: v1.ContainerRestartPolicyAlways},
					{name: "restartable-init-container2", restartPolicy: v1.ContainerRestartPolicyAlways},
				},
				[]*containerOptions{},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: true,
			topology:  topoSingleSocketHT,
		},
		{
			name: "Valid: Pod-level resources with only restartable one init container and one standard container, PLR: Enabled, PLRM: Enabled",
			pod: makeMultiContainerPodWithOptionsAndPodLevelResources(
				"2",
				[]*containerOptions{
					{name: "restartable-init-container1", request: "1", limit: "1", restartPolicy: v1.ContainerRestartPolicyAlways},
				},
				[]*containerOptions{
					{name: "container1", request: "1", limit: "1"},
				},
			),
			scope:     topologymanager.PodTopologyScope,
			features:  map[featuregate.Feature]bool{pkgfeatures.PodLevelResources: true, pkgfeatures.PodLevelResourceManagers: true},
			expectErr: false,
			topology:  topoSingleSocketHT,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)

			for feature, enabled := range tc.features {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, enabled)
			}

			fakeTopologyManager := topologymanager.NewFakeManagerWithScope(tc.scope)
			policy, err := NewStaticPolicy(logger, tc.topology, tc.numReserved, tc.reservedCPUs, fakeTopologyManager, tc.policyOptions)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			err = policy.(*staticPolicy).validatePodScopeResources(logger, tc.pod)

			if tc.expectErr && err == nil {
				t.Error("Expected an error, but got none")
			}
			if !tc.expectErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		})
	}
}

func newCPUSetPtr(cpus ...int) *cpuset.CPUSet {
	ret := cpuset.New(cpus...)
	return &ret
}

func getPodUncoreCacheIDs(s state.Reader, topo *topology.CPUTopology, pod *v1.Pod) ([]int, error) {
	var uncoreCacheIDs []int
	for idx := range pod.Spec.Containers {
		container := &pod.Spec.Containers[idx]
		cset, ok := s.GetCPUSet(string(pod.UID), container.Name)
		if !ok {
			return nil, fmt.Errorf("GetCPUSet(%s, %s) not ok", pod.UID, container.Name)
		}
		for _, cpuID := range cset.UnsortedList() {
			info, ok := topo.CPUDetails[cpuID]
			if !ok {
				return nil, fmt.Errorf("cpuID %v not in topo.CPUDetails", cpuID)
			}
			uncoreCacheIDs = append(uncoreCacheIDs, info.UncoreCacheID)
		}
	}
	return uncoreCacheIDs, nil
}
