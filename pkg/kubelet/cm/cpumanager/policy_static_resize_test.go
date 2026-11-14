/*
Copyright The Kubernetes Authors.

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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/ktesting"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/admission"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager/bitmask"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/cpuset"
)

// The following lifecycle tests verify that InPlacePodVerticalScalingExclusiveCPUs
// alongside static CPU manager policy processes resize operations as per KEP
// All following tests require InPlacePodVerticalScalingExclusiveCPUs enabled
type staticPolicyResizeTest struct {
	description                 string
	topo                        *topology.CPUTopology
	numReservedCPUs             int
	reservedCPUs                *cpuset.CPUSet
	podUID                      string
	options                     map[string]string
	containerName               string
	stAssignments               state.ContainerCPUAssignments
	stBaselines                 state.ContainerCPUBaselines
	stDefaultCPUSet             cpuset.CPUSet
	inconsistentState           bool
	stAssignmentsInconsistent   state.ContainerCPUAssignments
	stBaselinesInconsistent     state.ContainerCPUBaselines
	stDefaultCPUSetInconsistent cpuset.CPUSet
	resizeTwice                 bool
	secondResizeLimit           string
	secondResizeRequest         string
	pod                         *v1.Pod
	qosClass                    v1.PodQOSClass
	podAllocated                string
	resizeLimit                 string
	resizeRequest               string
	topologyHint                *topologymanager.TopologyHint
	expErr                      error
	expCPUAlloc                 bool
	expCSet                     cpuset.CPUSet
}

// this is not a real Clone() - hence Pseudo- - because we don't clone some
// objects which are accessed read-only
func (spt staticPolicyResizeTest) PseudoClone() staticPolicyResizeTest {
	return staticPolicyResizeTest{
		description:                 spt.description,
		topo:                        spt.topo, // accessed in read-only
		numReservedCPUs:             spt.numReservedCPUs,
		podUID:                      spt.podUID,
		options:                     spt.options, // accessed in read-only
		containerName:               spt.containerName,
		stAssignments:               spt.stAssignments.Clone(),
		stBaselines:                 spt.stBaselines.Clone(),
		stDefaultCPUSet:             spt.stDefaultCPUSet.Clone(),
		inconsistentState:           spt.inconsistentState,
		stAssignmentsInconsistent:   spt.stAssignmentsInconsistent,
		stBaselinesInconsistent:     spt.stBaselinesInconsistent,
		stDefaultCPUSetInconsistent: spt.stDefaultCPUSetInconsistent.Clone(),
		resizeTwice:                 spt.resizeTwice,
		pod:                         spt.pod, // accessed in read-only
		qosClass:                    spt.qosClass,
		podAllocated:                spt.podAllocated,
		resizeLimit:                 spt.resizeLimit,
		resizeRequest:               spt.resizeRequest,
		topologyHint:                spt.topologyHint, // accessed in read-only
		expErr:                      spt.expErr,
		expCPUAlloc:                 spt.expCPUAlloc,
		expCSet:                     spt.expCSet.Clone(),
	}
}

func TestStaticPolicyStartAlognsideInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)
	testCases := []staticPolicyResizeTest{
		{
			description: "non-corrupted state",
			topo:        topoDualSocketHT,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"0": cpuset.New(0),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"0": {Baseline: cpuset.New(0)},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(),
			expCSet:         cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
		},
		{
			description:     "reserved cores 0 & 6 are not present in available cpuset",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1),
			expErr:          fmt.Errorf("not all reserved cpus: \"0,6\" are present in defaultCpuSet: \"0-1\""),
		},
		{
			description:     "some of reserved cores are present in available cpuset (StrictCPUReservationOption)",
			topo:            topoDualSocketHT,
			numReservedCPUs: 2,
			options:         map[string]string{StrictCPUReservationOption: "true"},
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"0": {Baseline: cpuset.New(0, 1, 2)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"0": {Baseline: cpuset.New(0, 1, 2)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"0": {Baseline: cpuset.New(0, 1, 2)},
					"1": {Baseline: cpuset.New(3, 4)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"0": {Baseline: cpuset.New(0, 1, 2)},
					"1": {Baseline: cpuset.New(3, 4)},
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			p, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(logger), testCase.options)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}
			policy := p.(*staticPolicy)
			st := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
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

func TestStaticPolicyAddWithInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {
	var largeTopoCPUids []int
	var largeTopoSock0CPUids []int
	var largeTopoSock1CPUids []int
	largeTopo := *topoQuadSocketFourWayHT
	for cpuid, val := range largeTopo.CPUDetails {
		largeTopoCPUids = append(largeTopoCPUids, cpuid)
		switch val.SocketID {
		case 0:
			largeTopoSock0CPUids = append(largeTopoSock0CPUids, cpuid)
		case 1:
			largeTopoSock1CPUids = append(largeTopoSock1CPUids, cpuid)
		}
	}
	largeTopoCPUSet := cpuset.New(largeTopoCPUids...)
	largeTopoSock0CPUSet := cpuset.New(largeTopoSock0CPUids...)
	largeTopoSock1CPUSet := cpuset.New(largeTopoSock1CPUids...)

	// these are the cases which must behave the same regardless the policy options.
	// So we will permutate the options to ensure this holds true.

	optionsInsensitiveTestCases := []staticPolicyResizeTest{
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectAllocOneCore",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer100": cpuset.New(2, 3, 6, 7),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(2, 3, 6, 7)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(2)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(1, 5)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New()},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(4, 5)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(2)},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(3, 11, 4, 5, 6, 7)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: largeTopoCPUSet.Difference(cpuset.New(1, 25, 13, 38, 2, 9, 11, 35, 23, 48, 12, 51,
						53, 173, 113, 233, 54, 61))},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: largeTopoCPUSet.Difference(largeTopoSock1CPUSet.Union(cpuset.New(10, 34, 22, 47, 53,
						173, 61, 181, 108, 228, 115, 235)))},
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
	defaultOptionsTestCases := []staticPolicyResizeTest{
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: largeTopoCPUSet.Difference(cpuset.New(10, 11, 53, 37, 55, 67, 52))},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          admission.NewOutOfResourceError(v1.ResourceCPU, fmt.Errorf("not enough cpus available to satisfy request: requested=8, available=7")),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodMultipleCores, SingleSocketHT, ExpectSameAllocation",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 2, 5, 6),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 2, 5, 6)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 3, 4, 7),
			pod:             makePod("fakePod", "fakeContainer3", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 2, 5, 6),
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(1, 2, 3)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainer5", "10000m", "10000m"),
			expErr:          admission.NewOutOfResourceError(v1.ResourceCPU, fmt.Errorf("not enough cpus available to satisfy request: requested=10, available=8")),
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(1, 2, 3, 4, 5, 6)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 7),
			pod:             makePod("fakePod", "fakeContainer5", "2000m", "2000m"),
			expErr:          admission.NewOutOfResourceError(v1.ResourceCPU, fmt.Errorf("not enough cpus available to satisfy request: requested=2, available=1")),
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: largeTopoCPUSet.Difference(cpuset.New(10, 11, 53, 37, 55, 67, 52))},
				},
			},
			stDefaultCPUSet: cpuset.New(10, 11, 53, 37, 55, 67, 52),
			pod:             makePod("fakePod", "fakeContainer5", "76000m", "76000m"),
			expErr:          admission.NewOutOfResourceError(v1.ResourceCPU, fmt.Errorf("not enough cpus available to satisfy request: requested=76, available=7")),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
	}

	// testcases for the FullPCPUsOnlyOption
	smtalignOptionTestCases := []staticPolicyResizeTest{
		{
			description: "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
			pod:             makePod("fakePod", "fakeContainerBug113537_2", "10000m", "10000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 2, 3, 4, 5, 7, 8, 9, 10, 11),
		},
		{
			// The purpose of this test is to verify that SMTAlignment consider
			// pre-allocated CPUs assigned to the container.
			// for example, in case of kubelet restart.
			description: "GuPodManyCores, DualSocketHT, ExpectReAllocCPUs",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 4,
			reservedCPUs:    newCPUSetPtr(0, 1, 6, 7),
			pod:             makePod("fakePod", "fakeContainer", "2", "2"),
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(5, 11),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(5, 11)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 5, 6, 7, 11),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(5, 11),
		},
		{
			// The purpose of this test is to verify that SMTAlignment consider
			// pre-allocated CPUs assigned to the container.
			// for example, in case of kubelet restart.
			description: "GuPodManyCores, DualSocketHT, ExpectReAllocCPUs",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 4,
			reservedCPUs:    newCPUSetPtr(0, 1, 6, 7),
			pod:             makePod("fakePod", "fakeContainer", "6", "6"),
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(2, 3, 4, 8, 9, 10),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(2, 3, 4, 8, 9, 10)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 5, 6, 7, 11),
			expErr:          SMTAlignmentError{RequestedCPUs: 6, CpusPerCore: 2, AvailablePhysicalCPUs: 2, CausedByPhysicalCPUs: true},
			expCPUAlloc:     false,
		},
		{
			description: "GuPodManyCores, DualSocketHT, NotEnoughAvailable, NoAlloc",
			topo:        topoDualSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 4,
			reservedCPUs:    newCPUSetPtr(0, 1, 6, 7),
			pod:             makePod("fakePod", "fakeContainer", "10", "10"),
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expErr:          SMTAlignmentError{RequestedCPUs: 10, CpusPerCore: 2, AvailablePhysicalCPUs: 8, CausedByPhysicalCPUs: true},
			expCPUAlloc:     false,
		},
	}

	// testcases for podResize
	podResizeTestCases := []staticPolicyResizeTest{
		{
			description:     "podResize GuPodMultipleCores, SingleSocketHT, ExpectSameAllocation",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 2, 5, 6),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 2, 5, 6)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 3, 4, 7),
			pod:             makePod("fakePod", "fakeContainer3", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 2, 5, 6),
		},
		{
			description: "podResize GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 5),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 5)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 6, 7),
			pod:             makePod("fakePod", "fakeContainer3", "4000m", "4000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 5),
		},
		{
			description: "podResize GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 5),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 5)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 6, 7),
			pod:             makePod("fakePod", "fakeContainer3", "2000m", "2000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 5),
		},
		{
			description: "podResize",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 5),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 5)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 6, 7),
			pod:             makePod("fakePod", "fakeContainer3", "100m", "100m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 5),
		},
		{
			description: "podResize",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "false",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer3", "1000m", "1000m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(4),
		},
		{
			description: "podResize",
			topo:        topoSingleSocketHT,
			options: map[string]string{
				FullPCPUsOnlyOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments: state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{
					"fakeContainer3": cpuset.New(1, 5),
				},
			},
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer3": {Baseline: cpuset.New(1, 5)},
				},
			},
			stDefaultCPUSet: cpuset.New(0, 2, 3, 4, 6, 7),
			pod:             makePod("fakePod", "fakeContainer3", "100m", "100m"),
			expErr:          nil,
			expCPUAlloc:     true,
			expCSet:         cpuset.New(1, 5),
		},
	}

	newNUMAAffinity := func(bits ...int) bitmask.BitMask {
		affinity, _ := bitmask.NewBitMask(bits...)
		return affinity
	}
	alignBySocketOptionTestCases := []staticPolicyResizeTest{
		{
			description: "Align by socket: true, cpu's within same socket of numa in hint are part of allocation",
			topo:        topoDualSocketMultiNumaPerSocketHT,
			options: map[string]string{
				AlignBySocketOption: "true",
			},
			numReservedCPUs: 1,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t, tCase)
		}
	}

	for _, testCase := range defaultOptionsTestCases {
		runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t, testCase)
	}
	for _, testCase := range smtalignOptionTestCases {
		runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t, testCase)
	}
	for _, testCase := range alignBySocketOptionTestCases {
		runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t, testCase)
	}
	for _, testCase := range podResizeTestCases {
		runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t, testCase)
	}
}

func runStaticPolicyTestCaseWithInPlacePodVerticalScalingExclusiveCPUs(t *testing.T, testCase staticPolicyResizeTest) {
	logger, _ := ktesting.NewTestContext(t)
	tm := topologymanager.NewFakeManager(logger)
	if testCase.topologyHint != nil {
		tm = topologymanager.NewFakeManagerWithHint(logger, testCase.topologyHint)
	}
	cpus := cpuset.New()
	if testCase.reservedCPUs != nil {
		cpus = testCase.reservedCPUs.Clone()
	}
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
	// Below will be used only for alignBySocketOptionTestCases ( adding here not to duplicate function )
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)

	policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpus, tm, testCase.options)
	if err != nil {
		t.Fatalf("NewStaticPolicy() failed (%v) : %v", testCase.description, err)
	}

	st := &mockState{
		assignments:   testCase.stAssignments,
		baselines:     testCase.stBaselines,
		defaultCPUSet: testCase.stDefaultCPUSet,
	}

	container := &testCase.pod.Spec.Containers[0]
	err = policy.Allocate(logger, st, testCase.pod, container, lifecycle.AddOperation)
	if !reflect.DeepEqual(err, testCase.expErr) {
		t.Errorf("StaticPolicy Allocate() error (%v). expected add error: %q but got: %q",
			testCase.description, testCase.expErr, err)
	}

	if testCase.expCPUAlloc {
		original, found := st.baselines[string(testCase.pod.UID)][container.Name]
		if !found {
			t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
				testCase.description, container.Name, st.baselines)
		}

		if !original.Baseline.Equals(testCase.expCSet) {
			t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s but got %s",
				testCase.description, testCase.expCSet, original.Baseline)
		}

		if !original.Baseline.Intersection(st.defaultCPUSet).IsEmpty() {
			t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s to be disoint from the shared cpuset %s",
				testCase.description, original.Baseline, st.defaultCPUSet)
		}
	}

	if !testCase.expCPUAlloc {
		_, found := st.baselines[string(testCase.pod.UID)][container.Name]
		if found {
			t.Errorf("StaticPolicy Allocate() error (%v). Did not expect container %v to be present in assignments %v",
				testCase.description, container.Name, st.assignments)
		}
	}
}

func TestStaticPolicyPodResizeCPUsSingleContainerPod(t *testing.T) {
	testCases := []struct {
		staticPolicyResizeTest
		expAllocErr                                      error
		expDefaultCPUSetAfterAdd                         cpuset.CPUSet
		expDefaultCPUSetAfterUpdateWithInconsistentState cpuset.CPUSet
		expDefaultCPUSetAfterUpdate                      cpuset.CPUSet
		expDefaultCPUSetSizeAfterResize                  int
		expAllocErrAfterSecondResize                     error
		expCSetAfterSecondResize                         cpuset.CPUSet
		expCSetAfterSecondResizeSize                     int
		expDefaultCPUSetAfterRemove                      cpuset.CPUSet
	}{
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, Increase allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0, 4
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "4000m",
				resizeRequest:   "4000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, Keep same allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0, 4
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from shared to exclusive pool, (requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, Decrease allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "4000m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "not allowed to reduce container exclusively allocated pool below baseline, (baseline CPUs = 4, requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:        cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate:     cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetSizeAfterResize: 4,
			expDefaultCPUSetAfterRemove:     cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in shared pool with more than one core, Attempt to move to exclusively allocated pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2100m", limit: "2100m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2100m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from shared to exclusive pool, (requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in shared pool, Increase CPU and keep in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "100m", limit: "100m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "100m",
				resizeLimit:     "200m",
				resizeRequest:   "200m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in shared pool, Increase CPU and keep in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "1100m", limit: "1100m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "1100m",
				resizeLimit:     "1200m",
				resizeRequest:   "1200m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in shared pool with less than one core, Decrease CPU and keep in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "200m", limit: "200m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "200m",
				resizeLimit:     "100m",
				resizeRequest:   "100m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in shared pool with more than one core, Decrease CPU and keep in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "1200m", limit: "1200m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "1200m",
				resizeLimit:     "1100m",
				resizeRequest:   "1100m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, Move to shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "1500m",
				resizeRequest:   "1500m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from exclusive to shared pool, (requested CPUs = 1500m)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then decrease allocated CPUs equal to baselineCPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0, 4
				),
				qosClass:            v1.PodQOSGuaranteed,
				podAllocated:        "2000m",
				resizeLimit:         "4000m",
				resizeRequest:       "4000m",
				containerName:       "appContainer-0",
				stAssignments:       state.ContainerCPUAssignments{},
				stBaselines:         state.ContainerCPUBaselines{},
				stDefaultCPUSet:     cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				resizeTwice:         true,
				secondResizeLimit:   "2000m",
				secondResizeRequest: "2000m",
			},
			expDefaultCPUSetAfterAdd:     cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate:  cpuset.New(2, 3, 6, 7),
			expCSetAfterSecondResize:     cpuset.New(1, 2, 3, 5, 6, 7),
			expCSetAfterSecondResizeSize: 6,
			expDefaultCPUSetAfterRemove:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then decrease allocated CPUs above baselineCPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-4
				),
				qosClass:            v1.PodQOSGuaranteed,
				podAllocated:        "2000m",
				resizeLimit:         "6000m",
				resizeRequest:       "6000m",
				containerName:       "appContainer-0",
				stAssignments:       state.ContainerCPUAssignments{},
				stBaselines:         state.ContainerCPUBaselines{},
				stDefaultCPUSet:     cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				resizeTwice:         true,
				secondResizeLimit:   "4000m",
				secondResizeRequest: "4000m",
			},
			expDefaultCPUSetAfterAdd:     cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate:  cpuset.New(3, 7),
			expCSetAfterSecondResize:     cpuset.New(2, 3, 6, 7),
			expCSetAfterSecondResizeSize: 4,
			expDefaultCPUSetAfterRemove:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then fail to decrease allocated CPUs below baselineCPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:            v1.PodQOSGuaranteed,
				podAllocated:        "4000m",
				resizeLimit:         "6000m",
				resizeRequest:       "6000m",
				containerName:       "appContainer-0",
				stAssignments:       state.ContainerCPUAssignments{},
				stBaselines:         state.ContainerCPUBaselines{},
				stDefaultCPUSet:     cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				resizeTwice:         true,
				secondResizeLimit:   "2000m",
				secondResizeRequest: "2000m",
			},
			expAllocErrAfterSecondResize: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "not allowed to reduce container exclusively allocated pool below baseline, (baseline CPUs = 4, requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:     cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate:  cpuset.New(3, 7),
			expCSetAfterSecondResize:     cpuset.New(3, 7),
			expCSetAfterSecondResizeSize: 2,
			expDefaultCPUSetAfterRemove:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		// Test defense against corrupted checkpoint
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then fail to decrease allocated CPUs caused by temporary inconsistent recorded state missing entries",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:                    v1.PodQOSGuaranteed,
				podAllocated:                "4000m",
				resizeLimit:                 "6000m",
				resizeRequest:               "6000m",
				containerName:               "appContainer-0",
				stAssignments:               state.ContainerCPUAssignments{},
				stBaselines:                 state.ContainerCPUBaselines{},
				stDefaultCPUSet:             cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				inconsistentState:           true,
				stAssignmentsInconsistent:   state.ContainerCPUAssignments{},
				stBaselinesInconsistent:     state.ContainerCPUBaselines{},
				stDefaultCPUSetInconsistent: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			// In this case we demonstrate the case
			// where the container is in exclusive pool but
			// because of corrupted recorded state both assigned and baseline CPUs
			// are missing. In this case we treat the container as it is in shared pool
			// because under normal conditiions it should be in shared pool.
			// With the next reconcile the state should be updated and.
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from shared to exclusive pool, (requested CPUs = 6)",
			},
			expDefaultCPUSetAfterAdd:                         cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdateWithInconsistentState: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate:                      cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove:                      cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then fail to decrease allocated CPUs caused by temporary inconsistent recorded state with wrong empty baseline entries",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:          v1.PodQOSGuaranteed,
				podAllocated:      "4000m",
				resizeLimit:       "6000m",
				resizeRequest:     "6000m",
				containerName:     "appContainer-0",
				stAssignments:     state.ContainerCPUAssignments{},
				stBaselines:       state.ContainerCPUBaselines{},
				stDefaultCPUSet:   cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				inconsistentState: true,
				stAssignmentsInconsistent: state.ContainerCPUAssignments{
					"podUID": map[string]cpuset.CPUSet{
						"appContainer-0": cpuset.New(0, 1, 4, 5),
					},
				},
				stBaselinesInconsistent: state.ContainerCPUBaselines{
					"podUID": map[string]state.ContainerCPUBaseline{
						"appContainer-0": {Baseline: cpuset.New()},
					},
				},
				stDefaultCPUSetInconsistent: cpuset.New(2, 3, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "inconsistent recorded state, baseline CPUs are empty while assigned CPUs exist",
			},
			expDefaultCPUSetAfterAdd:                         cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdateWithInconsistentState: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate:                      cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove:                      cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then fail to decrease allocated CPUs caused by temporary inconsistent recorded state, with baseline CPUs not a subset of assigned CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:          v1.PodQOSGuaranteed,
				podAllocated:      "4000m",
				resizeLimit:       "6000m",
				resizeRequest:     "6000m",
				containerName:     "appContainer-0",
				stAssignments:     state.ContainerCPUAssignments{},
				stBaselines:       state.ContainerCPUBaselines{},
				stDefaultCPUSet:   cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				inconsistentState: true,
				stAssignmentsInconsistent: state.ContainerCPUAssignments{
					"podUID": map[string]cpuset.CPUSet{
						"appContainer-0": cpuset.New(0, 1),
					},
				},
				stBaselinesInconsistent: state.ContainerCPUBaselines{
					"podUID": map[string]state.ContainerCPUBaseline{
						"appContainer-0": {Baseline: cpuset.New(0, 1, 4, 5)},
					},
				},
				stDefaultCPUSetInconsistent: cpuset.New(2, 3, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "inconsistent recorded state, baseline CPUs are more than assigned CPUs",
			},
			expDefaultCPUSetAfterAdd:                         cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdateWithInconsistentState: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate:                      cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove:                      cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Container in exclusively allocated pool, first increase allocated CPUs and then fail to decrease allocated CPUs caused by temporary inconsistent recorded state with empty baselines while assignments are correct",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:          v1.PodQOSGuaranteed,
				podAllocated:      "4000m",
				resizeLimit:       "6000m",
				resizeRequest:     "6000m",
				containerName:     "appContainer-0",
				stAssignments:     state.ContainerCPUAssignments{},
				stBaselines:       state.ContainerCPUBaselines{},
				stDefaultCPUSet:   cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
				inconsistentState: true,
				stAssignmentsInconsistent: state.ContainerCPUAssignments{
					"podUID": map[string]cpuset.CPUSet{
						"appContainer-0": cpuset.New(0, 1, 4, 5),
					},
				},
				stBaselinesInconsistent: state.ContainerCPUBaselines{
					"podUID": map[string]state.ContainerCPUBaseline{
						"appContainer-0": {Baseline: cpuset.New()},
					},
				},
				stDefaultCPUSetInconsistent: cpuset.New(2, 3, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "inconsistent recorded state, baseline CPUs are empty while assigned CPUs exist",
			},
			expDefaultCPUSetAfterAdd:                         cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdateWithInconsistentState: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate:                      cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove:                      cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			stConsistent := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}
			pod := testCase.pod
			pod.Status.QOSClass = testCase.qosClass

			// allocate
			for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
				err := policy.Allocate(logger, stConsistent, pod, &container, lifecycle.AddOperation)
				if err != nil {
					t.Errorf("StaticPolicy Allocate() error (%v). expected no error but got %v",
						testCase.description, err)
				}
			}
			if !reflect.DeepEqual(stConsistent.defaultCPUSet, testCase.expDefaultCPUSetAfterAdd) {
				t.Errorf("StaticPolicy Allocate() error (%v) before pod resize. expected default cpuset %v but got %v",
					testCase.description, testCase.expDefaultCPUSetAfterAdd, stConsistent.defaultCPUSet)
			}

			// resize
			pod.Status.ContainerStatuses = []v1.ContainerStatus{
				{
					Name: testCase.containerName,
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(testCase.podAllocated),
					},
				},
			}
			pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.resizeLimit),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.resizeRequest),
				},
			}
			podResized := pod
			if testCase.inconsistentState {
				stInconsistent := &mockState{
					assignments:   testCase.stAssignmentsInconsistent,
					baselines:     testCase.stBaselinesInconsistent,
					defaultCPUSet: testCase.stDefaultCPUSetInconsistent,
				}
				for _, container := range append(podResized.Spec.InitContainers, podResized.Spec.Containers...) {
					err := policy.Allocate(logger, stInconsistent, podResized, &container, lifecycle.UpdateOperation)
					if err != nil {
						if !reflect.DeepEqual(err, testCase.expAllocErr) {
							t.Errorf("StaticPolicy Allocate() error (%v), expected error: %v but got: %v",
								testCase.description, testCase.expAllocErr, err)
						}
						if !reflect.DeepEqual(stInconsistent.defaultCPUSet, testCase.expDefaultCPUSetAfterUpdateWithInconsistentState) {
							t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset size to stay equal to %v but got %v",
								testCase.description, testCase.expDefaultCPUSetAfterUpdateWithInconsistentState, stInconsistent.defaultCPUSet)
						}
					}
				}
			} else {
				for _, container := range append(podResized.Spec.InitContainers, podResized.Spec.Containers...) {
					err := policy.Allocate(logger, stConsistent, podResized, &container, lifecycle.UpdateOperation)
					if err != nil {
						if !reflect.DeepEqual(err, testCase.expAllocErr) {
							t.Errorf("StaticPolicy Allocate() error (%v), expected error: %v but got: %v",
								testCase.description, testCase.expAllocErr, err)
						}
						if !reflect.DeepEqual(stConsistent.defaultCPUSet, testCase.expDefaultCPUSetAfterUpdate) {
							t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset size to stay equal to %v but got %v",
								testCase.description, testCase.expDefaultCPUSetAfterUpdate, stConsistent.defaultCPUSet)
						}
					}
				}
			}
			// Fot the tests, we assume that broken State is now recovered after CPU Managers Reconcile loop.

			if testCase.expDefaultCPUSetSizeAfterResize > 0 {
				// expDefaultCPUSetSizeAfterResize is used when testing scale down because allocated CPUs are not deterministic,
				// since size of defaultCPUSet is deterministic and also interesection  with expected allocation
				// should not be nill.
				if !reflect.DeepEqual(stConsistent.defaultCPUSet.Size(), testCase.expDefaultCPUSetSizeAfterResize) {
					t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset size equal to %v but got %v",
						testCase.description, testCase.expDefaultCPUSetSizeAfterResize, stConsistent.defaultCPUSet.Size())
				}
			} else {
				if !reflect.DeepEqual(stConsistent.defaultCPUSet, testCase.expDefaultCPUSetAfterUpdate) {
					t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset %v but got %v",
						testCase.description, testCase.expDefaultCPUSetAfterUpdate, stConsistent.defaultCPUSet)
				}
			}

			if testCase.resizeTwice {
				// resize for a second time, this is required to test scale downs on an already resize pod.
				pod.Status.ContainerStatuses = []v1.ContainerStatus{
					{
						Name: testCase.containerName,
						AllocatedResources: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse(testCase.podAllocated),
						},
					},
				}
				pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.secondResizeLimit),
					},
					Requests: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.secondResizeRequest),
					},
				}
				podResized := pod
				for _, container := range append(podResized.Spec.InitContainers, podResized.Spec.Containers...) {
					err := policy.Allocate(logger, stConsistent, podResized, &container, lifecycle.UpdateOperation)
					if err != nil {
						if !reflect.DeepEqual(err, testCase.expAllocErrAfterSecondResize) {
							t.Errorf("StaticPolicy Allocate() error (%v), expected error: %v but got: %v",
								testCase.description, testCase.expAllocErrAfterSecondResize, err)
						}
					}
				}
				if testCase.expCSetAfterSecondResizeSize > 0 {
					// expCSetAfterSecondResizeSize is used when testing scale down because allocated CPUs are not deterministic,
					// since size of defaultCPUSet is deterministic and also interesection  with expected allocation
					// should not be nill.
					if !reflect.DeepEqual(stConsistent.defaultCPUSet.Size(), testCase.expCSetAfterSecondResizeSize) {
						t.Errorf("StaticPolicy Allocate() error (%v) after second pod resize. expected default cpuset size equal to %v but got %v",
							testCase.description, testCase.expCSetAfterSecondResizeSize, stConsistent.defaultCPUSet.Size())
					}
				} else {
					if !reflect.DeepEqual(stConsistent.defaultCPUSet, testCase.expCSetAfterSecondResize) {
						t.Errorf("StaticPolicy Allocate() error (%v) after second pod resize. expected default cpuset %v but got %v",
							testCase.description, testCase.expCSetAfterSecondResize, stConsistent.defaultCPUSet)
					}
				}
			}

			// remove
			err = policy.RemoveContainer(logger, stConsistent, string(pod.UID), testCase.containerName)
			if err != nil {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected no error but got %v",
					testCase.description, err)
			}

			if !reflect.DeepEqual(stConsistent.defaultCPUSet, testCase.expDefaultCPUSetAfterRemove) {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected default cpuset %v but got %v",
					testCase.description, testCase.expDefaultCPUSetAfterRemove, stConsistent.defaultCPUSet)
			}
			if _, found := stConsistent.assignments[string(pod.UID)][testCase.containerName]; found {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected (pod %v, container %v) not be in assignments %v",
					testCase.description, testCase.podUID, testCase.containerName, stConsistent.assignments)
			}
		})
	}
}

func TestStaticPolicyPodResizeCPUsMultiContainerPod(t *testing.T) {
	testCases := []struct {
		staticPolicyResizeTest
		containerName2                  string
		expAllocErr                     error
		expDefaultCPUSetAfterAdd        cpuset.CPUSet
		expDefaultCPUSetAfterUpdate     cpuset.CPUSet
		expDefaultCPUSetSizeAfterResize int
		expDefaultCPUSetAfterRemove     cpuset.CPUSet
	}{
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Containers in exclusively allocated pool, Increase appContainer-0 allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // 0, 4
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 1, 5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "4000m",
				resizeRequest:   "4000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2:              "appContainer-1",
			expDefaultCPUSetAfterAdd:    cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(3, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Containers in exclusively allocated pool, Keep same allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // 0, 4
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 1, 5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2:              "appContainer-1",
			expAllocErr:                 nil,
			expDefaultCPUSetAfterAdd:    cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Containers in exclusively allocated pool, Decrease appContainer-0 allocated CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // appContainer-0 CPUs 0, 4, 1, 5
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // appContainer-1 CPUS 2, 6, 3, 7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "4000m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "not allowed to reduce container exclusively allocated pool below baseline, (baseline CPUs = 4, requested CPUs = 2)",
			},
			containerName2:              "appContainer-1",
			expDefaultCPUSetAfterAdd:    cpuset.New(),
			expDefaultCPUSetAfterUpdate: cpuset.New(),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Containers in shared pool with more than one core, Attempt to move to exclusively allocated pool reducing CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2100m", limit: "2100m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // 0-7
						{request: "2100m", limit: "2100m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2100m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2: "appContainer-1",
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from shared to exclusive pool, (requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, appContainer-0 in shared pool, Increase CPU and keep appContainer-0 in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "100m", limit: "100m", restartPolicy: v1.ContainerRestartPolicy("Never")},    // 2-3, 6-7
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "100m",
				resizeLimit:     "200m",
				resizeRequest:   "200m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2:              "appContainer-1",
			expDefaultCPUSetAfterAdd:    cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, appContainer-0 in shared pool with more than one core, Increase CPU and keep appContainer-0 in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "1100m", limit: "1100m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // 0-7
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "1100m",
				resizeLimit:     "1200m",
				resizeRequest:   "1200m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2:              "appContainer-1",
			expDefaultCPUSetAfterAdd:    cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, appContainer-0 in shared pool, appContainer-1 in exclusive pool, Decrease CPU and keep in shared pool",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "200m", limit: "200m", restartPolicy: v1.ContainerRestartPolicy("Never")},    // 0-7
						{request: "4000m", limit: "4000m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-1, 4-5
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "200m",
				resizeLimit:     "100m",
				resizeRequest:   "100m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2:              "appContainer-1",
			expDefaultCPUSetAfterAdd:    cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(2, 3, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, appContainer-0 in exclusively allocated pool with 2 CPUs, Attempt to move to shared pool requesting 1500m",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "2000m", limit: "2000m", restartPolicy: v1.ContainerRestartPolicy("Never")}, // 0-1, 4-5
						{request: "200m", limit: "200m", restartPolicy: v1.ContainerRestartPolicy("Never")}},  // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "2000m",
				resizeLimit:     "1500m",
				resizeRequest:   "1500m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2: "appContainer-1",
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from exclusive to shared pool, (requested CPUs = 1500m)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(1, 2, 3, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
		{
			staticPolicyResizeTest: staticPolicyResizeTest{
				description: "SingleSocketHT, PodResize, Containers in shared pool with more than one core, Attempt to move to exclusively allocated pool increasing CPUs",
				topo:        topoSingleSocketHT,
				pod: makeMultiContainerPodWithOptions(
					nil,
					[]*containerOptions{
						{request: "1500m", limit: "1500m", restartPolicy: v1.ContainerRestartPolicy("Never")},  // 0-7
						{request: "1500m", limit: "1500m", restartPolicy: v1.ContainerRestartPolicy("Never")}}, // 0-7
				),
				qosClass:        v1.PodQOSGuaranteed,
				podAllocated:    "1500m",
				resizeLimit:     "2000m",
				resizeRequest:   "2000m",
				containerName:   "appContainer-0",
				stAssignments:   state.ContainerCPUAssignments{},
				stBaselines:     state.ContainerCPUBaselines{},
				stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			},
			containerName2: "appContainer-1",
			expAllocErr: isResizeFeasibleError{
				PodUID:        "podUID",
				ContainerName: "appContainer-0",
				CausedBy:      "cannot move container from shared to exclusive pool, (requested CPUs = 2)",
			},
			expDefaultCPUSetAfterAdd:    cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterUpdate: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			expDefaultCPUSetAfterRemove: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
		},
	}

	for _, testCase := range testCases {
		logger, _ := ktesting.NewTestContext(t)
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			policy, _ := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(logger), nil)

			st := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}
			pod := testCase.pod
			pod.Status.QOSClass = testCase.qosClass

			// allocate
			for _, container := range append(pod.Spec.InitContainers, pod.Spec.Containers...) {
				err := policy.Allocate(logger, st, pod, &container, lifecycle.AddOperation)
				if err != nil {
					t.Errorf("StaticPolicy Allocate() error (%v). expected no error but got %v",
						testCase.description, err)
				}
			}
			if !reflect.DeepEqual(st.defaultCPUSet, testCase.expDefaultCPUSetAfterAdd) {
				t.Errorf("StaticPolicy Allocate() error (%v) before pod resize. expected default cpuset %v but got %v",
					testCase.description, testCase.expDefaultCPUSetAfterAdd, st.defaultCPUSet)
			}

			// resize
			pod.Status.ContainerStatuses = []v1.ContainerStatus{
				{
					Name: testCase.containerName,
					AllocatedResources: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse(testCase.podAllocated),
					},
				},
			}
			pod.Spec.Containers[0].Resources = v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.resizeLimit),
				},
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU): resource.MustParse(testCase.resizeRequest),
				},
			}
			podResized := pod
			for _, container := range append(podResized.Spec.InitContainers, podResized.Spec.Containers...) {
				err := policy.Allocate(logger, st, podResized, &container, lifecycle.UpdateOperation)
				if err != nil {
					if !reflect.DeepEqual(err, testCase.expAllocErr) {
						t.Errorf("StaticPolicy Allocate() error (%v), expected error: %v but got: %v",
							testCase.description, testCase.expAllocErr, err)
					}
				}
			}

			if testCase.expDefaultCPUSetSizeAfterResize > 0 {
				// expDefaultCPUSetSizeAfterResize is used when testing scale down because allocated CPUs are not deterministic,
				// since size of defaultCPUSet is deterministic and also interesection  with expected allocation
				// should not be nill.
				if !reflect.DeepEqual(st.defaultCPUSet.Size(), testCase.expDefaultCPUSetSizeAfterResize) {
					t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset size equal to %v but got %v",
						testCase.description, testCase.expDefaultCPUSetSizeAfterResize, st.defaultCPUSet.Size())
				}
			} else {
				if !reflect.DeepEqual(st.defaultCPUSet, testCase.expDefaultCPUSetAfterUpdate) {
					t.Errorf("StaticPolicy Allocate() error (%v) after pod resize. expected default cpuset %v but got %v",
						testCase.description, testCase.expDefaultCPUSetAfterUpdate, st.defaultCPUSet)
				}
			}

			// remove
			err := policy.RemoveContainer(logger, st, string(pod.UID), testCase.containerName)
			if err != nil {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected no error but got %v",
					testCase.description, err)
			}
			err = policy.RemoveContainer(logger, st, string(pod.UID), testCase.containerName2)
			if err != nil {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected no error but got %v",
					testCase.description, err)
			}

			if !reflect.DeepEqual(st.defaultCPUSet, testCase.expDefaultCPUSetAfterRemove) {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected default cpuset %v but got %v",
					testCase.description, testCase.expDefaultCPUSetAfterRemove, st.defaultCPUSet)
			}
			if _, found := st.assignments[string(pod.UID)][testCase.containerName]; found {
				t.Errorf("StaticPolicy RemoveContainer() error (%v) after pod resize. expected (pod %v, container %v) not be in assignments %v",
					testCase.description, testCase.podUID, testCase.containerName, st.assignments)
			}
		})
	}
}

func TestStaticPolicyRemoveAlognsideInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {

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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer1": {Baseline: cpuset.New(1, 2, 3)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer1": {Baseline: cpuset.New(1, 2, 3)},
					"fakeContainer2": {Baseline: cpuset.New(4, 5, 6, 7)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer1": {Baseline: cpuset.New(1, 3, 5)},
					"fakeContainer2": {Baseline: cpuset.New(2, 4)},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer1": {Baseline: cpuset.New(1, 3, 5)},
				},
			},
			stDefaultCPUSet: cpuset.New(2, 4, 6, 7),
			expCSet:         cpuset.New(2, 4, 6, 7),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			logger, _ := ktesting.NewTestContext(t)
			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, cpuset.New(), topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			st := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
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

			if _, found := st.baselines[testCase.podUID][testCase.containerName]; found {
				t.Errorf("StaticPolicy RemoveContainer() error (%v). expected (pod %v, container %v) not be in baselines %v",
					testCase.description, testCase.podUID, testCase.containerName, st.baselines)
			}

		})
	}
}

func TestTopologyAwareAllocateCPUsAlongsideInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {
	testCases := []struct {
		description     string
		topo            *topology.CPUTopology
		stAssignments   state.ContainerCPUAssignments
		stBaselines     state.ContainerCPUBaselines
		stDefaultCPUSet cpuset.CPUSet
		numRequested    int
		socketMask      bitmask.BitMask
		expCSet         cpuset.CPUSet
	}{
		{
			description:     "Request 2 CPUs, No BitMask",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			numRequested:    2,
			socketMask:      nil,
			expCSet:         cpuset.New(0, 6),
		},
		{
			description:     "Request 2 CPUs, BitMask on Socket 0",
			topo:            topoDualSocketHT,
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
		p, err := NewStaticPolicy(logger, tc.topo, 0, cpuset.New(), topologymanager.NewFakeManager(logger), nil)
		if err != nil {
			t.Fatalf("NewStaticPolicy() failed: %v", err)
		}
		policy := p.(*staticPolicy)
		st := &mockState{
			assignments:   tc.stAssignments,
			baselines:     tc.stBaselines,
			defaultCPUSet: tc.stDefaultCPUSet,
		}
		err = policy.Start(logger, st)
		if err != nil {
			t.Errorf("StaticPolicy Start() error (%v)", err)
			continue
		}

		cpuAlloc, err := policy.allocateCPUs(logger, st, tc.numRequested, tc.socketMask, cpuset.New())
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

func TestStaticPolicyAllocateCPUsForResize(t *testing.T) {
	// topoDualSocketHT: {cores} by NUMA:
	// NUMA0 (even CPUs): {0,6}, {2,8}, {4,10}
	// NUMA1 (odd  CPUs): {1,7}, {3,9}, {5,11}
	allCPUs := topoDualSocketHT.CPUDetails.CPUs()
	testCases := []struct {
		description  string
		options      map[string]string
		currentCPUs  cpuset.CPUSet
		baselineCPUs cpuset.CPUSet
		// otherCPUs are exclusively allocated to another container.
		otherCPUs    cpuset.CPUSet
		reusableCPUs cpuset.CPUSet
		numRequested int
		numaAffinity bitmask.BitMask
		expCSet      cpuset.CPUSet
		expErr       string
	}{
		// Scale up
		{
			description:  "scale up, NUMA affinity has enough CPUs, full cores are taken",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			// CPUs {2,8} are held in state by a completed init container of the
			// same pod, hence out of the default CPUSet, and are reusable.
			description:  "scale up, NUMA affinity satisfiable only via reusable CPUs of a completed init container",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(2, 4, 8, 10),
			reusableCPUs: cpuset.New(2, 8),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			description:  "scale up, NUMA affinity has enough CPUs, distribute-cpus-across-cores, one CPU per core is taken",
			options:      map[string]string{DistributeCPUsAcrossCoresOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 4, 6),
		},
		{
			description:  "scale up, NUMA affinity has not enough CPUs, remaining full core taken from other NUMA node",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(4, 10),
			numRequested: 6,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 1, 2, 6, 7, 8),
		},
		{
			description:  "scale up, NUMA affinity has not enough CPUs, distribute-cpus-across-cores, remaining CPUs taken from different cores of other NUMA node",
			options:      map[string]string{DistributeCPUsAcrossCoresOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(4, 10),
			numRequested: 6,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 1, 2, 3, 6, 8),
		},
		{
			description:  "scale up, no NUMA affinity, request exceeds a NUMA node, NUMA node of currently allocated CPUs is filled first",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 8,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 2, 4, 6, 7, 8, 10),
		},
		{
			description:  "scale up, no NUMA affinity, request exceeds a NUMA node, distribute-cpus-across-numa, CPUs evenly distributed across NUMA nodes",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 8,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 2, 3, 6, 7, 8, 9),
		},
		{
			description:  "scale up, no NUMA affinity, NUMA node of currently allocated CPUs has not enough CPUs",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(4, 10),
			numRequested: 6,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 2, 6, 7, 8),
		},
		{
			description:  "scale up, no NUMA affinity, NUMA node of currently allocated CPUs has not enough CPUs, distribute-cpus-across-numa, CPUs evenly distributed across NUMA nodes",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(4, 10),
			numRequested: 6,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 2, 3, 6, 7),
		},
		{
			description:  "scale up, no NUMA affinity, NUMA node of currently allocated CPUs has not enough CPUs, distribute-cpus-across-numa and full-pcpus-only, full cores distributed across NUMA nodes",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true", FullPCPUsOnlyOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(4, 10),
			numRequested: 6,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 3, 6, 7, 9),
		},
		{
			description:  "scale up, multi-NUMA affinity, CPUs packed on NUMA node of currently allocated CPUs",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0, 1),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			description:  "scale up, multi-NUMA affinity, distribute-cpus-across-numa, CPUs distributed across all NUMA nodes of the affinity",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0, 1),
			expCSet:      cpuset.New(0, 1, 6, 7),
		},
		{
			description:  "scale up, single NUMA affinity, distribute-cpus-across-numa, CPUs stay on NUMA node of the affinity",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			description:  "scale up, not enough CPUs available",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(1, 3, 4, 5, 7, 9, 10, 11),
			numRequested: 6,
			numaAffinity: newNUMAAffinity(0),
			expErr:       "unable to take remaining 2 of 6 requested CPUs outside of NUMA affinity 01 (2 CPUs retained, 2 CPUs taken from NUMA affinity): not enough cpus available to satisfy request: requested=6 (of which 4 already allocated), available=0",
		},
		{
			description:  "scale up, no NUMA affinity, not enough CPUs available",
			currentCPUs:  cpuset.New(0, 6),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(1, 3, 4, 5, 7, 9, 10, 11),
			numRequested: 6,
			numaAffinity: nil,
			expErr:       "unable to take 4 of 6 requested CPUs (2 CPUs retained): not enough cpus available to satisfy request: requested=6 (of which 2 already allocated), available=2",
		},
		// No scaling
		{
			description:  "no change of CPUs",
			currentCPUs:  cpuset.New(0, 2, 6, 8),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		// Scale down
		{
			description:  "scale down to baseline, only baseline CPUs are kept regardless of NUMA affinity",
			currentCPUs:  cpuset.New(0, 1, 6, 7),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 2,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 6),
		},
		{
			description:  "scale down above baseline, CPUs released from NUMA node outside of NUMA affinity",
			currentCPUs:  cpuset.New(0, 1, 2, 6, 7, 8),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			description:  "scale down above baseline, NUMA affinity has not enough CPUs, remaining CPUs taken from other NUMA node",
			currentCPUs:  cpuset.New(0, 1, 3, 6, 7, 9),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 1, 6, 7),
		},
		{
			description:  "scale down above baseline, no NUMA affinity, distribute-cpus-across-numa, kept CPUs evenly distributed across NUMA nodes",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 1, 2, 3, 6, 7, 8, 9),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 6,
			numaAffinity: nil,
			expCSet:      cpuset.New(0, 1, 2, 3, 6, 7),
		},
		{
			description:  "scale down above baseline, distribute-cpus-across-numa, CPUs from NUMA affinity kept first",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 1, 2, 3, 6, 7, 8, 9),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 6,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 1, 2, 6, 7, 8),
		},
		{
			description:  "scale down above baseline, single NUMA affinity, distribute-cpus-across-numa, kept CPUs stay on NUMA node of the affinity",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 1, 2, 4, 6, 7, 8, 10),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0),
			expCSet:      cpuset.New(0, 2, 6, 8),
		},
		{
			description:  "scale down above baseline, multi-NUMA affinity, distribute-cpus-across-numa, kept CPUs distributed across all NUMA nodes of the affinity",
			options:      map[string]string{DistributeCPUsAcrossNUMAOption: "true"},
			currentCPUs:  cpuset.New(0, 1, 2, 4, 6, 7, 8, 10),
			baselineCPUs: cpuset.New(0, 6),
			otherCPUs:    cpuset.New(),
			numRequested: 4,
			numaAffinity: newNUMAAffinity(0, 1),
			expCSet:      cpuset.New(0, 1, 6, 7),
		},
		{
			description:  "scale down, baseline CPUs are not part of currently allocated CPUs",
			currentCPUs:  cpuset.New(0, 1, 6, 7),
			baselineCPUs: cpuset.New(2, 8),
			otherCPUs:    cpuset.New(),
			numRequested: 2,
			numaAffinity: newNUMAAffinity(0),
			expErr:       "retained CPUs 2,8 are not a subset of allocatable CPUs 0-1,6-7",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)
			p, err := NewStaticPolicy(logger, topoDualSocketHT, 0, cpuset.New(), topologymanager.NewFakeManager(logger), tc.options)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}
			policy := p.(*staticPolicy)

			assignments := state.ContainerCPUAssignments{
				"fakePod": map[string]cpuset.CPUSet{"fakeContainer": tc.currentCPUs},
			}
			if !tc.otherCPUs.IsEmpty() {
				assignments["otherPod"] = map[string]cpuset.CPUSet{"otherContainer": tc.otherCPUs}
			}
			st := &mockState{
				assignments: assignments,
				baselines: state.ContainerCPUBaselines{
					"fakePod": map[string]state.ContainerCPUBaseline{"fakeContainer": {Baseline: tc.baselineCPUs}},
				},
				defaultCPUSet: allCPUs.Difference(tc.currentCPUs).Difference(tc.otherCPUs),
			}
			expDefaultCPUSet := st.defaultCPUSet

			cpuAlloc, err := policy.allocateCPUsForResize(logger, st, tc.numRequested, tc.numaAffinity, tc.reusableCPUs, tc.currentCPUs, tc.baselineCPUs)
			if tc.expErr != "" {
				if err == nil || err.Error() != tc.expErr {
					t.Fatalf("expected error %q but got %v", tc.expErr, err)
				}
				if !st.defaultCPUSet.Equals(expDefaultCPUSet) {
					t.Errorf("expected default CPUSet to stay %v but got %v", expDefaultCPUSet, st.defaultCPUSet)
				}
				return
			}
			if err != nil {
				t.Fatalf("expected CPUSet %v but got error %v", tc.expCSet, err)
			}
			if !tc.expCSet.Equals(cpuAlloc.CPUs) {
				t.Errorf("expected CPUSet %v but got %v", tc.expCSet, cpuAlloc.CPUs)
			}
			if !tc.baselineCPUs.IsSubsetOf(cpuAlloc.CPUs) {
				t.Errorf("expected baseline CPUs %v to be retained in %v", tc.baselineCPUs, cpuAlloc.CPUs)
			}

			if !st.defaultCPUSet.Equals(expDefaultCPUSet) {
				t.Errorf("expected default CPUSet to stay %v but got %v", expDefaultCPUSet, st.defaultCPUSet)
			}
		})
	}
}

// above test cases are without kubelet --reserved-cpus cmd option
// the following tests are with --reserved-cpus configured
// with InPlacePodVerticalScalingExclusiveCPUs enabled
type staticPolicyTestWithResvListWithInPlacePodVerticalScalingExclusiveCPUs struct {
	description     string
	topo            *topology.CPUTopology
	numReservedCPUs int
	reserved        cpuset.CPUSet
	stAssignments   state.ContainerCPUAssignments
	stBaselines     state.ContainerCPUBaselines
	stDefaultCPUSet cpuset.CPUSet
	pod             *v1.Pod
	expErr          error
	expCPUAlloc     bool
	expCSet         cpuset.CPUSet
}

func TestStaticPolicyAddWithResvListAlongsideInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {

	testCases := []staticPolicyTestWithResvListWithInPlacePodVerticalScalingExclusiveCPUs{
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectError",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 1,
			reserved:        cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			pod:             makePod("fakePod", "fakeContainer2", "8000m", "8000m"),
			expErr:          admission.NewOutOfResourceError(v1.ResourceCPU, fmt.Errorf("not enough cpus available to satisfy request: requested=8, available=7")),
			expCPUAlloc:     false,
			expCSet:         cpuset.New(),
		},
		{
			description:     "GuPodSingleCore, SingleSocketHT, ExpectAllocOneCPU",
			topo:            topoSingleSocketHT,
			numReservedCPUs: 2,
			reserved:        cpuset.New(0, 1),
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
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
			stBaselines: state.ContainerCPUBaselines{
				"fakePod": map[string]state.ContainerCPUBaseline{
					"fakeContainer100": {Baseline: cpuset.New(2, 3, 6, 7)},
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
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			logger, _ := ktesting.NewTestContext(t)
			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reserved, topologymanager.NewFakeManager(logger), nil)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			st := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}

			container := &testCase.pod.Spec.Containers[0]
			err = policy.Allocate(logger, st, testCase.pod, container, lifecycle.AddOperation)
			if !reflect.DeepEqual(err, testCase.expErr) {
				t.Errorf("StaticPolicy Allocate() error (%v). expected add error: %v but got: %v",
					testCase.description, testCase.expErr, err)
			}

			if testCase.expCPUAlloc {
				original, found := st.baselines[string(testCase.pod.UID)][container.Name]
				if !found {
					t.Errorf("StaticPolicy Allocate() error (%v). expected container %v to be present in assignments %v",
						testCase.description, container.Name, st.baselines)
				}

				if !original.Baseline.Equals(testCase.expCSet) {
					t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s but got %s",
						testCase.description, testCase.expCSet, original.Baseline)
				}

				if !original.Baseline.Intersection(st.defaultCPUSet).IsEmpty() {
					t.Errorf("StaticPolicy Allocate() error (%v). expected cpuset %s to be disoint from the shared cpuset %s",
						testCase.description, original.Baseline, st.defaultCPUSet)
				}
			}

			if !testCase.expCPUAlloc {
				_, found := st.baselines[string(testCase.pod.UID)][container.Name]
				if found {
					t.Errorf("StaticPolicy Allocate() error (%v). Did not expect container %v to be present in assignments %v",
						testCase.description, container.Name, st.baselines)
				}
			}

		})
	}
}

func TestStaticPolicyAllocatePodWithInPlacePodVerticalScalingExclusiveCPUs(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	testCases := []staticPolicyAllocatePodTest{
		{
			description:     "should successfully allocate CPUs for a guaranteed pod with pod-level resources",
			topo:            topoDualSocketHT,
			numReservedCPUs: 1,
			reservedCPUs:    cpuset.New(0),
			stAssignments:   state.ContainerCPUAssignments{},
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePodWithPodLevelResources("pod1", "2", "2", "container1", "1", "1"),
			topologyHint:    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:          nil,
			expPodBaselines: state.ContainerCPUBaselines{
				"pod1": map[string]state.ContainerCPUBaseline{
					"container1": {Baseline: cpuset.New(2)},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod:             makePodWithPodLevelResources("gu-pod-level-resources", "2", "2", "gu-container", "1", "1"),
			topologyHint:    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:          nil,
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-resources": map[string]state.ContainerCPUBaseline{
					"gu-container": {Baseline: cpuset.New(2)},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-resources", "1", "1", []containerSpec{}, []containerSpec{
				{name: "ngu-container"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-resources": map[string]state.ContainerCPUBaseline{
					"ngu-container": {Baseline: cpuset.New(6)},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-mix-ctn", "3", "3", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "gu-container-2", request: "1", limit: "1"},
				{name: "ngu-container"},
			}),
			topologyHint: topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:       nil,
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-mix-ctn": {
					"gu-container-1": {Baseline: cpuset.New(6)},
					"gu-container-2": {Baseline: cpuset.New(2)},
					"ngu-container":  {Baseline: cpuset.New(8)},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-gu-init-ctn": {
					"gu-init-container-1": {Baseline: cpuset.New(2)},
					"gu-init-container-2": {Baseline: cpuset.New(2)},
					"gu-container-1":      {Baseline: cpuset.New(2)},
					"gu-container-2":      {Baseline: cpuset.New(8)},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-gu-init-ctn": {
					"ngu-init-container-1":  {Baseline: cpuset.New(2, 6, 8)},
					"gu-init-restartable-2": {Baseline: cpuset.New(6)},
					"ngu-init-container-3":  {Baseline: cpuset.New(2, 8)},
					"gu-container-1":        {Baseline: cpuset.New(2)},
					"ngu-container-2":       {Baseline: cpuset.New(8)},
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
			stBaselines:     state.ContainerCPUBaselines{},
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
			expPodBaselines: state.ContainerCPUBaselines{
				"gu-pod-level-gu-init-ctn": {
					"ngu-init-container-1":   {Baseline: cpuset.New(2, 6, 8)},
					"ngu-init-restartable-2": {Baseline: cpuset.New(2, 8)},
					"ngu-init-container-3":   {Baseline: cpuset.New(2, 6, 8)},
					"gu-container-1":         {Baseline: cpuset.New(6)},
					"ngu-container-2":        {Baseline: cpuset.New(2, 8)},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("gu-pod-level-empty-pod-shared-pool", "2", "2", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
				{name: "gu-container-2", request: "1", limit: "1"},
				{name: "ngu-container"},
			}),
			topologyHint:                    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:                          admission.NewEmptyPodSharedPoolError(fmt.Errorf("pod rejected, sum of exclusive container cpu requests equals pod budget, leaving no cpus for shared containers")),
			expPodBaselines:                 state.ContainerCPUBaselines{},
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
			stBaselines:     state.ContainerCPUBaselines{},
			stDefaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			pod: makePodWithContainersAndPodLevelResources("ngu-pod-level-empty-pod-shared-pool", "2", "1", []containerSpec{}, []containerSpec{
				{name: "gu-container-1", request: "1", limit: "1"},
			}),
			topologyHint:                    topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true},
			expErr:                          nil,
			expPodBaselines:                 state.ContainerCPUBaselines{},
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)

			metrics.Register()
			metrics.ResourceManagerAllocationsTotal.Reset()
			metrics.ResourceManagerAllocationErrorsTotal.Reset()
			metrics.ResourceManagerContainerAssignments.Reset()

			policy, err := NewStaticPolicy(logger, testCase.topo, testCase.numReservedCPUs, testCase.reservedCPUs, topologymanager.NewFakeManager(logger), testCase.options)
			if err != nil {
				t.Fatalf("NewStaticPolicy() failed: %v", err)
			}

			st := &mockState{
				assignments:   testCase.stAssignments,
				baselines:     testCase.stBaselines,
				defaultCPUSet: testCase.stDefaultCPUSet,
			}

			err = policy.AllocatePod(logger, st, testCase.pod, lifecycle.AddOperation)
			if testCase.expErr != nil {
				require.Error(t, err)

				errors, err := testutil.GetCounterMetricValue(metrics.ResourceManagerAllocationErrorsTotal.WithLabelValues(metrics.ResourceManagerCPU, metrics.ResourceManagerPod))
				require.NoError(t, err)
				require.InDelta(t, float64(testCase.requiredMetrics.expTotalErrors), errors, 0.001, "expected allocation errors to be incremented")

				return
			}
			require.NoError(t, err)

			if !reflect.DeepEqual(st.GetCPUBaselines(), testCase.expPodBaselines) {
				t.Errorf("StaticPolicy AllocatePod() error (%v). expected allocations: %v but got: %v",
					testCase.description, testCase.expPodBaselines, st.GetCPUBaselines())
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

func TestGenerateCPUTopologyHintsForResize(t *testing.T) {
	// topoDualSocketNoHT: 8 CPUs, 2 NUMA nodes, 4 CPUs each
	// NUMA 0: CPUs {0,1,2,3}, NUMA 1: CPUs {4,5,6,7}

	testCases := []struct {
		description   string
		topo          *topology.CPUTopology
		reservedCPUs  cpuset.CPUSet
		availableCPUs cpuset.CPUSet
		reusableCPUs  cpuset.CPUSet
		retainedCPUs  cpuset.CPUSet
		freeCPUs      cpuset.CPUSet
		request       int
		expectedHints []topologymanager.TopologyHint
	}{
		{
			description:   "no reserverdCPUs, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with one CPU less than allocated size",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(2, 3, 4, 5, 6, 7),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(0, 1),
			retainedCPUs: cpuset.New(0, 1),
			freeCPUs:     cpuset.New(2, 3, 4, 5, 6, 7),
			request:      1,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 4 allocatable, 2 free, 2 assigned → Score = 50
			// NUMA 1: No hint generated
			// NUMA {0,1}: 8 allocatable, 6 free, 2 assigned → 2*100/8 = 25
			// This case is for a resize when allocated exists, requested is less than size
			// of allocated and baseline exists inside allocated.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 25},
			},
		},
		{
			description:   "one reservedCPU, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with one CPU less than allocated size",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(0),
			availableCPUs: cpuset.New(3, 4, 5, 6, 7),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(1, 2),
			retainedCPUs: cpuset.New(1, 2),
			freeCPUs:     cpuset.New(3, 4, 5, 6, 7),
			request:      1,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 3 allocatable, 2 free, 2 assigned → Score = 2*100/3 = 66
			// NUMA 1: No hint generated
			// NUMA {0,1}: 7 allocatable, 6 free, 2 assigned → 2*100/7 = 28
			// This case is for a resize when allocated exists, requested is less than size
			// of allocated and baseline exists inside allocated.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 66},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 28},
			},
		},
		{
			description:   "no reserverdCPUs, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with three CPUs larger than allocated size",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(0, 1),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(0, 1),
			retainedCPUs: cpuset.New(0, 1),
			freeCPUs:     cpuset.New(2, 3, 4, 5, 6, 7),
			request:      3,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 4 allocatable, 2 free, 2 assigned → Score = 2*100/4 = 50
			// NUMA 1: No hint generated
			// NUMA {0,1}: 4 allocatable, 2 free, 2 assigned → Score = 2*100/4 = 25
			// This case is for a resize when allocated exists, requested is larger than size
			// of allocated and baseline exists inside allocated.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 25},
			},
		},
		{
			description:   "One reservedCPU, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with one CPU",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(0),
			availableCPUs: cpuset.New(1, 2),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(1, 2),
			retainedCPUs: cpuset.New(1, 2),
			freeCPUs:     cpuset.New(3, 4, 5, 6, 7),
			request:      3,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 3 allocatable, 2 free, 2 assigned → Score = 2*100/3 = 66
			// NUMA 1: No hint generated
			// NUMA {0,1}: 7 allocatable, 6 free, 2 assigned → 2*100/7 = 28
			// This case is for a resize when allocated exists, requested is less than size
			// of allocated and baseline exists inside allocated.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 66},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 28},
			},
		},
		{
			description:   "no reserverdCPUs, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with three CPUs larger than allocated size",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(0, 1),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(),
			freeCPUs:     cpuset.New(2, 3, 4, 5, 6, 7),
			request:      2,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 4 allocatable, 2 free, 2 assigned → Score = 2*100/4 = 50
			// NUMA 1: No hint generated
			// NUMA {0,1}: 4 allocatable, 2 free, 2 assigned → Score = 2*100/4 = 25
			// This case is for a resize when allocated exists, requested is equal to the size
			// of allocated and baseline exist inside allocated.

			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 25},
			},
		},
		{
			description:   "One reservedCPU, 2 assigned in NUMA 0 as baselineCPUs, attempt to scale up with one CPU",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(0),
			availableCPUs: cpuset.New(1, 2),
			// Use case: ReusableCPUs are the baselines
			reusableCPUs: cpuset.New(),
			freeCPUs:     cpuset.New(3, 4, 5, 6, 7),
			request:      2,
			// Below Scores are based on node status when the decision was taken.
			// NUMA 0: 3 allocatable, 2 free, 2 assigned → Score = 2*100/3 = 66
			// NUMA 1: No hint generated
			// NUMA {0,1}: 7 allocatable, 6 free, 2 assigned → 2*100/7 = 28
			// This case is for a resize when allocated exists, requested is equal to the size
			// of allocated and baseline exist inside allocated.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 66},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 28},
			},
		},
		// Below cases are for a resize when allocated doesn't exist and a regeneration is performed.
		{
			description:   "nothing assigned, symmetric NUMA, all scores are 1",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			reusableCPUs:  cpuset.New(),
			freeCPUs:      cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			request:       1,
			// NUMA 0: 4 allocatable, 4 free, 0 assigned → max(1, 0) = 1
			// NUMA 1: 4 allocatable, 4 free, 0 assigned → max(1, 0) = 1
			// NUMA {0,1} 1: 8 allocatable, 8 free, 0 assigned → max(1, 0) = 1
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 1},
				{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true, Score: 1},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 1},
			},
		},
		{
			description:   "2 of 4 assigned on NUMA 0, score reflects utilization",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(2, 3, 4, 5, 6, 7),
			reusableCPUs:  cpuset.New(),
			freeCPUs:      cpuset.New(2, 3, 4, 5, 6, 7),
			request:       1,
			// NUMA 0: 4 allocatable, 2 free, 2 assigned → 2*100/4 = 50
			// NUMA 1: 4 allocatable, 4 free, 0 assigned → max(1, 0) = 1
			// NUMA {0,1}: 8 allocatable, 6 free, 2 assigned → 2*100/8 = 25
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true, Score: 1},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 25},
			},
		},
		{
			description:   "asymmetric reserved CPUs, partial assignment",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(0, 1),
			availableCPUs: cpuset.New(3, 4, 5, 6, 7),
			reusableCPUs:  cpuset.New(),
			freeCPUs:      cpuset.New(3, 4, 5, 6, 7),
			request:       1,
			// NUMA 0: allocatable={2,3}=2, free_on_0={3}=1, assigned=1 → 1*100/2 = 50
			// NUMA 1: allocatable={4,5,6,7}=4, free_on_1={4,5,6,7}=4, assigned=0 → max(1,0)=1
			// NUMA {0,1}: allocatable=6, free=5, assigned=1 → 1*100/6 = 16
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 50},
				{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true, Score: 1},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 16},
			},
		},
		{
			description:   "full utilization via reusable CPUs, score 100",
			topo:          topoSingleSocketHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(),
			reusableCPUs:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			retainedCPUs:  cpuset.New(0, 1, 2, 3, 4, 5, 6, 7),
			freeCPUs:      cpuset.New(),
			request:       1,
			// NUMA 0: 8 allocatable, 0 free, 8 assigned → 8*100/8 = 100
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 100},
			},
		},
		{
			description:   "3 of 4 assigned on each NUMA, high utilization",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(3, 7),
			reusableCPUs:  cpuset.New(),
			freeCPUs:      cpuset.New(3, 7),
			request:       1,
			// NUMA 0: 4 allocatable, 1 free, 3 assigned → 3*100/4 = 75
			// NUMA 1: 4 allocatable, 1 free, 3 assigned → 3*100/4 = 75
			// NUMA {0,1}: 8 allocatable, 2 free, 6 assigned → 6*100/8 = 75
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 75},
				{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true, Score: 75},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 75},
			},
		},
		{
			description:   "regeneration, usable set differs from free set",
			topo:          topoDualSocketNoHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(0, 1),
			reusableCPUs:  cpuset.New(),
			freeCPUs:      cpuset.New(4, 5, 6, 7),
			request:       2,
			// NUMA 0: 4 allocatable, 0 free, 4 assigned → 4*100/4 = 100
			// NUMA {0,1}: 8 allocatable, 4 free, 4 assigned → 4*100/8 = 50
			// NUMA 1 generates no hint: none of the usable CPUs live there.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true, Score: 100},
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: false, Score: 50},
			},
		},
		{
			// topoDualSocketHT: 12 CPUs, 2 NUMA nodes, 6 CPUs each
			// NUMA 0: CPUs {0,2,4,6,8,10}, NUMA 1: CPUs {1,3,5,7,9,11}
			description:   "retained CPUs span both NUMA nodes, request fits in one: the hint covering the retained CPUs is preferred",
			topo:          topoDualSocketHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(2, 3, 4, 5, 8, 9, 10, 11),
			// Use case: ReusableCPUs are the retained CPUs of a scale up from 4 to 5
			reusableCPUs: cpuset.New(0, 1, 6, 7),
			retainedCPUs: cpuset.New(0, 1, 6, 7),
			freeCPUs:     cpuset.New(2, 3, 4, 5, 8, 9, 10, 11),
			request:      5,
			// NUMA 0: no hint generated, it does not contain all retained CPUs
			// NUMA 1: no hint generated, it does not contain all retained CPUs
			// NUMA {0,1}: 12 allocatable, 8 free, 4 assigned → 4*100/12 = 33
			// The narrowest affinity by request size is a single NUMA node, but the
			// retained CPUs span both nodes, so the hint covering them is preferred.
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: true, Score: 33},
			},
		},
		{
			description:   "request outgrows the retained CPUs NUMA node: the narrowest hint still containing them is preferred",
			topo:          topoDualSocketHT,
			reservedCPUs:  cpuset.New(),
			availableCPUs: cpuset.New(1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			// Use case: ReusableCPUs are the retained CPUs of a scale up from 2 to 7
			reusableCPUs: cpuset.New(0, 2),
			retainedCPUs: cpuset.New(0, 2),
			freeCPUs:     cpuset.New(1, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			request:      7,
			// NUMA 0: no hint generated, 2 retained + 4 available CPUs cannot satisfy 7
			// NUMA 1: no hint generated, it does not contain all retained CPUs
			// NUMA {0,1}: 12 allocatable, 10 free, 2 assigned → 2*100/12 = 16
			expectedHints: []topologymanager.TopologyHint{
				{NUMANodeAffinity: newNUMAAffinity(1, 0), Preferred: true, Score: 16},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.InPlacePodVerticalScalingExclusiveCPUs, true)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.CPUManagerPolicyAlphaOptions, true)

			p := staticPolicy{
				topology:     testCase.topo,
				reservedCPUs: testCase.reservedCPUs,
				numaNodes:    testCase.topo.CPUDetails.NUMANodes().List(),
			}

			hints := p.generateCPUTopologyHintsForResize(testCase.availableCPUs, testCase.reusableCPUs, testCase.retainedCPUs, testCase.freeCPUs, testCase.request)

			if !reflect.DeepEqual(testCase.expectedHints, hints) {
				t.Errorf("Expected in result to be %v, got %v", testCase.expectedHints, hints)
			}
		})
	}
}
