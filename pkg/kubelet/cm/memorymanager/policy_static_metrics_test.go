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

package memorymanager

import (
	"log"
	"os"
	"slices"
	"strconv"
	"strings"
	"testing"

	cadvisorapi "github.com/google/cadvisor/lib/model"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
)

type mainTB struct {
	cleanups []func()
}

func (tb *mainTB) Cleanup(f func())                  { tb.cleanups = append(tb.cleanups, f) }
func (tb *mainTB) Helper()                           {}
func (tb *mainTB) Name() string                      { return "TestMain" }
func (tb *mainTB) Logf(format string, args ...any)   { log.Printf(format, args...) }
func (tb *mainTB) Error(args ...any)                 { tb.Fatal(args...) }
func (tb *mainTB) Errorf(format string, args ...any) { tb.Fatalf(format, args...) }
func (tb *mainTB) Fatal(args ...any)                 { log.Fatal(args...) }
func (tb *mainTB) Fatalf(format string, args ...any) { log.Fatalf(format, args...) }

func (tb *mainTB) restore() {
	for _, cleanup := range slices.Backward(tb.cleanups) {
		cleanup()
	}
	tb.cleanups = nil
}

func TestMain(m *testing.M) {
	tb := &mainTB{}
	featuregatetesting.SetFeatureGatesDuringTest(tb, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.MemoryManagerDriftTolerance: true,
		features.PodLevelResources:           true,
		features.PodLevelResourceManagers:    true,
	})
	metrics.Register()
	tb.restore()

	os.Exit(m.Run())
}

func TestStaticPolicyStartMemoryDriftMetrics(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	const total = 8 * gb
	numaNode := func(nodeID int) *state.NUMANodeState {
		return &state.NUMANodeState{
			MemoryMap: map[v1.ResourceName]*state.MemoryTable{
				v1.ResourceMemory: {Allocatable: total - gb, Free: total - gb, Reserved: 0, SystemReserved: gb, TotalMemSize: total},
			},
			Cells:               []int{nodeID},
			NumberOfAssignments: 0,
		}
	}
	machineInfo := func(memory0, memory1 uint64) *cadvisorapi.MachineInfo {
		return &cadvisorapi.MachineInfo{
			Topology: []cadvisorapi.Node{
				{Id: 0, Memory: memory0},
				{Id: 1, Memory: memory1},
			},
		}
	}
	systemReserved := systemReservedMemory{
		0: map[v1.ResourceName]uint64{v1.ResourceMemory: gb},
		1: map[v1.ResourceName]uint64{v1.ResourceMemory: gb},
	}
	expected := func(drift0, drift1, tolerance uint64) string {
		return `
# HELP kubelet_memory_manager_drift_bytes [ALPHA] The absolute difference in bytes between the memory recorded for a NUMA node and the memory the node reports, as seen at the last memory manager start.
# TYPE kubelet_memory_manager_drift_bytes gauge
kubelet_memory_manager_drift_bytes{numa_node="0"} ` + strconv.FormatUint(drift0, 10) + `
kubelet_memory_manager_drift_bytes{numa_node="1"} ` + strconv.FormatUint(drift1, 10) + `
# HELP kubelet_memory_manager_drift_tolerance_bytes [ALPHA] The per-NUMA-node memory drift in bytes the memory manager tolerates across a restart. 0 means the persisted state has to match the machine exactly.
# TYPE kubelet_memory_manager_drift_tolerance_bytes gauge
kubelet_memory_manager_drift_tolerance_bytes ` + strconv.FormatUint(tolerance, 10) + `
`
	}

	testCases := []struct {
		description    string
		maxMemoryDrift uint64
		machineState   state.NUMANodeMap
		machineInfo    *cadvisorapi.MachineInfo
		expected       string
	}{
		{
			description:    "a fresh start reports a zero drift for every NUMA node",
			maxMemoryDrift: testMemoryDriftBound,
			machineState:   state.NUMANodeMap{},
			machineInfo:    machineInfo(total, total),
			expected:       expected(0, 0, testMemoryDriftBound),
		},
		{
			description:    "a restart reports the drift observed on each NUMA node",
			maxMemoryDrift: testMemoryDriftBound,
			machineState:   state.NUMANodeMap{0: numaNode(0), 1: numaNode(1)},
			machineInfo:    machineInfo(total-4*mb, total+8*mb),
			expected:       expected(4*mb, 8*mb, testMemoryDriftBound),
		},
		{
			description:    "the exact comparison reports a zero tolerance",
			maxMemoryDrift: 0,
			machineState:   state.NUMANodeMap{0: numaNode(0), 1: numaNode(1)},
			machineInfo:    machineInfo(total, total),
			expected:       expected(0, 0, 0),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			metrics.MemoryManagerDriftBytes.Reset()
			tc := testStaticPolicy{
				assignments:    state.ContainerMemoryAssignments{},
				machineState:   testCase.machineState,
				systemReserved: systemReserved,
				machineInfo:    testCase.machineInfo,
			}
			p, s, err := initTests(t, &tc, nil, nil)
			require.NoError(t, err)
			p.(*staticPolicy).maxMemoryDrift = testCase.maxMemoryDrift
			require.NoError(t, p.Start(logger, s))
			require.NoError(t, testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(testCase.expected),
				"kubelet_"+metrics.MemoryManagerDriftBytesKey, "kubelet_"+metrics.MemoryManagerDriftToleranceBytesKey))
		})
	}
}
