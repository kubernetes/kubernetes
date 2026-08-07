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
	"log"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/klog/v2/ktesting"
	pkgfeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/topology"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/utils/cpuset"
)

// mainTB adapts TestMain to the minimal interface the feature gate test
// helpers need. The cleanups restoring the gates are collected instead of
// being deferred to the end of a test, so they can be run as soon as the
// metrics are registered.
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

// restore runs the collected cleanups in reverse order, the way testing does.
func (tb *mainTB) restore() {
	for i := len(tb.cleanups) - 1; i >= 0; i-- {
		tb.cleanups[i]()
	}
	tb.cleanups = nil
}

// TestMain makes sure every metric asserted on in this package exists before
// any test runs.
// metrics.Register is guarded by a sync.Once and registers the resource
// manager metrics only when PodLevelResourceManagers is enabled.
// So without this the first test calling metrics.Register decides which
// metrics are registered and exist.
// (an unregistered metric silently discards every update and always reads back zero)
func TestMain(m *testing.M) {
	tb := &mainTB{}
	featuregatetesting.SetFeatureGatesDuringTest(tb, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		pkgfeatures.PodLevelResources:        true,
		pkgfeatures.PodLevelResourceManagers: true,
	})
	metrics.Register()
	tb.restore()

	os.Exit(m.Run())
}

type cpuMetricsSnapshot struct {
	exclusiveCPUs   float64
	sharedPoolMilli float64
	perNUMA         map[string]float64
}

// readCPUMetrics reads the per-NUMA gauge by gathering the registry rather
// than through WithLabelValues, which would silently create a missing series:
// a NUMA node absent from the snapshot means no series exists for it.
func readCPUMetrics(t *testing.T) cpuMetricsSnapshot {
	t.Helper()

	exclusiveCPUs, err := testutil.GetGaugeMetricValue(metrics.CPUManagerExclusiveCPUsAllocationCount)
	require.NoError(t, err)
	sharedPoolMilli, err := testutil.GetGaugeMetricValue(metrics.CPUManagerSharedPoolSizeMilliCores)
	require.NoError(t, err)

	perNUMA := map[string]float64{}
	families, err := legacyregistry.DefaultGatherer.Gather()
	require.NoError(t, err)
	for _, family := range families {
		if family.GetName() != "kubelet_"+metrics.CPUManagerAllocationPerNUMAKey {
			continue
		}
		for _, sample := range family.GetMetric() {
			for _, label := range sample.GetLabel() {
				if label.GetName() == metrics.AlignedNUMANode {
					perNUMA[label.GetValue()] = sample.GetGauge().GetValue()
				}
			}
		}
	}
	return cpuMetricsSnapshot{exclusiveCPUs: exclusiveCPUs, sharedPoolMilli: sharedPoolMilli, perNUMA: perNUMA}
}

func assertCPUMetrics(t *testing.T, expected cpuMetricsSnapshot) {
	t.Helper()
	got := readCPUMetrics(t)
	require.InDelta(t, expected.exclusiveCPUs, got.exclusiveCPUs, 0.001, "exclusive CPU allocation count")
	require.InDelta(t, expected.sharedPoolMilli, got.sharedPoolMilli, 0.001, "shared pool size millicores")
	require.Equal(t, expected.perNUMA, got.perNUMA, "allocation per NUMA node")
}

func newMetricsTestPolicy(t *testing.T, topo *topology.CPUTopology, numReservedCPUs int, reservedCPUs cpuset.CPUSet, hint *topologymanager.TopologyHint) *staticPolicy {
	t.Helper()
	metrics.Register()
	metrics.CPUManagerAllocationPerNUMA.Reset()

	logger, _ := ktesting.NewTestContext(t)
	affinity := topologymanager.NewFakeManagerWithHint(logger, hint)
	policy, err := NewStaticPolicy(logger, topo, numReservedCPUs, reservedCPUs, affinity, nil)
	require.NoError(t, err)
	return policy.(*staticPolicy)
}

func scrambleCPUMetrics() {
	metrics.CPUManagerAllocationPerNUMA.Reset()
	metrics.CPUManagerExclusiveCPUsAllocationCount.Set(-1)
	metrics.CPUManagerSharedPoolSizeMilliCores.Set(-1)
}

func TestStaticPolicyMetricsInitialization(t *testing.T) {
	testCases := []struct {
		description                     string
		podLevelResourceManagersEnabled bool
		stAssignments                   state.ContainerCPUAssignments
		stPodAssignments                state.PodCPUAssignments
		stDefaultCPUSet                 cpuset.CPUSet
		expected                        cpuMetricsSnapshot
	}{
		{
			description:                     "empty state with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stAssignments:                   state.ContainerCPUAssignments{},
			stDefaultCPUSet:                 cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   0,
				sharedPoolMilli: 11000,
				perNUMA:         map[string]float64{"0": 0, "1": 0},
			},
		},
		{
			description:                     "empty state with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stAssignments:                   state.ContainerCPUAssignments{},
			stDefaultCPUSet:                 cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   0,
				sharedPoolMilli: 11000,
				perNUMA:         map[string]float64{"0": 0, "1": 0},
			},
		},
		{
			description:                     "container assignments with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{"cont": cpuset.New(2, 8)},
				"podB": map[string]cpuset.CPUSet{"cont": cpuset.New(1, 7)},
			},
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   4,
				sharedPoolMilli: 7000,
				perNUMA:         map[string]float64{"0": 2, "1": 2},
			},
		},
		{
			description:                     "container assignments with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{"cont": cpuset.New(2, 8)},
				"podB": map[string]cpuset.CPUSet{"cont": cpuset.New(1, 7)},
			},
			stDefaultCPUSet: cpuset.New(0, 3, 4, 5, 6, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   4,
				sharedPoolMilli: 7000,
				perNUMA:         map[string]float64{"0": 2, "1": 2},
			},
		},
		{
			description:                     "overlapping assignments of an init container reused by an app container with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"init-cont": cpuset.New(2, 8),
					"app-cont":  cpuset.New(2, 6, 8),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 4, 5, 7, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   3,
				sharedPoolMilli: 8000,
				perNUMA:         map[string]float64{"0": 3, "1": 0},
			},
		},
		{
			description:                     "overlapping assignments of an init container reused by an app container with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"init-cont": cpuset.New(2, 8),
					"app-cont":  cpuset.New(2, 6, 8),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 4, 5, 7, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   3,
				sharedPoolMilli: 8000,
				perNUMA:         map[string]float64{"0": 3, "1": 0},
			},
		},
		{
			description:                     "one-shot init container reused by app containers running alongside a sidecar with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"init-cont":    cpuset.New(2, 8),
					"sidecar-cont": cpuset.New(6),
					"app-cont-1":   cpuset.New(2, 8),
					"app-cont-2":   cpuset.New(4),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 5, 7, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   4,
				sharedPoolMilli: 7000,
				perNUMA:         map[string]float64{"0": 4, "1": 0},
			},
		},
		{
			description:                     "one-shot init container reused by app containers running alongside a sidecar with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"init-cont":    cpuset.New(2, 8),
					"sidecar-cont": cpuset.New(6),
					"app-cont-1":   cpuset.New(2, 8),
					"app-cont-2":   cpuset.New(4),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 1, 3, 5, 7, 9, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   4,
				sharedPoolMilli: 7000,
				perNUMA:         map[string]float64{"0": 4, "1": 0},
			},
		},
		{
			description:                     "pod-level bubble spanning both NUMA nodes with all container assignments with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 2, 3, 4, 6, 7, 8, 10)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"gu-cont":     cpuset.New(1, 7),
					"shared-cont": cpuset.New(2, 3, 4, 6, 8, 10),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 5, 9, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   8,
				sharedPoolMilli: 3000,
				perNUMA:         map[string]float64{"0": 5, "1": 3},
			},
		},
		{
			description:                     "pod-level bubble spanning both NUMA nodes with all container assignments with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 2, 3, 4, 6, 7, 8, 10)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"gu-cont":     cpuset.New(1, 7),
					"shared-cont": cpuset.New(2, 3, 4, 6, 8, 10),
				},
			},
			stDefaultCPUSet: cpuset.New(0, 5, 9, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   8,
				sharedPoolMilli: 3000,
				perNUMA:         map[string]float64{"0": 5, "1": 3},
			},
		},
		{
			description:                     "pod-level bubble spanning both NUMA nodes with partial container assignments with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 2, 3, 4, 6, 7, 8, 10)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{"shared-cont": cpuset.New(2, 3, 4, 6, 8, 10)},
			},
			stDefaultCPUSet: cpuset.New(0, 5, 9, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   6,
				sharedPoolMilli: 3000,
				perNUMA:         map[string]float64{"0": 5, "1": 1},
			},
		},
		{
			description:                     "pod-level bubble spanning both NUMA nodes with partial container assignments with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 2, 3, 4, 6, 7, 8, 10)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{"shared-cont": cpuset.New(2, 3, 4, 6, 8, 10)},
			},
			stDefaultCPUSet: cpuset.New(0, 5, 9, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   8,
				sharedPoolMilli: 3000,
				perNUMA:         map[string]float64{"0": 5, "1": 3},
			},
		},
		{
			description:                     "container assignments mixed with a pod-level bubble with PodLevelResourceManagers disabled",
			podLevelResourceManagersEnabled: false,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 3, 7, 9)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"gu-cont":     cpuset.New(1, 7),
					"shared-cont": cpuset.New(3, 9),
				},
				"podB": map[string]cpuset.CPUSet{"cont": cpuset.New(2, 8)},
			},
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   6,
				sharedPoolMilli: 5000,
				perNUMA:         map[string]float64{"0": 2, "1": 4},
			},
		},
		{
			description:                     "container assignments mixed with a pod-level bubble with PodLevelResourceManagers enabled",
			podLevelResourceManagersEnabled: true,
			stPodAssignments: state.PodCPUAssignments{
				"podA": state.PodEntry{CPUSet: cpuset.New(1, 3, 7, 9)},
			},
			stAssignments: state.ContainerCPUAssignments{
				"podA": map[string]cpuset.CPUSet{
					"gu-cont":     cpuset.New(1, 7),
					"shared-cont": cpuset.New(3, 9),
				},
				"podB": map[string]cpuset.CPUSet{"cont": cpuset.New(2, 8)},
			},
			stDefaultCPUSet: cpuset.New(0, 4, 5, 6, 10, 11),
			expected: cpuMetricsSnapshot{
				exclusiveCPUs:   6,
				sharedPoolMilli: 5000,
				perNUMA:         map[string]float64{"0": 2, "1": 4},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, testCase.podLevelResourceManagersEnabled)
			logger, _ := ktesting.NewTestContext(t)

			p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), nil)
			st := &mockState{
				assignments:    testCase.stAssignments,
				podAssignments: testCase.stPodAssignments,
				defaultCPUSet:  testCase.stDefaultCPUSet,
			}

			scrambleCPUMetrics()
			p.initializeMetrics(logger, st)

			assertCPUMetrics(t, testCase.expected)
		})
	}
}

func TestStaticPolicyMetricsContainerAllocation(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	allCPUs := cpuset.New(0, 1, 2, 3, 4, 5, 6, 7)
	idle := cpuMetricsSnapshot{
		exclusiveCPUs:   0,
		sharedPoolMilli: 7000,
		perNUMA:         map[string]float64{"0": 0},
	}

	setup := func(t *testing.T) (*staticPolicy, *mockState) {
		p := newMetricsTestPolicy(t, topoSingleSocketHT, 1, cpuset.New(0), nil)
		st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: allCPUs}
		require.NoError(t, p.Start(logger, st))
		assertCPUMetrics(t, idle)
		return p, st
	}

	t.Run("allocation and full release", func(t *testing.T) {
		p, st := setup(t)
		pod := makePod("podUID", "cont", "2", "2")

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})

		require.NoError(t, p.RemoveContainer(logger, st, "podUID", "cont"))
		assertCPUMetrics(t, idle)
	})

	t.Run("init container CPU reuse counted once", func(t *testing.T) {
		p, st := setup(t)
		pod := makeMultiContainerPod(
			[]struct{ request, limit string }{{"4000m", "4000m"}},
			[]struct{ request, limit string }{{"2000m", "2000m"}})

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.InitContainers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 4, sharedPoolMilli: 3000, perNUMA: map[string]float64{"0": 4}})

		// The app container reuses 2 CPUs of the init container, so the gauges must not change.
		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 4, sharedPoolMilli: 3000, perNUMA: map[string]float64{"0": 4}})

		require.NoError(t, p.RemoveContainer(logger, st, "podUID", "initContainer-0"))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})

		require.NoError(t, p.RemoveContainer(logger, st, "podUID", "appContainer-0"))
		assertCPUMetrics(t, idle)
	})

	t.Run("allocation skipped for a container already present in the state", func(t *testing.T) {
		p, st := setup(t)
		pod := makePod("podUID", "cont", "2", "2")

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})
	})

	t.Run("failed allocation leaves the metrics unchanged", func(t *testing.T) {
		p, st := setup(t)
		pod := makePod("podUID", "cont", "9", "9")

		require.Error(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, idle)
	})

	t.Run("release of an unknown pod container leaves the metrics unchanged", func(t *testing.T) {
		p, st := setup(t)

		require.NoError(t, p.RemoveContainer(logger, st, "no-such-pod", "no-such-cont"))
		assertCPUMetrics(t, idle)
	})

	t.Run("release of an unknown container leaves the metrics unchanged", func(t *testing.T) {
		p, st := setup(t)

		pod := makePod("podUID", "cont", "2", "2")

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})

		require.NoError(t, p.RemoveContainer(logger, st, "podUID", "no-such-container"))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 5000, perNUMA: map[string]float64{"0": 2}})
	})
}

func TestStaticPolicyMetricsPodLevelAllocation(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

	logger, _ := ktesting.NewTestContext(t)
	hint := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true}
	allCPUs := cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
	idle := cpuMetricsSnapshot{
		exclusiveCPUs:   0,
		sharedPoolMilli: 11000,
		perNUMA:         map[string]float64{"0": 0, "1": 0},
	}

	setup := func(t *testing.T) (*staticPolicy, *mockState) {
		p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hint)
		st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: allCPUs}
		require.NoError(t, p.Start(logger, st))
		assertCPUMetrics(t, idle)
		return p, st
	}

	t.Run("whole bubble counted on allocation, released with the last container", func(t *testing.T) {
		p, st := setup(t)
		pod := makePodWithContainersAndPodLevelResources("plrm-pod", "4", "4", nil, []containerSpec{
			{name: "gu-container", request: "2", limit: "2"},
			{name: "shared-container"},
		})

		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		busy := cpuMetricsSnapshot{
			exclusiveCPUs:   4,
			sharedPoolMilli: 7000,
			perNUMA:         map[string]float64{"0": 4, "1": 0},
		}
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "gu-container"))
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "shared-container"))
		assertCPUMetrics(t, idle)
	})

	t.Run("failed bubble partitioning rolls the allocation back", func(t *testing.T) {
		p, st := setup(t)
		// The containers request more exclusive CPUs than the pod-level
		// budget, so partitioning the bubble must fail after the bubble has
		// been allocated.
		pod := makePodWithContainersAndPodLevelResources("rollback-pod", "3", "3", nil, []containerSpec{
			{name: "gu-container-1", request: "2", limit: "2"},
			{name: "gu-container-2", request: "2", limit: "2"},
		})

		require.Error(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		assertCPUMetrics(t, idle)
		require.True(t, st.GetDefaultCPUSet().Equals(allCPUs), "default CPU set should be restored, got %s", st.GetDefaultCPUSet())
		_, hasPodCPUSet := st.GetPodCPUSet("rollback-pod")
		require.False(t, hasPodCPUSet, "pod-level CPU set should be removed")
		require.Empty(t, st.GetCPUAssignments(), "container assignments should be removed")
	})

	t.Run("rollback releases the container assignments already present in the state", func(t *testing.T) {
		p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hint)
		bubble := cpuset.New(2, 4, 8, 10)
		st := &mockState{
			assignments: state.ContainerCPUAssignments{
				"plrm-pod": {
					"gu-container":     cpuset.New(2, 8),
					"shared-container": cpuset.New(4, 10),
				},
			},
			podAssignments: state.PodCPUAssignments{
				"plrm-pod": state.PodEntry{CPUSet: bubble},
			},
			defaultCPUSet: allCPUs.Difference(bubble),
		}
		require.NoError(t, p.Start(logger, st))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 4, sharedPoolMilli: 7000, perNUMA: map[string]float64{"0": 4, "1": 0}})

		p.releasePodAllocation(logger, st, "plrm-pod", bubble)

		assertCPUMetrics(t, idle)
		require.True(t, st.GetDefaultCPUSet().Equals(allCPUs), "default CPU set should be restored, got %s", st.GetDefaultCPUSet())
		_, hasPodCPUSet := st.GetPodCPUSet("plrm-pod")
		require.False(t, hasPodCPUSet, "pod-level CPU set should be removed")
		require.Empty(t, st.GetCPUAssignments(), "container assignments should be removed")
	})

	t.Run("pod with non-integral pod-level CPUs leaves the metrics unchanged", func(t *testing.T) {
		p, st := setup(t)
		pod := makePodWithContainersAndPodLevelResources("shared-pod", "1500m", "1500m", nil, []containerSpec{
			{name: "container"},
		})

		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		assertCPUMetrics(t, idle)
	})

	t.Run("pod with non-integral pod-level CPUs and an integral container", func(t *testing.T) {
		p, st := setup(t)
		pod := makePodWithContainersAndPodLevelResources("mixed-pod", "3500m", "3500m", nil, []containerSpec{
			{name: "gu-container", request: "2", limit: "2"},
			{name: "shared-container"},
		})

		// No bubble: non-integral pod-level CPUs make the pod ineligible for a pod-scope allocation.
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		assertCPUMetrics(t, idle)
		_, hasPodCPUSet := st.GetPodCPUSet("mixed-pod")
		require.False(t, hasPodCPUSet, "no pod-level CPU set should be allocated")

		// The integral container still gets a node-scope exclusive allocation through the container-scope path.
		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 9000, perNUMA: map[string]float64{"0": 2, "1": 0}})

		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[1], lifecycle.AddOperation))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 9000, perNUMA: map[string]float64{"0": 2, "1": 0}})

		require.NoError(t, p.RemoveContainer(logger, st, "mixed-pod", "gu-container"))
		assertCPUMetrics(t, idle)
	})
}

func TestStaticPolicyMetricsPodLevelRestore(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

	logger, _ := ktesting.NewTestContext(t)
	hint := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true}
	allCPUs := cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
	bubble := cpuset.New(2, 4, 8, 10)
	idle := cpuMetricsSnapshot{
		exclusiveCPUs:   0,
		sharedPoolMilli: 11000,
		perNUMA:         map[string]float64{"0": 0, "1": 0},
	}
	busy := cpuMetricsSnapshot{
		exclusiveCPUs:   4,
		sharedPoolMilli: 7000,
		perNUMA:         map[string]float64{"0": 4, "1": 0},
	}
	pod := makePodWithContainersAndPodLevelResources("plrm-pod", "4", "4", nil, []containerSpec{
		{name: "gu-container", request: "2", limit: "2"},
		{name: "shared-container"},
	})

	newPolicyWithState := func(t *testing.T, assignments state.ContainerCPUAssignments, podAssignments state.PodCPUAssignments, defaultCPUSet cpuset.CPUSet) (*staticPolicy, *mockState) {
		p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hint)
		st := &mockState{assignments: assignments, podAssignments: podAssignments, defaultCPUSet: defaultCPUSet}
		require.NoError(t, p.Start(logger, st))
		return p, st
	}

	t.Run("re-admission with a fully restored allocation leaves the metrics unchanged", func(t *testing.T) {
		p, st := newPolicyWithState(t, state.ContainerCPUAssignments{}, nil, allCPUs)
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		assertCPUMetrics(t, busy)
		defaultBefore := st.GetDefaultCPUSet()

		scrambleCPUMetrics()
		p.initializeMetrics(logger, st)
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		assertCPUMetrics(t, busy)
		require.True(t, st.GetDefaultCPUSet().Equals(defaultBefore), "default CPU set should be unchanged, got %s", st.GetDefaultCPUSet())
	})

	t.Run("partially restored allocation is completed without changing the metrics", func(t *testing.T) {
		p, st := newPolicyWithState(t,
			state.ContainerCPUAssignments{"plrm-pod": {"gu-container": cpuset.New(2, 8)}},
			state.PodCPUAssignments{"plrm-pod": state.PodEntry{CPUSet: bubble}},
			allCPUs.Difference(bubble))
		assertCPUMetrics(t, busy)

		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		assertCPUMetrics(t, busy)
		cset, ok := st.GetCPUSet("plrm-pod", "shared-container")
		require.True(t, ok, "missing container assignment should be completed")
		require.True(t, cset.Equals(cpuset.New(4, 10)), "shared container should get the pod shared pool, got %s", cset)
	})

	t.Run("failed restore releases the pod CPUs and the metrics", func(t *testing.T) {
		p, st := newPolicyWithState(t,
			state.ContainerCPUAssignments{"plrm-pod": {"gu-container-1": cpuset.New(2, 8)}},
			state.PodCPUAssignments{"plrm-pod": state.PodEntry{CPUSet: bubble}},
			allCPUs.Difference(bubble))
		assertCPUMetrics(t, busy)

		// Inconsistent checkpoint: the pod-level CPU set is too small to fit
		// the assignment of gu-container-2 next to the restored assignment of
		// gu-container-1, so completing the restored allocation must fail.
		pod := makePodWithContainersAndPodLevelResources("plrm-pod", "5", "5", nil, []containerSpec{
			{name: "gu-container-1", request: "2", limit: "2"},
			{name: "gu-container-2", request: "3", limit: "3"},
		})
		require.Error(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		assertCPUMetrics(t, idle)
		require.True(t, st.GetDefaultCPUSet().Equals(allCPUs), "default CPU set should be restored, got %s", st.GetDefaultCPUSet())
		_, hasPodCPUSet := st.GetPodCPUSet("plrm-pod")
		require.False(t, hasPodCPUSet, "pod-level CPU set should be removed")
		require.Empty(t, st.GetCPUAssignments(), "container assignments should be removed")
	})

	t.Run("stale pod-level CPU set overlapping the default CPU set is dropped before a fresh allocation", func(t *testing.T) {
		p, st := newPolicyWithState(t, state.ContainerCPUAssignments{},
			state.PodCPUAssignments{"plrm-pod": state.PodEntry{CPUSet: bubble}},
			allCPUs)
		// The interrupted release left the bubble both in the pod-level entry
		// and in the default CPU set, so the seeded gauges report it as
		// exclusively allocated while the shared pool is already full.
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 4, sharedPoolMilli: 11000, perNUMA: map[string]float64{"0": 4, "1": 0}})

		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		assertCPUMetrics(t, busy)
	})

	t.Run("stale pod-level CPU set is dropped from the metrics even when the fresh allocation fails", func(t *testing.T) {
		p, st := newPolicyWithState(t, state.ContainerCPUAssignments{},
			state.PodCPUAssignments{"plrm-pod": state.PodEntry{CPUSet: bubble}},
			allCPUs)
		hugePod := makePodWithContainersAndPodLevelResources("plrm-pod", "12", "12", nil, []containerSpec{
			{name: "gu-container", request: "12", limit: "12"},
		})

		require.Error(t, p.AllocatePod(logger, st, hugePod, lifecycle.AddOperation))

		assertCPUMetrics(t, idle)
		_, hasPodCPUSet := st.GetPodCPUSet("plrm-pod")
		require.False(t, hasPodCPUSet, "stale pod-level CPU set should be removed")
	})

	t.Run("stale non-prefix assignments are released before a fresh allocation", func(t *testing.T) {
		p, st := newPolicyWithState(t,
			state.ContainerCPUAssignments{"plrm-pod": {"shared-container": cpuset.New(4, 10)}},
			state.PodCPUAssignments{"plrm-pod": state.PodEntry{CPUSet: bubble}},
			allCPUs.Difference(bubble))
		assertCPUMetrics(t, busy)

		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))

		// Only the fresh bubble is accounted, the stale allocation is gone.
		assertCPUMetrics(t, busy)
		cset, ok := st.GetCPUSet("plrm-pod", "gu-container")
		require.True(t, ok, "fresh allocation should assign the guaranteed container")
		require.True(t, cset.Equals(cpuset.New(2, 8)), "unexpected guaranteed container assignment %s", cset)
	})
}

func TestStaticPolicyMetricsPerNUMA(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	hintNUMA0 := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true}
	hintNUMA1 := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(1), Preferred: true}

	p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hintNUMA0)
	st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}
	require.NoError(t, p.Start(logger, st))
	idle := cpuMetricsSnapshot{
		exclusiveCPUs:   0,
		sharedPoolMilli: 11000,
		perNUMA:         map[string]float64{"0": 0, "1": 0},
	}
	assertCPUMetrics(t, idle)

	podA := makePod("podA", "contA", "2", "2")
	require.NoError(t, p.Allocate(logger, st, podA, &podA.Spec.Containers[0], lifecycle.AddOperation))
	assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 9000, perNUMA: map[string]float64{"0": 2, "1": 0}})

	p.affinity = topologymanager.NewFakeManagerWithHint(logger, hintNUMA1)
	podB := makePod("podB", "contB", "2", "2")
	require.NoError(t, p.Allocate(logger, st, podB, &podB.Spec.Containers[0], lifecycle.AddOperation))
	assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 4, sharedPoolMilli: 7000, perNUMA: map[string]float64{"0": 2, "1": 2}})

	// The NUMA node 0 series must drop to zero, not disappear or keep the last value.
	require.NoError(t, p.RemoveContainer(logger, st, "podA", "contA"))
	assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 2, sharedPoolMilli: 9000, perNUMA: map[string]float64{"0": 0, "1": 2}})

	require.NoError(t, p.RemoveContainer(logger, st, "podB", "contB"))
	assertCPUMetrics(t, idle)
}

func TestStaticPolicyMetricsRestartConsistency(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	simulateRestart := func(t *testing.T, p *staticPolicy, st state.State) {
		t.Helper()
		scrambleCPUMetrics()
		p.initializeMetrics(logger, st)
	}

	t.Run("container-scope allocations with init container reuse", func(t *testing.T) {
		p := newMetricsTestPolicy(t, topoSingleSocketHT, 1, cpuset.New(0), nil)
		st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7)}
		require.NoError(t, p.Start(logger, st))

		pod := makeMultiContainerPod(
			[]struct{ request, limit string }{{"4000m", "4000m"}},
			[]struct{ request, limit string }{{"2000m", "2000m"}})
		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.InitContainers[0], lifecycle.AddOperation))
		require.NoError(t, p.Allocate(logger, st, pod, &pod.Spec.Containers[0], lifecycle.AddOperation))
		busy := readCPUMetrics(t)

		simulateRestart(t, p, st)
		assertCPUMetrics(t, busy)
	})

	t.Run("pod-level allocation partially released before the restart", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

		hint := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true}
		p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hint)
		st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}
		require.NoError(t, p.Start(logger, st))

		pod := makePodWithContainersAndPodLevelResources("plrm-pod", "4", "4", nil, []containerSpec{
			{name: "gu-container", request: "2", limit: "2"},
			{name: "shared-container"},
		})
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		busy := cpuMetricsSnapshot{
			exclusiveCPUs:   4,
			sharedPoolMilli: 7000,
			perNUMA:         map[string]float64{"0": 4, "1": 0},
		}
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "gu-container"))

		// The whole bubble must still be accounted after the restart, even though some container assignments are gone.
		simulateRestart(t, p, st)
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "shared-container"))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 0, sharedPoolMilli: 11000, perNUMA: map[string]float64{"0": 0, "1": 0}})
	})

	t.Run("pod-level allocation restored after a restart and released to idle", func(t *testing.T) {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResources, true)
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, pkgfeatures.PodLevelResourceManagers, true)

		hint := &topologymanager.TopologyHint{NUMANodeAffinity: newNUMAAffinity(0), Preferred: true}
		p := newMetricsTestPolicy(t, topoDualSocketHT, 1, cpuset.New(0), hint)
		st := &mockState{assignments: state.ContainerCPUAssignments{}, defaultCPUSet: cpuset.New(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)}
		require.NoError(t, p.Start(logger, st))

		pod := makePodWithContainersAndPodLevelResources("plrm-pod", "4", "4", nil, []containerSpec{
			{name: "gu-container", request: "2", limit: "2"},
			{name: "shared-container"},
		})
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		busy := cpuMetricsSnapshot{
			exclusiveCPUs:   4,
			sharedPoolMilli: 7000,
			perNUMA:         map[string]float64{"0": 4, "1": 0},
		}
		assertCPUMetrics(t, busy)

		simulateRestart(t, p, st)
		require.NoError(t, p.AllocatePod(logger, st, pod, lifecycle.AddOperation))
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "gu-container"))
		assertCPUMetrics(t, busy)

		require.NoError(t, p.RemoveContainer(logger, st, "plrm-pod", "shared-container"))
		assertCPUMetrics(t, cpuMetricsSnapshot{exclusiveCPUs: 0, sharedPoolMilli: 11000, perNUMA: map[string]float64{"0": 0, "1": 0}})
	})
}
