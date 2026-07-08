//go:build windows

/*
Copyright 2026 The Kubernetes Authors.

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

package e2enodewindows

import (
	"context"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	windowsTMMinNUMANodes = 1
	windowsTMMinCoreCount = 4
)

var _ = SIGWindowsDescribe(feature.TopologyManager, feature.Windows, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("topologymanager-metrics-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod
		var minCPUsPerNUMA int
		var maxCPUsPerNUMA int

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			minCPUsPerNUMA, maxCPUsPerNUMA = windowsHostCheckForTopologyMetrics()

			newCfg := buildWindowsCPUManagerKubeletConfig(oldCfg, true)
			newCfg.TopologyManagerPolicy = topologymanager.PolicySingleNumaNode
			newCfg.TopologyManagerScope = topologymanager.PodTopologyScope
			updateWindowsKubeletConfig(ctx, f, newCfg)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				testPod = nil
			}
			updateWindowsKubeletConfig(ctx, f, oldCfg)
		})

		ginkgo.It("should report zero admission counters after a fresh restart", func(ctx context.Context) {
			// Note: the kubelet_container_aligned_compute_resources_* series with
			// boundary="numa_node" are created lazily. The topology manager's
			// metric initialization runs during container-manager construction,
			// which happens before the kubelet registers its metrics
			// (metrics.Register in initializeModules). component-base counters
			// return a no-op for WithLabelValues before registration, so those
			// numa_node series only appear after the first real admission. On a
			// truly fresh restart with no pods they are legitimately absent, so we
			// only assert the admission counters here (plain counters that read 0
			// immediately after registration).
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report admission failures when the topology manager alignment is known to fail", func(ctx context.Context) {
			cpuRequest := maxCPUsPerNUMA + 1

			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusivePausePodWindows("topology-affinity-err", cpuRequest))

			idFn := makeCustomPairID("scope", "boundary")
			// TopologyManagerScope is "pod", so only the pod-scope numa_node series is
			// touched at admission time. The container-scope numa_node series is never
			// created here (and the startup zero-init is a no-op because it runs before
			// metrics.Register), so we only assert the pod-scope failure series.
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"pod::numa_node": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": checkMetricValueGreaterThan(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should not report any admission failures when the topology manager alignment is expected to succeed", func(ctx context.Context) {
			// Request one fewer than minCPUsPerNUMA so the guaranteed pod always
			// fits within a single NUMA node's allocatable CPUs. On a single-NUMA
			// host minCPUsPerNUMA equals the whole node, and one CPU is reserved
			// (reservedSystemCPUs), so requesting minCPUsPerNUMA would exceed
			// allocatable and never be admitted.
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusivePausePodWindows("topology-alignment-ok", minCPUsPerNUMA-1))

			idFn := makeCustomPairID("scope", "boundary")
			// Pod scope: a successful admission increments the pod-scope numa_node
			// aligned count. The failure series are not created on the success path
			// (and the startup zero-init no-ops before registration), so assert the
			// admission counters plus the pod-scope aligned count.
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"pod::numa_node": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": checkMetricValueGreaterThan(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

	})
})

func windowsHostCheckForTopologyMetrics() (int, int) {
	perNodeCPUCounts := windowsDetectCPUsPerNUMANode()
	numaNodes := len(perNodeCPUCounts)
	if numaNodes < windowsTMMinNUMANodes {
		e2eskipper.Skipf("this test is intended to be run on a multi-node NUMA system")
	}

	totalLogicalCPUs := windowsDetectTotalLogicalCPUs()
	if totalLogicalCPUs < windowsTMMinCoreCount*windowsTMMinNUMANodes {
		e2eskipper.Skipf("this test is intended to be run on a system with at least %d logical CPUs", windowsTMMinCoreCount*windowsTMMinNUMANodes)
	}

	minCPUsPerNUMA := totalLogicalCPUs
	maxCPUsPerNUMA := 0
	for nodeID, cpus := range perNodeCPUCounts {
		if cpus < 1 {
			e2eskipper.Skipf("NUMA node %d has no online CPUs; skipping", nodeID)
		}
		if cpus < minCPUsPerNUMA {
			minCPUsPerNUMA = cpus
		}
		if cpus > maxCPUsPerNUMA {
			maxCPUsPerNUMA = cpus
		}
	}
	if minCPUsPerNUMA < 1 || maxCPUsPerNUMA < 1 {
		e2eskipper.Skipf("unable to derive per-NUMA CPU counts: %v", perNodeCPUCounts)
	}

	framework.Logf("numaNodes on the system %d", numaNodes)
	framework.Logf("logical CPUs on the system %d", totalLogicalCPUs)
	framework.Logf("Per-NUMA CPU counts on the system %v", perNodeCPUCounts)
	framework.Logf("Minimum CPUs per NUMA on the system %d", minCPUsPerNUMA)
	framework.Logf("Maximum CPUs per NUMA on the system %d", maxCPUsPerNUMA)

	return minCPUsPerNUMA, maxCPUsPerNUMA
}

func windowsDetectCPUsPerNUMANode() map[int]int {
	outData, err := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", "Get-CimInstance Win32_PerfRawData_Counters_ProcessorInformation | Select-Object -ExpandProperty Name").Output()
	framework.ExpectNoError(err)

	perNodeCPUCounts := map[int]int{}
	for _, line := range strings.Split(strings.TrimSpace(string(outData)), "\n") {
		name := strings.TrimSpace(line)
		if name == "" || name == "_Total" || strings.HasSuffix(name, "_Total") {
			continue
		}
		parts := strings.Split(name, ",")
		if len(parts) != 2 {
			continue
		}
		nodeID, err := strconv.Atoi(strings.TrimSpace(parts[0]))
		if err != nil {
			continue
		}
		cpuIDPart := strings.TrimSpace(parts[1])
		if cpuIDPart == "_Total" {
			continue
		}
		if _, err := strconv.Atoi(cpuIDPart); err != nil {
			continue
		}
		perNodeCPUCounts[nodeID]++
	}

	return perNodeCPUCounts
}

func windowsDetectTotalLogicalCPUs() int {
	outData, err := exec.Command("powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum").Output()
	framework.ExpectNoError(err)

	logicalCPUs, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return logicalCPUs
}

func makeGuaranteedCPUExclusivePausePodWindows(name string, cpus int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name + "-pod",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  name + "-cnt",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", cpus)),
							v1.ResourceMemory: resource.MustParse("64Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse(fmt.Sprintf("%d", cpus)),
							v1.ResourceMemory: resource.MustParse("64Mi"),
						},
					},
				},
			},
		},
	}
}

func getWindowsKubeletMetrics(ctx context.Context) (e2emetrics.KubeletMetrics, error) {
	return e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", nodeNameOrIP(), ports.KubeletReadOnlyPort), "/metrics")
}

func nodeID(_ interface{}) string {
	return ""
}

func makeCustomPairID(pri, sec string) func(interface{}) string {
	return func(element interface{}) string {
		el := element.(*testutil.Sample)
		return fmt.Sprintf("%s::%s", el.Metric[testutil.LabelName(pri)], el.Metric[testutil.LabelName(sec)])
	}
}

func timelessSample(value interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically("==", value),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}

func checkMetricValueGreaterThan(value interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically(">", value),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}
