//go:build windows

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

package e2enodewindows

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// Windows CPU manager metrics. The CPU manager pinning/pool metrics live in the
// cross-platform static policy, so they increment on Windows the same way as on
// Linux. They are plain counters/gauges (not label vectors), so they read
// correctly at a fresh restart without the lazy-init caveat that affects the
// topology manager numa_node series.
//
// Only the OS-agnostic Linux cases are mirrored here. The SMT (full-pcpus-only),
// uncore-cache, and allocation-per-NUMA cases are intentionally omitted because
// those options rely on SMT/L3/NUMA-distance topology that the Windows CRI does
// not expose.
var _ = SIGWindowsDescribe(feature.CPUManager, feature.Windows, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("cpumanager-metrics-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			newCfg := buildWindowsCPUManagerKubeletConfig(oldCfg, true)
			// Keep the topology manager out of the admission decision so a guaranteed
			// pod is admitted straight to the CPU manager's Allocate and the pinning /
			// exclusive-allocation counters increment.
			newCfg.TopologyManagerPolicy = topologymanager.PolicyNone
			// Switching the CPU manager policy invalidates the persisted state
			// checkpoint, so clear it before restarting to avoid the kubelet
			// refusing to start on a policy mismatch.
			updateWindowsKubeletConfigClearState(ctx, f, newCfg)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				testPod = nil
			}
			updateWindowsKubeletConfigClearState(ctx, f, oldCfg)
		})

		ginkgo.It("should report zero pinning counters and an idle CPU pool after a fresh restart", func(ctx context.Context) {
			// We restarted the kubelet in BeforeEach and, being [Serial], no other
			// pods are running: no pinning has happened, nothing is exclusively
			// allocated, and the shared pool holds the assignable CPUs.
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_cpu_manager_shared_pool_size_millicores": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": checkMetricValueGreaterThan(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report pinning failures when the cpumanager allocation is known to fail", func(ctx context.Context) {
			// The Linux failure spec provokes an SMTAlignmentError via the
			// full-pcpus-only option, which is unavailable on Windows (no SMT
			// sibling topology through the CRI). Instead, request more whole
			// exclusive CPUs than the node has assignable: the static policy's
			// Allocate runs (topology admit precedes the node-fit predicate) and
			// fails with "not enough cpus available", incrementing the generic
			// pinning request/error counters. The physical_cpu aligned-failure
			// series is intentionally not asserted since it only increments under
			// full-pcpus-only.
			_, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusivePausePodWindows("cpu-pin-fail", int(cpuAlloc)+1))

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report pinning success and exclusive allocation when a guaranteed pod runs", func(ctx context.Context) {
			// A Guaranteed pod requesting whole CPUs is pinned by the static policy:
			// the pinning request counter increments, no error occurs, and the
			// exclusive allocation count reflects the CPUs handed out.
			const exclusiveCPUs = 2
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), exclusiveCPUs)

			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusivePausePodWindows("cpu-pin-ok", exclusiveCPUs))

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(exclusiveCPUs),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})
	})
})
