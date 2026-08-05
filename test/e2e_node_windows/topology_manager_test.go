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
	internalapi "k8s.io/cri-api/pkg/apis"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
 * Windows Topology Manager coordination E2E tests.
 *
 * These tests validate the Windows-specific reconciliation that makes memory
 * placement follow the CPU manager's decision, implemented in
 * pkg/kubelet/cm/internal_container_lifecycle_windows.go (PreCreateContainer +
 * computeFinalCpuSet).
 *
 * The topology manager runs the "best-effort" policy here: it always admits the
 * pod (never rejects) but, unlike "none", it collects and merges the CPU and
 * memory managers' hints so the memory manager allocates on the same NUMA node
 * the CPU manager chose. The two branches of computeFinalCpuSet are exercised:
 *
 *   1. CPU manager allocated exclusive CPUs (Guaranteed + integer CPU): the CPU
 *      allocation is authoritative and used as-is. The final job-object affinity
 *      is exactly the CPU manager's set and is NOT widened to the whole NUMA
 *      node, even though the memory manager pinned memory on that same node.
 *   2. CPU manager allocated nothing (Guaranteed + fractional CPU): the final
 *      affinity is the memory manager's NUMA node CPU set, providing memory
 *      locality through CPU affinity (Windows has no cpuset.mems equivalent).
 *
 * The pure computeFinalCpuSet function (including the cross-NUMA and multi-group
 * cases) is unit-tested in
 * pkg/kubelet/cm/internal_container_lifecycle_windows_test.go; this suite covers
 * the live admission -> PreCreateContainer -> CRI path with both managers active.
 */
var _ = SIGWindowsDescribe(feature.TopologyManager, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("topologymanager-coordination-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("with the topology manager best-effort policy", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var criClient internalapi.RuntimeService
		var testPod *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			criClient, _, err = getCRIClient(ctx)
			framework.ExpectNoError(err, "failed to get CRI client")

			// Enabling the CPU (static) + memory (BestEffort) managers together and
			// switching the topology policy invalidates the persisted manager state
			// checkpoints, so clear them before restarting.
			newCfg := buildWindowsTopologyManagerKubeletConfig(oldCfg, topologymanager.PolicyBestEffort, topologymanager.ContainerTopologyScope)
			updateWindowsKubeletConfigClearState(ctx, f, newCfg)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				testPod = nil
			}
			// Restoring the original policy is another manager-policy change, so it
			// also needs the state checkpoints cleared.
			updateWindowsKubeletConfigClearState(ctx, f, oldCfg)
		})

		ginkgo.It("should use the CPU manager's exclusive CPUs as-is when a guaranteed integer-CPU pod runs", func(ctx context.Context) {
			// Branch 1 of computeFinalCpuSet: the CPU manager allocated exclusive
			// CPUs, so the memory manager's NUMA decision must not widen the final
			// affinity. The pod requests 2 whole CPUs; the job-object affinity must
			// be exactly 2 CPUs, not the full NUMA node.
			const exclusiveCPUs = 2
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), exclusiveCPUs)

			testPod = createGuaranteedTopologyPod(ctx, f, "topo-int", "2000m")

			ginkgo.By("verifying the container is pinned to exactly the CPU manager's exclusive set")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, testPod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(exclusiveCPUs),
				"guaranteed integer-CPU container must be pinned to exactly its exclusive CPUs, not widened to the NUMA node")

			ginkgo.By("verifying the host job-object affinity agrees with the CRI report")
			verifyHostMatchesCRI(ctx, criClient, testPod, "gu-ctr")

			ginkgo.By("verifying the memory manager also pinned memory (following the CPU decision)")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})
			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should set affinity to the memory manager's NUMA node when a guaranteed fractional-CPU pod runs", func(ctx context.Context) {
			// Branch 2 of computeFinalCpuSet: a Guaranteed pod with a fractional CPU
			// request gets no exclusive CPUs from the CPU manager, so the final
			// affinity is the memory manager's NUMA node CPU set. This deterministic
			// check requires a single NUMA node (we cannot read which node the
			// memory manager picked); the multi-NUMA path is covered by unit tests.
			perNodeCPUs := windowsDetectCPUsPerNUMANode()
			if len(perNodeCPUs) != 1 {
				e2eskipper.Skipf("memory-driven affinity assertion requires a single-NUMA node (memory node is deterministic); found %d NUMA nodes", len(perNodeCPUs))
			}
			var expectedNodeCPUs int
			for _, cpus := range perNodeCPUs {
				expectedNodeCPUs = cpus
			}

			testPod = createGuaranteedTopologyPod(ctx, f, "topo-frac", "500m")

			ginkgo.By("verifying no exclusive CPUs were allocated")
			matchNoExclusive := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})
			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchNoExclusive)

			ginkgo.By("verifying the container affinity equals the memory manager's NUMA node CPU set")
			gomega.Eventually(ctx, func(ctx context.Context) (int, error) {
				aff, err := getWindowsContainerCPUAffinity(ctx, criClient, testPod, "gu-ctr")
				if err != nil {
					return 0, err
				}
				return countCPUsInAffinities(aff), nil
			}, 30*time.Second, 2*time.Second).Should(gomega.Equal(expectedNodeCPUs),
				"guaranteed fractional-CPU container must be pinned to the memory manager's NUMA node CPUs")

			ginkgo.By("verifying the memory manager pinned memory")
			matchMemoryPinned := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})
			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchMemoryPinned)
		})

		ginkgo.It("should reject the pod when the memory manager cannot fit its own allocation", func(ctx context.Context) {
			// This exercises the memory manager's OWN-decision path, the only path
			// that can fail on capacity. On Windows the memory manager FOLLOWS the
			// CPU manager's NUMA decision (cpuFollowingStore) when the container owns
			// exclusive CPUs, and in that case it skips hint extension and binds
			// best-effort to the CPU's node (it does not fail here — an oversized
			// request is instead rejected later by the node-fit predicate). To reach
			// the path where the memory manager selects/extends NUMA nodes itself and
			// can return an allocation error, we use a Guaranteed pod with a
			// FRACTIONAL CPU request: it gets no exclusive CPUs, so cpuFollowingStore
			// falls back to the topology hint and the memory manager extends on its
			// own. The oversized memory request fits no NUMA node, so Allocate fails,
			// the pod is rejected, and both memory pinning counters increment.
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedMemoryManagerPodWindows("topo-mem-fail", "500m", "1000Gi"))

			ginkgo.By("verifying the pod is rejected (not admitted to run)")
			gomega.Eventually(ctx, func(ctx context.Context) (v1.PodPhase, error) {
				p, err := e2epod.NewPodClient(f).Get(ctx, testPod.Name, metav1.GetOptions{})
				if err != nil {
					return "", err
				}
				return p.Status.Phase, nil
			}, 1*time.Minute, 5*time.Second).Should(gomega.Equal(v1.PodFailed),
				"a pod the memory manager cannot satisfy on its own-decision path must fail admission")

			ginkgo.By("verifying the memory manager reported a pinning failure")
			matchMemoryPinningFailure := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
			})
			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchMemoryPinningFailure)
		})

		ginkgo.It("should not report a memory pinning failure for a guaranteed integer-CPU pod even when memory cannot fit", func(ctx context.Context) {
			// This is the counterpart to the fractional-CPU failure spec above and
			// exercises the memory manager's FOLLOW-CPU path (the added skip-extend
			// logic in policy_best_effort.go). The pod requests a whole integer CPU,
			// so the CPU manager gives it exclusive CPUs and the memory manager
			// follows that NUMA node via cpuFollowingStore. Because the container is
			// following the CPU manager, the memory manager does NOT extend the hint
			// to other NUMA nodes even though the oversized memory request cannot fit
			// the CPU's node — it binds best-effort and returns success. So, unlike
			// the fractional-CPU case, the memory manager records a pinning REQUEST
			// but NO pinning error; the pod is ultimately rejected by the node-fit
			// predicate for insufficient memory, not by the memory manager.
			//
			// Same oversized memory request as the spec above, only the CPU request
			// differs (integer vs fractional), which is exactly what flips the
			// skip-extend behavior — making this the observable signal of the added
			// logic on a single-NUMA node.
			const exclusiveCPUs = 2
			skipIfAllocatableCPUsLessThan(getLocalNode(ctx, f), exclusiveCPUs)

			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedMemoryManagerPodWindows("topo-mem-follow", "2", "1000Gi"))

			ginkgo.By("verifying the memory manager recorded a pinning request but no pinning error")
			matchNoPinningFailure := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})
			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchNoPinningFailure)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchNoPinningFailure)
		})
	})
})

// buildWindowsTopologyManagerKubeletConfig returns a KubeletConfiguration with
// the CPU manager (static) and memory manager (BestEffort) enabled together
// under the given topology manager policy and scope. It builds on the memory
// manager config (which also sets the reserved-memory numbers the memory
// manager validates at startup) and overrides the topology policy/scope so the
// topology manager acts as the coordinator between the two managers.
func buildWindowsTopologyManagerKubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration, policy, scope string) *kubeletconfig.KubeletConfiguration {
	newCfg := buildWindowsMemoryManagerKubeletConfig(oldCfg)
	newCfg.TopologyManagerPolicy = policy
	newCfg.TopologyManagerScope = scope
	return newCfg
}

// createGuaranteedTopologyPod creates a Guaranteed-QoS pod with a single
// container named "gu-ctr" requesting the given CPU quantity (requests == limits
// for both CPU and memory), waits for it to run, and returns the pod. A whole
// integer cpu (e.g. "2000m") yields exclusive CPUs; a fractional cpu (e.g.
// "500m") is still Guaranteed but gets no exclusive CPUs from the CPU manager.
func createGuaranteedTopologyPod(ctx context.Context, f *framework.Framework, name, cpu string) *v1.Pod {
	pod := makeWindowsCPUManagerPod(name, []windowsCtnAttribute{
		{name: "gu-ctr", cpuRequest: cpu, cpuLimit: cpu},
	})
	return e2epod.NewPodClient(f).CreateSync(ctx, pod)
}
