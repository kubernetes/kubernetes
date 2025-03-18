/*
Copyright 2023 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Topology Manager Metrics", framework.WithSerial(), feature.TopologyManager, func() {
	f := framework.NewDefaultFramework("topologymanager-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod
		var cpusNumPerNUMA, coresNumPerNUMA, numaNodes, threadsPerCore int

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			numaNodes, coresNumPerNUMA, threadsPerCore = hostCheck()
			cpusNumPerNUMA = coresNumPerNUMA * threadsPerCore

			// It is safe to assume that the CPUs are distributed equally across
			// NUMA nodes and therefore number of CPUs on all NUMA nodes are same
			// so we just check the CPUs on the first NUMA node

			framework.Logf("numaNodes on the system %d", numaNodes)
			framework.Logf("Cores per NUMA on the system %d", coresNumPerNUMA)
			framework.Logf("Threads per Core on the system %d", threadsPerCore)
			framework.Logf("CPUs per NUMA on the system %d", cpusNumPerNUMA)

			policy := topologymanager.PolicySingleNumaNode
			scope := podScopeTopology

			newCfg, _ := configureTopologyManagerInKubelet(oldCfg, policy, scope, nil, nil, 0)
			updateKubeletConfig(ctx, f, newCfg, true)

		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				deletePodSyncByName(ctx, f, testPod.Name)
			}
			updateKubeletConfig(ctx, f, oldCfg, true)
		})

		ginkgo.It("should report zero admission counters after a fresh restart", func(ctx context.Context) {
			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the topologymanager metrics right after the kubelet restart, with no pods running")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::numa_node": timelessSample(0),
					"pod::numa_node":       timelessSample(0),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report admission failures when the topology manager alignment is known to fail", func(ctx context.Context) {
			ginkgo.By("Creating the test pod which will be rejected for TopologyAffinity")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("topology-affinity-err", cpusNumPerNUMA+1))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the topologymanager metrics right after the kubelet restart, with pod failed to admit")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::numa_node": timelessSample(0),
					"pod::numa_node":       timelessSample(1),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": checkMetricValueGreaterThan(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should not report any admission failures when the topology manager alignment is expected to succeed", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("topology-alignment-ok", cpusNumPerNUMA))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the topologymanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_topology_manager_admission_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_topology_manager_admission_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::numa_node": timelessSample(0),
					"pod::numa_node":       timelessSample(0),
				}),
				"kubelet_topology_manager_admission_duration_ms_count": gstruct.MatchElements(nodeID, gstruct.IgnoreExtras, gstruct.Elements{
					"": checkMetricValueGreaterThan(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 2*time.Minute, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("[alignment] should return updated alignment counters when pod successfully run", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("count-align-numa-ok", cpusNumPerNUMA))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchAlignmentMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchAllElements(idFn, gstruct.Elements{
					"container::numa_node": timelessSample(0),
					"pod::numa_node":       timelessSample(1),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::numa_node": timelessSample(0),
					"pod::numa_node":       timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to update the alignment metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about alignment metrics a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
		})
	})
})

func hostCheck() (int, int, int) {
	// this is a very rough check. We just want to rule out system that does NOT have
	// multi-NUMA nodes or at least 4 cores

	numaNodes := detectNUMANodes()
	if numaNodes < minNumaNodes {
		e2eskipper.Skipf("this test is intended to be run on a multi-node NUMA system")
	}

	coreCount := detectCoresPerSocket()
	if coreCount < minCoreCount {
		e2eskipper.Skipf("this test is intended to be run on a system with at least %d cores per socket", minCoreCount)
	}

	threadsPerCore := detectThreadPerCore()

	return numaNodes, coreCount, threadsPerCore
}

func checkMetricValueGreaterThan(value interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		// We already check Metric when matching the Id
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically(">", value),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}
