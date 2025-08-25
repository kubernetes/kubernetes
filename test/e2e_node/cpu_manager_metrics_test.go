/*
Copyright 2022 The Kubernetes Authors.

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
	"fmt"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"
)

var _ = SIGDescribe("CPU Manager Metrics", framework.WithSerial(), feature.CPUManager, func() {
	f := framework.NewDefaultFramework("cpumanager-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod
		var cpuAlloc int64
		var smtLevel int
		var uncoreGroupSize int
		var hasSplitUncore bool

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
			_, cpuAlloc, _ := getLocalNodeCPUDetails(ctx, f)
			smtLevel = smtLevelFromSysFS()

			// strict SMT alignment is trivially verified and granted on non-SMT systems
			if smtLevel < 2 {
				e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
			}

			// our tests want to allocate up to a full core, so we need at last 2*2=4 virtual cpus
			if cpuAlloc < int64(smtLevel*2) {
				e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < 4", fullCPUsOnlyOpt)
			}

			framework.Logf("SMT level %d", smtLevel)

			uncoreGroupSize = getUncoreCPUGroupSize()
			if uncoreGroupSize == 0 {
				hasSplitUncore = false
			} else {
				// check we do physically have split Uncore but also we have enough CPUs available to run
				// meaningful tests. We need them both.
				hasSplitUncore = (cpuAlloc > int64(uncoreGroupSize))
			}

			framework.Logf("Uncore Group Size %d; Split Uncore detected=%v", uncoreGroupSize, hasSplitUncore)

			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.
			cpuPolicyOptions := map[string]string{
				cpumanager.FullPCPUsOnlyOption:            "true",
				cpumanager.PreferAlignByUnCoreCacheOption: strconv.FormatBool(hasSplitUncore),
			}
			newCfg := configureCPUManagerInKubelet(oldCfg,
				&cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      cpuset.New(0),
					enableCPUManagerOptions: true,
					options:                 cpuPolicyOptions,
				},
			)
			updateKubeletConfig(ctx, f, newCfg, true)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				deletePodSyncByName(ctx, f, testPod.Name)
				waitForContainerRemoval(ctx, testPod.Spec.Containers[0].Name, testPod.Name, testPod.Namespace)
			}
			updateKubeletConfig(ctx, f, oldCfg, true)
		})

		ginkgo.It("should report zero pinning counters after a fresh restart", func(ctx context.Context) {
			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with no pods running")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::physical_cpu": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report pinning failures when the cpumanager allocation is known to fail", func(ctx context.Context) {
			ginkgo.By("Creating the test pod which will be rejected for SMTAlignmentError")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("smt-align-err", 1))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod failed to admit")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::physical_cpu": timelessSample(1),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should not report any pinning failures when the cpumanager allocation is expected to succeed", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("smt-align-ok", smtLevel))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::physical_cpu": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should return updated SMT alignment counters when pod successfully run", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("count-align-smt-ok", smtLevel))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchAlignmentMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::physical_cpu": timelessSample(1),
				}),
				"kubelet_container_aligned_compute_resources_failure_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::physical_cpu": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to update the alignment metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about alignment metrics a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
		})

		ginkgo.It("should report the default idle cpu pool size", func(ctx context.Context) {
			ginkgo.By("Querying the podresources endpoint to get the baseline")
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
			defer func() {
				framework.ExpectNoError(conn.Close())
			}()

			ginkgo.By("Checking the pool allocatable resources from the kubelet")
			resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err, "failed to get the kubelet allocatable resources")
			allocatableCPUs, _ := demuxCPUsAndDevicesFromGetAllocatableResources(resp)

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_shared_pool_size_millicores": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(int(allocatableCPUs.Size() * 1000)),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics about idle pool size")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 10*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about idle pool size a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 30*time.Second, 10*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report mutating cpu pool size when handling guaranteed pods", func(ctx context.Context) {
			ginkgo.By("Querying the podresources endpoint to get the baseline")
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
			defer func() {
				framework.ExpectNoError(conn.Close())
			}()

			ginkgo.By("Checking the pool allocatable resources from the kubelet")
			resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err, "failed to get the kubelet allocatable resources")
			allocatableCPUs, _ := demuxCPUsAndDevicesFromGetAllocatableResources(resp)

			allocatableCPUsIdleMillis := int(allocatableCPUs.Size() * 1000)

			matchResourceMetricsIdle := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_shared_pool_size_millicores": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(allocatableCPUsIdleMillis),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})
			ginkgo.By(fmt.Sprintf("Pool allocatable resources from the kubelet: shared pool %d cpus %d millis", allocatableCPUs.Size(), allocatableCPUsIdleMillis))

			ginkgo.By("Giving the Kubelet time to start up and produce metrics about idle pool size")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 10*time.Second).Should(matchResourceMetricsIdle)
			ginkgo.By("Ensuring the metrics match the expectations about idle pool size a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 30*time.Second, 10*time.Second).Should(matchResourceMetricsIdle)

			ginkgo.By("Creating the test pod to consume exclusive cpus from the pool")
			testPod = e2epod.NewPodClient(f).CreateSync(ctx, makeGuaranteedCPUExclusiveSleeperPod("smt-cpupool", smtLevel))

			matchResourceMetricsBusy := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_cpu_manager_shared_pool_size_millicores": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(allocatableCPUsIdleMillis - (smtLevel * 1000)),
				}),
				"kubelet_cpu_manager_exclusive_cpu_allocation_count": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(smtLevel),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 10*time.Second).Should(matchResourceMetricsBusy)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 30*time.Second, 10*time.Second).Should(matchResourceMetricsBusy)

			deletePodSyncByName(ctx, f, testPod.Name)

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 10*time.Second).Should(matchResourceMetricsIdle)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 30*time.Second, 10*time.Second).Should(matchResourceMetricsIdle)
		})

		ginkgo.It("should update alignment counters when pod successfully run taking less than uncore cache group", func(ctx context.Context) {
			if !hasSplitUncore {
				e2eskipper.Skip("Skipping CPU Manager uncore alignment test - not split Uncore detected")
			}
			if smtLevel >= uncoreGroupSize {
				// this doesn't make sense according to the very definition of uncore cache (a cache which spans across core blocks,
				// and thread siblings belong to the same block and they share a exclusive cache block)
				// so it has to be a configuration or detection issue. Fail loudly.
				framework.Failf("Failed preconditions for CPU Manager uncore alignment test - SMT level more than Uncore group size - this is unexpected")
			}

			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("count-align-uncore-ok", smtLevel))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchAlignmentMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::uncore_cache": timelessSample(1),
				}),
			})

			ginkgo.By("Giving the Kubelet time to update the alignment metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about alignment metrics a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
		})

		ginkgo.It("should update alignment counters when pod successfully run taking a full uncore cache group", func(ctx context.Context) {
			if !hasSplitUncore {
				e2eskipper.Skip("Skipping CPU Manager uncore alignment test - not split Uncore detected")
			}

			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("count-align-uncore-ok", uncoreGroupSize))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchAlignmentMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::uncore_cache": timelessSample(1),
				}),
			})

			ginkgo.By("Giving the Kubelet time to update the alignment metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about alignment metrics a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
		})

		ginkgo.It("should not update alignment counters when pod successfully run taking more than a uncore cache group", func(ctx context.Context) {
			if !hasSplitUncore {
				e2eskipper.Skip("Skipping CPU Manager uncore alignment test - not split Uncore detected")
			}
			if cpuAlloc < int64(uncoreGroupSize+smtLevel) {
				e2eskipper.Skipf("Skipping CPU Manager uncore alignment test - not enough available CPUs (needs %d allocatable %d)", uncoreGroupSize+smtLevel, cpuAlloc)
			}

			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("count-align-uncore-ok", uncoreGroupSize+smtLevel))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the cpumanager metrics right after the kubelet restart, with pod should be admitted")

			idFn := makeCustomPairID("scope", "boundary")
			matchAlignmentMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_container_aligned_compute_resources_count": gstruct.MatchElements(idFn, gstruct.IgnoreExtras, gstruct.Elements{
					"container::uncore_cache": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to update the alignment metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
			ginkgo.By("Ensuring the metrics match the expectations about alignment metrics a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchAlignmentMetrics)
		})
		ginkgo.It("should report zero counters for allocation per NUMA after a fresh restart", func(ctx context.Context) {

			cpuPolicyOptions := map[string]string{
				cpumanager.DistributeCPUsAcrossNUMAOption: "true",
				cpumanager.FullPCPUsOnlyOption:            "true",
			}
			newCfg := configureCPUManagerInKubelet(oldCfg,
				&cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      cpuset.New(0),
					enableCPUManagerOptions: true,
					options:                 cpuPolicyOptions,
				},
			)

			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("Checking the cpumanager allocation per NUMA metric right after the kubelet restart, with no pods running")
			numaNodes, _, _, _ := hostCheck()

			framework.Logf("numaNodes on the system %d", numaNodes)

			keys := make(map[interface{}]types.GomegaMatcher)
			idFn := makeCustomLabelID(metrics.AlignedNUMANode)

			for i := 0; i < numaNodes; i++ {
				keys["kubelet_cpu_manager_allocation_per_numa"] = gstruct.MatchAllElements(idFn, gstruct.Elements{
					fmt.Sprintf("%d", i): timelessSample(0),
				})

			}

			matchSpreadMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, keys)

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchSpreadMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchSpreadMetrics)

		})

		ginkgo.It("should report allocation per NUMA metric when handling guaranteed pods", func(ctx context.Context) {
			cpuPolicyOptions := map[string]string{
				cpumanager.DistributeCPUsAcrossNUMAOption: "true",
				cpumanager.FullPCPUsOnlyOption:            "true",
			}
			newCfg := configureCPUManagerInKubelet(oldCfg,
				&cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      cpuset.New(0),
					enableCPUManagerOptions: true,
					options:                 cpuPolicyOptions,
				},
			)

			updateKubeletConfig(ctx, f, newCfg, true)

			numaNodes, _, _, cpusNumPerNUMA := hostCheck()

			smtLevel = smtLevelFromSysFS()
			framework.Logf("SMT Level on the system %d", smtLevel)

			ginkgo.By("Querying the podresources endpoint to get the baseline")
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed err: %v", err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed err: %v", err)
			defer func() {
				framework.ExpectNoError(conn.Close())
			}()

			ginkgo.By("Checking the pool allocatable resources from the kubelet")
			resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err, "failed to get the kubelet allocatable resources")
			allocatableCPUs, _ := demuxCPUsAndDevicesFromGetAllocatableResources(resp)

			// 'distribute-cpus-across-numa' policy option ensures that CPU allocations are evenly distributed
			//  across NUMA nodes in cases where more than one NUMA node is required to satisfy the allocation.
			// So, we want to ensure that the CPU Request exceeds the number of CPUs that can fit within a single
			// NUMA node. We have to pick cpuRequest such that:
			// 1. CPURequest > cpusNumPerNUMA
			// 2. Not occupy all the CPUs on the node ande leave room for reserved CPU
			// 3. CPURequest is a multiple if number of NUMA nodes to allow equal CPU distribution across NUMA nodes
			//
			// In summary: cpusNumPerNUMA < CPURequest < ((cpusNumPerNuma * numaNodes) - reservedCPUscount)
			// Considering all these constraints we select: CPURequest= (cpusNumPerNUMA-smtLevel)*numaNodes
			cpuRequest := (cpusNumPerNUMA - smtLevel) * numaNodes
			if cpuRequest > allocatableCPUs.Size() {
				e2eskipper.Skipf("Pod requesting %d CPUs which is more than allocatable CPUs:%d", cpuRequest, allocatableCPUs.Size())
			}

			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("test-pod-allocation-per-numa", cpuRequest))

			ginkgo.By("Checking the cpumanager metrics after pod creation")

			keys := make(map[interface{}]types.GomegaMatcher)
			idFn := makeCustomLabelID(metrics.AlignedNUMANode)

			// On a clean environment with no other pods running if distribute-across-numa policy option is enabled
			for i := 0; i < numaNodes; i++ {
				keys["kubelet_cpu_manager_allocation_per_numa"] = gstruct.MatchAllElements(idFn, gstruct.Elements{
					fmt.Sprintf("%d", i): timelessSample(2),
				})

			}

			matchSpreadMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, keys)

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchSpreadMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(ctx, getKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchSpreadMetrics)
		})

	})
})

func getKubeletMetrics(ctx context.Context) (e2emetrics.KubeletMetrics, error) {
	ginkgo.By("Getting Kubelet metrics from the metrics API")
	return e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, nodeNameOrIP()+":10255", "/metrics")
}

func makeGuaranteedCPUExclusiveSleeperPod(name string, cpus int) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name + "-pod",
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  name + "-cnt",
					Image: busyboxImage,
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
					Command: []string{"sh", "-c", "sleep 1d"},
				},
			},
		},
	}
}

func timelessSample(value interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		// We already check Metric when matching the Id
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically("==", value),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}
