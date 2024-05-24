//go:build linux

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
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Memory Manager Metrics", framework.WithSerial(), feature.MemoryManager, func() {
	f := framework.NewDefaultFramework("memorymanager-metrics")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var testPod *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			var oldCfg *kubeletconfig.KubeletConfiguration
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			newCfg := oldCfg.DeepCopy()
			updateKubeletConfigWithMemoryManagerParams(newCfg,
				&memoryManagerKubeletParams{
					policy: staticPolicy,
					systemReservedMemory: []kubeletconfig.MemoryReservation{
						{
							NumaNode: 0,
							Limits: v1.ResourceList{
								resourceMemory: resource.MustParse("1100Mi"),
							},
						},
					},
					systemReserved: map[string]string{resourceMemory: "500Mi"},
					kubeReserved:   map[string]string{resourceMemory: "500Mi"},
					evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
				},
			)
			updateKubeletConfig(ctx, f, newCfg, true)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				if testPod != nil {
					deletePodSyncByName(ctx, f, testPod.Name)
				}
				updateKubeletConfig(ctx, f, oldCfg, true)
			})
		})

		ginkgo.It("should report zero pinning counters after a fresh restart", func(ctx context.Context) {
			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume no one else but us is running pods.
			ginkgo.By("Checking the memorymanager metrics right after the kubelet restart, with no pods running")

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
		})

		ginkgo.It("should report pinning failures when the memorymanager allocation is known to fail", func(ctx context.Context) {
			ginkgo.By("Creating the test pod which will be rejected for memory request which is too big")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeMemoryManagerPod("memmngrpod", nil,
				[]memoryManagerCtnAttributes{
					{
						ctnName: "memmngrcnt",
						cpus:    "100m",
						memory:  "1000Gi"},
				}))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the memorymanager metrics right after the kubelet restart, with pod failed to admit")

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
		})

		ginkgo.It("should not report any pinning failures when the memorymanager allocation is expected to succeed", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeMemoryManagerPod("memmngrpod", nil,
				[]memoryManagerCtnAttributes{
					{
						ctnName: "memmngrcnt",
						cpus:    "100m",
						memory:  "64Mi"},
				}))

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the memorymanager metrics right after the kubelet restart, with pod should be admitted")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_cpu_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
		})
	})
})
