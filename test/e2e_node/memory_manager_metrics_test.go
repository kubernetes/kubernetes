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
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Memory Manager Metrics", framework.WithSerial(), feature.MemoryManager, func() {
	f := framework.NewDefaultFramework("memorymanager-metrics")
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

			count := printAllPodsOnNode(ctx, f.ClientSet, framework.TestContext.NodeName)
			gomega.Expect(count).To(gomega.BeZero(), "unexpected pods on %q, please check output above", framework.TestContext.NodeName)
		})

		ginkgo.It("should report zero pinning counters after a fresh restart", func(ctx context.Context) {
			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume no one else but us is running pods.
			ginkgo.By("Checking the memorymanager metrics right after the kubelet restart, with no pods running")

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0), // intentionally use stricter value
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0), // intentionally use stricter value
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
				}),
			)

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

			values, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err, "error getting the kubelet metrics for sanity check")
			err = validateMetrics(
				values,
				"kubelet_memory_manager_pinning_requests_total",
				"kubelet_memory_manager_pinning_errors_total",
				func(totVal, errVal float64) error {
					if int64(totVal) != int64(errVal) {
						return fmt.Errorf("expected total requests equal to total errors")
					}
					return nil
				},
			)
			framework.ExpectNoError(err, "error validating the kubelet metrics between each other")
		})

		ginkgo.It("should not report any pinning failures when the memorymanager allocation is expected to succeed", func(ctx context.Context) {
			ginkgo.By("Creating the test pod")
			testPod = e2epod.NewPodClient(f).Create(ctx, makeMemoryManagerPod("memmngrpod", nil,
				[]memoryManagerCtnAttributes{
					{
						ctnName: "memmngrcnt",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}),
			)

			printAllPodsOnNode(ctx, f.ClientSet, framework.TestContext.NodeName)

			// we updated the kubelet config in BeforeEach, so we can assume we start fresh.
			// being [Serial], we can also assume noone else but us is running pods.
			ginkgo.By("Checking the memorymanager metrics right after the kubelet restart, with pod should be admitted")
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			ginkgo.By("Giving the Kubelet time to start up and produce metrics")
			gomega.Eventually(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(getKubeletMetrics, 1*time.Minute, 15*time.Second).WithContext(ctx).Should(matchResourceMetrics)

			values, err := getKubeletMetrics(ctx)
			framework.ExpectNoError(err, "error getting the kubelet metrics for sanity check")
			err = validateMetrics(
				values,
				"kubelet_memory_manager_pinning_requests_total",
				"kubelet_memory_manager_pinning_errors_total",
				func(totVal, errVal float64) error {
					if int64(totVal-errVal) < 1 {
						return fmt.Errorf("expected total requests equal to total errors + 1")
					}
					return nil
				},
			)
			framework.ExpectNoError(err, "error validating the kubelet metrics between each other")
		})
	})
})

func validateMetrics(values e2emetrics.KubeletMetrics, totalKey, errorKey string, checkFn func(totVal, errVal float64) error) error {
	totalSamples := values[totalKey]
	errorSamples := values[errorKey]
	if len(totalSamples) != len(errorSamples) {
		return fmt.Errorf("inconsistent samples, total=%d error=%d", len(totalSamples), len(errorSamples))
	}
	for idx := range totalSamples {
		if err := checkFn(float64(totalSamples[idx].Value), float64(errorSamples[idx].Value)); err != nil {
			return err
		}
	}
	return nil
}

// printAllPodsOnNode outputs status of all kubelet pods into log.
// Note considering the e2e_node environment we will always have exactly 1 node, but still.
func printAllPodsOnNode(ctx context.Context, c clientset.Interface, nodeName string) int {
	nodeSelector := fields.Set{
		"spec.nodeName": nodeName,
	}.AsSelector().String()

	podList, err := c.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{
		FieldSelector: nodeSelector,
	})
	if err != nil {
		framework.Logf("Unable to retrieve pods for node %v: %v", nodeName, err)
		return 0
	}
	count := 0
	framework.Logf("begin listing pods: %d found", len(podList.Items))
	for _, p := range podList.Items {
		framework.Logf("%s/%s node %s (expected: %s) status %v QoS %s message %s reason %s (%d container statuses recorded)",
			p.Namespace, p.Name, p.Spec.NodeName, nodeName, p.Status.Phase, p.Status.QOSClass, p.Status.Message, p.Status.Reason, len(p.Status.ContainerStatuses))
		for _, c := range p.Status.ContainerStatuses {
			framework.Logf("\tContainer %v ready: %v, restart count %v",
				c.Name, c.Ready, c.RestartCount)
		}
		count++
	}
	framework.Logf("end listing pods: %d found", len(podList.Items))
	return count
}
