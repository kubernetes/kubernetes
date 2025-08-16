/*
Copyright 2025 The Kubernetes Authors.

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

package windows

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = sigDescribe(feature.Windows, "Kubelet metrics on Windows node",
	framework.WithSerial(),
	framework.WithSlow(),
	framework.WithDisruptive(),
	framework.WithFeatureGate(kubefeatures.PodAndContainerStatsFromCRI),
	func() {
		f := framework.NewDefaultFramework("windows-metrics")
		f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

		var (
			c, ec   clientset.Interface
			grabber *e2emetrics.Grabber
			node    v1.Node
		)

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			c = f.ClientSet
			ec = f.KubemarkExternalClusterClientSet

			ginkgo.By("Grabbing metrics endpoints")
			gomega.Eventually(ctx, func() error {
				grabber, err = e2emetrics.NewMetricsGrabber(ctx, c, ec, f.ClientConfig(), true, true, true, true, true, true)
				return err
			}, 5*time.Minute, 10*time.Second).Should(gomega.Succeed())

			ginkgo.By("Selecting a Windows node")

			node, err = findWindowsNode(ctx, f)
			framework.ExpectNoError(err)

			ginkgo.By("Scheduling test pods")
			powershellImage := imageutils.GetConfig(imageutils.BusyBox)
			pods := newKubeletStatsTestPods(2, powershellImage, node.Name)
			e2epod.NewPodClient(f).CreateBatch(ctx, pods)

			ginkgo.By("Waiting for pods to be running")
			timeout := 3 * time.Minute
			err = e2epod.WaitForPodsRunningReady(ctx, f.ClientSet, f.Namespace.Name, 2, timeout)
			framework.ExpectNoError(err)
		})

		ginkgo.It("validates /stats/summary", func(ctx context.Context) {
			ginkgo.By("Grabbing /stats/summary")
			response, err := grabber.GrabSummaryFromNode(ctx, node.Name)
			framework.ExpectNoError(err)
			gomega.Expect(response).NotTo(gomega.BeEmpty())

			summary, err := parseSummaryOutput(response)
			framework.ExpectNoError(err)

			gomega.Expect(summary.Node).NotTo(gomega.BeNil(), "Node summary should not be nil")

			// validate the metrics for the pod's memory
			for _, pod := range summary.Pods {
				if !strings.HasPrefix(pod.PodRef.Name, "statscollectiontest") {
					continue // Skip pods not created by this test
				}

				gomega.Expect(pod.Containers).To(gomega.HaveLen(1), "Expected 1 container in each pod")
				gomega.Expect(pod.Memory).NotTo(gomega.BeNil(), "Pod memory metrics should not be nil")
				gomega.Expect(pod.Memory.WorkingSetBytes).NotTo(gomega.BeNil(), "Pod working set bytes should not be nil")
				gomega.Expect(*pod.Memory.WorkingSetBytes).To(gomega.BeNumerically(">", 0), "Pod working set bytes should be greater than 0")
				// TODO: Remove the following check when containerd fix is released
				if pod.Memory.AvailableBytes != nil {
					framework.Logf("Pod %s has memory metrics: available %d bytes", pod.PodRef.Name, *pod.Memory.AvailableBytes)
					gomega.Expect(*pod.Memory.AvailableBytes).To(gomega.BeNumerically(">", 0), "Pod available bytes should be greater than 0")
				}

				for _, container := range pod.Containers {
					gomega.Expect(container.Memory).NotTo(gomega.BeNil(), "Container memory metrics should not be nil")
					gomega.Expect(container.Memory.WorkingSetBytes).NotTo(gomega.BeNil(), "WorkingSetBytes should not be nil")
					gomega.Expect(*container.Memory.WorkingSetBytes).To(gomega.BeNumerically(">", 0), "WorkingSetBytes should be greater than 0")
					// TODO: Remove the following check when containerd fix is released
					if container.Memory.AvailableBytes != nil {
						framework.Logf("Container %s in pod %s has memory metrics: available %d bytes", container.Name, pod.PodRef.Name, *container.Memory.AvailableBytes)
						gomega.Expect(*container.Memory.AvailableBytes).To(gomega.BeNumerically(">", 0), "Container available bytes should be greater than 0")
					}
				}
			}
		})

		ginkgo.It("validates /metrics/resource", func(ctx context.Context) {
			ginkgo.By("Grabbing /metrics/resource")
			resourceMetrics, err := grabber.GrabResourceMetricsFromKubelet(ctx, node.Name)
			if errors.Is(err, e2emetrics.MetricsGrabbingDisabledError) {
				e2eskipper.Skipf("%v", err)
			}

			gomega.Expect(resourceMetrics).NotTo(gomega.BeEmpty())
			ginkgo.By("Parsing and checking /metrics/resource")

			// Check if the metrics contain expected keys
			expectedKeys := []string{
				"pod_memory_working_set_bytes",
				// TODO: Add more keys as needed
			}

			for _, key := range expectedKeys {
				gomega.Expect(resourceMetrics).To(gomega.HaveKey(key), fmt.Sprintf("Expected key %s not found in /metrics/resource", key))
				sample := resourceMetrics[key]

				gomega.Expect(sample).NotTo(gomega.BeNil(), fmt.Sprintf("Metric sample for key %s should not be nil", key))
				gomega.Expect(sample).NotTo(gomega.BeEmpty(), fmt.Sprintf("Metric samples for key %s should not be empty", key))
				for _, s := range sample {
					podName, ok := s.Metric["pod"]
					if !ok {
						framework.Logf("Metric %s does not have 'pod' label, skipping validation", key)
						continue
					}

					if !strings.HasPrefix(string(podName), "statscollectiontest") {
						framework.Logf("Metric %s has 'pod' label %s, skipping validation", key, podName)
						continue
					}

					gomega.Expect(s.Value).To(gomega.BeNumerically(">", 0), fmt.Sprintf("Metric value for key %s should be greater than 0", key))
				}

			}
			framework.Logf("Successfully validated /metrics/resource for node %s", node.Name)
		})
	})

func parseSummaryOutput(output string) (*stats.Summary, error) {
	decoder := json.NewDecoder(strings.NewReader(output))
	summary := stats.Summary{}
	err := decoder.Decode(&summary)
	if err != nil {
		return nil, fmt.Errorf("failed to parse /stats/summary to go struct: %+v", output)
	}

	return &summary, nil
}
