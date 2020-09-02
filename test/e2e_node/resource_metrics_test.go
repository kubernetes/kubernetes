/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"

	"github.com/prometheus/common/model"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"
)

const (
	pod0        = "stats-busybox-0"
	pod1        = "stats-busybox-1"
	maxStatsAge = time.Minute
)

var _ = framework.KubeDescribe("ResourceMetricsAPI [NodeFeature:ResourceMetrics]", func() {
	f := framework.NewDefaultFramework("resource-metrics")
	ginkgo.Context("when querying /resource/metrics", func() {
		ginkgo.BeforeEach(func() {
			ginkgo.By("Creating test pods to measure their resource usage")
			numRestarts := int32(1)
			pods := getSummaryTestPods(f, numRestarts, pod0, pod1)
			f.PodClient().CreateBatch(pods)

			ginkgo.By("restarting the containers to ensure container metrics are still being gathered after a container is restarted")
			gomega.Eventually(func() error {
				for _, pod := range pods {
					err := verifyPodRestartCount(f, pod.Name, len(pod.Spec.Containers), numRestarts)
					if err != nil {
						return err
					}
				}
				return nil
			}, time.Minute, 5*time.Second).Should(gomega.Succeed())

			ginkgo.By("Waiting 15 seconds for cAdvisor to collect 2 stats points")
			time.Sleep(15 * time.Second)
		})
		ginkgo.It("should report resource usage through the resouce metrics api", func() {
			ginkgo.By("Fetching node so we can match against an appropriate memory limit")
			node := getLocalNode(f)
			memoryCapacity := node.Status.Capacity["memory"]
			memoryLimit := memoryCapacity.Value()

			matchResourceMetrics := gstruct.MatchAllKeys(gstruct.Keys{
				"scrape_error": gstruct.Ignore(),
				"node_cpu_usage_seconds_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": boundedSample(1, 1e6),
				}),
				"node_memory_working_set_bytes": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": boundedSample(10*e2evolume.Mb, memoryLimit),
				}),

				"container_cpu_usage_seconds_total": gstruct.MatchElements(containerID, gstruct.IgnoreExtras, gstruct.Elements{
					fmt.Sprintf("%s::%s::%s", f.Namespace.Name, pod0, "busybox-container"): boundedSample(0, 100),
					fmt.Sprintf("%s::%s::%s", f.Namespace.Name, pod1, "busybox-container"): boundedSample(0, 100),
				}),

				"container_memory_working_set_bytes": gstruct.MatchElements(containerID, gstruct.IgnoreExtras, gstruct.Elements{
					fmt.Sprintf("%s::%s::%s", f.Namespace.Name, pod0, "busybox-container"): boundedSample(10*e2evolume.Kb, 80*e2evolume.Mb),
					fmt.Sprintf("%s::%s::%s", f.Namespace.Name, pod1, "busybox-container"): boundedSample(10*e2evolume.Kb, 80*e2evolume.Mb),
				}),
			})
			ginkgo.By("Giving pods a minute to start up and produce metrics")
			gomega.Eventually(getResourceMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			ginkgo.By("Ensuring the metrics match the expectations a few more times")
			gomega.Consistently(getResourceMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})
		ginkgo.AfterEach(func() {
			ginkgo.By("Deleting test pods")
			var zero int64 = 0
			f.PodClient().DeleteSync(pod0, metav1.DeleteOptions{GracePeriodSeconds: &zero}, 10*time.Minute)
			f.PodClient().DeleteSync(pod1, metav1.DeleteOptions{GracePeriodSeconds: &zero}, 10*time.Minute)
			if !ginkgo.CurrentGinkgoTestDescription().Failed {
				return
			}
			if framework.TestContext.DumpLogsOnFailure {
				e2ekubectl.LogFailedContainers(f.ClientSet, f.Namespace.Name, framework.Logf)
			}
			ginkgo.By("Recording processes in system cgroups")
			recordSystemCgroupProcesses()
		})
	})
})

func getResourceMetrics() (e2emetrics.KubeletMetrics, error) {
	ginkgo.By("getting stable resource metrics API")
	return e2emetrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName+":10255", "/metrics/resource")
}

func nodeID(element interface{}) string {
	return ""
}

func containerID(element interface{}) string {
	el := element.(*model.Sample)
	return fmt.Sprintf("%s::%s::%s", el.Metric["namespace"], el.Metric["pod"], el.Metric["container"])
}

func boundedSample(lower, upper interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		// We already check Metric when matching the Id
		"Metric": gstruct.Ignore(),
		"Value":  gomega.And(gomega.BeNumerically(">=", lower), gomega.BeNumerically("<=", upper)),
		"Timestamp": gomega.WithTransform(func(t model.Time) time.Time {
			// model.Time is in Milliseconds since epoch
			return time.Unix(0, int64(t)*int64(time.Millisecond))
		},
			gomega.And(
				gomega.BeTemporally(">=", time.Now().Add(-maxStatsAge)),
				// Now() is the test start time, not the match time, so permit a few extra minutes.
				gomega.BeTemporally("<", time.Now().Add(2*time.Minute))),
		)}))
}
