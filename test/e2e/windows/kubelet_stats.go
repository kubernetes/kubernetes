/*
Copyright 2020 The Kubernetes Authors.

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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("[Feature:Windows] Kubelet-Stats [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-stats-test-windows")

	ginkgo.Describe("Kubelet stats collection for Windows nodes", func() {
		ginkgo.Context("when running 10 pods", func() {
			// 10 seconds is the default scrape timeout for metrics-server and kube-prometheus
			ginkgo.It("should return within 10 seconds", func() {

				ginkgo.By("Selecting a Windows node")
				targetNode, err := findWindowsNode(f)
				framework.ExpectNoError(err, "Error finding Windows node")
				e2elog.Logf("Using node: %v", targetNode.Name)

				ginkgo.By("Scheduling 10 pods")
				powershellImage := imageutils.GetConfig(imageutils.BusyBox)
				pods := newKubeletStatsTestPods(10, powershellImage, targetNode.Name)
				f.PodClient().CreateBatch(pods)

				ginkgo.By("Waiting up to 3 minutes for pods to be running")
				timeout := 3 * time.Minute
				framework.WaitForPodsRunningReady(f.ClientSet, f.Namespace.Name, 10, 0, timeout, make(map[string]string))

				ginkgo.By("Getting kubelet stats 5 times and checking average duration")
				iterations := 5
				var totalDurationSeconds float64

				for i := 0; i < iterations; i++ {
					start := time.Now()
					nodeStats, err := framework.GetStatsSummary(f.ClientSet, targetNode.Name)
					duration := time.Since(start)
					totalDurationSeconds += duration.Seconds()

					framework.ExpectNoError(err, "Error getting kubelet stats")

					// Perform some basic sanity checks on retrieved stats for pods in this test's namespace
					statsChecked := 0
					for _, podStats := range nodeStats.Pods {
						if podStats.PodRef.Namespace != f.Namespace.Name {
							continue
						}
						statsChecked = statsChecked + 1

						gomega.Expect(*podStats.CPU.UsageCoreNanoSeconds > 0).To(gomega.BeTrue(), "Pod status should not report 0 cpu usage")
						gomega.Expect(*podStats.Memory.WorkingSetBytes > 0).To(gomega.BeTrue(), "Pod stats should not report 0 bytes for memory working set")
					}
					gomega.Expect(statsChecked).To(gomega.Equal(10), "Should find stats for 10 pods in kubelet stats")

					time.Sleep(5 * time.Second)
				}

				avgDurationSeconds := totalDurationSeconds / float64(iterations)

				durationMatch := avgDurationSeconds <= time.Duration(10*time.Second).Seconds()
				e2elog.Logf("Getting kubelet stats for node %v took an average of %v seconds over %v iterations", targetNode.Name, avgDurationSeconds, iterations)
				gomega.Expect(durationMatch).To(gomega.BeTrue(), "Collecting kubelet stats should not take longer than 10 seconds")
			})
		})
	})
})

// findWindowsNode finds a Windows node that is Ready and Schedulable
func findWindowsNode(f *framework.Framework) (v1.Node, error) {
	selector := labels.Set{"beta.kubernetes.io/os": "windows"}.AsSelector()
	nodeList, err := f.ClientSet.CoreV1().Nodes().List(metav1.ListOptions{LabelSelector: selector.String()})

	if err != nil {
		return v1.Node{}, err
	}

	var targetNode v1.Node
	foundNode := false
	for _, n := range nodeList.Items {
		if framework.IsNodeConditionSetAsExpected(&n, v1.NodeReady, true) && !n.Spec.Unschedulable {
			targetNode = n
			foundNode = true
			break
		}
	}

	if foundNode == false {
		framework.Skipf("Could not find and ready and schedulable Windows nodes")
	}

	return targetNode, nil
}

// newKubeletStatsTestPods creates a list of pods (specification) for test.
func newKubeletStatsTestPods(numPods int, image imageutils.Config, nodeName string) []*v1.Pod {
	var pods []*v1.Pod

	for i := 0; i < numPods; i++ {
		podName := "statscollectiontest-" + string(uuid.NewUUID())
		pod := v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
				Labels: map[string]string{
					"name":    podName,
					"testapp": "stats-collection",
				},
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Image: image.GetE2EImage(),
						Name:  podName,
						Command: []string{
							"powershell.exe",
							"-Command",
							"sleep -Seconds 600",
						},
					},
				},

				NodeName: nodeName,
			},
		}

		pods = append(pods, &pod)
	}

	return pods
}
