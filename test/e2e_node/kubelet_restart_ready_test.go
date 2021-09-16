/*
Copyright 2021 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// Serial because the test restarts Kubelet
var _ = SIGDescribe("Kubelet Restart [Serial]", func() {
	f := framework.NewDefaultFramework("kubelet-restart")
	testKubeletRestart(f)
})

func testKubeletRestart(f *framework.Framework) {
	ginkgo.Context("KubeletRestart", func() {
		ginkgo.It("Verifies that Kubelet restart does not interfere with pod readiness.", func() {

			ginkgo.By("Create a pod with a readiness probe that always returns true")
			podName := "always-passing-readiness-probe-test-pod"
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: podName},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test-container",
							Image:   imageutils.GetE2EImage(imageutils.BusyBox),
							Command: []string{"/bin/sleep"},
							Args:    []string{"10000"},
							ReadinessProbe: &v1.Probe{
								Handler: v1.Handler{
									Exec: &v1.ExecAction{
										Command: []string{
											"/bin/true",
										},
									},
								},
								PeriodSeconds: 10, // A larger period here helps expose the bug
							},
						},
					},
				},
			})

			ginkgo.By("Wait for pod to become ready")
			gomega.Eventually(func() bool {
				return f.PodClient().PodIsReady(podName)
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())
			originalTransitionTime := getPodReadyLastTransisionTime(f, podName)

			originalProbeCount := getReadyProbeCount(f, podName)

			ginkgo.By("Restarting Kubelet")
			restartTime := time.Now()
			restartKubelet()

			// We wait for the node to become ready as a proxy for kubelet finishing restarting.
			// This is needed so that we don't error out trying to poll the kubelet probe metrics below.
			ginkgo.By("Wait for node to be ready")
			gomega.Eventually(func() bool {
				node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				for _, cond := range node.Status.Conditions {
					if cond.Type == v1.NodeReady && cond.Status == v1.ConditionTrue && cond.LastHeartbeatTime.After(restartTime) {
						return true
					}
				}
				return false
			}, 5*time.Minute, framework.Poll).Should(gomega.BeTrue())

			ginkgo.By("Wait until the pod has been probed again")
			gomega.Eventually(func() int {
				return getReadyProbeCount(f, podName)
			}, 5*time.Minute, framework.Poll).Should(gomega.BeNumerically(">", originalProbeCount))

			ginkgo.By("Check if pod readiness has changed")
			updatedTransitionTime := getPodReadyLastTransisionTime(f, podName)
			framework.ExpectEqual(updatedTransitionTime, originalTransitionTime)
		})
	})
}

func getReadyProbeCount(f *framework.Framework, podName string) int {
	total := 0
	m, err := e2emetrics.GrabKubeletMetricsWithoutProxy(framework.TestContext.NodeName+":10255", "/metrics/probes")
	framework.ExpectNoError(err)
	samples, ok := m["prober_probe_total"]
	// Note: We return 0 in the case that the metric doesn't exist yet after
	//       kubelet is restarted and no probes have been done yet.
	if ok {
		for _, sample := range samples {
			ns := string(sample.Metric["namespace"])
			pod := string(sample.Metric["pod"])
			probe_type := string(sample.Metric["probe_type"])
			if ns == f.Namespace.Name && pod == podName && probe_type == "Readiness" {
				total += int(sample.Value)
			}
		}
	}
	return total
}

func getPodReadyLastTransisionTime(f *framework.Framework, podName string) metav1.Time {
	pod, err := f.PodClient().Get(context.TODO(), podName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	cond := getPodReadyCondition(pod)
	framework.ExpectNotEqual(cond, nil)
	return cond.LastTransitionTime
}

func getPodReadyCondition(pod *v1.Pod) *v1.PodCondition {
	framework.ExpectNotEqual(pod.Status, nil)
	framework.ExpectNotEqual(pod.Status.Conditions, nil)
	for _, cond := range pod.Status.Conditions {
		if cond.Type == v1.PodReady {
			return &cond
		}
	}
	return nil // not found
}
