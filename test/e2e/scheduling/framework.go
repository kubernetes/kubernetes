/*
Copyright 2017 The Kubernetes Authors.

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

package scheduling

import (
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// SIGDescribe annotates the test with the SIG label.
func SIGDescribe(text string, body func()) bool {
	return ginkgo.Describe("[sig-scheduling] "+text, body)
}

// WaitForStableCluster waits until all existing pods are scheduled and returns their amount.
func WaitForStableCluster(c clientset.Interface, masterNodes sets.String) int {
	timeout := 10 * time.Minute
	startTime := time.Now()

	// Wait for all pods to be scheduled.
	allScheduledPods, allNotScheduledPods := getFilteredPods(c, masterNodes, metav1.NamespaceAll)
	for len(allNotScheduledPods) != 0 {
		time.Sleep(2 * time.Second)
		if startTime.Add(timeout).Before(time.Now()) {
			framework.Logf("Timed out waiting for the following pods to schedule")
			for _, p := range allNotScheduledPods {
				framework.Logf("%v/%v", p.Namespace, p.Name)
			}
			framework.Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
		allScheduledPods, allNotScheduledPods = getFilteredPods(c, masterNodes, metav1.NamespaceAll)
	}
	return len(allScheduledPods)
}

// getFilteredPods lists scheduled and not scheduled pods in the given namespace, with succeeded and failed pods filtered out.
func getFilteredPods(c clientset.Interface, masterNodes sets.String, ns string) (scheduledPods, notScheduledPods []v1.Pod) {
	pods, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
	framework.ExpectNoError(err, "listing all pods in kube-system namespace while waiting for stable cluster")
	// API server returns also Pods that succeeded. We need to filter them out.
	filteredPods := make([]v1.Pod, 0, len(pods.Items))
	for _, pod := range pods.Items {
		if pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
			filteredPods = append(filteredPods, pod)
		}
	}
	pods.Items = filteredPods
	return e2epod.GetPodsScheduled(masterNodes, pods)
}
