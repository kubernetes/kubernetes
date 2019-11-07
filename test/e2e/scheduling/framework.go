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

	allPods := getAllPods(c)
	scheduledSystemPods, currentlyNotScheduledSystemPods := getSystemPods(c)
	systemPods := scheduledSystemPods + currentlyNotScheduledSystemPods

	// Wait for system pods to be scheduled, and for pods in all other namespaces to be deleted
	for currentlyNotScheduledSystemPods != 0 || systemPods != allPods {
		time.Sleep(2 * time.Second)

		scheduledSystemPods, currentlyNotScheduledSystemPods := getSystemPods(c)
		systemPods = scheduledSystemPods + currentlyNotScheduledSystemPods
		allPods = getAllPods(c)

		if startTime.Add(timeout).Before(time.Now()) {
			framework.Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
	}
	return scheduledSystemPods
}

// getAllPods lists all pods in the cluster, with succeeded and failed pods filtered out and returns the count
func getAllPods(c clientset.Interface) int {
	allPods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
	framework.ExpectNoError(err, "listing all pods in kube-system namespace while waiting for stable cluster")
	// API server returns also Pods that succeeded. We need to filter them out.
	currentPods := make([]v1.Pod, 0, len(allPods.Items))
	for _, pod := range allPods.Items {
		if pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
			currentPods = append(currentPods, pod)
		}

	}
	allPods.Items = currentPods
	return len(allPods.Items)
}

// getSystemPods lists the pods in the kube-system namespace and returns the number of scheduled and unscheduled pods
func getSystemPods(c clientset.Interface) (int, int) {
	systemPods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(metav1.ListOptions{})
	framework.ExpectNoError(err, "listing all pods in kube-system namespace while waiting for stable cluster")
	scheduledSystemPods, currentlyNotScheduledSystemPods := e2epod.GetPodsScheduled(masterNodes, systemPods)
	return len(scheduledSystemPods), len(currentlyNotScheduledSystemPods)
}
