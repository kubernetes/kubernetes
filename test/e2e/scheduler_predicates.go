/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Returns a number of currently running and not running Pods.
func getPodsNumbers(pods *api.PodList) (runningPods, notRunningPods int) {
	for _, pod := range pods.Items {
		if pod.Status.Phase == api.PodRunning {
			runningPods += 1
		} else {
			notRunningPods += 1
		}
	}
	return
}

// Simplified version of RunRC, that does not create RC, but creates plain Pods and
// requires passing whole Pod definition, which is needed to test various Scheduler predicates.
func startPods(c *client.Client, replicas int, ns string, podNamePrefix string, pod api.Pod) {
	allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
	expectNoError(err)
	podsRunningBefore, _ := getPodsNumbers(allPods)

	for i := 0; i < replicas; i++ {
		podName := fmt.Sprintf("%v-%v", podNamePrefix, i)
		pod.ObjectMeta.Name = podName
		pod.ObjectMeta.Labels["name"] = podName
		pod.Spec.Containers[0].Name = podName
		_, err = c.Pods(ns).Create(&pod)
		expectNoError(err)
	}

	// Wait for pods to start running.  Note: this is a functional
	// test, not a performance test, so the timeout needs to be
	// sufficiently long that it's only triggered if things are
	// completely broken vs. running slowly.
	timeout := 10 * time.Minute
	startTime := time.Now()
	currentlyRunningPods := 0
	for podsRunningBefore+replicas != currentlyRunningPods {
		allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		runningPods := 0
		for _, pod := range allPods.Items {
			if pod.Status.Phase == api.PodRunning {
				runningPods += 1
			}
		}
		currentlyRunningPods = runningPods
		Logf("%v pods running", currentlyRunningPods)
		if startTime.Add(timeout).Before(time.Now()) {
			Logf("Timed out after %v waiting for pods to start running.", timeout)
			break
		}
		time.Sleep(5 * time.Second)
	}
	Expect(currentlyRunningPods).To(Equal(podsRunningBefore + replicas))
}

func getRequestedCPU(pod api.Pod) int64 {
	var result int64
	for _, container := range pod.Spec.Containers {
		result += container.Resources.Limits.Cpu().MilliValue()
	}
	return result
}

func verifyResult(c *client.Client, podName string, ns string, oldNotRunning int) {
	allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
	expectNoError(err)
	_, notRunningPods := getPodsNumbers(allPods)

	schedEvents, err := c.Events(ns).List(
		labels.Everything(),
		fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      podName,
			"involvedObject.namespace": ns,
			"source":                   "scheduler",
			"reason":                   "FailedScheduling",
		}.AsSelector())
	expectNoError(err)

	printed := false
	printOnce := func(msg string) string {
		if !printed {
			printed = true
			return msg
		} else {
			return ""
		}
	}

	Expect(notRunningPods).To(Equal(1+oldNotRunning), printOnce(fmt.Sprintf("Pods found in the cluster: %#v", allPods)))
	Expect(schedEvents.Items).ToNot(BeEmpty(), printOnce(fmt.Sprintf("Pods found in the cluster: %#v", allPods)))
}

func cleanupPods(c *client.Client, ns string) {
	By("Removing all pods in namespace " + ns)
	pods, err := c.Pods(ns).List(labels.Everything(), fields.Everything())
	expectNoError(err)
	opt := api.NewDeleteOptions(0)
	for _, p := range pods.Items {
		expectNoError(c.Pods(ns).Delete(p.ObjectMeta.Name, opt))
	}
}

var _ = Describe("SchedulerPredicates", func() {
	var c *client.Client
	var nodeList *api.NodeList
	var nodeCount int
	var totalPodCapacity int64
	var RCName string
	var ns string
	var uuid string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		nodeList, err = c.Nodes().List(labels.Everything(), fields.Everything())
		expectNoError(err)
		nodeCount = len(nodeList.Items)
		Expect(nodeCount).NotTo(BeZero())

		err = deleteTestingNS(c)
		expectNoError(err)

		nsForTesting, err := createTestingNS("sched-pred", c)
		ns = nsForTesting.Name
		expectNoError(err)
		uuid = string(util.NewUUID())
	})

	AfterEach(func() {
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			By("Cleaning up the replication controller")
			err := DeleteRC(c, ns, RCName)
			expectNoError(err)
		}

		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := deleteNS(c, ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	// This test verifies that max-pods flag works as advertised. It assumes that cluster add-on pods stay stable
	// and cannot be run in parallel with any other test that touches Nodes or Pods. It is so because to check
	// if max-pods is working we need to fully saturate the cluster and keep it in this state for few seconds.
	It("validates MaxPods limit number of pods that are allowed to run.", func() {
		totalPodCapacity = 0

		for _, node := range nodeList.Items {
			podCapacity, found := node.Status.Capacity["pods"]
			Expect(found).To(Equal(true))
			totalPodCapacity += podCapacity.Value()
			Logf("Node: %v", node)
		}

		allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		currentlyRunningPods, currentlyDeadPods := getPodsNumbers(allPods)
		podsNeededForSaturation := int(totalPodCapacity) - currentlyRunningPods

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster max pods and trying to start another one", podsNeededForSaturation))

		startPods(c, podsNeededForSaturation, ns, "maxp", api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: map[string]string{"name": ""},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "",
						Image: "gcr.io/google_containers/pause:go",
					},
				},
			},
		})

		podName := "additional-pod"
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": "additional"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:go",
					},
				},
			},
		})
		expectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns, currentlyDeadPods)
		cleanupPods(c, ns)
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of limits of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	It("validates resource limits of pods that are allowed to run.", func() {
		nodeToCapacityMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			capacity, found := node.Status.Capacity["cpu"]
			Expect(found).To(Equal(true))
			nodeToCapacityMap[node.Name] = capacity.MilliValue()
		}

		pods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		var currentlyDeadPods int
		for _, pod := range pods.Items {
			_, found := nodeToCapacityMap[pod.Spec.NodeName]
			Expect(found).To(Equal(true))
			if pod.Status.Phase == api.PodRunning {
				Logf("Pod %v requesting capacity %v on Node %v", pod.Name, getRequestedCPU(pod), pod.Spec.NodeName)
				nodeToCapacityMap[pod.Spec.NodeName] -= getRequestedCPU(pod)
			} else {
				currentlyDeadPods += 1
			}
		}

		var podsNeededForSaturation int
		milliCpuPerPod := int64(500)
		for name, leftCapacity := range nodeToCapacityMap {
			Logf("Node: %v has capacity: %v", name, leftCapacity)
			podsNeededForSaturation += (int)(leftCapacity / milliCpuPerPod)
		}

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster CPU and trying to start another one", podsNeededForSaturation))

		startPods(c, podsNeededForSaturation, ns, "overcommit", api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   "",
				Labels: map[string]string{"name": ""},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "",
						Image: "gcr.io/google_containers/pause:go",
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
							},
						},
					},
				},
			},
		})

		podName := "additional-pod"
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": "additional"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:go",
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
							},
						},
					},
				},
			},
		})
		expectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns, currentlyDeadPods)
		cleanupPods(c, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	It("validates that NodeSelector is respected.", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		allPods, err := c.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything())
		expectNoError(err)
		_, currentlyDeadPods := getPodsNumbers(allPods)

		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: api.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": "restricted"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:go",
					},
				},
				NodeSelector: map[string]string{
					"label": "nonempty",
				},
			},
		})
		expectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns, currentlyDeadPods)
		cleanupPods(c, ns)
	})
})
