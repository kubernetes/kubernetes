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
	"path/filepath"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

// variable set in BeforeEach, never modified afterwards
var masterNodes sets.String

// Returns a number of currently scheduled and not scheduled Pods.
func getPodsScheduled(pods *api.PodList) (scheduledPods, notScheduledPods []api.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				scheduledPods = append(scheduledPods, pod)
			} else {
				notScheduledPods = append(notScheduledPods, pod)
			}
		}
	}
	return
}

func getRequestedCPU(pod api.Pod) int64 {
	var result int64
	for _, container := range pod.Spec.Containers {
		result += container.Resources.Requests.Cpu().MilliValue()
	}
	return result
}

func verifyResult(c *client.Client, podName string, ns string) {
	allPods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
	framework.ExpectNoError(err)
	scheduledPods, notScheduledPods := getPodsScheduled(allPods)

	selector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      podName,
		"involvedObject.namespace": ns,
		"source":                   api.DefaultSchedulerName,
		"reason":                   "FailedScheduling",
	}.AsSelector()
	options := api.ListOptions{FieldSelector: selector}
	schedEvents, err := c.Events(ns).List(options)
	framework.ExpectNoError(err)
	// If we failed to find event with a capitalized first letter of reason
	// try looking for one starting with a small one for backward compatibility.
	// If we don't do it we end up in #15806.
	// TODO: remove this block when we don't care about supporting v1.0 too much.
	if len(schedEvents.Items) == 0 {
		selector := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      podName,
			"involvedObject.namespace": ns,
			"source":                   "scheduler",
			"reason":                   "failedScheduling",
		}.AsSelector()
		options := api.ListOptions{FieldSelector: selector}
		schedEvents, err = c.Events(ns).List(options)
		framework.ExpectNoError(err)
	}

	printed := false
	printOnce := func(msg string) string {
		if !printed {
			printed = true
			return msg
		} else {
			return ""
		}
	}

	Expect(len(notScheduledPods)).To(Equal(1), printOnce(fmt.Sprintf("Not scheduled Pods: %#v", notScheduledPods)))
	Expect(schedEvents.Items).ToNot(BeEmpty(), printOnce(fmt.Sprintf("Scheduled Pods: %#v", scheduledPods)))
}

func cleanupPods(c *client.Client, ns string) {
	By("Removing all pods in namespace " + ns)
	pods, err := c.Pods(ns).List(api.ListOptions{})
	framework.ExpectNoError(err)
	opt := api.NewDeleteOptions(0)
	for _, p := range pods.Items {
		framework.ExpectNoError(c.Pods(ns).Delete(p.ObjectMeta.Name, opt))
	}
}

// Waits until all existing pods are scheduled and returns their amount.
func waitForStableCluster(c *client.Client) int {
	timeout := 10 * time.Minute
	startTime := time.Now()

	allPods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
	framework.ExpectNoError(err)
	scheduledPods, currentlyNotScheduledPods := getPodsScheduled(allPods)
	for len(currentlyNotScheduledPods) != 0 {
		time.Sleep(2 * time.Second)

		allPods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
		framework.ExpectNoError(err)
		scheduledPods, currentlyNotScheduledPods = getPodsScheduled(allPods)

		if startTime.Add(timeout).Before(time.Now()) {
			framework.Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
	}
	return len(scheduledPods)
}

var _ = framework.KubeDescribe("SchedulerPredicates [Serial]", func() {
	var c *client.Client
	var nodeList *api.NodeList
	var systemPodsNo int
	var totalPodCapacity int64
	var RCName string
	var ns string

	AfterEach(func() {
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			By("Cleaning up the replication controller")
			err := framework.DeleteRC(c, ns, RCName)
			framework.ExpectNoError(err)
		}
	})

	f := framework.NewDefaultFramework("sched-pred")

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
		nodeList = &api.NodeList{}
		nodes, err := c.Nodes().List(api.ListOptions{})
		masterNodes = sets.NewString()
		for _, node := range nodes.Items {
			if system.IsMasterNode(&node) {
				masterNodes.Insert(node.Name)
			} else {
				nodeList.Items = append(nodeList.Items, node)
			}
		}

		err = framework.CheckTestingNSDeletedExcept(c, ns)
		framework.ExpectNoError(err)

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := c.Pods(api.NamespaceSystem).List(api.ListOptions{})
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = 0
		for _, pod := range systemPods.Items {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = framework.WaitForPodsRunningReady(api.NamespaceSystem, int32(systemPodsNo), framework.PodReadyBeforeTimeout)
		Expect(err).NotTo(HaveOccurred())

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			framework.PrintAllKubeletPods(c, node.Name)
		}

	})

	// This test verifies that max-pods flag works as advertised. It assumes that cluster add-on pods stay stable
	// and cannot be run in parallel with any other test that touches Nodes or Pods. It is so because to check
	// if max-pods is working we need to fully saturate the cluster and keep it in this state for few seconds.
	//
	// Slow PR #13315 (8 min)
	It("validates MaxPods limit number of pods that are allowed to run [Slow]", func() {
		totalPodCapacity = 0

		for _, node := range nodeList.Items {
			framework.Logf("Node: %v", node)
			podCapacity, found := node.Status.Capacity["pods"]
			Expect(found).To(Equal(true))
			totalPodCapacity += podCapacity.Value()
		}

		currentlyScheduledPods := waitForStableCluster(c)
		podsNeededForSaturation := int(totalPodCapacity) - currentlyScheduledPods

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster max pods and trying to start another one", podsNeededForSaturation))

		framework.StartPods(c, podsNeededForSaturation, ns, "maxp", api.Pod{
			TypeMeta: unversioned.TypeMeta{
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
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		}, true)

		podName := "additional-pod"
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
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
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns)
		cleanupPods(c, ns)
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of limits of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	It("validates resource limits of pods that are allowed to run [Conformance]", func() {
		nodeToCapacityMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			capacity, found := node.Status.Capacity["cpu"]
			Expect(found).To(Equal(true))
			nodeToCapacityMap[node.Name] = capacity.MilliValue()
		}
		waitForStableCluster(c)

		pods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToCapacityMap[pod.Spec.NodeName]
			if found && pod.Status.Phase == api.PodRunning {
				framework.Logf("Pod %v requesting resource %v on Node %v", pod.Name, getRequestedCPU(pod), pod.Spec.NodeName)
				nodeToCapacityMap[pod.Spec.NodeName] -= getRequestedCPU(pod)
			}
		}

		var podsNeededForSaturation int
		milliCpuPerPod := int64(500)
		for name, leftCapacity := range nodeToCapacityMap {
			framework.Logf("Node: %v has capacity: %v", name, leftCapacity)
			podsNeededForSaturation += (int)(leftCapacity / milliCpuPerPod)
		}

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster CPU and trying to start another one", podsNeededForSaturation))

		framework.StartPods(c, podsNeededForSaturation, ns, "overcommit", api.Pod{
			TypeMeta: unversioned.TypeMeta{
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
						Image: "gcr.io/google_containers/pause-amd64:3.0",
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
							},
							Requests: api.ResourceList{
								"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
							},
						},
					},
				},
			},
		}, true)

		podName := "additional-pod"
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
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
						Image: "gcr.io/google_containers/pause-amd64:3.0",
						Resources: api.ResourceRequirements{
							Limits: api.ResourceList{
								"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
							},
						},
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns)
		cleanupPods(c, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	It("validates that NodeSelector is respected if not matching [Conformance]", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		waitForStableCluster(c)

		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
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
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
				NodeSelector: map[string]string{
					"label": "nonempty",
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns)
		cleanupPods(c, ns)
	})

	It("validates that a pod with an invalid NodeAffinity is rejected", func() {

		By("Trying to launch a pod with an invalid Affinity data.")
		podName := "without-label"
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					api.AffinityAnnotationKey: `
					{"nodeAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [{
								"matchExpressions": []
							}]
						},
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})

		if err == nil || !errors.IsInvalid(err) {
			framework.Failf("Expect error of invalid, got : %v", err)
		}

		// Wait a bit to allow scheduler to do its thing if the pod is not rejected.
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		cleanupPods(c, ns)
	})

	It("validates that NodeSelector is respected if matching [Conformance]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a label to get a node which can launch it.")
		podName := "without-label"
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(util.NewUUID()))
		v := "42"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: labelPodName,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/hostname": nodeName,
					k: v,
				},
			},
		})
		framework.ExpectNoError(err)
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// non-nil NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.
	It("validates that NodeAffinity is respected if not matching", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		waitForStableCluster(c)

		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": "restricted"},
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
					{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
						"nodeSelectorTerms": [
							{
								"matchExpressions": [{
									"key": "foo",
									"operator": "In",
									"values": ["bar", "value2"]
								}]
							},
							{
								"matchExpressions": [{
									"key": "diffkey",
									"operator": "In",
									"values": ["wrong", "value2"]
								}]
							}
						]
					}}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns)
		cleanupPods(c, ns)
	})

	// Keep the same steps with the test on NodeSelector,
	// but specify Affinity in Pod.Annotations, instead of NodeSelector.
	It("validates that required NodeAffinity setting is respected if matching", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a label to get a node which can launch it.")
		podName := "without-label"
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(util.NewUUID()))
		v := "42"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: labelPodName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
						{"nodeAffinity": { "requiredDuringSchedulingIgnoredDuringExecution": {
							"nodeSelectorTerms": [
							{
								"matchExpressions": [{
									"key": "kubernetes.io/hostname",
									"operator": "In",
									"values": ["` + nodeName + `"]
								},{
									"key": "` + k + `",
									"operator": "In",
									"values": ["` + v + `"]
								}]
							}
						]
					}}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Verify that an escaped JSON string of NodeAffinity in a YAML PodSpec works.
	It("validates that embedding the JSON NodeAffinity setting as a string in the annotation value work", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a label to get a node which can launch it.")
		podName := "without-label"
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause-amd64:3.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a label with fake az info on the found node.")
		k := "kubernetes.io/e2e-az-name"
		v := "e2e-az1"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch a pod that with NodeAffinity setting as embedded JSON string in the annotation value.")
		labelPodName := "with-labels"
		nodeSelectionRoot := filepath.Join(framework.TestContext.RepoRoot, "test/e2e/node-selection")
		testPodPath := filepath.Join(nodeSelectionRoot, "pod-with-node-affinity.yaml")
		framework.RunKubectlOrDie("create", "-f", testPodPath, fmt.Sprintf("--namespace=%v", ns))
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// labelSelector Operator is DoesNotExist but values are there in requiredDuringSchedulingIgnoredDuringExecution
	// part of podAffinity,so validation fails.
	It("validates that a pod with an invalid podAffinity is rejected because of the LabelSelectorRequirement is invalid", func() {
		By("Trying to launch a pod with an invalid pod Affinity data.")
		podName := "without-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"name": "without-label"},
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
					{"podAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"weight": 0,
							"podAffinityTerm": {
								"labelSelector": {
									"matchExpressions": [{
										"key": "service",
										"operator": "DoesNotExist",
										"values":["securityscan"]
									}]
								},
								"namespaces": [],
								"topologyKey": "kubernetes.io/hostname"
							}
						}]
					 }}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})

		if err == nil || !errors.IsInvalid(err) {
			framework.Failf("Expect error of invalid, got : %v", err)
		}

		// Wait a bit to allow scheduler to do its thing if the pod is not rejected.
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		cleanupPods(c, ns)
	})

	// Test Nodes does not have any pod, hence it should be impossible to schedule a Pod with pod affinity.
	It("validates that Inter-pod-Affinity is respected if not matching [Feature:PodAffinity]", func() {
		By("Trying to schedule Pod with nonempty Pod Affinity.")
		podName := "without-label-" + string(util.NewUUID())

		waitForStableCluster(c)

		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector":{
								"matchExpressions": [{
									"key": "service",
									"operator": "In",
									"values": ["securityscan", "value2"]
								}]
							},
							"topologyKey": "kubernetes.io/hostname"
						}]
				 }}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, ns)
		cleanupPods(c, ns)
	})

	// test the pod affinity successful matching scenario.
	It("validates that InterPodAffinity is respected if matching [Feature:PodAffinity]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"security": "S1"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		defer c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch the pod, now with podAffinity.")
		labelPodName := "with-podaffinity-" + string(util.NewUUID())
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: labelPodName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector":{
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values": ["S1", "value2"]
								}]
							},
							"topologyKey": "` + k + `",
							"namespaces":["` + ns + `"]
						}]
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// test when the pod anti affinity rule is not satisfied, the pod would stay pending.
	It("validates that InterPodAntiAffinity is respected if matching 2 [Feature:PodAffinity]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"service": "S1"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch the pod, now with podAffinity with same Labels.")
		labelPodName := "with-podaffinity-" + string(util.NewUUID())
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   labelPodName,
				Labels: map[string]string{"service": "Diff"},
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
						{"podAntiAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector":{
									"matchExpressions": [{
										"key": "service",
										"operator": "In",
										"values": ["S1", "value2"]
									}]
								},
								"topologyKey": "` + k + `",
								"namespaces": ["` + ns + `"]
							}]
						}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, labelPodName, ns)
		cleanupPods(c, ns)
	})

	// test the pod affinity successful matching scenario with multiple Label Operators.
	It("validates that InterPodAffinity is respected if matching with multiple Affinities [Feature:PodAffinity]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"security": "S1"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		defer c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "kubernetes-e2e"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch the pod, now with multiple pod affinities with diff LabelOperators.")
		labelPodName := "with-podaffinity-" + string(util.NewUUID())
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: labelPodName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
						{"podAffinity": {
							"requiredDuringSchedulingIgnoredDuringExecution": [{
								"labelSelector":{
									"matchExpressions": [{
										"key": "security",
										"operator": "In",
										"values": ["S1", "value2"]
									},
									{
										"key": "security",
										"operator": "NotIn",
										"values": ["S2"]
									},
									{
										"key": "security",
										"operator":"Exists"
									}]
								},
								"topologyKey": "` + k + `"
							}]
					 }}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// test the pod affinity and anti affinity successful matching scenario.
	It("validates that InterPod Affinity and AntiAffinity is respected if matching [Feature:PodAffinity]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"security": "S1"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		defer c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-testing"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch the pod, now with Pod affinity and anti affinity.")
		labelPodName := "with-podantiaffinity-" + string(util.NewUUID())
		_, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: labelPodName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/affinity": `
					{"podAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S1"]
								}]
							},
							"topologyKey": "` + k + `"
						}]
					},
					"podAntiAffinity": {
						"requiredDuringSchedulingIgnoredDuringExecution": [{
							"labelSelector": {
								"matchExpressions": [{
									"key": "security",
									"operator": "In",
									"values":["S2"]
								}]
							},
							"topologyKey": "` + k + `"
						}]
					}}`,
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  labelPodName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Verify that an escaped JSON string of pod affinity and pod anti affinity in a YAML PodSpec works.
	It("validates that embedding the JSON PodAffinity and PodAntiAffinity setting as a string in the annotation value work", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with label to get a node which can launch it.")
		podName := "with-label-" + string(util.NewUUID())
		_, err := c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name:   podName,
				Labels: map[string]string{"security": "S1"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  podName,
						Image: "gcr.io/google_containers/pause:2.0",
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, podName, ns))
		pod, err := c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName
		defer c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))

		By("Trying to apply a label with fake az info on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-az1"
		patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, k, v)
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		framework.ExpectNoError(err)

		node, err := c.Nodes().Get(nodeName)
		framework.ExpectNoError(err)
		Expect(node.Labels[k]).To(Equal(v))

		By("Trying to launch a pod that with PodAffinity & PodAntiAffinity setting as embedded JSON string in the annotation value.")
		labelPodName := "with-newlabels"
		nodeSelectionRoot := filepath.Join(framework.TestContext.RepoRoot, "test/e2e/node-selection")
		testPodPath := filepath.Join(nodeSelectionRoot, "pod-with-pod-affinity.yaml")
		framework.RunKubectlOrDie("create", "-f", testPodPath, fmt.Sprintf("--namespace=%v", ns))
		defer c.Pods(ns).Delete(labelPodName, api.NewDeleteOptions(0))

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})
})
