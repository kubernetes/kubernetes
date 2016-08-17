/*
Copyright 2015 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

const maxNumberOfPods int64 = 10
const minPodCPURequest int64 = 500

// variable set in BeforeEach, never modified afterwards
var masterNodes sets.String

var podWithNodeAffinity = api.Pod{
	ObjectMeta: api.ObjectMeta{
		Name: "with-labels",
		Annotations: map[string]string{
			"scheduler.alpha.kubernetes.io/affinity": `{
        		"nodeAffinity": {
          			"requiredDuringSchedulingIgnoredDuringExecution": {
            			"nodeSelectorTerms": [{
                			"matchExpressions": [{
                    			"key": "kubernetes.io/e2e-az-name",
                    			"operator": "In",
                    			"values": ["e2e-az1", "e2e-az2"]
                  			}]
              			}]
          			}
        		}
      		}`,
			"another-annotation-key": "another-annotation-value",
		},
	},
	Spec: api.PodSpec{
		Containers: []api.Container{
			{
				Name:  "with-labels",
				Image: "gcr.io/google_containers/pause:2.0",
			},
		},
	},
}

var podWithPodAffinity = api.Pod{
	ObjectMeta: api.ObjectMeta{
		Name: "with-newlabels",
		Annotations: map[string]string{
			"scheduler.alpha.kubernetes.io/affinity": `{
        		"podAffinity": {
          			"requiredDuringSchedulingIgnoredDuringExecution": [{
            			"labelSelector": {
              				"matchExpressions": [{
                				"key": "security",
                				"operator": "In",
                				"values":["S1"]
              				}]
            			},
            			"topologyKey": "kubernetes.io/hostname"
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
            			"topologyKey": "kubernetes.io/hostname"
          			}]
        		}
      		}`,
			"another-annotation-key": "another-annotation-value",
		},
	},
	Spec: api.PodSpec{
		Containers: []api.Container{
			{
				Name:  "with-newlabels",
				Image: "gcr.io/google_containers/pause:2.0",
			},
		},
	},
}

// Returns a number of currently scheduled and not scheduled Pods.
func getPodsScheduled(pods *api.PodList) (scheduledPods, notScheduledPods []api.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				_, scheduledCondition := api.GetPodCondition(&pod.Status, api.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(api.ConditionTrue))
				scheduledPods = append(scheduledPods, pod)
			} else {
				_, scheduledCondition := api.GetPodCondition(&pod.Status, api.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(api.ConditionFalse))
				if scheduledCondition.Reason == "Unschedulable" {
					notScheduledPods = append(notScheduledPods, pod)
				}
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

// TODO: upgrade calls in PodAffinity tests when we're able to run them
func verifyResult(c *client.Client, podName string, expectedScheduled int, expectedNotScheduled int, ns string) {
	allPods, err := c.Pods(ns).List(api.ListOptions{})
	framework.ExpectNoError(err)
	scheduledPods, notScheduledPods := framework.GetPodsScheduled(masterNodes, allPods)

	printed := false
	printOnce := func(msg string) string {
		if !printed {
			printed = true
			return msg
		} else {
			return ""
		}
	}

	Expect(len(notScheduledPods)).To(Equal(expectedNotScheduled), printOnce(fmt.Sprintf("Not scheduled Pods: %#v", notScheduledPods)))
	Expect(len(scheduledPods)).To(Equal(expectedScheduled), printOnce(fmt.Sprintf("Scheduled Pods: %#v", scheduledPods)))
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

var _ = framework.KubeDescribe("SchedulerPredicates [Serial]", func() {
	var c *client.Client
	var nodeList *api.NodeList
	var systemPodsNo int
	var totalPodCapacity int64
	var RCName string
	var ns string
	ignoreLabels := framework.ImagePullerLabels

	AfterEach(func() {
		rc, err := c.ReplicationControllers(ns).Get(RCName)
		if err == nil && rc.Spec.Replicas != 0 {
			By("Cleaning up the replication controller")
			err := framework.DeleteRCAndPods(c, ns, RCName)
			framework.ExpectNoError(err)
		}
	})

	f := framework.NewDefaultFramework("sched-pred")

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
		nodeList = &api.NodeList{}

		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(c)

		err := framework.CheckTestingNSDeletedExcept(c, ns)
		framework.ExpectNoError(err)

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := framework.GetPodsInNamespace(c, ns, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = 0
		for _, pod := range systemPods {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = framework.WaitForPodsRunningReady(c, api.NamespaceSystem, int32(systemPodsNo), framework.PodReadyBeforeTimeout, ignoreLabels)
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

		currentlyScheduledPods := framework.WaitForStableCluster(c, masterNodes)
		podsNeededForSaturation := int(totalPodCapacity) - currentlyScheduledPods

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster max pods and trying to start another one", podsNeededForSaturation))

		// As the pods are distributed randomly among nodes,
		// it can easily happen that all nodes are satured
		// and there is no need to create additional pods.
		// StartPods requires at least one pod to replicate.
		if podsNeededForSaturation > 0 {
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
							Image: framework.GetPauseImageName(f.Client),
						},
					},
				},
			}, true)
		}
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, podsNeededForSaturation, 1, ns)
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of limits of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	It("validates resource limits of pods that are allowed to run [Conformance]", func() {
		nodeMaxCapacity := int64(0)

		nodeToCapacityMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			capacity, found := node.Status.Capacity["cpu"]
			Expect(found).To(Equal(true))
			nodeToCapacityMap[node.Name] = capacity.MilliValue()
			if nodeMaxCapacity < capacity.MilliValue() {
				nodeMaxCapacity = capacity.MilliValue()
			}
		}
		framework.WaitForStableCluster(c, masterNodes)

		pods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToCapacityMap[pod.Spec.NodeName]
			if found && pod.Status.Phase != api.PodSucceeded && pod.Status.Phase != api.PodFailed {
				framework.Logf("Pod %v requesting resource cpu=%vm on Node %v", pod.Name, getRequestedCPU(pod), pod.Spec.NodeName)
				nodeToCapacityMap[pod.Spec.NodeName] -= getRequestedCPU(pod)
			}
		}

		var podsNeededForSaturation int

		milliCpuPerPod := nodeMaxCapacity / maxNumberOfPods
		if milliCpuPerPod < minPodCPURequest {
			milliCpuPerPod = minPodCPURequest
		}
		framework.Logf("Using pod capacity: %vm", milliCpuPerPod)
		for name, leftCapacity := range nodeToCapacityMap {
			framework.Logf("Node: %v has cpu capacity: %vm", name, leftCapacity)
			podsNeededForSaturation += (int)(leftCapacity / milliCpuPerPod)
		}

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster CPU and trying to start another one", podsNeededForSaturation))

		// As the pods are distributed randomly among nodes,
		// it can easily happen that all nodes are saturated
		// and there is no need to create additional pods.
		// StartPods requires at least one pod to replicate.
		if podsNeededForSaturation > 0 {
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
							Image: framework.GetPauseImageName(f.Client),
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
		}
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
						Image: framework.GetPauseImageName(f.Client),
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

		verifyResult(c, podName, podsNeededForSaturation, 1, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	It("validates that NodeSelector is respected if not matching [Conformance]", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		framework.WaitForStableCluster(c, masterNodes)

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
						Image: framework.GetPauseImageName(f.Client),
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

		verifyResult(c, podName, 0, 1, ns)
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
						Image: framework.GetPauseImageName(f.Client),
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
	})

	It("validates that NodeSelector is respected if matching [Conformance]", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a label to get a node which can launch it.")
		podName := "without-label"
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Explicitly delete pod here to free the resource it takes.")
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		pod, err = c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
				NodeSelector: map[string]string{
					"kubernetes.io/hostname": nodeName,
					k: v,
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// non-nil NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.
	It("validates that NodeAffinity is respected if not matching", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		framework.WaitForStableCluster(c, masterNodes)

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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, 0, 1, ns)
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
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Explicitly delete pod here to free the resource it takes.")
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		pod, err = c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
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
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Explicitly delete pod here to free the resource it takes.")
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a label with fake az info on the found node.")
		k := "kubernetes.io/e2e-az-name"
		v := "e2e-az1"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to launch a pod that with NodeAffinity setting as embedded JSON string in the annotation value.")
		labelPodName := podWithNodeAffinity.Name
		pod, err = c.Pods(ns).Create(&podWithNodeAffinity)
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, ""))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// labelSelector Operator is DoesNotExist but values are there in requiredDuringSchedulingIgnoredDuringExecution
	// part of podAffinity,so validation fails.
	It("validates that a pod with an invalid podAffinity is rejected because of the LabelSelectorRequirement is invalid", func() {
		By("Trying to launch a pod with an invalid pod Affinity data.")
		podName := "without-label-" + string(uuid.NewUUID())
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
						Image: framework.GetPauseImageName(f.Client),
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
	})

	// Test Nodes does not have any pod, hence it should be impossible to schedule a Pod with pod affinity.
	It("validates that Inter-pod-Affinity is respected if not matching", func() {
		By("Trying to schedule Pod with nonempty Pod Affinity.")
		podName := "without-label-" + string(uuid.NewUUID())

		framework.WaitForStableCluster(c, masterNodes)

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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, podName, 0, 1, ns)
	})

	// test the pod affinity successful matching scenario.
	It("validates that InterPodAffinity is respected if matching", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(uuid.NewUUID())
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to launch the pod, now with podAffinity.")
		labelPodName := "with-podaffinity-" + string(uuid.NewUUID())
		pod, err = c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// test when the pod anti affinity rule is not satisfied, the pod would stay pending.
	It("validates that InterPodAntiAffinity is respected if matching 2", func() {
		// launch pods to find nodes which can launch a pod. We intentionally do
		// not just take the node list and choose the first and the second of them.
		// Depending on the cluster and the scheduler it might be that a "normal" pod
		// cannot be scheduled onto it.
		By("Launching two pods on two distinct nodes to get two node names")
		CreateHostPortPods(f, "host-port", 2, true)
		defer framework.DeleteRCAndPods(f.Client, f.Namespace.Name, "host-port")
		podList, err := c.Pods(ns).List(api.ListOptions{})
		ExpectNoError(err)
		Expect(len(podList.Items)).To(Equal(2))
		nodeNames := []string{podList.Items[0].Spec.NodeName, podList.Items[1].Spec.NodeName}
		Expect(nodeNames[0]).ToNot(Equal(nodeNames[1]))

		By("Applying a random label to both nodes.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		for _, nodeName := range nodeNames {
			framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
			framework.ExpectNodeHasLabel(c, nodeName, k, v)
			defer framework.RemoveLabelOffNode(c, nodeName, k)
		}

		By("Trying to launch another pod on the first node with the service label.")
		podName := "with-label-" + string(uuid.NewUUID())
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
				NodeSelector: map[string]string{k: v}, // only launch on our two nodes
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		By("Trying to launch another pod, now with podAntiAffinity with same Labels.")
		labelPodName := "with-podantiaffinity-" + string(uuid.NewUUID())
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
				NodeSelector: map[string]string{k: v}, // only launch on our two nodes, contradicting the podAntiAffinity
			},
		})
		framework.ExpectNoError(err)

		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)

		verifyResult(c, labelPodName, 3, 1, ns)
	})

	// test the pod affinity successful matching scenario with multiple Label Operators.
	It("validates that InterPodAffinity is respected if matching with multiple Affinities", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(uuid.NewUUID())
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "kubernetes-e2e"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to launch the pod, now with multiple pod affinities with diff LabelOperators.")
		labelPodName := "with-podaffinity-" + string(uuid.NewUUID())
		pod, err = c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// test the pod affinity and anti affinity successful matching scenario.
	It("validates that InterPod Affinity and AntiAffinity is respected if matching", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod with a label to get a node which can launch it.")
		podName := "with-label-" + string(uuid.NewUUID())
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-testing"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to launch the pod, now with Pod affinity and anti affinity.")
		labelPodName := "with-podantiaffinity-" + string(uuid.NewUUID())
		pod, err = c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
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
		podName := "with-label-" + string(uuid.NewUUID())
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Trying to apply a label with fake az info on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-az1"
		framework.AddOrUpdateLabelOnNode(c, nodeName, k, v)
		framework.ExpectNodeHasLabel(c, nodeName, k, v)
		defer framework.RemoveLabelOffNode(c, nodeName, k)

		By("Trying to launch a pod that with PodAffinity & PodAntiAffinity setting as embedded JSON string in the annotation value.")
		labelPodName := podWithPodAffinity.Name
		pod, err = c.Pods(ns).Create(&podWithPodAffinity)
		framework.ExpectNoError(err)
		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, labelPodName, pod.ResourceVersion))
		labelPod, err := c.Pods(ns).Get(labelPodName)
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod with tolerations tolerate the taints on node,
	// and the pod's nodeName specified to the name of node found in step 1
	It("validates that taints-tolerations is respected if matching", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a toleration to get a node which can launch it.")
		podName := "without-toleration"
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Explicitly delete pod here to free the resource it takes.")
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random taint on the found node.")
		taintName := fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID()))
		taintValue := "testing-taint-value"
		taintEffect := api.TaintEffectNoSchedule
		framework.AddOrUpdateTaintOnNode(c, nodeName, api.Taint{Key: taintName, Value: taintValue, Effect: taintEffect})
		framework.ExpectNodeHasTaint(c, nodeName, taintName)
		defer framework.RemoveTaintOffNode(c, nodeName, taintName)

		By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(c, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(c, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(c, nodeName, labelKey)

		By("Trying to relaunch the pod, now with tolerations.")
		tolerationPodName := "with-tolerations"
		pod, err = c.Pods(ns).Create(&api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: tolerationPodName,
				Annotations: map[string]string{
					"scheduler.alpha.kubernetes.io/tolerations": `
						[
							{
								"key": "` + taintName + `",
								"value": "` + taintValue + `",
								"effect": "` + string(taintEffect) + `"
							}
						]`,
				},
			},
			Spec: api.PodSpec{
				NodeSelector: map[string]string{labelKey: labelValue},
				Containers: []api.Container{
					{
						Name:  tolerationPodName,
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new taint yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(c, ns, tolerationPodName, pod.ResourceVersion))
		deployedPod, err := c.Pods(ns).Get(tolerationPodName)
		framework.ExpectNoError(err)
		Expect(deployedPod.Spec.NodeName).To(Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod still no tolerations,
	// and the pod's nodeName specified to the name of node found in step 1
	It("validates that taints-tolerations is respected if not matching", func() {
		// launch a pod to find a node which can launch a pod. We intentionally do
		// not just take the node list and choose the first of them. Depending on the
		// cluster and the scheduler it might be that a "normal" pod cannot be
		// scheduled onto it.
		By("Trying to launch a pod without a toleration to get a node which can launch it.")
		podName := "without-toleration"
		pod, err := c.Pods(ns).Create(&api.Pod{
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
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		})
		framework.ExpectNoError(err)
		framework.ExpectNoError(framework.WaitForPodRunningInNamespace(c, pod))
		pod, err = c.Pods(ns).Get(podName)
		framework.ExpectNoError(err)

		nodeName := pod.Spec.NodeName

		By("Explicitly delete pod here to free the resource it takes.")
		err = c.Pods(ns).Delete(podName, api.NewDeleteOptions(0))
		framework.ExpectNoError(err)

		By("Trying to apply a random taint on the found node.")
		taintName := fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID()))
		taintValue := "testing-taint-value"
		taintEffect := api.TaintEffectNoSchedule
		framework.AddOrUpdateTaintOnNode(c, nodeName, api.Taint{Key: taintName, Value: taintValue, Effect: taintEffect})
		framework.ExpectNodeHasTaint(c, nodeName, taintName)
		defer framework.RemoveTaintOffNode(c, nodeName, taintName)

		By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(c, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(c, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(c, nodeName, labelKey)

		By("Trying to relaunch the pod, still no tolerations.")
		podNameNoTolerations := "still-no-tolerations"
		podNoTolerations := &api.Pod{
			TypeMeta: unversioned.TypeMeta{
				Kind: "Pod",
			},
			ObjectMeta: api.ObjectMeta{
				Name: podNameNoTolerations,
			},
			Spec: api.PodSpec{
				NodeSelector: map[string]string{labelKey: labelValue},
				Containers: []api.Container{
					{
						Name:  podNameNoTolerations,
						Image: framework.GetPauseImageName(f.Client),
					},
				},
			},
		}
		_, err = c.Pods(ns).Create(podNoTolerations)
		framework.ExpectNoError(err)

		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)
		verifyResult(c, podNameNoTolerations, 0, 1, ns)

		By("Removing taint off the node")
		framework.RemoveTaintOffNode(c, nodeName, taintName)
		// Wait a bit to allow scheduler to do its thing
		// TODO: this is brittle; there's no guarantee the scheduler will have run in 10 seconds.
		framework.Logf("Sleeping 10 seconds and crossing our fingers that scheduler will run in that time.")
		time.Sleep(10 * time.Second)
		// as taint removed off the node, expect the pod can be successfully scheduled
		verifyResult(c, podNameNoTolerations, 1, 0, ns)
	})
})
