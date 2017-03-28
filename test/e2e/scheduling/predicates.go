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

package scheduling

import (
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"
)

const maxNumberOfPods int64 = 10
const minPodCPURequest int64 = 500

// variable set in BeforeEach, never modified afterwards
var masterNodes sets.String

type pausePodConfig struct {
	Name                              string
	Affinity                          *v1.Affinity
	Annotations, Labels, NodeSelector map[string]string
	Resources                         *v1.ResourceRequirements
	Tolerations                       []v1.Toleration
}

var _ = framework.KubeDescribe("SchedulerPredicates [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var systemPodsNo int
	var totalPodCapacity int64
	var RCName string
	var ns string
	f := framework.NewDefaultFramework("sched-pred")
	ignoreLabels := framework.ImagePullerLabels

	AfterEach(func() {
		rc, err := cs.Core().ReplicationControllers(ns).Get(RCName, metav1.GetOptions{})
		if err == nil && *(rc.Spec.Replicas) != 0 {
			By("Cleaning up the replication controller")
			err := framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, RCName)
			framework.ExpectNoError(err)
		}
	})

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}

		framework.WaitForAllNodesHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)

		// Every test case in this suite assumes that cluster add-on pods stay stable and
		// cannot be run in parallel with any other test that touches Nodes or Pods.
		// It is so because we need to have precise control on what's running in the cluster.
		systemPods, err := framework.GetPodsInNamespace(cs, ns, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
		systemPodsNo = 0
		for _, pod := range systemPods {
			if !masterNodes.Has(pod.Spec.NodeName) && pod.DeletionTimestamp == nil {
				systemPodsNo++
			}
		}

		err = framework.WaitForPodsRunningReady(cs, metav1.NamespaceSystem, int32(systemPodsNo), 0, framework.PodReadyBeforeTimeout, ignoreLabels, true)
		Expect(err).NotTo(HaveOccurred())

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			framework.PrintAllKubeletPods(cs, node.Name)
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

		currentlyScheduledPods := framework.WaitForStableCluster(cs, masterNodes)
		podsNeededForSaturation := int(totalPodCapacity) - currentlyScheduledPods

		By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster max pods and trying to start another one", podsNeededForSaturation))

		// As the pods are distributed randomly among nodes,
		// it can easily happen that all nodes are satured
		// and there is no need to create additional pods.
		// StartPods requires at least one pod to replicate.
		if podsNeededForSaturation > 0 {
			framework.ExpectNoError(testutils.StartPods(cs, podsNeededForSaturation, ns, "maxp",
				*initPausePod(f, pausePodConfig{
					Name:   "",
					Labels: map[string]string{"name": ""},
				}), true, framework.Logf))
		}
		podName := "additional-pod"
		WaitForSchedulerAfterAction(f, createPausePodAction(f, pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "additional"},
		}), podName, false)
		verifyResult(cs, podsNeededForSaturation, 1, ns)
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
		framework.WaitForStableCluster(cs, masterNodes)

		pods, err := cs.Core().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToCapacityMap[pod.Spec.NodeName]
			if found && pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
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
			framework.ExpectNoError(testutils.StartPods(cs, podsNeededForSaturation, ns, "overcommit",
				*initPausePod(f, pausePodConfig{
					Name:   "",
					Labels: map[string]string{"name": ""},
					Resources: &v1.ResourceRequirements{
						Limits: v1.ResourceList{
							"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
						},
						Requests: v1.ResourceList{
							"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
						},
					},
				}), true, framework.Logf))
		}
		podName := "additional-pod"
		conf := pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "additional"},
			Resources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					"cpu": *resource.NewMilliQuantity(milliCpuPerPod, "DecimalSI"),
				},
			},
		}
		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), podName, false)
		verifyResult(cs, podsNeededForSaturation, 1, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	It("validates that NodeSelector is respected if not matching [Conformance]", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		framework.WaitForStableCluster(cs, masterNodes)

		conf := pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "restricted"},
			NodeSelector: map[string]string{
				"label": "nonempty",
			},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), podName, false)
		verifyResult(cs, 0, 1, ns)
	})

	It("validates that a pod with an invalid NodeAffinity is rejected", func() {
		By("Trying to launch a pod with an invalid Affinity data.")
		podName := "without-label"
		_, err := cs.CoreV1().Pods(ns).Create(initPausePod(f, pausePodConfig{
			Name: podName,
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{},
							},
						},
					},
				},
			},
		}))

		if err == nil || !errors.IsInvalid(err) {
			framework.Failf("Expect error of invalid, got : %v", err)
		}
	})

	It("validates that NodeSelector is respected if matching [Conformance]", func() {
		nodeName := GetNodeThatCanRunPod(f)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		_ = createPausePod(f, pausePodConfig{
			Name: labelPodName,
			NodeSelector: map[string]string{
				"kubernetes.io/hostname": nodeName,
				k: v,
			},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.Core().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// non-nil NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.
	It("validates that NodeAffinity is respected if not matching", func() {
		By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		framework.WaitForStableCluster(cs, masterNodes)

		conf := pausePodConfig{
			Name: podName,
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "foo",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"bar", "value2"},
									},
								},
							}, {
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "diffkey",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{"wrong", "value2"},
									},
								},
							},
						},
					},
				},
			},
			Labels: map[string]string{"name": "restricted"},
		}
		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), podName, false)
		verifyResult(cs, 0, 1, ns)
	})

	// Keep the same steps with the test on NodeSelector,
	// but specify Affinity in Pod.Annotations, instead of NodeSelector.
	It("validates that required NodeAffinity setting is respected if matching", func() {
		nodeName := GetNodeThatCanRunPod(f)

		By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		_ = createPausePod(f, pausePodConfig{
			Name: labelPodName,
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
										Key:      "kubernetes.io/hostname",
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{nodeName},
									}, {
										Key:      k,
										Operator: v1.NodeSelectorOpIn,
										Values:   []string{v},
									},
								},
							},
						},
					},
				},
			},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// labelSelector Operator is DoesNotExist but values are there in requiredDuringSchedulingIgnoredDuringExecution
	// part of podAffinity,so validation fails.
	It("validates that a pod with an invalid podAffinity is rejected because of the LabelSelectorRequirement is invalid", func() {
		By("Trying to launch a pod with an invalid pod Affinity data.")
		podName := "without-label-" + string(uuid.NewUUID())
		_, err := cs.CoreV1().Pods(ns).Create(initPausePod(f, pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "without-label"},
			Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpDoesNotExist,
										Values:   []string{"securityscan"},
									},
								},
							},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			},
		}))

		if err == nil || !errors.IsInvalid(err) {
			framework.Failf("Expect error of invalid, got : %v", err)
		}
	})

	// Test Nodes does not have any pod, hence it should be impossible to schedule a Pod with pod affinity.
	It("validates that Inter-pod-Affinity is respected if not matching", func() {
		By("Trying to schedule Pod with nonempty Pod Affinity.")
		framework.WaitForStableCluster(cs, masterNodes)
		podName := "without-label-" + string(uuid.NewUUID())
		conf := pausePodConfig{
			Name: podName,
			Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "service",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"securityscan", "value2"},
									},
								},
							},
							TopologyKey: "kubernetes.io/hostname",
						},
					},
				},
			},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), podName, false)
		verifyResult(cs, 0, 1, ns)
	})

	// test the pod affinity successful matching scenario.
	It("validates that InterPodAffinity is respected if matching", func() {
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to launch the pod, now with podAffinity.")
		labelPodName := "with-podaffinity-" + string(uuid.NewUUID())
		_ = createPausePod(f, pausePodConfig{
			Name: labelPodName,
			Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S1", "value2"},
									},
								},
							},
							TopologyKey: k,
							Namespaces:  []string{ns},
						},
					},
				},
			},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
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
		defer framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, ns, "host-port")
		podList, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		Expect(len(podList.Items)).To(Equal(2))
		nodeNames := []string{podList.Items[0].Spec.NodeName, podList.Items[1].Spec.NodeName}
		Expect(nodeNames[0]).ToNot(Equal(nodeNames[1]))

		By("Applying a random label to both nodes.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "china-e2etest"
		for _, nodeName := range nodeNames {
			framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
			framework.ExpectNodeHasLabel(cs, nodeName, k, v)
			defer framework.RemoveLabelOffNode(cs, nodeName, k)
		}

		By("Trying to launch another pod on the first node with the service label.")
		podName := "with-label-" + string(uuid.NewUUID())

		runPausePod(f, pausePodConfig{
			Name:         podName,
			Labels:       map[string]string{"service": "S1"},
			NodeSelector: map[string]string{k: v}, // only launch on our two nodes
		})

		By("Trying to launch another pod, now with podAntiAffinity with same Labels.")
		labelPodName := "with-podantiaffinity-" + string(uuid.NewUUID())
		conf := pausePodConfig{
			Name:         labelPodName,
			Labels:       map[string]string{"service": "Diff"},
			NodeSelector: map[string]string{k: v}, // only launch on our two nodes, contradicting the podAntiAffinity
			Affinity: &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "service",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S1", "value2"},
									},
								},
							},
							TopologyKey: k,
							Namespaces:  []string{ns},
						},
					},
				},
			},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), labelPodName, false)
		verifyResult(cs, 3, 1, ns)
	})

	// test the pod affinity successful matching scenario with multiple Label Operators.
	It("validates that InterPodAffinity is respected if matching with multiple Affinities", func() {
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "kubernetes-e2e"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to launch the pod, now with multiple pod affinities with diff LabelOperators.")
		labelPodName := "with-podaffinity-" + string(uuid.NewUUID())
		_ = createPausePod(f, pausePodConfig{
			Name: labelPodName,
			Affinity: &v1.Affinity{
				PodAffinity: &v1.PodAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
						{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "security",
										Operator: metav1.LabelSelectorOpIn,
										Values:   []string{"S1", "value2"},
									}, {
										Key:      "security",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"S2"},
									}, {
										Key:      "security",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
							TopologyKey: k,
						},
					},
				},
			},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// test the pod affinity and anti affinity successful matching scenario.
	It("validates that InterPod Affinity and AntiAffinity is respected if matching", func() {
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a random label on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-testing"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to launch the pod, now with Pod affinity and anti affinity.")
		pod := createPodWithPodAffinity(f, k)

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, pod.Name))
		labelPod, err := cs.Core().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// Verify that an escaped JSON string of pod affinity and pod anti affinity in a YAML PodSpec works.
	It("validates that embedding the JSON PodAffinity and PodAntiAffinity setting as a string in the annotation value work", func() {
		nodeName, _ := runAndKeepPodWithLabelAndGetNodeName(f)

		By("Trying to apply a label with fake az info on the found node.")
		k := "e2e.inter-pod-affinity.kubernetes.io/zone"
		v := "e2e-az1"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		By("Trying to launch a pod that with PodAffinity & PodAntiAffinity setting as embedded JSON string in the annotation value.")
		pod := createPodWithPodAffinity(f, "kubernetes.io/hostname")
		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, pod.Name))
		labelPod, err := cs.Core().Pods(ns).Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(labelPod.Spec.NodeName).To(Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod with tolerations tolerate the taints on node,
	// and the pod's nodeName specified to the name of node found in step 1
	It("validates that taints-tolerations is respected if matching", func() {
		nodeName := getNodeThatCanRunPodWithoutToleration(f)

		By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(cs, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(cs, nodeName, labelKey)

		By("Trying to relaunch the pod, now with tolerations.")
		tolerationPodName := "with-tolerations"
		_ = createPausePod(f, pausePodConfig{
			Name:         tolerationPodName,
			Tolerations:  []v1.Toleration{{Key: testTaint.Key, Value: testTaint.Value, Effect: testTaint.Effect}},
			NodeSelector: map[string]string{labelKey: labelValue},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new taint yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(framework.WaitForPodNotPending(cs, ns, tolerationPodName))
		deployedPod, err := cs.Core().Pods(ns).Get(tolerationPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		Expect(deployedPod.Spec.NodeName).To(Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod still no tolerations,
	// and the pod's nodeName specified to the name of node found in step 1
	It("validates that taints-tolerations is respected if not matching", func() {
		nodeName := getNodeThatCanRunPodWithoutToleration(f)

		By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(cs, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(cs, nodeName, labelKey)

		By("Trying to relaunch the pod, still no tolerations.")
		podNameNoTolerations := "still-no-tolerations"
		conf := pausePodConfig{
			Name:         podNameNoTolerations,
			NodeSelector: map[string]string{labelKey: labelValue},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), podNameNoTolerations, false)
		verifyResult(cs, 0, 1, ns)

		By("Removing taint off the node")
		WaitForSchedulerAfterAction(f, removeTaintFromNodeAction(cs, nodeName, testTaint), podNameNoTolerations, true)
		verifyResult(cs, 1, 0, ns)
	})
})

func initPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        conf.Name,
			Labels:      conf.Labels,
			Annotations: conf.Annotations,
		},
		Spec: v1.PodSpec{
			NodeSelector: conf.NodeSelector,
			Affinity:     conf.Affinity,
			Containers: []v1.Container{
				{
					Name:  conf.Name,
					Image: framework.GetPauseImageName(f.ClientSet),
				},
			},
			Tolerations: conf.Tolerations,
		},
	}
	if conf.Resources != nil {
		pod.Spec.Containers[0].Resources = *conf.Resources
	}
	return pod
}

func createPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	pod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Create(initPausePod(f, conf))
	framework.ExpectNoError(err)
	return pod
}

func runPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	pod := createPausePod(f, conf)
	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(f.ClientSet, pod))
	pod, err := f.ClientSet.Core().Pods(f.Namespace.Name).Get(conf.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pod
}

func runPodAndGetNodeName(f *framework.Framework, conf pausePodConfig) string {
	// launch a pod to find a node which can launch a pod. We intentionally do
	// not just take the node list and choose the first of them. Depending on the
	// cluster and the scheduler it might be that a "normal" pod cannot be
	// scheduled onto it.
	pod := runPausePod(f, conf)

	By("Explicitly delete pod here to free the resource it takes.")
	err := f.ClientSet.Core().Pods(f.Namespace.Name).Delete(pod.Name, metav1.NewDeleteOptions(0))
	framework.ExpectNoError(err)

	return pod.Spec.NodeName
}

func createPodWithNodeAffinity(f *framework.Framework) *v1.Pod {
	return createPausePod(f, pausePodConfig{
		Name: "with-nodeaffinity-" + string(uuid.NewUUID()),
		Affinity: &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      "kubernetes.io/e2e-az-name",
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"e2e-az1", "e2e-az2"},
								},
							},
						},
					},
				},
			},
		},
	})
}

func createPodWithPodAffinity(f *framework.Framework, topologyKey string) *v1.Pod {
	return createPausePod(f, pausePodConfig{
		Name: "with-podantiaffinity-" + string(uuid.NewUUID()),
		Affinity: &v1.Affinity{
			PodAffinity: &v1.PodAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S1"},
								},
							},
						},
						TopologyKey: topologyKey,
					},
				},
			},
			PodAntiAffinity: &v1.PodAntiAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchExpressions: []metav1.LabelSelectorRequirement{
								{
									Key:      "security",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"S2"},
								},
							},
						},
						TopologyKey: topologyKey,
					},
				},
			},
		},
	})
}

// Returns a number of currently scheduled and not scheduled Pods.
func getPodsScheduled(pods *v1.PodList) (scheduledPods, notScheduledPods []v1.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				_, scheduledCondition := v1.GetPodCondition(&pod.Status, v1.PodScheduled)
				// We can't assume that the scheduledCondition is always set if Pod is assigned to Node,
				// as e.g. DaemonController doesn't set it when assigning Pod to a Node. Currently
				// Kubelet sets this condition when it gets a Pod without it, but if we were expecting
				// that it would always be not nil, this would cause a rare race condition.
				if scheduledCondition != nil {
					Expect(scheduledCondition.Status).To(Equal(v1.ConditionTrue))
				}
				scheduledPods = append(scheduledPods, pod)
			} else {
				_, scheduledCondition := v1.GetPodCondition(&pod.Status, v1.PodScheduled)
				if scheduledCondition != nil {
					Expect(scheduledCondition.Status).To(Equal(v1.ConditionFalse))
				}
				if scheduledCondition.Reason == "Unschedulable" {
					notScheduledPods = append(notScheduledPods, pod)
				}
			}
		}
	}
	return
}

func getRequestedCPU(pod v1.Pod) int64 {
	var result int64
	for _, container := range pod.Spec.Containers {
		result += container.Resources.Requests.Cpu().MilliValue()
	}
	return result
}

// removeTaintFromNodeAction returns a closure that removes the given taint
// from the given node upon invocation.
func removeTaintFromNodeAction(cs clientset.Interface, nodeName string, testTaint v1.Taint) common.Action {
	return func() error {
		framework.RemoveTaintOffNode(cs, nodeName, testTaint)
		return nil
	}
}

// createPausePodAction returns a closure that creates a pause pod upon invocation.
func createPausePodAction(f *framework.Framework, conf pausePodConfig) common.Action {
	return func() error {
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(initPausePod(f, conf))
		return err
	}
}

// WaitForSchedulerAfterAction performs the provided action and then waits for
// scheduler to act on the given pod.
func WaitForSchedulerAfterAction(f *framework.Framework, action common.Action, podName string, expectSuccess bool) {
	predicate := scheduleFailureEvent(podName)
	if expectSuccess {
		predicate = scheduleSuccessEvent(podName, "" /* any node */)
	}
	success, err := common.ObserveEventAfterAction(f, predicate, action)
	Expect(err).NotTo(HaveOccurred())
	Expect(success).To(Equal(true))
}

// TODO: upgrade calls in PodAffinity tests when we're able to run them
func verifyResult(c clientset.Interface, expectedScheduled int, expectedNotScheduled int, ns string) {
	allPods, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
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

func runAndKeepPodWithLabelAndGetNodeName(f *framework.Framework) (string, string) {
	// launch a pod to find a node which can launch a pod. We intentionally do
	// not just take the node list and choose the first of them. Depending on the
	// cluster and the scheduler it might be that a "normal" pod cannot be
	// scheduled onto it.
	By("Trying to launch a pod with a label to get a node which can launch it.")
	pod := runPausePod(f, pausePodConfig{
		Name:   "with-label-" + string(uuid.NewUUID()),
		Labels: map[string]string{"security": "S1"},
	})
	return pod.Spec.NodeName, pod.Name
}

func GetNodeThatCanRunPod(f *framework.Framework) string {
	By("Trying to launch a pod without a label to get a node which can launch it.")
	return runPodAndGetNodeName(f, pausePodConfig{Name: "without-label"})
}

func getNodeThatCanRunPodWithoutToleration(f *framework.Framework) string {
	By("Trying to launch a pod without a toleration to get a node which can launch it.")
	return runPodAndGetNodeName(f, pausePodConfig{Name: "without-toleration"})
}

func CreateHostPortPods(f *framework.Framework, id string, replicas int, expectRunning bool) {
	By(fmt.Sprintf("Running RC which reserves host port"))
	config := &testutils.RCConfig{
		Client:         f.ClientSet,
		InternalClient: f.InternalClientset,
		Name:           id,
		Namespace:      f.Namespace.Name,
		Timeout:        defaultTimeout,
		Image:          framework.GetPauseImageName(f.ClientSet),
		Replicas:       replicas,
		HostPorts:      map[string]int{"port1": 4321},
	}
	err := framework.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}
