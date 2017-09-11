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
	"encoding/json"
	"fmt"
	"math"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	_ "github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	priorityutil "k8s.io/kubernetes/plugin/pkg/scheduler/algorithm/priorities/util"
	"k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

type Resource struct {
	MilliCPU int64
	Memory   int64
}

var balancePodLabel map[string]string = map[string]string{"name": "priority-balanced-memory"}

var podRequestedResource *v1.ResourceRequirements = &v1.ResourceRequirements{
	Limits: v1.ResourceList{
		v1.ResourceMemory: resource.MustParse("100Mi"),
		v1.ResourceCPU:    resource.MustParse("100m"),
	},
	Requests: v1.ResourceList{
		v1.ResourceMemory: resource.MustParse("100Mi"),
		v1.ResourceCPU:    resource.MustParse("100m"),
	},
}

// This test suite is used to verifies scheduler priority functions based on the default provider
var _ = SIGDescribe("SchedulerPriorities [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var systemPodsNo int
	var ns string
	var masterNodes sets.String
	f := framework.NewDefaultFramework("sched-priority")
	ignoreLabels := framework.ImagePullerLabels

	AfterEach(func() {
	})

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}

		framework.WaitForAllNodesHealthy(cs, time.Minute)
		masterNodes, nodeList = framework.GetMasterAndWorkerNodesOrDie(cs)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
		err = framework.WaitForPodsRunningReady(cs, metav1.NamespaceSystem, int32(systemPodsNo), 0, framework.PodReadyBeforeTimeout, ignoreLabels)
		Expect(err).NotTo(HaveOccurred())
	})

	It("Pod should be schedule to node that don't match the PodAntiAffinity terms", func() {
		By("Trying to launch a pod with a label to get a node which can launch it.")
		pod := runPausePod(f, pausePodConfig{
			Name:   "pod-with-label-security-s1",
			Labels: map[string]string{"security": "S1"},
		})
		nodeName := pod.Spec.NodeName

		By("Trying to apply a label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", "node-topologyKey")
		v := "topologyvalue"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)
		// make the nodes have balanced cpu,mem usage
		err := createBalancedPodForNodes(f, cs, ns, nodeList.Items, podRequestedResource, 0.6)
		framework.ExpectNoError(err)
		By("Trying to launch the pod with podAntiAffinity.")
		labelPodName := "pod-with-pod-antiaffinity"
		pod = createPausePod(f, pausePodConfig{
			Resources: podRequestedResource,
			Name:      labelPodName,
			Affinity: &v1.Affinity{
				PodAntiAffinity: &v1.PodAntiAffinity{
					PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
						{
							PodAffinityTerm: v1.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "security",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"S1", "value2"},
										},
										{
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
								Namespaces:  []string{ns},
							},
							Weight: 10,
						},
					},
				},
			},
		})
		By("Wait the pod becomes running")
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		By("Verify the pod was scheduled to the expected node.")
		Expect(labelPod.Spec.NodeName).NotTo(Equal(nodeName))
	})

	It("Pod should avoid to schedule to node that have avoidPod annotation", func() {
		nodeName := nodeList.Items[0].Name
		// make the nodes have balanced cpu,mem usage
		err := createBalancedPodForNodes(f, cs, ns, nodeList.Items, podRequestedResource, 0.5)
		framework.ExpectNoError(err)
		By("Create a RC, with 0 replicas")
		rc := createRC(ns, "scheduler-priority-avoid-pod", int32(0), map[string]string{"name": "scheduler-priority-avoid-pod"}, f, podRequestedResource)
		// Cleanup the replication controller when we are done.
		defer func() {
			// Resize the replication controller to zero to get rid of pods.
			if err := framework.DeleteRCAndPods(f.ClientSet, f.InternalClientset, f.Namespace.Name, rc.Name); err != nil {
				framework.Logf("Failed to cleanup replication controller %v: %v.", rc.Name, err)
			}
		}()

		By("Trying to apply avoidPod annotations on the first node.")
		avoidPod := v1.AvoidPods{
			PreferAvoidPods: []v1.PreferAvoidPodsEntry{
				{
					PodSignature: v1.PodSignature{
						PodController: &metav1.OwnerReference{
							APIVersion: "v1",
							Kind:       "ReplicationController",
							Name:       rc.Name,
							UID:        rc.UID,
							Controller: func() *bool { b := true; return &b }(),
						},
					},
					Reason:  "some reson",
					Message: "some message",
				},
			},
		}
		action := func() error {
			framework.AddOrUpdateAvoidPodOnNode(cs, nodeName, avoidPod)
			return nil
		}
		predicate := func(node *v1.Node) bool {
			val, err := json.Marshal(avoidPod)
			if err != nil {
				return false
			}
			return node.Annotations[v1.PreferAvoidPodsAnnotationKey] == string(val)
		}
		success, err := common.ObserveNodeUpdateAfterAction(f, nodeName, predicate, action)
		Expect(err).NotTo(HaveOccurred())
		Expect(success).To(Equal(true))

		defer framework.RemoveAvoidPodsOffNode(cs, nodeName)

		By(fmt.Sprintf("Scale the RC: %s to len(nodeList.Item)-1 : %v.", rc.Name, len(nodeList.Items)-1))

		framework.ScaleRC(f.ClientSet, f.InternalClientset, ns, rc.Name, uint(len(nodeList.Items)-1), true)
		testPods, err := cs.CoreV1().Pods(ns).List(metav1.ListOptions{
			LabelSelector: "name=scheduler-priority-avoid-pod",
		})
		Expect(err).NotTo(HaveOccurred())
		By(fmt.Sprintf("Verify the pods should not scheduled to the node: %s", nodeName))
		for _, pod := range testPods.Items {
			Expect(pod.Spec.NodeName).NotTo(Equal(nodeName))
		}
	})

	It("Pod should perfer to scheduled to nodes pod can tolerate", func() {
		// make the nodes have balanced cpu,mem usage ratio
		err := createBalancedPodForNodes(f, cs, ns, nodeList.Items, podRequestedResource, 0.5)
		framework.ExpectNoError(err)
		//we need apply more taints on a node, because one match toleration only count 1
		By("Trying to apply 10 taint on the nodes except first one.")
		nodeName := nodeList.Items[0].Name

		for index, node := range nodeList.Items {
			if index == 0 {
				continue
			}
			for i := 0; i < 10; i++ {
				testTaint := addRandomTaitToNode(cs, node.Name)
				defer framework.RemoveTaintOffNode(cs, node.Name, *testTaint)
			}
		}
		By("Create a pod without any tolerations")
		tolerationPodName := "without-tolerations"
		pod := createPausePod(f, pausePodConfig{
			Name: tolerationPodName,
		})
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("Pod should prefer scheduled to the node don't have the taint.")
		tolePod, err := cs.CoreV1().Pods(ns).Get(tolerationPodName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(tolePod.Spec.NodeName).To(Equal(nodeName))

		By("Trying to apply 10 taint on the first node.")
		var tolerations []v1.Toleration
		for i := 0; i < 10; i++ {
			testTaint := addRandomTaitToNode(cs, nodeName)
			tolerations = append(tolerations, v1.Toleration{Key: testTaint.Key, Value: testTaint.Value, Effect: testTaint.Effect})
			defer framework.RemoveTaintOffNode(cs, nodeName, *testTaint)
		}
		tolerationPodName = "with-tolerations"
		By("Create a pod that tolerates all the taints of the first node.")
		pod = createPausePod(f, pausePodConfig{
			Name:        tolerationPodName,
			Tolerations: tolerations,
		})
		framework.ExpectNoError(f.WaitForPodRunning(pod.Name))

		By("Pod should prefer scheduled to the node that pod can tolerate.")
		tolePod, err = cs.CoreV1().Pods(ns).Get(tolerationPodName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(tolePod.Spec.NodeName).To(Equal(nodeName))
	})
})

// createBalancedPodForNodes creates a pod per node that asks for enough resources to make all nodes have the same mem/cpu usage ratio.
func createBalancedPodForNodes(f *framework.Framework, cs clientset.Interface, ns string, nodes []v1.Node, requestedResource *v1.ResourceRequirements, ratio float64) error {
	// find the max, if the node has the max,use the one, if not,use the ratio parameter
	var maxCPUFraction, maxMemFraction float64 = ratio, ratio
	var cpuFractionMap = make(map[string]float64)
	var memFractionMap = make(map[string]float64)
	for _, node := range nodes {
		cpuFraction, memFraction := computeCpuMemFraction(cs, node, requestedResource)
		cpuFractionMap[node.Name] = cpuFraction
		memFractionMap[node.Name] = memFraction
		if cpuFraction > maxCPUFraction {
			maxCPUFraction = cpuFraction
		}
		if memFraction > maxMemFraction {
			maxMemFraction = memFraction
		}
	}
	// we need the max one to keep the same cpu/mem use rate
	ratio = math.Max(maxCPUFraction, maxMemFraction)
	for _, node := range nodes {
		memAllocatable, found := node.Status.Allocatable[v1.ResourceMemory]
		Expect(found).To(Equal(true))
		memAllocatableVal := memAllocatable.Value()

		cpuAllocatable, found := node.Status.Allocatable[v1.ResourceCPU]
		Expect(found).To(Equal(true))
		cpuAllocatableMil := cpuAllocatable.MilliValue()

		needCreateResource := v1.ResourceList{}
		cpuFraction := cpuFractionMap[node.Name]
		memFraction := memFractionMap[node.Name]
		needCreateResource[v1.ResourceCPU] = *resource.NewMilliQuantity(int64((ratio-cpuFraction)*float64(cpuAllocatableMil)), resource.DecimalSI)

		needCreateResource[v1.ResourceMemory] = *resource.NewQuantity(int64((ratio-memFraction)*float64(memAllocatableVal)), resource.BinarySI)

		err := testutils.StartPods(cs, 1, ns, string(uuid.NewUUID()),
			*initPausePod(f, pausePodConfig{
				Name:   "",
				Labels: balancePodLabel,
				Resources: &v1.ResourceRequirements{
					Limits:   needCreateResource,
					Requests: needCreateResource,
				},
				NodeName: node.Name,
			}), true, framework.Logf)

		if err != nil {
			return err
		}
	}

	for _, node := range nodes {
		By("Compute Cpu, Mem Fraction after create balanced pods.")
		computeCpuMemFraction(cs, node, requestedResource)
	}

	return nil
}

func computeCpuMemFraction(cs clientset.Interface, node v1.Node, resource *v1.ResourceRequirements) (float64, float64) {
	framework.Logf("ComputeCpuMemFraction for node: %v", node.Name)
	totalRequestedCpuResource := resource.Requests.Cpu().MilliValue()
	totalRequestedMemResource := resource.Requests.Memory().Value()
	allpods, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
	if err != nil {
		framework.Failf("Expect error of invalid, got : %v", err)
	}
	for _, pod := range allpods.Items {
		if pod.Spec.NodeName == node.Name {
			framework.Logf("Pod for on the node: %v, Cpu: %v, Mem: %v", pod.Name, getNonZeroRequests(&pod).MilliCPU, getNonZeroRequests(&pod).Memory)
			totalRequestedCpuResource += getNonZeroRequests(&pod).MilliCPU
			totalRequestedMemResource += getNonZeroRequests(&pod).Memory
		}
	}
	cpuAllocatable, found := node.Status.Allocatable[v1.ResourceCPU]
	Expect(found).To(Equal(true))
	cpuAllocatableMil := cpuAllocatable.MilliValue()

	cpuFraction := float64(totalRequestedCpuResource) / float64(cpuAllocatableMil)
	memAllocatable, found := node.Status.Allocatable[v1.ResourceMemory]
	Expect(found).To(Equal(true))
	memAllocatableVal := memAllocatable.Value()
	memFraction := float64(totalRequestedMemResource) / float64(memAllocatableVal)

	framework.Logf("Node: %v, totalRequestedCpuResource: %v, cpuAllocatableMil: %v, cpuFraction: %v", node.Name, totalRequestedCpuResource, cpuAllocatableMil, cpuFraction)
	framework.Logf("Node: %v, totalRequestedMemResource: %v, memAllocatableVal: %v, memFraction: %v", node.Name, totalRequestedMemResource, memAllocatableVal, memFraction)

	return cpuFraction, memFraction
}

func getNonZeroRequests(pod *v1.Pod) Resource {
	result := Resource{}
	for i := range pod.Spec.Containers {
		container := &pod.Spec.Containers[i]
		cpu, memory := priorityutil.GetNonzeroRequests(&container.Resources.Requests)
		result.MilliCPU += cpu
		result.Memory += memory
	}
	return result
}

func createRC(ns, rsName string, replicas int32, rcPodLabels map[string]string, f *framework.Framework, resource *v1.ResourceRequirements) *v1.ReplicationController {
	rc := &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: rsName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: &replicas,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:      rsName,
							Image:     framework.GetPauseImageName(f.ClientSet),
							Resources: *resource,
						},
					},
				},
			},
		},
	}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(ns).Create(rc)
	Expect(err).NotTo(HaveOccurred())
	return rc
}

func addRandomTaitToNode(cs clientset.Interface, nodeName string) *v1.Taint {
	testTaint := v1.Taint{
		Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
		Value:  fmt.Sprintf("testing-taint-value-%s", string(uuid.NewUUID())),
		Effect: v1.TaintEffectPreferNoSchedule,
	}
	framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
	framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
	return &testTaint
}
