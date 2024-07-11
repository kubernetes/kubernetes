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
	"context"
	"encoding/json"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eruntimeclass "k8s.io/kubernetes/test/e2e/framework/node/runtimeclass"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	maxNumberOfPods int64 = 10
	defaultTimeout        = 3 * time.Minute
)

var localStorageVersion = utilversion.MustParseSemantic("v1.8.0-beta.0")

// variable populated in BeforeEach, never modified afterwards
var workerNodes = sets.Set[string]{}

type pausePodConfig struct {
	Name                              string
	Namespace                         string
	Finalizers                        []string
	Affinity                          *v1.Affinity
	Annotations, Labels, NodeSelector map[string]string
	Resources                         *v1.ResourceRequirements
	RuntimeClassHandler               *string
	Tolerations                       []v1.Toleration
	NodeName                          string
	Ports                             []v1.ContainerPort
	OwnerReferences                   []metav1.OwnerReference
	PriorityClassName                 string
	DeletionGracePeriodSeconds        *int64
	TopologySpreadConstraints         []v1.TopologySpreadConstraint
	SchedulingGates                   []v1.PodSchedulingGate
}

var _ = SIGDescribe("SchedulerPredicates", framework.WithSerial(), func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var RCName string
	var ns string
	f := framework.NewDefaultFramework("sched-pred")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.AfterEach(func(ctx context.Context) {
		rc, err := cs.CoreV1().ReplicationControllers(ns).Get(ctx, RCName, metav1.GetOptions{})
		if err == nil && *(rc.Spec.Replicas) != 0 {
			ginkgo.By("Cleaning up the replication controller")
			err := e2erc.DeleteRCAndWaitForGC(ctx, f.ClientSet, ns, RCName)
			framework.ExpectNoError(err)
		}
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		var err error

		e2enode.AllNodesReady(ctx, cs, time.Minute)

		nodeList, err = e2enode.GetReadySchedulableNodes(ctx, cs)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		framework.ExpectNoErrorWithOffset(0, err)
		for _, n := range nodeList.Items {
			workerNodes.Insert(n.Name)
		}

		err = framework.CheckTestingNSDeletedExcept(ctx, cs, ns)
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the apiserver thinks is on node %v before test", node.Name)
			printAllPodsOnNode(ctx, cs, node.Name)
		}

	})

	// This test verifies we don't allow scheduling of pods in a way that sum of local ephemeral storage resource requests of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	f.It("validates local ephemeral storage resource limits of pods that are allowed to run", feature.LocalStorageCapacityIsolation, func(ctx context.Context) {

		e2eskipper.SkipUnlessServerVersionGTE(localStorageVersion, f.ClientSet.Discovery())

		nodeMaxAllocatable := int64(0)

		nodeToAllocatableMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			allocatable, found := node.Status.Allocatable[v1.ResourceEphemeralStorage]
			if !found {
				framework.Failf("node.Status.Allocatable %v does not contain entry %v", node.Status.Allocatable, v1.ResourceEphemeralStorage)
			}
			nodeToAllocatableMap[node.Name] = allocatable.Value()
			if nodeMaxAllocatable < allocatable.Value() {
				nodeMaxAllocatable = allocatable.Value()
			}
		}
		WaitForStableCluster(cs, workerNodes)

		pods, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToAllocatableMap[pod.Spec.NodeName]
			if found && pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
				framework.Logf("Pod %v requesting local ephemeral resource =%v on Node %v", pod.Name, getRequestedStorageEphemeralStorage(pod), pod.Spec.NodeName)
				nodeToAllocatableMap[pod.Spec.NodeName] -= getRequestedStorageEphemeralStorage(pod)
			}
		}

		var podsNeededForSaturation int
		ephemeralStoragePerPod := nodeMaxAllocatable / maxNumberOfPods

		framework.Logf("Using pod capacity: %v", ephemeralStoragePerPod)
		for name, leftAllocatable := range nodeToAllocatableMap {
			framework.Logf("Node: %v has local ephemeral resource allocatable: %v", name, leftAllocatable)
			podsNeededForSaturation += (int)(leftAllocatable / ephemeralStoragePerPod)
		}

		ginkgo.By(fmt.Sprintf("Starting additional %v Pods to fully saturate the cluster local ephemeral resource and trying to start another one", podsNeededForSaturation))

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
							v1.ResourceEphemeralStorage: *resource.NewQuantity(ephemeralStoragePerPod, "DecimalSI"),
						},
						Requests: v1.ResourceList{
							v1.ResourceEphemeralStorage: *resource.NewQuantity(ephemeralStoragePerPod, "DecimalSI"),
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
					v1.ResourceEphemeralStorage: *resource.NewQuantity(ephemeralStoragePerPod, "DecimalSI"),
				},
				Requests: v1.ResourceList{
					v1.ResourceEphemeralStorage: *resource.NewQuantity(ephemeralStoragePerPod, "DecimalSI"),
				},
			},
		}
		WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(ctx, cs, podsNeededForSaturation, 1, ns)
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of limits +
	// associated overhead is greater than machine's capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel
	// with any other test that touches Nodes or Pods.
	// Because of this we need to have precise control on what's running in the cluster.
	// Test scenario:
	// 1. Find the first ready node on the system, and add a fake resource for test
	// 2. Create one with affinity to the particular node that uses 70% of the fake resource.
	// 3. Wait for the pod to be scheduled.
	// 4. Create another pod with affinity to the particular node that needs 20% of the fake resource and
	//    an overhead set as 25% of the fake resource.
	// 5. Make sure this additional pod is not scheduled.

	ginkgo.Context("validates pod overhead is considered along with resource limits of pods that are allowed to run", func() {
		var testNodeName string
		var handler string
		var beardsecond v1.ResourceName = "example.com/beardsecond"

		ginkgo.BeforeEach(func(ctx context.Context) {
			WaitForStableCluster(cs, workerNodes)
			ginkgo.By("Add RuntimeClass and fake resource")

			// find a node which can run a pod:
			testNodeName = GetNodeThatCanRunPod(ctx, f)

			// Get node object:
			node, err := cs.CoreV1().Nodes().Get(ctx, testNodeName, metav1.GetOptions{})
			framework.ExpectNoError(err, "unable to get node object for node %v", testNodeName)

			// update Node API object with a fake resource
			nodeCopy := node.DeepCopy()
			nodeCopy.ResourceVersion = "0"

			nodeCopy.Status.Capacity[beardsecond] = resource.MustParse("1000")
			_, err = cs.CoreV1().Nodes().UpdateStatus(ctx, nodeCopy, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "unable to apply fake resource to %v", testNodeName)

			// Register a runtimeClass with overhead set as 25% of the available beard-seconds
			handler = e2eruntimeclass.PreconfiguredRuntimeClassHandler

			rc := &nodev1.RuntimeClass{
				ObjectMeta: metav1.ObjectMeta{Name: handler},
				Handler:    handler,
				Overhead: &nodev1.Overhead{
					PodFixed: v1.ResourceList{
						beardsecond: resource.MustParse("250"),
					},
				},
			}
			_, err = cs.NodeV1().RuntimeClasses().Create(ctx, rc, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create RuntimeClass resource")
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("Remove fake resource and RuntimeClass")
			// remove fake resource:
			if testNodeName != "" {
				// Get node object:
				node, err := cs.CoreV1().Nodes().Get(ctx, testNodeName, metav1.GetOptions{})
				framework.ExpectNoError(err, "unable to get node object for node %v", testNodeName)

				nodeCopy := node.DeepCopy()
				// force it to update
				nodeCopy.ResourceVersion = "0"
				delete(nodeCopy.Status.Capacity, beardsecond)
				_, err = cs.CoreV1().Nodes().UpdateStatus(ctx, nodeCopy, metav1.UpdateOptions{})
				framework.ExpectNoError(err, "unable to update node %v", testNodeName)
			}

			// remove RuntimeClass
			_ = cs.NodeV1().RuntimeClasses().Delete(ctx, e2eruntimeclass.PreconfiguredRuntimeClassHandler, metav1.DeleteOptions{})
		})

		ginkgo.It("verify pod overhead is accounted for", func(ctx context.Context) {
			if testNodeName == "" {
				framework.Fail("unable to find a node which can run a pod")
			}

			ginkgo.By("Starting Pod to consume most of the node's resource.")

			// Create pod which requires 70% of the available beard-seconds.
			fillerPod := createPausePod(ctx, f, pausePodConfig{
				Name: "filler-pod-" + string(uuid.NewUUID()),
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{beardsecond: resource.MustParse("700")},
					Limits:   v1.ResourceList{beardsecond: resource.MustParse("700")},
				},
			})

			// Wait for filler pod to schedule.
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, fillerPod))

			ginkgo.By("Creating another pod that requires unavailable amount of resources.")
			// Create another pod that requires 20% of available beard-seconds, but utilizes the RuntimeClass
			// which defines a pod overhead that requires an additional 25%.
			// This pod should remain pending as at least 70% of beard-second in
			// the node are already consumed.
			podName := "additional-pod" + string(uuid.NewUUID())
			conf := pausePodConfig{
				RuntimeClassHandler: &handler,
				Name:                podName,
				Labels:              map[string]string{"name": "additional"},
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{beardsecond: resource.MustParse("200")},
				},
			}

			WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podName, false)
			verifyResult(ctx, cs, 1, 1, ns)
		})
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of
	// resource requests of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel
	// with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	// Test scenario:
	// 1. Find the amount CPU resources on each node.
	// 2. Create one pod with affinity to each node that uses 70% of the node CPU.
	// 3. Wait for the pods to be scheduled.
	// 4. Create another pod with no affinity to any node that need 50% of the largest node CPU.
	// 5. Make sure this additional pod is not scheduled.
	/*
		Release: v1.9
		Testname: Scheduler, resource limits
		Description: Scheduling Pods MUST fail if the resource requests exceed Machine capacity.
	*/
	framework.ConformanceIt("validates resource limits of pods that are allowed to run", func(ctx context.Context) {
		WaitForStableCluster(cs, workerNodes)
		nodeMaxAllocatable := int64(0)
		nodeToAllocatableMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			nodeReady := false
			for _, condition := range node.Status.Conditions {
				if condition.Type == v1.NodeReady && condition.Status == v1.ConditionTrue {
					nodeReady = true
					break
				}
			}
			if !nodeReady {
				continue
			}
			// Apply node label to each node
			e2enode.AddOrUpdateLabelOnNode(cs, node.Name, "node", node.Name)
			e2enode.ExpectNodeHasLabel(ctx, cs, node.Name, "node", node.Name)
			// Find allocatable amount of CPU.
			allocatable, found := node.Status.Allocatable[v1.ResourceCPU]
			if !found {
				framework.Failf("node.Status.Allocatable %v does not contain entry %v", node.Status.Allocatable, v1.ResourceCPU)
			}
			nodeToAllocatableMap[node.Name] = allocatable.MilliValue()
			if nodeMaxAllocatable < allocatable.MilliValue() {
				nodeMaxAllocatable = allocatable.MilliValue()
			}
		}
		// Clean up added labels after this test.
		defer func() {
			for nodeName := range nodeToAllocatableMap {
				e2enode.RemoveLabelOffNode(cs, nodeName, "node")
			}
		}()

		pods, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToAllocatableMap[pod.Spec.NodeName]
			if found && pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
				framework.Logf("Pod %v requesting resource cpu=%vm on Node %v", pod.Name, getRequestedCPU(pod), pod.Spec.NodeName)
				nodeToAllocatableMap[pod.Spec.NodeName] -= getRequestedCPU(pod)
			}
		}

		ginkgo.By("Starting Pods to consume most of the cluster CPU.")
		// Create one pod per node that requires 70% of the node remaining CPU.
		fillerPods := []*v1.Pod{}
		for nodeName, cpu := range nodeToAllocatableMap {
			requestedCPU := cpu * 7 / 10
			framework.Logf("Creating a pod which consumes cpu=%vm on Node %v", requestedCPU, nodeName)
			fillerPods = append(fillerPods, createPausePod(ctx, f, pausePodConfig{
				Name: "filler-pod-" + string(uuid.NewUUID()),
				Resources: &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(requestedCPU, "DecimalSI"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU: *resource.NewMilliQuantity(requestedCPU, "DecimalSI"),
					},
				},
				Affinity: &v1.Affinity{
					NodeAffinity: &v1.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
							NodeSelectorTerms: []v1.NodeSelectorTerm{
								{
									MatchExpressions: []v1.NodeSelectorRequirement{
										{
											Key:      "node",
											Operator: v1.NodeSelectorOpIn,
											Values:   []string{nodeName},
										},
									},
								},
							},
						},
					},
				},
			}))
		}
		// Wait for filler pods to schedule.
		for _, pod := range fillerPods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pod))
		}
		ginkgo.By("Creating another pod that requires unavailable amount of CPU.")
		// Create another pod that requires 50% of the largest node CPU resources.
		// This pod should remain pending as at least 70% of CPU of other nodes in
		// the cluster are already consumed.
		podName := "additional-pod"
		conf := pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "additional"},
			Resources: &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(nodeMaxAllocatable*5/10, "DecimalSI"),
				},
				Requests: v1.ResourceList{
					v1.ResourceCPU: *resource.NewMilliQuantity(nodeMaxAllocatable*5/10, "DecimalSI"),
				},
			},
		}
		WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(ctx, cs, len(fillerPods), 1, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	/*
		Release: v1.9
		Testname: Scheduler, node selector not matching
		Description: Create a Pod with a NodeSelector set to a value that does not match a node in the cluster. Since there are no nodes matching the criteria the Pod MUST not be scheduled.
	*/
	framework.ConformanceIt("validates that NodeSelector is respected if not matching", func(ctx context.Context) {
		ginkgo.By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		WaitForStableCluster(cs, workerNodes)

		conf := pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "restricted"},
			NodeSelector: map[string]string{
				"label": "nonempty",
			},
		}

		WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(ctx, cs, 0, 1, ns)
	})

	/*
		Release: v1.9
		Testname: Scheduler, node selector matching
		Description: Create a label on the node {k: v}. Then create a Pod with a NodeSelector set to {k: v}. Check to see if the Pod is scheduled. When the NodeSelector matches then Pod MUST be scheduled on that node.
	*/
	framework.ConformanceIt("validates that NodeSelector is respected if matching", func(ctx context.Context) {
		nodeName := GetNodeThatCanRunPod(ctx, f)

		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, k, v)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		createPausePod(ctx, f, pausePodConfig{
			Name: labelPodName,
			NodeSelector: map[string]string{
				k: v,
			},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new label yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(ctx, labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(labelPod.Spec.NodeName).To(gomega.Equal(nodeName))
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// non-nil NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.
	ginkgo.It("validates that NodeAffinity is respected if not matching", func(ctx context.Context) {
		ginkgo.By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		WaitForStableCluster(cs, workerNodes)

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
		WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(ctx, cs, 0, 1, ns)
	})

	// Keep the same steps with the test on NodeSelector,
	// but specify Affinity in Pod.Spec.Affinity, instead of NodeSelector.
	ginkgo.It("validates that required NodeAffinity setting is respected if matching", func(ctx context.Context) {
		nodeName := GetNodeThatCanRunPod(ctx, f)

		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, k, v)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		createPausePod(ctx, f, pausePodConfig{
			Name: labelPodName,
			Affinity: &v1.Affinity{
				NodeAffinity: &v1.NodeAffinity{
					RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
						NodeSelectorTerms: []v1.NodeSelectorTerm{
							{
								MatchExpressions: []v1.NodeSelectorRequirement{
									{
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
		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(ctx, labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(labelPod.Spec.NodeName).To(gomega.Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod with tolerations tolerate the taints on node,
	// and the pod's nodeName specified to the name of node found in step 1
	ginkgo.It("validates that taints-tolerations is respected if matching", func(ctx context.Context) {
		nodeName := getNodeThatCanRunPodWithoutToleration(ctx, f)

		ginkgo.By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		e2enode.AddOrUpdateTaintOnNode(ctx, cs, nodeName, testTaint)
		e2enode.ExpectNodeHasTaint(ctx, cs, nodeName, &testTaint)
		ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, cs, nodeName, testTaint)

		ginkgo.By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, labelKey, labelValue)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, labelKey)

		ginkgo.By("Trying to relaunch the pod, now with tolerations.")
		tolerationPodName := "with-tolerations"
		createPausePod(ctx, f, pausePodConfig{
			Name:         tolerationPodName,
			Tolerations:  []v1.Toleration{{Key: testTaint.Key, Value: testTaint.Value, Effect: testTaint.Effect}},
			NodeSelector: map[string]string{labelKey: labelValue},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new taint yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, cs, ns, tolerationPodName))
		deployedPod, err := cs.CoreV1().Pods(ns).Get(ctx, tolerationPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(deployedPod.Spec.NodeName).To(gomega.Equal(nodeName))
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod still no tolerations,
	// and the pod's nodeName specified to the name of node found in step 1
	ginkgo.It("validates that taints-tolerations is respected if not matching", func(ctx context.Context) {
		nodeName := getNodeThatCanRunPodWithoutToleration(ctx, f)

		ginkgo.By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		e2enode.AddOrUpdateTaintOnNode(ctx, cs, nodeName, testTaint)
		e2enode.ExpectNodeHasTaint(ctx, cs, nodeName, &testTaint)
		ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, cs, nodeName, testTaint)

		ginkgo.By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, labelKey, labelValue)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, labelKey)

		ginkgo.By("Trying to relaunch the pod, still no tolerations.")
		podNameNoTolerations := "still-no-tolerations"
		conf := pausePodConfig{
			Name:         podNameNoTolerations,
			NodeSelector: map[string]string{labelKey: labelValue},
		}

		WaitForSchedulerAfterAction(ctx, f, createPausePodAction(f, conf), ns, podNameNoTolerations, false)
		verifyResult(ctx, cs, 0, 1, ns)

		ginkgo.By("Removing taint off the node")
		WaitForSchedulerAfterAction(ctx, f, removeTaintFromNodeAction(cs, nodeName, testTaint), ns, podNameNoTolerations, true)
		verifyResult(ctx, cs, 1, 0, ns)
	})

	ginkgo.It("validates that there is no conflict between pods with same hostPort but different hostIP and protocol", func(ctx context.Context) {

		nodeName := GetNodeThatCanRunPod(ctx, f)
		localhost := "127.0.0.1"
		if framework.TestContext.ClusterIsIPv6() {
			localhost = "::1"
		}
		hostIP := getNodeHostIP(ctx, f, nodeName)

		// use nodeSelector to make sure the testing pods get assigned on the same node to explicitly verify there exists conflict or not
		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "90"

		nodeSelector := make(map[string]string)
		nodeSelector[k] = v

		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, k, v)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, k)

		port := int32(54321)
		ginkgo.By(fmt.Sprintf("Trying to create a pod(pod1) with hostport %v and hostIP %s and expect scheduled", port, localhost))
		createHostPortPodOnNode(ctx, f, "pod1", ns, localhost, port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create another pod(pod2) with hostport %v but hostIP %s on the node which pod1 resides and expect scheduled", port, hostIP))
		createHostPortPodOnNode(ctx, f, "pod2", ns, hostIP, port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create a third pod(pod3) with hostport %v, hostIP %s but use UDP protocol on the node which pod2 resides", port, hostIP))
		createHostPortPodOnNode(ctx, f, "pod3", ns, hostIP, port, v1.ProtocolUDP, nodeSelector, true)

	})

	/*
		Release: v1.16
		Testname: Scheduling, HostPort and Protocol match, HostIPs different but one is default HostIP (0.0.0.0)
		Description: Pods with the same HostPort and Protocol, but different HostIPs, MUST NOT schedule to the
		same node if one of those IPs is the default HostIP of 0.0.0.0, which represents all IPs on the host.
	*/
	framework.ConformanceIt("validates that there exists conflict between pods with same hostPort and protocol but one using 0.0.0.0 hostIP", func(ctx context.Context) {
		nodeName := GetNodeThatCanRunPod(ctx, f)
		hostIP := getNodeHostIP(ctx, f, nodeName)
		// use nodeSelector to make sure the testing pods get assigned on the same node to explicitly verify there exists conflict or not
		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "95"

		nodeSelector := make(map[string]string)
		nodeSelector[k] = v

		e2enode.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		e2enode.ExpectNodeHasLabel(ctx, cs, nodeName, k, v)
		defer e2enode.RemoveLabelOffNode(cs, nodeName, k)

		port := int32(54322)
		ginkgo.By(fmt.Sprintf("Trying to create a pod(pod4) with hostport %v and hostIP 0.0.0.0(empty string here) and expect scheduled", port))
		createHostPortPodOnNode(ctx, f, "pod4", ns, "", port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create another pod(pod5) with hostport %v but hostIP %s on the node which pod4 resides and expect not scheduled", port, hostIP))
		createHostPortPodOnNode(ctx, f, "pod5", ns, hostIP, port, v1.ProtocolTCP, nodeSelector, false)
	})

	ginkgo.Context("PodTopologySpread Filtering", func() {
		var nodeNames []string
		topologyKey := "kubernetes.io/e2e-pts-filter"

		ginkgo.BeforeEach(func(ctx context.Context) {
			if len(nodeList.Items) < 2 {
				ginkgo.Skip("At least 2 nodes are required to run the test")
			}
			ginkgo.By("Trying to get 2 available nodes which can run pod")
			nodeNames = Get2NodesThatCanRunPod(ctx, f)
			ginkgo.By(fmt.Sprintf("Apply dedicated topologyKey %v for this test on the 2 nodes.", topologyKey))
			for _, nodeName := range nodeNames {
				e2enode.AddOrUpdateLabelOnNode(cs, nodeName, topologyKey, nodeName)
			}
		})
		ginkgo.AfterEach(func() {
			for _, nodeName := range nodeNames {
				e2enode.RemoveLabelOffNode(cs, nodeName, topologyKey)
			}
		})

		ginkgo.It("validates 4 pods with MaxSkew=1 are evenly distributed into 2 nodes", func(ctx context.Context) {
			podLabel := "e2e-pts-filter"
			replicas := 4
			rsConfig := pauseRSConfig{
				Replicas: int32(replicas),
				PodConfig: pausePodConfig{
					Name:      podLabel,
					Namespace: ns,
					Labels:    map[string]string{podLabel: ""},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchExpressions: []v1.NodeSelectorRequirement{
											{
												Key:      topologyKey,
												Operator: v1.NodeSelectorOpIn,
												Values:   nodeNames,
											},
										},
									},
								},
							},
						},
					},
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           1,
							TopologyKey:       topologyKey,
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      podLabel,
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
					},
				},
			}
			runPauseRS(ctx, f, rsConfig)
			podList, err := cs.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)
			numInNode1, numInNode2 := 0, 0
			for _, pod := range podList.Items {
				if pod.Spec.NodeName == nodeNames[0] {
					numInNode1++
				} else if pod.Spec.NodeName == nodeNames[1] {
					numInNode2++
				}
			}
			expected := replicas / len(nodeNames)
			gomega.Expect(numInNode1).To(gomega.Equal(expected), fmt.Sprintf("Pods are not distributed as expected on node %q", nodeNames[0]))
			gomega.Expect(numInNode2).To(gomega.Equal(expected), fmt.Sprintf("Pods are not distributed as expected on node %q", nodeNames[1]))
		})
	})

	ginkgo.It("validates Pods with non-empty schedulingGates are blocked on scheduling", func(ctx context.Context) {
		podLabel := "e2e-scheduling-gates"
		replicas := 3
		ginkgo.By(fmt.Sprintf("Creating a ReplicaSet with replicas=%v, carrying scheduling gates [foo bar]", replicas))
		rsConfig := pauseRSConfig{
			Replicas: int32(replicas),
			PodConfig: pausePodConfig{
				Name:      podLabel,
				Namespace: ns,
				Labels:    map[string]string{podLabel: ""},
				SchedulingGates: []v1.PodSchedulingGate{
					{Name: "foo"},
					{Name: "bar"},
				},
			},
		}
		createPauseRS(ctx, f, rsConfig)

		ginkgo.By("Expect all pods stay in pending state")
		podList, err := e2epod.WaitForNumberOfPods(ctx, cs, ns, replicas, time.Minute)
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForPodsSchedulingGated(ctx, cs, ns, replicas, time.Minute))

		ginkgo.By("Remove one scheduling gate")
		want := []v1.PodSchedulingGate{{Name: "bar"}}
		var pods []*v1.Pod
		for _, pod := range podList.Items {
			clone := pod.DeepCopy()
			clone.Spec.SchedulingGates = want
			live, err := patchPod(cs, &pod, clone)
			framework.ExpectNoError(err)
			pods = append(pods, live)
		}

		ginkgo.By("Expect all pods carry one scheduling gate and are still in pending state")
		framework.ExpectNoError(e2epod.WaitForPodsWithSchedulingGates(ctx, cs, ns, replicas, time.Minute, want))
		framework.ExpectNoError(e2epod.WaitForPodsSchedulingGated(ctx, cs, ns, replicas, time.Minute))

		ginkgo.By("Remove the remaining scheduling gates")
		for _, pod := range pods {
			clone := pod.DeepCopy()
			clone.Spec.SchedulingGates = nil
			_, err := patchPod(cs, pod, clone)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Expect all pods are scheduled and running")
		framework.ExpectNoError(e2epod.WaitForPodsRunning(ctx, cs, ns, replicas, time.Minute))
	})

	// Regression test for an extended scenario for https://issues.k8s.io/123465
	ginkgo.It("when PVC has node-affinity to non-existent/illegal nodes, the pod should be scheduled normally if suitable nodes exist", func(ctx context.Context) {
		nodeName := GetNodeThatCanRunPod(ctx, f)
		nonExistentNodeName1 := string(uuid.NewUUID())
		nonExistentNodeName2 := string(uuid.NewUUID())
		hostLabel := "kubernetes.io/hostname"
		localPath := "/tmp"
		podName := "bind-pv-with-non-existent-nodes"
		pvcName := "pvc-" + string(uuid.NewUUID())
		_, pvc, err := e2epv.CreatePVPVC(ctx, cs, f.Timeouts, e2epv.PersistentVolumeConfig{
			PVSource: v1.PersistentVolumeSource{
				Local: &v1.LocalVolumeSource{
					Path: localPath,
				},
			},
			Prebind: &v1.PersistentVolumeClaim{
				ObjectMeta: metav1.ObjectMeta{Name: pvcName, Namespace: ns},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      hostLabel,
									Operator: v1.NodeSelectorOpIn,
									// add non-existent nodes to the list
									Values: []string{nodeName, nonExistentNodeName1, nonExistentNodeName2},
								},
							},
						},
					},
				},
			},
		}, e2epv.PersistentVolumeClaimConfig{
			Name: pvcName,
		}, ns, true)
		framework.ExpectNoError(err)
		bindPvPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: imageutils.GetE2EImage(imageutils.Pause),
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "data",
								MountPath: "/tmp",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "data",
						VolumeSource: v1.VolumeSource{
							PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
								ClaimName: pvc.Name,
							},
						},
					},
				},
			},
		}
		_, err = f.ClientSet.CoreV1().Pods(ns).Create(ctx, bindPvPod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, f.ClientSet, ns, podName))
	})
})

func patchPod(cs clientset.Interface, old, new *v1.Pod) (*v1.Pod, error) {
	oldData, err := json.Marshal(old)
	if err != nil {
		return nil, err
	}

	newData, err := json.Marshal(new)
	if err != nil {
		return nil, err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Pod{})
	if err != nil {
		return nil, fmt.Errorf("failed to create merge patch for Pod %q: %w", old.Name, err)
	}
	return cs.CoreV1().Pods(new.Namespace).Patch(context.TODO(), new.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
}

// printAllPodsOnNode outputs status of all kubelet pods into log.
func printAllPodsOnNode(ctx context.Context, c clientset.Interface, nodeName string) {
	podList, err := c.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{FieldSelector: "spec.nodeName=" + nodeName})
	if err != nil {
		framework.Logf("Unable to retrieve pods for node %v: %v", nodeName, err)
		return
	}
	for _, p := range podList.Items {
		framework.Logf("%v from %v started at %v (%d container statuses recorded)", p.Name, p.Namespace, p.Status.StartTime, len(p.Status.ContainerStatuses))
		for _, c := range p.Status.ContainerStatuses {
			framework.Logf("\tContainer %v ready: %v, restart count %v",
				c.Name, c.Ready, c.RestartCount)
		}
	}
}

func initPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	var gracePeriod = int64(1)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            conf.Name,
			Namespace:       conf.Namespace,
			Labels:          map[string]string{},
			Annotations:     map[string]string{},
			OwnerReferences: conf.OwnerReferences,
			Finalizers:      conf.Finalizers,
		},
		Spec: v1.PodSpec{
			SecurityContext:           e2epod.GetRestrictedPodSecurityContext(),
			NodeSelector:              conf.NodeSelector,
			Affinity:                  conf.Affinity,
			TopologySpreadConstraints: conf.TopologySpreadConstraints,
			RuntimeClassName:          conf.RuntimeClassHandler,
			Containers: []v1.Container{
				{
					Name:            conf.Name,
					Image:           imageutils.GetPauseImageName(),
					Ports:           conf.Ports,
					SecurityContext: e2epod.GetRestrictedContainerSecurityContext(),
				},
			},
			Tolerations:                   conf.Tolerations,
			PriorityClassName:             conf.PriorityClassName,
			TerminationGracePeriodSeconds: &gracePeriod,
			SchedulingGates:               conf.SchedulingGates,
		},
	}
	for key, value := range conf.Labels {
		pod.ObjectMeta.Labels[key] = value
	}
	for key, value := range conf.Annotations {
		pod.ObjectMeta.Annotations[key] = value
	}
	// TODO: setting the Pod's nodeAffinity instead of setting .spec.nodeName works around the
	// Preemption e2e flake (#88441), but we should investigate deeper to get to the bottom of it.
	if len(conf.NodeName) != 0 {
		e2epod.SetNodeAffinity(&pod.Spec, conf.NodeName)
	}
	if conf.Resources != nil {
		pod.Spec.Containers[0].Resources = *conf.Resources
	}
	if conf.DeletionGracePeriodSeconds != nil {
		pod.ObjectMeta.DeletionGracePeriodSeconds = conf.DeletionGracePeriodSeconds
	}
	return pod
}

func createPausePod(ctx context.Context, f *framework.Framework, conf pausePodConfig) *v1.Pod {
	namespace := conf.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, initPausePod(f, conf), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return pod
}

func runPausePod(ctx context.Context, f *framework.Framework, conf pausePodConfig) *v1.Pod {
	pod := createPausePod(ctx, f, conf)
	framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, pod.Name, pod.Namespace, f.Timeouts.PodStartShort))
	pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, conf.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pod
}

func runPodAndGetNodeName(ctx context.Context, f *framework.Framework, conf pausePodConfig) string {
	// launch a pod to find a node which can launch a pod. We intentionally do
	// not just take the node list and choose the first of them. Depending on the
	// cluster and the scheduler it might be that a "normal" pod cannot be
	// scheduled onto it.
	pod := runPausePod(ctx, f, conf)

	ginkgo.By("Explicitly delete pod here to free the resource it takes.")
	err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(ctx, pod.Name, *metav1.NewDeleteOptions(0))
	framework.ExpectNoError(err)

	return pod.Spec.NodeName
}

func getRequestedCPU(pod v1.Pod) int64 {
	var result int64
	for _, container := range pod.Spec.Containers {
		result += container.Resources.Requests.Cpu().MilliValue()
	}
	return result
}

func getRequestedStorageEphemeralStorage(pod v1.Pod) int64 {
	var result int64
	for _, container := range pod.Spec.Containers {
		result += container.Resources.Requests.StorageEphemeral().Value()
	}
	return result
}

// removeTaintFromNodeAction returns a closure that removes the given taint
// from the given node upon invocation.
func removeTaintFromNodeAction(cs clientset.Interface, nodeName string, testTaint v1.Taint) Action {
	return func(ctx context.Context) error {
		e2enode.RemoveTaintOffNode(ctx, cs, nodeName, testTaint)
		return nil
	}
}

// createPausePodAction returns a closure that creates a pause pod upon invocation.
func createPausePodAction(f *framework.Framework, conf pausePodConfig) Action {
	return func(ctx context.Context) error {
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, initPausePod(f, conf), metav1.CreateOptions{})
		return err
	}
}

// WaitForSchedulerAfterAction performs the provided action and then waits for
// scheduler to act on the given pod.
func WaitForSchedulerAfterAction(ctx context.Context, f *framework.Framework, action Action, ns, podName string, expectSuccess bool) {
	predicate := scheduleFailureEvent(podName)
	if expectSuccess {
		predicate = scheduleSuccessEvent(ns, podName, "" /* any node */)
	}
	observed, err := observeEventAfterAction(ctx, f.ClientSet, f.Namespace.Name, predicate, action)
	framework.ExpectNoError(err)
	if expectSuccess && !observed {
		framework.Failf("Did not observe success event after performing the supplied action for pod %v", podName)
	}
	if !expectSuccess && !observed {
		framework.Failf("Did not observe failed event after performing the supplied action for pod %v", podName)
	}
}

// TODO: upgrade calls in PodAffinity tests when we're able to run them
func verifyResult(ctx context.Context, c clientset.Interface, expectedScheduled int, expectedNotScheduled int, ns string) {
	allPods, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)
	scheduledPods, notScheduledPods := GetPodsScheduled(workerNodes, allPods)

	gomega.Expect(notScheduledPods).To(gomega.HaveLen(expectedNotScheduled), fmt.Sprintf("Not scheduled Pods: %#v", notScheduledPods))
	gomega.Expect(scheduledPods).To(gomega.HaveLen(expectedScheduled), fmt.Sprintf("Scheduled Pods: %#v", scheduledPods))
}

// GetNodeThatCanRunPod trying to launch a pod without a label to get a node which can launch it
func GetNodeThatCanRunPod(ctx context.Context, f *framework.Framework) string {
	ginkgo.By("Trying to launch a pod without a label to get a node which can launch it.")
	return runPodAndGetNodeName(ctx, f, pausePodConfig{Name: "without-label"})
}

// Get2NodesThatCanRunPod return a 2-node slice where can run pod.
func Get2NodesThatCanRunPod(ctx context.Context, f *framework.Framework) []string {
	firstNode := GetNodeThatCanRunPod(ctx, f)
	ginkgo.By("Trying to launch a pod without a label to get a node which can launch it.")
	pod := pausePodConfig{
		Name: "without-label",
		Affinity: &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchFields: []v1.NodeSelectorRequirement{
								{Key: "metadata.name", Operator: v1.NodeSelectorOpNotIn, Values: []string{firstNode}},
							},
						},
					},
				},
			},
		},
	}
	secondNode := runPodAndGetNodeName(ctx, f, pod)
	return []string{firstNode, secondNode}
}

func getNodeThatCanRunPodWithoutToleration(ctx context.Context, f *framework.Framework) string {
	ginkgo.By("Trying to launch a pod without a toleration to get a node which can launch it.")
	return runPodAndGetNodeName(ctx, f, pausePodConfig{Name: "without-toleration"})
}

// CreateHostPortPods creates RC with host port 4321
func CreateHostPortPods(ctx context.Context, f *framework.Framework, id string, replicas int, expectRunning bool) {
	ginkgo.By("Running RC which reserves host port")
	config := &testutils.RCConfig{
		Client:    f.ClientSet,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   defaultTimeout,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  replicas,
		HostPorts: map[string]int{"port1": 4321},
	}
	err := e2erc.RunRC(ctx, *config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}

// CreateNodeSelectorPods creates RC with host port 4321 and defines node selector
func CreateNodeSelectorPods(ctx context.Context, f *framework.Framework, id string, replicas int, nodeSelector map[string]string, expectRunning bool) error {
	ginkgo.By("Running RC which reserves host port and defines node selector")

	config := &testutils.RCConfig{
		Client:       f.ClientSet,
		Name:         id,
		Namespace:    f.Namespace.Name,
		Timeout:      defaultTimeout,
		Image:        imageutils.GetPauseImageName(),
		Replicas:     replicas,
		HostPorts:    map[string]int{"port1": 4321},
		NodeSelector: nodeSelector,
	}
	err := e2erc.RunRC(ctx, *config)
	if expectRunning {
		return err
	}
	return nil
}

// create pod which using hostport on the specified node according to the nodeSelector
// it starts an http server on the exposed port
func createHostPortPodOnNode(ctx context.Context, f *framework.Framework, podName, ns, hostIP string, port int32, protocol v1.Protocol, nodeSelector map[string]string, expectScheduled bool) {
	hostPortPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "agnhost",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"netexec", "--http-port=8080", "--udp-port=8080"},
					Ports: []v1.ContainerPort{
						{
							HostPort:      port,
							ContainerPort: 8080,
							Protocol:      protocol,
							HostIP:        hostIP,
						},
					},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path: "/hostname",
								Port: intstr.IntOrString{
									IntVal: int32(8080),
								},
								Scheme: v1.URISchemeHTTP,
							},
						},
					},
				},
			},
			NodeSelector: nodeSelector,
		},
	}
	_, err := f.ClientSet.CoreV1().Pods(ns).Create(ctx, hostPortPod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = e2epod.WaitForPodNotPending(ctx, f.ClientSet, ns, podName)
	if expectScheduled {
		framework.ExpectNoError(err)
	}
}

// GetPodsScheduled returns a number of currently scheduled and not scheduled Pods on worker nodes.
func GetPodsScheduled(workerNodes sets.Set[string], pods *v1.PodList) (scheduledPods, notScheduledPods []v1.Pod) {
	for _, pod := range pods.Items {
		if pod.Spec.NodeName != "" && workerNodes.Has(pod.Spec.NodeName) {
			_, scheduledCondition := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
			if scheduledCondition == nil {
				framework.Failf("Did not find 'scheduled' condition for pod %+v", podName)
			}
			if scheduledCondition.Status != v1.ConditionTrue {
				framework.Failf("PodStatus isn't 'true' for pod %+v", podName)
			}
			scheduledPods = append(scheduledPods, pod)
		} else if pod.Spec.NodeName == "" {
			notScheduledPods = append(notScheduledPods, pod)
		}
	}
	return
}

// getNodeHostIP returns the first internal IP on the node matching the main Cluster IP family
func getNodeHostIP(ctx context.Context, f *framework.Framework, nodeName string) string {
	// Get the internal HostIP of the node
	family := v1.IPv4Protocol
	if framework.TestContext.ClusterIsIPv6() {
		family = v1.IPv6Protocol
	}
	node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ips := e2enode.GetAddressesByTypeAndFamily(node, v1.NodeInternalIP, family)
	gomega.Expect(ips).ToNot(gomega.BeEmpty())
	return ips[0]
}
