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
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2erc "k8s.io/kubernetes/test/e2e/framework/rc"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	k8utilnet "k8s.io/utils/net"

	"github.com/onsi/ginkgo"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

const (
	maxNumberOfPods int64 = 10
	defaultTimeout        = 3 * time.Minute
)

var localStorageVersion = utilversion.MustParseSemantic("v1.8.0-beta.0")

// variable set in BeforeEach, never modified afterwards
var masterNodes sets.String

type pausePodConfig struct {
	Name                              string
	Namespace                         string
	Affinity                          *v1.Affinity
	Annotations, Labels, NodeSelector map[string]string
	Resources                         *v1.ResourceRequirements
	Tolerations                       []v1.Toleration
	NodeName                          string
	Ports                             []v1.ContainerPort
	OwnerReferences                   []metav1.OwnerReference
	PriorityClassName                 string
	DeletionGracePeriodSeconds        *int64
}

var _ = SIGDescribe("SchedulerPredicates [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var RCName string
	var ns string
	f := framework.NewDefaultFramework("sched-pred")

	ginkgo.AfterEach(func() {
		rc, err := cs.CoreV1().ReplicationControllers(ns).Get(RCName, metav1.GetOptions{})
		if err == nil && *(rc.Spec.Replicas) != 0 {
			ginkgo.By("Cleaning up the replication controller")
			err := e2erc.DeleteRCAndWaitForGC(f.ClientSet, ns, RCName)
			framework.ExpectNoError(err)
		}
	})

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		var err error

		framework.AllNodesReady(cs, time.Minute)

		// NOTE: Here doesn't get nodeList for supporting a master nodes which can host workload pods.
		masterNodes, _, err = e2enode.GetMasterAndWorkerNodes(cs)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		nodeList, err = e2enode.GetReadySchedulableNodes(cs)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}

		// TODO: write a wrapper for ExpectNoErrorWithOffset()
		framework.ExpectNoErrorWithOffset(0, err)

		err = framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)

		for _, node := range nodeList.Items {
			framework.Logf("\nLogging pods the kubelet thinks is on node %v before test", node.Name)
			printAllKubeletPods(cs, node.Name)
		}

	})

	// This test verifies we don't allow scheduling of pods in a way that sum of local ephemeral storage limits of pods is greater than machines capacity.
	// It assumes that cluster add-on pods stay stable and cannot be run in parallel with any other test that touches Nodes or Pods.
	// It is so because we need to have precise control on what's running in the cluster.
	ginkgo.It("validates local ephemeral storage resource limits of pods that are allowed to run [Feature:LocalStorageCapacityIsolation]", func() {

		e2eskipper.SkipUnlessServerVersionGTE(localStorageVersion, f.ClientSet.Discovery())

		nodeMaxAllocatable := int64(0)

		nodeToAllocatableMap := make(map[string]int64)
		for _, node := range nodeList.Items {
			allocatable, found := node.Status.Allocatable[v1.ResourceEphemeralStorage]
			framework.ExpectEqual(found, true)
			nodeToAllocatableMap[node.Name] = allocatable.Value()
			if nodeMaxAllocatable < allocatable.Value() {
				nodeMaxAllocatable = allocatable.Value()
			}
		}
		WaitForStableCluster(cs, masterNodes)

		pods, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, pod := range pods.Items {
			_, found := nodeToAllocatableMap[pod.Spec.NodeName]
			if found && pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
				framework.Logf("Pod %v requesting local ephemeral resource =%v on Node %v", pod.Name, getRequestedStorageEphemeralStorage(pod), pod.Spec.NodeName)
				nodeToAllocatableMap[pod.Spec.NodeName] -= getRequestedStorageEphemeralStorage(pod)
			}
		}

		var podsNeededForSaturation int
		var ephemeralStoragePerPod int64

		ephemeralStoragePerPod = nodeMaxAllocatable / maxNumberOfPods

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
			},
		}
		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(cs, podsNeededForSaturation, 1, ns)
	})

	// This test verifies we don't allow scheduling of pods in a way that sum of
	// limits of pods is greater than machines capacity.
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
		Release : v1.9
		Testname: Scheduler, resource limits
		Description: Scheduling Pods MUST fail if the resource limits exceed Machine capacity.
	*/
	framework.ConformanceIt("validates resource limits of pods that are allowed to run ", func() {
		WaitForStableCluster(cs, masterNodes)
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
			framework.AddOrUpdateLabelOnNode(cs, node.Name, "node", node.Name)
			framework.ExpectNodeHasLabel(cs, node.Name, "node", node.Name)
			// Find allocatable amount of CPU.
			allocatable, found := node.Status.Allocatable[v1.ResourceCPU]
			framework.ExpectEqual(found, true)
			nodeToAllocatableMap[node.Name] = allocatable.MilliValue()
			if nodeMaxAllocatable < allocatable.MilliValue() {
				nodeMaxAllocatable = allocatable.MilliValue()
			}
		}
		// Clean up added labels after this test.
		defer func() {
			for nodeName := range nodeToAllocatableMap {
				framework.RemoveLabelOffNode(cs, nodeName, "node")
			}
		}()

		pods, err := cs.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
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
			fillerPods = append(fillerPods, createPausePod(f, pausePodConfig{
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
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
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
			},
		}
		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(cs, len(fillerPods), 1, ns)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// nonempty Selector set.
	/*
		Release : v1.9
		Testname: Scheduler, node selector not matching
		Description: Create a Pod with a NodeSelector set to a value that does not match a node in the cluster. Since there are no nodes matching the criteria the Pod MUST not be scheduled.
	*/
	framework.ConformanceIt("validates that NodeSelector is respected if not matching ", func() {
		ginkgo.By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		WaitForStableCluster(cs, masterNodes)

		conf := pausePodConfig{
			Name:   podName,
			Labels: map[string]string{"name": "restricted"},
			NodeSelector: map[string]string{
				"label": "nonempty",
			},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(cs, 0, 1, ns)
	})

	/*
		Release : v1.9
		Testname: Scheduler, node selector matching
		Description: Create a label on the node {k: v}. Then create a Pod with a NodeSelector set to {k: v}. Check to see if the Pod is scheduled. When the NodeSelector matches then Pod MUST be scheduled on that node.
	*/
	framework.ConformanceIt("validates that NodeSelector is respected if matching ", func() {
		nodeName := GetNodeThatCanRunPod(f)

		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		createPausePod(f, pausePodConfig{
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
		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(labelPod.Spec.NodeName, nodeName)
	})

	// Test Nodes does not have any label, hence it should be impossible to schedule Pod with
	// non-nil NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.
	ginkgo.It("validates that NodeAffinity is respected if not matching", func() {
		ginkgo.By("Trying to schedule Pod with nonempty NodeSelector.")
		podName := "restricted-pod"

		WaitForStableCluster(cs, masterNodes)

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
		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), ns, podName, false)
		verifyResult(cs, 0, 1, ns)
	})

	// Keep the same steps with the test on NodeSelector,
	// but specify Affinity in Pod.Spec.Affinity, instead of NodeSelector.
	ginkgo.It("validates that required NodeAffinity setting is respected if matching", func() {
		nodeName := GetNodeThatCanRunPod(f)

		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "42"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Trying to relaunch the pod, now with labels.")
		labelPodName := "with-labels"
		createPausePod(f, pausePodConfig{
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
		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, labelPodName))
		labelPod, err := cs.CoreV1().Pods(ns).Get(labelPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(labelPod.Spec.NodeName, nodeName)
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod with tolerations tolerate the taints on node,
	// and the pod's nodeName specified to the name of node found in step 1
	ginkgo.It("validates that taints-tolerations is respected if matching", func() {
		nodeName := getNodeThatCanRunPodWithoutToleration(f)

		ginkgo.By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		ginkgo.By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(cs, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(cs, nodeName, labelKey)

		ginkgo.By("Trying to relaunch the pod, now with tolerations.")
		tolerationPodName := "with-tolerations"
		createPausePod(f, pausePodConfig{
			Name:         tolerationPodName,
			Tolerations:  []v1.Toleration{{Key: testTaint.Key, Value: testTaint.Value, Effect: testTaint.Effect}},
			NodeSelector: map[string]string{labelKey: labelValue},
		})

		// check that pod got scheduled. We intentionally DO NOT check that the
		// pod is running because this will create a race condition with the
		// kubelet and the scheduler: the scheduler might have scheduled a pod
		// already when the kubelet does not know about its new taint yet. The
		// kubelet will then refuse to launch the pod.
		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, tolerationPodName))
		deployedPod, err := cs.CoreV1().Pods(ns).Get(tolerationPodName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(deployedPod.Spec.NodeName, nodeName)
	})

	// 1. Run a pod to get an available node, then delete the pod
	// 2. Taint the node with a random taint
	// 3. Try to relaunch the pod still no tolerations,
	// and the pod's nodeName specified to the name of node found in step 1
	ginkgo.It("validates that taints-tolerations is respected if not matching", func() {
		nodeName := getNodeThatCanRunPodWithoutToleration(f)

		ginkgo.By("Trying to apply a random taint on the found node.")
		testTaint := v1.Taint{
			Key:    fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID())),
			Value:  "testing-taint-value",
			Effect: v1.TaintEffectNoSchedule,
		}
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		ginkgo.By("Trying to apply a random label on the found node.")
		labelKey := fmt.Sprintf("kubernetes.io/e2e-label-key-%s", string(uuid.NewUUID()))
		labelValue := "testing-label-value"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, labelKey, labelValue)
		framework.ExpectNodeHasLabel(cs, nodeName, labelKey, labelValue)
		defer framework.RemoveLabelOffNode(cs, nodeName, labelKey)

		ginkgo.By("Trying to relaunch the pod, still no tolerations.")
		podNameNoTolerations := "still-no-tolerations"
		conf := pausePodConfig{
			Name:         podNameNoTolerations,
			NodeSelector: map[string]string{labelKey: labelValue},
		}

		WaitForSchedulerAfterAction(f, createPausePodAction(f, conf), ns, podNameNoTolerations, false)
		verifyResult(cs, 0, 1, ns)

		ginkgo.By("Removing taint off the node")
		WaitForSchedulerAfterAction(f, removeTaintFromNodeAction(cs, nodeName, testTaint), ns, podNameNoTolerations, true)
		verifyResult(cs, 1, 0, ns)
	})

	/*
		Release : v1.16
		Testname: Scheduling, HostPort matching and HostIP and Protocol not-matching
		Description: Pods with the same HostPort value MUST be able to be scheduled to the same node
		if the HostIP or Protocol is different.
	*/
	framework.ConformanceIt("validates that there is no conflict between pods with same hostPort but different hostIP and protocol", func() {

		nodeName := GetNodeThatCanRunPod(f)

		// use nodeSelector to make sure the testing pods get assigned on the same node to explicitly verify there exists conflict or not
		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "90"

		nodeSelector := make(map[string]string)
		nodeSelector[k] = v

		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		port := int32(54321)
		ginkgo.By(fmt.Sprintf("Trying to create a pod(pod1) with hostport %v and hostIP 127.0.0.1 and expect scheduled", port))
		createHostPortPodOnNode(f, "pod1", ns, "127.0.0.1", port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create another pod(pod2) with hostport %v but hostIP 127.0.0.2 on the node which pod1 resides and expect scheduled", port))
		createHostPortPodOnNode(f, "pod2", ns, "127.0.0.2", port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create a third pod(pod3) with hostport %v, hostIP 127.0.0.2 but use UDP protocol on the node which pod2 resides", port))
		createHostPortPodOnNode(f, "pod3", ns, "127.0.0.2", port, v1.ProtocolUDP, nodeSelector, true)
	})

	/*
		Release : v1.16
		Testname: Scheduling, HostPort and Protocol match, HostIPs different but one is default HostIP (0.0.0.0)
		Description: Pods with the same HostPort and Protocol, but different HostIPs, MUST NOT schedule to the
		same node if one of those IPs is the default HostIP of 0.0.0.0, which represents all IPs on the host.
	*/
	framework.ConformanceIt("validates that there exists conflict between pods with same hostPort and protocol but one using 0.0.0.0 hostIP", func() {
		nodeName := GetNodeThatCanRunPod(f)

		// use nodeSelector to make sure the testing pods get assigned on the same node to explicitly verify there exists conflict or not
		ginkgo.By("Trying to apply a random label on the found node.")
		k := fmt.Sprintf("kubernetes.io/e2e-%s", string(uuid.NewUUID()))
		v := "95"

		nodeSelector := make(map[string]string)
		nodeSelector[k] = v

		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		port := int32(54322)
		ginkgo.By(fmt.Sprintf("Trying to create a pod(pod4) with hostport %v and hostIP 0.0.0.0(empty string here) and expect scheduled", port))
		createHostPortPodOnNode(f, "pod4", ns, "", port, v1.ProtocolTCP, nodeSelector, true)

		ginkgo.By(fmt.Sprintf("Trying to create another pod(pod5) with hostport %v but hostIP 127.0.0.1 on the node which pod4 resides and expect not scheduled", port))
		createHostPortPodOnNode(f, "pod5", ns, "127.0.0.1", port, v1.ProtocolTCP, nodeSelector, false)
	})
})

// printAllKubeletPods outputs status of all kubelet pods into log.
func printAllKubeletPods(c clientset.Interface, nodeName string) {
	podList, err := e2ekubelet.GetKubeletPods(c, nodeName)
	if err != nil {
		framework.Logf("Unable to retrieve kubelet pods for node %v: %v", nodeName, err)
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
			Labels:          conf.Labels,
			Annotations:     conf.Annotations,
			OwnerReferences: conf.OwnerReferences,
		},
		Spec: v1.PodSpec{
			NodeSelector: conf.NodeSelector,
			Affinity:     conf.Affinity,
			Containers: []v1.Container{
				{
					Name:  conf.Name,
					Image: imageutils.GetPauseImageName(),
					Ports: conf.Ports,
				},
			},
			Tolerations:                   conf.Tolerations,
			NodeName:                      conf.NodeName,
			PriorityClassName:             conf.PriorityClassName,
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	if conf.Resources != nil {
		pod.Spec.Containers[0].Resources = *conf.Resources
	}
	if conf.DeletionGracePeriodSeconds != nil {
		pod.ObjectMeta.DeletionGracePeriodSeconds = conf.DeletionGracePeriodSeconds
	}
	return pod
}

func createPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	namespace := conf.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Create(initPausePod(f, conf))
	framework.ExpectNoError(err)
	return pod
}

func runPausePod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	pod := createPausePod(f, conf)
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(f.ClientSet, pod))
	pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(conf.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pod
}

func runPodAndGetNodeName(f *framework.Framework, conf pausePodConfig) string {
	// launch a pod to find a node which can launch a pod. We intentionally do
	// not just take the node list and choose the first of them. Depending on the
	// cluster and the scheduler it might be that a "normal" pod cannot be
	// scheduled onto it.
	pod := runPausePod(f, conf)

	ginkgo.By("Explicitly delete pod here to free the resource it takes.")
	err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(pod.Name, metav1.NewDeleteOptions(0))
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
func removeTaintFromNodeAction(cs clientset.Interface, nodeName string, testTaint v1.Taint) e2eevents.Action {
	return func() error {
		framework.RemoveTaintOffNode(cs, nodeName, testTaint)
		return nil
	}
}

// createPausePodAction returns a closure that creates a pause pod upon invocation.
func createPausePodAction(f *framework.Framework, conf pausePodConfig) e2eevents.Action {
	return func() error {
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(initPausePod(f, conf))
		return err
	}
}

// WaitForSchedulerAfterAction performs the provided action and then waits for
// scheduler to act on the given pod.
func WaitForSchedulerAfterAction(f *framework.Framework, action e2eevents.Action, ns, podName string, expectSuccess bool) {
	predicate := scheduleFailureEvent(podName)
	if expectSuccess {
		predicate = scheduleSuccessEvent(ns, podName, "" /* any node */)
	}
	success, err := e2eevents.ObserveEventAfterAction(f.ClientSet, f.Namespace.Name, predicate, action)
	framework.ExpectNoError(err)
	framework.ExpectEqual(success, true)
}

// TODO: upgrade calls in PodAffinity tests when we're able to run them
func verifyResult(c clientset.Interface, expectedScheduled int, expectedNotScheduled int, ns string) {
	allPods, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
	framework.ExpectNoError(err)
	scheduledPods, notScheduledPods := GetPodsScheduled(masterNodes, allPods)

	framework.ExpectEqual(len(notScheduledPods), expectedNotScheduled, fmt.Sprintf("Not scheduled Pods: %#v", notScheduledPods))
	framework.ExpectEqual(len(scheduledPods), expectedScheduled, fmt.Sprintf("Scheduled Pods: %#v", scheduledPods))
}

// GetNodeThatCanRunPod trying to launch a pod without a label to get a node which can launch it
func GetNodeThatCanRunPod(f *framework.Framework) string {
	ginkgo.By("Trying to launch a pod without a label to get a node which can launch it.")
	return runPodAndGetNodeName(f, pausePodConfig{Name: "without-label"})
}

func getNodeThatCanRunPodWithoutToleration(f *framework.Framework) string {
	ginkgo.By("Trying to launch a pod without a toleration to get a node which can launch it.")
	return runPodAndGetNodeName(f, pausePodConfig{Name: "without-toleration"})
}

// CreateHostPortPods creates RC with host port 4321
func CreateHostPortPods(f *framework.Framework, id string, replicas int, expectRunning bool) {
	ginkgo.By(fmt.Sprintf("Running RC which reserves host port"))
	config := &testutils.RCConfig{
		Client:    f.ClientSet,
		Name:      id,
		Namespace: f.Namespace.Name,
		Timeout:   defaultTimeout,
		Image:     imageutils.GetPauseImageName(),
		Replicas:  replicas,
		HostPorts: map[string]int{"port1": 4321},
	}
	err := e2erc.RunRC(*config)
	if expectRunning {
		framework.ExpectNoError(err)
	}
}

// CreateNodeSelectorPods creates RC with host port 4321 and defines node selector
func CreateNodeSelectorPods(f *framework.Framework, id string, replicas int, nodeSelector map[string]string, expectRunning bool) error {
	ginkgo.By(fmt.Sprintf("Running RC which reserves host port and defines node selector"))

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
	err := e2erc.RunRC(*config)
	if expectRunning {
		return err
	}
	return nil
}

// create pod which using hostport on the specified node according to the nodeSelector
func createHostPortPodOnNode(f *framework.Framework, podName, ns, hostIP string, port int32, protocol v1.Protocol, nodeSelector map[string]string, expectScheduled bool) {
	hostIP = translateIPv4ToIPv6(hostIP)
	createPausePod(f, pausePodConfig{
		Name: podName,
		Ports: []v1.ContainerPort{
			{
				HostPort:      port,
				ContainerPort: 80,
				Protocol:      protocol,
				HostIP:        hostIP,
			},
		},
		NodeSelector: nodeSelector,
	})

	err := e2epod.WaitForPodNotPending(f.ClientSet, ns, podName)
	if expectScheduled {
		framework.ExpectNoError(err)
	}
}

// translateIPv4ToIPv6 maps an IPv4 address into a valid IPv6 address
// adding the well known prefix "0::ffff:" https://tools.ietf.org/html/rfc2765
// if the ip is IPv4 and the cluster IPFamily is IPv6, otherwise returns the same ip
func translateIPv4ToIPv6(ip string) string {
	if framework.TestContext.IPFamily == "ipv6" && !k8utilnet.IsIPv6String(ip) && ip != "" {
		ip = "0::ffff:" + ip
	}
	return ip
}

// GetPodsScheduled returns a number of currently scheduled and not scheduled Pods.
func GetPodsScheduled(masterNodes sets.String, pods *v1.PodList) (scheduledPods, notScheduledPods []v1.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				_, scheduledCondition := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
				framework.ExpectEqual(scheduledCondition != nil, true)
				framework.ExpectEqual(scheduledCondition.Status, v1.ConditionTrue)
				scheduledPods = append(scheduledPods, pod)
			} else {
				_, scheduledCondition := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
				framework.ExpectEqual(scheduledCondition != nil, true)
				framework.ExpectEqual(scheduledCondition.Status, v1.ConditionFalse)
				if scheduledCondition.Reason == "Unschedulable" {

					notScheduledPods = append(notScheduledPods, pod)
				}
			}
		}
	}
	return
}
