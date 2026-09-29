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
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"github.com/google/uuid"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	policyv1 "k8s.io/api/policy/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/features"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

type priorityPair struct {
	name  string
	value int32
}

var testExtendedResource = v1.ResourceName("scheduling.k8s.io/foo")

const (
	testFinalizer = "example.com/test-finalizer"
)

var _ = SIGDescribe("SchedulerPreemption", framework.WithSerial(), func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var ns string
	f := framework.NewDefaultFramework("sched-preemption")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	lowPriority, mediumPriority, highPriority := int32(1), int32(100), int32(1000)
	lowPriorityClassName := f.BaseName + "-low-priority"
	mediumPriorityClassName := f.BaseName + "-medium-priority"
	highPriorityClassName := f.BaseName + "-high-priority"
	priorityPairs := []priorityPair{
		{name: lowPriorityClassName, value: lowPriority},
		{name: mediumPriorityClassName, value: mediumPriority},
		{name: highPriorityClassName, value: highPriority},
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		for _, pair := range priorityPairs {
			_ = cs.SchedulingV1().PriorityClasses().Delete(ctx, pair.name, *metav1.NewDeleteOptions(0))
		}
		for _, node := range nodeList.Items {
			e2enode.RemoveExtendedResource(ctx, cs, node.Name, testExtendedResource)
		}
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		var err error
		for _, pair := range priorityPairs {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: pair.name}, Value: pair.value}, metav1.CreateOptions{})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
			}
		}

		e2enode.WaitForTotalHealthy(ctx, cs, time.Minute)
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
	})

	/*
		Release: v1.19
		Testname: Scheduler, Basic Preemption
		Description: When a higher priority pod is created and no node with enough
		resources is found, the scheduler MUST preempt a lower priority pod and
		schedule the high priority pod.
	*/
	framework.ConformanceIt("validates basic preemption works", func(ctx context.Context) {
		var podRes v1.ResourceList

		// Create two pods per node that uses a lot of the node's resources.
		ginkgo.By("Create pods that use 4/5 of node resources.")
		pods := make([]*v1.Pod, 0, 2*len(nodeList.Items))
		// Create pods in the cluster.
		// One of them has low priority, making it the victim for preemption.
		for i, node := range nodeList.Items {
			// Update each node to advertise 3 available extended resources
			e2enode.AddExtendedResource(ctx, cs, node.Name, testExtendedResource, resource.MustParse("5"))

			for j := range 2 {
				// Request 2 of the available resources for the victim pods
				podRes = v1.ResourceList{}
				podRes[testExtendedResource] = resource.MustParse("2")

				// make the first pod low priority and the rest medium priority.
				priorityName := mediumPriorityClassName
				if len(pods) == 0 {
					priorityName = lowPriorityClassName
				}
				pausePod := createPausePod(ctx, f, pausePodConfig{
					Name:              fmt.Sprintf("pod%d-%d-%v", i, j, priorityName),
					PriorityClassName: priorityName,
					Resources: &v1.ResourceRequirements{
						Requests: podRes,
						Limits:   podRes,
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{node.Name}},
										},
									},
								},
							},
						},
					},
				})
				pods = append(pods, pausePod)
				framework.Logf("Created pod: %v", pausePod.Name)
			}
		}
		if len(pods) < 2 {
			framework.Failf("We need at least two pods to be created but " +
				"all nodes are already heavily utilized, so preemption tests cannot be run")
		}
		ginkgo.By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pod))
		}

		// Set the pod request to the first pod's resources (should be low priority pod)
		podRes = pods[0].Spec.Containers[0].Resources.Requests

		ginkgo.By("Run a high priority pod that has same requirements as that of lower priority pod")
		// Create a high priority pod and make sure it is scheduled on the same node as the low priority pod.
		runPausePod(ctx, f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
		})

		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(ctx, pods[0].Name, metav1.GetOptions{})
		podPreempted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		if !podPreempted {
			framework.Failf("expected pod to be preempted, instead got pod %+v and error %v", preemptedPod, err)
		}
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(ctx, pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil())
		}
	})

	/*
		Release: v1.19
		Testname: Scheduler, Preemption for critical pod
		Description: When a critical pod is created and no node with enough
		resources is found, the scheduler MUST preempt a lower priority pod to
		schedule the critical pod.
	*/
	framework.ConformanceIt("validates lower priority pod preemption by critical pod", func(ctx context.Context) {
		var podRes v1.ResourceList

		ginkgo.By("Create pods that use 4/5 of node resources.")
		pods := make([]*v1.Pod, 0, len(nodeList.Items))
		for i, node := range nodeList.Items {
			// Update each node to advertise 3 available extended resources
			e2enode.AddExtendedResource(ctx, cs, node.Name, testExtendedResource, resource.MustParse("5"))

			for j := range 2 {
				// Request 2 of the available resources for the victim pods
				podRes = v1.ResourceList{}
				podRes[testExtendedResource] = resource.MustParse("2")

				// make the first pod low priority and the rest medium priority.
				priorityName := mediumPriorityClassName
				if len(pods) == 0 {
					priorityName = lowPriorityClassName
				}
				pausePod := createPausePod(ctx, f, pausePodConfig{
					Name:              fmt.Sprintf("pod%d-%d-%v", i, j, priorityName),
					PriorityClassName: priorityName,
					Resources: &v1.ResourceRequirements{
						Requests: podRes,
						Limits:   podRes,
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{node.Name}},
										},
									},
								},
							},
						},
					},
				})
				pods = append(pods, pausePod)
				framework.Logf("Created pod: %v", pausePod.Name)
			}
		}
		if len(pods) < 2 {
			framework.Failf("We need at least two pods to be created but " +
				"all nodes are already heavily utilized, so preemption tests cannot be run")
		}
		ginkgo.By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pod))
		}

		// We want this pod to be preempted
		podRes = pods[0].Spec.Containers[0].Resources.Requests
		ginkgo.By("Run a critical pod that use same resources as that of a lower priority pod")
		// Create a critical pod and make sure it is scheduled.
		defer func() {
			// Clean-up the critical pod
			// Always run cleanup to make sure the pod is properly cleaned up.
			err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(ctx, "critical-pod", *metav1.NewDeleteOptions(0))
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error cleanup pod `%s/%s`: %v", metav1.NamespaceSystem, "critical-pod", err)
			}
		}()
		runPausePod(ctx, f, pausePodConfig{
			Name:              "critical-pod",
			Namespace:         metav1.NamespaceSystem,
			PriorityClassName: scheduling.SystemClusterCritical,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
		})

		defer func() {
			// Clean-up the critical pod
			err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(ctx, "critical-pod", *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)
		}()
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(ctx, pods[0].Name, metav1.GetOptions{})
		podPreempted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(ctx, pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil())
		}

		if !podPreempted {
			framework.Failf("expected pod to be preempted, instead got pod %+v and error %v", preemptedPod, err)
		}
	})

	/*
		Testname: Scheduler, PodDisruptionBudget compliance with empty selector across namespaces
		Description: When an empty selector PodDisruptionBudget protects all workloads in a namespace,
		including unlabeled pods, the scheduler MUST respect disruption budgets during preemption:
		1. When budget is exhausted (minAvailable == total), lower-priority pods covered by the empty selector
		   PDB must not be disrupted, and higher-priority workloads in other namespaces without PDBs are preempted instead.
		2. When disruption budget becomes available, preemption proceeds against the lower-priority unlabeled pods.
	*/
	framework.It("validates PodDisruptionBudget compliance and empty selector protection across namespaces during preemption", func(ctx context.Context) {
		ginkgo.By("Selecting a node and advertising extended resources")
		node := nodeList.Items[0]
		e2enode.AddExtendedResource(ctx, cs, node.Name, testExtendedResource, resource.MustParse("10"))

		nsA := f.Namespace.Name
		nsBObj, err := f.CreateNamespace(ctx, f.BaseName+"-pdb-b", nil)
		framework.ExpectNoError(err)
		nsB := nsBObj.Name

		testNodeAffinity := &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchFields: []v1.NodeSelectorRequirement{
								{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{node.Name}},
							},
						},
					},
				},
			},
		}

		twoRes := v1.ResourceList{testExtendedResource: resource.MustParse("2")}
		twoResReq := &v1.ResourceRequirements{Requests: twoRes, Limits: twoRes}

		ginkgo.By("Creating 2 unlabeled low-priority pods in namespace A")
		lowPodA1 := createPausePod(ctx, f, pausePodConfig{
			Name:              "unlabeled-low-pod-a1",
			Namespace:         nsA,
			PriorityClassName: lowPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})
		lowPodA2 := createPausePod(ctx, f, pausePodConfig{
			Name:              "unlabeled-low-pod-a2",
			Namespace:         nsA,
			PriorityClassName: lowPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})

		ginkgo.By("Creating 2 medium-priority pods in namespace B")
		midPodB1 := createPausePod(ctx, f, pausePodConfig{
			Name:              "mid-pod-b1",
			Namespace:         nsB,
			PriorityClassName: mediumPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})
		midPodB2 := createPausePod(ctx, f, pausePodConfig{
			Name:              "mid-pod-b2",
			Namespace:         nsB,
			PriorityClassName: mediumPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})

		ginkgo.By("Creating 1 high-priority filler pod in namespace A to exhaust node capacity")
		fillerPod := createPausePod(ctx, f, pausePodConfig{
			Name:              "filler-pod",
			Namespace:         nsA,
			PriorityClassName: highPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})

		ginkgo.By("Waiting for existing pods to be running")
		for _, p := range []*v1.Pod{lowPodA1, lowPodA2, midPodB1, midPodB2, fillerPod} {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, p))
		}

		ginkgo.By("Creating an empty selector PodDisruptionBudget in namespace A with minAvailable=3")
		minAvail3 := intstr.FromInt32(3)
		pdbA := &policyv1.PodDisruptionBudget{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "empty-selector-pdb-a",
				Namespace: nsA,
			},
			Spec: policyv1.PodDisruptionBudgetSpec{
				MinAvailable: &minAvail3,
				Selector:     &metav1.LabelSelector{},
			},
		}
		pdbA, err = cs.PolicyV1().PodDisruptionBudgets(nsA).Create(ctx, pdbA, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for PDB status to reflect 3 healthy pods and 0 disruptions allowed")
		err = wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, false, func(ctx context.Context) (bool, error) {
			curPDB, err := cs.PolicyV1().PodDisruptionBudgets(nsA).Get(ctx, pdbA.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return curPDB.Status.CurrentHealthy == 3 && curPDB.Status.DisruptionsAllowed == 0, nil
		})
		framework.ExpectNoError(err, "PDB in namespace A failed to stabilize with 0 disruptions allowed")

		ginkgo.By("Creating high-priority preemptor1 in namespace A; it should preempt a medium-priority pod from namespace B rather than violate PDB on lower-priority pods in namespace A")
		preemptor1 := createPausePod(ctx, f, pausePodConfig{
			Name:              "preemptor1",
			Namespace:         nsA,
			PriorityClassName: highPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, preemptor1))

		ginkgo.By("Verifying low-priority pods in namespace A were protected and not preempted")
		for _, pod := range []*v1.Pod{lowPodA1, lowPodA2, fillerPod} {
			livePod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil(), fmt.Sprintf("expected pod %s/%s to remain un-preempted", pod.Namespace, pod.Name))
		}

		ginkgo.By("Verifying at least one medium-priority pod in namespace B was preempted")
		b1, err1 := cs.CoreV1().Pods(nsB).Get(ctx, midPodB1.Name, metav1.GetOptions{})
		b2, err2 := cs.CoreV1().Pods(nsB).Get(ctx, midPodB2.Name, metav1.GetOptions{})
		b1Preempted := (err1 != nil && apierrors.IsNotFound(err1)) || (err1 == nil && b1.DeletionTimestamp != nil)
		b2Preempted := (err2 != nil && apierrors.IsNotFound(err2)) || (err2 == nil && b2.DeletionTimestamp != nil)
		gomega.Expect(b1Preempted || b2Preempted).To(gomega.BeTrue(), "expected at least one medium-priority pod in namespace B to be preempted")

		ginkgo.By("Updating PDB in namespace A to minAvailable=2 so disruption budget becomes available")
		curPDB, err := cs.PolicyV1().PodDisruptionBudgets(nsA).Get(ctx, pdbA.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		minAvail2 := intstr.FromInt32(2)
		curPDB.Spec.MinAvailable = &minAvail2
		_, err = cs.PolicyV1().PodDisruptionBudgets(nsA).Update(ctx, curPDB, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		err = wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, false, func(ctx context.Context) (bool, error) {
			p, err := cs.PolicyV1().PodDisruptionBudgets(nsA).Get(ctx, pdbA.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			return p.Status.DisruptionsAllowed >= 1, nil
		})
		framework.ExpectNoError(err, "PDB in namespace A failed to reflect available disruption budget")

		ginkgo.By("Creating high-priority preemptor2 in namespace A; it should now preempt a low-priority unlabeled pod in namespace A")
		preemptor2 := createPausePod(ctx, f, pausePodConfig{
			Name:              "preemptor2",
			Namespace:         nsA,
			PriorityClassName: highPriorityClassName,
			Resources:         twoResReq,
			Affinity:          testNodeAffinity,
		})
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, preemptor2))

		ginkgo.By("Verifying that a low-priority pod in namespace A was preempted")
		a1, errA1 := cs.CoreV1().Pods(nsA).Get(ctx, lowPodA1.Name, metav1.GetOptions{})
		a2, errA2 := cs.CoreV1().Pods(nsA).Get(ctx, lowPodA2.Name, metav1.GetOptions{})
		a1Preempted := (errA1 != nil && apierrors.IsNotFound(errA1)) || (errA1 == nil && a1.DeletionTimestamp != nil)
		a2Preempted := (errA2 != nil && apierrors.IsNotFound(errA2)) || (errA2 == nil && a2.DeletionTimestamp != nil)
		gomega.Expect(a1Preempted || a2Preempted).To(gomega.BeTrue(), "expected at least one low-priority pod in namespace A to be preempted when PDB budget is available")
	})

	/*
		Release: v1.32
		Testname: Scheduler runs the preemption with various priority classes expectedly
		Description: When there are Pods with various priority classes running the preemption,
		the scheduler must prioritize the Pods with the higher priority class.
	*/
	framework.It("validates various priority Pods preempt expectedly with the async preemption", feature.SchedulerAsyncPreemption, framework.WithFeatureGate(features.SchedulerAsyncPreemption), func(ctx context.Context) {
		var podRes v1.ResourceList
		// Create 10 pods per node that will eat up all the node's resources.
		ginkgo.By("Create 10 low-priority pods on each node.")
		nodeListLen := len(nodeList.Items)
		lowPriorityPods := make([]*v1.Pod, 0, 10*nodeListLen)
		// Create pods in the cluster.
		for i, node := range nodeList.Items {
			// Update each node to advertise 3 available extended resources
			e2enode.AddExtendedResource(ctx, cs, node.Name, testExtendedResource, resource.MustParse("10"))

			// Create 10 low priority pods on each node, which will use up 10/10 of the node's resources.
			for j := range 10 {
				// Request 1 of the available resources for the victim pods
				podRes = v1.ResourceList{}
				podRes[testExtendedResource] = resource.MustParse("1")
				pausePod := createPausePod(ctx, f, pausePodConfig{
					Name:              fmt.Sprintf("pod%d-%d-%v", i, j, lowPriorityClassName),
					PriorityClassName: lowPriorityClassName,
					Resources: &v1.ResourceRequirements{
						Requests: podRes,
						Limits:   podRes,
					},
					Affinity: &v1.Affinity{
						NodeAffinity: &v1.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
								NodeSelectorTerms: []v1.NodeSelectorTerm{
									{
										MatchFields: []v1.NodeSelectorRequirement{
											{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{node.Name}},
										},
									},
								},
							},
						},
					},
					// This victim pod will be preempted by the high priority pod.
					// But, the deletion will be blocked by the preStop hook with
					// TerminationGracePeriodSeconds set.
					//
					// The preStop hook + TerminationGracePeriodSeconds are needed to prevent the medium Pods
					// from being scheduled instead of the high Pods,
					// depending on when the scheduler notices the existence of all the high Pods we create.
					TerminationGracePeriodSeconds: ptr.To[int64](80),
					PreStopHookSleepSeconds:       ptr.To[int64](79),
				})
				lowPriorityPods = append(lowPriorityPods, pausePod)
				framework.Logf("Created pod: %v", pausePod.Name)
			}
		}

		ginkgo.By("Wait for lower priority pods to be scheduled.")
		for _, pod := range lowPriorityPods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pod))
		}

		highPriorityPods := make([]*v1.Pod, 0, 5*nodeListLen)
		mediumPriorityPods := make([]*v1.Pod, 0, 10*nodeListLen)

		ginkgo.By("Run medium priority pods that have same requirements as that of lower priority pod")
		for i := range nodeList.Items {
			// Create medium priority pods first
			// to confirm the scheduler finally prioritize the high priority pods, ignoring the medium priority pods.
			for j := range 10 {
				// 5 pods per node will be unschedulable
				// because the node only has 10 resource, and high priority pods will use 5 resource.
				p := createPausePod(ctx, f, pausePodConfig{
					Name:              fmt.Sprintf("pod%d-%d-%v", i, j, mediumPriorityClassName),
					PriorityClassName: mediumPriorityClassName,
					Resources: &v1.ResourceRequirements{
						// Set the pod request to the low priority pod's resources
						Requests: lowPriorityPods[0].Spec.Containers[0].Resources.Requests,
						Limits:   lowPriorityPods[0].Spec.Containers[0].Resources.Requests,
					},
				})
				mediumPriorityPods = append(mediumPriorityPods, p)
			}
		}

		// All low priority Pods should be the target of preemption.
		// Those Pods have a finalizer and hence should not be deleted yet at this point.
		ginkgo.By("Check all low priority pods to be about to preempted.")
		for _, pod := range lowPriorityPods {
			framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
				preemptedPod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return preemptedPod.DeletionTimestamp != nil, nil
			}))
		}

		ginkgo.By("Run high priority pods that have same requirements as that of lower priority pod")
		for i := range nodeList.Items {
			for j := range 5 {
				p := createPausePod(ctx, f, pausePodConfig{
					Name:              fmt.Sprintf("pod%d-%d-%v", i, j, highPriorityClassName),
					PriorityClassName: highPriorityClassName,
					Resources: &v1.ResourceRequirements{
						// Set the pod request to the low priority pod's resources
						Requests: lowPriorityPods[0].Spec.Containers[0].Resources.Requests,
						Limits:   lowPriorityPods[0].Spec.Containers[0].Resources.Requests,
					},
				})
				highPriorityPods = append(highPriorityPods, p)
			}
		}

		// All high priority Pods should be schedulable by removing the low priority Pods.
		ginkgo.By("Wait for high priority pods to be ready for the preemption.")
		for _, pod := range highPriorityPods {
			framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
				highPod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				return highPod.Status.NominatedNodeName != "", nil
			}))
		}

		ginkgo.By("Delete all low priority pods to proceed the preemption faster.")
		for _, pod := range lowPriorityPods {
			err := cs.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](0)})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Logf("Deleting %v pod failed: %v", pod.Name, err)
			}
		}

		ginkgo.By("Wait for high priority pods to be scheduled.")
		for _, pod := range highPriorityPods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, pod))
		}

		ginkgo.By(fmt.Sprintf("Wait for %v medium priority pods to be scheduled.", 5*nodeListLen))
		framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, framework.PodStartTimeout, false, func(ctx context.Context) (bool, error) {
			scheduled := 0
			for _, pod := range mediumPriorityPods {
				medPod, err := cs.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}

				if medPod.Spec.NodeName != "" {
					scheduled++
				}
			}
			if scheduled > 5*nodeListLen {
				return false, fmt.Errorf("expected %v medium priority pods to be scheduled, but got %d", 5*nodeListLen, scheduled)
			}

			return scheduled == 5*nodeListLen, nil
		}))
	})

	/*
		Release: v1.31
		Testname: Verify the DisruptionTarget condition is added to the preempted pod
		Description:
		1. Run a low priority pod with finalizer which consumes 1/1 of node resources
		2. Schedule a higher priority pod which also consumes 1/1 of node resources
		3. See if the pod with lower priority is preempted and has the pod disruption condition
		4. Remove the finalizer so that the pod can be deleted by GC
	*/
	framework.ConformanceIt("validates pod disruption condition is added to the preempted pod", func(ctx context.Context) {
		podRes := v1.ResourceList{testExtendedResource: resource.MustParse("1")}

		ginkgo.By("Select a node to run the lower and higher priority pods")
		gomega.Expect(nodeList.Items).ToNot(gomega.BeEmpty(), "We need at least one node for the test to run")
		node := nodeList.Items[0]
		e2enode.AddExtendedResource(ctx, cs, node.Name, testExtendedResource, resource.MustParse("1"))

		// prepare node affinity to make sure both the lower and higher priority pods are scheduled on the same node
		testNodeAffinity := v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchFields: []v1.NodeSelectorRequirement{
								{Key: "metadata.name", Operator: v1.NodeSelectorOpIn, Values: []string{node.Name}},
							},
						},
					},
				},
			},
		}

		ginkgo.By("Create a low priority pod that consumes 1/1 of node resources")
		victimPod := createPausePod(ctx, f, pausePodConfig{
			Name:              "victim-pod",
			PriorityClassName: lowPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
			Finalizers: []string{testFinalizer},
			Affinity:   &testNodeAffinity,
		})
		framework.Logf("Created pod: %v", victimPod.Name)

		ginkgo.By("Wait for the victim pod to be scheduled")
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, cs, victimPod))

		// Remove the finalizer so that the victim pod can be GCed
		defer e2epod.NewPodClient(f).RemoveFinalizer(ctx, victimPod.Name, testFinalizer)

		ginkgo.By("Create a high priority pod to trigger preemption of the lower priority pod")
		preemptorPod := createPausePod(ctx, f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
			Affinity: &testNodeAffinity,
		})
		framework.Logf("Created pod: %v", preemptorPod.Name)

		ginkgo.By("Waiting for the victim pod to be terminating")
		err := e2epod.WaitForPodTerminatingInNamespaceTimeout(ctx, f.ClientSet, victimPod.Name, victimPod.Namespace, framework.PodDeleteTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying the pod has the pod disruption condition")
		e2epod.VerifyPodHasConditionWithType(ctx, f, victimPod, v1.DisruptionTarget)
	})

	ginkgo.Context("PodTopologySpread Preemption", func() {
		var nodeNames []string
		var nodes []*v1.Node
		topologyKey := "kubernetes.io/e2e-pts-preemption"
		var fakeRes v1.ResourceName = "example.com/fakePTSRes"

		ginkgo.BeforeEach(func(ctx context.Context) {
			if len(nodeList.Items) < 2 {
				ginkgo.Skip("At least 2 nodes are required to run the test")
			}
			ginkgo.By("Trying to get 2 available nodes which can run pod")
			nodeNames = Get2NodesThatCanRunPod(ctx, f)
			ginkgo.By(fmt.Sprintf("Apply dedicated topologyKey %v for this test on the 2 nodes.", topologyKey))
			for _, nodeName := range nodeNames {
				e2enode.AddOrUpdateLabelOnNode(cs, nodeName, topologyKey, nodeName)

				node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				// update Node API object with a fake resource
				ginkgo.By(fmt.Sprintf("Apply 10 fake resource to node %v.", node.Name))
				e2enode.AddExtendedResource(ctx, cs, node.Name, fakeRes, resource.MustParse("10"))
				nodes = append(nodes, node)
			}
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			for _, nodeName := range nodeNames {
				e2enode.RemoveLabelOffNode(cs, nodeName, topologyKey)
			}
			for _, node := range nodes {
				e2enode.RemoveExtendedResource(ctx, cs, node.Name, fakeRes)
			}
		})

		ginkgo.It("validates proper pods are preempted", func(ctx context.Context) {
			podLabel := "e2e-pts-preemption"
			nodeAffinity := &v1.Affinity{
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
			}
			highPodCfg := pausePodConfig{
				Name:              "high",
				Namespace:         ns,
				Labels:            map[string]string{podLabel: ""},
				PriorityClassName: highPriorityClassName,
				Affinity:          nodeAffinity,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{fakeRes: resource.MustParse("9")},
					Limits:   v1.ResourceList{fakeRes: resource.MustParse("9")},
				},
			}
			lowPodCfg := pausePodConfig{
				Namespace:         ns,
				Labels:            map[string]string{podLabel: ""},
				PriorityClassName: lowPriorityClassName,
				Affinity:          nodeAffinity,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{fakeRes: resource.MustParse("3")},
					Limits:   v1.ResourceList{fakeRes: resource.MustParse("3")},
				},
			}

			ginkgo.By("Create 1 High Pod and 3 Low Pods to occupy 9/10 of fake resources on both nodes.")
			// Prepare 1 High Pod and 3 Low Pods
			runPausePod(ctx, f, highPodCfg)
			for i := 1; i <= 3; i++ {
				lowPodCfg.Name = fmt.Sprintf("low-%v", i)
				runPausePod(ctx, f, lowPodCfg)
			}

			ginkgo.By("Create 1 Medium Pod with TopologySpreadConstraints")
			mediumPodCfg := pausePodConfig{
				Name:              "medium",
				Namespace:         ns,
				Labels:            map[string]string{podLabel: ""},
				PriorityClassName: mediumPriorityClassName,
				Affinity:          nodeAffinity,
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{fakeRes: resource.MustParse("3")},
					Limits:   v1.ResourceList{fakeRes: resource.MustParse("3")},
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
			}
			// To fulfil resource.requests, the medium Pod only needs to preempt one low pod.
			// However, in that case, the Pods spread becomes [<high>, <medium, low, low>], which doesn't
			// satisfy the pod topology spread constraints. Hence it needs to preempt another low pod
			// to make the Pods spread like [<high>, <medium, low>].
			runPausePod(ctx, f, mediumPodCfg)

			ginkgo.By("Verify there are 3 Pods left in this namespace")
			wantPods := sets.New("high", "medium", "low")

			// Wait until the number of pods stabilizes. Note that `medium` pod can get scheduled once the
			// second low priority pod is marked as terminating.
			pods, err := e2epod.WaitForNumberOfPods(ctx, cs, ns, 3, framework.PollShortTimeout)
			framework.ExpectNoError(err)

			for _, pod := range pods.Items {
				// Remove the ordinal index for low pod.
				podName := strings.Split(pod.Name, "-")[0]
				if wantPods.Has(podName) {
					ginkgo.By(fmt.Sprintf("Pod %q is as expected to be running.", pod.Name))
					wantPods.Delete(podName)
				} else {
					framework.Failf("Pod %q conflicted with expected PodSet %v", podName, wantPods)
				}
			}
		})
	})

	ginkgo.Context("PreemptionExecutionPath", func() {
		// construct a fakecpu so as to set it to status of Node object
		// otherwise if we update CPU/Memory/etc, those values will be corrected back by kubelet
		var fakecpu v1.ResourceName = "example.com/fakecpu"
		var cs clientset.Interface
		var node *v1.Node
		var ns, nodeHostNameLabel string
		f := framework.NewDefaultFramework("sched-preemption-path")
		f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

		priorityPairs := make([]priorityPair, 0)

		ginkgo.AfterEach(func(ctx context.Context) {
			// print out additional info if tests failed
			if ginkgo.CurrentSpecReport().Failed() {
				// List existing PriorityClasses.
				priorityList, err := cs.SchedulingV1().PriorityClasses().List(ctx, metav1.ListOptions{})
				if err != nil {
					framework.Logf("Unable to list PriorityClasses: %v", err)
				} else {
					framework.Logf("List existing PriorityClasses:")
					for _, p := range priorityList.Items {
						framework.Logf("%v/%v created at %v", p.Name, p.Value, p.CreationTimestamp)
					}
				}
			}

			if node != nil {
				e2enode.RemoveExtendedResource(ctx, cs, node.Name, fakecpu)
			}
			for _, pair := range priorityPairs {
				_ = cs.SchedulingV1().PriorityClasses().Delete(ctx, pair.name, *metav1.NewDeleteOptions(0))
			}
		})

		ginkgo.BeforeEach(func(ctx context.Context) {
			cs = f.ClientSet
			ns = f.Namespace.Name

			// find an available node
			ginkgo.By("Finding an available node")
			nodeName := GetNodeThatCanRunPod(ctx, f)
			framework.Logf("found a healthy node: %s", nodeName)

			// get the node API object
			var err error
			node, err = cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("error getting node %q: %v", nodeName, err)
			}
			var ok bool
			nodeHostNameLabel, ok = node.GetObjectMeta().GetLabels()["kubernetes.io/hostname"]
			if !ok {
				framework.Failf("error getting kubernetes.io/hostname label on node %s", nodeName)
			}

			// update Node API object with a fake resource
			e2enode.AddExtendedResource(ctx, cs, node.Name, fakecpu, resource.MustParse("1000"))

			// create four PriorityClass: p1, p2, p3, p4
			for i := 1; i <= 4; i++ {
				priorityName := fmt.Sprintf("p%d", i)
				priorityVal := int32(i)
				priorityPairs = append(priorityPairs, priorityPair{name: priorityName, value: priorityVal})
				_, err := cs.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: priorityName}, Value: priorityVal}, metav1.CreateOptions{})
				if err != nil {
					framework.Logf("Failed to create priority '%v/%v'. Reason: %v. Msg: %v", priorityName, priorityVal, apierrors.ReasonForError(err), err)
				}
				if err != nil && !apierrors.IsAlreadyExists(err) {
					framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
				}
			}
		})

		/*
			Release: v1.19
			Testname: Pod preemption verification
			Description: Four levels of Pods in ReplicaSets with different levels of Priority, restricted by given CPU limits MUST launch. Priority 1 - 3 Pods MUST spawn first followed by Priority 4 Pod. The ReplicaSets with Replicas MUST contain the expected number of Replicas.
		*/
		framework.ConformanceIt("runs ReplicaSets to verify preemption running path", func(ctx context.Context) {
			var podNamesSeen [3]atomic.Int32

			_, podController := cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						obj, err := f.ClientSet.CoreV1().Pods(ns).List(ctx, options)
						return runtime.Object(obj), err
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return f.ClientSet.CoreV1().Pods(ns).Watch(ctx, options)
					},
				},
				&v1.Pod{},
				0,
				cache.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						if pod, ok := obj.(*v1.Pod); ok {
							switch {
							case strings.HasPrefix(pod.Name, "rs-pod1"):
								podNamesSeen[0].Add(1)
							case strings.HasPrefix(pod.Name, "rs-pod2"):
								podNamesSeen[1].Add(1)
							case strings.HasPrefix(pod.Name, "rs-pod3"):
								podNamesSeen[2].Add(1)
							}
						}
					},
				},
			)

			go podController.Run(ctx.Done())

			// prepare three ReplicaSet
			rsConfs := []pauseRSConfig{
				{
					Replicas: int32(1),
					PodConfig: pausePodConfig{
						Name:              "pod1",
						Namespace:         ns,
						Labels:            map[string]string{"name": "pod1"},
						PriorityClassName: "p1",
						NodeSelector:      map[string]string{"kubernetes.io/hostname": nodeHostNameLabel},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{fakecpu: resource.MustParse("200")},
							Limits:   v1.ResourceList{fakecpu: resource.MustParse("200")},
						},
					},
				},
				{
					Replicas: int32(1),
					PodConfig: pausePodConfig{
						Name:              "pod2",
						Namespace:         ns,
						Labels:            map[string]string{"name": "pod2"},
						PriorityClassName: "p2",
						NodeSelector:      map[string]string{"kubernetes.io/hostname": nodeHostNameLabel},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{fakecpu: resource.MustParse("300")},
							Limits:   v1.ResourceList{fakecpu: resource.MustParse("300")},
						},
					},
				},
				{
					Replicas: int32(1),
					PodConfig: pausePodConfig{
						Name:              "pod3",
						Namespace:         ns,
						Labels:            map[string]string{"name": "pod3"},
						PriorityClassName: "p3",
						NodeSelector:      map[string]string{"kubernetes.io/hostname": nodeHostNameLabel},
						Resources: &v1.ResourceRequirements{
							Requests: v1.ResourceList{fakecpu: resource.MustParse("450")},
							Limits:   v1.ResourceList{fakecpu: resource.MustParse("450")},
						},
					},
				},
			}
			// create ReplicaSet{1,2,3} so as to occupy 950/1000 fake resource
			for i := range rsConfs {
				runPauseRS(ctx, f, rsConfs[i])
			}

			framework.Logf(
				"pods created so far: [%d %d %d]",
				podNamesSeen[0].Load(),
				podNamesSeen[1].Load(),
				podNamesSeen[2].Load(),
			)
			framework.Logf("length of pods created so far: %v", len(podNamesSeen))

			// create a Preemptor Pod
			preemptorPodConf := pausePodConfig{
				Name:              "pod4",
				Namespace:         ns,
				Labels:            map[string]string{"name": "pod4"},
				PriorityClassName: "p4",
				NodeSelector:      map[string]string{"kubernetes.io/hostname": nodeHostNameLabel},
				Resources: &v1.ResourceRequirements{
					Requests: v1.ResourceList{fakecpu: resource.MustParse("500")},
					Limits:   v1.ResourceList{fakecpu: resource.MustParse("500")},
				},
			}
			preemptorPod := createPod(ctx, f, preemptorPodConf)
			waitForPreemptingWithTimeout(ctx, f, preemptorPod, framework.PodGetTimeout)

			framework.Logf(
				"pods created so far: [%d %d %d]",
				podNamesSeen[0].Load(),
				podNamesSeen[1].Load(),
				podNamesSeen[2].Load(),
			)

			// count pods number of ReplicaSet{1,2,3}:
			// - if it's more than expected replicas, it denotes its pods have been over-preempted
			// - if it's less than expected replicas, it denotes its pods are under-preempted
			// "*2" means pods of ReplicaSet{1,2} are expected to be only preempted once.
			expectedRSPods := []int32{1 * 2, 1 * 2, 1}
			err := wait.PollUntilContextTimeout(ctx, framework.Poll, framework.PollShortTimeout, false, func(ctx context.Context) (bool, error) {
				for i := range len(podNamesSeen) {
					got := podNamesSeen[i].Load()
					if got < expectedRSPods[i] {
						framework.Logf("waiting for rs%d to observe %d pod creations, got %d", i+1, expectedRSPods[i], got)
						return false, nil
					} else if got > expectedRSPods[i] {
						return false, fmt.Errorf("rs%d had more than %d pods created: %d", i+1, expectedRSPods[i], got)
					}
				}
				return true, nil
			})
			if err != nil {
				framework.Logf(
					"pods created so far: [%d %d %d]",
					podNamesSeen[0].Load(),
					podNamesSeen[1].Load(),
					podNamesSeen[2].Load(),
				)
				framework.Failf("failed pod observation expectations: %v", err)
			}

			// If logic continues to here, we should do a final check to ensure within a time period,
			// the state is stable; otherwise, pods may be over-preempted.
			time.Sleep(5 * time.Second)
			for i := range len(podNamesSeen) {
				got := podNamesSeen[i].Load()
				if got < expectedRSPods[i] {
					framework.Failf("pods of ReplicaSet%d have been under-preempted: expect %v pod names, but got %d", i+1, expectedRSPods[i], got)
				} else if got > expectedRSPods[i] {
					framework.Failf("pods of ReplicaSet%d have been over-preempted: expect %v pod names, but got %d", i+1, expectedRSPods[i], got)
				}
			}
		})
	})

	ginkgo.Context("PriorityClass endpoints", func() {
		var cs clientset.Interface
		f := framework.NewDefaultFramework("sched-preemption-path")
		f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
		testUUID := uuid.New().String()
		var pcs []*schedulingv1.PriorityClass

		ginkgo.BeforeEach(func(ctx context.Context) {
			cs = f.ClientSet
			// Create 2 PriorityClass: p1, p2.
			for i := 1; i <= 2; i++ {
				name, val := fmt.Sprintf("p%d", i), int32(i)
				pc, err := cs.SchedulingV1().PriorityClasses().Create(ctx, &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"e2e": testUUID}}, Value: val}, metav1.CreateOptions{})
				if err != nil {
					framework.Logf("Failed to create priority '%v/%v'. Reason: %v. Msg: %v", name, val, apierrors.ReasonForError(err), err)
				}
				if err != nil && !apierrors.IsAlreadyExists(err) {
					framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
				}
				pcs = append(pcs, pc)
			}
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			// Print out additional info if tests failed.
			if ginkgo.CurrentSpecReport().Failed() {
				// List existing PriorityClasses.
				priorityList, err := cs.SchedulingV1().PriorityClasses().List(ctx, metav1.ListOptions{})
				if err != nil {
					framework.Logf("Unable to list PriorityClasses: %v", err)
				} else {
					framework.Logf("List existing PriorityClasses:")
					for _, p := range priorityList.Items {
						framework.Logf("%v/%v created at %v", p.Name, p.Value, p.CreationTimestamp)
					}
				}
			}

			// Collection deletion on created PriorityClasses.
			err := cs.SchedulingV1().PriorityClasses().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: fmt.Sprintf("e2e=%v", testUUID)})
			framework.ExpectNoError(err)
		})

		/*
			Release: v1.20
			Testname: Scheduler, Verify PriorityClass endpoints
			Description: Verify that PriorityClass endpoints can be listed. When any mutable field is
			either patched or updated it MUST succeed. When any immutable field is either patched or
			updated it MUST fail.
		*/
		framework.ConformanceIt("verify PriorityClass endpoints can be operated with different HTTP methods", func(ctx context.Context) {
			// 1. Patch/Update on immutable fields will fail.
			pcCopy := pcs[0].DeepCopy()
			pcCopy.Value = pcCopy.Value * 10
			err := patchPriorityClass(ctx, cs, pcs[0], pcCopy)
			gomega.Expect(err).To(gomega.HaveOccurred(), "expect a patch error on an immutable field")
			framework.Logf("%v", err)

			pcCopy = pcs[1].DeepCopy()
			pcCopy.Value = pcCopy.Value * 10
			_, err = cs.SchedulingV1().PriorityClasses().Update(ctx, pcCopy, metav1.UpdateOptions{})
			gomega.Expect(err).To(gomega.HaveOccurred(), "expect an update error on an immutable field")
			framework.Logf("%v", err)

			// 2. Patch/Update on mutable fields will succeed.
			newDesc := "updated description"
			pcCopy = pcs[0].DeepCopy()
			pcCopy.Description = newDesc
			err = patchPriorityClass(ctx, cs, pcs[0], pcCopy)
			framework.ExpectNoError(err)

			pcCopy = pcs[1].DeepCopy()
			pcCopy.Description = newDesc
			_, err = cs.SchedulingV1().PriorityClasses().Update(ctx, pcCopy, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// 3. List existing PriorityClasses.
			_, err = cs.SchedulingV1().PriorityClasses().List(ctx, metav1.ListOptions{})
			framework.ExpectNoError(err)

			// 4. Verify fields of updated PriorityClasses.
			for _, pc := range pcs {
				livePC, err := cs.SchedulingV1().PriorityClasses().Get(ctx, pc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				gomega.Expect(livePC.Value).To(gomega.Equal(pc.Value))
				gomega.Expect(livePC.Description).To(gomega.Equal(newDesc))
				gomega.Expect(livePC).To(apimachineryutils.HaveValidResourceVersion())
				gomega.Expect(resourceversion.CompareResourceVersion(pc.ResourceVersion, livePC.ResourceVersion)).To(gomega.BeNumerically("==", -1), "changed object should have a larger resource version")
			}
		})
	})
})

type pauseRSConfig struct {
	Replicas  int32
	PodConfig pausePodConfig
}

func initPauseRS(f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	pausePod := initPausePod(f, conf.PodConfig)
	pauseRS := &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "rs-" + pausePod.Name,
			Namespace: pausePod.Namespace,
		},
		Spec: appsv1.ReplicaSetSpec{
			Replicas: &conf.Replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: pausePod.Labels,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: pausePod.ObjectMeta.Labels},
				Spec:       pausePod.Spec,
			},
		},
	}
	return pauseRS
}

func createPauseRS(ctx context.Context, f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	namespace := conf.PodConfig.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	rs, err := f.ClientSet.AppsV1().ReplicaSets(namespace).Create(ctx, initPauseRS(f, conf), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return rs
}

func runPauseRS(ctx context.Context, f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	rs := createPauseRS(ctx, f, conf)
	framework.ExpectNoError(e2ereplicaset.WaitForReplicaSetTargetAvailableReplicasWithTimeout(ctx, f.ClientSet, rs, conf.Replicas, framework.PodGetTimeout))
	return rs
}

func createPod(ctx context.Context, f *framework.Framework, conf pausePodConfig) *v1.Pod {
	namespace := conf.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Create(ctx, initPausePod(f, conf), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return pod
}

// waitForPreemptingWithTimeout verifies if 'pod' is preempting within 'timeout', specifically it checks
// if the 'spec.NodeName' field of preemptor 'pod' has been set.
func waitForPreemptingWithTimeout(ctx context.Context, f *framework.Framework, pod *v1.Pod, timeout time.Duration) {
	err := wait.PollUntilContextTimeout(ctx, 2*time.Second, timeout, false, func(ctx context.Context) (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(pod.Spec.NodeName) > 0 {
			return true, nil
		}
		return false, err
	})
	framework.ExpectNoError(err, "pod %v/%v failed to preempt other pods", pod.Namespace, pod.Name)
}

func patchPriorityClass(ctx context.Context, cs clientset.Interface, old, new *schedulingv1.PriorityClass) error {
	oldData, err := json.Marshal(old)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(new)
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &schedulingv1.PriorityClass{})
	if err != nil {
		return fmt.Errorf("failed to create merge patch for PriorityClass %q: %w", old.Name, err)
	}
	_, err = cs.SchedulingV1().PriorityClasses().Patch(ctx, old.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	return err
}
