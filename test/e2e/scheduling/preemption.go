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

	"github.com/google/uuid"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

type priorityPair struct {
	name  string
	value int32
}

var testExtendedResource = v1.ResourceName("scheduling.k8s.io/foo")

var _ = SIGDescribe("SchedulerPreemption [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var ns string
	f := framework.NewDefaultFramework("sched-preemption")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	lowPriority, mediumPriority, highPriority := int32(1), int32(100), int32(1000)
	lowPriorityClassName := f.BaseName + "-low-priority"
	mediumPriorityClassName := f.BaseName + "-medium-priority"
	highPriorityClassName := f.BaseName + "-high-priority"
	priorityPairs := []priorityPair{
		{name: lowPriorityClassName, value: lowPriority},
		{name: mediumPriorityClassName, value: mediumPriority},
		{name: highPriorityClassName, value: highPriority},
	}

	ginkgo.AfterEach(func() {
		for _, pair := range priorityPairs {
			cs.SchedulingV1().PriorityClasses().Delete(context.TODO(), pair.name, *metav1.NewDeleteOptions(0))
		}
		for _, node := range nodeList.Items {
			nodeCopy := node.DeepCopy()
			delete(nodeCopy.Status.Capacity, testExtendedResource)
			err := patchNode(cs, &node, nodeCopy)
			framework.ExpectNoError(err)
		}
	})

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		var err error
		for _, pair := range priorityPairs {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: pair.name}, Value: pair.value}, metav1.CreateOptions{})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
			}
		}

		e2enode.WaitForTotalHealthy(cs, time.Minute)
		nodeList, err = e2enode.GetReadySchedulableNodes(cs)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		framework.ExpectNoErrorWithOffset(0, err)
		for _, n := range nodeList.Items {
			workerNodes.Insert(n.Name)
		}

		err = framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.19
		Testname: Scheduler, Basic Preemption
		Description: When a higher priority pod is created and no node with enough
		resources is found, the scheduler MUST preempt a lower priority pod and
		schedule the high priority pod.
	*/
	framework.ConformanceIt("validates basic preemption works", func() {
		var podRes v1.ResourceList

		// Create two pods per node that uses a lot of the node's resources.
		ginkgo.By("Create pods that use 4/5 of node resources.")
		pods := make([]*v1.Pod, 0, 2*len(nodeList.Items))
		// Create pods in the cluster.
		// One of them has low priority, making it the victim for preemption.
		for i, node := range nodeList.Items {
			// Update each node to advertise 3 available extended resources
			nodeCopy := node.DeepCopy()
			nodeCopy.Status.Capacity[testExtendedResource] = resource.MustParse("5")
			err := patchNode(cs, &node, nodeCopy)
			framework.ExpectNoError(err)

			for j := 0; j < 2; j++ {
				// Request 2 of the available resources for the victim pods
				podRes = v1.ResourceList{}
				podRes[testExtendedResource] = resource.MustParse("2")

				// make the first pod low priority and the rest medium priority.
				priorityName := mediumPriorityClassName
				if len(pods) == 0 {
					priorityName = lowPriorityClassName
				}
				pausePod := createPausePod(f, pausePodConfig{
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
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
		}

		// Set the pod request to the first pod's resources (should be low priority pod)
		podRes = pods[0].Spec.Containers[0].Resources.Requests

		ginkgo.By("Run a high priority pod that has same requirements as that of lower priority pod")
		// Create a high priority pod and make sure it is scheduled on the same node as the low priority pod.
		runPausePodWithTimeout(f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
		}, framework.PodStartShortTimeout)

		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(context.TODO(), pods[0].Name, metav1.GetOptions{})
		podPreempted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		if !podPreempted {
			framework.Failf("expected pod to be preempted, instead got pod %+v and error %v", preemptedPod, err)
		}
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(context.TODO(), pods[i].Name, metav1.GetOptions{})
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
	framework.ConformanceIt("validates lower priority pod preemption by critical pod", func() {
		var podRes v1.ResourceList

		ginkgo.By("Create pods that use 4/5 of node resources.")
		pods := make([]*v1.Pod, 0, len(nodeList.Items))
		for i, node := range nodeList.Items {
			// Update each node to advertise 3 available extended resources
			nodeCopy := node.DeepCopy()
			nodeCopy.Status.Capacity[testExtendedResource] = resource.MustParse("5")
			err := patchNode(cs, &node, nodeCopy)
			framework.ExpectNoError(err)

			for j := 0; j < 2; j++ {
				// Request 2 of the available resources for the victim pods
				podRes = v1.ResourceList{}
				podRes[testExtendedResource] = resource.MustParse("2")

				// make the first pod low priority and the rest medium priority.
				priorityName := mediumPriorityClassName
				if len(pods) == 0 {
					priorityName = lowPriorityClassName
				}
				pausePod := createPausePod(f, pausePodConfig{
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
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
		}

		// We want this pod to be preempted
		podRes = pods[0].Spec.Containers[0].Resources.Requests
		ginkgo.By("Run a critical pod that use same resources as that of a lower priority pod")
		// Create a critical pod and make sure it is scheduled.
		defer func() {
			// Clean-up the critical pod
			// Always run cleanup to make sure the pod is properly cleaned up.
			err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(context.TODO(), "critical-pod", *metav1.NewDeleteOptions(0))
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error cleanup pod `%s/%s`: %v", metav1.NamespaceSystem, "critical-pod", err)
			}
		}()
		runPausePodWithTimeout(f, pausePodConfig{
			Name:              "critical-pod",
			Namespace:         metav1.NamespaceSystem,
			PriorityClassName: scheduling.SystemClusterCritical,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
		}, framework.PodStartShortTimeout)

		defer func() {
			// Clean-up the critical pod
			err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete(context.TODO(), "critical-pod", *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)
		}()
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(context.TODO(), pods[0].Name, metav1.GetOptions{})
		podPreempted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(context.TODO(), pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil())
		}

		if !podPreempted {
			framework.Failf("expected pod to be preempted, instead got pod %+v and error %v", preemptedPod, err)
		}
	})

	ginkgo.Context("PodTopologySpread Preemption", func() {
		var nodeNames []string
		var nodes []*v1.Node
		topologyKey := "kubernetes.io/e2e-pts-preemption"
		var fakeRes v1.ResourceName = "example.com/fakePTSRes"

		ginkgo.BeforeEach(func() {
			if len(nodeList.Items) < 2 {
				ginkgo.Skip("At least 2 nodes are required to run the test")
			}
			ginkgo.By("Trying to get 2 available nodes which can run pod")
			nodeNames = Get2NodesThatCanRunPod(f)
			ginkgo.By(fmt.Sprintf("Apply dedicated topologyKey %v for this test on the 2 nodes.", topologyKey))
			for _, nodeName := range nodeNames {
				framework.AddOrUpdateLabelOnNode(cs, nodeName, topologyKey, nodeName)

				node, err := cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				// update Node API object with a fake resource
				ginkgo.By(fmt.Sprintf("Apply 10 fake resource to node %v.", node.Name))
				nodeCopy := node.DeepCopy()
				nodeCopy.Status.Capacity[fakeRes] = resource.MustParse("10")
				err = patchNode(cs, node, nodeCopy)
				framework.ExpectNoError(err)
				nodes = append(nodes, node)
			}
		})
		ginkgo.AfterEach(func() {
			for _, nodeName := range nodeNames {
				framework.RemoveLabelOffNode(cs, nodeName, topologyKey)
			}
			for _, node := range nodes {
				nodeCopy := node.DeepCopy()
				delete(nodeCopy.Status.Capacity, fakeRes)
				err := patchNode(cs, node, nodeCopy)
				framework.ExpectNoError(err)
			}
		})

		ginkgo.It("validates proper pods are preempted", func() {
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
			runPausePod(f, highPodCfg)
			for i := 1; i <= 3; i++ {
				lowPodCfg.Name = fmt.Sprintf("low-%v", i)
				runPausePod(f, lowPodCfg)
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
			runPausePod(f, mediumPodCfg)

			ginkgo.By("Verify there are 3 Pods left in this namespace")
			wantPods := sets.NewString("high", "medium", "low")

			// Wait until the number of pods stabilizes. Note that `medium` pod can get scheduled once the
			// second low priority pod is marked as terminating.
			pods, err := e2epod.WaitForNumberOfPods(cs, ns, 3, framework.PollShortTimeout)
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
		f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

		priorityPairs := make([]priorityPair, 0)

		ginkgo.AfterEach(func() {
			// print out additional info if tests failed
			if ginkgo.CurrentSpecReport().Failed() {
				// List existing PriorityClasses.
				priorityList, err := cs.SchedulingV1().PriorityClasses().List(context.TODO(), metav1.ListOptions{})
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
				nodeCopy := node.DeepCopy()
				delete(nodeCopy.Status.Capacity, fakecpu)
				err := patchNode(cs, node, nodeCopy)
				framework.ExpectNoError(err)
			}
			for _, pair := range priorityPairs {
				cs.SchedulingV1().PriorityClasses().Delete(context.TODO(), pair.name, *metav1.NewDeleteOptions(0))
			}
		})

		ginkgo.BeforeEach(func() {
			cs = f.ClientSet
			ns = f.Namespace.Name

			// find an available node
			ginkgo.By("Finding an available node")
			nodeName := GetNodeThatCanRunPod(f)
			framework.Logf("found a healthy node: %s", nodeName)

			// get the node API object
			var err error
			node, err = cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
			if err != nil {
				framework.Failf("error getting node %q: %v", nodeName, err)
			}
			var ok bool
			nodeHostNameLabel, ok = node.GetObjectMeta().GetLabels()["kubernetes.io/hostname"]
			if !ok {
				framework.Failf("error getting kubernetes.io/hostname label on node %s", nodeName)
			}

			// update Node API object with a fake resource
			nodeCopy := node.DeepCopy()
			nodeCopy.Status.Capacity[fakecpu] = resource.MustParse("1000")
			err = patchNode(cs, node, nodeCopy)
			framework.ExpectNoError(err)

			// create four PriorityClass: p1, p2, p3, p4
			for i := 1; i <= 4; i++ {
				priorityName := fmt.Sprintf("p%d", i)
				priorityVal := int32(i)
				priorityPairs = append(priorityPairs, priorityPair{name: priorityName, value: priorityVal})
				_, err := cs.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: priorityName}, Value: priorityVal}, metav1.CreateOptions{})
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
		framework.ConformanceIt("runs ReplicaSets to verify preemption running path", func() {
			podNamesSeen := []int32{0, 0, 0}
			stopCh := make(chan struct{})

			// create a pod controller to list/watch pod events from the test framework namespace
			_, podController := cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
						obj, err := f.ClientSet.CoreV1().Pods(ns).List(context.TODO(), options)
						return runtime.Object(obj), err
					},
					WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
						return f.ClientSet.CoreV1().Pods(ns).Watch(context.TODO(), options)
					},
				},
				&v1.Pod{},
				0,
				cache.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						if pod, ok := obj.(*v1.Pod); ok {
							if strings.HasPrefix(pod.Name, "rs-pod1") {
								atomic.AddInt32(&podNamesSeen[0], 1)
							} else if strings.HasPrefix(pod.Name, "rs-pod2") {
								atomic.AddInt32(&podNamesSeen[1], 1)
							} else if strings.HasPrefix(pod.Name, "rs-pod3") {
								atomic.AddInt32(&podNamesSeen[2], 1)
							}
						}
					},
				},
			)
			go podController.Run(stopCh)
			defer close(stopCh)

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
				runPauseRS(f, rsConfs[i])
			}

			framework.Logf("pods created so far: %v", podNamesSeen)
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
			preemptorPod := createPod(f, preemptorPodConf)
			waitForPreemptingWithTimeout(f, preemptorPod, framework.PodGetTimeout)

			framework.Logf("pods created so far: %v", podNamesSeen)

			// count pods number of ReplicaSet{1,2,3}:
			// - if it's more than expected replicas, it denotes its pods have been over-preempted
			// - if it's less than expected replicas, it denotes its pods are under-preempted
			// "*2" means pods of ReplicaSet{1,2} are expected to be only preempted once.
			expectedRSPods := []int32{1 * 2, 1 * 2, 1}
			err := wait.Poll(framework.Poll, framework.PollShortTimeout, func() (bool, error) {
				for i := 0; i < len(podNamesSeen); i++ {
					got := atomic.LoadInt32(&podNamesSeen[i])
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
				framework.Logf("pods created so far: %v", podNamesSeen)
				framework.Failf("failed pod observation expectations: %v", err)
			}

			// If logic continues to here, we should do a final check to ensure within a time period,
			// the state is stable; otherwise, pods may be over-preempted.
			time.Sleep(5 * time.Second)
			for i := 0; i < len(podNamesSeen); i++ {
				got := atomic.LoadInt32(&podNamesSeen[i])
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
		f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
		testUUID := uuid.New().String()
		var pcs []*schedulingv1.PriorityClass

		ginkgo.BeforeEach(func() {
			cs = f.ClientSet
			// Create 2 PriorityClass: p1, p2.
			for i := 1; i <= 2; i++ {
				name, val := fmt.Sprintf("p%d", i), int32(i)
				pc, err := cs.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"e2e": testUUID}}, Value: val}, metav1.CreateOptions{})
				if err != nil {
					framework.Logf("Failed to create priority '%v/%v'. Reason: %v. Msg: %v", name, val, apierrors.ReasonForError(err), err)
				}
				if err != nil && !apierrors.IsAlreadyExists(err) {
					framework.Failf("expected 'alreadyExists' as error, got instead: %v", err)
				}
				pcs = append(pcs, pc)
			}
		})

		ginkgo.AfterEach(func() {
			// Print out additional info if tests failed.
			if ginkgo.CurrentSpecReport().Failed() {
				// List existing PriorityClasses.
				priorityList, err := cs.SchedulingV1().PriorityClasses().List(context.TODO(), metav1.ListOptions{})
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
			err := cs.SchedulingV1().PriorityClasses().DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: fmt.Sprintf("e2e=%v", testUUID)})
			framework.ExpectNoError(err)
		})

		/*
			Release: v1.20
			Testname: Scheduler, Verify PriorityClass endpoints
			Description: Verify that PriorityClass endpoints can be listed. When any mutable field is
			either patched or updated it MUST succeed. When any immutable field is either patched or
			updated it MUST fail.
		*/
		framework.ConformanceIt("verify PriorityClass endpoints can be operated with different HTTP methods", func() {
			// 1. Patch/Update on immutable fields will fail.
			pcCopy := pcs[0].DeepCopy()
			pcCopy.Value = pcCopy.Value * 10
			err := patchPriorityClass(cs, pcs[0], pcCopy)
			framework.ExpectError(err, "expect a patch error on an immutable field")
			framework.Logf("%v", err)

			pcCopy = pcs[1].DeepCopy()
			pcCopy.Value = pcCopy.Value * 10
			_, err = cs.SchedulingV1().PriorityClasses().Update(context.TODO(), pcCopy, metav1.UpdateOptions{})
			framework.ExpectError(err, "expect an update error on an immutable field")
			framework.Logf("%v", err)

			// 2. Patch/Update on mutable fields will succeed.
			newDesc := "updated description"
			pcCopy = pcs[0].DeepCopy()
			pcCopy.Description = newDesc
			err = patchPriorityClass(cs, pcs[0], pcCopy)
			framework.ExpectNoError(err)

			pcCopy = pcs[1].DeepCopy()
			pcCopy.Description = newDesc
			_, err = cs.SchedulingV1().PriorityClasses().Update(context.TODO(), pcCopy, metav1.UpdateOptions{})
			framework.ExpectNoError(err)

			// 3. List existing PriorityClasses.
			_, err = cs.SchedulingV1().PriorityClasses().List(context.TODO(), metav1.ListOptions{})
			framework.ExpectNoError(err)

			// 4. Verify fields of updated PriorityClasses.
			for _, pc := range pcs {
				livePC, err := cs.SchedulingV1().PriorityClasses().Get(context.TODO(), pc.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				framework.ExpectEqual(livePC.Value, pc.Value)
				framework.ExpectEqual(livePC.Description, newDesc)
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

func createPauseRS(f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	namespace := conf.PodConfig.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	rs, err := f.ClientSet.AppsV1().ReplicaSets(namespace).Create(context.TODO(), initPauseRS(f, conf), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return rs
}

func runPauseRS(f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	rs := createPauseRS(f, conf)
	framework.ExpectNoError(e2ereplicaset.WaitForReplicaSetTargetAvailableReplicasWithTimeout(f.ClientSet, rs, conf.Replicas, framework.PodGetTimeout))
	return rs
}

func createPod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	namespace := conf.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Create(context.TODO(), initPausePod(f, conf), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	return pod
}

// waitForPreemptingWithTimeout verifies if 'pod' is preempting within 'timeout', specifically it checks
// if the 'spec.NodeName' field of preemptor 'pod' has been set.
func waitForPreemptingWithTimeout(f *framework.Framework, pod *v1.Pod, timeout time.Duration) {
	err := wait.Poll(2*time.Second, timeout, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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

func patchNode(client clientset.Interface, old *v1.Node, new *v1.Node) error {
	oldData, err := json.Marshal(old)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(new)
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Node{})
	if err != nil {
		return fmt.Errorf("failed to create merge patch for node %q: %v", old.Name, err)
	}
	_, err = client.CoreV1().Nodes().Patch(context.TODO(), old.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	return err
}

func patchPriorityClass(cs clientset.Interface, old, new *schedulingv1.PriorityClass) error {
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
		return fmt.Errorf("failed to create merge patch for PriorityClass %q: %v", old.Name, err)
	}
	_, err = cs.SchedulingV1().PriorityClasses().Patch(context.TODO(), old.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	return err
}
