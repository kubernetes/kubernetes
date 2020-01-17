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
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/framework/replicaset"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

type priorityPair struct {
	name  string
	value int32
}

var _ = SIGDescribe("SchedulerPreemption [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var ns string
	f := framework.NewDefaultFramework("sched-preemption")

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
			cs.SchedulingV1().PriorityClasses().Delete(pair.name, metav1.NewDeleteOptions(0))
		}
	})

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}
		var err error
		for _, pair := range priorityPairs {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(&schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: pair.name}, Value: pair.value})
			framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		}

		e2enode.WaitForTotalHealthy(cs, time.Minute)
		masterNodes, nodeList, err = e2enode.GetMasterAndWorkerNodes(cs)
		if err != nil {
			framework.Logf("Unexpected error occurred: %v", err)
		}
		// TODO: write a wrapper for ExpectNoErrorWithOffset()
		framework.ExpectNoErrorWithOffset(0, err)

		err = framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	// This test verifies that when a higher priority pod is created and no node with
	// enough resources is found, scheduler preempts a lower priority pod to schedule
	// the high priority pod.
	ginkgo.It("validates basic preemption works", func() {
		var podRes v1.ResourceList
		// Create one pod per node that uses a lot of the node's resources.
		ginkgo.By("Create pods that use 60% of node resources.")
		pods := make([]*v1.Pod, len(nodeList.Items))
		for i, node := range nodeList.Items {
			cpuAllocatable, found := node.Status.Allocatable["cpu"]
			framework.ExpectEqual(found, true)
			milliCPU := cpuAllocatable.MilliValue() * 40 / 100
			memAllocatable, found := node.Status.Allocatable["memory"]
			framework.ExpectEqual(found, true)
			memory := memAllocatable.Value() * 60 / 100
			podRes = v1.ResourceList{}
			podRes[v1.ResourceCPU] = *resource.NewMilliQuantity(int64(milliCPU), resource.DecimalSI)
			podRes[v1.ResourceMemory] = *resource.NewQuantity(int64(memory), resource.BinarySI)

			// make the first pod low priority and the rest medium priority.
			priorityName := mediumPriorityClassName
			if i == 0 {
				priorityName = lowPriorityClassName
			}
			pods[i] = createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, priorityName),
				PriorityClassName: priorityName,
				Resources: &v1.ResourceRequirements{
					Requests: podRes,
				},
			})
			framework.Logf("Created pod: %v", pods[i].Name)
		}
		ginkgo.By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
		}

		ginkgo.By("Run a high priority pod that use 60% of a node resources.")
		// Create a high priority pod and make sure it is scheduled.
		runPausePod(f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
			},
		})
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(pods[0].Name, metav1.GetOptions{})
		podDeleted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		framework.ExpectEqual(podDeleted, true)
		// Other pods (mid priority ones) should be present.
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil())
		}
	})

	// This test verifies that when a critical pod is created and no node with
	// enough resources is found, scheduler preempts a lower priority pod to schedule
	// this critical pod.
	ginkgo.It("validates lower priority pod preemption by critical pod", func() {
		var podRes v1.ResourceList
		// Create one pod per node that uses a lot of the node's resources.
		ginkgo.By("Create pods that use 60% of node resources.")
		pods := make([]*v1.Pod, len(nodeList.Items))
		for i, node := range nodeList.Items {
			cpuAllocatable, found := node.Status.Allocatable["cpu"]
			framework.ExpectEqual(found, true)
			milliCPU := cpuAllocatable.MilliValue() * 40 / 100
			memAllocatable, found := node.Status.Allocatable["memory"]
			framework.ExpectEqual(found, true)
			memory := memAllocatable.Value() * 60 / 100
			podRes = v1.ResourceList{}
			podRes[v1.ResourceCPU] = *resource.NewMilliQuantity(int64(milliCPU), resource.DecimalSI)
			podRes[v1.ResourceMemory] = *resource.NewQuantity(int64(memory), resource.BinarySI)

			// make the first pod low priority and the rest medium priority.
			priorityName := mediumPriorityClassName
			if i == 0 {
				priorityName = lowPriorityClassName
			}
			pods[i] = createPausePod(f, pausePodConfig{
				Name:              fmt.Sprintf("pod%d-%v", i, priorityName),
				PriorityClassName: priorityName,
				Resources: &v1.ResourceRequirements{
					Requests: podRes,
				},
			})
			framework.Logf("Created pod: %v", pods[i].Name)
		}
		ginkgo.By("Wait for pods to be scheduled.")
		for _, pod := range pods {
			framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, pod))
		}

		ginkgo.By("Run a critical pod that use 60% of a node resources.")
		// Create a critical pod and make sure it is scheduled.
		defer func() {
			// Clean-up the critical pod
			// Always run cleanup to make sure the pod is properly cleaned up.
			err := f.ClientSet.CoreV1().Pods(metav1.NamespaceSystem).Delete("critical-pod", metav1.NewDeleteOptions(0))
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error cleanup pod `%s/%s`: %v", metav1.NamespaceSystem, "critical-pod", err)
			}
		}()
		runPausePod(f, pausePodConfig{
			Name:              "critical-pod",
			Namespace:         metav1.NamespaceSystem,
			PriorityClassName: scheduling.SystemClusterCritical,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
			},
		})
		// Make sure that the lowest priority pod is deleted.
		preemptedPod, err := cs.CoreV1().Pods(pods[0].Namespace).Get(pods[0].Name, metav1.GetOptions{})
		podDeleted := (err != nil && apierrors.IsNotFound(err)) ||
			(err == nil && preemptedPod.DeletionTimestamp != nil)
		framework.ExpectEqual(podDeleted, true)
		// Other pods (mid priority ones) should be present.
		for i := 1; i < len(pods); i++ {
			livePod, err := cs.CoreV1().Pods(pods[i].Namespace).Get(pods[i].Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(livePod.DeletionTimestamp).To(gomega.BeNil())
		}
	})
})

// construct a fakecpu so as to set it to status of Node object
// otherwise if we update CPU/Memory/etc, those values will be corrected back by kubelet
var fakecpu v1.ResourceName = "example.com/fakecpu"

var _ = SIGDescribe("PreemptionExecutionPath", func() {
	var cs clientset.Interface
	var node *v1.Node
	var ns, nodeHostNameLabel string
	f := framework.NewDefaultFramework("sched-preemption-path")

	priorityPairs := make([]priorityPair, 0)

	ginkgo.AfterEach(func() {
		// print out additional info if tests failed
		if ginkgo.CurrentGinkgoTestDescription().Failed {
			// list existing priorities
			priorityList, err := cs.SchedulingV1().PriorityClasses().List(metav1.ListOptions{})
			if err != nil {
				framework.Logf("Unable to list priorities: %v", err)
			} else {
				framework.Logf("List existing priorities:")
				for _, p := range priorityList.Items {
					framework.Logf("%v/%v created at %v", p.Name, p.Value, p.CreationTimestamp)
				}
			}
		}

		if node != nil {
			nodeCopy := node.DeepCopy()
			// force it to update
			nodeCopy.ResourceVersion = "0"
			delete(nodeCopy.Status.Capacity, fakecpu)
			_, err := cs.CoreV1().Nodes().UpdateStatus(nodeCopy)
			framework.ExpectNoError(err)
		}
		for _, pair := range priorityPairs {
			cs.SchedulingV1().PriorityClasses().Delete(pair.name, metav1.NewDeleteOptions(0))
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
		node, err = cs.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
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
		// force it to update
		nodeCopy.ResourceVersion = "0"
		nodeCopy.Status.Capacity[fakecpu] = resource.MustParse("1000")
		node, err = cs.CoreV1().Nodes().UpdateStatus(nodeCopy)
		framework.ExpectNoError(err)

		// create four PriorityClass: p1, p2, p3, p4
		for i := 1; i <= 4; i++ {
			priorityName := fmt.Sprintf("p%d", i)
			priorityVal := int32(i)
			priorityPairs = append(priorityPairs, priorityPair{name: priorityName, value: priorityVal})
			_, err := cs.SchedulingV1().PriorityClasses().Create(&schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: priorityName}, Value: priorityVal})
			if err != nil {
				framework.Logf("Failed to create priority '%v/%v': %v", priorityName, priorityVal, err)
				framework.Logf("Reason: %v. Msg: %v", apierrors.ReasonForError(err), err)
			}
			framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		}
	})

	ginkgo.It("runs ReplicaSets to verify preemption running path", func() {
		podNamesSeen := []int32{0, 0, 0}
		stopCh := make(chan struct{})

		// create a pod controller to list/watch pod events from the test framework namespace
		_, podController := cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					obj, err := f.ClientSet.CoreV1().Pods(ns).List(options)
					return runtime.Object(obj), err
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return f.ClientSet.CoreV1().Pods(ns).Watch(options)
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
	rs, err := f.ClientSet.AppsV1().ReplicaSets(namespace).Create(initPauseRS(f, conf))
	framework.ExpectNoError(err)
	return rs
}

func runPauseRS(f *framework.Framework, conf pauseRSConfig) *appsv1.ReplicaSet {
	rs := createPauseRS(f, conf)
	framework.ExpectNoError(replicaset.WaitForReplicaSetTargetAvailableReplicasWithTimeout(f.ClientSet, rs, conf.Replicas, framework.PodGetTimeout))
	return rs
}

func createPod(f *framework.Framework, conf pausePodConfig) *v1.Pod {
	namespace := conf.Namespace
	if len(namespace) == 0 {
		namespace = f.Namespace.Name
	}
	pod, err := f.ClientSet.CoreV1().Pods(namespace).Create(initPausePod(f, conf))
	framework.ExpectNoError(err)
	return pod
}

// waitForPreemptingWithTimeout verifies if 'pod' is preempting within 'timeout', specifically it checks
// if the 'spec.NodeName' field of preemptor 'pod' has been set.
func waitForPreemptingWithTimeout(f *framework.Framework, pod *v1.Pod, timeout time.Duration) {
	err := wait.Poll(2*time.Second, timeout, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
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
