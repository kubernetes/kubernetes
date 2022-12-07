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
	"time"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"

	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

type priorityPair struct {
	name  string
	value int32
}

var testExtendedResource = v1.ResourceName("scheduling.k8s.io/foo")

const (
	testFinalizer = "example.com/test-finalizer"
)

var _ = SIGDescribe("SchedulerPreemption", func() {
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

	// 1. Run a low priority pod with finalizer which consumes 1/1 of node resources
	// 2. Schedule a higher priority pod which also consumes 1/1 of node resources
	// 3. See if the pod with lower priority is preempted and has the pod disruption condition
	// 4. Remove the finalizer so that the pod can be deleted by GC
	ginkgo.It("validates pod disruption condition is added to the preempted pod", func() {
		podRes := v1.ResourceList{testExtendedResource: resource.MustParse("1")}

		ginkgo.By("Select a node to run the lower and higher priority pods")
		framework.ExpectNotEqual(len(nodeList.Items), 0, "We need at least one node for the test to run")
		node := nodeList.Items[0]
		nodeCopy := node.DeepCopy()
		nodeCopy.Status.Capacity[testExtendedResource] = resource.MustParse("1")
		err := patchNode(cs, &node, nodeCopy)
		framework.ExpectNoError(err)

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
		victimPod := createPausePod(f, pausePodConfig{
			Name:              "victim-pod",
			PriorityClassName: lowPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
			Finalizers: []string{testFinalizer},
			Affinity:   &testNodeAffinity,
		})
		framework.Logf("Created pod: %v on node: %s", victimPod.Name, victimPod.Spec.NodeName)

		ginkgo.By("Wait for the victim pod to be scheduled")
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(cs, victimPod))

		// Remove the finalizer so that the victim pod can be GCed
		defer e2epod.NewPodClient(f).RemoveFinalizer(victimPod.Name, testFinalizer)

		ginkgo.By("Create a high priority pod to trigger preemption of the lower priority pod")
		preemptorPod := createPausePod(f, pausePodConfig{
			Name:              "preemptor-pod",
			PriorityClassName: highPriorityClassName,
			Resources: &v1.ResourceRequirements{
				Requests: podRes,
				Limits:   podRes,
			},
			Affinity: &testNodeAffinity,
		})
		framework.Logf("Created pod: %v on node: %s", preemptorPod.Name, preemptorPod.Spec.NodeName)

		ginkgo.By("Waiting for the victim pod to be terminating")
		err = e2epod.WaitForPodTerminatingInNamespaceTimeout(f.ClientSet, victimPod.Name, victimPod.Namespace, framework.PodDeleteTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying the pod has the pod disruption condition")
		e2epod.VerifyPodHasConditionWithType(f, victimPod, v1.DisruptionTarget)
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
