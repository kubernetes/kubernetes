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
	"fmt"
	"k8s.io/apimachinery/pkg/util/uuid"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	// ensure libs have a chance to initialize
	_ "github.com/stretchr/testify/assert"
)

const (
	nodeAffinityEvictionDelaySeconds = 30
)

// Create specified NodeAffinity
func createRequiredDuringExecutionNodeAffinity(key string, operator v1.NodeSelectorOperator, values ...string) *v1.Affinity {
	return &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingRequiredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      key,
								Operator: operator,
								Values:   values,
							},
						},
					},
				},
			},
		},
	}
}

// Creates and starts a controller that watches updates on a pod in given namespace.
// puts the deleted pod name into observedDeletion channel when it sees pod get deleted
// It podName is not empty, it only observe the pod with given name
func createTestNodeAffinityController(cs clientset.Interface, observedDeletions chan string, stopCh chan struct{}, podName, ns string) {
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				obj, err := cs.CoreV1().Pods(ns).List(context.TODO(), options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return cs.CoreV1().Pods(ns).Watch(context.TODO(), options)
			},
		},
		&v1.Pod{},
		0,
		cache.ResourceEventHandlerFuncs{
			DeleteFunc: func(oldObj interface{}) {
				if delPod, ok := oldObj.(*v1.Pod); ok && (podName == "" || delPod.Name == podName) {
					observedDeletions <- delPod.Name
				}
			},
		},
	)
	go controller.Run(stopCh)
}

// Tests the behavior of ExecutionNodeAffinityManager. All operators are included.
var _ = SIGDescribe("ExecutionNodeAffinityManager Single Pod [Serial]", func() {
	var cs clientset.Interface
	var ns string
	f := framework.NewDefaultFramework("node-affinity-evict-single-pod")

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

		framework.AllNodesReady(cs, time.Minute)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	ginkgo.It("pod should survive", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label"
		v2 := "testing-node-affinity-label-2"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-in"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpIn, v, v2),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// re-label the node
		ginkgo.By("Re-label the node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v2)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v2)

		ginkgo.By("Waiting some time to make sure pod still alive")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Logf("Pod wasn't evicted. Test successful.")
		case <-observedDeletions:
			framework.Failf("Noticed Pod eviction but should survive.")
		}
	})

	ginkgo.It("evicts pods with In expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-in"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpIn, v),
		})

		// RequiredDuringSchedulingRequiredDuringExecution should also make sure the pod is scheduled to the wanted node
		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// re-label the node
		ginkgo.By("Re-label the node")
		changedV := v + "-changed"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, changedV)
		framework.ExpectNodeHasLabel(cs, nodeName, k, changedV)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	ginkgo.It("evicts pods with NotIn expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label"
		nodeName := GetNodeThatCanRunPod(f)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-not-in"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			NodeName: nodeName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpNotIn, v),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// label the node
		ginkgo.By("Trying to apply label on node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	ginkgo.It("evicts pods with Exists expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node.")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-exists"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpExists),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// delete the label
		ginkgo.By("Delete the label from node")
		framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	ginkgo.It("evicts pods with DoesNotExist expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label"
		nodeName := GetNodeThatCanRunPod(f)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-does-not-exists"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			NodeName: nodeName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpDoesNotExist),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// label the node
		ginkgo.By("Trying to apply label on node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	ginkgo.It("evicts pods with Gt expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "10"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node.")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-gt"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpGt, "5"),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// re-label the node
		ginkgo.By("Re-label the node")
		changedV := "0"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, changedV)
		framework.ExpectNodeHasLabel(cs, nodeName, k, changedV)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	ginkgo.It("evicts pods with Lt expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "0"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node.")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		// observe the pod
		observedDeletions := make(chan string, 1)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-op-lt"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, podName, ns)

		// Run the pod
		ginkgo.By("Starting pod " + podName)
		createPausePod(f, pausePodConfig{
			Name:     podName,
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpLt, "5"),
		})

		framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
		gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)

		// re-label the node
		ginkgo.By("Re-label the node")
		changedV := "10"
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, changedV)
		framework.ExpectNodeHasLabel(cs, nodeName, k, changedV)

		ginkgo.By("Waiting pod to be deleted")
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})
})

var _ = SIGDescribe("ExecutionNodeAffinityManager Multiple Pods [Serial]", func() {
	var cs clientset.Interface
	var ns string
	f := framework.NewDefaultFramework("node-affinity-evict-multiple-pod")

	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name

		framework.AllNodesReady(cs, time.Minute)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	ginkgo.It("evicts pods with In expression", func() {
		k := fmt.Sprintf("kubernetes.io/e2e-node-affinity-%s", string(uuid.NewUUID()))
		v := "testing-node-affinity-label-1"
		v2 := "testing-node-affinity-label-2"
		v3 := "testing-node-affinity-label-3"
		nodeName := GetNodeThatCanRunPod(f)

		// label the node
		ginkgo.By("Trying to apply label on node")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v)
		defer framework.RemoveLabelOffNode(cs, nodeName, k)

		// observe the pod
		observedDeletions := make(chan string, 10)
		stopCh := make(chan struct{})
		podName := "pod-with-node-affinity-"
		createTestNodeAffinityController(cs, observedDeletions, stopCh, "", ns)

		// Run the pod
		ginkgo.By("Starting pods...")
		createPausePod(f, pausePodConfig{
			Name:     podName + "1",
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpIn, v),
		})
		createPausePod(f, pausePodConfig{
			Name:     podName + "2",
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpNotIn, v3),
		})
		createPausePod(f, pausePodConfig{
			Name:     podName + "3",
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpIn, v, v2),
		})
		createPausePod(f, pausePodConfig{
			Name:     podName + "4",
			Affinity: createRequiredDuringExecutionNodeAffinity(k, v1.NodeSelectorOpExists),
		})

		expectPodRunning := func(podName string) {
			framework.ExpectNoError(e2epod.WaitForPodNotPending(cs, ns, podName))
			gotPod, err := cs.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(gotPod.Spec.NodeName, nodeName)
		}
		expectPodRunning(podName + "1")
		expectPodRunning(podName + "2")
		expectPodRunning(podName + "3")
		expectPodRunning(podName + "4")

		ginkgo.By("Re-label the node to evict Pod1")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v2)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v2)

		ginkgo.By("Waiting Pod1 to be deleted")
		evicted := 0
		timeoutChannel := time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
	out:
		for {
			select {
			case <-timeoutChannel:
				if evicted == 0 {
					framework.Failf("Failed to evict Pods.")
				} else if evicted > 1 {
					framework.Failf("Pod1 is evicted. But others Pods also get evicted.")
				}
				break out
			case pod := <-observedDeletions:
				evicted++
				if pod == podName+"1" {
					framework.Logf("Noticed Pod %q gets evicted.", pod)
				} else {
					framework.Failf("Unexepected Pod %q gets evicted.", pod)
					break out
				}
			}
		}

		ginkgo.By("Re-label the node to evict Pod1 and Pod2")
		framework.AddOrUpdateLabelOnNode(cs, nodeName, k, v3)
		framework.ExpectNodeHasLabel(cs, nodeName, k, v3)

		ginkgo.By("Waiting Pod2 and Pod3 to be deleted")
		evicted = 0
		timeoutChannel = time.NewTimer(nodeAffinityEvictionDelaySeconds * time.Second).C
		for {
			select {
			case <-timeoutChannel:
				if evicted == 0 {
					framework.Failf("Failed to evict Pods.")
				} else if evicted != 2 {
					framework.Failf("Expect 2 Pods be evicted but got %v", evicted)
				}
				return
			case pod := <-observedDeletions:
				evicted++
				if pod == podName+"2" || pod == podName+"3" {
					framework.Logf("Noticed Pod %q gets evicted.", pod)
				} else {
					framework.Failf("Unexepected Pod %q gets evicted.", pod)
					return
				}
			}
		}
	})
})
