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
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"

	"k8s.io/client-go/tools/cache"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	testutils "k8s.io/kubernetes/test/utils"

	. "github.com/onsi/ginkgo"
	_ "github.com/stretchr/testify/assert"
)

func getTestTaint() v1.Taint {
	return v1.Taint{
		Key:       "kubernetes.io/e2e-evict-taint-key",
		Value:     "evictTaintVal",
		Effect:    v1.TaintEffectNoExecute,
		TimeAdded: metav1.Now(),
	}
}

// Creates a defaut pod for this test, with argument saying if the Pod should have
// toleration for Taits used in this test.
func createPodForTaintsTest(hasToleration bool, tolerationSeconds int, podName, ns string) *v1.Pod {
	grace := int64(1)
	if !hasToleration {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      podName,
				Namespace: ns,
				Labels:    map[string]string{"name": podName},
				DeletionGracePeriodSeconds: &grace,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "pause",
						Image: "kubernetes/pause",
					},
				},
			},
		}
	} else {
		if tolerationSeconds <= 0 {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: ns,
					Labels:    map[string]string{"name": podName},
					DeletionGracePeriodSeconds: &grace,
					// default - tolerate forever
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "pause",
							Image: "kubernetes/pause",
						},
					},
					Tolerations: []v1.Toleration{{Key: "kubernetes.io/e2e-evict-taint-key", Value: "evictTaintVal", Effect: v1.TaintEffectNoExecute}},
				},
			}
		} else {
			ts := int64(tolerationSeconds)
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      podName,
					Namespace: ns,
					Labels:    map[string]string{"name": podName},
					DeletionGracePeriodSeconds: &grace,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "pause",
							Image: "kubernetes/pause",
						},
					},
					// default - tolerate forever
					Tolerations: []v1.Toleration{{Key: "kubernetes.io/e2e-evict-taint-key", Value: "evictTaintVal", Effect: v1.TaintEffectNoExecute, TolerationSeconds: &ts}},
				},
			}
		}
	}
}

// Creates and starts a controller (informer) that watches updates on a pod in given namespace with given name. It puts a new
// struct into observedDeletion channel for every deletion it sees.
func createTestController(cs clientset.Interface, observedDeletions chan struct{}, stopCh chan struct{}, podName, ns string) {
	_, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.FieldSelector = fields.SelectorFromSet(fields.Set{"metadata.name": podName}).String()
				obj, err := cs.CoreV1().Pods(ns).List(options)
				return runtime.Object(obj), err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.FieldSelector = fields.SelectorFromSet(fields.Set{"metadata.name": podName}).String()
				return cs.CoreV1().Pods(ns).Watch(options)
			},
		},
		&v1.Pod{},
		0,
		cache.ResourceEventHandlerFuncs{
			DeleteFunc: func(oldObj interface{}) { observedDeletions <- struct{}{} },
		},
	)
	framework.Logf("Starting informer...")
	go controller.Run(stopCh)
}

const (
	KubeletPodDeletionDelaySeconds = 60
	AdditionalWaitPerDeleteSeconds = 5
)

// Tests the behavior of NoExecuteTaintManager. Following scenarios are included:
// - eviction of non-tolerating pods from a tainted node,
// - lack of eviction of tolerating pods from a tainted node,
// - delayed eviction of short-tolerating pod from a tainted node,
// - lack of eviction of short-tolerating pod after taint removal.
var _ = SIGDescribe("NoExecuteTaintManager [Serial]", func() {
	var cs clientset.Interface
	var nodeList *v1.NodeList
	var ns string
	f := framework.NewDefaultFramework("taint-control")

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		nodeList = &v1.NodeList{}

		framework.WaitForAllNodesHealthy(cs, time.Minute)

		err := framework.CheckTestingNSDeletedExcept(cs, ns)
		framework.ExpectNoError(err)
	})

	// 1. Run a pod
	// 2. Taint the node running this pod with a no-execute taint
	// 3. See if pod will get evicted
	It("evicts pods from tainted nodes", func() {
		podName := "taint-eviction-1"
		pod := createPodForTaintsTest(false, 0, podName, ns)
		observedDeletions := make(chan struct{}, 100)
		stopCh := make(chan struct{})
		createTestController(cs, observedDeletions, stopCh, podName, ns)

		By("Staring pod...")
		nodeName, err := testutils.RunPodAndGetNodeName(cs, pod, 2*time.Minute)
		framework.ExpectNoError(err)
		framework.Logf("Pod is running on %v. Tainting Node", nodeName)

		By("Trying to apply a taint on the Node")
		testTaint := getTestTaint()
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		// Wait a bit
		By("Waiting for Pod to be deleted")
		timeoutChannel := time.NewTimer(time.Duration(KubeletPodDeletionDelaySeconds+AdditionalWaitPerDeleteSeconds) * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Failed to evict Pod")
		case <-observedDeletions:
			framework.Logf("Noticed Pod eviction. Test successful")
		}
	})

	// 1. Run a pod with toleration
	// 2. Taint the node running this pod with a no-execute taint
	// 3. See if pod wont get evicted
	It("doesn't evict pod with tolerations from tainted nodes", func() {
		podName := "taint-eviction-2"
		pod := createPodForTaintsTest(true, 0, podName, ns)
		observedDeletions := make(chan struct{}, 100)
		stopCh := make(chan struct{})
		createTestController(cs, observedDeletions, stopCh, podName, ns)

		By("Staring pod...")
		nodeName, err := testutils.RunPodAndGetNodeName(cs, pod, 2*time.Minute)
		framework.ExpectNoError(err)
		framework.Logf("Pod is running on %v. Tainting Node", nodeName)

		By("Trying to apply a taint on the Node")
		testTaint := getTestTaint()
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		// Wait a bit
		By("Waiting for Pod to be deleted")
		timeoutChannel := time.NewTimer(time.Duration(KubeletPodDeletionDelaySeconds+AdditionalWaitPerDeleteSeconds) * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Logf("Pod wasn't evicted. Test successful")
		case <-observedDeletions:
			framework.Failf("Pod was evicted despite toleration")
		}
	})

	// 1. Run a pod with a finite toleration
	// 2. Taint the node running this pod with a no-execute taint
	// 3. See if pod wont get evicted before toleration time runs out
	// 4. See if pod will get evicted after toleration time runs out
	It("eventually evict pod with finite tolerations from tainted nodes", func() {
		podName := "taint-eviction-3"
		pod := createPodForTaintsTest(true, KubeletPodDeletionDelaySeconds+2*AdditionalWaitPerDeleteSeconds, podName, ns)
		observedDeletions := make(chan struct{}, 100)
		stopCh := make(chan struct{})
		createTestController(cs, observedDeletions, stopCh, podName, ns)

		By("Staring pod...")
		nodeName, err := testutils.RunPodAndGetNodeName(cs, pod, 2*time.Minute)
		framework.ExpectNoError(err)
		framework.Logf("Pod is running on %v. Tainting Node", nodeName)

		By("Trying to apply a taint on the Node")
		testTaint := getTestTaint()
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		defer framework.RemoveTaintOffNode(cs, nodeName, testTaint)

		// Wait a bit
		By("Waiting to see if a Pod won't be deleted")
		timeoutChannel := time.NewTimer(time.Duration(KubeletPodDeletionDelaySeconds+AdditionalWaitPerDeleteSeconds) * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Logf("Pod wasn't evicted")
		case <-observedDeletions:
			framework.Failf("Pod was evicted despite toleration")
			return
		}
		By("Waiting for Pod to be deleted")
		timeoutChannel = time.NewTimer(time.Duration(KubeletPodDeletionDelaySeconds+AdditionalWaitPerDeleteSeconds) * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Failf("Pod wasn't evicted")
		case <-observedDeletions:
			framework.Logf("Pod was evicted after toleration time run out. Test successful")
			return
		}
	})

	// 1. Run a pod with short toleration
	// 2. Taint the node running this pod with a no-execute taint
	// 3. Wait some time
	// 4. Remove the taint
	// 5. See if Pod won't be evicted.
	It("removing taint cancels eviction", func() {
		podName := "taint-eviction-4"
		pod := createPodForTaintsTest(true, 2*AdditionalWaitPerDeleteSeconds, podName, ns)
		observedDeletions := make(chan struct{}, 100)
		stopCh := make(chan struct{})
		createTestController(cs, observedDeletions, stopCh, podName, ns)

		By("Staring pod...")
		nodeName, err := testutils.RunPodAndGetNodeName(cs, pod, 2*time.Minute)
		framework.ExpectNoError(err)
		framework.Logf("Pod is running on %v. Tainting Node", nodeName)

		By("Trying to apply a taint on the Node")
		testTaint := getTestTaint()
		framework.AddOrUpdateTaintOnNode(cs, nodeName, testTaint)
		framework.ExpectNodeHasTaint(cs, nodeName, &testTaint)
		taintRemoved := false
		defer func() {
			if !taintRemoved {
				framework.RemoveTaintOffNode(cs, nodeName, testTaint)
			}
		}()

		// Wait a bit
		By("Waiting short time to make sure Pod is queued for deletion")
		timeoutChannel := time.NewTimer(AdditionalWaitPerDeleteSeconds).C
		select {
		case <-timeoutChannel:
			framework.Logf("Pod wasn't evicted. Proceeding")
		case <-observedDeletions:
			framework.Failf("Pod was evicted despite toleration")
			return
		}
		framework.Logf("Removing taint from Node")
		framework.RemoveTaintOffNode(cs, nodeName, testTaint)
		taintRemoved = true
		By("Waiting some time to make sure that toleration time passed.")
		timeoutChannel = time.NewTimer(time.Duration(KubeletPodDeletionDelaySeconds+3*AdditionalWaitPerDeleteSeconds) * time.Second).C
		select {
		case <-timeoutChannel:
			framework.Logf("Pod wasn't evicted. Test successful")
		case <-observedDeletions:
			framework.Failf("Pod was evicted despite toleration")
		}
	})
})
