/*
Copyright 2022 The Kubernetes Authors.

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

package podgc

import (
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/podgc"
	"k8s.io/kubernetes/pkg/features"
	testutils "k8s.io/kubernetes/test/integration/util"
	"k8s.io/utils/pointer"
)

// TestPodGcOrphanedPodsWithFinalizer tests deletion of orphaned pods
func TestPodGcOrphanedPodsWithFinalizer(t *testing.T) {
	testCtx := setup(t, "podgc-orphaned")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node",
		},
		Spec: v1.NodeSpec{},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionTrue,
				},
			},
		},
	}
	node, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node '%v', err: %v", node.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "testpod",
			Namespace:  testCtx.NS.Name,
			Finalizers: []string{"test.k8s.io/finalizer"},
		},
		Spec: v1.PodSpec{
			NodeName: node.Name,
			Containers: []v1.Container{
				{Name: "foo", Image: "bar"},
			},
		},
	}

	pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error %v, while creating pod: %v", err, klog.KObj(pod))
	}
	defer testutils.RemovePodFinalizers(testCtx.ClientSet, t, []*v1.Pod{pod})
	pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Error: '%v' while updating pod info: '%v'", err, klog.KObj(pod))
	}

	// we delete the node to orphan the pod
	err = cs.CoreV1().Nodes().Delete(testCtx.Ctx, pod.Spec.NodeName, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete node: %v, err: %v", pod.Spec.NodeName, err)
	}

	err = wait.PollImmediate(time.Second, time.Second*15, func() (bool, error) {
		updatedPod, err := cs.CoreV1().Pods(testCtx.NS.Name).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err != nil {
			return true, err
		}
		if updatedPod.ObjectMeta.DeletionTimestamp != nil {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Error '%v' while waiting for the pod '%v' to be deleted", err, klog.KObj(pod))
	}
}

// TestTerminatingOnOutOfServiceNode tests deletion pods terminating on out-of-service nodes
func TestTerminatingOnOutOfServiceNode(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeOutOfServiceVolumeDetach, true)()
	testCtx := setup(t, "podgc-out-of-service")
	defer testutils.CleanupTest(t, testCtx)
	cs := testCtx.ClientSet

	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node",
		},
		Spec: v1.NodeSpec{},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:   v1.NodeReady,
					Status: v1.ConditionFalse,
				},
			},
		},
	}
	node, err := cs.CoreV1().Nodes().Create(testCtx.Ctx, node, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create node '%v', err: %v", node.Name, err)
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testpod",
			Namespace: testCtx.NS.Name,
		},
		Spec: v1.PodSpec{
			NodeName: node.Name,
			Containers: []v1.Container{
				{Name: "foo", Image: "bar"},
			},
		},
	}

	pod, err = cs.CoreV1().Pods(testCtx.NS.Name).Create(testCtx.Ctx, pod, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error %v, while creating pod: %v", err, klog.KObj(pod))
	}

	// trigger termination of the pod, but with long grace period so that it is not removed immediately
	err = cs.CoreV1().Pods(testCtx.NS.Name).Delete(testCtx.Ctx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: pointer.Int64(300)})
	if err != nil {
		t.Fatalf("Error: '%v' while deleting pod: '%v'", err, klog.KObj(pod))
	}

	// taint the node with the out-of-service taint
	err = testutils.AddTaintToNode(cs, pod.Spec.NodeName, v1.Taint{Key: v1.TaintNodeOutOfService, Value: "", Effect: v1.TaintEffectNoExecute})
	if err != nil {
		t.Fatalf("Failed to taint node: %v, err: %v", pod.Spec.NodeName, err)
	}

	// wait until the pod is deleted
	err = wait.PollImmediate(time.Second, time.Second*15, func() (bool, error) {
		updatedPod, err := cs.CoreV1().Pods(pod.Namespace).Get(testCtx.Ctx, pod.Name, metav1.GetOptions{})
		if err == nil {
			return updatedPod == nil, nil
		}
		// there was an error
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		t.Fatalf("Error '%v' while waiting for the pod '%v' to be deleted", err, klog.KObj(pod))
	}
}

func setup(t *testing.T, name string) *testutils.TestContext {
	testCtx := testutils.InitTestAPIServer(t, name, nil)
	externalInformers := informers.NewSharedInformerFactory(testCtx.ClientSet, time.Second)

	podgc := podgc.NewPodGCInternal(testCtx.Ctx,
		testCtx.ClientSet,
		externalInformers.Core().V1().Pods(),
		externalInformers.Core().V1().Nodes(),
		0,
		500*time.Millisecond,
		time.Second)

	// Waiting for all controllers to sync
	externalInformers.Start(testCtx.Ctx.Done())
	externalInformers.WaitForCacheSync(testCtx.Ctx.Done())

	go podgc.Run(testCtx.Ctx)
	return testCtx
}
