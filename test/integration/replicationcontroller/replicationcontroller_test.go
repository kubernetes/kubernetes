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

package replicationcontroller

import (
	"context"
	"fmt"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	interval = 100 * time.Millisecond
	timeout  = 60 * time.Second
)

func labelMap() map[string]string {
	return map[string]string{"foo": "bar"}
}

func newRC(name, namespace string, replicas int) *v1.ReplicationController {
	replicasCopy := int32(replicas)
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Selector: labelMap(),
			Replicas: &replicasCopy,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labelMap(),
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "fake-name",
							Image: "fakeimage",
						},
					},
				},
			},
		},
	}
}

func newMatchingPod(podName, namespace string) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: namespace,
			Labels:    labelMap(),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
		},
	}
}

func rmSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *replication.ReplicationManager, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "rc-informers")), resyncPeriod)

	rm := replication.NewReplicationManager(
		informers.Core().V1().Pods(),
		informers.Core().V1().ReplicationControllers(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "replication-controller")),
		replication.BurstReplicas,
	)

	if err != nil {
		t.Fatalf("Failed to create replication controller")
	}
	return s, closeFn, rm, informers, clientSet
}

// Run RC controller and informers
func runControllerAndInformers(t *testing.T, rm *replication.ReplicationManager, informers informers.SharedInformerFactory, podNum int) chan struct{} {
	stopCh := make(chan struct{})
	informers.Start(stopCh)
	waitToObservePods(t, informers.Core().V1().Pods().Informer(), podNum)
	go rm.Run(5, stopCh)
	return stopCh
}

// wait for the podInformer to observe the pods. Call this function before
// running the RC controller to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		return len(objects) == podNum, nil
	}); err != nil {
		t.Fatalf("Error encountered when waiting for podInformer to observe the pods: %v", err)
	}
}

func createRCsPods(t *testing.T, clientSet clientset.Interface, rcs []*v1.ReplicationController, pods []*v1.Pod) ([]*v1.ReplicationController, []*v1.Pod) {
	var createdRCs []*v1.ReplicationController
	var createdPods []*v1.Pod
	for _, rc := range rcs {
		createdRC, err := clientSet.CoreV1().ReplicationControllers(rc.Namespace).Create(context.TODO(), rc, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create replication controller %s: %v", rc.Name, err)
		}
		createdRCs = append(createdRCs, createdRC)
	}
	for _, pod := range pods {
		createdPod, err := clientSet.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
		createdPods = append(createdPods, createdPod)
	}

	return createdRCs, createdPods
}

// Verify .Status.Replicas is equal to .Spec.Replicas
func waitRCStable(t *testing.T, clientSet clientset.Interface, rc *v1.ReplicationController) {
	rcClient := clientSet.CoreV1().ReplicationControllers(rc.Namespace)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newRC.Status.Replicas == *rc.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.Replicas is equal to .Spec.Replicas for rc %s: %v", rc.Name, err)
	}
}

// Update .Spec.Replicas to replicas and verify .Status.Replicas is changed accordingly
func scaleRC(t *testing.T, c clientset.Interface, rc *v1.ReplicationController, replicas int32) {
	rcClient := c.CoreV1().ReplicationControllers(rc.Namespace)
	rc = updateRC(t, rcClient, rc.Name, func(rc *v1.ReplicationController) {
		*rc.Spec.Replicas = replicas
	})
	waitRCStable(t, c, rc)
}

func updatePod(t *testing.T, podClient typedv1.PodInterface, podName string, updateFunc func(*v1.Pod)) *v1.Pod {
	var pod *v1.Pod
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newPod)
		pod, err = podClient.Update(context.TODO(), newPod, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update pod %s: %v", podName, err)
	}
	return pod
}

func updatePodStatus(t *testing.T, podClient typedv1.PodInterface, pod *v1.Pod, updateStatusFunc func(*v1.Pod)) {
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateStatusFunc(newPod)
		_, err = podClient.UpdateStatus(context.TODO(), newPod, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}
}

func getPods(t *testing.T, podClient typedv1.PodInterface, labelMap map[string]string) *v1.PodList {
	podSelector := labels.Set(labelMap).AsSelector()
	options := metav1.ListOptions{LabelSelector: podSelector.String()}
	pods, err := podClient.List(context.TODO(), options)
	if err != nil {
		t.Fatalf("Failed obtaining a list of pods that match the pod labels %v: %v", labelMap, err)
	}
	return pods
}

func updateRC(t *testing.T, rcClient typedv1.ReplicationControllerInterface, rcName string, updateFunc func(*v1.ReplicationController)) *v1.ReplicationController {
	var rc *v1.ReplicationController
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newRC, err := rcClient.Get(context.TODO(), rcName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newRC)
		rc, err = rcClient.Update(context.TODO(), newRC, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update rc %s: %v", rcName, err)
	}
	return rc
}

// Verify ControllerRef of a RC pod that has incorrect attributes is automatically patched by the RC
func testPodControllerRefPatch(t *testing.T, c clientset.Interface, pod *v1.Pod, ownerReference *metav1.OwnerReference, rc *v1.ReplicationController, expectedOwnerReferenceNum int) {
	ns := rc.Namespace
	podClient := c.CoreV1().Pods(ns)
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.OwnerReferences = []metav1.OwnerReference{*ownerReference}
	})

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return metav1.GetControllerOf(newPod) != nil, nil
	}); err != nil {
		t.Fatalf("Failed to verify ControllerRef for the pod %s is not nil: %v", pod.Name, err)
	}

	newPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain pod %s: %v", pod.Name, err)
	}
	controllerRef := metav1.GetControllerOf(newPod)
	if controllerRef.UID != rc.UID {
		t.Fatalf("RC owner of the pod %s has a different UID: Expected %v, got %v", newPod.Name, rc.UID, controllerRef.UID)
	}
	ownerReferenceNum := len(newPod.GetOwnerReferences())
	if ownerReferenceNum != expectedOwnerReferenceNum {
		t.Fatalf("Unexpected number of owner references for pod %s: Expected %d, got %d", newPod.Name, expectedOwnerReferenceNum, ownerReferenceNum)
	}
}

func setPodsReadyCondition(t *testing.T, clientSet clientset.Interface, pods *v1.PodList, conditionStatus v1.ConditionStatus, lastTransitionTime time.Time) {
	replicas := int32(len(pods.Items))
	var readyPods int32
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		readyPods = 0
		for i := range pods.Items {
			pod := &pods.Items[i]
			if podutil.IsPodReady(pod) {
				readyPods++
				continue
			}
			pod.Status.Phase = v1.PodRunning
			_, condition := podutil.GetPodCondition(&pod.Status, v1.PodReady)
			if condition != nil {
				condition.Status = conditionStatus
				condition.LastTransitionTime = metav1.Time{Time: lastTransitionTime}
			} else {
				condition = &v1.PodCondition{
					Type:               v1.PodReady,
					Status:             conditionStatus,
					LastTransitionTime: metav1.Time{Time: lastTransitionTime},
				}
				pod.Status.Conditions = append(pod.Status.Conditions, *condition)
			}
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(context.TODO(), pod, metav1.UpdateOptions{})
			if err != nil {
				// When status fails to be updated, we continue to next pod
				continue
			}
			readyPods++
		}
		return readyPods >= replicas, nil
	})
	if err != nil {
		t.Fatalf("failed to mark all ReplicationController pods to ready: %v", err)
	}
}

func testScalingUsingScaleSubresource(t *testing.T, c clientset.Interface, rc *v1.ReplicationController, replicas int32) {
	ns := rc.Namespace
	rcClient := c.CoreV1().ReplicationControllers(ns)
	newRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain rc %s: %v", rc.Name, err)
	}
	scale, err := c.CoreV1().ReplicationControllers(ns).GetScale(context.TODO(), rc.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain scale subresource for rc %s: %v", rc.Name, err)
	}
	if scale.Spec.Replicas != *newRC.Spec.Replicas {
		t.Fatalf("Scale subresource for rc %s does not match .Spec.Replicas: expected %d, got %d", rc.Name, *newRC.Spec.Replicas, scale.Spec.Replicas)
	}

	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		scale, err := c.CoreV1().ReplicationControllers(ns).GetScale(context.TODO(), rc.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		scale.Spec.Replicas = replicas
		_, err = c.CoreV1().ReplicationControllers(ns).UpdateScale(context.TODO(), rc.Name, scale, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to set .Spec.Replicas of scale subresource for rc %s: %v", rc.Name, err)
	}

	newRC, err = rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain rc %s: %v", rc.Name, err)
	}
	if *newRC.Spec.Replicas != replicas {
		t.Fatalf(".Spec.Replicas of rc %s does not match its scale subresource: expected %d, got %d", rc.Name, replicas, *newRC.Spec.Replicas)
	}
}

func TestAdoption(t *testing.T) {
	boolPtr := func(b bool) *bool { return &b }
	testCases := []struct {
		name                    string
		existingOwnerReferences func(rc *v1.ReplicationController) []metav1.OwnerReference
		expectedOwnerReferences func(rc *v1.ReplicationController) []metav1.OwnerReference
	}{
		{
			"pod refers rc as an owner, not a controller",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController"}}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod doesn't have owner references",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod refers rc as a controller",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true)}}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true)}}
			},
		},
		{
			"pod refers other rc as the controller, refers the rc as an owner",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRC", APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true)},
					{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController"},
				}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRC", APIVersion: "v1", Kind: "ReplicationController", Controller: boolPtr(true)},
					{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController"},
				}
			},
		},
	}
	for i, tc := range testCases {
		func() {
			s, closeFn, rm, informers, clientSet := rmSetup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace(fmt.Sprintf("rc-adoption-%d", i), s, t)
			defer framework.DeleteTestingNamespace(ns, s, t)

			rcClient := clientSet.CoreV1().ReplicationControllers(ns.Name)
			podClient := clientSet.CoreV1().Pods(ns.Name)
			const rcName = "rc"
			rc, err := rcClient.Create(context.TODO(), newRC(rcName, ns.Name, 1), metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create replication controllers: %v", err)
			}
			podName := fmt.Sprintf("pod%d", i)
			pod := newMatchingPod(podName, ns.Name)
			pod.OwnerReferences = tc.existingOwnerReferences(rc)
			_, err = podClient.Create(context.TODO(), pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create Pod: %v", err)
			}

			stopCh := runControllerAndInformers(t, rm, informers, 1)
			defer close(stopCh)
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				updatedPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}

				e, a := tc.expectedOwnerReferences(rc), updatedPod.OwnerReferences
				if reflect.DeepEqual(e, a) {
					return true, nil
				}

				t.Logf("ownerReferences don't match, expect %v, got %v", e, a)
				return false, nil
			}); err != nil {
				t.Fatalf("test %q failed: %v", tc.name, err)
			}
		}()
	}
}

func TestSpecReplicasChange(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-spec-replicas-change", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 2)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// Update .Spec.Replicas and verify .Status.Replicas is changed accordingly
	scaleRC(t, c, rc, 3)
	scaleRC(t, c, rc, 0)
	scaleRC(t, c, rc, 2)

	// Add a template annotation change to test RC's status does update
	// without .Spec.Replicas change
	rcClient := c.CoreV1().ReplicationControllers(ns.Name)
	var oldGeneration int64
	newRC := updateRC(t, rcClient, rc.Name, func(rc *v1.ReplicationController) {
		oldGeneration = rc.Generation
		rc.Spec.Template.Annotations = map[string]string{"test": "annotation"}
	})
	savedGeneration := newRC.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("Failed to verify .Generation has incremented for rc %s", rc.Name)
	}

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newRC.Status.ObservedGeneration >= savedGeneration, nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.ObservedGeneration has incremented for rc %s: %v", rc.Name, err)
	}
}

func TestDeletingAndFailedPods(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-deleting-and-failed-pods", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 2)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// Verify RC creates 2 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
	}

	// Set first pod as deleting pod
	// Set finalizers for the pod to simulate pending deletion status
	deletingPod := &pods.Items[0]
	updatePod(t, podClient, deletingPod.Name, func(pod *v1.Pod) {
		pod.Finalizers = []string{"fake.example.com/blockDeletion"}
	})
	if err := c.CoreV1().Pods(ns.Name).Delete(context.TODO(), deletingPod.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Error deleting pod %s: %v", deletingPod.Name, err)
	}

	// Set second pod as failed pod
	failedPod := &pods.Items[1]
	updatePodStatus(t, podClient, failedPod, func(pod *v1.Pod) {
		pod.Status.Phase = v1.PodFailed
	})

	// Pool until 2 new pods have been created to replace deleting and failed pods
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods = getPods(t, podClient, labelMap())
		return len(pods.Items) == 4, nil
	}); err != nil {
		t.Fatalf("Failed to verify 2 new pods have been created (expected 4 pods): %v", err)
	}

	// Verify deleting and failed pods are among the four pods
	foundDeletingPod := false
	foundFailedPod := false
	for _, pod := range pods.Items {
		if pod.UID == deletingPod.UID {
			foundDeletingPod = true
		}
		if pod.UID == failedPod.UID {
			foundFailedPod = true
		}
	}
	// Verify deleting pod exists
	if !foundDeletingPod {
		t.Fatalf("expected deleting pod %s exists, but it is not found", deletingPod.Name)
	}
	// Verify failed pod exists
	if !foundFailedPod {
		t.Fatalf("expected failed pod %s exists, but it is not found", failedPod.Name)
	}
}

func TestOverlappingRCs(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-overlapping-rcs", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	// Create 2 RCs with identical selectors
	for i := 0; i < 2; i++ {
		// One RC has 1 replica, and another has 2 replicas
		rc := newRC(fmt.Sprintf("rc-%d", i+1), ns.Name, i+1)
		rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
		waitRCStable(t, c, rcs[0])
	}

	// Expect 3 total Pods to be created
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 3 {
		t.Errorf("len(pods) = %d, want 3", len(pods.Items))
	}

	// Expect both RCs have .status.replicas = .spec.replicas
	for i := 0; i < 2; i++ {
		newRC, err := c.CoreV1().ReplicationControllers(ns.Name).Get(context.TODO(), fmt.Sprintf("rc-%d", i+1), metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to obtain rc rc-%d: %v", i+1, err)
		}
		if newRC.Status.Replicas != *newRC.Spec.Replicas {
			t.Fatalf(".Status.Replicas %d is not equal to .Spec.Replicas %d", newRC.Status.Replicas, *newRC.Spec.Replicas)
		}
	}
}

func TestPodOrphaningAndAdoptionWhenLabelsChange(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-pod-orphaning-and-adoption-when-labels-change", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 1)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// Orphaning: RC should remove OwnerReference from a pod when the pod's labels change to not match its labels
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 1 {
		t.Fatalf("len(pods) = %d, want 1", len(pods.Items))
	}
	pod := &pods.Items[0]

	// Start by verifying ControllerRef for the pod is not nil
	if metav1.GetControllerOf(pod) == nil {
		t.Fatalf("ControllerRef of pod %s is nil", pod.Name)
	}
	newLabelMap := map[string]string{"new-foo": "new-bar"}
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.Labels = newLabelMap
	})
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		pod = newPod
		return metav1.GetControllerOf(newPod) == nil, nil
	}); err != nil {
		t.Fatalf("Failed to verify ControllerRef for the pod %s is nil: %v", pod.Name, err)
	}

	// Adoption: RC should add ControllerRef to a pod when the pod's labels change to match its labels
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.Labels = labelMap()
	})
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newPod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
		if err != nil {
			// If the pod is not found, it means the RC picks the pod for deletion (it is extra)
			// Verify there is only one pod in namespace and it has ControllerRef to the RC
			if apierrors.IsNotFound(err) {
				pods := getPods(t, podClient, labelMap())
				if len(pods.Items) != 1 {
					return false, fmt.Errorf("Expected 1 pod in current namespace, got %d", len(pods.Items))
				}
				// Set the pod accordingly
				pod = &pods.Items[0]
				return true, nil
			}
			return false, err
		}
		// Always update the pod so that we can save a GET call to API server later
		pod = newPod
		// If the pod is found, verify the pod has a ControllerRef
		return metav1.GetControllerOf(newPod) != nil, nil
	}); err != nil {
		t.Fatalf("Failed to verify ControllerRef for pod %s is not nil: %v", pod.Name, err)
	}
	// Verify the pod has a ControllerRef to the RC
	// Do nothing if the pod is nil (i.e., has been picked for deletion)
	if pod != nil {
		controllerRef := metav1.GetControllerOf(pod)
		if controllerRef.UID != rc.UID {
			t.Fatalf("RC owner of the pod %s has a different UID: Expected %v, got %v", pod.Name, rc.UID, controllerRef.UID)
		}
	}
}

func TestGeneralPodAdoption(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-general-pod-adoption", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 1)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 1 {
		t.Fatalf("len(pods) = %d, want 1", len(pods.Items))
	}
	pod := &pods.Items[0]
	var falseVar = false

	// When the only OwnerReference of the pod points to another type of API object such as statefulset
	// with Controller=false, the RC should add a second OwnerReference (ControllerRef) pointing to itself
	// with Controller=true
	ownerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "apps/v1beta1", Kind: "StatefulSet", Name: rc.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rc, 2)

	// When the only OwnerReference of the pod points to the RC, but Controller=false
	ownerReference = metav1.OwnerReference{UID: rc.UID, APIVersion: "v1", Kind: "ReplicationController", Name: rc.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rc, 1)
}

func TestReadyAndAvailableReplicas(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-ready-and-available-replicas", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 3)
	rc.Spec.MinReadySeconds = 3600
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// First verify no pod is available
	if rc.Status.AvailableReplicas != 0 {
		t.Fatalf("Unexpected .Status.AvailableReplicas: Expected 0, saw %d", rc.Status.AvailableReplicas)
	}

	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 3 {
		t.Fatalf("len(pods) = %d, want 3", len(pods.Items))
	}

	// Separate 3 pods into their own list
	firstPodList := &v1.PodList{Items: pods.Items[:1]}
	secondPodList := &v1.PodList{Items: pods.Items[1:2]}
	thirdPodList := &v1.PodList{Items: pods.Items[2:]}
	// First pod: Running, but not Ready
	// by setting the Ready condition to false with LastTransitionTime to be now
	setPodsReadyCondition(t, c, firstPodList, v1.ConditionFalse, time.Now())
	// Second pod: Running and Ready, but not Available
	// by setting LastTransitionTime to now
	setPodsReadyCondition(t, c, secondPodList, v1.ConditionTrue, time.Now())
	// Third pod: Running, Ready, and Available
	// by setting LastTransitionTime to more than 3600 seconds ago
	setPodsReadyCondition(t, c, thirdPodList, v1.ConditionTrue, time.Now().Add(-120*time.Minute))

	rcClient := c.CoreV1().ReplicationControllers(ns.Name)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// Verify 3 pods exist, 2 pods are Ready, and 1 pod is Available
		return newRC.Status.Replicas == 3 && newRC.Status.ReadyReplicas == 2 && newRC.Status.AvailableReplicas == 1, nil
	}); err != nil {
		t.Fatalf("Failed to verify number of Replicas, ReadyReplicas and AvailableReplicas of rc %s to be as expected: %v", rc.Name, err)
	}
}

func TestRCScaleSubresource(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-rc-scale-subresource", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rc := newRC("rc", ns.Name, 1)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// Use scale subresource to scale up .Spec.Replicas to 3
	testScalingUsingScaleSubresource(t, c, rc, 3)
	// Use the scale subresource to scale down .Spec.Replicas to 0
	testScalingUsingScaleSubresource(t, c, rc, 0)
}

func TestExtraPodsAdoptionAndDeletion(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-extra-pods-adoption-and-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	rc := newRC("rc", ns.Name, 2)
	// Create 3 pods, RC should adopt only 2 of them
	podList := []*v1.Pod{}
	for i := 0; i < 3; i++ {
		pod := newMatchingPod(fmt.Sprintf("pod-%d", i+1), ns.Name)
		pod.Labels = labelMap()
		podList = append(podList, pod)
	}
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, podList)
	rc = rcs[0]
	stopCh := runControllerAndInformers(t, rm, informers, 3)
	defer close(stopCh)
	waitRCStable(t, c, rc)

	// Verify the extra pod is deleted eventually by determining whether number of
	// all pods within namespace matches .spec.replicas of the RC (2 in this case)
	podClient := c.CoreV1().Pods(ns.Name)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		// All pods have labelMap as their labels
		pods := getPods(t, podClient, labelMap())
		return int32(len(pods.Items)) == *rc.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("Failed to verify number of all pods within current namespace matches .spec.replicas of rc %s: %v", rc.Name, err)
	}
}

func TestFullyLabeledReplicas(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-fully-labeled-replicas", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	extraLabelMap := map[string]string{"foo": "bar", "extraKey": "extraValue"}
	rc := newRC("rc", ns.Name, 2)
	rcs, _ := createRCsPods(t, c, []*v1.ReplicationController{rc}, []*v1.Pod{})
	rc = rcs[0]
	waitRCStable(t, c, rc)

	// Change RC's template labels to have extra labels, but not its selector
	rcClient := c.CoreV1().ReplicationControllers(ns.Name)
	updateRC(t, rcClient, rc.Name, func(rc *v1.ReplicationController) {
		rc.Spec.Template.Labels = extraLabelMap
	})

	// Set one of the pods to have extra labels
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
	}
	fullyLabeledPod := &pods.Items[0]
	updatePod(t, podClient, fullyLabeledPod.Name, func(pod *v1.Pod) {
		pod.Labels = extraLabelMap
	})

	// Verify only one pod is fully labeled
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return (newRC.Status.Replicas == 2 && newRC.Status.FullyLabeledReplicas == 1), nil
	}); err != nil {
		t.Fatalf("Failed to verify only one pod is fully labeled: %v", err)
	}
}
