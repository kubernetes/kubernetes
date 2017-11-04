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

package replicaset

import (
	"fmt"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	typedv1beta1 "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	interval = 100 * time.Millisecond
	timeout  = 60 * time.Second
)

func labelMap() map[string]string {
	return map[string]string{"foo": "bar"}
}

func newRS(name, namespace string, replicas int) *v1beta1.ReplicaSet {
	replicasCopy := int32(replicas)
	return &v1beta1.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1beta1.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labelMap(),
			},
			Replicas: &replicasCopy,
			Template: v1.PodTemplateSpec{
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

// verifyRemainingObjects verifies if the number of the remaining replica
// sets and pods are rsNum and podNum. It returns error if the
// communication with the API server fails.
func verifyRemainingObjects(t *testing.T, clientSet clientset.Interface, namespace string, rsNum, podNum int) (bool, error) {
	rsClient := clientSet.Extensions().ReplicaSets(namespace)
	podClient := clientSet.CoreV1().Pods(namespace)
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		t.Logf("expect %d pods, got %d pods", podNum, len(pods.Items))
	}
	rss, err := rsClient.List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replica sets: %v", err)
	}
	if len(rss.Items) != rsNum {
		ret = false
		t.Logf("expect %d RSs, got %d RSs", rsNum, len(rss.Items))
	}
	return ret, nil
}

func rmSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *replicaset.ReplicaSetController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "rs-informers")), resyncPeriod)

	rm := replicaset.NewReplicaSetController(
		informers.Extensions().V1beta1().ReplicaSets(),
		informers.Core().V1().Pods(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "replicaset-controller")),
		replicaset.BurstReplicas,
	)

	if err != nil {
		t.Fatalf("Failed to create replicaset controller")
	}
	return s, closeFn, rm, informers, clientSet
}

func rmSimpleSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	return s, closeFn, clientSet
}

// Run RS controller and informers
func runControllerAndInformers(t *testing.T, rm *replicaset.ReplicaSetController, informers informers.SharedInformerFactory, podNum int) chan struct{} {
	stopCh := make(chan struct{})
	informers.Start(stopCh)
	waitToObservePods(t, informers.Core().V1().Pods().Informer(), podNum)
	go rm.Run(5, stopCh)
	return stopCh
}

// wait for the podInformer to observe the pods. Call this function before
// running the RS controller to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		return len(objects) == podNum, nil
	}); err != nil {
		t.Fatalf("Error encountered when waiting for podInformer to observe the pods: %v", err)
	}
}

func createRSsPods(t *testing.T, clientSet clientset.Interface, rss []*v1beta1.ReplicaSet, pods []*v1.Pod) ([]*v1beta1.ReplicaSet, []*v1.Pod) {
	var createdRSs []*v1beta1.ReplicaSet
	var createdPods []*v1.Pod
	for _, rs := range rss {
		createdRS, err := clientSet.Extensions().ReplicaSets(rs.Namespace).Create(rs)
		if err != nil {
			t.Fatalf("Failed to create replica set %s: %v", rs.Name, err)
		}
		createdRSs = append(createdRSs, createdRS)
	}
	for _, pod := range pods {
		createdPod, err := clientSet.CoreV1().Pods(pod.Namespace).Create(pod)
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
		createdPods = append(createdPods, createdPod)
	}

	return createdRSs, createdPods
}

// Verify .Status.Replicas is equal to .Spec.Replicas
func waitRSStable(t *testing.T, clientSet clientset.Interface, rs *v1beta1.ReplicaSet) {
	rsClient := clientSet.Extensions().ReplicaSets(rs.Namespace)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newRS.Status.Replicas == *rs.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.Replicas is equal to .Spec.Replicas for rs %s: %v", rs.Name, err)
	}
}

// Update .Spec.Replicas to replicas and verify .Status.Replicas is changed accordingly
func scaleRS(t *testing.T, c clientset.Interface, rs *v1beta1.ReplicaSet, replicas int32) {
	rsClient := c.Extensions().ReplicaSets(rs.Namespace)
	rs = updateRS(t, rsClient, rs.Name, func(rs *v1beta1.ReplicaSet) {
		*rs.Spec.Replicas = replicas
	})
	waitRSStable(t, c, rs)
}

func updatePod(t *testing.T, podClient typedv1.PodInterface, podName string, updateFunc func(*v1.Pod)) *v1.Pod {
	var pod *v1.Pod
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newPod)
		pod, err = podClient.Update(newPod)
		return err
	}); err != nil {
		t.Fatalf("Failed to update pod %s: %v", podName, err)
	}
	return pod
}

func updatePodStatus(t *testing.T, podClient typedv1.PodInterface, pod *v1.Pod, updateStatusFunc func(*v1.Pod)) {
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateStatusFunc(newPod)
		_, err = podClient.UpdateStatus(newPod)
		return err
	}); err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", pod.Name, err)
	}
}

func getPods(t *testing.T, podClient typedv1.PodInterface, labelMap map[string]string) *v1.PodList {
	podSelector := labels.Set(labelMap).AsSelector()
	options := metav1.ListOptions{LabelSelector: podSelector.String()}
	pods, err := podClient.List(options)
	if err != nil {
		t.Fatalf("Failed obtaining a list of pods that match the pod labels %v: %v", labelMap, err)
	}
	return pods
}

func updateRS(t *testing.T, rsClient typedv1beta1.ReplicaSetInterface, rsName string, updateFunc func(*v1beta1.ReplicaSet)) *v1beta1.ReplicaSet {
	var rs *v1beta1.ReplicaSet
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newRS, err := rsClient.Get(rsName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newRS)
		rs, err = rsClient.Update(newRS)
		return err
	}); err != nil {
		t.Fatalf("Failed to update rs %s: %v", rsName, err)
	}
	return rs
}

// Verify ControllerRef of a RS pod that has incorrect attributes is automatically patched by the RS
func testPodControllerRefPatch(t *testing.T, c clientset.Interface, pod *v1.Pod, ownerReference *metav1.OwnerReference, rs *v1beta1.ReplicaSet, expectedOwnerReferenceNum int) {
	ns := rs.Namespace
	podClient := c.CoreV1().Pods(ns)
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.OwnerReferences = []metav1.OwnerReference{*ownerReference}
	})

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return metav1.GetControllerOf(newPod) != nil, nil
	}); err != nil {
		t.Fatalf("Failed to verify ControllerRef for the pod %s is not nil: %v", pod.Name, err)
	}

	newPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain pod %s: %v", pod.Name, err)
	}
	controllerRef := metav1.GetControllerOf(newPod)
	if controllerRef.UID != rs.UID {
		t.Fatalf("RS owner of the pod %s has a different UID: Expected %v, got %v", newPod.Name, rs.UID, controllerRef.UID)
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
			_, err := clientSet.CoreV1().Pods(pod.Namespace).UpdateStatus(pod)
			if err != nil {
				// When status fails to be updated, we continue to next pod
				continue
			}
			readyPods++
		}
		return readyPods >= replicas, nil
	})
	if err != nil {
		t.Fatalf("failed to mark all ReplicaSet pods to ready: %v", err)
	}
}

func testScalingUsingScaleSubresource(t *testing.T, c clientset.Interface, rs *v1beta1.ReplicaSet, replicas int32) {
	ns := rs.Namespace
	rsClient := c.Extensions().ReplicaSets(ns)
	newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain rs %s: %v", rs.Name, err)
	}
	kind := "ReplicaSet"
	scaleClient := c.ExtensionsV1beta1().Scales(ns)
	scale, err := scaleClient.Get(kind, rs.Name)
	if err != nil {
		t.Fatalf("Failed to obtain scale subresource for rs %s: %v", rs.Name, err)
	}
	if scale.Spec.Replicas != *newRS.Spec.Replicas {
		t.Fatalf("Scale subresource for rs %s does not match .Spec.Replicas: expected %d, got %d", rs.Name, *newRS.Spec.Replicas, scale.Spec.Replicas)
	}

	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		scale, err := scaleClient.Get(kind, rs.Name)
		if err != nil {
			return err
		}
		scale.Spec.Replicas = replicas
		_, err = scaleClient.Update(kind, scale)
		return err
	}); err != nil {
		t.Fatalf("Failed to set .Spec.Replicas of scale subresource for rs %s: %v", rs.Name, err)
	}

	newRS, err = rsClient.Get(rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain rs %s: %v", rs.Name, err)
	}
	if *newRS.Spec.Replicas != replicas {
		t.Fatalf(".Spec.Replicas of rs %s does not match its scale subresource: expected %d, got %d", rs.Name, replicas, *newRS.Spec.Replicas)
	}
}

func TestAdoption(t *testing.T) {
	boolPtr := func(b bool) *bool { return &b }
	testCases := []struct {
		name                    string
		existingOwnerReferences func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference
		expectedOwnerReferences func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference
	}{
		{
			"pod refers rs as an owner, not a controller",
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet"}}
			},
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod doesn't have owner references",
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{}
			},
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod refers rs as a controller",
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true)}}
			},
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true)}}
			},
		},
		{
			"pod refers other rs as the controller, refers the rs as an owner",
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRS", APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true)},
					{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet"},
				}
			},
			func(rs *v1beta1.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRS", APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Controller: boolPtr(true)},
					{UID: rs.UID, Name: rs.Name, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet"},
				}
			},
		},
	}
	for i, tc := range testCases {
		func() {
			s, closeFn, rm, informers, clientSet := rmSetup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace(fmt.Sprintf("rs-adoption-%d", i), s, t)
			defer framework.DeleteTestingNamespace(ns, s, t)

			rsClient := clientSet.Extensions().ReplicaSets(ns.Name)
			podClient := clientSet.CoreV1().Pods(ns.Name)
			const rsName = "rs"
			rs, err := rsClient.Create(newRS(rsName, ns.Name, 1))
			if err != nil {
				t.Fatalf("Failed to create replica set: %v", err)
			}
			podName := fmt.Sprintf("pod%d", i)
			pod := newMatchingPod(podName, ns.Name)
			pod.OwnerReferences = tc.existingOwnerReferences(rs)
			_, err = podClient.Create(pod)
			if err != nil {
				t.Fatalf("Failed to create Pod: %v", err)
			}

			stopCh := runControllerAndInformers(t, rm, informers, 1)
			defer close(stopCh)
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				updatedPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if e, a := tc.expectedOwnerReferences(rs), updatedPod.OwnerReferences; reflect.DeepEqual(e, a) {
					return true, nil
				} else {
					t.Logf("ownerReferences don't match, expect %v, got %v", e, a)
					return false, nil
				}
			}); err != nil {
				t.Fatalf("test %q failed: %v", tc.name, err)
			}
		}()
	}
}

// selectors are IMMUTABLE for all API versions except extensions/v1beta1
func TestRSSelectorImmutability(t *testing.T) {
	s, closeFn, clientSet := rmSimpleSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("rs-selector-immutability", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rs := newRS("rs", ns.Name, 0)
	createRSsPods(t, clientSet, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})

	// test to ensure extensions/v1beta1 selector is mutable
	newSelectorLabels := map[string]string{"changed_name_extensions_v1beta1": "changed_test_extensions_v1beta1"}
	rs.Spec.Selector.MatchLabels = newSelectorLabels
	rs.Spec.Template.Labels = newSelectorLabels
	replicaset, err := clientSet.ExtensionsV1beta1().ReplicaSets(ns.Name).Update(rs)
	if err != nil {
		t.Fatalf("failed to update extensions/v1beta1 replicaset %s: %v", replicaset.Name, err)
	}
	if !reflect.DeepEqual(replicaset.Spec.Selector.MatchLabels, newSelectorLabels) {
		t.Errorf("selector should be changed for extensions/v1beta1, expected: %v, got: %v", newSelectorLabels, replicaset.Spec.Selector.MatchLabels)
	}

	// test to ensure apps/v1beta2 selector is immutable
	rsV1beta2, err := clientSet.AppsV1beta2().ReplicaSets(ns.Name).Get(replicaset.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get apps/v1beta2 replicaset %s: %v", replicaset.Name, err)
	}
	newSelectorLabels = map[string]string{"changed_name_apps_v1beta2": "changed_test_apps_v1beta2"}
	rsV1beta2.Spec.Selector.MatchLabels = newSelectorLabels
	rsV1beta2.Spec.Template.Labels = newSelectorLabels
	_, err = clientSet.AppsV1beta2().ReplicaSets(ns.Name).Update(rsV1beta2)
	if err == nil {
		t.Fatalf("failed to provide validation error when changing immutable selector when updating apps/v1beta2 replicaset %s", rsV1beta2.Name)
	}
	expectedErrType := "Invalid value"
	expectedErrDetail := "field is immutable"
	if !strings.Contains(err.Error(), expectedErrType) || !strings.Contains(err.Error(), expectedErrDetail) {
		t.Errorf("error message does not match, expected type: %s, expected detail: %s, got: %s", expectedErrType, expectedErrDetail, err.Error())
	}
}

func TestSpecReplicasChange(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-spec-replicas-change", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Update .Spec.Replicas and verify .Status.Replicas is changed accordingly
	scaleRS(t, c, rs, 3)
	scaleRS(t, c, rs, 0)
	scaleRS(t, c, rs, 2)

	// Add a template annotation change to test RS's status does update
	// without .Spec.Replicas change
	rsClient := c.Extensions().ReplicaSets(ns.Name)
	var oldGeneration int64
	newRS := updateRS(t, rsClient, rs.Name, func(rs *v1beta1.ReplicaSet) {
		oldGeneration = rs.Generation
		rs.Spec.Template.Annotations = map[string]string{"test": "annotation"}
	})
	savedGeneration := newRS.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("Failed to verify .Generation has incremented for rs %s", rs.Name)
	}

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newRS.Status.ObservedGeneration >= savedGeneration, nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.ObservedGeneration has incremented for rs %s: %v", rs.Name, err)
	}
}

func TestDeletingAndFailedPods(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-deleting-and-failed-pods", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Verify RS creates 2 pods
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
	if err := c.CoreV1().Pods(ns.Name).Delete(deletingPod.Name, &metav1.DeleteOptions{}); err != nil {
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

func TestOverlappingRSs(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-overlapping-rss", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	// Create 2 RSs with identical selectors
	for i := 0; i < 2; i++ {
		// One RS has 1 replica, and another has 2 replicas
		rs := newRS(fmt.Sprintf("rs-%d", i+1), ns.Name, i+1)
		rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
		waitRSStable(t, c, rss[0])
	}

	// Expect 3 total Pods to be created
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 3 {
		t.Errorf("len(pods) = %d, want 3", len(pods.Items))
	}

	// Expect both RSs have .status.replicas = .spec.replicas
	for i := 0; i < 2; i++ {
		newRS, err := c.ExtensionsV1beta1().ReplicaSets(ns.Name).Get(fmt.Sprintf("rs-%d", i+1), metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to obtain rs rs-%d: %v", i+1, err)
		}
		if newRS.Status.Replicas != *newRS.Spec.Replicas {
			t.Fatalf(".Status.Replicas %d is not equal to .Spec.Replicas %d", newRS.Status.Replicas, *newRS.Spec.Replicas)
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

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Orphaning: RS should remove OwnerReference from a pod when the pod's labels change to not match its labels
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
		newPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		pod = newPod
		return metav1.GetControllerOf(newPod) == nil, nil
	}); err != nil {
		t.Fatalf("Failed to verify ControllerRef for the pod %s is nil: %v", pod.Name, err)
	}

	// Adoption: RS should add ControllerRef to a pod when the pod's labels change to match its labels
	updatePod(t, podClient, pod.Name, func(pod *v1.Pod) {
		pod.Labels = labelMap()
	})
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			// If the pod is not found, it means the RS picks the pod for deletion (it is extra)
			// Verify there is only one pod in namespace and it has ControllerRef to the RS
			if errors.IsNotFound(err) {
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
	// Verify the pod has a ControllerRef to the RS
	// Do nothing if the pod is nil (i.e., has been picked for deletion)
	if pod != nil {
		controllerRef := metav1.GetControllerOf(pod)
		if controllerRef.UID != rs.UID {
			t.Fatalf("RS owner of the pod %s has a different UID: Expected %v, got %v", pod.Name, rs.UID, controllerRef.UID)
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

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 1 {
		t.Fatalf("len(pods) = %d, want 1", len(pods.Items))
	}
	pod := &pods.Items[0]
	var falseVar = false

	// When the only OwnerReference of the pod points to another type of API object such as statefulset
	// with Controller=false, the RS should add a second OwnerReference (ControllerRef) pointing to itself
	// with Controller=true
	ownerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "apps/v1beta1", Kind: "StatefulSet", Name: rs.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rs, 2)

	// When the only OwnerReference of the pod points to the RS, but Controller=false
	ownerReference = metav1.OwnerReference{UID: rs.UID, APIVersion: "extensions/v1beta1", Kind: "ReplicaSet", Name: rs.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rs, 1)
}

func TestReadyAndAvailableReplicas(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-ready-and-available-replicas", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rs := newRS("rs", ns.Name, 3)
	rs.Spec.MinReadySeconds = 3600
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// First verify no pod is available
	if rs.Status.AvailableReplicas != 0 {
		t.Fatalf("Unexpected .Status.AvailableReplicas: Expected 0, saw %d", rs.Status.AvailableReplicas)
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

	rsClient := c.Extensions().ReplicaSets(ns.Name)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		// Verify 3 pods exist, 2 pods are Ready, and 1 pod is Available
		return newRS.Status.Replicas == 3 && newRS.Status.ReadyReplicas == 2 && newRS.Status.AvailableReplicas == 1, nil
	}); err != nil {
		t.Fatalf("Failed to verify number of Replicas, ReadyReplicas and AvailableReplicas of rs %s to be as expected: %v", rs.Name, err)
	}
}

func TestRSScaleSubresource(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-rs-scale-subresource", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Use scale subresource to scale up .Spec.Replicas to 3
	testScalingUsingScaleSubresource(t, c, rs, 3)
	// Use the scale subresource to scale down .Spec.Replicas to 0
	testScalingUsingScaleSubresource(t, c, rs, 0)
}

func TestExtraPodsAdoptionAndDeletion(t *testing.T) {
	s, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-extra-pods-adoption-and-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	rs := newRS("rs", ns.Name, 2)
	// Create 3 pods, RS should adopt only 2 of them
	podList := []*v1.Pod{}
	for i := 0; i < 3; i++ {
		pod := newMatchingPod(fmt.Sprintf("pod-%d", i+1), ns.Name)
		pod.Labels = labelMap()
		podList = append(podList, pod)
	}
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, podList)
	rs = rss[0]
	stopCh := runControllerAndInformers(t, rm, informers, 3)
	defer close(stopCh)
	waitRSStable(t, c, rs)

	// Verify the extra pod is deleted eventually by determining whether number of
	// all pods within namespace matches .spec.replicas of the RS (2 in this case)
	podClient := c.CoreV1().Pods(ns.Name)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		// All pods have labelMap as their labels
		pods := getPods(t, podClient, labelMap())
		return int32(len(pods.Items)) == *rs.Spec.Replicas, nil
	}); err != nil {
		t.Fatalf("Failed to verify number of all pods within current namespace matches .spec.replicas of rs %s: %v", rs.Name, err)
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
	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Change RS's template labels to have extra labels, but not its selector
	rsClient := c.Extensions().ReplicaSets(ns.Name)
	updateRS(t, rsClient, rs.Name, func(rs *v1beta1.ReplicaSet) {
		rs.Spec.Template.Labels = extraLabelMap
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
		newRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return (newRS.Status.Replicas == 2 && newRS.Status.FullyLabeledReplicas == 1), nil
	}); err != nil {
		t.Fatalf("Failed to verify only one pod is fully labeled: %v", err)
	}
}
