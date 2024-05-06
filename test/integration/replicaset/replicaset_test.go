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
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	appsclient "k8s.io/client-go/kubernetes/typed/apps/v1"
	typedv1 "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	testutil "k8s.io/kubernetes/test/utils"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const (
	interval = 100 * time.Millisecond
	timeout  = 60 * time.Second
)

func labelMap() map[string]string {
	return map[string]string{"foo": "bar"}
}

func newRS(name, namespace string, replicas int) *apps.ReplicaSet {
	replicasCopy := int32(replicas)
	return &apps.ReplicaSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicaSet",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: apps.ReplicaSetSpec{
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

func rmSetup(t *testing.T) (context.Context, kubeapiservertesting.TearDownFunc, *replicaset.ReplicaSetController, informers.SharedInformerFactory, clientset.Interface) {
	tCtx := ktesting.Init(t)
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "rs-informers")), resyncPeriod)

	rm := replicaset.NewReplicaSetController(
		tCtx,
		informers.Apps().V1().ReplicaSets(),
		informers.Core().V1().Pods(),
		clientset.NewForConfigOrDie(restclient.AddUserAgent(config, "replicaset-controller")),
		replicaset.BurstReplicas,
	)

	newTeardown := func() {
		tCtx.Cancel("tearing down controller")
		server.TearDownFn()
	}

	return tCtx, newTeardown, rm, informers, clientSet
}

func rmSimpleSetup(t *testing.T) (kubeapiservertesting.TearDownFunc, clientset.Interface) {
	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())

	config := restclient.CopyConfig(server.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	return server.TearDownFn, clientSet
}

// Run RS controller and informers
func runControllerAndInformers(t *testing.T, rm *replicaset.ReplicaSetController, informers informers.SharedInformerFactory, podNum int) func() {
	ctx, cancelFn := context.WithCancel(context.Background())
	informers.Start(ctx.Done())
	waitToObservePods(t, informers.Core().V1().Pods().Informer(), podNum)
	go rm.Run(ctx, 5)
	return cancelFn
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

func createRSsPods(t *testing.T, clientSet clientset.Interface, rss []*apps.ReplicaSet, pods []*v1.Pod) ([]*apps.ReplicaSet, []*v1.Pod) {
	var createdRSs []*apps.ReplicaSet
	var createdPods []*v1.Pod
	for _, rs := range rss {
		createdRS, err := clientSet.AppsV1().ReplicaSets(rs.Namespace).Create(context.TODO(), rs, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create replica set %s: %v", rs.Name, err)
		}
		createdRSs = append(createdRSs, createdRS)
	}
	for _, pod := range pods {
		createdPod, err := clientSet.CoreV1().Pods(pod.Namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
		createdPods = append(createdPods, createdPod)
	}

	return createdRSs, createdPods
}

// Verify .Status.Replicas is equal to .Spec.Replicas
func waitRSStable(t *testing.T, clientSet clientset.Interface, rs *apps.ReplicaSet) {
	if err := testutil.WaitRSStable(t, clientSet, rs, interval, timeout); err != nil {
		t.Fatal(err)
	}
}

// Update .Spec.Replicas to replicas and verify .Status.Replicas is changed accordingly
func scaleRS(t *testing.T, c clientset.Interface, rs *apps.ReplicaSet, replicas int32) {
	rsClient := c.AppsV1().ReplicaSets(rs.Namespace)
	rs = updateRS(t, rsClient, rs.Name, func(rs *apps.ReplicaSet) {
		*rs.Spec.Replicas = replicas
	})
	waitRSStable(t, c, rs)
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

func updatePodStatus(t *testing.T, podClient typedv1.PodInterface, podName string, updateStatusFunc func(*v1.Pod)) {
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newPod, err := podClient.Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateStatusFunc(newPod)
		_, err = podClient.UpdateStatus(context.TODO(), newPod, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update status of pod %s: %v", podName, err)
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

func updateRS(t *testing.T, rsClient appsclient.ReplicaSetInterface, rsName string, updateFunc func(*apps.ReplicaSet)) *apps.ReplicaSet {
	var rs *apps.ReplicaSet
	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		newRS, err := rsClient.Get(context.TODO(), rsName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		updateFunc(newRS)
		rs, err = rsClient.Update(context.TODO(), newRS, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to update rs %s: %v", rsName, err)
	}
	return rs
}

// Verify ControllerRef of a RS pod that has incorrect attributes is automatically patched by the RS
func testPodControllerRefPatch(t *testing.T, c clientset.Interface, pod *v1.Pod, ownerReference *metav1.OwnerReference, rs *apps.ReplicaSet, expectedOwnerReferenceNum int) {
	ns := rs.Namespace
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
		t.Fatalf("failed to mark all ReplicaSet pods to ready: %v", err)
	}
}

func testScalingUsingScaleSubresource(t *testing.T, c clientset.Interface, rs *apps.ReplicaSet, replicas int32) {
	ns := rs.Namespace
	rsClient := c.AppsV1().ReplicaSets(ns)
	newRS, err := rsClient.Get(context.TODO(), rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain rs %s: %v", rs.Name, err)
	}
	scale, err := c.AppsV1().ReplicaSets(ns).GetScale(context.TODO(), rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to obtain scale subresource for rs %s: %v", rs.Name, err)
	}
	if scale.Spec.Replicas != *newRS.Spec.Replicas {
		t.Fatalf("Scale subresource for rs %s does not match .Spec.Replicas: expected %d, got %d", rs.Name, *newRS.Spec.Replicas, scale.Spec.Replicas)
	}

	if err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		scale, err := c.AppsV1().ReplicaSets(ns).GetScale(context.TODO(), rs.Name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		scale.Spec.Replicas = replicas
		_, err = c.AppsV1().ReplicaSets(ns).UpdateScale(context.TODO(), rs.Name, scale, metav1.UpdateOptions{})
		return err
	}); err != nil {
		t.Fatalf("Failed to set .Spec.Replicas of scale subresource for rs %s: %v", rs.Name, err)
	}

	newRS, err = rsClient.Get(context.TODO(), rs.Name, metav1.GetOptions{})
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
		existingOwnerReferences func(rs *apps.ReplicaSet) []metav1.OwnerReference
		expectedOwnerReferences func(rs *apps.ReplicaSet) []metav1.OwnerReference
	}{
		{
			"pod refers rs as an owner, not a controller",
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet"}}
			},
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod doesn't have owner references",
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{}
			},
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod refers rs as a controller",
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true)}}
			},
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true)}}
			},
		},
		{
			"pod refers other rs as the controller, refers the rs as an owner",
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRS", APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true)},
					{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet"},
				}
			},
			func(rs *apps.ReplicaSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRS", APIVersion: "apps/v1", Kind: "ReplicaSet", Controller: boolPtr(true)},
					{UID: rs.UID, Name: rs.Name, APIVersion: "apps/v1", Kind: "ReplicaSet"},
				}
			},
		},
	}
	for i, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tCtx, closeFn, rm, informers, clientSet := rmSetup(t)
			defer closeFn()

			ns := framework.CreateNamespaceOrDie(clientSet, fmt.Sprintf("rs-adoption-%d", i), t)
			defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

			rsClient := clientSet.AppsV1().ReplicaSets(ns.Name)
			podClient := clientSet.CoreV1().Pods(ns.Name)
			const rsName = "rs"
			rs, err := rsClient.Create(tCtx, newRS(rsName, ns.Name, 1), metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create replica set: %v", err)
			}
			podName := fmt.Sprintf("pod%d", i)
			pod := newMatchingPod(podName, ns.Name)
			pod.OwnerReferences = tc.existingOwnerReferences(rs)
			_, err = podClient.Create(tCtx, pod, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create Pod: %v", err)
			}

			stopControllers := runControllerAndInformers(t, rm, informers, 1)
			defer stopControllers()
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				updatedPod, err := podClient.Get(tCtx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false, err
				}

				e, a := tc.expectedOwnerReferences(rs), updatedPod.OwnerReferences
				if reflect.DeepEqual(e, a) {
					return true, nil
				}

				t.Logf("ownerReferences don't match, expect %v, got %v", e, a)
				return false, nil
			}); err != nil {
				t.Fatalf("test %q failed: %v", tc.name, err)
			}
		})
	}
}

// selectors are IMMUTABLE for all API versions except extensions/v1beta1
func TestRSSelectorImmutability(t *testing.T) {
	closeFn, clientSet := rmSimpleSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(clientSet, "rs-selector-immutability", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)
	rs := newRS("rs", ns.Name, 0)
	createRSsPods(t, clientSet, []*apps.ReplicaSet{rs}, []*v1.Pod{})

	// test to ensure apps/v1 selector is immutable
	rsV1, err := clientSet.AppsV1().ReplicaSets(ns.Name).Get(context.TODO(), rs.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to get apps/v1 replicaset %s: %v", rs.Name, err)
	}
	newSelectorLabels := map[string]string{"changed_name_apps_v1": "changed_test_apps_v1"}
	rsV1.Spec.Selector.MatchLabels = newSelectorLabels
	rsV1.Spec.Template.Labels = newSelectorLabels
	_, err = clientSet.AppsV1().ReplicaSets(ns.Name).Update(context.TODO(), rsV1, metav1.UpdateOptions{})
	if err == nil {
		t.Fatalf("failed to provide validation error when changing immutable selector when updating apps/v1 replicaset %s", rsV1.Name)
	}
	expectedErrType := "Invalid value"
	expectedErrDetail := "field is immutable"
	if !strings.Contains(err.Error(), expectedErrType) || !strings.Contains(err.Error(), expectedErrDetail) {
		t.Errorf("error message does not match, expected type: %s, expected detail: %s, got: %s", expectedErrType, expectedErrDetail, err.Error())
	}
}

func TestSpecReplicasChange(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-spec-replicas-change", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Update .Spec.Replicas and verify .Status.Replicas is changed accordingly
	scaleRS(t, c, rs, 3)
	scaleRS(t, c, rs, 0)
	scaleRS(t, c, rs, 2)

	// Add a template annotation change to test RS's status does update
	// without .Spec.Replicas change
	rsClient := c.AppsV1().ReplicaSets(ns.Name)
	var oldGeneration int64
	newRS := updateRS(t, rsClient, rs.Name, func(rs *apps.ReplicaSet) {
		oldGeneration = rs.Generation
		rs.Spec.Template.Annotations = map[string]string{"test": "annotation"}
	})
	savedGeneration := newRS.Generation
	if savedGeneration == oldGeneration {
		t.Fatalf("Failed to verify .Generation has incremented for rs %s", rs.Name)
	}

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(tCtx, rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return newRS.Status.ObservedGeneration >= savedGeneration, nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.ObservedGeneration has incremented for rs %s: %v", rs.Name, err)
	}
}

func TestDeletingAndFailedPods(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()

	ns := framework.CreateNamespaceOrDie(c, "test-deleting-and-failed-pods", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
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
	if err := c.CoreV1().Pods(ns.Name).Delete(tCtx, deletingPod.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatalf("Error deleting pod %s: %v", deletingPod.Name, err)
	}

	// Set second pod as failed pod
	failedPod := &pods.Items[1]
	updatePodStatus(t, podClient, failedPod.Name, func(pod *v1.Pod) {
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

func TestPodDeletionCost(t *testing.T) {
	tests := []struct {
		name              string
		costs             []string
		restarts          []int32
		enabled           bool
		remainingPodIndex int
	}{
		{
			name:              "enabled-with-different-costs",
			costs:             []string{"1000", "100"},
			restarts:          []int32{5, 0},
			enabled:           true,
			remainingPodIndex: 0,
		},
		{
			name:              "enabled-with-same-costs",
			costs:             []string{"100", "100"},
			restarts:          []int32{5, 0},
			enabled:           true,
			remainingPodIndex: 1,
		},
		{
			name:              "enabled-with-no-costs",
			restarts:          []int32{5, 0},
			enabled:           true,
			remainingPodIndex: 1,
		},
		{
			name:              "disabled-with-different-costs",
			costs:             []string{"1000", "100"},
			restarts:          []int32{5, 0},
			enabled:           false,
			remainingPodIndex: 1,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDeletionCost, tc.enabled)
			_, closeFn, rm, informers, c := rmSetup(t)
			defer closeFn()
			ns := framework.CreateNamespaceOrDie(c, tc.name, t)
			defer framework.DeleteNamespaceOrDie(c, ns, t)
			stopControllers := runControllerAndInformers(t, rm, informers, 0)
			defer stopControllers()

			rs := newRS("rs", ns.Name, 2)
			rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
			rs = rss[0]
			waitRSStable(t, c, rs)

			// Verify RS creates 2 pods.
			podClient := c.CoreV1().Pods(ns.Name)
			pods := getPods(t, podClient, labelMap())
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				pods = getPods(t, podClient, labelMap())
				return len(pods.Items) == 2, nil
			}); err != nil {
				t.Fatalf("Failed to verify replicaset has 2 pods: %v", err)
			}

			// Set a higher deletion cost to the pod that is supposed to remain after scale down.
			remainingPodUID := pods.Items[tc.remainingPodIndex].UID
			for i := range pods.Items {
				podName := pods.Items[i].Name
				if len(tc.costs) != 0 {
					updatePod(t, podClient, podName, func(pod *v1.Pod) {
						pod.Annotations = map[string]string{core.PodDeletionCost: tc.costs[i]}
					})
				}
				updatePodStatus(t, podClient, podName, func(pod *v1.Pod) {
					pod.Status.ContainerStatuses = []v1.ContainerStatus{{RestartCount: tc.restarts[i]}}
				})
			}

			// Change RS's number of replics to 1
			rsClient := c.AppsV1().ReplicaSets(ns.Name)
			updateRS(t, rsClient, rs.Name, func(rs *apps.ReplicaSet) {
				rs.Spec.Replicas = ptr.To[int32](1)
			})

			// Poll until ReplicaSet is downscaled to 1.
			if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
				pods = getPods(t, podClient, labelMap())
				return len(pods.Items) == 1, nil
			}); err != nil {
				t.Fatalf("Failed to downscale replicaset to 1 pod: %v", err)
			}

			if pods.Items[0].UID != remainingPodUID {
				t.Errorf("expected remaining Pod UID %v, got UID %v with container statues %v",
					remainingPodUID, pods.Items[0].UID, pods.Items[0].Status.ContainerStatuses)
			}
		})
	}
}

func TestOverlappingRSs(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-overlapping-rss", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	// Create 2 RSs with identical selectors
	for i := 0; i < 2; i++ {
		// One RS has 1 replica, and another has 2 replicas
		rs := newRS(fmt.Sprintf("rs-%d", i+1), ns.Name, i+1)
		rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
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
		newRS, err := c.AppsV1().ReplicaSets(ns.Name).Get(tCtx, fmt.Sprintf("rs-%d", i+1), metav1.GetOptions{})
		if err != nil {
			t.Fatalf("failed to obtain rs rs-%d: %v", i+1, err)
		}
		if newRS.Status.Replicas != *newRS.Spec.Replicas {
			t.Fatalf(".Status.Replicas %d is not equal to .Spec.Replicas %d", newRS.Status.Replicas, *newRS.Spec.Replicas)
		}
	}
}

func TestPodOrphaningAndAdoptionWhenLabelsChange(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-pod-orphaning-and-adoption-when-labels-change", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
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
		newPod, err := podClient.Get(tCtx, pod.Name, metav1.GetOptions{})
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
		newPod, err := podClient.Get(tCtx, pod.Name, metav1.GetOptions{})
		if err != nil {
			// If the pod is not found, it means the RS picks the pod for deletion (it is extra)
			// Verify there is only one pod in namespace and it has ControllerRef to the RS
			if !apierrors.IsNotFound(err) {
				return false, err
			}

			pods := getPods(t, podClient, labelMap())
			if len(pods.Items) != 1 {
				return false, fmt.Errorf("Expected 1 pod in current namespace, got %d", len(pods.Items))
			}
			// Set the pod accordingly
			pod = &pods.Items[0]
			return true, nil
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
	_, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-general-pod-adoption", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
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
	ownerReference := metav1.OwnerReference{UID: uuid.NewUUID(), APIVersion: "apps/v1", Kind: "StatefulSet", Name: rs.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rs, 2)

	// When the only OwnerReference of the pod points to the RS, but Controller=false
	ownerReference = metav1.OwnerReference{UID: rs.UID, APIVersion: "apps/v1", Kind: "ReplicaSet", Name: rs.Name, Controller: &falseVar}
	testPodControllerRefPatch(t, c, pod, &ownerReference, rs, 1)
}

func TestReadyAndAvailableReplicas(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-ready-and-available-replicas", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 3)
	rs.Spec.MinReadySeconds = 3600
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
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

	rsClient := c.AppsV1().ReplicaSets(ns.Name)
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(tCtx, rs.Name, metav1.GetOptions{})
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
	_, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-rs-scale-subresource", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 1)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Use scale subresource to scale up .Spec.Replicas to 3
	testScalingUsingScaleSubresource(t, c, rs, 3)
	// Use the scale subresource to scale down .Spec.Replicas to 0
	testScalingUsingScaleSubresource(t, c, rs, 0)
}

func TestExtraPodsAdoptionAndDeletion(t *testing.T) {
	_, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-extra-pods-adoption-and-deletion", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)

	rs := newRS("rs", ns.Name, 2)
	// Create 3 pods, RS should adopt only 2 of them
	podList := []*v1.Pod{}
	for i := 0; i < 3; i++ {
		pod := newMatchingPod(fmt.Sprintf("pod-%d", i+1), ns.Name)
		pod.Labels = labelMap()
		podList = append(podList, pod)
	}
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, podList)
	rs = rss[0]
	stopControllers := runControllerAndInformers(t, rm, informers, 3)
	defer stopControllers()
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
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-fully-labeled-replicas", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	extraLabelMap := map[string]string{"foo": "bar", "extraKey": "extraValue"}
	rs := newRS("rs", ns.Name, 2)
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Change RS's template labels to have extra labels, but not its selector
	rsClient := c.AppsV1().ReplicaSets(ns.Name)
	updateRS(t, rsClient, rs.Name, func(rs *apps.ReplicaSet) {
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
		newRS, err := rsClient.Get(tCtx, rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return (newRS.Status.Replicas == 2 && newRS.Status.FullyLabeledReplicas == 1), nil
	}); err != nil {
		t.Fatalf("Failed to verify only one pod is fully labeled: %v", err)
	}
}

func TestReplicaSetsAppsV1DefaultGCPolicy(t *testing.T) {
	tCtx, closeFn, rm, informers, c := rmSetup(t)
	defer closeFn()
	ns := framework.CreateNamespaceOrDie(c, "test-default-gc-v1", t)
	defer framework.DeleteNamespaceOrDie(c, ns, t)
	stopControllers := runControllerAndInformers(t, rm, informers, 0)
	defer stopControllers()

	rs := newRS("rs", ns.Name, 2)
	fakeFinalizer := "kube.io/dummy-finalizer"
	rs.Finalizers = []string{fakeFinalizer}
	rss, _ := createRSsPods(t, c, []*apps.ReplicaSet{rs}, []*v1.Pod{})
	rs = rss[0]
	waitRSStable(t, c, rs)

	// Verify RS creates 2 pods
	podClient := c.CoreV1().Pods(ns.Name)
	pods := getPods(t, podClient, labelMap())
	if len(pods.Items) != 2 {
		t.Fatalf("len(pods) = %d, want 2", len(pods.Items))
	}

	rsClient := c.AppsV1().ReplicaSets(ns.Name)
	err := rsClient.Delete(tCtx, rs.Name, metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("Failed to delete rs: %v", err)
	}

	// Verify no new finalizer has been added
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newRS, err := rsClient.Get(tCtx, rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if newRS.DeletionTimestamp == nil {
			return false, nil
		}
		if got, want := newRS.Finalizers, []string{fakeFinalizer}; !reflect.DeepEqual(got, want) {
			return false, fmt.Errorf("got finalizers: %+v; want: %+v", got, want)
		}
		return true, nil
	}); err != nil {
		t.Fatalf("Failed to verify the finalizer: %v", err)
	}

	updateRS(t, c.AppsV1().ReplicaSets(ns.Name), rs.Name, func(rs *apps.ReplicaSet) {
		var finalizers []string
		// remove fakeFinalizer
		for _, finalizer := range rs.Finalizers {
			if finalizer != fakeFinalizer {
				finalizers = append(finalizers, finalizer)
			}
		}
		rs.Finalizers = finalizers
	})

	_ = rsClient.Delete(tCtx, rs.Name, metav1.DeleteOptions{})
}
