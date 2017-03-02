// +build integration,!no-etcd

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
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/test/integration/framework"
)

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
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
				MatchLabels: testLabels(),
			},
			Replicas: &replicasCopy,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: testLabels(),
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
			Labels:    testLabels(),
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
	podClient := clientSet.Core().Pods(namespace)
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

func rmSetup(t *testing.T) (*httptest.Server, *replicaset.ReplicaSetController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s := framework.RunAMaster(masterConfig)

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
	return s, rm, informers, clientSet
}

// wait for the podInformer to observe the pods. Call this function before
// running the RS controller to prevent the rc manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		if len(objects) == podNum {
			return true, nil
		} else {
			return false, nil
		}
	}); err != nil {
		t.Fatal(err)
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
		s, rm, informers, clientSet := rmSetup(t)
		podInformer := informers.Core().V1().Pods().Informer()
		ns := framework.CreateTestingNamespace(fmt.Sprintf("rs-adoption-%d", i), s, t)
		defer framework.DeleteTestingNamespace(ns, s, t)

		rsClient := clientSet.Extensions().ReplicaSets(ns.Name)
		podClient := clientSet.Core().Pods(ns.Name)
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

		stopCh := make(chan struct{})
		informers.Start(stopCh)
		waitToObservePods(t, podInformer, 1)
		go rm.Run(5, stopCh)
		if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
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
		close(stopCh)
	}
}

func createRSsPods(t *testing.T, clientSet clientset.Interface, rss []*v1beta1.ReplicaSet, pods []*v1.Pod, ns string) {
	rsClient := clientSet.Extensions().ReplicaSets(ns)
	podClient := clientSet.Core().Pods(ns)
	for _, rs := range rss {
		if _, err := rsClient.Create(rs); err != nil {
			t.Fatalf("Failed to create replica set %s: %v", rs.Name, err)
		}
	}
	for _, pod := range pods {
		if _, err := podClient.Create(pod); err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
	}
}

func waitRSStable(t *testing.T, clientSet clientset.Interface, rs *v1beta1.ReplicaSet, ns string) {
	rsClient := clientSet.Extensions().ReplicaSets(ns)
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		updatedRS, err := rsClient.Get(rs.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if updatedRS.Status.Replicas != *rs.Spec.Replicas {
			return false, nil
		} else {
			return true, nil
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func TestUpdateSelectorToAdopt(t *testing.T) {
	// We have pod1, pod2 and rs. rs.spec.replicas=1. At first rs.Selector
	// matches pod1 only; change the selector to match pod2 as well. Verify
	// there is only one pod left.
	s, rm, informers, clientSet := rmSetup(t)
	ns := framework.CreateTestingNamespace("rs-update-selector-to-adopt", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rs := newRS("rs", ns.Name, 1)
	// let rs's selector only match pod1
	rs.Spec.Selector.MatchLabels["uniqueKey"] = "1"
	rs.Spec.Template.Labels["uniqueKey"] = "1"
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRSsPods(t, clientSet, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{pod1, pod2}, ns.Name)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	waitRSStable(t, clientSet, rs, ns.Name)

	// change the rs's selector to match both pods
	patch := `{"spec":{"selector":{"matchLabels": {"uniqueKey":null}}}}`
	rsClient := clientSet.Extensions().ReplicaSets(ns.Name)
	rs, err := rsClient.Patch(rs.Name, types.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch replica set: %v", err)
	}
	t.Logf("patched rs = %#v", rs)
	// wait for the rs select both pods and delete one of them
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 1)
	}); err != nil {
		t.Fatal(err)
	}
	close(stopCh)
}

func TestUpdateSelectorToRemoveControllerRef(t *testing.T) {
	// We have pod1, pod2 and rs. rs.spec.replicas=2. At first rs.Selector
	// matches pod1 and pod2; change the selector to match only pod1. Verify
	// that rs creates one more pod, so there are 3 pods. Also verify that
	// pod2's controllerRef is cleared.
	s, rm, informers, clientSet := rmSetup(t)
	podInformer := informers.Core().V1().Pods().Informer()
	ns := framework.CreateTestingNamespace("rs-update-selector-to-remove-controllerref", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rs := newRS("rs", ns.Name, 2)
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRSsPods(t, clientSet, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{pod1, pod2}, ns.Name)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	waitToObservePods(t, podInformer, 2)
	go rm.Run(5, stopCh)
	waitRSStable(t, clientSet, rs, ns.Name)

	// change the rs's selector to match both pods
	patch := `{"spec":{"selector":{"matchLabels": {"uniqueKey":"1"}},"template":{"metadata":{"labels":{"uniqueKey":"1"}}}}}`
	rsClient := clientSet.Extensions().ReplicaSets(ns.Name)
	rs, err := rsClient.Patch(rs.Name, types.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch replica set: %v", err)
	}
	t.Logf("patched rs = %#v", rs)
	// wait for the rs to create one more pod
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 3)
	}); err != nil {
		t.Fatal(err)
	}
	podClient := clientSet.Core().Pods(ns.Name)
	pod2, err = podClient.Get(pod2.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod2: %v", err)
	}
	if len(pod2.OwnerReferences) != 0 {
		t.Fatalf("ownerReferences of pod2 is not cleared, got %#v", pod2.OwnerReferences)
	}
	close(stopCh)
}

func TestUpdateLabelToRemoveControllerRef(t *testing.T) {
	// We have pod1, pod2 and rs. rs.spec.replicas=2. At first rs.Selector
	// matches pod1 and pod2; change pod2's labels to non-matching. Verify
	// that rs creates one more pod, so there are 3 pods. Also verify that
	// pod2's controllerRef is cleared.
	s, rm, informers, clientSet := rmSetup(t)
	ns := framework.CreateTestingNamespace("rs-update-label-to-remove-controllerref", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rs := newRS("rs", ns.Name, 2)
	pod1 := newMatchingPod("pod1", ns.Name)
	pod2 := newMatchingPod("pod2", ns.Name)
	createRSsPods(t, clientSet, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{pod1, pod2}, ns.Name)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	waitRSStable(t, clientSet, rs, ns.Name)

	// change the rs's selector to match both pods
	patch := `{"metadata":{"labels":{"name":null}}}`
	podClient := clientSet.Core().Pods(ns.Name)
	pod2, err := podClient.Patch(pod2.Name, types.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch pod2: %v", err)
	}
	t.Logf("patched pod2 = %#v", pod2)
	// wait for the rs to create one more pod
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 3)
	}); err != nil {
		t.Fatal(err)
	}
	pod2, err = podClient.Get(pod2.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get pod2: %v", err)
	}
	if len(pod2.OwnerReferences) != 0 {
		t.Fatalf("ownerReferences of pod2 is not cleared, got %#v", pod2.OwnerReferences)
	}
	close(stopCh)
}

func TestUpdateLabelToBeAdopted(t *testing.T) {
	// We have pod1, pod2 and rs. rs.spec.replicas=1. At first rs.Selector
	// matches pod1 only; change pod2's labels to be matching. Verify the RS
	// controller adopts pod2 and delete one of them, so there is only 1 pod
	// left.
	s, rm, informers, clientSet := rmSetup(t)
	ns := framework.CreateTestingNamespace("rs-update-label-to-be-adopted", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rs := newRS("rs", ns.Name, 1)
	// let rs's selector only matches pod1
	rs.Spec.Selector.MatchLabels["uniqueKey"] = "1"
	rs.Spec.Template.Labels["uniqueKey"] = "1"
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRSsPods(t, clientSet, []*v1beta1.ReplicaSet{rs}, []*v1.Pod{pod1, pod2}, ns.Name)

	stopCh := make(chan struct{})
	informers.Start(stopCh)
	go rm.Run(5, stopCh)
	waitRSStable(t, clientSet, rs, ns.Name)

	// change the rs's selector to match both pods
	patch := `{"metadata":{"labels":{"uniqueKey":"1"}}}`
	podClient := clientSet.Core().Pods(ns.Name)
	pod2, err := podClient.Patch(pod2.Name, types.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch pod2: %v", err)
	}
	t.Logf("patched pod2 = %#v", pod2)
	// wait for the rs to select both pods and delete one of them
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 1)
	}); err != nil {
		t.Fatal(err)
	}
	close(stopCh)
}
