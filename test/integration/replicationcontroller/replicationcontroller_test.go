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

package replicationcontroller

import (
	"fmt"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/test/integration/framework"
)

func testLabels() map[string]string {
	return map[string]string{"name": "test"}
}

func newRC(name, namespace string, replicas int) *v1.ReplicationController {
	replicasCopy := int32(replicas)
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Selector: testLabels(),
			Replicas: &replicasCopy,
			Template: &v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
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
		ObjectMeta: v1.ObjectMeta{
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

// verifyRemainingObjects verifies if the number of the remaining replication
// controllers and pods are rcNum and podNum. It returns error if the
// communication with the API server fails.
func verifyRemainingObjects(t *testing.T, clientSet clientset.Interface, namespace string, rcNum, podNum int) (bool, error) {
	rcClient := clientSet.Core().ReplicationControllers(namespace)
	podClient := clientSet.Core().Pods(namespace)
	pods, err := podClient.List(v1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		t.Logf("expect %d pods, got %d pods", podNum, len(pods.Items))
	}
	rcs, err := rcClient.List(v1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != rcNum {
		ret = false
		t.Logf("expect %d RCs, got %d RCs", rcNum, len(rcs.Items))
	}
	return ret, nil
}

func rmSetup(t *testing.T, stopCh chan struct{}, enableGarbageCollector bool) (*httptest.Server, *replication.ReplicationManager, cache.SharedIndexInformer, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour

	informers := informers.NewSharedInformerFactory(clientSet, nil, resyncPeriod)
	podInformer := informers.Pods().Informer()
	rcInformer := informers.ReplicationControllers().Informer()
	rm := replication.NewReplicationManager(podInformer, rcInformer, clientSet, replication.BurstReplicas, 4096, enableGarbageCollector)
	informers.Start(stopCh)

	return s, rm, podInformer, clientSet
}

// wait for the podInformer to observe the pods. Call this function before
// running the RC manager to prevent the rc manager from creating new pods
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
	var trueVar = true
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
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar}}
			},
		},
		{
			"pod doesn't have owner references",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar}}
			},
		},
		{
			"pod refers rc as a controller",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar}}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar}}
			},
		},
		{
			"pod refers other rc as the controller, refers the rc as an owner",
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRC", APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar},
					{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController"},
				}
			},
			func(rc *v1.ReplicationController) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherRC", APIVersion: "v1", Kind: "ReplicationController", Controller: &trueVar},
					{UID: rc.UID, Name: rc.Name, APIVersion: "v1", Kind: "ReplicationController"},
				}
			},
		},
	}
	for i, tc := range testCases {
		stopCh := make(chan struct{})
		s, rm, podInformer, clientSet := rmSetup(t, stopCh, true)
		ns := framework.CreateTestingNamespace(fmt.Sprintf("adoption-%d", i), s, t)
		defer framework.DeleteTestingNamespace(ns, s, t)

		rcClient := clientSet.Core().ReplicationControllers(ns.Name)
		podClient := clientSet.Core().Pods(ns.Name)
		const rcName = "rc"
		rc, err := rcClient.Create(newRC(rcName, ns.Name, 1))
		if err != nil {
			t.Fatalf("Failed to create replication controller: %v", err)
		}
		podName := fmt.Sprintf("pod%d", i)
		pod := newMatchingPod(podName, ns.Name)
		pod.OwnerReferences = tc.existingOwnerReferences(rc)
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}

		go podInformer.Run(stopCh)
		waitToObservePods(t, podInformer, 1)
		go rm.Run(5, stopCh)
		if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
			updatedPod, err := podClient.Get(pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if e, a := tc.expectedOwnerReferences(rc), updatedPod.OwnerReferences; reflect.DeepEqual(e, a) {
				return true, nil
			} else {
				t.Logf("ownerReferences don't match, expect %v, got %v", e, a)
				return false, nil
			}
		}); err != nil {
			t.Fatal(err)
		}
		close(stopCh)
	}
}

func createRCsPods(t *testing.T, clientSet clientset.Interface, rcs []*v1.ReplicationController, pods []*v1.Pod, ns string) {
	rcClient := clientSet.Core().ReplicationControllers(ns)
	podClient := clientSet.Core().Pods(ns)
	for _, rc := range rcs {
		if _, err := rcClient.Create(rc); err != nil {
			t.Fatalf("Failed to create replication controller %s: %v", rc.Name, err)
		}
	}
	for _, pod := range pods {
		if _, err := podClient.Create(pod); err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
	}
}

func waitRCStable(t *testing.T, clientSet clientset.Interface, rc *v1.ReplicationController, ns string) {
	rcClient := clientSet.Core().ReplicationControllers(ns)
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		updatedRC, err := rcClient.Get(rc.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if updatedRC.Status.Replicas != *rc.Spec.Replicas {
			return false, nil
		} else {
			return true, nil
		}
	}); err != nil {
		t.Fatal(err)
	}
}

func TestUpdateSelectorToAdopt(t *testing.T) {
	// We have pod1, pod2 and rc. rc.spec.replicas=1. At first rc.Selector
	// matches pod1 only; change the selector to match pod2 as well. Verify
	// there is only one pod left.
	stopCh := make(chan struct{})
	s, rm, _, clientSet := rmSetup(t, stopCh, true)
	ns := framework.CreateTestingNamespace("update-selector-to-adopt", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rc := newRC("rc", ns.Name, 1)
	// let rc's selector only match pod1
	rc.Spec.Selector["uniqueKey"] = "1"
	rc.Spec.Template.Labels["uniqueKey"] = "1"
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRCsPods(t, clientSet, []*v1.ReplicationController{rc}, []*v1.Pod{pod1, pod2}, ns.Name)

	go rm.Run(5, stopCh)
	waitRCStable(t, clientSet, rc, ns.Name)

	// change the rc's selector to match both pods
	patch := `{"spec":{"selector":{"uniqueKey":null}}}`
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	rc, err := rcClient.Patch(rc.Name, api.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch replication controller: %v", err)
	}
	t.Logf("patched rc = %#v", rc)
	// wait for the rc select both pods and delete one of them
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 1)
	}); err != nil {
		t.Fatal(err)
	}
	close(stopCh)
}

func TestUpdateSelectorToRemoveControllerRef(t *testing.T) {
	// We have pod1, pod2 and rc. rc.spec.replicas=2. At first rc.Selector
	// matches pod1 and pod2; change the selector to match only pod1. Verify
	// that rc creates one more pod, so there are 3 pods. Also verify that
	// pod2's controllerRef is cleared.
	stopCh := make(chan struct{})
	s, rm, podInformer, clientSet := rmSetup(t, stopCh, true)
	ns := framework.CreateTestingNamespace("update-selector-to-remove-controllerref", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rc := newRC("rc", ns.Name, 2)
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRCsPods(t, clientSet, []*v1.ReplicationController{rc}, []*v1.Pod{pod1, pod2}, ns.Name)

	waitToObservePods(t, podInformer, 2)
	go rm.Run(5, stopCh)
	waitRCStable(t, clientSet, rc, ns.Name)

	// change the rc's selector to match both pods
	patch := `{"spec":{"selector":{"uniqueKey":"1"},"template":{"metadata":{"labels":{"uniqueKey":"1"}}}}}`
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	rc, err := rcClient.Patch(rc.Name, api.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch replication controller: %v", err)
	}
	t.Logf("patched rc = %#v", rc)
	// wait for the rc to create one more pod
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
	// We have pod1, pod2 and rc. rc.spec.replicas=2. At first rc.Selector
	// matches pod1 and pod2; change pod2's labels to non-matching. Verify
	// that rc creates one more pod, so there are 3 pods. Also verify that
	// pod2's controllerRef is cleared.
	stopCh := make(chan struct{})
	s, rm, _, clientSet := rmSetup(t, stopCh, true)
	ns := framework.CreateTestingNamespace("update-label-to-remove-controllerref", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rc := newRC("rc", ns.Name, 2)
	pod1 := newMatchingPod("pod1", ns.Name)
	pod2 := newMatchingPod("pod2", ns.Name)
	createRCsPods(t, clientSet, []*v1.ReplicationController{rc}, []*v1.Pod{pod1, pod2}, ns.Name)

	go rm.Run(5, stopCh)
	waitRCStable(t, clientSet, rc, ns.Name)

	// change the rc's selector to match both pods
	patch := `{"metadata":{"labels":{"name":null}}}`
	podClient := clientSet.Core().Pods(ns.Name)
	pod2, err := podClient.Patch(pod2.Name, api.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch pod2: %v", err)
	}
	t.Logf("patched pod2 = %#v", pod2)
	// wait for the rc to create one more pod
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
	// We have pod1, pod2 and rc. rc.spec.replicas=1. At first rc.Selector
	// matches pod1 only; change pod2's labels to be matching. Verify the RC
	// controller adopts pod2 and delete one of them, so there is only 1 pod
	// left.
	stopCh := make(chan struct{})
	s, rm, _, clientSet := rmSetup(t, stopCh, true)
	ns := framework.CreateTestingNamespace("update-label-to-be-adopted", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	rc := newRC("rc", ns.Name, 1)
	// let rc's selector only matches pod1
	rc.Spec.Selector["uniqueKey"] = "1"
	rc.Spec.Template.Labels["uniqueKey"] = "1"
	pod1 := newMatchingPod("pod1", ns.Name)
	pod1.Labels["uniqueKey"] = "1"
	pod2 := newMatchingPod("pod2", ns.Name)
	pod2.Labels["uniqueKey"] = "2"
	createRCsPods(t, clientSet, []*v1.ReplicationController{rc}, []*v1.Pod{pod1, pod2}, ns.Name)

	go rm.Run(5, stopCh)
	waitRCStable(t, clientSet, rc, ns.Name)

	// change the rc's selector to match both pods
	patch := `{"metadata":{"labels":{"uniqueKey":"1"}}}`
	podClient := clientSet.Core().Pods(ns.Name)
	pod2, err := podClient.Patch(pod2.Name, api.StrategicMergePatchType, []byte(patch))
	if err != nil {
		t.Fatalf("Failed to patch pod2: %v", err)
	}
	t.Logf("patched pod2 = %#v", pod2)
	// wait for the rc to select both pods and delete one of them
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return verifyRemainingObjects(t, clientSet, ns.Name, 1, 1)
	}); err != nil {
		t.Fatal(err)
	}
	close(stopCh)
}
