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

package daemonset

import (
	"fmt"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/daemon"
	"k8s.io/kubernetes/test/integration/controller/common"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	interval = 100 * time.Millisecond
	timeout  = 60 * time.Second
)

func labelMap() map[string]string {
	return map[string]string{"foo": "bar"}
}

func dsSetup(t *testing.T) (*httptest.Server, framework.CloseFunc, *daemon.DaemonSetsController, informers.SharedInformerFactory, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	_, s, closeFn := framework.RunAMaster(masterConfig)

	config := restclient.Config{Host: s.URL}
	clientSet, err := clientset.NewForConfig(&config)
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	resyncPeriod := 12 * time.Hour
	informers := informers.NewSharedInformerFactory(clientset.NewForConfigOrDie(restclient.AddUserAgent(&config, "ds-informers")), resyncPeriod)

	ds := daemon.NewDaemonSetsController(
		informers.Extensions().V1beta1().DaemonSets(),
		informers.Apps().V1beta1().ControllerRevisions(),
		informers.Core().V1().Pods(),
		informers.Core().V1().Nodes(),
		clientSet,
	)

	if err != nil {
		t.Fatalf("Failed to create daemonset controller")
	}
	return s, closeFn, ds, informers, clientSet
}

// Run DS controller and informers
func runControllerAndInformers(t *testing.T, rm *daemon.DaemonSetsController, informers informers.SharedInformerFactory, podNum int) chan struct{} {
	stopCh := make(chan struct{})
	informers.Start(stopCh)
	waitToObservePods(t, informers.Core().V1().Pods().Informer(), podNum)
	go rm.Run(5, stopCh)
	return stopCh
}

// wait for the podInformer to observe the pods. Call this function before
// running the DS controller to prevent the DS manager from creating new pods
// rather than adopting the existing ones.
func waitToObservePods(t *testing.T, podInformer cache.SharedIndexInformer, podNum int) {
	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		objects := podInformer.GetIndexer().List()
		return len(objects) == podNum, nil
	}); err != nil {
		t.Fatalf("Error encountered when waiting for podInformer to observe the pods: %v", err)
	}
}

func newDS(name, namespace string) *v1beta1.DaemonSet {
	return &v1beta1.DaemonSet{
		TypeMeta: metav1.TypeMeta{
			Kind:       "DaemonSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1beta1.DaemonSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labelMap(),
			},
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

func createDSsPods(t *testing.T, clientSet clientset.Interface, dss []*v1beta1.DaemonSet, pods []*v1.Pod) ([]*v1beta1.DaemonSet, []*v1.Pod) {
	var createdDSs []*v1beta1.DaemonSet
	var createdPods []*v1.Pod
	for _, ds := range dss {
		createdDS, err := clientSet.Extensions().DaemonSets(ds.Namespace).Create(ds)
		if err != nil {
			t.Fatalf("Failed to create daemon set %s: %v", ds.Name, err)
		}
		createdDSs = append(createdDSs, createdDS)
	}
	for _, pod := range pods {
		createdPod, err := clientSet.Core().Pods(pod.Namespace).Create(pod)
		if err != nil {
			t.Fatalf("Failed to create pod %s: %v", pod.Name, err)
		}
		createdPods = append(createdPods, createdPod)
	}
	return createdDSs, createdPods
}

// Verify DS has spun up one per node.
func waitDSStable(t *testing.T, clientSet clientset.Interface, ds *v1beta1.DaemonSet) {
	dsClient := clientSet.Extensions().DaemonSets(ds.Namespace)
	nodeClient := clientSet.CoreV1().Nodes()
	nodeList, err := nodeClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to get node list: %v", err)
	}

	if err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		newDS, err := dsClient.Get(ds.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		println(newDS)
		return newDS.Status.DesiredNumberScheduled == int32(len(nodeList.Items)), nil
	}); err != nil {
		t.Fatalf("Failed to verify .Status.Replicas is equal to .Spec.Replicas for ds %s: %v", ds.Name, err)
	}
}

func TestDaemonSetBasic(t *testing.T) {
	s, closeFn, rm, informers, c := dsSetup(t)
	defer closeFn()
	ns := framework.CreateTestingNamespace("test-ready", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)
	stopCh := runControllerAndInformers(t, rm, informers, 0)
	defer close(stopCh)

	// Create a new daemonset & nodes.
	common.CreateNodes(c, "node", nil, 3)
	defer common.DeleteNodes(c, "node", 3)
	ds := newDS("ds1", ns.Name)
	ds.Spec.MinReadySeconds = 3600
	dss, _ := createDSsPods(t, c, []*v1beta1.DaemonSet{ds}, []*v1.Pod{})
	ds = dss[0]
	waitDSStable(t, c, ds)

	// Test that pods have been scheduled.
	podClient := c.Core().Pods(ns.Name)
	pods := common.GetPods(t, podClient, labelMap())
	if len(pods.Items) != 3 {
		t.Fatalf("len(pods) = %d, want 3", len(pods.Items))
	}
}

func TestAdoption(t *testing.T) {
	boolPtr := func(b bool) *bool { return &b }
	testCases := []struct {
		name                    string
		existingOwnerReferences func(ds *v1beta1.DaemonSet) []metav1.OwnerReference
		expectedOwnerReferences func(ds *v1beta1.DaemonSet) []metav1.OwnerReference
	}{
		{
			"pod references ds as an owner, not a controller",
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet"}}
			},
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod doesn't have any owner references",
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{}
			},
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true), BlockOwnerDeletion: boolPtr(true)}}
			},
		},
		{
			"pod references ds as a controller",
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true)}}
			},
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true)}}
			},
		},
		{
			"pod references other-ds as the controller, references ds as an owner",
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherDS", APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true)},
					{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet"},
				}
			},
			func(ds *v1beta1.DaemonSet) []metav1.OwnerReference {
				return []metav1.OwnerReference{
					{UID: "1", Name: "anotherDS", APIVersion: "extensions/v1beta1", Kind: "DaemonSet", Controller: boolPtr(true)},
					{UID: ds.UID, Name: ds.Name, APIVersion: "extensions/v1beta1", Kind: "DaemonSet"},
				}
			},
		},
	}
	for i, tc := range testCases {
		func() {
			s, closeFn, rm, informers, clientSet := dsSetup(t)
			defer closeFn()
			ns := framework.CreateTestingNamespace(fmt.Sprintf("ds-adoption-%d", i), s, t)
			defer framework.DeleteTestingNamespace(ns, s, t)
			common.CreateNodes(clientSet, "node", nil, 1)
			defer common.DeleteNodes(clientSet, "node", 1)

			dsClient := clientSet.Extensions().DaemonSets(ns.Name)
			podClient := clientSet.Core().Pods(ns.Name)
			const dsName = "ds"
			ds, err := dsClient.Create(newDS(dsName, ns.Name))
			if err != nil {
				t.Fatalf("Failed to create daemon set: %v", err)
			}
			podName := fmt.Sprintf("pod%d", i)
			pod := common.NewMatchingPod(podName, ns.Name, labelMap())
			pod.OwnerReferences = tc.existingOwnerReferences(ds)
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
				if e, a := tc.expectedOwnerReferences(ds), updatedPod.OwnerReferences; reflect.DeepEqual(e, a) {
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
