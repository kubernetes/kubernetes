// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package integration

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/integration/framework"
)

const garbageCollectedPodName = "test.pod.1"
const independentPodName = "test.pod.2"
const oneValidOwnerPodName = "test.pod.3"
const toBeDeletedRCName = "test.rc.1"
const remainingRCName = "test.rc.2"

func newPod(podName string, ownerReferences []v1.OwnerReference) *v1.Pod {
	for i := 0; i < len(ownerReferences); i++ {
		ownerReferences[i].Kind = "ReplicationController"
		ownerReferences[i].APIVersion = "v1"
	}
	return &v1.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:            podName,
			Namespace:       framework.TestNS,
			OwnerReferences: ownerReferences,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-name",
					Image: "fakeimage",
				},
			},
		},
	}
}

func newOwnerRC(name string) *v1.ReplicationController {
	return &v1.ReplicationController{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Namespace: framework.TestNS,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Selector: map[string]string{"name": "test"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: v1.ObjectMeta{
					Labels: map[string]string{"name": "test"},
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

func setup(t *testing.T) (*garbagecollector.GarbageCollector, clientset.Interface) {
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	// TODO: close the http server

	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.EnableCoreControllers = false
	m, err := master.New(masterConfig)
	if err != nil {
		t.Fatalf("Error in bringing up the master: %v", err)
	}

	framework.DeleteAllEtcdKeys()
	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	groupVersionResources, err := clientSet.Discovery().ServerPreferredResources()
	if err != nil {
		t.Fatalf("Failed to get supported resources from server: %v", err)
	}
	clientPool := dynamic.NewClientPool(&restclient.Config{Host: s.URL}, dynamic.LegacyAPIPathResolverFunc)
	gc, err := garbagecollector.NewGarbageCollector(clientPool, groupVersionResources)
	if err != nil {
		t.Fatalf("Failed to create garbage collector")
	}
	return gc, clientSet
}

// This test simulates the cascading deletion.
func TestCascadingDeletion(t *testing.T) {
	gc, clientSet := setup(t)
	rcClient := clientSet.Core().ReplicationControllers(framework.TestNS)
	podClient := clientSet.Core().Pods(framework.TestNS)

	toBeDeletedRC, err := rcClient.Create(newOwnerRC(toBeDeletedRCName))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	remainingRC, err := rcClient.Create(newOwnerRC(remainingRCName))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	rcs, err := rcClient.List(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != 2 {
		t.Fatalf("Expect only 2 replication controller")
	}

	// this pod should be cascadingly deleted.
	pod := newPod(garbageCollectedPodName, []v1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it has a valid referenece.
	pod = newPod(oneValidOwnerPodName, []v1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRCName},
	})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it doesn't have an owner.
	pod = newPod(independentPodName, []v1.OwnerReference{})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 3 {
		t.Fatalf("Expect only 3 pods")
	}
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)
	// delete one of the replication controller
	if err := rcClient.Delete(toBeDeletedRCName, nil); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}

	// wait for the garbage collector to drain its queue
	if err := wait.Poll(10*time.Second, 120*time.Second, func() (bool, error) {
		return gc.QueuesDrained(), nil
	}); err != nil {
		t.Fatal(err)
	}
	// sometimes the deletion of the RC takes long time to be observed by
	// the gc, so wait for the garbage collector to observe the deletion of
	// the toBeDeletedRC
	if err := wait.Poll(10*time.Second, 120*time.Second, func() (bool, error) {
		return !gc.GraphHasUID([]types.UID{toBeDeletedRC.ObjectMeta.UID}), nil
	}); err != nil {
		t.Fatal(err)
	}
	// wait for the garbage collector to drain its queue again because it's
	// possible it just processed the delete of the toBeDeletedRC.
	if err := wait.Poll(10*time.Second, 120*time.Second, func() (bool, error) {
		return gc.QueuesDrained(), nil
	}); err != nil {
		t.Fatal(err)
	}

	t.Logf("garbage collector queues drained")
	// checks the garbage collect doesn't delete pods it shouldn't do.
	if _, err := podClient.Get(independentPodName); err != nil {
		t.Fatal(err)
	}
	if _, err := podClient.Get(oneValidOwnerPodName); err != nil {
		t.Fatal(err)
	}
	if _, err := podClient.Get(garbageCollectedPodName); err == nil || !errors.IsNotFound(err) {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}
}

// This test simulates the case where an object is created with an owner that
// doesn't exist. It verifies the GC will delete such an object.
func TestCreateWithNonExisitentOwner(t *testing.T) {
	gc, clientSet := setup(t)
	podClient := clientSet.Core().Pods(framework.TestNS)

	pod := newPod(garbageCollectedPodName, []v1.OwnerReference{{UID: "doesn't matter", Name: toBeDeletedRCName}})
	_, err := podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Fatalf("Expect only 1 pod")
	}
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)
	// wait for the garbage collector to drain its queue
	if err := wait.Poll(10*time.Second, 120*time.Second, func() (bool, error) {
		return gc.QueuesDrained(), nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("garbage collector queues drained")
	if _, err := podClient.Get(garbageCollectedPodName); err == nil || !errors.IsNotFound(err) {
		t.Fatalf("expect pod %s to be garbage collected", garbageCollectedPodName)
	}
}

func createRemoveRCsPods(t *testing.T, clientSet clientset.Interface, id int, wg *sync.WaitGroup, rcUIDs chan types.UID) {
	defer wg.Done()
	rcClient := clientSet.Core().ReplicationControllers(framework.TestNS)
	podClient := clientSet.Core().Pods(framework.TestNS)
	// create rc.
	rcName := toBeDeletedRCName + strconv.Itoa(id)
	toBeDeletedRC, err := rcClient.Create(newOwnerRC(rcName))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	rcUIDs <- toBeDeletedRC.ObjectMeta.UID
	// create pods. These pods should be cascadingly deleted.
	for j := 0; j < 3; j++ {
		podName := garbageCollectedPodName + strconv.Itoa(id) + "-" + strconv.Itoa(j)
		pod := newPod(podName, []v1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: rcName}})
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
	}
	// delete the rc
	if err := rcClient.Delete(rcName, nil); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}
}

func allObjectsRemoved(clientSet clientset.Interface) (bool, error) {
	rcClient := clientSet.Core().ReplicationControllers(framework.TestNS)
	podClient := clientSet.Core().Pods(framework.TestNS)
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 0 {
		return false, nil
	}
	rcs, err := rcClient.List(api.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != 0 {
		return false, nil
	}
	return true, nil
}

// This stress test the garbage collector
func TestStressingCascadingDeletion(t *testing.T) {
	t.Logf("starts garbage collector stress test")
	gc, clientSet := setup(t)
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)

	const collections = 50
	var wg sync.WaitGroup
	wg.Add(collections)
	rcUIDs := make(chan types.UID, collections)
	for i := 0; i < collections; i++ {
		go createRemoveRCsPods(t, clientSet, i, &wg, rcUIDs)
	}
	wg.Wait()
	t.Logf("all pods are created, all replications controllers are created then deleted")
	// wait for the garbage collector to drain its queue
	if err := wait.Poll(10*time.Second, 300*time.Second, func() (bool, error) {
		return gc.QueuesDrained(), nil
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("garbage collector queues drained")
	// wait for all replication controllers and pods to be deleted. This
	// shouldn't take long, because the queues are already drained.
	if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
		return allObjectsRemoved(clientSet)
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("all replication controllers and pods are deleted")

	// verify there is no node representing replication controllers in the gc's graph
	uids := make([]types.UID, 0, collections)
	for i := 0; i < collections; i++ {
		uid := <-rcUIDs
		uids = append(uids, uid)
	}
	if gc.GraphHasUID(uids) {
		t.Errorf("Expect all nodes representing replication controllers are removed from the Propagator's graph")
	}
}
