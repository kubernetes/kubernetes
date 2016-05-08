//// +build integration,!no-etcd

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
	"flag"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/master"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/integration/framework"
)

const garbageCollectedPodName = "test.pod.1"
const independentPodName = "test.pod.2"
const oneValidOwnerPodName = "test.pod.3"
const rcName = "test.rc.1"
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

func observePodDeletion(t *testing.T, w watch.Interface) (deletedPod *api.Pod) {
	deleted := false
	timeout := false
	timer := time.After(60 * time.Second)
	for !deleted && !timeout {
		select {
		case event, _ := <-w.ResultChan():
			if event.Type == watch.Deleted {
				// TODO: used the commented code once we fix the client.
				// deletedPod = event.Object.(*v1.Pod)
				deletedPod = event.Object.(*api.Pod)
				deleted = true
			}
		case <-timer:
			timeout = true
		}
	}
	if !deleted {
		t.Fatalf("Failed to observe pod deletion")
	}
	return
}

func setup(t *testing.T) (*garbagecollector.GarbageCollector, clientset.Interface) {
	flag.Set("v", "9")
	var m *master.Master
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))
	// TODO: Uncomment when fix #19254
	// defer s.Close()

	masterConfig := framework.NewIntegrationTestMasterConfig()
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

	toBeDeletedRC, err := rcClient.Create(newOwnerRC(rcName))
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

	pod := newPod(garbageCollectedPodName, []v1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: rcName}})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	pod = newPod(oneValidOwnerPodName, []v1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: rcName},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRCName},
	})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

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
	options := api.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	w, err := podClient.Watch(options)
	if err != nil {
		t.Fatalf("Failed to set up watch: %v", err)
	}
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)
	if err := rcClient.Delete(rcName, nil); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}

	deletedPod := observePodDeletion(t, w)
	if deletedPod == nil {
		t.Fatalf("empty deletedPod")
	}
	if deletedPod.Name != garbageCollectedPodName {
		t.Fatalf("deleted unexpected pod: %v", *deletedPod)
	}
	// wait for another 30 seconds to give garbage collect a chance to make mistakes.
	time.Sleep(30 * time.Second)
	if _, err := podClient.Get(independentPodName); err != nil {
		t.Fatal(err)
	}
	if _, err := podClient.Get(oneValidOwnerPodName); err != nil {
		t.Fatal(err)
	}
}

// This test simulates the case where an object is created with an owner that
// doesn't exist. It verifies the GC will delete such delete such an object.
func TestCreateWithNonExisitentOwner(t *testing.T) {
	gc, clientSet := setup(t)
	podClient := clientSet.Core().Pods(framework.TestNS)

	pod := newPod(garbageCollectedPodName, []v1.OwnerReference{{UID: "doesn't matter", Name: rcName}})
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
	options := api.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	w, err := podClient.Watch(options)
	if err != nil {
		t.Fatalf("Failed to set up watch: %v", err)
	}
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)
	deletedPod := observePodDeletion(t, w)
	if deletedPod == nil {
		t.Fatalf("empty deletedPod")
	}
	if deletedPod.Name != garbageCollectedPodName {
		t.Fatalf("deleted unexpected pod: %v", *deletedPod)
	}
}
