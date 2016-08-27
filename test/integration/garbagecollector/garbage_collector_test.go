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

package garbagecollector

import (
	"fmt"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/golang/glog"
	dto "github.com/prometheus/client_model/go"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/integration/framework"
)

func getOrphanOptions() *api.DeleteOptions {
	var trueVar = true
	return &api.DeleteOptions{OrphanDependents: &trueVar}
}

func getNonOrphanOptions() *api.DeleteOptions {
	var falseVar = false
	return &api.DeleteOptions{OrphanDependents: &falseVar}
}

const garbageCollectedPodName = "test.pod.1"
const independentPodName = "test.pod.2"
const oneValidOwnerPodName = "test.pod.3"
const toBeDeletedRCName = "test.rc.1"
const remainingRCName = "test.rc.2"

func newPod(podName, podNamespace string, ownerReferences []v1.OwnerReference) *v1.Pod {
	for i := 0; i < len(ownerReferences); i++ {
		if len(ownerReferences[i].Kind) == 0 {
			ownerReferences[i].Kind = "ReplicationController"
		}
		ownerReferences[i].APIVersion = "v1"
	}
	return &v1.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Name:            podName,
			Namespace:       podNamespace,
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

func newOwnerRC(name, namespace string) *v1.ReplicationController {
	return &v1.ReplicationController{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: v1.ObjectMeta{
			Namespace: namespace,
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

func setup(t *testing.T) (*httptest.Server, *garbagecollector.GarbageCollector, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.EnableCoreControllers = false
	_, s := framework.RunAMaster(masterConfig)

	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	groupVersionResources, err := clientSet.Discovery().ServerPreferredResources()
	if err != nil {
		t.Fatalf("Failed to get supported resources from server: %v", err)
	}
	config := &restclient.Config{Host: s.URL}
	config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
	metaOnlyClientPool := dynamic.NewClientPool(config, dynamic.LegacyAPIPathResolverFunc)
	config.ContentConfig.NegotiatedSerializer = nil
	clientPool := dynamic.NewClientPool(config, dynamic.LegacyAPIPathResolverFunc)
	gc, err := garbagecollector.NewGarbageCollector(metaOnlyClientPool, clientPool, groupVersionResources)
	if err != nil {
		t.Fatalf("Failed to create garbage collector")
	}
	return s, gc, clientSet
}

// This test simulates the cascading deletion.
func TestCascadingDeletion(t *testing.T) {
	glog.V(6).Infof("TestCascadingDeletion starts")
	defer glog.V(6).Infof("TestCascadingDeletion ends")
	s, gc, clientSet := setup(t)
	defer s.Close()

	ns := framework.CreateTestingNamespace("gc-cascading-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	oldEnableGarbageCollector := registry.EnableGarbageCollector
	registry.EnableGarbageCollector = true
	defer func() { registry.EnableGarbageCollector = oldEnableGarbageCollector }()
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	podClient := clientSet.Core().Pods(ns.Name)

	toBeDeletedRC, err := rcClient.Create(newOwnerRC(toBeDeletedRCName, ns.Name))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	remainingRC, err := rcClient.Create(newOwnerRC(remainingRCName, ns.Name))
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
	pod := newPod(garbageCollectedPodName, ns.Name, []v1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it has a valid reference.
	pod = newPod(oneValidOwnerPodName, ns.Name, []v1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRCName},
	})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it doesn't have an owner.
	pod = newPod(independentPodName, ns.Name, []v1.OwnerReference{})
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
	if err := rcClient.Delete(toBeDeletedRCName, getNonOrphanOptions()); err != nil {
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
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
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
func TestCreateWithNonExistentOwner(t *testing.T) {
	glog.V(6).Infof("TestCreateWithNonExistentOwner starts")
	defer glog.V(6).Infof("TestCreateWithNonExistentOwner ends")
	s, gc, clientSet := setup(t)
	defer s.Close()

	ns := framework.CreateTestingNamespace("gc-non-existing-owner", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	oldEnableGarbageCollector := registry.EnableGarbageCollector
	registry.EnableGarbageCollector = true
	defer func() { registry.EnableGarbageCollector = oldEnableGarbageCollector }()
	podClient := clientSet.Core().Pods(ns.Name)

	pod := newPod(garbageCollectedPodName, ns.Name, []v1.OwnerReference{{UID: "doesn't matter", Name: toBeDeletedRCName}})
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

func setupRCsPods(t *testing.T, gc *garbagecollector.GarbageCollector, clientSet clientset.Interface, nameSuffix, namespace string, initialFinalizers []string, options *api.DeleteOptions, wg *sync.WaitGroup, rcUIDs chan types.UID) {
	defer wg.Done()
	rcClient := clientSet.Core().ReplicationControllers(namespace)
	podClient := clientSet.Core().Pods(namespace)
	// create rc.
	rcName := "test.rc." + nameSuffix
	rc := newOwnerRC(rcName, namespace)
	rc.ObjectMeta.Finalizers = initialFinalizers
	rc, err := rcClient.Create(rc)
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	rcUIDs <- rc.ObjectMeta.UID
	// create pods.
	var podUIDs []types.UID
	for j := 0; j < 3; j++ {
		podName := "test.pod." + nameSuffix + "-" + strconv.Itoa(j)
		pod := newPod(podName, namespace, []v1.OwnerReference{{UID: rc.ObjectMeta.UID, Name: rc.ObjectMeta.Name}})
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
		podUIDs = append(podUIDs, pod.ObjectMeta.UID)
	}
	orphan := (options != nil && options.OrphanDependents != nil && *options.OrphanDependents) || (options == nil && len(initialFinalizers) != 0 && initialFinalizers[0] == api.FinalizerOrphan)
	// if we intend to orphan the pods, we need wait for the gc to observe the
	// creation of the pods, otherwise if the deletion of RC is observed before
	// the creation of the pods, the pods will not be orphaned.
	if orphan {
		wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) { return gc.GraphHasUID(podUIDs), nil })
	}
	// delete the rc
	if err := rcClient.Delete(rc.ObjectMeta.Name, options); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}
}

func verifyRemainingObjects(t *testing.T, clientSet clientset.Interface, namespace string, rcNum, podNum int) (bool, error) {
	rcClient := clientSet.Core().ReplicationControllers(namespace)
	podClient := clientSet.Core().Pods(namespace)
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		t.Logf("expect %d pods, got %d pods", podNum, len(pods.Items))
	}
	rcs, err := rcClient.List(api.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != rcNum {
		ret = false
		t.Logf("expect %d RCs, got %d RCs", rcNum, len(rcs.Items))
	}
	return ret, nil
}

// The stress test is not very stressful, because we need to control the running
// time of our pre-submit tests to increase submit-queue throughput. We'll add
// e2e tests that put more stress.
func TestStressingCascadingDeletion(t *testing.T) {
	t.Logf("starts garbage collector stress test")
	s, gc, clientSet := setup(t)
	defer s.Close()

	ns := framework.CreateTestingNamespace("gc-stressing-cascading-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	oldEnableGarbageCollector := registry.EnableGarbageCollector
	registry.EnableGarbageCollector = true
	defer func() { registry.EnableGarbageCollector = oldEnableGarbageCollector }()
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)

	const collections = 10
	var wg sync.WaitGroup
	wg.Add(collections * 4)
	rcUIDs := make(chan types.UID, collections*4)
	for i := 0; i < collections; i++ {
		// rc is created with empty finalizers, deleted with nil delete options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection1-"+strconv.Itoa(i), ns.Name, []string{}, nil, &wg, rcUIDs)
		// rc is created with the orphan finalizer, deleted with nil options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection2-"+strconv.Itoa(i), ns.Name, []string{api.FinalizerOrphan}, nil, &wg, rcUIDs)
		// rc is created with the orphan finalizer, deleted with DeleteOptions.OrphanDependents=false, pods will be deleted.
		go setupRCsPods(t, gc, clientSet, "collection3-"+strconv.Itoa(i), ns.Name, []string{api.FinalizerOrphan}, getNonOrphanOptions(), &wg, rcUIDs)
		// rc is created with empty finalizers, deleted with DeleteOptions.OrphanDependents=true, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection4-"+strconv.Itoa(i), ns.Name, []string{}, getOrphanOptions(), &wg, rcUIDs)
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
	// wait for the RCs and Pods to reach the expected numbers. This shouldn't
	// take long, because the queues are already drained.
	if err := wait.Poll(5*time.Second, 30*time.Second, func() (bool, error) {
		podsInEachCollection := 3
		// see the comments on the calls to setupRCsPods for details
		remainingGroups := 3
		return verifyRemainingObjects(t, clientSet, ns.Name, 0, collections*podsInEachCollection*remainingGroups)
	}); err != nil {
		t.Fatal(err)
	}
	t.Logf("number of remaining replication controllers and pods are as expected")

	// verify the remaining pods all have "orphan" in their names.
	podClient := clientSet.Core().Pods(ns.Name)
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, pod := range pods.Items {
		if !strings.Contains(pod.ObjectMeta.Name, "collection1-") && !strings.Contains(pod.ObjectMeta.Name, "collection2-") && !strings.Contains(pod.ObjectMeta.Name, "collection4-") {
			t.Errorf("got unexpected remaining pod: %#v", pod)
		}
	}

	// verify there is no node representing replication controllers in the gc's graph
	uids := make([]types.UID, 0, collections)
	for i := 0; i < collections; i++ {
		uid := <-rcUIDs
		uids = append(uids, uid)
	}
	if gc.GraphHasUID(uids) {
		t.Errorf("Expect all nodes representing replication controllers are removed from the Propagator's graph")
	}
	metric := &dto.Metric{}
	garbagecollector.EventProcessingLatency.Write(metric)
	count := float64(metric.Summary.GetSampleCount())
	sum := metric.Summary.GetSampleSum()
	t.Logf("Average time spent in GC's eventQueue is %.1f microseconds", sum/count)
	garbagecollector.DirtyProcessingLatency.Write(metric)
	count = float64(metric.Summary.GetSampleCount())
	sum = metric.Summary.GetSampleSum()
	t.Logf("Average time spent in GC's dirtyQueue is %.1f microseconds", sum/count)
	garbagecollector.OrphanProcessingLatency.Write(metric)
	count = float64(metric.Summary.GetSampleCount())
	sum = metric.Summary.GetSampleSum()
	t.Logf("Average time spent in GC's orphanQueue is %.1f microseconds", sum/count)
}

func TestOrphaning(t *testing.T) {
	s, gc, clientSet := setup(t)
	defer s.Close()

	ns := framework.CreateTestingNamespace("gc-orphaning", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	oldEnableGarbageCollector := registry.EnableGarbageCollector
	registry.EnableGarbageCollector = true
	defer func() { registry.EnableGarbageCollector = oldEnableGarbageCollector }()
	podClient := clientSet.Core().Pods(ns.Name)
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC := newOwnerRC(toBeDeletedRCName, ns.Name)
	toBeDeletedRC, err := rcClient.Create(toBeDeletedRC)
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	// these pods should be ophaned.
	var podUIDs []types.UID
	podsNum := 3
	for i := 0; i < podsNum; i++ {
		podName := garbageCollectedPodName + strconv.Itoa(i)
		pod := newPod(podName, ns.Name, []v1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
		podUIDs = append(podUIDs, pod.ObjectMeta.UID)
	}
	stopCh := make(chan struct{})
	go gc.Run(5, stopCh)
	defer close(stopCh)

	// we need wait for the gc to observe the creation of the pods, otherwise if
	// the deletion of RC is observed before the creation of the pods, the pods
	// will not be orphaned.
	wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) { return gc.GraphHasUID(podUIDs), nil })

	err = rcClient.Delete(toBeDeletedRCName, getOrphanOptions())
	if err != nil {
		t.Fatalf("Failed to gracefully delete the rc: %v", err)
	}

	// wait for the garbage collector to drain its queue
	if err := wait.Poll(10*time.Second, 300*time.Second, func() (bool, error) {
		return gc.QueuesDrained(), nil
	}); err != nil {
		t.Fatal(err)
	}

	// verify pods don't have the ownerPod as an owner anymore
	pods, err := podClient.List(api.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != podsNum {
		t.Errorf("Expect %d pod(s), but got %#v", podsNum, pods)
	}
	for _, pod := range pods.Items {
		if len(pod.ObjectMeta.OwnerReferences) != 0 {
			t.Errorf("pod %s still has non-empty OwnerRefereces: %v", pod.ObjectMeta.Name, pod.ObjectMeta.OwnerReferences)
		}
	}
	// verify the toBeDeleteRC is deleted
	rcs, err := rcClient.List(api.ListOptions{})
	if len(rcs.Items) != 0 {
		t.Errorf("Expect RCs to be deleted, but got %#v", rcs.Items)
	}
}
