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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	"k8s.io/kubernetes/pkg/controller/garbagecollector"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

func getForegroundOptions() *metav1.DeleteOptions {
	policy := metav1.DeletePropagationForeground
	return &metav1.DeleteOptions{PropagationPolicy: &policy}
}

func getOrphanOptions() *metav1.DeleteOptions {
	var trueVar = true
	return &metav1.DeleteOptions{OrphanDependents: &trueVar}
}

func getNonOrphanOptions() *metav1.DeleteOptions {
	var falseVar = false
	return &metav1.DeleteOptions{OrphanDependents: &falseVar}
}

const garbageCollectedPodName = "test.pod.1"
const independentPodName = "test.pod.2"
const oneValidOwnerPodName = "test.pod.3"
const toBeDeletedRCName = "test.rc.1"
const remainingRCName = "test.rc.2"

func newPod(podName, podNamespace string, ownerReferences []metav1.OwnerReference) *v1.Pod {
	for i := 0; i < len(ownerReferences); i++ {
		if len(ownerReferences[i].Kind) == 0 {
			ownerReferences[i].Kind = "ReplicationController"
		}
		ownerReferences[i].APIVersion = "v1"
	}
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
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
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
		},
		Spec: v1.ReplicationControllerSpec{
			Selector: map[string]string{"name": "test"},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
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

func setup(t *testing.T, stop chan struct{}) (*httptest.Server, framework.CloseFunc, *garbagecollector.GarbageCollector, clientset.Interface) {
	masterConfig := framework.NewIntegrationTestMasterConfig()
	masterConfig.EnableCoreControllers = false
	_, s, closeFn := framework.RunAMaster(masterConfig)

	clientSet, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
	if err != nil {
		t.Fatalf("Error in create clientset: %v", err)
	}
	preferredResources, err := clientSet.Discovery().ServerPreferredResources()
	if err != nil {
		t.Fatalf("Failed to get supported resources from server: %v", err)
	}
	deletableResources := discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"delete"}}, preferredResources)
	deletableGroupVersionResources, err := discovery.GroupVersionResources(deletableResources)
	if err != nil {
		t.Fatalf("Failed to parse supported resources from server: %v", err)
	}
	config := &restclient.Config{Host: s.URL}
	config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
	metaOnlyClientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	config.ContentConfig.NegotiatedSerializer = nil
	clientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	sharedInformers := informers.NewSharedInformerFactory(clientSet, 0)
	gc, err := garbagecollector.NewGarbageCollector(
		metaOnlyClientPool,
		clientPool,
		api.Registry.RESTMapper(),
		deletableGroupVersionResources,
		garbagecollector.DefaultIgnoredResources(),
		sharedInformers,
	)
	if err != nil {
		t.Fatalf("Failed to create garbage collector")
	}

	go sharedInformers.Start(stop)

	return s, closeFn, gc, clientSet
}

// This test simulates the cascading deletion.
func TestCascadingDeletion(t *testing.T) {
	stopCh := make(chan struct{})

	glog.V(6).Infof("TestCascadingDeletion starts")
	defer glog.V(6).Infof("TestCascadingDeletion ends")
	s, closeFn, gc, clientSet := setup(t, stopCh)
	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-cascading-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

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

	rcs, err := rcClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list replication controllers: %v", err)
	}
	if len(rcs.Items) != 2 {
		t.Fatalf("Expect only 2 replication controller")
	}

	// this pod should be cascadingly deleted.
	pod := newPod(garbageCollectedPodName, ns.Name, []metav1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it has a valid reference.
	pod = newPod(oneValidOwnerPodName, ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRCName},
	})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// this pod shouldn't be cascadingly deleted, because it doesn't have an owner.
	pod = newPod(independentPodName, ns.Name, []metav1.OwnerReference{})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 3 {
		t.Fatalf("Expect only 3 pods")
	}
	go gc.Run(5, stopCh)
	// delete one of the replication controller
	if err := rcClient.Delete(toBeDeletedRCName, getNonOrphanOptions()); err != nil {
		t.Fatalf("failed to delete replication controller: %v", err)
	}
	// sometimes the deletion of the RC takes long time to be observed by
	// the gc, so wait for the garbage collector to observe the deletion of
	// the toBeDeletedRC
	if err := wait.Poll(10*time.Second, 60*time.Second, func() (bool, error) {
		return !gc.GraphHasUID([]types.UID{toBeDeletedRC.ObjectMeta.UID}), nil
	}); err != nil {
		t.Fatal(err)
	}
	if err := integration.WaitForPodToDisappear(podClient, garbageCollectedPodName, 5*time.Second, 30*time.Second); err != nil {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}
	// checks the garbage collect doesn't delete pods it shouldn't delete.
	if _, err := podClient.Get(independentPodName, metav1.GetOptions{}); err != nil {
		t.Fatal(err)
	}
	if _, err := podClient.Get(oneValidOwnerPodName, metav1.GetOptions{}); err != nil {
		t.Fatal(err)
	}
}

// This test simulates the case where an object is created with an owner that
// doesn't exist. It verifies the GC will delete such an object.
func TestCreateWithNonExistentOwner(t *testing.T) {
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)
	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-non-existing-owner", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	podClient := clientSet.Core().Pods(ns.Name)

	pod := newPod(garbageCollectedPodName, ns.Name, []metav1.OwnerReference{{UID: "doesn't matter", Name: toBeDeletedRCName}})
	_, err := podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	// set up watch
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Fatalf("Expect only 1 pod")
	}
	go gc.Run(5, stopCh)
	// wait for the garbage collector to delete the pod
	if err := integration.WaitForPodToDisappear(podClient, garbageCollectedPodName, 5*time.Second, 30*time.Second); err != nil {
		t.Fatalf("expect pod %s to be garbage collected, got err= %v", garbageCollectedPodName, err)
	}
}

func setupRCsPods(t *testing.T, gc *garbagecollector.GarbageCollector, clientSet clientset.Interface, nameSuffix, namespace string, initialFinalizers []string, options *metav1.DeleteOptions, wg *sync.WaitGroup, rcUIDs chan types.UID) {
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
		pod := newPod(podName, namespace, []metav1.OwnerReference{{UID: rc.ObjectMeta.UID, Name: rc.ObjectMeta.Name}})
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
		podUIDs = append(podUIDs, pod.ObjectMeta.UID)
	}
	orphan := (options != nil && options.OrphanDependents != nil && *options.OrphanDependents) || (options == nil && len(initialFinalizers) != 0 && initialFinalizers[0] == metav1.FinalizerOrphanDependents)
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
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		return false, fmt.Errorf("Failed to list pods: %v", err)
	}
	var ret = true
	if len(pods.Items) != podNum {
		ret = false
		t.Logf("expect %d pods, got %d pods", podNum, len(pods.Items))
	}
	rcs, err := rcClient.List(metav1.ListOptions{})
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
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)

	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-stressing-cascading-deletion", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	go gc.Run(5, stopCh)

	const collections = 10
	var wg sync.WaitGroup
	wg.Add(collections * 4)
	rcUIDs := make(chan types.UID, collections*4)
	for i := 0; i < collections; i++ {
		// rc is created with empty finalizers, deleted with nil delete options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection1-"+strconv.Itoa(i), ns.Name, []string{}, nil, &wg, rcUIDs)
		// rc is created with the orphan finalizer, deleted with nil options, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection2-"+strconv.Itoa(i), ns.Name, []string{metav1.FinalizerOrphanDependents}, nil, &wg, rcUIDs)
		// rc is created with the orphan finalizer, deleted with DeleteOptions.OrphanDependents=false, pods will be deleted.
		go setupRCsPods(t, gc, clientSet, "collection3-"+strconv.Itoa(i), ns.Name, []string{metav1.FinalizerOrphanDependents}, getNonOrphanOptions(), &wg, rcUIDs)
		// rc is created with empty finalizers, deleted with DeleteOptions.OrphanDependents=true, pods will remain.
		go setupRCsPods(t, gc, clientSet, "collection4-"+strconv.Itoa(i), ns.Name, []string{}, getOrphanOptions(), &wg, rcUIDs)
	}
	wg.Wait()
	t.Logf("all pods are created, all replications controllers are created then deleted")
	// wait for the RCs and Pods to reach the expected numbers.
	if err := wait.Poll(5*time.Second, 300*time.Second, func() (bool, error) {
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
	pods, err := podClient.List(metav1.ListOptions{})
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
}

func TestOrphaning(t *testing.T) {
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)

	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-orphaning", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	podClient := clientSet.Core().Pods(ns.Name)
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC := newOwnerRC(toBeDeletedRCName, ns.Name)
	toBeDeletedRC, err := rcClient.Create(toBeDeletedRC)
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}

	// these pods should be orphaned.
	var podUIDs []types.UID
	podsNum := 3
	for i := 0; i < podsNum; i++ {
		podName := garbageCollectedPodName + strconv.Itoa(i)
		pod := newPod(podName, ns.Name, []metav1.OwnerReference{{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRCName}})
		_, err = podClient.Create(pod)
		if err != nil {
			t.Fatalf("Failed to create Pod: %v", err)
		}
		podUIDs = append(podUIDs, pod.ObjectMeta.UID)
	}
	go gc.Run(5, stopCh)

	// we need wait for the gc to observe the creation of the pods, otherwise if
	// the deletion of RC is observed before the creation of the pods, the pods
	// will not be orphaned.
	wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) { return gc.GraphHasUID(podUIDs), nil })

	err = rcClient.Delete(toBeDeletedRCName, getOrphanOptions())
	if err != nil {
		t.Fatalf("Failed to gracefully delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(5*time.Second, 30*time.Second, func() (bool, error) {
		rcs, err := rcClient.List(metav1.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(rcs.Items) == 0 {
			t.Logf("Still has %d RCs", len(rcs.Items))
			return true, nil
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods don't have the ownerPod as an owner anymore
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != podsNum {
		t.Errorf("Expect %d pod(s), but got %#v", podsNum, pods)
	}
	for _, pod := range pods.Items {
		if len(pod.ObjectMeta.OwnerReferences) != 0 {
			t.Errorf("pod %s still has non-empty OwnerReferences: %v", pod.ObjectMeta.Name, pod.ObjectMeta.OwnerReferences)
		}
	}
}

func TestSolidOwnerDoesNotBlockWaitingOwner(t *testing.T) {
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)

	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-foreground1", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	podClient := clientSet.Core().Pods(ns.Name)
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(newOwnerRC(toBeDeletedRCName, ns.Name))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	remainingRC, err := rcClient.Create(newOwnerRC(remainingRCName, ns.Name))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	trueVar := true
	pod := newPod("pod", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &trueVar},
		{UID: remainingRC.ObjectMeta.UID, Name: remainingRC.Name},
	})
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	go gc.Run(5, stopCh)

	err = rcClient.Delete(toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(5*time.Second, 30*time.Second, func() (bool, error) {
		_, err := rcClient.Get(toBeDeletedRC.Name, metav1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods don't have the toBeDeleteRC as an owner anymore
	pod, err = podClient.Get("pod", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pod.ObjectMeta.OwnerReferences) != 1 {
		t.Errorf("expect pod to have only one ownerReference: got %#v", pod.ObjectMeta.OwnerReferences)
	} else if pod.ObjectMeta.OwnerReferences[0].Name != remainingRC.Name {
		t.Errorf("expect pod to have an ownerReference pointing to %s, got %#v", remainingRC.Name, pod.ObjectMeta.OwnerReferences)
	}
}

func TestNonBlockingOwnerRefDoesNotBlock(t *testing.T) {
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)

	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-foreground2", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	podClient := clientSet.Core().Pods(ns.Name)
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(newOwnerRC(toBeDeletedRCName, ns.Name))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	// BlockingOwnerDeletion is not set
	pod1 := newPod("pod1", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod1.ObjectMeta.Finalizers = []string{"x/y"}
	// BlockingOwnerDeletion is false
	falseVar := false
	pod2 := newPod("pod2", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &falseVar},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod2.ObjectMeta.Finalizers = []string{"x/y"}
	_, err = podClient.Create(pod1)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}
	_, err = podClient.Create(pod2)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	go gc.Run(5, stopCh)

	err = rcClient.Delete(toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	// verify the toBeDeleteRC is deleted
	if err := wait.PollImmediate(5*time.Second, 30*time.Second, func() (bool, error) {
		_, err := rcClient.Get(toBeDeletedRC.Name, metav1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods are still there
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 2 {
		t.Errorf("expect there to be 2 pods, got %#v", pods.Items)
	}
}

func TestBlockingOwnerRefDoesBlock(t *testing.T) {
	stopCh := make(chan struct{})
	s, closeFn, gc, clientSet := setup(t, stopCh)

	defer func() {
		// We have to close the stop channel first, so the shared informers can terminate their watches;
		// otherwise closeFn() will hang waiting for active client connections to finish.
		close(stopCh)
		closeFn()
	}()

	ns := framework.CreateTestingNamespace("gc-foreground3", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	podClient := clientSet.Core().Pods(ns.Name)
	rcClient := clientSet.Core().ReplicationControllers(ns.Name)
	// create the RC with the orphan finalizer set
	toBeDeletedRC, err := rcClient.Create(newOwnerRC(toBeDeletedRCName, ns.Name))
	if err != nil {
		t.Fatalf("Failed to create replication controller: %v", err)
	}
	trueVar := true
	pod := newPod("pod", ns.Name, []metav1.OwnerReference{
		{UID: toBeDeletedRC.ObjectMeta.UID, Name: toBeDeletedRC.Name, BlockOwnerDeletion: &trueVar},
	})
	// adding finalizer that no controller handles, so that the pod won't be deleted
	pod.ObjectMeta.Finalizers = []string{"x/y"}
	_, err = podClient.Create(pod)
	if err != nil {
		t.Fatalf("Failed to create Pod: %v", err)
	}

	go gc.Run(5, stopCh)

	// this makes sure the garbage collector will have added the pod to its
	// dependency graph before handling the foreground deletion of the rc.
	timeout := make(chan struct{})
	go func() {
		select {
		case <-time.After(5 * time.Second):
			close(timeout)
		}
	}()
	if !cache.WaitForCacheSync(timeout, gc.HasSynced) {
		t.Fatalf("failed to wait for garbage collector to be synced")
	}

	err = rcClient.Delete(toBeDeletedRCName, getForegroundOptions())
	if err != nil {
		t.Fatalf("Failed to delete the rc: %v", err)
	}
	time.Sleep(30 * time.Second)
	// verify the toBeDeleteRC is NOT deleted
	_, err = rcClient.Get(toBeDeletedRC.Name, metav1.GetOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// verify pods are still there
	pods, err := podClient.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list pods: %v", err)
	}
	if len(pods.Items) != 1 {
		t.Errorf("expect there to be 1 pods, got %#v", pods.Items)
	}
}
