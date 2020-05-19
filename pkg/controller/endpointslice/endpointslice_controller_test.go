/*
Copyright 2019 The Kubernetes Authors.

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

package endpointslice

import (
	"context"
	"fmt"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller"
	endpointutil "k8s.io/kubernetes/pkg/controller/util/endpoint"
	utilpointer "k8s.io/utils/pointer"
)

// Most of the tests related to EndpointSlice allocation can be found in reconciler_test.go
// Tests here primarily focus on unique controller functionality before the reconciler begins

var alwaysReady = func() bool { return true }

type endpointSliceController struct {
	*Controller
	endpointSliceStore cache.Store
	nodeStore          cache.Store
	podStore           cache.Store
	serviceStore       cache.Store
}

func newController(nodeNames []string, batchPeriod time.Duration) (*fake.Clientset, *endpointSliceController) {
	client := newClientset()
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	nodeInformer := informerFactory.Core().V1().Nodes()
	indexer := nodeInformer.Informer().GetIndexer()
	for _, nodeName := range nodeNames {
		indexer.Add(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}

	esController := NewController(
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Services(),
		nodeInformer,
		informerFactory.Discovery().V1beta1().EndpointSlices(),
		int32(100),
		client,
		batchPeriod)

	esController.nodesSynced = alwaysReady
	esController.podsSynced = alwaysReady
	esController.servicesSynced = alwaysReady
	esController.endpointSlicesSynced = alwaysReady

	return client, &endpointSliceController{
		esController,
		informerFactory.Discovery().V1beta1().EndpointSlices().Informer().GetStore(),
		informerFactory.Core().V1().Nodes().Informer().GetStore(),
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Services().Informer().GetStore(),
	}
}

// Ensure SyncService for service with no selector results in no action
func TestSyncServiceNoSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: ns},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{TargetPort: intstr.FromInt(80)}},
		},
	})

	err := esController.syncService(fmt.Sprintf("%s/%s", ns, serviceName))
	assert.Nil(t, err)
	assert.Len(t, client.Actions(), 0)
}

// Ensure SyncService for service with selector but no pods results in placeholder EndpointSlice
func TestSyncServiceWithSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	standardSyncService(t, esController, ns, serviceName, "true")
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	sliceList, err := client.DiscoveryV1beta1().EndpointSlices(ns).List(context.TODO(), metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, sliceList.Items, 1, "Expected 1 endpoint slices")
	slice := sliceList.Items[0]
	assert.Regexp(t, "^"+serviceName, slice.Name)
	assert.Equal(t, serviceName, slice.Labels[discovery.LabelServiceName])
	assert.EqualValues(t, []discovery.EndpointPort{}, slice.Ports)
	assert.EqualValues(t, []discovery.Endpoint{}, slice.Endpoints)
	assert.NotEmpty(t, slice.Annotations["endpoints.kubernetes.io/last-change-trigger-time"])
}

// Ensure SyncService gracefully handles a missing service. This test also
// populates another existing service to ensure a clean up process doesn't
// remove too much.
func TestSyncServiceMissing(t *testing.T) {
	namespace := metav1.NamespaceDefault
	client, esController := newController([]string{"node-1"}, time.Duration(0))

	// Build up existing service
	existingServiceName := "stillthere"
	existingServiceKey := endpointutil.ServiceKey{Name: existingServiceName, Namespace: namespace}
	esController.triggerTimeTracker.ServiceStates[existingServiceKey] = endpointutil.ServiceState{}
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: existingServiceName, Namespace: namespace},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt(80)}},
			Selector: map[string]string{"foo": "bar"},
		},
	})

	// Add missing service to triggerTimeTracker to ensure the reference is cleaned up
	missingServiceName := "notthere"
	missingServiceKey := endpointutil.ServiceKey{Name: missingServiceName, Namespace: namespace}
	esController.triggerTimeTracker.ServiceStates[missingServiceKey] = endpointutil.ServiceState{}

	err := esController.syncService(fmt.Sprintf("%s/%s", namespace, missingServiceName))

	// nil should be returned when the service doesn't exist
	assert.Nil(t, err, "Expected no error syncing service")

	// That should mean no client actions were performed
	assert.Len(t, client.Actions(), 0)

	// TriggerTimeTracker should have removed the reference to the missing service
	assert.NotContains(t, esController.triggerTimeTracker.ServiceStates, missingServiceKey)

	// TriggerTimeTracker should have left the reference to the missing service
	assert.Contains(t, esController.triggerTimeTracker.ServiceStates, existingServiceKey)
}

// Ensure SyncService correctly selects Pods.
func TestSyncServicePodSelection(t *testing.T) {
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault

	pod1 := newPod(1, ns, true, 0)
	esController.podStore.Add(pod1)

	// ensure this pod will not match the selector
	pod2 := newPod(2, ns, true, 0)
	pod2.Labels["foo"] = "boo"
	esController.podStore.Add(pod2)

	standardSyncService(t, esController, ns, "testing-1", "true")
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	// an endpoint slice should be created, it should only reference pod1 (not pod2)
	slices, err := client.DiscoveryV1beta1().EndpointSlices(ns).List(context.TODO(), metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, slices.Items, 1, "Expected 1 endpoint slices")
	slice := slices.Items[0]
	assert.Len(t, slice.Endpoints, 1, "Expected 1 endpoint in first slice")
	assert.NotEmpty(t, slice.Annotations["endpoints.kubernetes.io/last-change-trigger-time"])
	endpoint := slice.Endpoints[0]
	assert.EqualValues(t, endpoint.TargetRef, &v1.ObjectReference{Kind: "Pod", Namespace: ns, Name: pod1.Name})
}

// Ensure SyncService correctly selects and labels EndpointSlices.
func TestSyncServiceEndpointSliceLabelSelection(t *testing.T) {
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"

	// 5 slices, 3 with matching labels for our service
	endpointSlices := []*discovery.EndpointSlice{{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "matching-1",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   controllerName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "matching-2",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   controllerName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "partially-matching-1",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "not-matching-1",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: "something-else",
				discovery.LabelManagedBy:   controllerName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:      "not-matching-2",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   "something-else",
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}}

	cmc := newCacheMutationCheck(endpointSlices)

	// need to add them to both store and fake clientset
	for _, endpointSlice := range endpointSlices {
		err := esController.endpointSliceStore.Add(endpointSlice)
		if err != nil {
			t.Fatalf("Expected no error adding EndpointSlice: %v", err)
		}
		_, err = client.DiscoveryV1beta1().EndpointSlices(ns).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Expected no error creating EndpointSlice: %v", err)
		}
	}

	// +1 for extra action involved in Service creation before syncService call.
	numActionsBefore := len(client.Actions()) + 1
	standardSyncService(t, esController, ns, serviceName, "false")

	if len(client.Actions()) != numActionsBefore+2 {
		t.Errorf("Expected 2 more actions, got %d", len(client.Actions())-numActionsBefore)
	}

	// only 2 slices should match, 2 should be deleted, 1 should be updated as a placeholder
	expectAction(t, client.Actions(), numActionsBefore, "update", "endpointslices")
	expectAction(t, client.Actions(), numActionsBefore+1, "delete", "endpointslices")

	// ensure cache mutation has not occurred
	cmc.Check(t)
}

// Ensure SyncService handles a variety of protocols and IPs appropriately.
func TestSyncServiceFull(t *testing.T) {
	client, esController := newController([]string{"node-1"}, time.Duration(0))
	namespace := metav1.NamespaceDefault
	serviceName := "all-the-protocols"
	ipv6Family := v1.IPv6Protocol

	pod1 := newPod(1, namespace, true, 0)
	pod1.Status.PodIPs = []v1.PodIP{{IP: "1.2.3.4"}}
	esController.podStore.Add(pod1)

	pod2 := newPod(2, namespace, true, 0)
	pod2.Status.PodIPs = []v1.PodIP{{IP: "1.2.3.5"}, {IP: "1234::5678:0000:0000:9abc:def0"}}
	esController.podStore.Add(pod2)

	// create service with all protocols and multiple ports
	serviceCreateTime := time.Now()
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(serviceCreateTime),
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{Name: "tcp-example", TargetPort: intstr.FromInt(80), Protocol: v1.ProtocolTCP},
				{Name: "udp-example", TargetPort: intstr.FromInt(161), Protocol: v1.ProtocolUDP},
				{Name: "sctp-example", TargetPort: intstr.FromInt(3456), Protocol: v1.ProtocolSCTP},
			},
			Selector: map[string]string{"foo": "bar"},
			IPFamily: &ipv6Family,
		},
	}
	esController.serviceStore.Add(service)
	_, err := esController.client.CoreV1().Services(namespace).Create(context.TODO(), service, metav1.CreateOptions{})
	assert.Nil(t, err, "Expected no error creating service")

	// run through full sync service loop
	err = esController.syncService(fmt.Sprintf("%s/%s", namespace, serviceName))
	assert.Nil(t, err)

	// last action should be to create endpoint slice
	expectActions(t, client.Actions(), 1, "create", "endpointslices")
	sliceList, err := client.DiscoveryV1beta1().EndpointSlices(namespace).List(context.TODO(), metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, sliceList.Items, 1, "Expected 1 endpoint slices")

	// ensure all attributes of endpoint slice match expected state
	slice := sliceList.Items[0]
	assert.Len(t, slice.Endpoints, 1, "Expected 1 endpoints in first slice")
	assert.Equal(t, slice.Annotations["endpoints.kubernetes.io/last-change-trigger-time"], serviceCreateTime.Format(time.RFC3339Nano))
	assert.EqualValues(t, []discovery.EndpointPort{{
		Name:     utilpointer.StringPtr("sctp-example"),
		Protocol: protoPtr(v1.ProtocolSCTP),
		Port:     utilpointer.Int32Ptr(int32(3456)),
	}, {
		Name:     utilpointer.StringPtr("udp-example"),
		Protocol: protoPtr(v1.ProtocolUDP),
		Port:     utilpointer.Int32Ptr(int32(161)),
	}, {
		Name:     utilpointer.StringPtr("tcp-example"),
		Protocol: protoPtr(v1.ProtocolTCP),
		Port:     utilpointer.Int32Ptr(int32(80)),
	}}, slice.Ports)

	assert.ElementsMatch(t, []discovery.Endpoint{{
		Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
		Addresses:  []string{"1234::5678:0000:0000:9abc:def0"},
		TargetRef:  &v1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pod2.Name},
		Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
	}}, slice.Endpoints)
}

// TestPodAddsBatching verifies that endpoint updates caused by pod addition are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodAddsBatching(t *testing.T) {
	type podAdd struct {
		delay time.Duration
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		adds             []podAdd
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three adds with no batching",
			batchPeriod: 0 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three adds in one batch",
			batchPeriod: 1 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three adds in two batches",
			batchPeriod: 1 * time.Second,
			adds: []podAdd{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay: 200 * time.Millisecond,
				},
				{
					delay: 100 * time.Millisecond,
				},
				{
					delay: 1 * time.Second,
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := metav1.NamespaceDefault
			client, esController := newController([]string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			go esController.Run(1, stopCh)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
				},
			})

			for i, add := range tc.adds {
				time.Sleep(add.delay)

				p := newPod(i, ns, true, 0)
				esController.podStore.Add(p)
				esController.addPod(p)
			}

			time.Sleep(tc.finalDelay)
			assert.Len(t, client.Actions(), tc.wantRequestCount)
			// In case of error, make debugging easier.
			for _, action := range client.Actions() {
				t.Logf("action: %v %v", action.GetVerb(), action.GetResource())
			}
		})
	}
}

// TestPodUpdatesBatching verifies that endpoint updates caused by pod updates are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodUpdatesBatching(t *testing.T) {
	resourceVersion := 1
	type podUpdate struct {
		delay   time.Duration
		podName string
		podIP   string
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		podsCount        int
		updates          []podUpdate
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three updates with no batching",
			batchPeriod: 0 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three updates in one batch",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three updates in two batches",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			updates: []podUpdate{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
					podIP:   "10.0.0.0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
					podIP:   "10.0.0.1",
				},
				{
					delay:   1 * time.Second,
					podName: "pod2",
					podIP:   "10.0.0.2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := metav1.NamespaceDefault
			client, esController := newController([]string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			go esController.Run(1, stopCh)

			addPods(t, esController, ns, tc.podsCount)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
				},
			})

			for _, update := range tc.updates {
				time.Sleep(update.delay)

				old, exists, err := esController.podStore.GetByKey(fmt.Sprintf("%s/%s", ns, update.podName))
				if err != nil {
					t.Fatalf("Error while retrieving old value of %q: %v", update.podName, err)
				}
				if !exists {
					t.Fatalf("Pod %q doesn't exist", update.podName)
				}
				oldPod := old.(*v1.Pod)
				newPod := oldPod.DeepCopy()
				newPod.Status.PodIPs[0].IP = update.podIP
				newPod.ResourceVersion = strconv.Itoa(resourceVersion)
				resourceVersion++

				esController.podStore.Update(newPod)
				esController.updatePod(oldPod, newPod)
			}

			time.Sleep(tc.finalDelay)
			assert.Len(t, client.Actions(), tc.wantRequestCount)
			// In case of error, make debugging easier.
			for _, action := range client.Actions() {
				t.Logf("action: %v %v", action.GetVerb(), action.GetResource())
			}
		})
	}
}

// TestPodDeleteBatching verifies that endpoint updates caused by pod deletion are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodDeleteBatching(t *testing.T) {
	type podDelete struct {
		delay   time.Duration
		podName string
	}

	tests := []struct {
		name             string
		batchPeriod      time.Duration
		podsCount        int
		deletes          []podDelete
		finalDelay       time.Duration
		wantRequestCount int
	}{
		{
			name:        "three deletes with no batching",
			batchPeriod: 0 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 3,
		},
		{
			name:        "three deletes in one batch",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 1,
		},
		{
			name:        "three deletes in two batches",
			batchPeriod: 1 * time.Second,
			podsCount:   10,
			deletes: []podDelete{
				{
					// endpoints.Run needs ~100 ms to start processing updates.
					delay:   200 * time.Millisecond,
					podName: "pod0",
				},
				{
					delay:   100 * time.Millisecond,
					podName: "pod1",
				},
				{
					delay:   1 * time.Second,
					podName: "pod2",
				},
			},
			finalDelay:       3 * time.Second,
			wantRequestCount: 2,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ns := metav1.NamespaceDefault
			client, esController := newController([]string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			go esController.Run(1, stopCh)

			addPods(t, esController, ns, tc.podsCount)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
				},
			})

			for _, update := range tc.deletes {
				time.Sleep(update.delay)

				old, exists, err := esController.podStore.GetByKey(fmt.Sprintf("%s/%s", ns, update.podName))
				assert.Nil(t, err, "error while retrieving old value of %q: %v", update.podName, err)
				assert.Equal(t, true, exists, "pod should exist")
				esController.podStore.Delete(old)
				esController.deletePod(old)
			}

			time.Sleep(tc.finalDelay)
			assert.Len(t, client.Actions(), tc.wantRequestCount)
			// In case of error, make debugging easier.
			for _, action := range client.Actions() {
				t.Logf("action: %v %v", action.GetVerb(), action.GetResource())
			}
		})
	}
}

// Test helpers
func addPods(t *testing.T, esController *endpointSliceController, namespace string, podsCount int) {
	t.Helper()
	for i := 0; i < podsCount; i++ {
		pod := newPod(i, namespace, true, 0)
		esController.podStore.Add(pod)
	}
}

func standardSyncService(t *testing.T, esController *endpointSliceController, namespace, serviceName, managedBySetup string) {
	t.Helper()
	createService(t, esController, namespace, serviceName, managedBySetup)

	err := esController.syncService(fmt.Sprintf("%s/%s", namespace, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")
}

func createService(t *testing.T, esController *endpointSliceController, namespace, serviceName, managedBySetup string) *v1.Service {
	t.Helper()
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(time.Now()),
		},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt(80)}},
			Selector: map[string]string{"foo": "bar"},
		},
	}
	esController.serviceStore.Add(service)
	_, err := esController.client.CoreV1().Services(namespace).Create(context.TODO(), service, metav1.CreateOptions{})
	assert.Nil(t, err, "Expected no error creating service")
	return service
}

func expectAction(t *testing.T, actions []k8stesting.Action, index int, verb, resource string) {
	t.Helper()
	if len(actions) <= index {
		t.Fatalf("Expected at least %d actions, got %d", index+1, len(actions))
	}

	action := actions[index]
	if action.GetVerb() != verb {
		t.Errorf("Expected action %d verb to be %s, got %s", index, verb, action.GetVerb())
	}

	if action.GetResource().Resource != resource {
		t.Errorf("Expected action %d resource to be %s, got %s", index, resource, action.GetResource().Resource)
	}
}

// protoPtr takes a Protocol and returns a pointer to it.
func protoPtr(proto v1.Protocol) *v1.Protocol {
	return &proto
}

// cacheMutationCheck helps ensure that cached objects have not been changed
// in any way throughout a test run.
type cacheMutationCheck struct {
	objects []cacheObject
}

// cacheObject stores a reference to an original object as well as a deep copy
// of that object to track any mutations in the original object.
type cacheObject struct {
	original runtime.Object
	deepCopy runtime.Object
}

// newCacheMutationCheck initializes a cacheMutationCheck with EndpointSlices.
func newCacheMutationCheck(endpointSlices []*discovery.EndpointSlice) cacheMutationCheck {
	cmc := cacheMutationCheck{}
	for _, endpointSlice := range endpointSlices {
		cmc.Add(endpointSlice)
	}
	return cmc
}

// Add appends a runtime.Object and a deep copy of that object into the
// cacheMutationCheck.
func (cmc *cacheMutationCheck) Add(o runtime.Object) {
	cmc.objects = append(cmc.objects, cacheObject{
		original: o,
		deepCopy: o.DeepCopyObject(),
	})
}

// Check verifies that no objects in the cacheMutationCheck have been mutated.
func (cmc *cacheMutationCheck) Check(t *testing.T) {
	for _, o := range cmc.objects {
		if !reflect.DeepEqual(o.original, o.deepCopy) {
			// Cached objects can't be safely mutated and instead should be deep
			// copied before changed in any way.
			t.Errorf("Cached object was unexpectedly mutated. Original: %+v, Mutated: %+v", o.deepCopy, o.original)
		}
	}
}
