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
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
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

func newController(nodeNames []string) (*fake.Clientset, *endpointSliceController) {
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
		informerFactory.Discovery().V1alpha1().EndpointSlices(),
		int32(100),
		client)

	esController.nodesSynced = alwaysReady
	esController.podsSynced = alwaysReady
	esController.servicesSynced = alwaysReady
	esController.endpointSlicesSynced = alwaysReady

	return client, &endpointSliceController{
		esController,
		informerFactory.Discovery().V1alpha1().EndpointSlices().Informer().GetStore(),
		informerFactory.Core().V1().Nodes().Informer().GetStore(),
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Services().Informer().GetStore(),
	}
}

// Ensure SyncService for service with no selector results in no action
func TestSyncServiceNoSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	client, esController := newController([]string{"node-1"})
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
	client, esController := newController([]string{"node-1"})
	standardSyncService(t, esController, ns, serviceName)
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	sliceList, err := client.DiscoveryV1alpha1().EndpointSlices(ns).List(metav1.ListOptions{})
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
	client, esController := newController([]string{"node-1"})

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
	client, esController := newController([]string{"node-1"})
	ns := metav1.NamespaceDefault

	pod1 := newPod(1, ns, true, 0)
	esController.podStore.Add(pod1)

	// ensure this pod will not match the selector
	pod2 := newPod(2, ns, true, 0)
	pod2.Labels["foo"] = "boo"
	esController.podStore.Add(pod2)

	standardSyncService(t, esController, ns, "testing-1")
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	// an endpoint slice should be created, it should only reference pod1 (not pod2)
	slices, err := client.DiscoveryV1alpha1().EndpointSlices(ns).List(metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, slices.Items, 1, "Expected 1 endpoint slices")
	slice := slices.Items[0]
	assert.Len(t, slice.Endpoints, 1, "Expected 1 endpoint in first slice")
	assert.NotEmpty(t, slice.Annotations["endpoints.kubernetes.io/last-change-trigger-time"])
	endpoint := slice.Endpoints[0]
	assert.EqualValues(t, endpoint.TargetRef, &v1.ObjectReference{Kind: "Pod", Namespace: ns, Name: pod1.Name})
}

// Ensure SyncService correctly selects EndpointSlices.
func TestSyncServiceEndpointSliceSelection(t *testing.T) {
	client, esController := newController([]string{"node-1"})
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"

	// 3 slices, 2 with matching labels for our service
	endpointSlices := []*discovery.EndpointSlice{{
		ObjectMeta: metav1.ObjectMeta{Name: "matching-1", Namespace: ns, Labels: map[string]string{discovery.LabelServiceName: serviceName}},
	}, {
		ObjectMeta: metav1.ObjectMeta{Name: "matching-2", Namespace: ns, Labels: map[string]string{discovery.LabelServiceName: serviceName}},
	}, {
		ObjectMeta: metav1.ObjectMeta{Name: "not-matching-1", Namespace: ns, Labels: map[string]string{discovery.LabelServiceName: "something-else"}},
	}}

	// need to add them to both store and fake clientset
	for _, endpointSlice := range endpointSlices {
		addErr := esController.endpointSliceStore.Add(endpointSlice)
		assert.Nil(t, addErr, "Expected no error adding EndpointSlice")
		_, err := client.DiscoveryV1alpha1().EndpointSlices(ns).Create(endpointSlice)
		assert.Nil(t, err, "Expected no error creating EndpointSlice")
	}

	numActionsBefore := len(client.Actions())
	standardSyncService(t, esController, ns, serviceName)

	// should only have 2 additional actions
	assert.Len(t, client.Actions(), numActionsBefore+2)

	// only 2 slices should match, 1 of those should be deleted, 1 should be updated as a placeholder
	assert.Equal(t, "update", client.Actions()[numActionsBefore].GetVerb())
	assert.Equal(t, client.Actions()[numActionsBefore].GetResource().Resource, "endpointslices")
	assert.Equal(t, "delete", client.Actions()[numActionsBefore+1].GetVerb())
	assert.Equal(t, client.Actions()[numActionsBefore+1].GetResource().Resource, "endpointslices")
}

// Ensure SyncService handles a variety of protocols and IPs appropriately.
func TestSyncServiceFull(t *testing.T) {
	client, esController := newController([]string{"node-1"})
	namespace := metav1.NamespaceDefault
	serviceName := "all-the-protocols"

	// pod 1 only uses PodIP status attr
	pod1 := newPod(1, namespace, true, 0)
	pod1.Status.PodIP = "1.2.3.4"
	pod1.Status.PodIPs = []v1.PodIP{}
	esController.podStore.Add(pod1)

	// pod 2 only uses PodIPs status attr
	pod2 := newPod(2, namespace, true, 0)
	pod2.Status.PodIP = ""
	pod2.Status.PodIPs = []v1.PodIP{{IP: "1.2.3.5"}, {IP: "1234::5678:0000:0000:9abc:def0"}}
	esController.podStore.Add(pod2)

	// create service with all protocols and multiple ports
	serviceCreateTime := time.Now()
	esController.serviceStore.Add(&v1.Service{
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
		},
	})

	// run through full sync service loop
	err := esController.syncService(fmt.Sprintf("%s/%s", namespace, serviceName))
	assert.Nil(t, err)

	// should only have 1 action - to create endpoint slice
	assert.Len(t, client.Actions(), 1)
	expectActions(t, client.Actions(), 1, "create", "endpointslices")
	sliceList, err := client.DiscoveryV1alpha1().EndpointSlices(namespace).List(metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, sliceList.Items, 1, "Expected 1 endpoint slices")

	// ensure all attributes of endpoint slice match expected state
	slice := sliceList.Items[0]
	assert.Len(t, slice.Endpoints, 2, "Expected 2 endpoints in first slice")
	assert.Equal(t, slice.Annotations["endpoints.kubernetes.io/last-change-trigger-time"], serviceCreateTime.Format(time.RFC3339Nano))
	assert.EqualValues(t, []discovery.EndpointPort{{
		Name:     strPtr("tcp-example"),
		Protocol: protoPtr(v1.ProtocolTCP),
		Port:     int32Ptr(int32(80)),
	}, {
		Name:     strPtr("udp-example"),
		Protocol: protoPtr(v1.ProtocolUDP),
		Port:     int32Ptr(int32(161)),
	}, {
		Name:     strPtr("sctp-example"),
		Protocol: protoPtr(v1.ProtocolSCTP),
		Port:     int32Ptr(int32(3456)),
	}}, slice.Ports)
	assert.ElementsMatch(t, []discovery.Endpoint{{
		Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
		Addresses:  []string{"1.2.3.4"},
		TargetRef:  &v1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pod1.Name},
		Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
	}, {
		Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
		Addresses:  []string{"1.2.3.5", "1234::5678:0000:0000:9abc:def0"},
		TargetRef:  &v1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pod2.Name},
		Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
	}}, slice.Endpoints)
}

// Test helpers

func standardSyncService(t *testing.T, esController *endpointSliceController, namespace, serviceName string) {
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(time.Now()),
		},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt(80)}},
			Selector: map[string]string{"foo": "bar"},
		},
	})

	err := esController.syncService(fmt.Sprintf("%s/%s", namespace, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")
}

func strPtr(str string) *string {
	return &str
}

func protoPtr(proto v1.Protocol) *v1.Protocol {
	return &proto
}

func int32Ptr(num int32) *int32 {
	return &num
}
