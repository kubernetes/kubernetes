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
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/endpointslice/topologycache"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/controller"
	endpointslicepkg "k8s.io/kubernetes/pkg/controller/util/endpointslice"
	"k8s.io/utils/pointer"
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

func newController(t *testing.T, nodeNames []string, batchPeriod time.Duration) (*fake.Clientset, *endpointSliceController) {
	client := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	nodeInformer := informerFactory.Core().V1().Nodes()
	indexer := nodeInformer.Informer().GetIndexer()
	for _, nodeName := range nodeNames {
		indexer.Add(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	}

	esInformer := informerFactory.Discovery().V1().EndpointSlices()
	esIndexer := esInformer.Informer().GetIndexer()

	// These reactors are required to mock functionality that would be covered
	// automatically if we weren't using the fake client.
	client.PrependReactor("create", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)

		if endpointSlice.ObjectMeta.GenerateName != "" {
			endpointSlice.ObjectMeta.Name = fmt.Sprintf("%s-%s", endpointSlice.ObjectMeta.GenerateName, rand.String(8))
			endpointSlice.ObjectMeta.GenerateName = ""
		}
		endpointSlice.Generation = 1
		esIndexer.Add(endpointSlice)

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.Generation++
		esIndexer.Update(endpointSlice)

		return false, endpointSlice, nil
	}))

	_, ctx := ktesting.NewTestContext(t)
	esController := NewController(
		ctx,
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().Services(),
		nodeInformer,
		esInformer,
		int32(100),
		client,
		batchPeriod)

	esController.nodesSynced = alwaysReady
	esController.podsSynced = alwaysReady
	esController.servicesSynced = alwaysReady
	esController.endpointSlicesSynced = alwaysReady

	return client, &endpointSliceController{
		esController,
		informerFactory.Discovery().V1().EndpointSlices().Informer().GetStore(),
		informerFactory.Core().V1().Nodes().Informer().GetStore(),
		informerFactory.Core().V1().Pods().Informer().GetStore(),
		informerFactory.Core().V1().Services().Informer().GetStore(),
	}
}

func newPod(n int, namespace string, ready bool, nPorts int, terminating bool) *v1.Pod {
	status := v1.ConditionTrue
	if !ready {
		status = v1.ConditionFalse
	}

	var deletionTimestamp *metav1.Time
	if terminating {
		deletionTimestamp = &metav1.Time{
			Time: time.Now(),
		}
	}

	p := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         namespace,
			Name:              fmt.Sprintf("pod%d", n),
			Labels:            map[string]string{"foo": "bar"},
			DeletionTimestamp: deletionTimestamp,
			ResourceVersion:   fmt.Sprint(n),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{
				Name: "container-1",
			}},
			NodeName: "node-1",
		},
		Status: v1.PodStatus{
			PodIP: fmt.Sprintf("1.2.3.%d", 4+n),
			PodIPs: []v1.PodIP{{
				IP: fmt.Sprintf("1.2.3.%d", 4+n),
			}},
			Conditions: []v1.PodCondition{
				{
					Type:   v1.PodReady,
					Status: status,
				},
			},
		},
	}

	return p
}

func expectActions(t *testing.T, actions []k8stesting.Action, num int, verb, resource string) {
	t.Helper()
	// if actions are less the below logic will panic
	if num > len(actions) {
		t.Fatalf("len of actions %v is unexpected. Expected to be at least %v", len(actions), num+1)
	}

	for i := 0; i < num; i++ {
		relativePos := len(actions) - i - 1
		assert.Equal(t, verb, actions[relativePos].GetVerb(), "Expected action -%d verb to be %s", i, verb)
		assert.Equal(t, resource, actions[relativePos].GetResource().Resource, "Expected action -%d resource to be %s", i, resource)
	}
}

// Ensure SyncService for service with no selector results in no action
func TestSyncServiceNoSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: ns},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{TargetPort: intstr.FromInt32(80)}},
		},
	})

	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
	assert.NoError(t, err)
	assert.Len(t, client.Actions(), 0)
}

func TestServiceExternalNameTypeSync(t *testing.T) {
	serviceName := "testing-1"
	namespace := metav1.NamespaceDefault

	testCases := []struct {
		desc    string
		service *v1.Service
	}{
		{
			desc: "External name with selector and ports should not receive endpoint slices",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{Port: 80}},
					Type:     v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name with ports should not receive endpoint slices",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{Port: 80}},
					Type:  v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name with selector should not receive endpoint slices",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Type:     v1.ServiceTypeExternalName,
				},
			},
		},
		{
			desc: "External name without selector and ports should not receive endpoint slices",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: namespace},
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeExternalName,
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			client, esController := newController(t, []string{"node-1"}, time.Duration(0))
			logger, _ := ktesting.NewTestContext(t)

			pod := newPod(1, namespace, true, 0, false)
			err := esController.podStore.Add(pod)
			assert.NoError(t, err)

			err = esController.serviceStore.Add(tc.service)
			assert.NoError(t, err)

			err = esController.syncService(logger, fmt.Sprintf("%s/%s", namespace, serviceName))
			assert.NoError(t, err)
			assert.Len(t, client.Actions(), 0)

			sliceList, err := client.DiscoveryV1().EndpointSlices(namespace).List(context.TODO(), metav1.ListOptions{})
			assert.NoError(t, err)
			assert.Len(t, sliceList.Items, 0, "Expected 0 endpoint slices")
		})
	}
}

// Ensure SyncService for service with pending deletion results in no action
func TestSyncServicePendingDeletion(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	deletionTimestamp := metav1.Now()
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: ns, DeletionTimestamp: &deletionTimestamp},
		Spec: v1.ServiceSpec{
			Selector: map[string]string{"foo": "bar"},
			Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt32(80)}},
		},
	})

	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
	assert.NoError(t, err)
	assert.Len(t, client.Actions(), 0)
}

// Ensure SyncService for service with selector but no pods results in placeholder EndpointSlice
func TestSyncServiceWithSelector(t *testing.T) {
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	standardSyncService(t, esController, ns, serviceName)
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	sliceList, err := client.DiscoveryV1().EndpointSlices(ns).List(context.TODO(), metav1.ListOptions{})
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
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))

	// Build up existing service
	existingServiceName := "stillthere"
	existingServiceKey := endpointsliceutil.ServiceKey{Name: existingServiceName, Namespace: namespace}
	esController.triggerTimeTracker.ServiceStates[existingServiceKey] = endpointsliceutil.ServiceState{}
	esController.serviceStore.Add(&v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: existingServiceName, Namespace: namespace},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt32(80)}},
			Selector: map[string]string{"foo": "bar"},
		},
	})

	// Add missing service to triggerTimeTracker to ensure the reference is cleaned up
	missingServiceName := "notthere"
	missingServiceKey := endpointsliceutil.ServiceKey{Name: missingServiceName, Namespace: namespace}
	esController.triggerTimeTracker.ServiceStates[missingServiceKey] = endpointsliceutil.ServiceState{}

	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", namespace, missingServiceName))

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
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault

	pod1 := newPod(1, ns, true, 0, false)
	esController.podStore.Add(pod1)

	// ensure this pod will not match the selector
	pod2 := newPod(2, ns, true, 0, false)
	pod2.Labels["foo"] = "boo"
	esController.podStore.Add(pod2)

	standardSyncService(t, esController, ns, "testing-1")
	expectActions(t, client.Actions(), 1, "create", "endpointslices")

	// an endpoint slice should be created, it should only reference pod1 (not pod2)
	slices, err := client.DiscoveryV1().EndpointSlices(ns).List(context.TODO(), metav1.ListOptions{})
	assert.Nil(t, err, "Expected no error fetching endpoint slices")
	assert.Len(t, slices.Items, 1, "Expected 1 endpoint slices")
	slice := slices.Items[0]
	assert.Len(t, slice.Endpoints, 1, "Expected 1 endpoint in first slice")
	assert.NotEmpty(t, slice.Annotations[v1.EndpointsLastChangeTriggerTime])
	endpoint := slice.Endpoints[0]
	assert.EqualValues(t, endpoint.TargetRef, &v1.ObjectReference{Kind: "Pod", Namespace: ns, Name: pod1.Name})
}

func TestSyncServiceEndpointSlicePendingDeletion(t *testing.T) {
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	service := createService(t, esController, ns, serviceName)
	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(service, gvk)

	deletedTs := metav1.Now()
	endpointSlice := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "epSlice-1",
			Namespace:       ns,
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   controllerName,
			},
			DeletionTimestamp: &deletedTs,
		},
		AddressType: discovery.AddressTypeIPv4,
	}
	err = esController.endpointSliceStore.Add(endpointSlice)
	if err != nil {
		t.Fatalf("Expected no error adding EndpointSlice: %v", err)
	}
	_, err = client.DiscoveryV1().EndpointSlices(ns).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Expected no error creating EndpointSlice: %v", err)
	}

	logger, _ = ktesting.NewTestContext(t)
	numActionsBefore := len(client.Actions())
	err = esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")

	// The EndpointSlice marked for deletion should be ignored by the controller, and thus
	// should not result in any action.
	if len(client.Actions()) != numActionsBefore {
		t.Errorf("Expected 0 more actions, got %d", len(client.Actions())-numActionsBefore)
	}
}

// Ensure SyncService correctly selects and labels EndpointSlices.
func TestSyncServiceEndpointSliceLabelSelection(t *testing.T) {
	client, esController := newController(t, []string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	service := createService(t, esController, ns, serviceName)

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(service, gvk)

	// 5 slices, 3 with matching labels for our service
	endpointSlices := []*discovery.EndpointSlice{{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "matching-1",
			Namespace:       ns,
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   controllerName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}, {
		ObjectMeta: metav1.ObjectMeta{
			Name:            "matching-2",
			Namespace:       ns,
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
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
		_, err = client.DiscoveryV1().EndpointSlices(ns).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Expected no error creating EndpointSlice: %v", err)
		}
	}

	numActionsBefore := len(client.Actions())
	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")

	if len(client.Actions()) != numActionsBefore+2 {
		t.Errorf("Expected 2 more actions, got %d", len(client.Actions())-numActionsBefore)
	}

	// only 2 slices should match, 2 should be deleted, 1 should be updated as a placeholder
	expectAction(t, client.Actions(), numActionsBefore, "update", "endpointslices")
	expectAction(t, client.Actions(), numActionsBefore+1, "delete", "endpointslices")

	// ensure cache mutation has not occurred
	cmc.Check(t)
}

func TestOnEndpointSliceUpdate(t *testing.T) {
	_, esController := newController(t, []string{"node-1"}, time.Duration(0))
	ns := metav1.NamespaceDefault
	serviceName := "testing-1"
	epSlice1 := &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "matching-1",
			Namespace: ns,
			Labels: map[string]string{
				discovery.LabelServiceName: serviceName,
				discovery.LabelManagedBy:   controllerName,
			},
		},
		AddressType: discovery.AddressTypeIPv4,
	}

	logger, _ := ktesting.NewTestContext(t)
	epSlice2 := epSlice1.DeepCopy()
	epSlice2.Labels[discovery.LabelManagedBy] = "something else"

	assert.Equal(t, 0, esController.queue.Len())
	esController.onEndpointSliceUpdate(logger, epSlice1, epSlice2)
	err := wait.PollImmediate(100*time.Millisecond, 3*time.Second, func() (bool, error) {
		if esController.queue.Len() > 0 {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("unexpected error waiting for add to queue")
	}
	assert.Equal(t, 1, esController.queue.Len())
}

func TestSyncService(t *testing.T) {
	creationTimestamp := metav1.Now()
	deletionTimestamp := metav1.Now()

	testcases := []struct {
		name                  string
		service               *v1.Service
		pods                  []*v1.Pod
		expectedEndpointPorts []discovery.EndpointPort
		expectedEndpoints     []discovery.Endpoint
	}{
		{
			name: "pods with multiple IPs and Service with ipFamilies=ipv4",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.2",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.2",
							},
							{
								IP: "fd08::5678:0000:0000:9abc:def0",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.String("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32(3456),
				},
				{
					Name:     pointer.String("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32(161),
				},
				{
					Name:     pointer.String("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32(80),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(true),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod0"},
					NodeName:  pointer.String("node-1"),
				},
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(true),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(false),
					},
					Addresses: []string{"10.0.0.2"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.String("node-1"),
				},
			},
		},
		{
			name: "pods with multiple IPs and Service with ipFamilies=ipv6",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv6Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.2",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.2",
							},
							{
								IP: "fd08::5678:0000:0000:9abc:def0",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.String("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32(3456),
				},
				{
					Name:     pointer.String("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32(161),
				},
				{
					Name:     pointer.String("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32(80),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(true),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(false),
					},
					Addresses: []string{"fd08::5678:0000:0000:9abc:def0"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.String("node-1"),
				},
			},
		},
		{
			name: "Terminating pods",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					// one ready pod for comparison
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: &deletionTimestamp,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.2",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.2",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.String("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32(3456),
				},
				{
					Name:     pointer.String("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32(161),
				},
				{
					Name:     pointer.String("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32(80),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(true),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod0"},
					NodeName:  pointer.String("node-1"),
				},
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(false),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(true),
					},
					Addresses: []string{"10.0.0.2"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.String("node-1"),
				},
			},
		},
		{
			name: "Not ready terminating pods",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					// one ready pod for comparison
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: &deletionTimestamp,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.2",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.2",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodReady,
								Status: v1.ConditionFalse,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.String("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32(3456),
				},
				{
					Name:     pointer.String("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32(161),
				},
				{
					Name:     pointer.String("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32(80),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(true),
						Serving:     pointer.Bool(true),
						Terminating: pointer.Bool(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod0"},
					NodeName:  pointer.String("node-1"),
				},
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.Bool(false),
						Serving:     pointer.Bool(false),
						Terminating: pointer.Bool(true),
					},
					Addresses: []string{"10.0.0.2"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.String("node-1"),
				},
			},
		},
		{
			name: "Ready and Complete pods with same IPs",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodInitialized,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.ContainersReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.1",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodInitialized,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.PodReady,
								Status: v1.ConditionFalse,
							},
							{
								Type:   v1.ContainersReady,
								Status: v1.ConditionFalse,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.StringPtr("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32Ptr(int32(3456)),
				},
				{
					Name:     pointer.StringPtr("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32Ptr(int32(161)),
				},
				{
					Name:     pointer.StringPtr("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32Ptr(int32(80)),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.BoolPtr(true),
						Serving:     pointer.BoolPtr(true),
						Terminating: pointer.BoolPtr(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod0"},
					NodeName:  pointer.StringPtr("node-1"),
				},
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.BoolPtr(false),
						Serving:     pointer.BoolPtr(false),
						Terminating: pointer.BoolPtr(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.StringPtr("node-1"),
				},
			},
		},
		{
			// Any client reading EndpointSlices already has to handle deduplicating endpoints by IP address.
			// If 2 pods are ready, something has gone wrong further up the stack, we shouldn't try to hide that.
			name: "Two Ready pods with same IPs",
			service: &v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:              "foobar",
					Namespace:         "default",
					CreationTimestamp: creationTimestamp,
				},
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{
						{Name: "tcp-example", TargetPort: intstr.FromInt32(80), Protocol: v1.ProtocolTCP},
						{Name: "udp-example", TargetPort: intstr.FromInt32(161), Protocol: v1.ProtocolUDP},
						{Name: "sctp-example", TargetPort: intstr.FromInt32(3456), Protocol: v1.ProtocolSCTP},
					},
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod0",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{{
							IP: "10.0.0.1",
						}},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodInitialized,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.ContainersReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{
						Namespace:         "default",
						Name:              "pod1",
						Labels:            map[string]string{"foo": "bar"},
						DeletionTimestamp: nil,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name: "container-1",
						}},
						NodeName: "node-1",
					},
					Status: v1.PodStatus{
						PodIP: "10.0.0.1",
						PodIPs: []v1.PodIP{
							{
								IP: "10.0.0.1",
							},
						},
						Conditions: []v1.PodCondition{
							{
								Type:   v1.PodInitialized,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.PodReady,
								Status: v1.ConditionTrue,
							},
							{
								Type:   v1.ContainersReady,
								Status: v1.ConditionTrue,
							},
						},
					},
				},
			},
			expectedEndpointPorts: []discovery.EndpointPort{
				{
					Name:     pointer.StringPtr("sctp-example"),
					Protocol: protoPtr(v1.ProtocolSCTP),
					Port:     pointer.Int32Ptr(int32(3456)),
				},
				{
					Name:     pointer.StringPtr("udp-example"),
					Protocol: protoPtr(v1.ProtocolUDP),
					Port:     pointer.Int32Ptr(int32(161)),
				},
				{
					Name:     pointer.StringPtr("tcp-example"),
					Protocol: protoPtr(v1.ProtocolTCP),
					Port:     pointer.Int32Ptr(int32(80)),
				},
			},
			expectedEndpoints: []discovery.Endpoint{
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.BoolPtr(true),
						Serving:     pointer.BoolPtr(true),
						Terminating: pointer.BoolPtr(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod0"},
					NodeName:  pointer.StringPtr("node-1"),
				},
				{
					Conditions: discovery.EndpointConditions{
						Ready:       pointer.BoolPtr(true),
						Serving:     pointer.BoolPtr(true),
						Terminating: pointer.BoolPtr(false),
					},
					Addresses: []string{"10.0.0.1"},
					TargetRef: &v1.ObjectReference{Kind: "Pod", Namespace: "default", Name: "pod1"},
					NodeName:  pointer.StringPtr("node-1"),
				},
			},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			client, esController := newController(t, []string{"node-1"}, time.Duration(0))

			for _, pod := range testcase.pods {
				esController.podStore.Add(pod)
			}
			esController.serviceStore.Add(testcase.service)

			_, err := esController.client.CoreV1().Services(testcase.service.Namespace).Create(context.TODO(), testcase.service, metav1.CreateOptions{})
			assert.Nil(t, err, "Expected no error creating service")

			logger, _ := ktesting.NewTestContext(t)
			err = esController.syncService(logger, fmt.Sprintf("%s/%s", testcase.service.Namespace, testcase.service.Name))
			assert.Nil(t, err)

			// last action should be to create endpoint slice
			expectActions(t, client.Actions(), 1, "create", "endpointslices")
			sliceList, err := client.DiscoveryV1().EndpointSlices(testcase.service.Namespace).List(context.TODO(), metav1.ListOptions{})
			assert.Nil(t, err, "Expected no error fetching endpoint slices")
			assert.Len(t, sliceList.Items, 1, "Expected 1 endpoint slices")

			// ensure all attributes of endpoint slice match expected state
			slice := sliceList.Items[0]
			assert.Equal(t, slice.Annotations[v1.EndpointsLastChangeTriggerTime], creationTimestamp.UTC().Format(time.RFC3339Nano))
			assert.ElementsMatch(t, testcase.expectedEndpointPorts, slice.Ports)
			assert.ElementsMatch(t, testcase.expectedEndpoints, slice.Endpoints)
		})
	}
}

// TestPodAddsBatching verifies that endpoint updates caused by pod addition are batched together.
// This test uses real time.Sleep, as there is no easy way to mock time in endpoints controller now.
// TODO(mborsz): Migrate this test to mock clock when possible.
func TestPodAddsBatching(t *testing.T) {
	t.Parallel()

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
			client, esController := newController(t, []string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			_, ctx := ktesting.NewTestContext(t)
			go esController.Run(ctx, 1)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
					Ports:      []v1.ServicePort{{Port: 80}},
				},
			})

			for i, add := range tc.adds {
				time.Sleep(add.delay)

				p := newPod(i, ns, true, 0, false)
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
	t.Parallel()

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
			client, esController := newController(t, []string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			_, ctx := ktesting.NewTestContext(t)
			go esController.Run(ctx, 1)

			addPods(t, esController, ns, tc.podsCount)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
					Ports:      []v1.ServicePort{{Port: 80}},
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
	t.Parallel()

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
			client, esController := newController(t, []string{"node-1"}, tc.batchPeriod)
			stopCh := make(chan struct{})
			defer close(stopCh)

			_, ctx := ktesting.NewTestContext(t)
			go esController.Run(ctx, 1)

			addPods(t, esController, ns, tc.podsCount)

			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
					Ports:      []v1.ServicePort{{Port: 80}},
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

func TestSyncServiceStaleInformer(t *testing.T) {
	testcases := []struct {
		name                     string
		informerGenerationNumber int64
		trackerGenerationNumber  int64
		expectError              bool
	}{
		{
			name:                     "informer cache outdated",
			informerGenerationNumber: 10,
			trackerGenerationNumber:  12,
			expectError:              true,
		},
		{
			name:                     "cache and tracker synced",
			informerGenerationNumber: 10,
			trackerGenerationNumber:  10,
			expectError:              false,
		},
		{
			name:                     "tracker outdated",
			informerGenerationNumber: 10,
			trackerGenerationNumber:  1,
			expectError:              false,
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			_, esController := newController(t, []string{"node-1"}, time.Duration(0))
			ns := metav1.NamespaceDefault
			serviceName := "testing-1"

			// Store Service in the cache
			esController.serviceStore.Add(&v1.Service{
				ObjectMeta: metav1.ObjectMeta{Name: serviceName, Namespace: ns},
				Spec: v1.ServiceSpec{
					Selector: map[string]string{"foo": "bar"},
					Ports:    []v1.ServicePort{{TargetPort: intstr.FromInt32(80)}},
				},
			})

			// Create EndpointSlice in the informer cache with informerGenerationNumber
			epSlice1 := &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "matching-1",
					Namespace:  ns,
					Generation: testcase.informerGenerationNumber,
					Labels: map[string]string{
						discovery.LabelServiceName: serviceName,
						discovery.LabelManagedBy:   controllerName,
					},
				},
				AddressType: discovery.AddressTypeIPv4,
			}
			err := esController.endpointSliceStore.Add(epSlice1)
			if err != nil {
				t.Fatalf("Expected no error adding EndpointSlice: %v", err)
			}

			// Create EndpointSlice in the tracker with trackerGenerationNumber
			epSlice2 := epSlice1.DeepCopy()
			epSlice2.Generation = testcase.trackerGenerationNumber
			esController.endpointSliceTracker.Update(epSlice2)

			logger, _ := ktesting.NewTestContext(t)
			err = esController.syncService(logger, fmt.Sprintf("%s/%s", ns, serviceName))
			// Check if we got a StaleInformerCache error
			if endpointslicepkg.IsStaleInformerCacheErr(err) != testcase.expectError {
				t.Fatalf("Expected error because informer cache is outdated")
			}

		})
	}
}

func Test_checkNodeTopologyDistribution(t *testing.T) {
	zoneA := "zone-a"
	zoneB := "zone-b"
	zoneC := "zone-c"

	readyTrue := true
	readyFalse := false

	cpu100 := resource.MustParse("100m")
	cpu1000 := resource.MustParse("1000m")
	cpu2000 := resource.MustParse("2000m")

	type nodeInfo struct {
		zoneLabel *string
		ready     *bool
		cpu       *resource.Quantity
	}

	testCases := []struct {
		name                 string
		nodes                []nodeInfo
		topologyCacheEnabled bool
		endpointZoneInfo     map[string]topologycache.EndpointZoneInfo
		expectedQueueLen     int
	}{{
		name:                 "empty",
		nodes:                []nodeInfo{},
		topologyCacheEnabled: false,
		endpointZoneInfo:     map[string]topologycache.EndpointZoneInfo{},
		expectedQueueLen:     0,
	}, {
		name: "lopsided, queue required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneA, ready: &readyTrue, cpu: &cpu100},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu1000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneA: 1, zoneB: 2, zoneC: 3},
		},
		expectedQueueLen: 1,
	}, {
		name: "lopsided but 1 unready, queue required because unready node means 0 CPU in one zone",
		nodes: []nodeInfo{
			{zoneLabel: &zoneA, ready: &readyFalse, cpu: &cpu100},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu1000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneA: 1, zoneB: 2, zoneC: 3},
		},
		expectedQueueLen: 1,
	}, {
		name: "even zones, uneven endpoint distribution but within threshold, no sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneB: 5, zoneC: 4},
		},
		expectedQueueLen: 0,
	}, {
		name: "even zones but node missing zone, sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneB: 5, zoneC: 4},
		},
		expectedQueueLen: 1,
	}, {
		name: "even zones but node missing cpu, sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneB, ready: &readyTrue},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneB: 5, zoneC: 4},
		},
		expectedQueueLen: 1,
	}, {
		name: "even zones, uneven endpoint distribution beyond threshold, no sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu2000},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneB: 6, zoneC: 4},
		},
		expectedQueueLen: 1,
	}, {
		name: "3 uneven zones, matching endpoint distribution, no sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneA, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu1000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu100},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneA: 20, zoneB: 10, zoneC: 1},
		},
		expectedQueueLen: 0,
	}, {
		name: "3 uneven zones, endpoint distribution within threshold but below 1, sync required",
		nodes: []nodeInfo{
			{zoneLabel: &zoneA, ready: &readyTrue, cpu: &cpu2000},
			{zoneLabel: &zoneB, ready: &readyTrue, cpu: &cpu1000},
			{zoneLabel: &zoneC, ready: &readyTrue, cpu: &cpu100},
		},
		topologyCacheEnabled: true,
		endpointZoneInfo: map[string]topologycache.EndpointZoneInfo{
			"ns/svc1": {zoneA: 20, zoneB: 10, zoneC: 0},
		},
		expectedQueueLen: 1,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			_, esController := newController(t, []string{}, time.Duration(0))

			for i, nodeInfo := range tc.nodes {
				node := &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("node-%d", i)},
					Status:     v1.NodeStatus{},
				}
				if nodeInfo.zoneLabel != nil {
					node.Labels = map[string]string{v1.LabelTopologyZone: *nodeInfo.zoneLabel}
				}
				if nodeInfo.ready != nil {
					status := v1.ConditionFalse
					if *nodeInfo.ready {
						status = v1.ConditionTrue
					}
					node.Status.Conditions = []v1.NodeCondition{{
						Type:   v1.NodeReady,
						Status: status,
					}}
				}
				if nodeInfo.cpu != nil {
					node.Status.Allocatable = v1.ResourceList{
						v1.ResourceCPU: *nodeInfo.cpu,
					}
				}
				esController.nodeStore.Add(node)
				if tc.topologyCacheEnabled {
					esController.topologyCache = topologycache.NewTopologyCache()
					for serviceKey, endpointZoneInfo := range tc.endpointZoneInfo {
						esController.topologyCache.SetHints(serviceKey, discovery.AddressTypeIPv4, endpointZoneInfo)
					}
				}
			}

			logger, _ := ktesting.NewTestContext(t)
			esController.checkNodeTopologyDistribution(logger)

			if esController.queue.Len() != tc.expectedQueueLen {
				t.Errorf("Expected %d services to be queued, got %d", tc.expectedQueueLen, esController.queue.Len())
			}
		})
	}
}

func TestUpdateNode(t *testing.T) {
	nodeReadyStatus := v1.NodeStatus{
		Allocatable: map[v1.ResourceName]resource.Quantity{
			v1.ResourceCPU: resource.MustParse("100m"),
		},
		Conditions: []v1.NodeCondition{
			{
				Type:   v1.NodeReady,
				Status: v1.ConditionTrue,
			},
		},
	}
	_, esController := newController(t, nil, time.Duration(0))
	sliceInfo := &topologycache.SliceInfo{
		ServiceKey:  "ns/svc",
		AddressType: discovery.AddressTypeIPv4,
		ToCreate: []*discovery.EndpointSlice{
			{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "svc-abc",
					Namespace: "ns",
					Labels: map[string]string{
						discovery.LabelServiceName: "svc",
						discovery.LabelManagedBy:   controllerName,
					},
				},
				Endpoints: []discovery.Endpoint{
					{
						Addresses:  []string{"172.18.0.2"},
						Zone:       pointer.String("zone-a"),
						Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
					},
					{
						Addresses:  []string{"172.18.1.2"},
						Zone:       pointer.String("zone-b"),
						Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
					},
				},
				AddressType: discovery.AddressTypeIPv4,
			},
		},
	}
	node1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-1"},
		Status:     nodeReadyStatus,
	}
	node2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "node-2"},
		Status:     nodeReadyStatus,
	}
	logger, _ := ktesting.NewTestContext(t)
	esController.nodeStore.Add(node1)
	esController.nodeStore.Add(node2)
	esController.addNode(logger, node1)
	esController.addNode(logger, node2)
	// The Nodes don't have the zone label, AddHints should fail.
	_, _, eventsBuilders := esController.topologyCache.AddHints(logger, sliceInfo)
	require.Len(t, eventsBuilders, 1)
	assert.Contains(t, eventsBuilders[0].Message, topologycache.InsufficientNodeInfo)

	updateNode1 := node1.DeepCopy()
	updateNode1.Labels = map[string]string{v1.LabelTopologyZone: "zone-a"}
	updateNode2 := node2.DeepCopy()
	updateNode2.Labels = map[string]string{v1.LabelTopologyZone: "zone-b"}

	// After adding the zone label to the Nodes and calling the event handler updateNode, AddHints should succeed.
	esController.nodeStore.Update(updateNode1)
	esController.nodeStore.Update(updateNode2)
	esController.updateNode(logger, node1, updateNode1)
	esController.updateNode(logger, node2, updateNode2)
	_, _, eventsBuilders = esController.topologyCache.AddHints(logger, sliceInfo)
	require.Len(t, eventsBuilders, 1)
	assert.Contains(t, eventsBuilders[0].Message, topologycache.TopologyAwareHintsEnabled)
}

// Test helpers
func addPods(t *testing.T, esController *endpointSliceController, namespace string, podsCount int) {
	t.Helper()
	for i := 0; i < podsCount; i++ {
		pod := newPod(i, namespace, true, 0, false)
		esController.podStore.Add(pod)
	}
}

func standardSyncService(t *testing.T, esController *endpointSliceController, namespace, serviceName string) {
	t.Helper()
	createService(t, esController, namespace, serviceName)

	logger, _ := ktesting.NewTestContext(t)
	err := esController.syncService(logger, fmt.Sprintf("%s/%s", namespace, serviceName))
	assert.Nil(t, err, "Expected no error syncing service")
}

func createService(t *testing.T, esController *endpointSliceController, namespace, serviceName string) *v1.Service {
	t.Helper()
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:              serviceName,
			Namespace:         namespace,
			CreationTimestamp: metav1.NewTime(time.Now()),
			UID:               types.UID(namespace + "-" + serviceName),
		},
		Spec: v1.ServiceSpec{
			Ports:      []v1.ServicePort{{TargetPort: intstr.FromInt32(80)}},
			Selector:   map[string]string{"foo": "bar"},
			IPFamilies: []v1.IPFamily{v1.IPv4Protocol},
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

func Test_dropEndpointSlicesPendingDeletion(t *testing.T) {
	now := metav1.Now()
	endpointSlices := []*discovery.EndpointSlice{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name:              "epSlice1",
				DeletionTimestamp: &now,
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "epSlice2",
			},
			AddressType: discovery.AddressTypeIPv4,
			Endpoints: []discovery.Endpoint{
				{
					Addresses: []string{"172.18.0.2"},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "epSlice3",
			},
			AddressType: discovery.AddressTypeIPv6,
			Endpoints: []discovery.Endpoint{
				{
					Addresses: []string{"3001:0da8:75a3:0000:0000:8a2e:0370:7334"},
				},
			},
		},
	}

	epSlice2 := endpointSlices[1]
	epSlice3 := endpointSlices[2]

	result := dropEndpointSlicesPendingDeletion(endpointSlices)

	assert.Len(t, result, 2)
	for _, epSlice := range result {
		if epSlice.Name == "epSlice1" {
			t.Errorf("Expected EndpointSlice marked for deletion to be dropped.")
		}
	}

	// We don't use endpointSlices and instead check manually for equality, because
	// `dropEndpointSlicesPendingDeletion` mutates the slice it receives, so it's easy
	// to break this test later. This way, we can be absolutely sure that the result
	// has exactly what we expect it to.
	if !reflect.DeepEqual(epSlice2, result[0]) {
		t.Errorf("EndpointSlice was unexpectedly mutated. Expected: %+v, Mutated: %+v", epSlice2, result[0])
	}
	if !reflect.DeepEqual(epSlice3, result[1]) {
		t.Errorf("EndpointSlice was unexpectedly mutated. Expected: %+v, Mutated: %+v", epSlice3, result[1])
	}
}
