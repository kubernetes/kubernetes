/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
	klogtesting "k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestNewServicesSourceApi_UpdatesAndMultipleServices(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	service1v1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}}}
	service1v2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s1"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}}}
	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "s2"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 30}}}}

	// Setup fake api client.
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	defer fakeWatch.Close()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	handler := NewServiceHandlerMock()

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	serviceConfig := NewServiceConfig(ctx, sharedInformers.Core().V1().Services(), time.Minute)
	serviceConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go serviceConfig.Run(stopCh)

	// Add the first service
	fakeWatch.Add(service1v1)
	handler.ValidateServices(t, []*v1.Service{service1v1})

	// Add another service
	fakeWatch.Add(service2)
	handler.ValidateServices(t, []*v1.Service{service1v1, service2})

	// Modify service1
	fakeWatch.Modify(service1v2)
	handler.ValidateServices(t, []*v1.Service{service1v2, service2})

	// Delete service1
	fakeWatch.Delete(service1v2)
	handler.ValidateServices(t, []*v1.Service{service2})

	// Delete service2
	fakeWatch.Delete(service2)
	handler.ValidateServices(t, []*v1.Service{})
}

func TestNewEndpointsSourceApi_UpdatesAndMultipleEndpoints(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	endpoints1v1 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{
				"1.2.3.4",
			},
		}},
		Ports: []discoveryv1.EndpointPort{{
			Port:     ptr.To[int32](8080),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
	}
	endpoints1v2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "e1"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{
				"1.2.3.4",
				"4.3.2.1",
			},
		}},
		Ports: []discoveryv1.EndpointPort{{
			Port:     ptr.To[int32](8080),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
	}
	endpoints2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "e2"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{
				"5.6.7.8",
			},
		}},
		Ports: []discoveryv1.EndpointPort{{
			Port:     ptr.To[int32](8080),
			Protocol: ptr.To(v1.ProtocolTCP),
		}},
	}

	// Setup fake api client.
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	defer fakeWatch.Close()
	client.PrependWatchReactor("endpointslices", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	handler := NewEndpointSliceHandlerMock()

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	endpointsliceConfig := NewEndpointSliceConfig(ctx, sharedInformers.Discovery().V1().EndpointSlices(), time.Minute)
	endpointsliceConfig.RegisterEventHandler(handler)
	go sharedInformers.Start(stopCh)
	go endpointsliceConfig.Run(stopCh)

	// Add the first endpoints
	fakeWatch.Add(endpoints1v1)
	handler.ValidateEndpointSlices(t, []*discoveryv1.EndpointSlice{endpoints1v1})

	// Add another endpoints
	fakeWatch.Add(endpoints2)
	handler.ValidateEndpointSlices(t, []*discoveryv1.EndpointSlice{endpoints1v1, endpoints2})

	// Modify endpoints1
	fakeWatch.Modify(endpoints1v2)
	handler.ValidateEndpointSlices(t, []*discoveryv1.EndpointSlice{endpoints1v2, endpoints2})

	// Delete endpoints1
	fakeWatch.Delete(endpoints1v2)
	handler.ValidateEndpointSlices(t, []*discoveryv1.EndpointSlice{endpoints2})

	// Delete endpoints2
	fakeWatch.Delete(endpoints2)
	handler.ValidateEndpointSlices(t, []*discoveryv1.EndpointSlice{})
}

func TestInitialSync(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	svc1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	svc2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	eps1 := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
	}
	eps2 := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
	}

	expectedSvcState := map[types.NamespacedName]*v1.Service{
		{Name: svc1.Name, Namespace: svc1.Namespace}: svc1,
		{Name: svc2.Name, Namespace: svc2.Namespace}: svc2,
	}
	expectedEpsState := map[types.NamespacedName]*discoveryv1.EndpointSlice{
		{Name: eps1.Name, Namespace: eps1.Namespace}: eps1,
		{Name: eps2.Name, Namespace: eps2.Namespace}: eps2,
	}

	// Setup fake api client.
	client := fake.NewSimpleClientset(svc1, svc2, eps2, eps1)
	sharedInformers := informers.NewSharedInformerFactory(client, 0)

	svcConfig := NewServiceConfig(ctx, sharedInformers.Core().V1().Services(), 0)
	svcHandler := NewServiceHandlerMock()
	svcConfig.RegisterEventHandler(svcHandler)

	epsConfig := NewEndpointSliceConfig(ctx, sharedInformers.Discovery().V1().EndpointSlices(), 0)
	epsHandler := NewEndpointSliceHandlerMock()
	epsConfig.RegisterEventHandler(epsHandler)

	stopCh := make(chan struct{})
	defer close(stopCh)
	sharedInformers.Start(stopCh)

	err := wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		svcHandler.lock.Lock()
		defer svcHandler.lock.Unlock()
		if reflect.DeepEqual(svcHandler.state, expectedSvcState) {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatal("Timed out waiting for the completion of handler `OnServiceAdd`")
	}

	err = wait.PollImmediate(time.Millisecond*10, wait.ForeverTestTimeout, func() (bool, error) {
		epsHandler.lock.Lock()
		defer epsHandler.lock.Unlock()
		if reflect.DeepEqual(epsHandler.state, expectedEpsState) {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		t.Fatal("Timed out waiting for the completion of handler `OnEndpointsAdd`")
	}

	svcConfig.Run(stopCh)
	epsConfig.Run(stopCh)

	gotSvc := <-svcHandler.updated
	gotSvcState := make(map[types.NamespacedName]*v1.Service, len(gotSvc))
	for _, svc := range gotSvc {
		gotSvcState[types.NamespacedName{Namespace: svc.Namespace, Name: svc.Name}] = svc
	}
	if !reflect.DeepEqual(gotSvcState, expectedSvcState) {
		t.Fatalf("Expected service state: %v\nGot: %v\n", expectedSvcState, gotSvcState)
	}

	gotEps := <-epsHandler.updated
	gotEpsState := make(map[types.NamespacedName]*discoveryv1.EndpointSlice, len(gotEps))
	for _, eps := range gotEps {
		gotEpsState[types.NamespacedName{Namespace: eps.Namespace, Name: eps.Name}] = eps
	}
	if !reflect.DeepEqual(gotEpsState, expectedEpsState) {
		t.Fatalf("Expected endpoints state: %v\nGot: %v\n", expectedEpsState, gotEpsState)
	}
}
