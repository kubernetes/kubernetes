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
	"fmt"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	informers "k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	ktesting "k8s.io/client-go/testing"
	klogtesting "k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

type ServiceHandlerMock struct {
	lock sync.Mutex

	state   map[types.NamespacedName]*v1.Service
	synced  bool
	updated chan []*v1.Service
	process func([]*v1.Service)
}

func NewServiceHandlerMock() *ServiceHandlerMock {
	shm := &ServiceHandlerMock{
		state:   make(map[types.NamespacedName]*v1.Service),
		updated: make(chan []*v1.Service, 5),
	}
	shm.process = func(services []*v1.Service) {
		shm.updated <- services
	}
	return shm
}

func (h *ServiceHandlerMock) OnServiceAdd(service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	h.state[namespacedName] = service
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceUpdate(oldService, service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	h.state[namespacedName] = service
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceDelete(service *v1.Service) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}
	delete(h.state, namespacedName)
	h.sendServices()
}

func (h *ServiceHandlerMock) OnServiceSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendServices()
}

func (h *ServiceHandlerMock) sendServices() {
	if !h.synced {
		return
	}
	services := make([]*v1.Service, 0, len(h.state))
	for _, svc := range h.state {
		services = append(services, svc)
	}
	slices.SortFunc(services, func(a, b *v1.Service) int { return strings.Compare(a.Name, b.Name) })
	h.process(services)
}

func (h *ServiceHandlerMock) ValidateServices(t *testing.T, expectedServices []*v1.Service) {
	// We might get 1 or more updates for N service updates, because we
	// over write older snapshots of services from the producer go-routine
	// if the consumer falls behind.
	var services []*v1.Service
	for {
		select {
		case services = <-h.updated:
			if reflect.DeepEqual(services, expectedServices) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedServices, services)
			return
		}
	}
}

func TestNewServiceAddedAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(ctx, sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service)
	handler.ValidateServices(t, []*v1.Service{service})
}

func TestServiceAddedRemovedSetAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(ctx, sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	fakeWatch.Add(service1)
	handler.ValidateServices(t, []*v1.Service{service1})

	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service2)
	services := []*v1.Service{service2, service1}
	handler.ValidateServices(t, services)

	fakeWatch.Delete(service1)
	services = []*v1.Service{service2}
	handler.ValidateServices(t, services)
}

func TestNewServicesMultipleHandlersAddedAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("services", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceConfig(ctx, sharedInformers.Core().V1().Services(), time.Minute)
	handler := NewServiceHandlerMock()
	handler2 := NewServiceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	service1 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 10}}},
	}
	service2 := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		Spec:       v1.ServiceSpec{Ports: []v1.ServicePort{{Protocol: "TCP", Port: 20}}},
	}
	fakeWatch.Add(service1)
	fakeWatch.Add(service2)

	services := []*v1.Service{service2, service1}
	handler.ValidateServices(t, services)
	handler2.ValidateServices(t, services)
}

type EndpointSliceHandlerMock struct {
	lock sync.Mutex

	state   map[types.NamespacedName]*discoveryv1.EndpointSlice
	synced  bool
	updated chan []*discoveryv1.EndpointSlice
	process func([]*discoveryv1.EndpointSlice)
}

func NewEndpointSliceHandlerMock() *EndpointSliceHandlerMock {
	ehm := &EndpointSliceHandlerMock{
		state:   make(map[types.NamespacedName]*discoveryv1.EndpointSlice),
		updated: make(chan []*discoveryv1.EndpointSlice, 5),
	}
	ehm.process = func(endpoints []*discoveryv1.EndpointSlice) {
		ehm.updated <- endpoints
	}
	return ehm
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceAdd(slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	h.state[namespacedName] = slice
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceUpdate(oldSlice, slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	h.state[namespacedName] = slice
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSliceDelete(slice *discoveryv1.EndpointSlice) {
	h.lock.Lock()
	defer h.lock.Unlock()
	namespacedName := types.NamespacedName{Namespace: slice.Namespace, Name: slice.Name}
	delete(h.state, namespacedName)
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) OnEndpointSlicesSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendEndpointSlices()
}

func (h *EndpointSliceHandlerMock) sendEndpointSlices() {
	if !h.synced {
		return
	}
	endpointSlices := make([]*discoveryv1.EndpointSlice, 0, len(h.state))
	for _, eps := range h.state {
		endpointSlices = append(endpointSlices, eps)
	}
	slices.SortFunc(endpointSlices, func(a, b *discoveryv1.EndpointSlice) int { return strings.Compare(a.Name, b.Name) })
	h.process(endpointSlices)
}

func (h *EndpointSliceHandlerMock) ValidateEndpointSlices(t *testing.T, expectedSlices []*discoveryv1.EndpointSlice) {
	// We might get 1 or more updates for N endpointslice updates, because we
	// over write older snapshots of endpointslices from the producer go-routine
	// if the consumer falls behind. Unittests will hard timeout in 5m.
	var slices []*discoveryv1.EndpointSlice
	for {
		select {
		case slices = <-h.updated:
			if reflect.DeepEqual(slices, expectedSlices) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedSlices, slices)
			return
		}
	}
}

func TestNewEndpointsMultipleHandlersAddedAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpointslices", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointSliceConfig(ctx, sharedInformers.Discovery().V1().EndpointSlices(), time.Minute)
	handler := NewEndpointSliceHandlerMock()
	handler2 := NewEndpointSliceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"2.2.2.2"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	endpoints2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"3.3.3.3"},
		}, {
			Addresses: []string{"4.4.4.4"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*discoveryv1.EndpointSlice{endpoints2, endpoints1}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)
}

func TestNewEndpointsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("endpointslices", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewEndpointSliceConfig(ctx, sharedInformers.Discovery().V1().EndpointSlices(), time.Minute)
	handler := NewEndpointSliceHandlerMock()
	handler2 := NewEndpointSliceHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	endpoints1 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"1.1.1.1"},
		}, {
			Addresses: []string{"2.2.2.2"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	endpoints2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "bar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"3.3.3.3"},
		}, {
			Addresses: []string{"4.4.4.4"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints1)
	fakeWatch.Add(endpoints2)

	endpoints := []*discoveryv1.EndpointSlice{endpoints2, endpoints1}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Add one more
	endpoints3 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foobar"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"5.5.5.5"},
		}, {
			Addresses: []string{"6.6.6.6"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Add(endpoints3)
	endpoints = []*discoveryv1.EndpointSlice{endpoints2, endpoints1, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Update the "foo" service with new endpoints
	endpoints1v2 := &discoveryv1.EndpointSlice{
		ObjectMeta:  metav1.ObjectMeta{Namespace: "testnamespace", Name: "foo"},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints: []discoveryv1.Endpoint{{
			Addresses: []string{"7.7.7.7"},
		}},
		Ports: []discoveryv1.EndpointPort{{Port: ptr.To[int32](80)}},
	}
	fakeWatch.Modify(endpoints1v2)
	endpoints = []*discoveryv1.EndpointSlice{endpoints2, endpoints1v2, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)

	// Remove "bar" endpoints
	fakeWatch.Delete(endpoints2)
	endpoints = []*discoveryv1.EndpointSlice{endpoints1v2, endpoints3}
	handler.ValidateEndpointSlices(t, endpoints)
	handler2.ValidateEndpointSlices(t, endpoints)
}

type NodeHandlerMock struct {
	lock sync.Mutex

	state   map[string]*v1.Node
	synced  bool
	updated chan []*v1.Node
	process func([]*v1.Node)
}

func NewNodeHandlerMock() *NodeHandlerMock {
	h := &NodeHandlerMock{
		state:   make(map[string]*v1.Node),
		updated: make(chan []*v1.Node, 5),
	}
	h.process = func(nodes []*v1.Node) {
		h.updated <- nodes
	}
	return h
}

func (h *NodeHandlerMock) OnNodeChange(node *v1.Node) {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.state[node.Name] = node
	h.sendNodes()
}

func (h *NodeHandlerMock) OnNodeDelete(node *v1.Node) {
	h.lock.Lock()
	defer h.lock.Unlock()
	delete(h.state, node.Name)
	h.sendNodes()
}

func (h *NodeHandlerMock) OnNodeSynced() {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.synced = true
	h.sendNodes()
}

func (h *NodeHandlerMock) sendNodes() {
	if !h.synced {
		return
	}
	nodes := make([]*v1.Node, 0, len(h.state))
	for _, svc := range h.state {
		nodes = append(nodes, svc)
	}
	slices.SortFunc(nodes, func(a, b *v1.Node) int { return strings.Compare(a.Name, b.Name) })
	h.process(nodes)
}

func (h *NodeHandlerMock) ValidateNodes(t *testing.T, expectedNodes []*v1.Node) {
	// We might get 1 or more updates for N node updates, because we
	// over write older snapshots of nodes from the producer go-routine
	// if the consumer falls behind.
	var nodes []*v1.Node
	for {
		select {
		case nodes = <-h.updated:
			if reflect.DeepEqual(nodes, expectedNodes) {
				return
			}
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedNodes, nodes)
			return
		}
	}
}

func TestNewNodesMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("nodes", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewNodeConfig(ctx, sharedInformers.Core().V1().Nodes(), time.Minute)
	handler := NewNodeHandlerMock()
	handler2 := NewNodeHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	nodes1 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: "1.1.1.1",
				},
				{
					Type:    v1.NodeExternalIP,
					Address: "2.2.2.2",
				},
			},
		},
	}
	nodes2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "bar"},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: "3.3.3.3",
				},
				{
					Type:    v1.NodeInternalIP,
					Address: "fc00::4",
				},
			},
		},
	}
	fakeWatch.Add(nodes1)
	fakeWatch.Add(nodes2)

	nodes := []*v1.Node{nodes2, nodes1}
	handler.ValidateNodes(t, nodes)
	handler2.ValidateNodes(t, nodes)

	// Add one more
	nodes3 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "foobar"},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: "5.5.5.5",
				},
			},
		},
	}
	fakeWatch.Add(nodes3)
	nodes = []*v1.Node{nodes2, nodes1, nodes3}
	handler.ValidateNodes(t, nodes)
	handler2.ValidateNodes(t, nodes)

	// Update the "foo" node with a new address
	nodes1v2 := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Status: v1.NodeStatus{
			Addresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: "6.6.6.6",
				},
			},
		},
	}
	fakeWatch.Modify(nodes1v2)
	nodes = []*v1.Node{nodes2, nodes1v2, nodes3}
	handler.ValidateNodes(t, nodes)
	handler2.ValidateNodes(t, nodes)

	// Remove "bar" node
	fakeWatch.Delete(nodes2)
	nodes = []*v1.Node{nodes1v2, nodes3}
	handler.ValidateNodes(t, nodes)
	handler2.ValidateNodes(t, nodes)
}

type ServiceCIDRHandlerMock struct {
	lock sync.Mutex

	state   []string
	updated chan []string
	process func([]string)
}

func NewServiceCIDRHandlerMock() *ServiceCIDRHandlerMock {
	h := &ServiceCIDRHandlerMock{
		updated: make(chan []string, 5),
	}
	h.process = func(serviceCIDRs []string) {
		h.updated <- serviceCIDRs
	}
	return h
}

func (h *ServiceCIDRHandlerMock) OnServiceCIDRsChanged(serviceCIDRs []string) {
	h.lock.Lock()
	defer h.lock.Unlock()
	h.state = serviceCIDRs
	h.sendServiceCIDRs()
}

func (h *ServiceCIDRHandlerMock) sendServiceCIDRs() {
	serviceCIDRs := append([]string{}, h.state...)
	slices.Sort(serviceCIDRs)
	h.process(serviceCIDRs)
}

func (h *ServiceCIDRHandlerMock) ValidateServiceCIDRs(t *testing.T, expectedServiceCIDRs []string) {
	// We might get 1 or more updates for N serviceCIDR updates, because we
	// over write older snapshots of nodes from the producer go-routine
	// if the consumer falls behind.
	var serviceCIDRs []string
	for {
		select {
		case serviceCIDRs = <-h.updated:
			if reflect.DeepEqual(serviceCIDRs, expectedServiceCIDRs) {
				return
			}
			t.Logf("Expected %#v, Got %#v", expectedServiceCIDRs, serviceCIDRs)
		// Unittests will hard timeout in 5m with a stack trace, prevent that
		// and surface a clearer reason for failure.
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("Timed out. Expected %#v, Got %#v", expectedServiceCIDRs, serviceCIDRs)
			return
		}
	}
}

func TestNewServiceCIDRsMultipleHandlersAddRemoveSetAndNotified(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewSimpleClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("servicecidrs", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)

	config := NewServiceCIDRConfig(ctx, sharedInformers.Networking().V1().ServiceCIDRs(), time.Minute)
	handler := NewServiceCIDRHandlerMock()
	handler2 := NewServiceCIDRHandlerMock()
	config.RegisterEventHandler(handler)
	config.RegisterEventHandler(handler2)
	sharedInformers.Start(stopCh)
	go config.Run(stopCh)

	serviceCIDRs1 := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{"1.1.1.0/24"},
		},
	}
	serviceCIDRs2 := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{Name: "bar"},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{"2.2.2.0/24", "fc00::/64"},
		},
	}
	fakeWatch.Add(serviceCIDRs1)
	fakeWatch.Add(serviceCIDRs2)

	serviceCIDRs := []string{"1.1.1.0/24", "2.2.2.0/24", "fc00::/64"}
	handler.ValidateServiceCIDRs(t, serviceCIDRs)
	handler2.ValidateServiceCIDRs(t, serviceCIDRs)

	// Add one more
	serviceCIDRs3 := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{Name: "foobar"},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{"3.3.3.0/24", "2001:db8::/64"},
		},
	}
	fakeWatch.Add(serviceCIDRs3)
	serviceCIDRs = []string{"1.1.1.0/24", "2.2.2.0/24", "2001:db8::/64", "3.3.3.0/24", "fc00::/64"}
	handler.ValidateServiceCIDRs(t, serviceCIDRs)
	handler2.ValidateServiceCIDRs(t, serviceCIDRs)

	// Update the "foo" ServiceCIDR
	serviceCIDRs1v2 := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec: networkingv1.ServiceCIDRSpec{
			CIDRs: []string{"4.4.4.0/24"},
		},
	}
	fakeWatch.Modify(serviceCIDRs1v2)
	serviceCIDRs = []string{"2.2.2.0/24", "2001:db8::/64", "3.3.3.0/24", "4.4.4.0/24", "fc00::/64"}
	handler.ValidateServiceCIDRs(t, serviceCIDRs)
	handler2.ValidateServiceCIDRs(t, serviceCIDRs)

	// Remove "bar" ServiceCIDR
	fakeWatch.Delete(serviceCIDRs2)
	serviceCIDRs = []string{"2001:db8::/64", "3.3.3.0/24", "4.4.4.0/24"}
	handler.ValidateServiceCIDRs(t, serviceCIDRs)
	handler2.ValidateServiceCIDRs(t, serviceCIDRs)
}

type nodeTopologyHandlerMock struct {
	topologyLabels map[string]string
}

func (n *nodeTopologyHandlerMock) OnTopologyChange(topologyLabels map[string]string) {
	n.topologyLabels = topologyLabels
}

// waitForInvocation waits for event handler to complete processing of the invocation.
func waitForInvocation(invoked <-chan struct{}) error {
	select {
	// unit tests will hard timeout in 5m with a stack trace, prevent that
	// and surface a clearer reason for failure.
	case <-time.After(wait.ForeverTestTimeout):
		return fmt.Errorf("timed out waiting for event handler to process update")
	case <-invoked:
		return nil
	}
}

func TestNewNodeTopologyConfig(t *testing.T) {
	_, ctx := klogtesting.NewTestContext(t)
	client := fake.NewClientset()
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("nodes", ktesting.DefaultWatchReactor(fakeWatch, nil))

	stopCh := make(chan struct{})
	defer close(stopCh)

	sharedInformers := informers.NewSharedInformerFactory(client, time.Minute)
	nodeInformer := sharedInformers.Core().V1().Nodes()
	invoked := make(chan struct{})
	config := newNodeTopologyConfig(ctx, nodeInformer, time.Minute, func() {
		// The callback is invoked after the event has been processed by the
		// handlers. For this unit test, we write to the channel here and wait
		// for it in waitForInvocation() which is called before doing assertions.
		invoked <- struct{}{}
	})

	handler := &nodeTopologyHandlerMock{
		topologyLabels: make(map[string]string),
	}
	config.RegisterEventHandler(handler)
	sharedInformers.Start(stopCh)

	testNodeName := "test-node"

	// add non-topology labels, handle should receive no notification
	fakeWatch.Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Labels: map[string]string{
				v1.LabelInstanceType: "m4.large",
				v1.LabelOSStable:     "linux",
			},
		},
	})
	err := waitForInvocation(invoked)
	require.NoError(t, err)
	require.Empty(t, handler.topologyLabels)

	// add topology label not relevant to kube-proxy, handle should receive no notification
	fakeWatch.Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Labels: map[string]string{
				v1.LabelInstanceType:   "m4.large",
				v1.LabelOSStable:       "linux",
				v1.LabelTopologyRegion: "us-east-1",
			},
		},
	})
	err = waitForInvocation(invoked)
	require.NoError(t, err)
	require.Empty(t, handler.topologyLabels)

	// add relevant zone topology label, handle should receive notification
	fakeWatch.Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Labels: map[string]string{
				v1.LabelInstanceType: "c6.large",
				v1.LabelOSStable:     "windows",
				v1.LabelTopologyZone: "us-west-2a",
			},
		},
	})

	err = waitForInvocation(invoked)
	require.NoError(t, err)
	require.Len(t, handler.topologyLabels, 1)
	require.Equal(t, map[string]string{
		v1.LabelTopologyZone: "us-west-2a",
	}, handler.topologyLabels)

	// add region topology label, handle should not receive notification
	// because kube-proxy doesn't do any region-based topology.
	fakeWatch.Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Labels: map[string]string{
				v1.LabelInstanceType:   "m3.medium",
				v1.LabelOSStable:       "windows",
				v1.LabelTopologyRegion: "us-east-1",
				v1.LabelTopologyZone:   "us-east-1b",
			},
		},
	})
	err = waitForInvocation(invoked)
	require.NoError(t, err)
	require.Len(t, handler.topologyLabels, 1)
	require.Equal(t, map[string]string{
		v1.LabelTopologyZone: "us-east-1b",
	}, handler.topologyLabels)

	// update non-topology label, handle should not receive notification
	fakeWatch.Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNodeName,
			Labels: map[string]string{
				v1.LabelInstanceType:   "m3.large",
				v1.LabelOSStable:       "windows",
				v1.LabelTopologyRegion: "us-east-1",
				v1.LabelTopologyZone:   "us-east-1b",
			},
		},
	})
	err = waitForInvocation(invoked)
	require.NoError(t, err)
	require.Len(t, handler.topologyLabels, 1)
}

// TODO: Add a unittest for interrupts getting processed in a timely manner.
