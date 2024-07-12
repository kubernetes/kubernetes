/*
Copyright 2024 The Kubernetes Authors.

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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	corelisters "k8s.io/client-go/listers/core/v1"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestPodToEndpoint(t *testing.T) {
	ns := "test"
	svc, _, _ := newServicePortsAddressType("foo", ns)
	svcPublishNotReady, _, _ := newServicePortsAddressType("publishnotready", ns)
	svcPublishNotReady.Spec.PublishNotReadyAddresses = true

	readyPod := newPod(1, ns, true, 1, false)
	readyTerminatingPod := newPod(1, ns, true, 1, true)
	readyPodHostname := newPod(1, ns, true, 1, false)
	readyPodHostname.Spec.Subdomain = svc.Name
	readyPodHostname.Spec.Hostname = "example-hostname"

	unreadyPod := newPod(1, ns, false, 1, false)
	unreadyTerminatingPod := newPod(1, ns, false, 1, true)
	multiIPPod := newPod(1, ns, true, 1, false)
	multiIPPod.Status.PodIPs = []corev1.PodIP{{IP: "1.2.3.4"}, {IP: "1234::5678:0000:0000:9abc:def0"}}

	node1 := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: readyPod.Spec.NodeName,
			Labels: map[string]string{
				"topology.kubernetes.io/zone":   "us-central1-a",
				"topology.kubernetes.io/region": "us-central1",
			},
		},
	}

	testCases := []struct {
		name                     string
		pod                      *corev1.Pod
		node                     *corev1.Node
		svc                      *corev1.Service
		expectedEndpoint         discovery.Endpoint
		publishNotReadyAddresses bool
	}{
		{
			name: "Ready pod",
			pod:  readyPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Ready pod + publishNotReadyAddresses",
			pod:  readyPod,
			svc:  &svcPublishNotReady,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Unready pod",
			pod:  unreadyPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(false),
					Serving:     ptr.To(false),
					Terminating: ptr.To(false),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Unready pod + publishNotReadyAddresses",
			pod:  unreadyPod,
			svc:  &svcPublishNotReady,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(false),
					Terminating: ptr.To(false),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Ready pod + node labels",
			pod:  readyPod,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				Zone:     ptr.To("us-central1-a"),
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Multi IP Ready pod + node labels",
			pod:  multiIPPod,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.4"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				Zone:     ptr.To("us-central1-a"),
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Ready pod + hostname",
			pod:  readyPodHostname,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				Hostname: &readyPodHostname.Spec.Hostname,
				Zone:     ptr.To("us-central1-a"),
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPodHostname.Name,
					UID:       readyPodHostname.UID,
				},
			},
		},
		{
			name: "Ready pod",
			pod:  readyPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(true),
					Serving:     ptr.To(true),
					Terminating: ptr.To(false),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Ready terminating pod",
			pod:  readyTerminatingPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(false),
					Serving:     ptr.To(true),
					Terminating: ptr.To(true),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
		{
			name: "Not ready terminating pod",
			pod:  unreadyTerminatingPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses: []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{
					Ready:       ptr.To(false),
					Serving:     ptr.To(false),
					Terminating: ptr.To(true),
				},
				NodeName: ptr.To("node-1"),
				TargetRef: &corev1.ObjectReference{
					Kind:      "Pod",
					Namespace: ns,
					Name:      readyPod.Name,
					UID:       readyPod.UID,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			endpoint := podToEndpoint(testCase.pod, testCase.node, testCase.svc, discovery.AddressTypeIPv4)
			if !reflect.DeepEqual(testCase.expectedEndpoint, endpoint) {
				t.Errorf("Expected endpoint: %+v, got: %+v", testCase.expectedEndpoint, endpoint)
			}
		})
	}
}

func TestServiceControllerKey(t *testing.T) {
	testCases := map[string]struct {
		endpointSlice *discovery.EndpointSlice
		expectedKey   string
		expectedErr   error
	}{
		"nil EndpointSlice": {
			endpointSlice: nil,
			expectedKey:   "",
			expectedErr:   fmt.Errorf("nil EndpointSlice passed to ServiceControllerKey()"),
		},
		"empty EndpointSlice": {
			endpointSlice: &discovery.EndpointSlice{},
			expectedKey:   "",
			expectedErr:   fmt.Errorf("EndpointSlice missing kubernetes.io/service-name label"),
		},
		"valid EndpointSlice": {
			endpointSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "ns",
					Labels: map[string]string{
						discovery.LabelServiceName: "svc",
					},
				},
			},
			expectedKey: "ns/svc",
			expectedErr: nil,
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actualKey, actualErr := ServiceControllerKey(tc.endpointSlice)
			if !reflect.DeepEqual(actualErr, tc.expectedErr) {
				t.Errorf("Expected %s, got %s", tc.expectedErr, actualErr)
			}
			if actualKey != tc.expectedKey {
				t.Errorf("Expected %s, got %s", tc.expectedKey, actualKey)
			}
		})
	}
}

func TestGetEndpointPorts(t *testing.T) {
	protoTCP := corev1.ProtocolTCP

	testCases := map[string]struct {
		service       *corev1.Service
		pod           *corev1.Pod
		expectedPorts []*discovery.EndpointPort
	}{
		"service with AppProtocol on one port": {
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{{
						Name:        "http",
						Port:        80,
						TargetPort:  intstr.FromInt32(80),
						Protocol:    protoTCP,
						AppProtocol: ptr.To("example.com/custom-protocol"),
					}},
				},
			},
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Ports: []corev1.ContainerPort{},
					}},
				},
			},
			expectedPorts: []*discovery.EndpointPort{{
				Name:        ptr.To("http"),
				Port:        ptr.To(int32(80)),
				Protocol:    &protoTCP,
				AppProtocol: ptr.To("example.com/custom-protocol"),
			}},
		},
		"service with named port and AppProtocol on one port": {
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{{
						Name:       "http",
						Port:       80,
						TargetPort: intstr.FromInt32(80),
						Protocol:   protoTCP,
					}, {
						Name:        "https",
						Protocol:    protoTCP,
						TargetPort:  intstr.FromString("https"),
						AppProtocol: ptr.To("https"),
					}},
				},
			},
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Ports: []corev1.ContainerPort{{
							Name:          "https",
							ContainerPort: int32(443),
							Protocol:      protoTCP,
						}},
					}},
				},
			},
			expectedPorts: []*discovery.EndpointPort{{
				Name:     ptr.To("http"),
				Port:     ptr.To(int32(80)),
				Protocol: &protoTCP,
			}, {
				Name:        ptr.To("https"),
				Port:        ptr.To(int32(443)),
				Protocol:    &protoTCP,
				AppProtocol: ptr.To("https"),
			}},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			actualPorts := getEndpointPorts(logger, tc.service, tc.pod)

			if len(actualPorts) != len(tc.expectedPorts) {
				t.Fatalf("Expected %d ports, got %d", len(tc.expectedPorts), len(actualPorts))
			}

			for i, actualPort := range actualPorts {
				if !reflect.DeepEqual(&actualPort, tc.expectedPorts[i]) {
					t.Errorf("Expected port: %+v, got %+v", tc.expectedPorts[i], &actualPort)
				}
			}
		})
	}
}

func TestSupportedServiceAddressType(t *testing.T) {
	testCases := []struct {
		name                 string
		service              corev1.Service
		expectedAddressTypes []discovery.AddressType
	}{
		{
			name:                 "v4 service with no ip families (cluster upgrade)",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv4},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP:  "10.0.0.10",
					IPFamilies: nil,
				},
			},
		},
		{
			name:                 "v6 service with no ip families (cluster upgrade)",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv6},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP:  "2000::1",
					IPFamilies: nil,
				},
			},
		},
		{
			name:                 "v4 service",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv4},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol},
				},
			},
		},
		{
			name:                 "v6 services",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv6},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					IPFamilies: []corev1.IPFamily{corev1.IPv6Protocol},
				},
			},
		},
		{
			name:                 "v4,v6 service",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol},
				},
			},
		},
		{
			name:                 "v6,v4 service",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv6, discovery.AddressTypeIPv4},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					IPFamilies: []corev1.IPFamily{corev1.IPv6Protocol, corev1.IPv4Protocol},
				},
			},
		},
		{
			name:                 "headless with no selector and no families (old api-server)",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv6, discovery.AddressTypeIPv4},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP:  corev1.ClusterIPNone,
					IPFamilies: nil,
				},
			},
		},
		{
			name:                 "headless with selector and no families (old api-server)",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv6, discovery.AddressTypeIPv4},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					ClusterIP:  corev1.ClusterIPNone,
					IPFamilies: nil,
				},
			},
		},

		{
			name:                 "headless with no selector with families",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP:  corev1.ClusterIPNone,
					IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol},
				},
			},
		},
		{
			name:                 "headless with selector with families",
			expectedAddressTypes: []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6},
			service: corev1.Service{
				Spec: corev1.ServiceSpec{
					Selector:   map[string]string{"foo": "bar"},
					ClusterIP:  corev1.ClusterIPNone,
					IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol},
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			addressTypes := getAddressTypesForService(logger, &testCase.service)
			if len(addressTypes) != len(testCase.expectedAddressTypes) {
				t.Fatalf("expected count address types %v got %v", len(testCase.expectedAddressTypes), len(addressTypes))
			}

			// compare
			for _, expectedAddressType := range testCase.expectedAddressTypes {
				found := false
				for key := range addressTypes {
					if key == expectedAddressType {
						found = true
						break

					}
				}
				if !found {
					t.Fatalf("expected address type %v was not found in the result", expectedAddressType)
				}
			}
		})
	}
}

func TestSetEndpointSliceLabels(t *testing.T) {

	service := corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: corev1.ServiceSpec{
			Ports:     []corev1.ServicePort{{Port: 80}},
			Selector:  map[string]string{"foo": "bar"},
			ClusterIP: "1.1.1.1",
		},
	}

	testCases := []struct {
		name           string
		epSlice        *discovery.EndpointSlice
		updateSvc      func(svc corev1.Service) corev1.Service // given basic valid services, each test case can customize them
		expectedLabels map[string]string
		expectedUpdate bool
	}{
		{
			name:    "Service without labels and empty endpoint slice",
			epSlice: &discovery.EndpointSlice{},
			updateSvc: func(svc corev1.Service) corev1.Service {
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name:    "Headless service with labels and empty endpoint slice",
			epSlice: &discovery.EndpointSlice{},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Spec.ClusterIP = corev1.ClusterIPNone
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				corev1.IsHeadlessService:   "",
				"foo":                      "bar",
			},
			expectedUpdate: true,
		},
		{
			name:    "Headless service without labels and empty endpoint slice",
			epSlice: &discovery.EndpointSlice{},
			updateSvc: func(svc corev1.Service) corev1.Service {
				svc.Spec.ClusterIP = corev1.ClusterIPNone
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				corev1.IsHeadlessService:   "",
			},
			expectedUpdate: false,
		},
		{
			name:    "Non Headless service with Headless label and empty endpoint slice",
			epSlice: &discovery.EndpointSlice{},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{corev1.IsHeadlessService: ""}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name: "Headless Service change to ClusterIP Service with headless label",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						corev1.IsHeadlessService:   "",
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{corev1.IsHeadlessService: ""}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name: "Headless Service change to ClusterIP Service",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						corev1.IsHeadlessService:   "",
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name: "Headless service and endpoint slice with same labels",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						"foo":                      "bar",
					},
				},
			}, updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Spec.ClusterIP = corev1.ClusterIPNone
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				corev1.IsHeadlessService:   "",
				"foo":                      "bar",
			},
			expectedUpdate: false,
		},
		{
			name:    "Service with labels and empty endpoint slice",
			epSlice: &discovery.EndpointSlice{},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				"foo":                      "bar",
			},
			expectedUpdate: true,
		},
		{
			name: "Slice with labels and service without labels",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						"foo":                      "bar",
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: true,
		},
		{
			name: "Slice with headless label and service with ClusterIP",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
						corev1.IsHeadlessService:   "",
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name: "Slice with reserved labels and service with labels",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{"foo": "bar"}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				"foo":                      "bar",
			},
			expectedUpdate: true,
		},
		{
			name: "Evil service trying to hijack slice labels only well-known slice labels",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{
					discovery.LabelServiceName: "bad",
					discovery.LabelManagedBy:   "actor",
					corev1.IsHeadlessService:   "invalid",
				}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			expectedUpdate: false,
		},
		{
			name: "Evil service trying to hijack slice labels with updates",
			epSlice: &discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: service.Name,
						discovery.LabelManagedBy:   controllerName,
					},
				},
			},
			updateSvc: func(svc corev1.Service) corev1.Service {
				labels := map[string]string{
					discovery.LabelServiceName: "bad",
					discovery.LabelManagedBy:   "actor",
					"foo":                      "bar",
				}
				svc.Labels = labels
				return svc
			},
			expectedLabels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
				"foo":                      "bar",
			},
			expectedUpdate: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			svc := tc.updateSvc(service)
			labelsFromService := LabelsFromService{Service: &svc}
			labels, _, updated := labelsFromService.SetLabels(logger, tc.epSlice, controllerName)
			assert.EqualValues(t, updated, tc.expectedUpdate)
			assert.EqualValues(t, tc.expectedLabels, labels)
		})
	}
}

func TestDesiredEndpointSlicesFromServicePods(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	client := newClientset()

	namespace := "test"

	nodes := make([]*corev1.Node, 2)
	nodes[0] = &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-0"}}
	nodes[1] = &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "node-1"}}

	pods := make([]*corev1.Pod, 2)
	pods[0] = newPod(0, namespace, true, 1, false)
	pods[0].Spec.NodeName = nodes[0].Name
	pods[1] = newPod(1, namespace, false, 1, false)
	pods[1].Spec.NodeName = nodes[1].Name

	protoTCP := corev1.ProtocolTCP

	type args struct {
		logger        klog.Logger
		pods          []*corev1.Pod
		service       *corev1.Service
		existingNodes []*corev1.Node
	}
	tests := []struct {
		name                      string
		args                      args
		wantDesiredEndpointSlices []*EndpointPortAddressType
		wantSupportedAddressTypes sets.Set[discovery.AddressType]
		wantErr                   bool
	}{
		{
			name: "ipv4 service",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv4: {},
			},
			wantErr: false,
		},
		{
			name: "ipv6 service",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv6Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv6: {},
			},
			wantErr: false,
		},
		{
			name: "ipv4,ipv6 service",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol, corev1.IPv6Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv4: {},
				discovery.AddressTypeIPv6: {},
			},
			wantErr: false,
		},
		{
			name: "no node",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{pods[0]},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: endpointsliceutil.EndpointSet{},
					Ports: []discovery.EndpointPort{
						{Name: ptr.To(""), Protocol: &protoTCP, Port: ptr.To(int32(80))},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv4: {},
			},
			wantErr: true,
		},
		{
			name: "one pod",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{pods[0]},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv4Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{nodes[0]},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(&discovery.Endpoint{
							Addresses:  []string{pods[0].Status.PodIP},
							Conditions: discovery.EndpointConditions{Ready: ptr.To(true), Serving: ptr.To(true), Terminating: ptr.To(false)},
							NodeName:   &nodes[0].Name,
							TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pods[0].Name},
						})
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: ptr.To(""), Protocol: &protoTCP, Port: ptr.To(int32(80))},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv4: {},
			},
			wantErr: false,
		},
		{
			name: "two pods, one is not ready",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{pods[0], pods[1]},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies:               []corev1.IPFamily{corev1.IPv4Protocol},
						PublishNotReadyAddresses: false,
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{nodes[0], nodes[1]},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{
				{
					EndpointSet: func() endpointsliceutil.EndpointSet {
						es := endpointsliceutil.EndpointSet{}
						es.Insert(&discovery.Endpoint{
							Addresses:  []string{pods[0].Status.PodIP},
							Conditions: discovery.EndpointConditions{Ready: ptr.To(true), Serving: ptr.To(true), Terminating: ptr.To(false)},
							NodeName:   &nodes[0].Name,
							TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pods[0].Name},
						})
						es.Insert(&discovery.Endpoint{
							Addresses:  []string{pods[1].Status.PodIP},
							Conditions: discovery.EndpointConditions{Ready: ptr.To(false), Serving: ptr.To(false), Terminating: ptr.To(false)},
							NodeName:   &nodes[1].Name,
							TargetRef:  &corev1.ObjectReference{Kind: "Pod", Namespace: namespace, Name: pods[1].Name},
						})
						return es
					}(),
					Ports: []discovery.EndpointPort{
						{Name: ptr.To(""), Protocol: &protoTCP, Port: ptr.To(int32(80))},
					},
					AddressType: discovery.AddressTypeIPv4,
				},
			},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv4: {},
			},
			wantErr: false,
		},
		{
			name: "one pod with no address type",
			args: args{
				logger: logger,
				pods:   []*corev1.Pod{pods[0]},
				service: &corev1.Service{
					Spec: corev1.ServiceSpec{
						IPFamilies: []corev1.IPFamily{corev1.IPv6Protocol},
						Ports: []corev1.ServicePort{
							{TargetPort: intstr.FromInt32(80), Protocol: protoTCP},
						},
					},
				},
				existingNodes: []*corev1.Node{nodes[0]},
			},
			wantDesiredEndpointSlices: []*EndpointPortAddressType{},
			wantSupportedAddressTypes: sets.Set[discovery.AddressType]{
				discovery.AddressTypeIPv6: {},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			informerFactory := informers.NewSharedInformerFactory(client, 0)
			nodeInformer := informerFactory.Core().V1().Nodes()
			indexer := nodeInformer.Informer().GetIndexer()
			for _, node := range tt.args.existingNodes {
				indexer.Add(node)
			}

			got, got1, err := DesiredEndpointSlicesFromServicePods(tt.args.logger, tt.args.pods, tt.args.service, corelisters.NewNodeLister(indexer))
			if (err != nil) != tt.wantErr {
				t.Errorf("TestDesiredEndpointSlicesFromServicePods() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !CompareEndpointPortAddressTypeSlices(got, tt.wantDesiredEndpointSlices) {
				t.Errorf("TestDesiredEndpointSlicesFromServicePods() got (desiredEndpointSlices) = %v, want %v", got, tt.wantDesiredEndpointSlices)
			}
			if !reflect.DeepEqual(got1, tt.wantSupportedAddressTypes) {
				t.Errorf("TestDesiredEndpointSlicesFromServicePods() got (supportedAddressTypes) = %v, want %v", got1, tt.wantSupportedAddressTypes)
			}
		})
	}
}
