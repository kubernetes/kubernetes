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
	"reflect"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	utilpointer "k8s.io/utils/pointer"
)

func TestNewEndpointSlice(t *testing.T) {
	ipAddressType := discovery.AddressTypeIPv4
	portName := "foo"
	protocol := v1.ProtocolTCP
	endpointMeta := endpointMeta{
		Ports:       []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}},
		AddressType: ipAddressType,
	}
	service := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec: v1.ServiceSpec{
			Ports:    []v1.ServicePort{{Port: 80}},
			Selector: map[string]string{"foo": "bar"},
		},
	}

	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Service"}
	ownerRef := metav1.NewControllerRef(&service, gvk)

	expectedSlice := discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{
				discovery.LabelServiceName: service.Name,
				discovery.LabelManagedBy:   controllerName,
			},
			GenerateName:    fmt.Sprintf("%s-", service.Name),
			OwnerReferences: []metav1.OwnerReference{*ownerRef},
			Namespace:       service.Namespace,
		},
		Ports:       endpointMeta.Ports,
		AddressType: endpointMeta.AddressType,
		Endpoints:   []discovery.Endpoint{},
	}
	generatedSlice := newEndpointSlice(&service, &endpointMeta)

	assert.EqualValues(t, expectedSlice, *generatedSlice)
}

func TestPodToEndpoint(t *testing.T) {
	ns := "test"
	svc, _ := newServiceAndEndpointMeta("foo", ns)
	svcPublishNotReady, _ := newServiceAndEndpointMeta("publishnotready", ns)
	svcPublishNotReady.Spec.PublishNotReadyAddresses = true

	readyPod := newPod(1, ns, true, 1)
	readyPodHostname := newPod(1, ns, true, 1)
	readyPodHostname.Spec.Subdomain = svc.Name
	readyPodHostname.Spec.Hostname = "example-hostname"

	unreadyPod := newPod(1, ns, false, 1)
	multiIPPod := newPod(1, ns, true, 1)
	multiIPPod.Status.PodIPs = []v1.PodIP{{IP: "1.2.3.4"}, {IP: "1234::5678:0000:0000:9abc:def0"}}

	node1 := &v1.Node{
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
		pod                      *v1.Pod
		node                     *v1.Node
		svc                      *v1.Service
		expectedEndpoint         discovery.Endpoint
		publishNotReadyAddresses bool
	}{
		{
			name: "Ready pod",
			pod:  readyPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Ready pod + publishNotReadyAddresses",
			pod:  readyPod,
			svc:  &svcPublishNotReady,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Unready pod",
			pod:  unreadyPod,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(false)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Unready pod + publishNotReadyAddresses",
			pod:  unreadyPod,
			svc:  &svcPublishNotReady,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology:   map[string]string{"kubernetes.io/hostname": "node-1"},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Ready pod + node labels",
			pod:  readyPod,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology: map[string]string{
					"kubernetes.io/hostname":        "node-1",
					"topology.kubernetes.io/zone":   "us-central1-a",
					"topology.kubernetes.io/region": "us-central1",
				},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Multi IP Ready pod + node labels",
			pod:  multiIPPod,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.4"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Topology: map[string]string{
					"kubernetes.io/hostname":        "node-1",
					"topology.kubernetes.io/zone":   "us-central1-a",
					"topology.kubernetes.io/region": "us-central1",
				},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPod.Name,
					UID:             readyPod.UID,
					ResourceVersion: readyPod.ResourceVersion,
				},
			},
		},
		{
			name: "Ready pod + hostname",
			pod:  readyPodHostname,
			node: node1,
			svc:  &svc,
			expectedEndpoint: discovery.Endpoint{
				Addresses:  []string{"1.2.3.5"},
				Conditions: discovery.EndpointConditions{Ready: utilpointer.BoolPtr(true)},
				Hostname:   &readyPodHostname.Spec.Hostname,
				Topology: map[string]string{
					"kubernetes.io/hostname":        "node-1",
					"topology.kubernetes.io/zone":   "us-central1-a",
					"topology.kubernetes.io/region": "us-central1",
				},
				TargetRef: &v1.ObjectReference{
					Kind:            "Pod",
					Namespace:       ns,
					Name:            readyPodHostname.Name,
					UID:             readyPodHostname.UID,
					ResourceVersion: readyPodHostname.ResourceVersion,
				},
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			endpoint := podToEndpoint(testCase.pod, testCase.node, testCase.svc)
			if !reflect.DeepEqual(testCase.expectedEndpoint, endpoint) {
				t.Errorf("Expected endpoint: %v, got: %v", testCase.expectedEndpoint, endpoint)
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
			expectedErr:   fmt.Errorf("nil EndpointSlice passed to serviceControllerKey()"),
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
			actualKey, actualErr := serviceControllerKey(tc.endpointSlice)
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
	protoTCP := v1.ProtocolTCP

	testCases := map[string]struct {
		service       *v1.Service
		pod           *v1.Pod
		expectedPorts []*discovery.EndpointPort
	}{
		"service with AppProtocol on one port": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Name:        "http",
						Port:        80,
						TargetPort:  intstr.FromInt(80),
						Protocol:    protoTCP,
						AppProtocol: utilpointer.StringPtr("example.com/custom-protocol"),
					}},
				},
			},
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Ports: []v1.ContainerPort{},
					}},
				},
			},
			expectedPorts: []*discovery.EndpointPort{{
				Name:        utilpointer.StringPtr("http"),
				Port:        utilpointer.Int32Ptr(80),
				Protocol:    &protoTCP,
				AppProtocol: utilpointer.StringPtr("example.com/custom-protocol"),
			}},
		},
		"service with named port and AppProtocol on one port": {
			service: &v1.Service{
				Spec: v1.ServiceSpec{
					Ports: []v1.ServicePort{{
						Name:       "http",
						Port:       80,
						TargetPort: intstr.FromInt(80),
						Protocol:   protoTCP,
					}, {
						Name:        "https",
						Protocol:    protoTCP,
						TargetPort:  intstr.FromString("https"),
						AppProtocol: utilpointer.StringPtr("https"),
					}},
				},
			},
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Ports: []v1.ContainerPort{{
							Name:          "https",
							ContainerPort: int32(443),
							Protocol:      protoTCP,
						}},
					}},
				},
			},
			expectedPorts: []*discovery.EndpointPort{{
				Name:     utilpointer.StringPtr("http"),
				Port:     utilpointer.Int32Ptr(80),
				Protocol: &protoTCP,
			}, {
				Name:        utilpointer.StringPtr("https"),
				Port:        utilpointer.Int32Ptr(443),
				Protocol:    &protoTCP,
				AppProtocol: utilpointer.StringPtr("https"),
			}},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actualPorts := getEndpointPorts(tc.service, tc.pod)

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

// Test helpers

func newPod(n int, namespace string, ready bool, nPorts int) *v1.Pod {
	status := v1.ConditionTrue
	if !ready {
		status = v1.ConditionFalse
	}

	p := &v1.Pod{
		TypeMeta: metav1.TypeMeta{APIVersion: "v1"},
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      fmt.Sprintf("pod%d", n),
			Labels:    map[string]string{"foo": "bar"},
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

func newClientset() *fake.Clientset {
	client := fake.NewSimpleClientset()

	client.PrependReactor("create", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)

		if endpointSlice.ObjectMeta.GenerateName != "" {
			endpointSlice.ObjectMeta.Name = fmt.Sprintf("%s-%s", endpointSlice.ObjectMeta.GenerateName, rand.String(8))
			endpointSlice.ObjectMeta.GenerateName = ""
		}
		endpointSlice.ObjectMeta.ResourceVersion = "100"

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.ObjectMeta.ResourceVersion = "200"
		return false, endpointSlice, nil
	}))

	return client
}

func newServiceAndEndpointMeta(name, namespace string) (v1.Service, endpointMeta) {
	portNum := int32(80)
	portNameIntStr := intstr.IntOrString{
		Type:   intstr.Int,
		IntVal: portNum,
	}

	svc := v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				TargetPort: portNameIntStr,
				Protocol:   v1.ProtocolTCP,
				Name:       name,
			}},
			Selector: map[string]string{"foo": "bar"},
		},
	}

	addressType := discovery.AddressTypeIPv4
	protocol := v1.ProtocolTCP
	endpointMeta := endpointMeta{
		AddressType: addressType,
		Ports:       []discovery.EndpointPort{{Name: &name, Port: &portNum, Protocol: &protocol}},
	}

	return svc, endpointMeta
}

func newEmptyEndpointSlice(n int, namespace string, endpointMeta endpointMeta, svc v1.Service) *discovery.EndpointSlice {
	return &discovery.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("%s.%d", svc.Name, n),
			Namespace: namespace,
		},
		Ports:       endpointMeta.Ports,
		AddressType: endpointMeta.AddressType,
		Endpoints:   []discovery.Endpoint{},
	}
}
