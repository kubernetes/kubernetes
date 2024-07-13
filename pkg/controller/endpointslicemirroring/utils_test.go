/*
Copyright 2020 The Kubernetes Authors.

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

package endpointslicemirroring

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/utils/ptr"
)

func TestNewEndpointSlice(t *testing.T) {
	portName := "foo"
	protocol := v1.ProtocolTCP

	ports := []discovery.EndpointPort{{Name: &portName, Protocol: &protocol}}
	addrType := discovery.AddressTypeIPv4
	gvk := schema.GroupVersionKind{Version: "v1", Kind: "Endpoints"}

	endpoints := v1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "test",
		},
		Subsets: []v1.EndpointSubset{{
			Ports: []v1.EndpointPort{{Port: 80}},
		}},
	}
	ownerRef := metav1.NewControllerRef(&endpoints, gvk)

	testCases := []struct {
		name          string
		tweakEndpoint func(ep *v1.Endpoints)
		expectedSlice discovery.EndpointSlice
	}{
		{
			name: "create slice from endpoints",
			tweakEndpoint: func(ep *v1.Endpoints) {
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with annotations",
			tweakEndpoint: func(ep *v1.Endpoints) {
				annotations := map[string]string{"foo": "bar"}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo": "bar"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels",
			tweakEndpoint: func(ep *v1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels and annotations",
			tweakEndpoint: func(ep *v1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
				annotations := map[string]string{"foo2": "bar2"}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo2": "bar2"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
		{
			name: "create slice from endpoints with labels and annotations triggertime",
			tweakEndpoint: func(ep *v1.Endpoints) {
				labels := map[string]string{"foo": "bar"}
				ep.Labels = labels
				annotations := map[string]string{
					"foo2":                            "bar2",
					v1.EndpointsLastChangeTriggerTime: "date",
				}
				ep.Annotations = annotations
			},
			expectedSlice: discovery.EndpointSlice{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo":                      "bar",
						discovery.LabelServiceName: endpoints.Name,
						discovery.LabelManagedBy:   controllerName,
					},
					Annotations:     map[string]string{"foo2": "bar2"},
					GenerateName:    fmt.Sprintf("%s-", endpoints.Name),
					Namespace:       endpoints.Namespace,
					OwnerReferences: []metav1.OwnerReference{*ownerRef},
				},
				Ports:       ports,
				AddressType: addrType,
				Endpoints:   []discovery.Endpoint{},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ep := endpoints.DeepCopy()
			tc.tweakEndpoint(ep)
			generatedSlice := newEndpointSlice(ep, ports, addrType, "")
			assert.EqualValues(t, tc.expectedSlice, *generatedSlice)
			if len(endpoints.Labels) > 1 {
				t.Errorf("Expected Endpoints labels to not be modified, got %+v", endpoints.Labels)
			}
		})
	}
}

func TestAddressToEndpoint(t *testing.T) {
	//name: "simple + gate enabled",
	epAddress := v1.EndpointAddress{
		IP:       "10.1.2.3",
		Hostname: "foo",
		NodeName: ptr.To("node-abc"),
		TargetRef: &v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Namespace:  "default",
			Name:       "foo",
		},
	}
	ready := true
	expectedEndpoint := discovery.Endpoint{
		Addresses: []string{"10.1.2.3"},
		Hostname:  ptr.To("foo"),
		Conditions: discovery.EndpointConditions{
			Ready: ptr.To(true),
		},
		TargetRef: &v1.ObjectReference{
			APIVersion: "v1",
			Kind:       "Pod",
			Namespace:  "default",
			Name:       "foo",
		},
		NodeName: ptr.To("node-abc"),
	}

	ep := addressToEndpoint(epAddress, ready)
	assert.EqualValues(t, expectedEndpoint, *ep)
}

// Test helpers

func newClientset() *fake.Clientset {
	client := fake.NewSimpleClientset()

	client.PrependReactor("create", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)

		if endpointSlice.ObjectMeta.GenerateName != "" {
			endpointSlice.ObjectMeta.Name = fmt.Sprintf("%s-%s", endpointSlice.ObjectMeta.GenerateName, rand.String(8))
			endpointSlice.ObjectMeta.GenerateName = ""
		}
		endpointSlice.ObjectMeta.Generation = 1

		return false, endpointSlice, nil
	}))
	client.PrependReactor("update", "endpointslices", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		endpointSlice := action.(k8stesting.CreateAction).GetObject().(*discovery.EndpointSlice)
		endpointSlice.ObjectMeta.Generation++
		return false, endpointSlice, nil
	}))

	return client
}
