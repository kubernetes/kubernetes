/*
Copyright 2022 The Kubernetes Authors.

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

package reconcilers

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

func makeEndpointsArray(name string, ips []string, ports []corev1.EndpointPort) []runtime.Object {
	return []runtime.Object{
		makeEndpoints(name, ips, ports),
		makeEndpointSlice(name, ips, ports),
	}
}

func makeEndpoints(name string, ips []string, ports []corev1.EndpointPort) *corev1.Endpoints {
	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      name,
			Labels: map[string]string{
				discoveryv1.LabelSkipMirror: "true",
			},
		},
	}
	if len(ips) > 0 || len(ports) > 0 {
		endpoints.Subsets = []corev1.EndpointSubset{{
			Addresses: make([]corev1.EndpointAddress, len(ips)),
			Ports:     ports,
		}}
		for i := range ips {
			endpoints.Subsets[0].Addresses[i].IP = ips[i]
		}
	}
	return endpoints
}

func makeEndpointSlice(name string, ips []string, ports []corev1.EndpointPort) *discoveryv1.EndpointSlice {
	slice := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: metav1.NamespaceDefault,
			Name:      name,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: name,
			},
		},
		AddressType: discoveryv1.AddressTypeIPv4,
		Endpoints:   make([]discoveryv1.Endpoint, len(ips)),
		Ports:       make([]discoveryv1.EndpointPort, len(ports)),
	}
	ready := true
	for i := range ips {
		slice.Endpoints[i].Addresses = []string{ips[i]}
		slice.Endpoints[i].Conditions.Ready = &ready
	}
	for i := range ports {
		slice.Ports[i].Name = &ports[i].Name
		slice.Ports[i].Protocol = &ports[i].Protocol
		slice.Ports[i].Port = &ports[i].Port
	}
	return slice
}

func verifyCreatesAndUpdates(fakeClient *fake.Clientset, expectedCreates, expectedUpdates []runtime.Object) error {
	errors := []error{}

	updates := []k8stesting.UpdateAction{}
	creates := []k8stesting.CreateAction{}
	for _, action := range fakeClient.Actions() {
		if action.GetVerb() == "update" {
			updates = append(updates, action.(k8stesting.UpdateAction))
		} else if action.GetVerb() == "create" {
			creates = append(creates, action.(k8stesting.CreateAction))
		}
	}

	if len(creates) != len(expectedCreates) {
		errors = append(errors, fmt.Errorf("expected %d creates got %d", len(expectedCreates), len(creates)))
	}
	for i := 0; i < len(creates) || i < len(expectedCreates); i++ {
		var expected, actual runtime.Object
		if i < len(creates) {
			actual = creates[i].GetObject()
		}
		if i < len(expectedCreates) {
			expected = expectedCreates[i]
		}
		if !apiequality.Semantic.DeepEqual(expected, actual) {
			errors = append(errors, fmt.Errorf("expected create %d to be:\n%#v\ngot:\n%#v\n", i, expected, actual))
		}
	}

	if len(updates) != len(expectedUpdates) {
		errors = append(errors, fmt.Errorf("expected %d updates got %d", len(expectedUpdates), len(updates)))
	}
	for i := 0; i < len(updates) || i < len(expectedUpdates); i++ {
		var expected, actual runtime.Object
		if i < len(updates) {
			actual = updates[i].GetObject()
		}
		if i < len(expectedUpdates) {
			expected = expectedUpdates[i]
		}
		if !apiequality.Semantic.DeepEqual(expected, actual) {
			errors = append(errors, fmt.Errorf("expected update %d to be:\n%#v\ngot:\n%#v\n", i, expected, actual))
		}
	}

	return utilerrors.NewAggregate(errors)
}
