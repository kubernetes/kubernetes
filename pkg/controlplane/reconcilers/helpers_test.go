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

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
)

const (
	testServiceNamespace = metav1.NamespaceDefault
	testServiceName      = "kubernetes"
)

func makeEndpointsArray(ips []string, ports []corev1.EndpointPort) []runtime.Object {
	return []runtime.Object{
		makeEndpoints(ips, ports),
		makeEndpointSlice(ips, ports),
	}
}

func makeEndpoints(ips []string, ports []corev1.EndpointPort) *corev1.Endpoints {
	endpoints := &corev1.Endpoints{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testServiceNamespace,
			Name:      testServiceName,
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

func makeEndpointSlice(ips []string, ports []corev1.EndpointPort) *discoveryv1.EndpointSlice {
	slice := &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testServiceNamespace,
			Name:      testServiceName,
			Labels: map[string]string{
				discoveryv1.LabelServiceName: testServiceName,
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

func verifyActions(fakeClient *fake.Clientset, expectedCreates, expectedUpdates, expectedDeletes []runtime.Object) error {
	errors := []error{}

	updates := []k8stesting.UpdateAction{}
	creates := []k8stesting.CreateAction{}
	deletes := []k8stesting.DeleteAction{}
	for _, action := range fakeClient.Actions() {
		if action.GetVerb() == "update" {
			updates = append(updates, action.(k8stesting.UpdateAction))
		} else if action.GetVerb() == "create" {
			creates = append(creates, action.(k8stesting.CreateAction))
		} else if action.GetVerb() == "delete" {
			deletes = append(deletes, action.(k8stesting.DeleteAction))
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
			errors = append(errors, fmt.Errorf("create %d has diff:\n%s\n", i+1, cmp.Diff(expected, actual)))
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
			errors = append(errors, fmt.Errorf("update %d has diff:\n%s\n", i+1, cmp.Diff(expected, actual)))
		}
	}

	if len(deletes) != len(expectedDeletes) {
		errors = append(errors, fmt.Errorf("expected %d deletes got %d", len(expectedDeletes), len(deletes)))
	}
	for i := 0; i < len(deletes) || i < len(expectedDeletes); i++ {
		// testing.DeleteAction doesn't include the actual object so we just make sure
		// that it has the expected resource type and name.
		var expected, actual string
		if i < len(deletes) {
			actual = fmt.Sprintf("%s %s/%s", deletes[i].GetResource().Resource, deletes[i].GetNamespace(), deletes[i].GetName())
		}
		if i < len(expectedDeletes) {
			metaObject := expectedDeletes[i].(metav1.Object)
			// NB: expectedDeletes[i].GetObjectKind() returns an empty ObjectKind here
			resource := "unknown type"
			switch expectedDeletes[i].(type) {
			case *corev1.Endpoints:
				resource = "endpoints"
			case *discoveryv1.EndpointSlice:
				resource = "endpointslices"
			}
			expected = fmt.Sprintf("%s %s/%s", resource, metaObject.GetNamespace(), metaObject.GetName())
		}
		if expected != actual {
			errors = append(errors, fmt.Errorf("expected delete %d to be %q got %q\n", i, expected, actual))
		}
	}

	return utilerrors.NewAggregate(errors)
}
