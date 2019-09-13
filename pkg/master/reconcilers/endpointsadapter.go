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

package reconcilers

import (
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryclient "k8s.io/client-go/kubernetes/typed/discovery/v1alpha1"
)

// EndpointsAdapter provides a simple interface for reading and writing both
// Endpoints and Endpoint Slices.
// NOTE: This is an incomplete adapter implementation that is only suitable for
// use in this package. This takes advantage of the Endpoints used in this
// package always having a consistent set of ports, a single subset, and a small
// set of addresses. Any more complex Endpoints resource would likely translate
// into multiple Endpoint Slices creating significantly more complexity instead
// of the 1:1 mapping this allows.
type EndpointsAdapter struct {
	endpointClient      corev1client.EndpointsGetter
	endpointSliceClient discoveryclient.EndpointSlicesGetter
}

// NewEndpointsAdapter returns a new EndpointsAdapter.
func NewEndpointsAdapter(endpointClient corev1client.EndpointsGetter, endpointSliceClient discoveryclient.EndpointSlicesGetter) EndpointsAdapter {
	return EndpointsAdapter{
		endpointClient:      endpointClient,
		endpointSliceClient: endpointSliceClient,
	}
}

// Get takes the name and namespace of the Endpoints resource, and returns a
// corresponding Endpoints object if it exists, and an error if there is any.
func (adapter *EndpointsAdapter) Get(namespace, name string, getOpts metav1.GetOptions) (*corev1.Endpoints, error) {
	return adapter.endpointClient.Endpoints(namespace).Get(name, getOpts)
}

// Create accepts a namespace and Endpoints object and creates the Endpoints
// object. If an endpointSliceClient exists, a matching EndpointSlice will also
// be created or updated. The created Endpoints object or an error will be
// returned.
func (adapter *EndpointsAdapter) Create(namespace string, endpoints *corev1.Endpoints) (*corev1.Endpoints, error) {
	endpoints, err := adapter.endpointClient.Endpoints(namespace).Create(endpoints)
	if err == nil && adapter.endpointSliceClient != nil {
		_, err = adapter.ensureEndpointSliceFromEndpoints(namespace, endpoints)
	}
	return endpoints, err
}

// Update accepts a namespace and Endpoints object and updates it. If an
// endpointSliceClient exists, a matching EndpointSlice will also be created or
// updated. The updated Endpoints object or an error will be returned.
func (adapter *EndpointsAdapter) Update(namespace string, endpoints *corev1.Endpoints) (*corev1.Endpoints, error) {
	endpoints, err := adapter.endpointClient.Endpoints(namespace).Update(endpoints)
	if err == nil && adapter.endpointSliceClient != nil {
		_, err = adapter.ensureEndpointSliceFromEndpoints(namespace, endpoints)
	}
	return endpoints, err
}

// ensureEndpointSliceFromEndpoints accepts a namespace and Endpoints resource
// and creates or updates a corresponding EndpointSlice. The EndpointSlice
// and/or an error will be returned.
func (adapter *EndpointsAdapter) ensureEndpointSliceFromEndpoints(namespace string, endpoints *corev1.Endpoints) (*discovery.EndpointSlice, error) {
	endpointSlice := endpointSliceFromEndpoints(endpoints)
	_, err := adapter.endpointSliceClient.EndpointSlices(namespace).Get(endpointSlice.Name, metav1.GetOptions{})

	if err != nil {
		if errors.IsNotFound(err) {
			return adapter.endpointSliceClient.EndpointSlices(namespace).Create(endpointSlice)
		}
		return nil, err
	}

	return adapter.endpointSliceClient.EndpointSlices(namespace).Update(endpointSlice)
}

// endpointSliceFromEndpoints generates an EndpointSlice from an Endpoints
// resource.
func endpointSliceFromEndpoints(endpoints *corev1.Endpoints) *discovery.EndpointSlice {
	endpointSlice := &discovery.EndpointSlice{}
	endpointSlice.Name = endpoints.Name
	endpointSlice.Labels = map[string]string{discovery.LabelServiceName: endpoints.Name}

	ipAddressType := discovery.AddressTypeIP
	endpointSlice.AddressType = &ipAddressType

	if len(endpoints.Subsets) > 0 {
		subset := endpoints.Subsets[0]
		for i := range subset.Ports {
			endpointSlice.Ports = append(endpointSlice.Ports, discovery.EndpointPort{
				Port:     &subset.Ports[i].Port,
				Name:     &subset.Ports[i].Name,
				Protocol: &subset.Ports[i].Protocol,
			})
		}
		for _, address := range subset.Addresses {
			endpointSlice.Endpoints = append(endpointSlice.Endpoints, endpointFromAddress(address, true))
		}
		for _, address := range subset.NotReadyAddresses {
			endpointSlice.Endpoints = append(endpointSlice.Endpoints, endpointFromAddress(address, false))
		}
	}

	return endpointSlice
}

// endpointFromAddress generates an Endpoint from an EndpointAddress resource.
func endpointFromAddress(address corev1.EndpointAddress, ready bool) discovery.Endpoint {
	topology := map[string]string{}
	if address.NodeName != nil {
		topology["kubernetes.io/hostname"] = *address.NodeName
	}

	return discovery.Endpoint{
		Addresses:  []string{address.IP},
		Conditions: discovery.EndpointConditions{Ready: &ready},
		TargetRef:  address.TargetRef,
		Topology:   topology,
	}
}
