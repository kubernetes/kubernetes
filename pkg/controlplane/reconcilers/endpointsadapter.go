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
	"context"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryclient "k8s.io/client-go/kubernetes/typed/discovery/v1"
	utilnet "k8s.io/utils/net"
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
	return adapter.endpointClient.Endpoints(namespace).Get(context.TODO(), name, getOpts)
}

// Create accepts a namespace and Endpoints object and creates the Endpoints
// object and matching EndpointSlice. The created Endpoints object or an error will be
// returned.
func (adapter *EndpointsAdapter) Create(namespace string, endpoints *corev1.Endpoints) (*corev1.Endpoints, error) {
	endpoints, err := adapter.endpointClient.Endpoints(namespace).Create(context.TODO(), endpoints, metav1.CreateOptions{})
	if err == nil {
		err = adapter.EnsureEndpointSliceFromEndpoints(namespace, endpoints)
	}
	return endpoints, err
}

// Update accepts a namespace and Endpoints object and updates it and its
// matching EndpointSlice. The updated Endpoints object or an error will be returned.
func (adapter *EndpointsAdapter) Update(namespace string, endpoints *corev1.Endpoints) (*corev1.Endpoints, error) {
	endpoints, err := adapter.endpointClient.Endpoints(namespace).Update(context.TODO(), endpoints, metav1.UpdateOptions{})
	if err == nil {
		err = adapter.EnsureEndpointSliceFromEndpoints(namespace, endpoints)
	}
	return endpoints, err
}

// EnsureEndpointSliceFromEndpoints accepts a namespace and Endpoints resource
// and creates or updates a corresponding EndpointSlice. An error will be returned
// if it fails to sync the EndpointSlice.
func (adapter *EndpointsAdapter) EnsureEndpointSliceFromEndpoints(namespace string, endpoints *corev1.Endpoints) error {
	endpointSlice := endpointSliceFromEndpoints(endpoints)
	currentEndpointSlice, err := adapter.endpointSliceClient.EndpointSlices(namespace).Get(context.TODO(), endpointSlice.Name, metav1.GetOptions{})

	if err != nil {
		if errors.IsNotFound(err) {
			if _, err = adapter.endpointSliceClient.EndpointSlices(namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{}); errors.IsAlreadyExists(err) {
				err = nil
			}
		}
		return err
	}

	// required for transition from IP to IPv4 address type.
	if currentEndpointSlice.AddressType != endpointSlice.AddressType {
		err = adapter.endpointSliceClient.EndpointSlices(namespace).Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
		if err != nil {
			return err
		}
		_, err = adapter.endpointSliceClient.EndpointSlices(namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		return err
	}

	if apiequality.Semantic.DeepEqual(currentEndpointSlice.Endpoints, endpointSlice.Endpoints) &&
		apiequality.Semantic.DeepEqual(currentEndpointSlice.Ports, endpointSlice.Ports) &&
		apiequality.Semantic.DeepEqual(currentEndpointSlice.Labels, endpointSlice.Labels) {
		return nil
	}

	_, err = adapter.endpointSliceClient.EndpointSlices(namespace).Update(context.TODO(), endpointSlice, metav1.UpdateOptions{})
	return err
}

// endpointSliceFromEndpoints generates an EndpointSlice from an Endpoints
// resource.
func endpointSliceFromEndpoints(endpoints *corev1.Endpoints) *discovery.EndpointSlice {
	endpointSlice := &discovery.EndpointSlice{}
	endpointSlice.Name = endpoints.Name
	endpointSlice.Namespace = endpoints.Namespace
	endpointSlice.Labels = map[string]string{discovery.LabelServiceName: endpoints.Name}

	// TODO: Add support for dual stack here (and in the rest of
	// EndpointsAdapter).
	endpointSlice.AddressType = discovery.AddressTypeIPv4

	if len(endpoints.Subsets) > 0 {
		subset := endpoints.Subsets[0]
		for i := range subset.Ports {
			endpointSlice.Ports = append(endpointSlice.Ports, discovery.EndpointPort{
				Port:     &subset.Ports[i].Port,
				Name:     &subset.Ports[i].Name,
				Protocol: &subset.Ports[i].Protocol,
			})
		}

		if allAddressesIPv6(append(subset.Addresses, subset.NotReadyAddresses...)) {
			endpointSlice.AddressType = discovery.AddressTypeIPv6
		}

		endpointSlice.Endpoints = append(endpointSlice.Endpoints, getEndpointsFromAddresses(subset.Addresses, endpointSlice.AddressType, true)...)
		endpointSlice.Endpoints = append(endpointSlice.Endpoints, getEndpointsFromAddresses(subset.NotReadyAddresses, endpointSlice.AddressType, false)...)
	}

	return endpointSlice
}

// getEndpointsFromAddresses returns a list of Endpoints from addresses that
// match the provided address type.
func getEndpointsFromAddresses(addresses []corev1.EndpointAddress, addressType discovery.AddressType, ready bool) []discovery.Endpoint {
	endpoints := []discovery.Endpoint{}
	isIPv6AddressType := addressType == discovery.AddressTypeIPv6

	for _, address := range addresses {
		if utilnet.IsIPv6String(address.IP) == isIPv6AddressType {
			endpoints = append(endpoints, endpointFromAddress(address, ready))
		}
	}

	return endpoints
}

// endpointFromAddress generates an Endpoint from an EndpointAddress resource.
func endpointFromAddress(address corev1.EndpointAddress, ready bool) discovery.Endpoint {
	ep := discovery.Endpoint{
		Addresses:  []string{address.IP},
		Conditions: discovery.EndpointConditions{Ready: &ready},
		TargetRef:  address.TargetRef,
	}

	if address.NodeName != nil {
		ep.NodeName = address.NodeName
	}

	return ep
}

// allAddressesIPv6 returns true if all provided addresses are IPv6.
func allAddressesIPv6(addresses []corev1.EndpointAddress) bool {
	if len(addresses) == 0 {
		return false
	}

	for _, address := range addresses {
		if !utilnet.IsIPv6String(address.IP) {
			return false
		}
	}

	return true
}

// setSkipMirrorTrue sets endpointslice.kubernetes.io/skip-mirror to true. It
// returns true if this has resulted in a change to the Endpoints resource.
func setSkipMirrorTrue(e *corev1.Endpoints) bool {
	skipMirrorVal, ok := e.Labels[discovery.LabelSkipMirror]
	if !ok || skipMirrorVal != "true" {
		if e.Labels == nil {
			e.Labels = map[string]string{}
		}
		e.Labels[discovery.LabelSkipMirror] = "true"
		return true
	}
	return false
}
