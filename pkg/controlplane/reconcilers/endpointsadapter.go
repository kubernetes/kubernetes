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
	"net"

	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	discoveryclient "k8s.io/client-go/kubernetes/typed/discovery/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
	utilnet "k8s.io/utils/net"
	"k8s.io/utils/pointer"
)

// EndpointsAdapter provides a simple interface for reading and writing both
// Endpoints and EndpointSlices for an EndpointReconciler.
type EndpointsAdapter struct {
	endpointClient      corev1client.EndpointsGetter
	endpointSliceClient discoveryclient.EndpointSlicesGetter

	serviceNamespace string
	serviceName      string
	addressType      discovery.AddressType
}

// NewEndpointsAdapter returns a new EndpointsAdapter
func NewEndpointsAdapter(endpointClient corev1client.EndpointsGetter, endpointSliceClient discoveryclient.EndpointSlicesGetter, serviceNamespace, serviceName string, serviceIP net.IP) *EndpointsAdapter {
	addressType := discovery.AddressTypeIPv4
	if utilnet.IsIPv6(serviceIP) {
		addressType = discovery.AddressTypeIPv6
	}

	return &EndpointsAdapter{
		endpointClient:      endpointClient,
		endpointSliceClient: endpointSliceClient,

		serviceNamespace: serviceNamespace,
		serviceName:      serviceName,
		addressType:      addressType,
	}
}

// Get returns the IPs from the existing apiserver Endpoints/EndpointSlice objects. If an
// error (beside "not found") occurs fetching the data, that error will be returned.
func (adapter *EndpointsAdapter) Get() (sets.Set[string], error) {
	ips := sets.New[string]()

	endpoints, err := adapter.endpointClient.Endpoints(adapter.serviceNamespace).Get(context.TODO(), adapter.serviceName, metav1.GetOptions{})
	if err != nil {
		if errors.IsNotFound(err) {
			return ips, nil
		}
		return nil, err
	}

	if len(endpoints.Subsets) == 1 {
		for _, addr := range endpoints.Subsets[0].Addresses {
			ips.Insert(addr.IP)
		}
	}
	return ips, nil
}

// Sync updates the apiserver Endpoints/EndpointSlice objects with the new set of IPs. If
// reconcilePorts is true it will also ensure that the objects have the correct ports. If
// an error occurs while updating, that error will be returned.
func (adapter *EndpointsAdapter) Sync(ips sets.Set[string], endpointPorts []corev1.EndpointPort, reconcilePorts bool) error {
	var sortedIPs []string
	for _, ip := range sets.List(ips) {
		if adapter.addressType == discovery.AddressTypeIPv4 && utilnet.IsIPv4String(ip) {
			sortedIPs = append(sortedIPs, ip)
		} else if adapter.addressType == discovery.AddressTypeIPv6 && utilnet.IsIPv6String(ip) {
			sortedIPs = append(sortedIPs, ip)
		}
	}

	// Sync Endpoints
	var endpoints *corev1.Endpoints
	err := retry.OnError(retry.DefaultBackoff, isRetriableError, func() error {
		currentEndpoints, err := adapter.endpointClient.Endpoints(adapter.serviceNamespace).Get(context.TODO(), adapter.serviceName, metav1.GetOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				return err
			}
			currentEndpoints = nil
		}
		endpoints = adapter.updateEndpoints(currentEndpoints, sortedIPs, endpointPorts, reconcilePorts)

		if currentEndpoints == nil {
			_, err = adapter.endpointClient.Endpoints(endpoints.Namespace).Create(context.TODO(), endpoints, metav1.CreateOptions{})
		} else if !apiequality.Semantic.DeepEqual(currentEndpoints.Subsets, endpoints.Subsets) ||
			!apiequality.Semantic.DeepEqual(currentEndpoints.Labels, endpoints.Labels) {
			klog.Warningf("Resetting endpoints for master service %q to %v", endpoints.Name, endpoints)
			_, err = adapter.endpointClient.Endpoints(endpoints.Namespace).Update(context.TODO(), endpoints, metav1.UpdateOptions{})
		}
		return err
	})
	if err != nil {
		return err
	}

	// Sync EndpointSlice
	err = retry.OnError(retry.DefaultBackoff, isRetriableError, func() error {
		currentEndpointSlice, err := adapter.endpointSliceClient.EndpointSlices(adapter.serviceNamespace).Get(context.TODO(), adapter.serviceName, metav1.GetOptions{})
		if err != nil {
			if !errors.IsNotFound(err) {
				return err
			}
			currentEndpointSlice = nil
		}

		endpointSlice := adapter.updateEndpointSlice(currentEndpointSlice, sortedIPs, endpointPorts, reconcilePorts)

		// required for transition from IP to IPv4 address type.
		if currentEndpointSlice != nil && currentEndpointSlice.AddressType != endpointSlice.AddressType {
			err = adapter.endpointSliceClient.EndpointSlices(endpointSlice.Namespace).Delete(context.TODO(), endpointSlice.Name, metav1.DeleteOptions{})
			if err != nil && !errors.IsNotFound(err) {
				return err
			}
			currentEndpointSlice = nil
		}

		if currentEndpointSlice == nil {
			_, err = adapter.endpointSliceClient.EndpointSlices(endpointSlice.Namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		} else if !apiequality.Semantic.DeepEqual(currentEndpointSlice.Endpoints, endpointSlice.Endpoints) ||
			!apiequality.Semantic.DeepEqual(currentEndpointSlice.Ports, endpointSlice.Ports) ||
			!apiequality.Semantic.DeepEqual(currentEndpointSlice.Labels, endpointSlice.Labels) {
			_, err = adapter.endpointSliceClient.EndpointSlices(endpointSlice.Namespace).Update(context.TODO(), endpointSlice, metav1.UpdateOptions{})
		}
		return err
	})
	return err
}

// isRetriableError is used by the call to retry.RetryOnError in Sync; we want to retry if
// an Update() call returns Conflict (another apiserver updated the object we were about
// to update), or if a Create() call returns Exists (another apiserver created the object
// we were about to create). (OTOH if a Delete() call returns Not Found (another apiserver
// deleted the object we were about to delete), Sync just ignores the error and continues
// since the new state is correct anyway). Any other errors are fatal. (In particular, a
// Not Found in response to an Update() call can't occur in a correctly-configured
// cluster, so it's better to let it be logged as an error.)
func isRetriableError(err error) bool {
	return errors.IsConflict(err) || errors.IsAlreadyExists(err)
}

// updateEndpoints updates endpoints to reflect sortedIPs and (optionally) endpointPorts
func (adapter *EndpointsAdapter) updateEndpoints(currentEndpoints *corev1.Endpoints, sortedIPs []string, endpointPorts []corev1.EndpointPort, reconcilePorts bool) *corev1.Endpoints {
	var endpoints *corev1.Endpoints

	if currentEndpoints != nil {
		endpoints = currentEndpoints.DeepCopy()
	} else {
		endpoints = &corev1.Endpoints{
			ObjectMeta: metav1.ObjectMeta{
				Name:      adapter.serviceName,
				Namespace: adapter.serviceNamespace,
			},
		}
	}

	// Ensure correct labels
	endpoints.Labels = map[string]string{
		discovery.LabelSkipMirror: "true",
	}
	// Ensure correct format
	if len(endpoints.Subsets) != 1 {
		endpoints.Subsets = make([]corev1.EndpointSubset, 1)
	}

	// Set addresses
	endpoints.Subsets[0].Addresses = make([]corev1.EndpointAddress, len(sortedIPs))
	for i := range sortedIPs {
		endpoints.Subsets[0].Addresses[i].IP = sortedIPs[i]
	}

	// Set ports
	if len(endpoints.Subsets[0].Ports) == 0 || (reconcilePorts && !apiequality.Semantic.DeepEqual(endpoints.Subsets[0].Ports, endpointPorts)) {
		endpoints.Subsets[0].Ports = endpointPorts
	}

	return endpoints
}

// updateEndpointSlice takes currentEndpointSlice (which may be nil), and updates it to
// reflect ips and (optionally) endpointPorts.
func (adapter *EndpointsAdapter) updateEndpointSlice(currentEndpointSlice *discovery.EndpointSlice, sortedIPs []string, endpointPorts []corev1.EndpointPort, reconcilePorts bool) *discovery.EndpointSlice {
	var endpointSlice *discovery.EndpointSlice

	if currentEndpointSlice != nil {
		endpointSlice = currentEndpointSlice.DeepCopy()
	} else {
		endpointSlice = &discovery.EndpointSlice{
			ObjectMeta: metav1.ObjectMeta{
				Name:      adapter.serviceName,
				Namespace: adapter.serviceNamespace,
			},
		}
	}

	// Ensure correct labels
	endpointSlice.Labels = map[string]string{
		discovery.LabelServiceName: adapter.serviceName,
	}

	// Ensure correct AddressType
	endpointSlice.AddressType = adapter.addressType

	// Set addresses
	endpointSlice.Endpoints = make([]discovery.Endpoint, len(sortedIPs))
	for i := range sortedIPs {
		endpointSlice.Endpoints[i] = discovery.Endpoint{
			Addresses:  []string{sortedIPs[i]},
			Conditions: discovery.EndpointConditions{Ready: pointer.Bool(true)},
		}
	}

	// Set ports
	endpointSlicePorts := convertEndpointPorts(endpointPorts)
	if len(endpointSlice.Ports) == 0 || (reconcilePorts && !apiequality.Semantic.DeepEqual(endpointSlice.Ports, endpointSlicePorts)) {
		endpointSlice.Ports = endpointSlicePorts
	}

	return endpointSlice
}

// convertEndpointPorts converts an array of corev1.EndpointPort to discovery.EndpointPort
func convertEndpointPorts(endpointPorts []corev1.EndpointPort) []discovery.EndpointPort {
	endpointSlicePorts := make([]discovery.EndpointPort, len(endpointPorts))
	for i := range endpointPorts {
		endpointSlicePorts[i].Port = &endpointPorts[i].Port
		endpointSlicePorts[i].Name = &endpointPorts[i].Name
		endpointSlicePorts[i].Protocol = &endpointPorts[i].Protocol
	}
	return endpointSlicePorts
}
