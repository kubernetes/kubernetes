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
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/util/sets"
	endpointsliceutil "k8s.io/endpointslice/util"
	"k8s.io/klog/v2"
	endpointsv1 "k8s.io/kubernetes/pkg/api/v1/endpoints"
	netutils "k8s.io/utils/net"
)

func DesiredEndpointSlicesFromEndpoints(
	endpoints *corev1.Endpoints,
	maxEndpointsPerSubset int32,
) ([]*EndpointPortAddressType, sets.Set[discovery.AddressType]) {
	desiredEndpointSlicesByAddrTypePort := map[endpointsliceutil.PortMapKey]*EndpointPortAddressType{}
	supportedAddressesTypes := sets.New[discovery.AddressType]()

	subsets := endpointsv1.RepackSubsets(endpoints.Subsets)

	for _, subset := range subsets {
		totalAddressesAdded := 0
		endpointPorts := epPortsToEpsPorts(subset.Ports)

		for _, address := range subset.Addresses {
			// Break if we've reached the max number of addresses to mirror
			// per EndpointSubset. This allows for a simple 1:1 mapping between
			// EndpointSubset and EndpointSlice.
			if totalAddressesAdded >= int(maxEndpointsPerSubset) {
				break
			}

			addrType := getAddressType(address.IP)
			if addrType == nil {
				continue
			}

			epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(endpointPorts), *addrType)
			endpointPortAddrType, exists := desiredEndpointSlicesByAddrTypePort[epHash]
			if !exists {
				endpointPortAddrType = &EndpointPortAddressType{
					EndpointSet: endpointsliceutil.EndpointSet{},
					Ports:       endpointPorts,
					AddressType: *addrType,
				}
				desiredEndpointSlicesByAddrTypePort[epHash] = endpointPortAddrType
			}

			totalAddressesAdded++
			supportedAddressesTypes.Insert(*addrType)
			endpointPortAddrType.EndpointSet.Insert(addressToEndpoint(address, true))
		}

		for _, address := range subset.NotReadyAddresses {
			// Break if we've reached the max number of addresses to mirror
			// per EndpointSubset. This allows for a simple 1:1 mapping between
			// EndpointSubset and EndpointSlice.
			if totalAddressesAdded >= int(maxEndpointsPerSubset) {
				break
			}

			addrType := getAddressType(address.IP)
			if addrType == nil {
				continue
			}

			epHash := newAddrTypePortMapKey(endpointsliceutil.NewPortMapKey(endpointPorts), *addrType)
			endpointPortAddrType, exists := desiredEndpointSlicesByAddrTypePort[epHash]
			if !exists {
				endpointPortAddrType = &EndpointPortAddressType{
					EndpointSet: endpointsliceutil.EndpointSet{},
					Ports:       endpointPorts,
					AddressType: *addrType,
				}
				desiredEndpointSlicesByAddrTypePort[epHash] = endpointPortAddrType
			}

			totalAddressesAdded++
			supportedAddressesTypes.Insert(*addrType)
			endpointPortAddrType.EndpointSet.Insert(addressToEndpoint(address, false))
		}
	}

	desiredEndpointSlices := []*EndpointPortAddressType{}
	for _, endpointPortAddrType := range desiredEndpointSlicesByAddrTypePort {
		desiredEndpointSlices = append(desiredEndpointSlices, endpointPortAddrType)
	}

	return desiredEndpointSlices, supportedAddressesTypes
}

type LabelsAnnotationsFromEndpoints struct {
	Endpoints *corev1.Endpoints
}

func (lafe *LabelsAnnotationsFromEndpoints) SetLabelsAnnotations(logger klog.Logger, epSlice *discovery.EndpointSlice, controllerName string) (map[string]string, map[string]string, bool) {
	updated := false

	// generated slices must mirror all endpoints annotations but EndpointsLastChangeTriggerTime and LastAppliedConfigAnnotation
	epAnnotations := cloneAndRemoveKeys(lafe.Endpoints.Annotations, corev1.EndpointsLastChangeTriggerTime, corev1.LastAppliedConfigAnnotation)
	epLabels := cloneAndRemoveKeys(lafe.Endpoints.Labels, discovery.LabelManagedBy, discovery.LabelServiceName)
	epLabels[discovery.LabelServiceName] = lafe.Endpoints.Name
	epLabels[discovery.LabelManagedBy] = controllerName

	// if the labels are not identical update the slice with the corresponding service labels
	if !apiequality.Semantic.DeepEqual(epLabels, epSlice.Labels) || !apiequality.Semantic.DeepEqual(epAnnotations, epSlice.Annotations) {
		updated = true
	}

	return epLabels, epAnnotations, updated
}

// epPortsToEpsPorts converts ports from an Endpoints resource to ports for an
// EndpointSlice resource.
func epPortsToEpsPorts(epPorts []corev1.EndpointPort) []discovery.EndpointPort {
	epsPorts := []discovery.EndpointPort{}
	for _, epPort := range epPorts {
		epp := epPort.DeepCopy()
		epsPorts = append(epsPorts, discovery.EndpointPort{
			Name:        &epp.Name,
			Port:        &epp.Port,
			Protocol:    &epp.Protocol,
			AppProtocol: epp.AppProtocol,
		})
	}
	return epsPorts
}

func getAddressType(address string) *discovery.AddressType {
	ip := netutils.ParseIPSloppy(address)
	if ip == nil {
		return nil
	}
	addressType := discovery.AddressTypeIPv4
	if ip.To4() == nil {
		addressType = discovery.AddressTypeIPv6
	}
	return &addressType
}

// cloneAndRemoveKeys is a copy of CloneAndRemoveLabels
// it is used here for annotations and labels
func cloneAndRemoveKeys(a map[string]string, keys ...string) map[string]string {
	if len(keys) == 0 {
		// Don't need to remove a key.
		return a
	}
	// Clone.
	newMap := map[string]string{}
	for k, v := range a {
		newMap[k] = v
	}
	// remove keys
	for _, key := range keys {
		delete(newMap, key)
	}
	return newMap
}

// addressToEndpoint converts an address from an Endpoints resource to an
// EndpointSlice endpoint.
func addressToEndpoint(address corev1.EndpointAddress, ready bool) *discovery.Endpoint {
	endpoint := &discovery.Endpoint{
		Addresses: []string{address.IP},
		Conditions: discovery.EndpointConditions{
			Ready: &ready,
		},
		TargetRef: address.TargetRef,
	}

	if address.NodeName != nil {
		endpoint.NodeName = address.NodeName
	}
	if address.Hostname != "" {
		endpoint.Hostname = &address.Hostname
	}

	return endpoint
}
