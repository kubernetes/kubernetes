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

package validation

import (
	apimachineryvalidation "k8s.io/apimachinery/pkg/api/validation"
	metavalidation "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/discovery"
)

var (
	supportedAddressTypes  = sets.NewString(string(discovery.AddressTypeIP))
	supportedPortProtocols = sets.NewString(string(api.ProtocolTCP), string(api.ProtocolUDP), string(api.ProtocolSCTP))
	maxTopologyLabels      = 16
	maxAddresses           = 100
	maxPorts               = 100
	maxEndpoints           = 1000
)

// ValidateEndpointSliceName can be used to check whether the given endpoint
// slice name is valid. Prefix indicates this name will be used as part of
// generation, in which case trailing dashes are allowed.
var ValidateEndpointSliceName = apimachineryvalidation.NameIsDNSSubdomain

// ValidateEndpointSlice validates an EndpointSlice.
func ValidateEndpointSlice(endpointSlice *discovery.EndpointSlice) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&endpointSlice.ObjectMeta, true, ValidateEndpointSliceName, field.NewPath("metadata"))

	addrType := discovery.AddressType("")
	if endpointSlice.AddressType == nil {
		allErrs = append(allErrs, field.Required(field.NewPath("addressType"), ""))
	} else {
		addrType = *endpointSlice.AddressType
	}

	if endpointSlice.AddressType != nil && !supportedAddressTypes.Has(string(*endpointSlice.AddressType)) {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("addressType"), *endpointSlice.AddressType, supportedAddressTypes.List()))
	}

	allErrs = append(allErrs, validateEndpoints(endpointSlice.Endpoints, addrType, field.NewPath("endpoints"))...)
	allErrs = append(allErrs, validatePorts(endpointSlice.Ports, field.NewPath("ports"))...)

	return allErrs
}

// ValidateEndpointSliceUpdate validates an EndpointSlice when it is updated.
func ValidateEndpointSliceUpdate(newEndpointSlice, oldEndpointSlice *discovery.EndpointSlice) field.ErrorList {
	allErrs := ValidateEndpointSlice(newEndpointSlice)

	allErrs = append(allErrs, apivalidation.ValidateImmutableField(*newEndpointSlice.AddressType, *oldEndpointSlice.AddressType, field.NewPath("addressType"))...)

	return allErrs
}

func validateEndpoints(endpoints []discovery.Endpoint, addrType discovery.AddressType, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(endpoints) > maxEndpoints {
		allErrs = append(allErrs, field.TooMany(fldPath, len(endpoints), maxEndpoints))
		return allErrs
	}

	for i, endpoint := range endpoints {
		idxPath := fldPath.Index(i)
		addressPath := idxPath.Child("addresses")

		if addrType == discovery.AddressTypeIP {
			if len(endpoint.Addresses) == 0 {
				allErrs = append(allErrs, field.Required(addressPath, "must contain at least 1 address"))
			} else if len(endpoint.Addresses) > maxAddresses {
				allErrs = append(allErrs, field.TooMany(addressPath, len(endpoint.Addresses), maxAddresses))
			}

			for i, address := range endpoint.Addresses {
				for _, msg := range validation.IsValidIP(address) {
					allErrs = append(allErrs, field.Invalid(addressPath.Index(i), address, msg))
				}
			}
		}

		topologyPath := idxPath.Child("topology")
		if len(endpoint.Topology) > maxTopologyLabels {
			allErrs = append(allErrs, field.TooMany(topologyPath, len(endpoint.Topology), maxTopologyLabels))
		}
		allErrs = append(allErrs, metavalidation.ValidateLabels(endpoint.Topology, topologyPath)...)

		if endpoint.Hostname != nil {
			for _, msg := range validation.IsDNS1123Label(*endpoint.Hostname) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("hostname"), *endpoint.Hostname, msg))
			}
		}
	}

	return allErrs
}

func validatePorts(endpointPorts []discovery.EndpointPort, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(endpointPorts) > maxPorts {
		allErrs = append(allErrs, field.TooMany(fldPath, len(endpointPorts), maxPorts))
		return allErrs
	}

	portNames := sets.String{}
	for i, endpointPort := range endpointPorts {
		idxPath := fldPath.Index(i)

		if len(*endpointPort.Name) > 0 {
			for _, msg := range validation.IsValidPortName(*endpointPort.Name) {
				allErrs = append(allErrs, field.Invalid(idxPath.Child("name"), endpointPort.Name, msg))
			}
		}

		if portNames.Has(*endpointPort.Name) {
			allErrs = append(allErrs, field.Duplicate(idxPath.Child("name"), endpointPort.Name))
		} else {
			portNames.Insert(*endpointPort.Name)
		}

		if endpointPort.Protocol == nil {
			allErrs = append(allErrs, field.Required(idxPath.Child("protocol"), ""))
		} else if !supportedPortProtocols.Has(string(*endpointPort.Protocol)) {
			allErrs = append(allErrs, field.NotSupported(idxPath.Child("protocol"), *endpointPort.Protocol, supportedPortProtocols.List()))
		}
	}

	return allErrs
}
