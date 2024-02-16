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
	"fmt"

	corev1 "k8s.io/api/core/v1"
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
	supportedAddressTypes = sets.NewString(
		string(discovery.AddressTypeIPv4),
		string(discovery.AddressTypeIPv6),
		string(discovery.AddressTypeFQDN),
	)
	supportedPortProtocols = sets.NewString(
		string(api.ProtocolTCP),
		string(api.ProtocolUDP),
		string(api.ProtocolSCTP),
	)
	maxTopologyLabels = 16
	maxAddresses      = 100
	maxPorts          = 20000
	maxEndpoints      = 1000
	maxZoneHints      = 8
)

// ValidateEndpointSliceName can be used to check whether the given endpoint
// slice name is valid. Prefix indicates this name will be used as part of
// generation, in which case trailing dashes are allowed.
var ValidateEndpointSliceName = apimachineryvalidation.NameIsDNSSubdomain

// ValidateEndpointSlice validates an EndpointSlice.
func ValidateEndpointSlice(endpointSlice *discovery.EndpointSlice) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMeta(&endpointSlice.ObjectMeta, true, ValidateEndpointSliceName, field.NewPath("metadata"))
	allErrs = append(allErrs, validateAddressType(endpointSlice.AddressType)...)
	allErrs = append(allErrs, validateEndpoints(endpointSlice.Endpoints, endpointSlice.AddressType, field.NewPath("endpoints"))...)
	allErrs = append(allErrs, validatePorts(endpointSlice.Ports, field.NewPath("ports"))...)

	return allErrs
}

// ValidateEndpointSliceCreate validates an EndpointSlice when it is created.
func ValidateEndpointSliceCreate(endpointSlice *discovery.EndpointSlice) field.ErrorList {
	return ValidateEndpointSlice(endpointSlice)
}

// ValidateEndpointSliceUpdate validates an EndpointSlice when it is updated.
func ValidateEndpointSliceUpdate(newEndpointSlice, oldEndpointSlice *discovery.EndpointSlice) field.ErrorList {
	allErrs := ValidateEndpointSlice(newEndpointSlice)
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(newEndpointSlice.AddressType, oldEndpointSlice.AddressType, field.NewPath("addressType"))...)

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

		if len(endpoint.Addresses) == 0 {
			allErrs = append(allErrs, field.Required(addressPath, "must contain at least 1 address"))
		} else if len(endpoint.Addresses) > maxAddresses {
			allErrs = append(allErrs, field.TooMany(addressPath, len(endpoint.Addresses), maxAddresses))
		}

		for i, address := range endpoint.Addresses {
			// This validates known address types, unknown types fall through
			// and do not get validated.
			switch addrType {
			case discovery.AddressTypeIPv4:
				allErrs = append(allErrs, validation.IsValidIPv4Address(addressPath.Index(i), address)...)
				allErrs = append(allErrs, apivalidation.ValidateEndpointIP(address, addressPath.Index(i))...)
			case discovery.AddressTypeIPv6:
				allErrs = append(allErrs, validation.IsValidIPv6Address(addressPath.Index(i), address)...)
				allErrs = append(allErrs, apivalidation.ValidateEndpointIP(address, addressPath.Index(i))...)
			case discovery.AddressTypeFQDN:
				allErrs = append(allErrs, validation.IsFullyQualifiedDomainName(addressPath.Index(i), address)...)
			}
		}

		if endpoint.NodeName != nil {
			nnPath := idxPath.Child("nodeName")
			for _, msg := range apivalidation.ValidateNodeName(*endpoint.NodeName, false) {
				allErrs = append(allErrs, field.Invalid(nnPath, *endpoint.NodeName, msg))
			}
		}

		topologyPath := idxPath.Child("deprecatedTopology")
		if len(endpoint.DeprecatedTopology) > maxTopologyLabels {
			allErrs = append(allErrs, field.TooMany(topologyPath, len(endpoint.DeprecatedTopology), maxTopologyLabels))
		}
		allErrs = append(allErrs, metavalidation.ValidateLabels(endpoint.DeprecatedTopology, topologyPath)...)
		if _, found := endpoint.DeprecatedTopology[corev1.LabelTopologyZone]; found {
			allErrs = append(allErrs, field.InternalError(topologyPath.Key(corev1.LabelTopologyZone), fmt.Errorf("reserved key was not removed in conversion")))
		}

		if endpoint.Hostname != nil {
			allErrs = append(allErrs, apivalidation.ValidateDNS1123Label(*endpoint.Hostname, idxPath.Child("hostname"))...)
		}

		if endpoint.Hints != nil {
			allErrs = append(allErrs, validateHints(endpoint.Hints, idxPath.Child("hints"))...)
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
			allErrs = append(allErrs, apivalidation.ValidateDNS1123Label(*endpointPort.Name, idxPath.Child("name"))...)
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

		if endpointPort.AppProtocol != nil {
			allErrs = append(allErrs, apivalidation.ValidateQualifiedName(*endpointPort.AppProtocol, idxPath.Child("appProtocol"))...)
		}
	}

	return allErrs
}

func validateAddressType(addressType discovery.AddressType) field.ErrorList {
	allErrs := field.ErrorList{}

	if addressType == "" {
		allErrs = append(allErrs, field.Required(field.NewPath("addressType"), ""))
	} else if !supportedAddressTypes.Has(string(addressType)) {
		allErrs = append(allErrs, field.NotSupported(field.NewPath("addressType"), addressType, supportedAddressTypes.List()))
	}

	return allErrs
}

func validateHints(endpointHints *discovery.EndpointHints, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	fzPath := fldPath.Child("forZones")
	if len(endpointHints.ForZones) > maxZoneHints {
		allErrs = append(allErrs, field.TooMany(fzPath, len(endpointHints.ForZones), maxZoneHints))
		return allErrs
	}

	zoneNames := sets.String{}
	for i, forZone := range endpointHints.ForZones {
		zonePath := fzPath.Index(i).Child("name")
		if zoneNames.Has(forZone.Name) {
			allErrs = append(allErrs, field.Duplicate(zonePath, forZone.Name))
		} else {
			zoneNames.Insert(forZone.Name)
		}

		for _, msg := range validation.IsValidLabelValue(forZone.Name) {
			allErrs = append(allErrs, field.Invalid(zonePath, forZone.Name, msg))
		}
	}

	return allErrs
}
