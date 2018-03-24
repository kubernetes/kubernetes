/*
Copyright 2014 The Kubernetes Authors.

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
	"net"

	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/core"
)

// ValidateEndpointsName can be used to check whether the given endpoints name is valid.
// Prefix indicates this name will be used as part of generation, in which case
// trailing dashes are allowed.
var ValidateEndpointsName = NameIsDNSSubdomain

// ValidateEndpoints tests if required fields are set.
func ValidateEndpoints(endpoints *core.Endpoints) field.ErrorList {
	allErrs := ValidateObjectMeta(&endpoints.ObjectMeta, true, ValidateEndpointsName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEndpointsSpecificAnnotations(endpoints.Annotations, field.NewPath("annotations"))...)
	allErrs = append(allErrs, validateEndpointSubsets(endpoints.Subsets, []core.EndpointSubset{}, field.NewPath("subsets"))...)
	return allErrs
}

// ValidateEndpointsUpdate tests to make sure an endpoints update can be applied.
func ValidateEndpointsUpdate(newEndpoints, oldEndpoints *core.Endpoints) field.ErrorList {
	allErrs := ValidateObjectMetaUpdate(&newEndpoints.ObjectMeta, &oldEndpoints.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, validateEndpointSubsets(newEndpoints.Subsets, oldEndpoints.Subsets, field.NewPath("subsets"))...)
	allErrs = append(allErrs, ValidateEndpointsSpecificAnnotations(newEndpoints.Annotations, field.NewPath("annotations"))...)
	return allErrs
}

func ValidateEndpointsSpecificAnnotations(annotations map[string]string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func validateEndpointSubsets(subsets []core.EndpointSubset, oldSubsets []core.EndpointSubset, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	ipToNodeName := buildEndpointAddressNodeNameMap(oldSubsets)
	for i := range subsets {
		ss := &subsets[i]
		idxPath := fldPath.Index(i)

		// EndpointSubsets must include endpoint address. For headless service, we allow its endpoints not to have ports.
		if len(ss.Addresses) == 0 && len(ss.NotReadyAddresses) == 0 {
			//TODO: consider adding a RequiredOneOf() error for this and similar cases
			allErrs = append(allErrs, field.Required(idxPath, "must specify `addresses` or `notReadyAddresses`"))
		}
		for addr := range ss.Addresses {
			allErrs = append(allErrs, validateEndpointAddress(&ss.Addresses[addr], idxPath.Child("addresses").Index(addr), ipToNodeName)...)
		}
		for addr := range ss.NotReadyAddresses {
			allErrs = append(allErrs, validateEndpointAddress(&ss.NotReadyAddresses[addr], idxPath.Child("notReadyAddresses").Index(addr), ipToNodeName)...)
		}
		for port := range ss.Ports {
			allErrs = append(allErrs, validateEndpointPort(&ss.Ports[port], len(ss.Ports) > 1, idxPath.Child("ports").Index(port))...)
		}
	}

	return allErrs
}

func validateEndpointAddress(address *core.EndpointAddress, fldPath *field.Path, ipToNodeName map[string]string) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, msg := range validation.IsValidIP(address.IP) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("ip"), address.IP, msg))
	}
	if len(address.Hostname) > 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(address.Hostname, fldPath.Child("hostname"))...)
	}
	// During endpoint update, verify that NodeName is a DNS subdomain and transition rules allow the update
	if address.NodeName != nil {
		for _, msg := range ValidateNodeName(*address.NodeName, false) {
			allErrs = append(allErrs, field.Invalid(fldPath.Child("nodeName"), *address.NodeName, msg))
		}
	}
	allErrs = append(allErrs, validateEpAddrNodeNameTransition(address, ipToNodeName, fldPath.Child("nodeName"))...)
	if len(allErrs) > 0 {
		return allErrs
	}
	allErrs = append(allErrs, validateNonSpecialIP(address.IP, fldPath.Child("ip"))...)
	return allErrs
}

func validateNonSpecialIP(ipAddress string, fldPath *field.Path) field.ErrorList {
	// We disallow some IPs as endpoints or external-ips.  Specifically,
	// unspecified and loopback addresses are nonsensical and link-local
	// addresses tend to be used for node-centric purposes (e.g. metadata
	// service).
	allErrs := field.ErrorList{}
	ip := net.ParseIP(ipAddress)
	if ip == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "must be a valid IP address"))
		return allErrs
	}
	if ip.IsUnspecified() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be unspecified (0.0.0.0)"))
	}
	if ip.IsLoopback() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the loopback range (127.0.0.0/8)"))
	}
	if ip.IsLinkLocalUnicast() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the link-local range (169.254.0.0/16)"))
	}
	if ip.IsLinkLocalMulticast() {
		allErrs = append(allErrs, field.Invalid(fldPath, ipAddress, "may not be in the link-local multicast range (224.0.0.0/24)"))
	}
	return allErrs
}

func validateEndpointPort(port *core.EndpointPort, requireName bool, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if requireName && len(port.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), ""))
	} else if len(port.Name) != 0 {
		allErrs = append(allErrs, ValidateDNS1123Label(port.Name, fldPath.Child("name"))...)
	}
	for _, msg := range validation.IsValidPortNum(int(port.Port)) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("port"), port.Port, msg))
	}
	if len(port.Protocol) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("protocol"), ""))
	} else if !supportedPortProtocols.Has(string(port.Protocol)) {
		allErrs = append(allErrs, field.NotSupported(fldPath.Child("protocol"), port.Protocol, supportedPortProtocols.List()))
	}
	return allErrs
}

// Construct lookup map of old subset IPs to NodeNames.
func updateEpAddrToNodeNameMap(ipToNodeName map[string]string, addresses []core.EndpointAddress) {
	for n := range addresses {
		if addresses[n].NodeName == nil {
			continue
		}
		ipToNodeName[addresses[n].IP] = *addresses[n].NodeName
	}
}

// Build a map across all subsets of IP -> NodeName
func buildEndpointAddressNodeNameMap(subsets []core.EndpointSubset) map[string]string {
	ipToNodeName := make(map[string]string)
	for i := range subsets {
		updateEpAddrToNodeNameMap(ipToNodeName, subsets[i].Addresses)
		updateEpAddrToNodeNameMap(ipToNodeName, subsets[i].NotReadyAddresses)
	}
	return ipToNodeName
}

func validateEpAddrNodeNameTransition(addr *core.EndpointAddress, ipToNodeName map[string]string, fldPath *field.Path) field.ErrorList {
	errList := field.ErrorList{}
	existingNodeName, found := ipToNodeName[addr.IP]
	if !found {
		return errList
	}
	if addr.NodeName == nil || *addr.NodeName == existingNodeName {
		return errList
	}
	// NodeName entry found for this endpoint IP, but user is attempting to change NodeName
	return append(errList, field.Forbidden(fldPath, fmt.Sprintf("Cannot change NodeName for %s to %s", addr.IP, *addr.NodeName)))
}
