/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"net"

	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
)

func ValidateMasterConfiguration(c *kubeadm.MasterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	return allErrs
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	return allErrs
}

func ValidateDiscovery(c *kubeadm.Discovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	var count int
	if c.Token != nil {
		allErrs = append(allErrs, ValidateTokenDiscovery(c.Token, fldPath)...)
		count++
	}
	if c.File != nil {
		allErrs = append(allErrs, ValidateFileDiscovery(c.File, fldPath)...)
		count++
	}
	if c.HTTPS != nil {
		allErrs = append(allErrs, ValidateHTTPSDiscovery(c.HTTPS, fldPath)...)
		count++
	}
	if count != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "exactly one discovery strategy can be provided"))
	}
	return allErrs
}

func ValidateFileDiscovery(c *kubeadm.FileDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func ValidateHTTPSDiscovery(c *kubeadm.HTTPSDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func ValidateTokenDiscovery(c *kubeadm.TokenDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

func ValidateCertificatesPhase(c *kubeadm.CertificatesPhase, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	// If SelfSign specified, validate its properties
	if c.SelfSign != nil {
		for _, altname := range c.SelfSign.AltNames {
			ip := net.ParseIP(altname)
			if len(validation.IsDNS1123Subdomain(altname)) != 0 && ip == nil {
				allErrs = append(allErrs, field.Invalid(fldPath, nil, fmt.Sprintf("altname %s is not a valid dns label or ip address", altname)))
			}
		}
	}
	return allErrs
}

func ValidateIPFromString(ipaddr string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if net.ParseIP(ipaddr) == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "ip address is not valid"))
	}
	return allErrs
}

func ValidateIPNetFromString(ipaddr string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, _, err := net.ParseCIDR(ipaddr); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "cidr is not valid"))
	}
	return allErrs
}

func ValidateNetworking(c *kubeadm.Networking) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateDNS1123Subdomain(c.DNSDomain, field.NewPath("dns-domain"))...)
	allErrs = append(allErrs, ValidateIPNetFromString(c.ServiceSubnet, field.NewPath("service-subnet"))...)
	allErrs = append(allErrs, ValidateIPNetFromString(c.PodSubnet, field.NewPath("pod-subnet"))...)
	return allErrs
}
