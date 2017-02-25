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
	"net"
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

// TODO: Break out the cloudprovider functionality out of core and only support the new flow
// described in https://github.com/kubernetes/community/pull/128
var cloudproviders = []string{
	"aws",
	"azure",
	"cloudstack",
	"gce",
	"mesos",
	"openstack",
	"ovirt",
	"photon",
	"rackspace",
	"vsphere",
}

func ValidateMasterConfiguration(c *kubeadm.MasterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateServiceSubnet(c.Networking.ServiceSubnet, field.NewPath("service subnet"))...)
	allErrs = append(allErrs, ValidateCloudProvider(c.CloudProvider, field.NewPath("cloudprovider"))...)
	allErrs = append(allErrs, ValidateAuthorizationMode(c.AuthorizationMode, field.NewPath("authorization-mode"))...)
	return allErrs
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(&c.Discovery, field.NewPath("discovery"))...)

	if !path.IsAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, field.Invalid(field.NewPath("ca-cert-path"), nil, "the ca certificate path must be an absolute path"))
	}
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
	if len(c.ID) == 0 || len(c.Secret) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "token must be specific as <ID>:<Secret>"))
	}
	if len(c.Addresses) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "at least one address is required"))
	}
	return allErrs
}

func ValidateAuthorizationMode(authzMode string, fldPath *field.Path) field.ErrorList {
	if !authzmodes.IsValidAuthorizationMode(authzMode) {
		return field.ErrorList{field.Invalid(fldPath, nil, "invalid authorization mode")}
	}
	return field.ErrorList{}
}

func ValidateServiceSubnet(subnet string, fldPath *field.Path) field.ErrorList {
	_, svcSubnet, err := net.ParseCIDR(subnet)
	if err != nil {
		return field.ErrorList{field.Invalid(fldPath, nil, "couldn't parse the service subnet")}
	}
	numAddresses := ipallocator.RangeSize(svcSubnet)
	if numAddresses < kubeadmconstants.MinimumAddressesInServiceSubnet {
		return field.ErrorList{field.Invalid(fldPath, nil, "service subnet is too small")}
	}
	return field.ErrorList{}
}

func ValidateCloudProvider(provider string, fldPath *field.Path) field.ErrorList {
	if len(provider) == 0 {
		return field.ErrorList{}
	}
	for _, supported := range cloudproviders {
		if provider == supported {
			return field.ErrorList{}
		}
	}
	return field.ErrorList{field.Invalid(fldPath, nil, "cloudprovider not supported")}
}
