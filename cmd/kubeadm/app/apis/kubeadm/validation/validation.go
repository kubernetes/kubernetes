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
	"net"
	"net/url"
	"os"
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
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
	allErrs = append(allErrs, ValidateServiceSubnet(c.Networking.ServiceSubnet, field.NewPath("service subnet"))...)
	allErrs = append(allErrs, ValidateCloudProvider(c.CloudProvider, field.NewPath("cloudprovider"))...)
	allErrs = append(allErrs, ValidateAuthorizationMode(c.AuthorizationMode, field.NewPath("authorization-mode"))...)
	return allErrs
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(c, field.NewPath("discovery"))...)

	if !path.IsAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, field.Invalid(field.NewPath("ca-cert-path"), c.CACertPath, "the ca certificate path must be an absolute path"))
	}
	return allErrs
}

func ValidateAuthorizationMode(authzMode string, fldPath *field.Path) field.ErrorList {
	if !authzmodes.IsValidAuthorizationMode(authzMode) {
		return field.ErrorList{field.Invalid(fldPath, authzMode, "invalid authorization mode")}
	}
	return field.ErrorList{}
}

func ValidateDiscovery(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(c.DiscoveryToken) != 0 {
		allErrs = append(allErrs, ValidateToken(c.DiscoveryToken, fldPath)...)
	}
	if len(c.DiscoveryFile) != 0 {
		allErrs = append(allErrs, ValidateDiscoveryFile(c.DiscoveryFile, fldPath)...)
	}
	allErrs = append(allErrs, ValidateArgSelection(c, fldPath)...)
	allErrs = append(allErrs, ValidateToken(c.TLSBootstrapToken, fldPath)...)
	allErrs = append(allErrs, ValidateJoinDiscoveryTokenAPIServer(c, fldPath)...)
	return allErrs
}

func ValidateArgSelection(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.DiscoveryToken) != 0 && len(cfg.DiscoveryFile) != 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "DiscoveryToken and DiscoveryFile cannot both be set"))
	}
	if len(cfg.DiscoveryToken) == 0 && len(cfg.DiscoveryFile) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, "", "DiscoveryToken or DiscoveryFile must be set"))
	}
	if len(cfg.DiscoveryTokenAPIServers) < 1 && len(cfg.DiscoveryToken) != 0 {
		allErrs = append(allErrs, field.Required(fldPath, "DiscoveryTokenAPIServers not set"))
	}
	// TODO remove once we support multiple api servers
	if len(cfg.DiscoveryTokenAPIServers) > 1 {
		fmt.Println("[validation] WARNING: kubeadm doesn't fully support multiple API Servers yet")
	}
	return allErrs
}

func ValidateJoinDiscoveryTokenAPIServer(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, m := range c.DiscoveryTokenAPIServers {
		_, _, err := net.SplitHostPort(m)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, m, err.Error()))
		}
	}
	return allErrs
}

func ValidateDiscoveryFile(discoveryFile string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	u, err := url.Parse(discoveryFile)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "not a valid HTTPS URL or a file on disk"))
		return allErrs
	}

	if u.Scheme == "" {
		// URIs with no scheme should be treated as files
		if _, err := os.Stat(discoveryFile); os.IsNotExist(err) {
			allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "not a valid HTTPS URL or a file on disk"))
		}
		return allErrs
	}

	if u.Scheme != "https" {
		allErrs = append(allErrs, field.Invalid(fldPath, discoveryFile, "if an URL is used, the scheme must be https"))
	}
	return allErrs
}

func ValidateToken(t string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	id, secret, err := tokenutil.ParseToken(t)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
	}

	if len(id) == 0 || len(secret) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "token must be of form '[a-z0-9]{6}.[a-z0-9]{16}'"))
	}
	return allErrs
}

func ValidateServiceSubnet(subnet string, fldPath *field.Path) field.ErrorList {
	_, svcSubnet, err := net.ParseCIDR(subnet)
	if err != nil {
		return field.ErrorList{field.Invalid(fldPath, nil, "couldn't parse the service subnet")}
	}
	numAddresses := ipallocator.RangeSize(svcSubnet)
	if numAddresses < constants.MinimumAddressesInServiceSubnet {
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
