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
		allErrs = append(allErrs, field.Invalid(field.NewPath("ca-cert-path"), nil, "the ca certificate path must be an absolute path"))
	}
	return allErrs
}

func ValidateAuthorizationMode(authzMode string, fldPath *field.Path) field.ErrorList {
	if !authzmodes.IsValidAuthorizationMode(authzMode) {
		return field.ErrorList{field.Invalid(fldPath, nil, "invalid authorization mode")}
	}
	return field.ErrorList{}
}

func ValidateDiscovery(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateAndSetArgSelection(c, fldPath)...)
	allErrs = append(allErrs, ValidateToken(c.Token, fldPath)...)
	allErrs = append(allErrs, ValidateDiscoveryFile(c, fldPath)...)
	allErrs = append(allErrs, ValidateDiscoveryURL(c, fldPath)...)
	allErrs = append(allErrs, ValidateToken(c.TLSBootstrapToken, fldPath)...)
	allErrs = append(allErrs, ValidateJoinMastersArgs(c, fldPath)...)
	return allErrs
}

func ValidateAndSetArgSelection(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateTokenArg(cfg, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateDiscoveryTokenArg(cfg, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateTLSBootstrapTokenArg(cfg, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateDiscoveryFileArg(cfg, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateDiscoveryURLArg(cfg, field.NewPath("discovery"))...)
	if len(cfg.Token) != 0 {
		cfg.TLSBootstrapToken = cfg.Token
		cfg.DiscoveryToken = cfg.Token
	}
	return allErrs
}

func ValidateTokenArg(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.Token) != 0 {
		if len(cfg.TLSBootstrapToken) != 0 || len(cfg.DiscoveryToken) != 0 || len(cfg.DiscoveryURL) != 0 || len(cfg.DiscoveryFile) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--token is mutually exclusive with TLSBootstrapToken, --discovery-token, --discovery-file, and --discovery-url",
				),
			)
		}
		if len(cfg.Masters) < 1 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--token must also specificy --masters",
				),
			)
		}
	}
	return allErrs
}

func ValidateDiscoveryTokenArg(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.DiscoveryToken) != 0 {
		if len(cfg.Token) != 0 || len(cfg.DiscoveryURL) != 0 || len(cfg.DiscoveryFile) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-token is mutually exclusive with --token, --discovery-file, and --discovery-url",
				),
			)
		}
		if len(cfg.Masters) < 1 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-token must also specificy --masters",
				),
			)
		}
		if len(cfg.TLSBootstrapToken) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-token must also specificy TLSBootstrapToken",
				),
			)
		}
	}
	return allErrs
}

func ValidateTLSBootstrapTokenArg(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.TLSBootstrapToken) != 0 {
		if len(cfg.Token) != 0 || len(cfg.DiscoveryURL) != 0 || len(cfg.DiscoveryFile) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"TLSBootstrapToken is mutually exclusive with --token, --discovery-file, and --discovery-url",
				),
			)
		}
		if len(cfg.Masters) < 1 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"TLSBootstrapToken must also specificy --masters",
				),
			)
		}
		if len(cfg.DiscoveryToken) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"TLSBootstrapToken must also specificy --discovery-token",
				),
			)
		}
	}
	return allErrs
}

func ValidateDiscoveryFileArg(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.DiscoveryFile) != 0 {
		if len(cfg.Token) != 0 || len(cfg.DiscoveryToken) != 0 || len(cfg.DiscoveryURL) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-file is mutually exclusive with --token, --discovery-token, and --discovery-url",
				),
			)
		}
		if len(cfg.Masters) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-file must not specificy --masters",
				),
			)
		}
		if len(cfg.TLSBootstrapToken) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-file must also specificy TLSBootstrapToken",
				),
			)
		}
	}
	return allErrs
}

func ValidateDiscoveryURLArg(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.DiscoveryURL) != 0 {
		if len(cfg.Token) != 0 || len(cfg.DiscoveryToken) != 0 || len(cfg.DiscoveryFile) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-url is mutually exclusive with --token, --discovery-token, and --discovery-file",
				),
			)
		}
		if len(cfg.Masters) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-url must not specificy --masters",
				),
			)
		}
		if len(cfg.TLSBootstrapToken) != 0 {
			allErrs = append(
				allErrs,
				field.Invalid(
					fldPath,
					nil,
					"--discovery-url must also specificy TLSBootstrapToken",
				),
			)
		}
	}
	return allErrs
}

func ValidateJoinMastersArgs(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	for _, m := range c.Masters {
		_, _, err := net.SplitHostPort(m)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
		}
	}
	return allErrs
}

func ValidateDiscoveryFile(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if _, err := os.Stat(c.DiscoveryFile); os.IsNotExist(err) {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "file does not exist"))
	}

	return allErrs
}

func ValidateDiscoveryURL(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	u, err := url.Parse(c.DiscoveryURL)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "invalide URL"))
	}
	if u.Scheme != "https" {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "must be https"))
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
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "token must be specific as <ID>:<Secret>"))
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
