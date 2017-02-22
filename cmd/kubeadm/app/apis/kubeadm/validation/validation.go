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
	"path"
	"strings"

	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/file"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/https"
	"k8s.io/kubernetes/cmd/kubeadm/app/discovery/token"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
)

func ValidateMasterConfiguration(c *kubeadm.MasterConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscoveryStruct(&c.Discovery, field.NewPath("discovery"))...)
	allErrs = append(allErrs, ValidateDiscoveryStruct(&c.Discovery, field.NewPath("service subnet"))...)
	return allErrs
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateDiscovery(c, field.NewPath("discovery"))...)
	return allErrs
}

//This will be removed once Discovery Type gets removed
func ValidateDiscoveryStruct(c *kubeadm.Discovery, fldPath *field.Path) field.ErrorList {
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

//This will be removed once Discovery Type gets removed
func ValidateFileDiscovery(c *kubeadm.FileDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

//This will be removed once Discovery Type gets removed
func ValidateHTTPSDiscovery(c *kubeadm.HTTPSDiscovery, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	return allErrs
}

//This will be removed once Discovery Type gets removed
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

func ValidateDiscovery(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, ValidateTLSBootstrapToken(c, fldPath)...)
	var count int
	if len(c.DiscoveryToken) != 0 {
		allErrs = append(allErrs, ValidateDiscoveryToken(c, fldPath)...)
		count++
	}
	if len(c.DiscoveryFile) != 0 {
		allErrs = append(allErrs, ValidateDiscoveryFile(c, fldPath)...)
		count++
	}
	if len(c.DiscoveryURL) != 0 {
		allErrs = append(allErrs, ValidateDiscoveryURL(c, fldPath)...)
		count++
	}
	if count != 1 {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "exactly one discovery strategy can be provided"))
	}
	return allErrs
}

func ValidateTLSBootstrapToken(cfg *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(cfg.Token) != 0 && (len(cfg.TLSBootstrapToken) != 0 || len(cfg.DiscoveryToken) != 0) {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "--token are mutually exclusive with --bootstrap-token and --discovery-token"))
	}
	if len(cfg.Token) == 0 && (len(cfg.TLSBootstrapToken) == 0 || len(cfg.DiscoveryToken) == 0) {
		allErrs = append(
			allErrs,
			field.Invalid(fldPath, nil, "If --token is not specified, both --bootstrap-token and --discovery-token have to be provided"),
		)
	}
	if len(cfg.Token) != 0 {
		cfg.TLSBootstrapToken = cfg.Token
		cfg.DiscoveryToken = cfg.Token
	}
	return allErrs
}

func ValidateDiscoveryFile(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if !path.IsAbs(c.DiscoveryFile) {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "not an absolute file path"))
	}

	// will remove url parsing of file once Discovery struct is removed
	u, err := url.Parse(c.DiscoveryURL)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
	}

	if len(allErrs) == 0 {
		file.Parse(u, &c.Discovery)
	}
	return allErrs
}

func ValidateDiscoveryURL(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	u, err := url.Parse(c.DiscoveryURL)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
	}
	if u.Scheme != "https" {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, "must be https"))
	}
	if len(allErrs) == 0 {
		https.Parse(u, &c.Discovery)
	}
	return allErrs
}

func ValidateDiscoveryToken(c *kubeadm.NodeConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	u, err := url.Parse(c.DiscoveryToken)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
	}
	if !strings.Contains(c.DiscoveryToken, "@") {
		c.DiscoveryToken = c.DiscoveryToken + "@"
		u, err = url.Parse(c.DiscoveryToken)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, nil, err.Error()))
		}
	}
	if len(allErrs) == 0 {
		token.Parse(u, &c.Discovery)
	}

	id, secret, err := tokenutil.ParseToken(c.DiscoveryToken)
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
