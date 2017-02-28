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
	"bytes"
	"fmt"
	"net"
	"net/url"
	"os"
	"path"
	"strings"

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

type Error struct {
	Msg string
}

func (e *Error) Error() string {
	return fmt.Sprintf("[validation] Some errors occurred:\n%s", e.Msg)
}

func ValidateMasterConfiguration(c *kubeadm.MasterConfiguration) error {
	allErrs := []error{}
	allErrs = append(allErrs, ValidateServiceSubnet(c.Networking.ServiceSubnet)...)
	allErrs = append(allErrs, ValidateCloudProvider(c.CloudProvider)...)
	allErrs = append(allErrs, ValidateAuthorizationMode(c.AuthorizationMode)...)
	if len(allErrs) > 0 {
		var errs bytes.Buffer
		for _, i := range allErrs {
			errs.WriteString("\t" + i.Error() + "\n")
		}
		return &Error{Msg: errs.String()}
	}
	return nil
}

func ValidateNodeConfiguration(c *kubeadm.NodeConfiguration) error {
	allErrs := []error{}
	allErrs = append(allErrs, ValidateDiscovery(c)...)

	if !path.IsAbs(c.CACertPath) || !strings.HasSuffix(c.CACertPath, ".crt") {
		allErrs = append(allErrs, fmt.Errorf("the ca certificate path must be an absolute path"))
	}
	if len(allErrs) > 0 {
		var errs bytes.Buffer
		for _, i := range allErrs {
			errs.WriteString("\t" + i.Error() + "\n")
		}
		return &Error{Msg: errs.String()}
	}
	return nil
}

func ValidateAuthorizationMode(authzMode string) []error {
	if !authzmodes.IsValidAuthorizationMode(authzMode) {
		return []error{fmt.Errorf("invalid authorization mode")}
	}
	return []error{}
}

func ValidateDiscovery(c *kubeadm.NodeConfiguration) []error {
	allErrs := []error{}
	allErrs = append(allErrs, ValidateArgSelection(c)...)
	allErrs = append(allErrs, ValidateTLSBootstrapTokenArg(c)...)
	allErrs = append(allErrs, ValidateToken(c.DiscoveryToken)...)
	allErrs = append(allErrs, ValidateDiscoveryFile(c)...)
	allErrs = append(allErrs, ValidateToken(c.TLSBootstrapToken)...)
	allErrs = append(allErrs, ValidateJoinDiscoveryTokenAPIServer(c)...)
	return allErrs
}

func ValidateArgSelection(cfg *kubeadm.NodeConfiguration) []error {
	allErrs := []error{}
	if len(cfg.DiscoveryToken) != 0 && len(cfg.DiscoveryFile) != 0 {
		allErrs = append(allErrs, fmt.Errorf("DiscoveryToken and DiscoveryFile cannot both be set"))
	}
	if len(cfg.DiscoveryToken) == 0 {
		allErrs = append(allErrs, fmt.Errorf("Token value not found in DiscoveryToken or Token"))
	}
	if len(cfg.DiscoveryTokenAPIServers) < 1 && len(cfg.DiscoveryToken) != 0 {
		allErrs = append(allErrs, fmt.Errorf("DiscoveryTokenAPIServers not set"))
	}
	// TODO remove once we support multiple api servers
	if len(cfg.DiscoveryTokenAPIServers) > 1 {
		fmt.Println("[validation] WARNING: kubeadm doesn't fully support multiple API Servers yet")
	}
	return allErrs
}

func ValidateTLSBootstrapTokenArg(cfg *kubeadm.NodeConfiguration) []error {
	allErrs := []error{}
	if len(cfg.TLSBootstrapToken) != 0 {
		allErrs = append(allErrs, fmt.Errorf("TLSBootstrapToken or Token must be set"))
	}
	return allErrs
}

func ValidateJoinDiscoveryTokenAPIServer(c *kubeadm.NodeConfiguration) []error {
	allErrs := []error{}
	for _, m := range c.DiscoveryTokenAPIServers {
		_, _, err := net.SplitHostPort(m)
		if err != nil {
			allErrs = append(allErrs, err)
		}
	}
	return allErrs
}

func ValidateDiscoveryFile(c *kubeadm.NodeConfiguration) []error {
	allErrs := []error{}
	u, err := url.Parse(c.DiscoveryFile)
	if err != nil {
		// Ok, something without a URI scheme is passed, assume a file
		if _, err := os.Stat(c.DiscoveryFile); os.IsNotExist(err) {
			allErrs = append(allErrs, fmt.Errorf("not a valid HTTPS URL or a file on disk"))
		}
	} else {
		// Parsable URL, but require HTTPS
		if u.Scheme != "https" {
			allErrs = append(allErrs, fmt.Errorf("must be https"))
		}
	}

	return allErrs
}

func ValidateToken(t string) []error {
	allErrs := []error{}

	id, secret, err := tokenutil.ParseToken(t)
	if err != nil {
		allErrs = append(allErrs, err)
	}

	if len(id) == 0 || len(secret) == 0 {
		allErrs = append(allErrs, fmt.Errorf("token must be specific as <ID>.<Secret>"))
	}
	return allErrs
}

func ValidateServiceSubnet(subnet string) []error {
	_, svcSubnet, err := net.ParseCIDR(subnet)
	if err != nil {
		return []error{fmt.Errorf("couldn't parse the service subnet")}
	}
	numAddresses := ipallocator.RangeSize(svcSubnet)
	if numAddresses < constants.MinimumAddressesInServiceSubnet {
		return []error{fmt.Errorf("service subnet is too small")}
	}
	return []error{}
}

func ValidateCloudProvider(provider string) []error {
	if len(provider) == 0 {
		return []error{}
	}
	for _, supported := range cloudproviders {
		if provider == supported {
			return []error{}
		}
	}
	return []error{fmt.Errorf("cloudprovider not supported")}
}
