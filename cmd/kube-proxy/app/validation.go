/*
Copyright 2017 The Kubernetes Authors.

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

package app

import (
	"fmt"
	"net"
	"strconv"
	"strings"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	apivalidation "k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

// Validate validates the configuration of kube-proxy
func Validate(config *componentconfig.KubeProxyConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}

	newPath := field.NewPath("KubeProxyConfiguration")

	allErrs = append(allErrs, validateKubeProxyIPTablesConfiguration(config.IPTables, newPath.Child("KubeProxyIPTablesConfiguration"))...)
	allErrs = append(allErrs, validateKubeProxyConntrackConfiguration(config.Conntrack, newPath.Child("KubeProxyConntrackConfiguration"))...)
	allErrs = append(allErrs, validateProxyMode(config.Mode, newPath.Child("Mode"))...)
	allErrs = append(allErrs, validateClientConnectionConfiguration(config.ClientConnection, newPath.Child("ClientConnection"))...)

	if config.OOMScoreAdj != nil && (*config.OOMScoreAdj < -1000 || *config.OOMScoreAdj > 1000) {
		allErrs = append(allErrs, field.Invalid(newPath.Child("OOMScoreAdj"), *config.OOMScoreAdj, "must be within the range [-1000, 1000]"))
	}

	if config.UDPIdleTimeout.Duration <= 0 {
		allErrs = append(allErrs, field.Invalid(newPath.Child("UDPIdleTimeout"), config.UDPIdleTimeout, "must be greater than 0"))
	}

	if config.ConfigSyncPeriod.Duration <= 0 {
		allErrs = append(allErrs, field.Invalid(newPath.Child("ConfigSyncPeriod"), config.ConfigSyncPeriod, "must be greater than 0"))
	}

	if net.ParseIP(config.BindAddress) == nil {
		allErrs = append(allErrs, field.Invalid(newPath.Child("BindAddress"), config.BindAddress, "not a valid textual representation of an IP address"))
	}

	allErrs = append(allErrs, validateHostPort(config.HealthzBindAddress, newPath.Child("HealthzBindAddress"))...)
	allErrs = append(allErrs, validateHostPort(config.MetricsBindAddress, newPath.Child("MetricsBindAddress"))...)

	if config.ClusterCIDR != "" {
		if _, _, err := net.ParseCIDR(config.ClusterCIDR); err != nil {
			allErrs = append(allErrs, field.Invalid(newPath.Child("ClusterCIDR"), config.ClusterCIDR, "must be a valid CIDR block (e.g. 10.100.0.0/16)"))
		}
	}

	if _, err := utilnet.ParsePortRange(config.PortRange); err != nil {
		allErrs = append(allErrs, field.Invalid(newPath.Child("PortRange"), config.PortRange, "must be a valid port range (e.g. 300-2000)"))
	}

	return allErrs
}

func validateKubeProxyIPTablesConfiguration(config componentconfig.KubeProxyIPTablesConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if config.MasqueradeBit != nil && (*config.MasqueradeBit < 0 || *config.MasqueradeBit > 31) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("MasqueradeBit"), config.MasqueradeBit, "must be within the range [0, 31]"))
	}

	if config.SyncPeriod.Duration <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("SyncPeriod"), config.SyncPeriod, "must be greater than 0"))
	}

	if config.MinSyncPeriod.Duration < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("MinSyncPeriod"), config.MinSyncPeriod, "must be greater than or equal to 0"))
	}

	return allErrs
}

func validateKubeProxyConntrackConfiguration(config componentconfig.KubeProxyConntrackConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if config.Max < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("Max"), config.Max, "must be greater than or equal to 0"))
	}

	if config.MaxPerCore < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("MaxPerCore"), config.MaxPerCore, "must be greater than or equal to 0"))
	}

	if config.Min < 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("Min"), config.Min, "must be greater than or equal to 0"))
	}

	if config.TCPEstablishedTimeout.Duration <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("TCPEstablishedTimeout"), config.TCPEstablishedTimeout, "must be greater than 0"))
	}

	if config.TCPCloseWaitTimeout.Duration <= 0 {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("TCPCloseWaitTimeout"), config.TCPCloseWaitTimeout, "must be greater than 0"))
	}

	return allErrs
}

func validateProxyMode(mode componentconfig.ProxyMode, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch mode {
	case componentconfig.ProxyModeUserspace:
	case componentconfig.ProxyModeIPTables:
	case componentconfig.ProxyModeIPVS:
	case "":
	default:
		modes := []string{string(componentconfig.ProxyModeUserspace), string(componentconfig.ProxyModeIPTables), string(componentconfig.ProxyModeIPVS)}
		errMsg := fmt.Sprintf("must be %s or blank (blank means the best-available proxy (currently iptables)", strings.Join(modes, ","))
		allErrs = append(allErrs, field.Invalid(fldPath.Child("ProxyMode"), string(mode), errMsg))
	}
	return allErrs
}

func validateClientConnectionConfiguration(config componentconfig.ClientConnectionConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateNonnegativeField(int64(config.Burst), fldPath.Child("Burst"))...)
	return allErrs
}

func validateHostPort(input string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	hostIP, port, err := net.SplitHostPort(input)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, input, "must be IP:port"))
		return allErrs
	}

	if ip := net.ParseIP(hostIP); ip == nil {
		allErrs = append(allErrs, field.Invalid(fldPath, hostIP, "must be a valid IP"))
	}

	if p, err := strconv.Atoi(port); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, port, "must be a valid port"))
	} else if p < 1 || p > 65535 {
		allErrs = append(allErrs, field.Invalid(fldPath, port, "must be a valid port"))
	}

	return allErrs
}
