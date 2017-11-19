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

package validation

import (
	"fmt"
	"net"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/validate"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/proxy/apis/kubeproxyconfig"
)

// Validate validates the configuration of kube-proxy
func Validate(config *kubeproxyconfig.KubeProxyConfiguration) field.ErrorList {
	allErrs := field.ErrorList{}

	newPath := field.NewPath("KubeProxyConfiguration")

	allErrs = append(allErrs, validateKubeProxyIPTablesConfiguration(config.IPTables, newPath.Child("KubeProxyIPTablesConfiguration"))...)
	allErrs = append(allErrs, validateKubeProxyIPVSConfiguration(config.IPVS, newPath.Child("KubeProxyIPVSConfiguration"))...)
	allErrs = append(allErrs, validateKubeProxyConntrackConfiguration(config.Conntrack, newPath.Child("KubeProxyConntrackConfiguration"))...)
	allErrs = append(allErrs, validateProxyMode(config.Mode, newPath.Child("Mode"))...)
	allErrs = append(allErrs, validateClientConnectionConfiguration(config.ClientConnection, newPath.Child("ClientConnection"))...)

	if config.OOMScoreAdj != nil && (*config.OOMScoreAdj < -1000 || *config.OOMScoreAdj > 1000) {
		allErrs = append(allErrs, field.Invalid(newPath.Child("OOMScoreAdj"), *config.OOMScoreAdj, "must be within the range [-1000, 1000]"))
	}

	allErrs = append(allErrs, validate.Positive(int64(config.UDPIdleTimeout.Duration), newPath.Child("UDPIdleTimeout"))...)
	allErrs = append(allErrs, validate.Positive(int64(config.ConfigSyncPeriod.Duration), newPath.Child("ConfigSyncPeriod"))...)

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

func validateKubeProxyIPTablesConfiguration(config kubeproxyconfig.KubeProxyIPTablesConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if config.MasqueradeBit != nil && (*config.MasqueradeBit < 0 || *config.MasqueradeBit > 31) {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("MasqueradeBit"), config.MasqueradeBit, "must be within the range [0, 31]"))
	}

	allErrs = append(allErrs, validate.Positive(int64(config.SyncPeriod.Duration), fldPath.Child("SyncPeriod"))...)

	allErrs = append(allErrs, validate.NonNegativeDuration(config.MinSyncPeriod.Duration, fldPath.Child("MinSyncPeriod"))...)

	if config.MinSyncPeriod.Duration > config.SyncPeriod.Duration {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("SyncPeriod"), config.MinSyncPeriod, fmt.Sprintf("must be greater than or equal to %s", fldPath.Child("MinSyncPeriod").String())))
	}

	return allErrs
}

func validateKubeProxyIPVSConfiguration(config kubeproxyconfig.KubeProxyIPVSConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	allErrs = append(allErrs, validate.Positive(int64(config.SyncPeriod.Duration), fldPath.Child("SyncPeriod"))...)

	allErrs = append(allErrs, validate.NonNegativeDuration(config.MinSyncPeriod.Duration, fldPath.Child("MinSyncPeriod"))...)

	if config.MinSyncPeriod.Duration > config.SyncPeriod.Duration {
		allErrs = append(allErrs, field.Invalid(fldPath.Child("SyncPeriod"), config.MinSyncPeriod, fmt.Sprintf("must be greater than or equal to %s", fldPath.Child("MinSyncPeriod").String())))
	}

	allErrs = append(allErrs, validateIPVSSchedulerMethod(kubeproxyconfig.IPVSSchedulerMethod(config.Scheduler), fldPath.Child("Scheduler"))...)

	return allErrs
}

func validateKubeProxyConntrackConfiguration(config kubeproxyconfig.KubeProxyConntrackConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if config.Max != nil {
		allErrs = append(allErrs, validate.NonNegative(int64(*config.Max), fldPath.Child("Max"))...)
	}

	if config.MaxPerCore != nil {
		allErrs = append(allErrs, validate.NonNegative(int64(*config.MaxPerCore), fldPath.Child("MaxPerCore"))...)
	}

	if config.Min != nil {
		allErrs = append(allErrs, validate.NonNegative(int64(*config.Min), fldPath.Child("Min"))...)
	}

	allErrs = append(allErrs, validate.NonNegativeDuration(config.TCPEstablishedTimeout.Duration, fldPath.Child("TCPEstablishedTimeout"))...)
	allErrs = append(allErrs, validate.NonNegativeDuration(config.TCPCloseWaitTimeout.Duration, fldPath.Child("TCPCloseWaitTimeout"))...)

	return allErrs
}

func validateProxyMode(mode kubeproxyconfig.ProxyMode, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	switch mode {
	case kubeproxyconfig.ProxyModeUserspace:
	case kubeproxyconfig.ProxyModeIPTables:
	case kubeproxyconfig.ProxyModeIPVS:
	case "":
	default:
		modes := []string{string(kubeproxyconfig.ProxyModeUserspace), string(kubeproxyconfig.ProxyModeIPTables), string(kubeproxyconfig.ProxyModeIPVS)}
		errMsg := fmt.Sprintf("must be %s or blank (blank means the best-available proxy (currently iptables)", strings.Join(modes, ","))
		allErrs = append(allErrs, field.Invalid(fldPath.Child("ProxyMode"), string(mode), errMsg))
	}
	return allErrs
}

func validateClientConnectionConfiguration(config kubeproxyconfig.ClientConnectionConfiguration, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, validate.NonNegative(int64(config.Burst), fldPath.Child("Burst"))...)
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

func validateIPVSSchedulerMethod(scheduler kubeproxyconfig.IPVSSchedulerMethod, fldPath *field.Path) field.ErrorList {
	supportedMethod := []kubeproxyconfig.IPVSSchedulerMethod{
		kubeproxyconfig.RoundRobin,
		kubeproxyconfig.WeightedRoundRobin,
		kubeproxyconfig.LeastConnection,
		kubeproxyconfig.WeightedLeastConnection,
		kubeproxyconfig.LocalityBasedLeastConnection,
		kubeproxyconfig.LocalityBasedLeastConnectionWithReplication,
		kubeproxyconfig.SourceHashing,
		kubeproxyconfig.DestinationHashing,
		kubeproxyconfig.ShortestExpectedDelay,
		kubeproxyconfig.NeverQueue,
		"",
	}
	allErrs := field.ErrorList{}
	var found bool
	for i := range supportedMethod {
		if scheduler == supportedMethod[i] {
			found = true
			break
		}
	}
	// Not found
	if !found {
		errMsg := fmt.Sprintf("must be in %v, blank means the default algorithm method (currently rr)", supportedMethod)
		allErrs = append(allErrs, field.Invalid(fldPath.Child("Scheduler"), string(scheduler), errMsg))
	}
	return allErrs
}
