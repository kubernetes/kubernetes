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
	"runtime"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbaseconfig "k8s.io/component-base/config"
	logsapi "k8s.io/component-base/logs/api/v1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/ptr"
)

func TestValidateKubeProxyConfiguration(t *testing.T) {
	getBaseConfig := func() *kubeproxyconfig.KubeProxyConfiguration {
		return &kubeproxyconfig.KubeProxyConfiguration{
			BindAddress:        "192.168.59.103",
			HealthzBindAddress: "0.0.0.0:10256",
			MetricsBindAddress: "127.0.0.1:10249",
			ClusterCIDR:        "192.168.59.0/24",
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
			Logging: logsapi.LoggingConfiguration{
				Format: "text",
			},
		}
	}

	successCases := []struct {
		name             string
		mutateConfigFunc func(*kubeproxyconfig.KubeProxyConfiguration)
	}{
		{
			name:             "base case",
			mutateConfigFunc: func(_ *kubeproxyconfig.KubeProxyConfiguration) {},
		},
		{
			name: "different proxy mode",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				if runtime.GOOS == "windows" {
					config.Mode = kubeproxyconfig.ProxyModeKernelspace
				} else {
					config.Mode = kubeproxyconfig.ProxyModeIPVS
					config.IPVS = kubeproxyconfig.KubeProxyIPVSConfiguration{
						SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
						MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
					}
				}
			},
		},
		{
			name: "empty healthz bind address",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = ""
			},
		},
		{
			name: "ipv6 only cluster",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.BindAddress = "fd00:192:168:59::103"
				config.HealthzBindAddress = ""
				config.MetricsBindAddress = "[::1]:10249"
				config.ClusterCIDR = "fd00:192:168:59::/64"
			},
		},
		{
			name: "custom port for healthz bind address",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = "0.0.0.0:12345"
			},
		},
		{
			name: "ipv6 cluster cidr",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "fd00:192:168::/64"
			},
		},
		{
			name: "dual stack cluster cidr",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0/24,fd00:192:168::/64"
			},
		},
		{
			name: "detect local mode interface prefix",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeInterfaceNamePrefix
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "vethabcde",
				}
			},
		},
		{
			name: "detect local mode bridge prefix",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeBridgeInterface
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					BridgeInterface: "avz",
				}
			},
		},
	}
	for _, tc := range successCases {
		t.Run(tc.name, func(t *testing.T) {
			config := getBaseConfig()
			tc.mutateConfigFunc(config)

			errs := Validate(config)
			assert.Equalf(t, 0, len(errs), "expected 0 errors, got %v", errs)
		})
	}

	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := []struct {
		name             string
		mutateConfigFunc func(*kubeproxyconfig.KubeProxyConfiguration)
		expectedErrs     field.ErrorList
	}{
		{
			name: "invalid BindAddress",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.BindAddress = "10.10.12.11:2000"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("BindAddress"), "10.10.12.11:2000", "not a valid textual representation of an IP address")},
		},
		{
			name: "invalid HealthzBindAddress",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = "0.0.0.0"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "0.0.0.0", "must be IP:port")},
		},
		{
			name: "invalid MetricsBindAddress",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MetricsBindAddress = "127.0.0.1"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MetricsBindAddress"), "127.0.0.1", "must be IP:port")},
		},
		{
			name: "clusterCIDR missing subset range",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		{
			name: "invalid number of ClusterCIDRs",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16", "only one CIDR allowed or a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)")},
		},
		{
			name: "configSyncPeriod must be > 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ConfigSyncPeriod = metav1.Duration{Duration: -1 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ConfigSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than 0")},
		},
		{
			name: "ipvs mode selected without providing required SyncPeriod",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.Mode = kubeproxyconfig.ProxyModeIPVS
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 0}, "must be greater than 0")},
		},
		{
			name: "interfacePrefix is empty",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeInterfaceNamePrefix
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "",
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfacePrefix"), "", "must not be empty")},
		},
		{
			name: "bridgeInterfaceName is empty",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeBridgeInterface
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "eth0", // we won't care about prefix since mode is not prefix
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfaceName"), "", "must not be empty")},
		},
		{
			name: "invalid DetectLocalMode",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = "Guess"
			},
			expectedErrs: field.ErrorList{field.NotSupported(newPath.Child("DetectLocalMode"), "Guess", []string{"ClusterCIDR", "NodeCIDR", "BridgeInterface", "InterfaceNamePrefix", ""})},
		},
		{
			name: "invalid logging format",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.Logging = logsapi.LoggingConfiguration{
					Format: "unsupported format",
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("logging.format"), "unsupported format", "Unsupported log format")},
		},
	}

	for _, tc := range testCases {
		config := getBaseConfig()
		tc.mutateConfigFunc(config)
		if runtime.GOOS == "windows" && config.Mode == kubeproxyconfig.ProxyModeIPVS {
			// IPVS is not supported on Windows.
			t.Log("Skipping test on Windows: ", tc.name)
			continue
		}
		t.Run(tc.name, func(t *testing.T) {
			errs := Validate(config)
			assert.Equal(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateKubeProxyIPTablesConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name         string
		config       kubeproxyconfig.KubeProxyIPTablesConfiguration
		expectedErrs field.ErrorList
	}{
		{
			name: "valid iptables config",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "valid custom MasqueradeBit",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "SyncPeriod must be > 0",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: 2 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPTablesConfiguration.MinSyncPeriod")},
		},
		{
			name: "MinSyncPeriod must be > 0",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.MinSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "MasqueradeBit cannot be < 0",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](-10),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.MasqueradeBit"), -10, "must be within the range [0, 31]")},
		},
		{
			name: "SyncPeriod must be >= MinSyncPeriod",
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: 5 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPTablesConfiguration.MinSyncPeriod")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateKubeProxyIPTablesConfiguration(tc.config, newPath.Child("KubeIPTablesConfiguration"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateKubeProxyIPVSConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := []struct {
		name         string
		config       kubeproxyconfig.KubeProxyIPVSConfiguration
		expectedErrs field.ErrorList
	}{
		{
			name: "SyncPeriod is not greater than 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 2 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		{
			name: "SyncPeriod cannot be 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 0 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 0}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 10 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		{
			name: "MinSyncPeriod cannot be less than 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.MinSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "SyncPeriod must be greater than MinSyncPeriod",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 5 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		{
			name: "SyncPeriod == MinSyncPeriod",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "SyncPeriod should be > MinSyncPeriod",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "MinSyncPeriod can be 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "IPVS Timeout can be 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 0 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 0 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "IPVS Timeout > 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 1 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 2 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 3 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "TCP,TCPFin,UDP Timeouts < 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: -1 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: -1 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.TCPTimeout"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.TCPFinTimeout"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.UDPTimeout"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateKubeProxyIPVSConfiguration(tc.config, newPath.Child("KubeIPVSConfiguration"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateKubeProxyConntrackConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := []struct {
		name         string
		config       kubeproxyconfig.KubeProxyConntrackConfiguration
		expectedErrs field.ErrorList
	}{
		{
			name: "valid 5 second timeouts",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "valid duration equal to 0 second timeout",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 0 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 0 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 0 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "invalid MaxPerCore < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](-1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.MaxPerCore"), -1, "must be greater than or equal to 0")},
		},
		{
			name: "invalid minimum < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](-1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.Min"), -1, "must be greater than or equal to 0")},
		},
		{
			name: "invalid TCPEstablishedTimeout < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: -5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPEstablishedTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid TCPCloseWaitTimeout < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: -5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPCloseWaitTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid UDPTimeout < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: -5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.UDPTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid UDPStreamTimeout < 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: -5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.UDPStreamTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateKubeProxyConntrackConfiguration(tc.config, newPath.Child("KubeConntrackConfiguration"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateProxyMode(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	successCases := []kubeproxyconfig.ProxyMode{""}
	expectedNonExistentErrorMsg := "must be iptables, ipvs or blank (blank means the best-available proxy [currently iptables])"

	if runtime.GOOS == "windows" {
		successCases = append(successCases, kubeproxyconfig.ProxyModeKernelspace)
		expectedNonExistentErrorMsg = "must be kernelspace or blank (blank means the most-available proxy [currently kernelspace])"
	} else {
		successCases = append(successCases, kubeproxyconfig.ProxyModeIPTables, kubeproxyconfig.ProxyModeIPVS)
	}

	for _, successCase := range successCases {
		if errs := validateProxyMode(successCase, newPath.Child("ProxyMode")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	testCases := []struct {
		name         string
		mode         kubeproxyconfig.ProxyMode
		expectedErrs field.ErrorList
	}{
		{
			name:         "blank mode should default",
			mode:         kubeproxyconfig.ProxyMode(""),
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "invalid mode non-existent",
			mode:         kubeproxyconfig.ProxyMode("non-existing"),
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "non-existing", expectedNonExistentErrorMsg)},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateProxyMode(tc.mode, newPath)
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateClientConnectionConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name         string
		ccc          componentbaseconfig.ClientConnectionConfiguration
		expectedErrs field.ErrorList
	}{
		{
			name:         "successful 0 value",
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: 0},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "successful 5 value",
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: 5},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "burst < 0",
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: -5},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("Burst"), -5, "must be greater than or equal to 0")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateClientConnectionConfiguration(tc.ccc, newPath)
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateHostPort(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []string{
		"0.0.0.0:10256",
		"127.0.0.1:10256",
		"10.10.10.10:10256",
	}

	for _, successCase := range successCases {
		if errs := validateHostPort(successCase, newPath.Child("HealthzBindAddress")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		name         string
		ip           string
		expectedErrs field.ErrorList
	}{
		{
			name:         "missing port",
			ip:           "10.10.10.10",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "10.10.10.10", "must be IP:port")},
		},
		{
			name:         "digits outside of 1-255",
			ip:           "123.456.789.10:12345",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "123.456.789.10", "must be a valid IP")},
		},
		{
			name:         "invalid named-port",
			ip:           "10.10.10.10:foo",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "foo", "must be a valid port")},
		},
		{
			name:         "port cannot be 0",
			ip:           "10.10.10.10:0",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "0", "must be a valid port")},
		},
		{
			name:         "port is greater than allowed range",
			ip:           "10.10.10.10:65536",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "65536", "must be a valid port")},
		},
	}

	for _, tc := range errorCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateHostPort(tc.ip, newPath.Child("HealthzBindAddress"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateKubeProxyNodePortAddress(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []struct {
		addresses []string
	}{
		{[]string{}},
		{[]string{"127.0.0.0/8"}},
		{[]string{"0.0.0.0/0"}},
		{[]string{"::/0"}},
		{[]string{"127.0.0.1/32", "1.2.3.0/24"}},
		{[]string{"127.0.0.0/8"}},
		{[]string{"127.0.0.1/32"}},
		{[]string{"::1/128"}},
		{[]string{"1.2.3.4/32"}},
		{[]string{"10.20.30.0/24"}},
		{[]string{"10.20.0.0/16", "100.200.0.0/16"}},
		{[]string{"10.0.0.0/8"}},
		{[]string{"2001:db8::/32"}},
	}

	for _, successCase := range successCases {
		if errs := validateKubeProxyNodePortAddress(successCase.addresses, newPath.Child("NodePortAddresses")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	testCases := []struct {
		name         string
		addresses    []string
		expectedErrs field.ErrorList
	}{
		{
			name:         "invalid foo address",
			addresses:    []string{"foo"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[0]"), "foo", "must be a valid CIDR")},
		},
		{
			name:         "invalid octet address",
			addresses:    []string{"10.0.0.0/0", "1.2.3"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "1.2.3", "must be a valid CIDR")},
		},
		{
			name:         "address cannot be 0",
			addresses:    []string{"127.0.0.1/32", "0", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "0", "must be a valid CIDR")},
		},
		{
			name:         "address missing subnet range",
			addresses:    []string{"127.0.0.1/32", "10.20.30.40", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "10.20.30.40", "must be a valid CIDR")},
		},
		{
			name:      "missing ipv6 subnet ranges",
			addresses: []string{"::0", "::1", "2001:db8::/32"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[0]"), "::0", "must be a valid CIDR"),
				field.Invalid(newPath.Child("NodePortAddresses[1]"), "::1", "must be a valid CIDR")},
		},
		{
			name:         "invalid ipv6 ip format",
			addresses:    []string{"::1/128", "2001:db8::/32", "2001:db8:xyz/64"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[2]"), "2001:db8:xyz/64", "must be a valid CIDR")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateKubeProxyNodePortAddress(tc.addresses, newPath.Child("NodePortAddresses"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateKubeProxyExcludeCIDRs(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []struct {
		addresses []string
	}{
		{[]string{}},
		{[]string{"127.0.0.0/8"}},
		{[]string{"0.0.0.0/0"}},
		{[]string{"::/0"}},
		{[]string{"127.0.0.1/32", "1.2.3.0/24"}},
		{[]string{"127.0.0.0/8"}},
		{[]string{"127.0.0.1/32"}},
		{[]string{"::1/128"}},
		{[]string{"1.2.3.4/32"}},
		{[]string{"10.20.30.0/24"}},
		{[]string{"10.20.0.0/16", "100.200.0.0/16"}},
		{[]string{"10.0.0.0/8"}},
		{[]string{"2001:db8::/32"}},
	}

	for _, successCase := range successCases {
		if errs := validateIPVSExcludeCIDRs(successCase.addresses, newPath.Child("ExcludeCIDRs")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	testCases := []struct {
		name         string
		addresses    []string
		expectedErrs field.ErrorList
	}{
		{
			name:         "invalid foo address",
			addresses:    []string{"foo"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[0]"), "foo", "must be a valid CIDR")},
		},
		{
			name:         "invalid octet address",
			addresses:    []string{"10.0.0.0/0", "1.2.3"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "1.2.3", "must be a valid CIDR")},
		},
		{
			name:         "address cannot be 0",
			addresses:    []string{"127.0.0.1/32", "0", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "0", "must be a valid CIDR")},
		},
		{
			name:         "address missing subnet range",
			addresses:    []string{"127.0.0.1/32", "10.20.30.40", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "10.20.30.40", "must be a valid CIDR")},
		},
		{
			name:      "missing ipv6 subnet ranges",
			addresses: []string{"::0", "::1", "2001:db8::/32"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[0]"), "::0", "must be a valid CIDR"),
				field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "::1", "must be a valid CIDR")},
		},
		{
			name:         "invalid ipv6 ip format",
			addresses:    []string{"::1/128", "2001:db8::/32", "2001:db8:xyz/64"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[2]"), "2001:db8:xyz/64", "must be a valid CIDR")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateIPVSExcludeCIDRs(tc.addresses, newPath.Child("ExcludeCIDRS"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}
