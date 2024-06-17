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
	baseConfig := &kubeproxyconfig.KubeProxyConfiguration{
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
	newPath := field.NewPath("KubeProxyConfiguration")

	for name, testCase := range map[string]struct {
		mutateConfigFunc func(*kubeproxyconfig.KubeProxyConfiguration)
		expectedErrs     field.ErrorList
	}{
		"basic config, unspecified Mode": {
			mutateConfigFunc: func(_ *kubeproxyconfig.KubeProxyConfiguration) {},
		},
		"Mode specified, extra mode-specific configs": {
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
		"empty HealthzBindAddress": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = ""
			},
		},
		"IPv6": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.BindAddress = "fd00:192:168:59::103"
				config.HealthzBindAddress = ""
				config.MetricsBindAddress = "[::1]:10249"
				config.ClusterCIDR = "fd00:192:168:59::/64"
			},
		},
		"alternate healthz port": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = "0.0.0.0:12345"
			},
		},
		"ClusterCIDR is wrong IP family": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "fd00:192:168::/64"
			},
		},
		"ClusterCIDR is dual-stack": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0/24,fd00:192:168::/64"
			},
		},
		"LocalModeInterfaceNamePrefix": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeInterfaceNamePrefix
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "vethabcde",
				}
			},
		},
		"LocalModeBridgeInterface": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeBridgeInterface
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					BridgeInterface: "avz",
				}
			},
		},
		"invalid BindAddress": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.BindAddress = "10.10.12.11:2000"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("BindAddress"), "10.10.12.11:2000", "not a valid textual representation of an IP address")},
		},
		"invalid HealthzBindAddress": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddress = "0.0.0.0"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "0.0.0.0", "must be IP:port")},
		},
		"invalid MetricsBindAddress": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MetricsBindAddress = "127.0.0.1"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MetricsBindAddress"), "127.0.0.1", "must be IP:port")},
		},
		"ClusterCIDR missing subset range": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		"Invalid number of ClusterCIDRs": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ClusterCIDR = "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16"
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16", "only one CIDR allowed or a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)")},
		},
		"ConfigSyncPeriod must be > 0": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ConfigSyncPeriod = metav1.Duration{Duration: -1 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ConfigSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than 0")},
		},
		"IPVS mode selected without providing required SyncPeriod": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.Mode = kubeproxyconfig.ProxyModeIPVS
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 0}, "must be greater than 0")},
		},
		"interfacePrefix is empty": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeInterfaceNamePrefix
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "",
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfacePrefix"), "", "must not be empty")},
		},
		"bridgeInterfaceName is empty": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = kubeproxyconfig.LocalModeBridgeInterface
				config.DetectLocal = kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "eth0", // we won't care about prefix since mode is not prefix
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfaceName"), "", "must not be empty")},
		},
		"invalid DetectLocalMode": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.DetectLocalMode = "Guess"
			},
			expectedErrs: field.ErrorList{field.NotSupported(newPath.Child("DetectLocalMode"), "Guess", []string{"ClusterCIDR", "NodeCIDR", "BridgeInterface", "InterfaceNamePrefix", ""})},
		},
		"invalid logging format": {
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.Logging = logsapi.LoggingConfiguration{
					Format: "unsupported format",
				}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("logging.format"), "unsupported format", "Unsupported log format")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			config := baseConfig.DeepCopy()
			testCase.mutateConfigFunc(config)
			errs := Validate(config)
			if len(testCase.expectedErrs) == 0 {
				assert.Equal(t, field.ErrorList{}, errs, "expected no validation errors")
			} else {
				assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
			}
		})
	}
}

func TestValidateKubeProxyIPTablesConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	for name, testCase := range map[string]struct {
		config       kubeproxyconfig.KubeProxyIPTablesConfiguration
		expectedErrs field.ErrorList
	}{
		"valid iptables config": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"valid custom MasqueradeBit": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"SyncPeriod must be > 0": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: 2 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPTablesConfiguration.MinSyncPeriod")},
		},
		"MinSyncPeriod must be > 0": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.MinSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
		"MasqueradeBit cannot be < 0": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](-10),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.MasqueradeBit"), ptr.To[int32](-10), "must be within the range [0, 31]")},
		},
		"SyncPeriod must be >= MinSyncPeriod": {
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: ptr.To[int32](5),
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.SyncPeriod"), metav1.Duration{Duration: 5 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPTablesConfiguration.MinSyncPeriod")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateKubeProxyIPTablesConfiguration(testCase.config, newPath.Child("KubeIPTablesConfiguration"))
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func TestValidateKubeProxyIPVSConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		config       kubeproxyconfig.KubeProxyIPVSConfiguration
		expectedErrs field.ErrorList
	}{
		"SyncPeriod is not greater than 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 2 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		"SyncPeriod cannot be 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 0 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 0}, "must be greater than 0"),
				field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 10 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		"MinSyncPeriod cannot be less than 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.MinSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
		"SyncPeriod must be greater than MinSyncPeriod": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 5 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.KubeIPVSConfiguration.MinSyncPeriod")},
		},
		"SyncPeriod == MinSyncPeriod": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"SyncPeriod should be > MinSyncPeriod": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"MinSyncPeriod can be 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"IPVS Timeout can be 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 0 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 0 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"IPVS Timeout > 0": {
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 1 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 2 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 3 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		"TCP,TCPFin,UDP Timeouts < 0": {
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
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateKubeProxyIPVSConfiguration(testCase.config, newPath.Child("KubeIPVSConfiguration"))
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func TestValidateKubeProxyConntrackConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		config       kubeproxyconfig.KubeProxyConntrackConfiguration
		expectedErrs field.ErrorList
	}{
		"valid 5 second timeouts": {
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
		"valid duration equal to 0 second timeout": {
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
		"invalid MaxPerCore < 0": {
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](-1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.MaxPerCore"), ptr.To[int32](-1), "must be greater than or equal to 0")},
		},
		"invalid minimum < 0": {
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](-1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.Min"), ptr.To[int32](-1), "must be greater than or equal to 0")},
		},
		"invalid TCPEstablishedTimeout < 0": {
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: -5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPEstablishedTimeout"), &metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		"invalid TCPCloseWaitTimeout < 0": {
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: -5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPCloseWaitTimeout"), &metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		"invalid UDPTimeout < 0": {
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
		"invalid UDPStreamTimeout < 0": {
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
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateKubeProxyConntrackConfiguration(testCase.config, newPath.Child("KubeConntrackConfiguration"))
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func TestValidateProxyMode(t *testing.T) {
	if runtime.GOOS == "windows" {
		testValidateProxyModeWindows(t)
	} else {
		testValidateProxyModeLinux(t)
	}
}

func testValidateProxyModeLinux(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		mode         kubeproxyconfig.ProxyMode
		expectedErrs field.ErrorList
	}{
		"blank mode should default": {
			mode: kubeproxyconfig.ProxyMode(""),
		},
		"iptables is allowed": {
			mode: kubeproxyconfig.ProxyModeIPTables,
		},
		"ipvs is allowed": {
			mode: kubeproxyconfig.ProxyModeIPVS,
		},
		"nftables is allowed": {
			mode: kubeproxyconfig.ProxyModeNFTables,
		},
		"winkernel is not allowed": {
			mode:         kubeproxyconfig.ProxyModeKernelspace,
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "kernelspace", "must be iptables, ipvs, nftables or blank (blank means the best-available proxy [currently iptables])")},
		},
		"invalid mode non-existent": {
			mode:         kubeproxyconfig.ProxyMode("non-existing"),
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "non-existing", "must be iptables, ipvs, nftables or blank (blank means the best-available proxy [currently iptables])")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateProxyMode(testCase.mode, newPath)
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func testValidateProxyModeWindows(t *testing.T) {
	// TODO: remove skip once the test has been fixed.
	if runtime.GOOS == "windows" {
		t.Skip("Skipping failing test on Windows.")
	}
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		mode         kubeproxyconfig.ProxyMode
		expectedErrs field.ErrorList
	}{
		"blank mode should default": {
			mode: kubeproxyconfig.ProxyMode(""),
		},
		"winkernel is allowed": {
			mode: kubeproxyconfig.ProxyModeKernelspace,
		},
		"iptables is not allowed": {
			mode:         kubeproxyconfig.ProxyModeIPTables,
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "iptables", "must be kernelspace or blank (blank means the most-available proxy [currently kernelspace])")},
		},
		"ipvs is not allowed": {
			mode:         kubeproxyconfig.ProxyModeIPVS,
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "ipvs", "must be kernelspace or blank (blank means the most-available proxy [currently kernelspace])")},
		},
		"nftables is not allowed": {
			mode:         kubeproxyconfig.ProxyModeNFTables,
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "nftables", "must be kernelspace or blank (blank means the most-available proxy [currently kernelspace])")},
		},
		"invalid mode non-existent": {
			mode:         kubeproxyconfig.ProxyMode("non-existing"),
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "non-existing", "must be kernelspace or blank (blank means the most-available proxy [currently kernelspace])")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateProxyMode(testCase.mode, newPath)
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func TestValidateClientConnectionConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		ccc          componentbaseconfig.ClientConnectionConfiguration
		expectedErrs field.ErrorList
	}{
		"successful 0 value": {
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: 0},
			expectedErrs: field.ErrorList{},
		},
		"successful 5 value": {
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: 5},
			expectedErrs: field.ErrorList{},
		},
		"burst < 0": {
			ccc:          componentbaseconfig.ClientConnectionConfiguration{Burst: -5},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("Burst"), int64(-5), "must be greater than or equal to 0")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateClientConnectionConfiguration(testCase.ccc, newPath)
			assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
		})
	}
}

func TestValidateHostPort(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		ip           string
		expectedErrs field.ErrorList
	}{
		"all IPs": {
			ip: "0.0.0.0:10256",
		},
		"localhost": {
			ip: "127.0.0.1:10256",
		},
		"specific IP": {
			ip: "10.10.10.10:10256",
		},
		"missing port": {
			ip:           "10.10.10.10",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "10.10.10.10", "must be IP:port")},
		},
		"digits outside of 1-255": {
			ip:           "123.456.789.10:12345",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "123.456.789.10", "must be a valid IP")},
		},
		"invalid named-port": {
			ip:           "10.10.10.10:foo",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "foo", "must be a valid port")},
		},
		"port cannot be 0": {
			ip:           "10.10.10.10:0",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "0", "must be a valid port")},
		},
		"port is greater than allowed range": {
			ip:           "10.10.10.10:65536",
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "65536", "must be a valid port")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateHostPort(testCase.ip, newPath.Child("HealthzBindAddress"))
			if len(testCase.expectedErrs) == 0 {
				assert.Equal(t, field.ErrorList{}, errs, "expected no validation errors")
			} else {
				assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
			}
		})
	}
}

func TestValidateKubeProxyNodePortAddress(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		addresses    []string
		expectedErrs field.ErrorList
	}{
		"no addresses": {
			addresses: []string{},
		},
		"valid 1": {
			addresses: []string{"127.0.0.0/8"},
		},
		"valid 2": {
			addresses: []string{"0.0.0.0/0"},
		},
		"valid 3": {
			addresses: []string{"::/0"},
		},
		"valid 4": {
			addresses: []string{"127.0.0.1/32", "1.2.3.0/24"},
		},
		"valid 5": {
			addresses: []string{"127.0.0.1/32"},
		},
		"valid 6": {
			addresses: []string{"::1/128"},
		},
		"valid 7": {
			addresses: []string{"1.2.3.4/32"},
		},
		"valid 8": {
			addresses: []string{"10.20.30.0/24"},
		},
		"valid 9": {
			addresses: []string{"10.20.0.0/16", "100.200.0.0/16"},
		},
		"valid 10": {
			addresses: []string{"10.0.0.0/8"},
		},
		"valid 11": {
			addresses: []string{"2001:db8::/32"},
		},
		"primary": {
			addresses: []string{kubeproxyconfig.NodePortAddressesPrimary},
		},
		"invalid foo address": {
			addresses:    []string{"foo"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[0]"), "foo", "must be a valid CIDR")},
		},
		"invalid octet address": {
			addresses:    []string{"10.0.0.0/0", "1.2.3"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "1.2.3", "must be a valid CIDR")},
		},
		"address cannot be 0": {
			addresses:    []string{"127.0.0.1/32", "0", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "0", "must be a valid CIDR")},
		},
		"address missing subnet range": {
			addresses:    []string{"127.0.0.1/32", "10.20.30.40", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "10.20.30.40", "must be a valid CIDR")},
		},
		"missing ipv6 subnet ranges": {
			addresses: []string{"::0", "::1", "2001:db8::/32"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[0]"), "::0", "must be a valid CIDR"),
				field.Invalid(newPath.Child("NodePortAddresses[1]"), "::1", "must be a valid CIDR")},
		},
		"invalid ipv6 ip format": {
			addresses:    []string{"::1/128", "2001:db8::/32", "2001:db8:xyz/64"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[2]"), "2001:db8:xyz/64", "must be a valid CIDR")},
		},
		"invalid primary/CIDR mix 1": {
			addresses:    []string{"primary", "127.0.0.1/32"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[0]"), "primary", "can't use both 'primary' and CIDRs")},
		},
		"invalid primary/CIDR mix 2": {
			addresses:    []string{"127.0.0.1/32", "primary"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodePortAddresses[1]"), "primary", "can't use both 'primary' and CIDRs")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateKubeProxyNodePortAddress(testCase.addresses, newPath.Child("NodePortAddresses"))
			if len(testCase.expectedErrs) == 0 {
				assert.Equal(t, field.ErrorList{}, errs, "expected no validation errors")
			} else {
				assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
			}
		})
	}
}

func TestValidateKubeProxyExcludeCIDRs(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	for name, testCase := range map[string]struct {
		addresses    []string
		expectedErrs field.ErrorList
	}{
		"no cidrs": {
			addresses: []string{},
		},
		"valid 1": {
			addresses: []string{"127.0.0.0/8"},
		},
		"valid 2": {
			addresses: []string{"0.0.0.0/0"},
		},
		"valid 3": {
			addresses: []string{"::/0"},
		},
		"valid 4": {
			addresses: []string{"127.0.0.1/32", "1.2.3.0/24"},
		},
		"valid 5": {
			addresses: []string{"127.0.0.1/32"},
		},
		"valid 6": {
			addresses: []string{"::1/128"},
		},
		"valid 7": {
			addresses: []string{"1.2.3.4/32"},
		},
		"valid 8": {
			addresses: []string{"10.20.30.0/24"},
		},
		"valid 9": {
			addresses: []string{"10.20.0.0/16", "100.200.0.0/16"},
		},
		"valid 10": {
			addresses: []string{"10.0.0.0/8"},
		},
		"valid 11": {
			addresses: []string{"2001:db8::/32"},
		},
		"invalid foo address": {
			addresses:    []string{"foo"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[0]"), "foo", "must be a valid CIDR")},
		},
		"invalid octet address": {
			addresses:    []string{"10.0.0.0/0", "1.2.3"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "1.2.3", "must be a valid CIDR")},
		},
		"address cannot be 0": {
			addresses:    []string{"127.0.0.1/32", "0", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "0", "must be a valid CIDR")},
		},
		"address missing subnet range": {
			addresses:    []string{"127.0.0.1/32", "10.20.30.40", "1.2.3.0/24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "10.20.30.40", "must be a valid CIDR")},
		},
		"missing ipv6 subnet ranges": {
			addresses: []string{"::0", "::1", "2001:db8::/32"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[0]"), "::0", "must be a valid CIDR"),
				field.Invalid(newPath.Child("ExcludeCIDRS[1]"), "::1", "must be a valid CIDR")},
		},
		"invalid ipv6 ip format": {
			addresses:    []string{"::1/128", "2001:db8::/32", "2001:db8:xyz/64"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ExcludeCIDRS[2]"), "2001:db8:xyz/64", "must be a valid CIDR")},
		},
	} {
		t.Run(name, func(t *testing.T) {
			errs := validateIPVSExcludeCIDRs(testCase.addresses, newPath.Child("ExcludeCIDRS"))
			if len(testCase.expectedErrs) == 0 {
				assert.Equal(t, field.ErrorList{}, errs, "expected no validation errors")
			} else {
				assert.Equal(t, testCase.expectedErrs, errs, "did not get expected validation errors")
			}
		})
	}
}
