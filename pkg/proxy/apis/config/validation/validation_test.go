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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbaseconfig "k8s.io/component-base/config"
	logsapi "k8s.io/component-base/logs/api/v1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"

	"k8s.io/utils/ptr"
)

func TestValidateKubeProxyConfiguration(t *testing.T) {
	var proxyMode kubeproxyconfig.ProxyMode
	if runtime.GOOS == "windows" {
		proxyMode = kubeproxyconfig.ProxyModeKernelspace
	} else {
		proxyMode = kubeproxyconfig.ProxyModeIPVS
	}
	successCases := []kubeproxyconfig.KubeProxyConfiguration{{
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
		Mode: proxyMode,
		IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
			SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
			MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
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
	}, {
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
	}, {
		BindAddress:        "192.168.59.103",
		HealthzBindAddress: "",
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
	}, {
		BindAddress:        "fd00:192:168:59::103",
		HealthzBindAddress: "",
		MetricsBindAddress: "[::1]:10249",
		ClusterCIDR:        "fd00:192:168:59::/64",
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
	}, {
		BindAddress:        "10.10.12.11",
		HealthzBindAddress: "0.0.0.0:12345",
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
	}, {
		BindAddress:        "10.10.12.11",
		HealthzBindAddress: "0.0.0.0:12345",
		MetricsBindAddress: "127.0.0.1:10249",
		ClusterCIDR:        "fd00:192:168::/64",
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
	}, {
		BindAddress:        "10.10.12.11",
		HealthzBindAddress: "0.0.0.0:12345",
		MetricsBindAddress: "127.0.0.1:10249",
		ClusterCIDR:        "192.168.59.0/24,fd00:192:168::/64",
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
	}, {
		BindAddress:        "10.10.12.11",
		HealthzBindAddress: "0.0.0.0:12345",
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
		DetectLocalMode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
		DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
			InterfaceNamePrefix: "vethabcde",
		},
		Logging: logsapi.LoggingConfiguration{
			Format: "text",
		},
	}, {
		BindAddress:        "10.10.12.11",
		HealthzBindAddress: "0.0.0.0:12345",
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
		DetectLocalMode: kubeproxyconfig.LocalModeBridgeInterface,
		DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
			BridgeInterface: "avz",
		},
		Logging: logsapi.LoggingConfiguration{
			Format: "text",
		},
	}}

	for _, successCase := range successCases {
		if errs := Validate(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := map[string]struct {
		config       kubeproxyconfig.KubeProxyConfiguration
		expectedErrs field.ErrorList
	}{
		"invalid BindAddress": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11:2000",
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("BindAddress"), "10.10.12.11:2000", "not a valid textual representation of an IP address")},
		},
		"invalid HealthzBindAddress": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0",
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddress"), "0.0.0.0", "must be IP:port")},
		},
		"invalid MetricsBindAddress": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1",
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MetricsBindAddress"), "127.0.0.1", "must be IP:port")},
		},
		"ClusterCIDR missing subset range": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0",
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		"Invalid number of ClusterCIDRs": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16",
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ClusterCIDR"), "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16", "only one CIDR allowed or a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)")},
		},
		"ConfigSyncPeriod must be > 0": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				ConfigSyncPeriod:   metav1.Duration{Duration: -1 * time.Second},
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
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ConfigSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than 0")},
		},
		"IPVS mode selected without providing required SyncPeriod": {
			config: kubeproxyconfig.KubeProxyConfiguration{
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
				// not specifying valid period in IPVS mode.
				Mode: kubeproxyconfig.ProxyModeIPVS,
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            ptr.To[int32](1),
					Min:                   ptr.To[int32](1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
				Logging: logsapi.LoggingConfiguration{
					Format: "text",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyIPVSConfiguration.SyncPeriod"), metav1.Duration{Duration: 0}, "must be greater than 0")},
		},
		"interfacePrefix is empty": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
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
				DetectLocalMode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "",
				},
				Logging: logsapi.LoggingConfiguration{
					Format: "text",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfacePrefix"), "", "must not be empty")},
		},
		"bridgeInterfaceName is empty": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
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
				DetectLocalMode: kubeproxyconfig.LocalModeBridgeInterface,
				DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
					InterfaceNamePrefix: "eth0", // we won't care about prefix since mode is not prefix
				},
				Logging: logsapi.LoggingConfiguration{
					Format: "text",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("InterfaceName"), "", "must not be empty")},
		},
		"invalid DetectLocalMode": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
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
				DetectLocalMode: "Guess",
				Logging: logsapi.LoggingConfiguration{
					Format: "text",
				},
			},
			expectedErrs: field.ErrorList{field.NotSupported(newPath.Child("DetectLocalMode"), "Guess", []string{"ClusterCIDR", "NodeCIDR", "BridgeInterface", "InterfaceNamePrefix", ""})},
		},
		"invalid logging format": {
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
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
					Format: "unsupported format",
				},
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("logging.format"), "unsupported format", "Unsupported log format")},
		},
	}

	for name, testCase := range testCases {
		if runtime.GOOS == "windows" && testCase.config.Mode == kubeproxyconfig.ProxyModeIPVS {
			// IPVS is not supported on Windows.
			t.Log("Skipping test on Windows: ", name)
			continue
		}
		t.Run(name, func(t *testing.T) {
			errs := Validate(&testCase.config)
			if len(testCase.expectedErrs) != len(errs) {
				t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
			}
			for i, err := range errs {
				if err.Error() != testCase.expectedErrs[i].Error() {
					t.Fatalf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
				}
			}
		})
	}
}

func TestValidateKubeProxyIPTablesConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := map[string]struct {
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeIPTablesConfiguration.MasqueradeBit"), -10, "must be within the range [0, 31]")},
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
	}

	for _, testCase := range testCases {
		errs := validateKubeProxyIPTablesConfiguration(testCase.config, newPath.Child("KubeIPTablesConfiguration"))
		if len(testCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
	}
}

func TestValidateKubeProxyIPVSConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := map[string]struct {
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
	}
	for _, testCase := range testCases {
		errs := validateKubeProxyIPVSConfiguration(testCase.config, newPath.Child("KubeIPVSConfiguration"))
		if len(testCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
	}
}

func TestValidateKubeProxyConntrackConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := map[string]struct {
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.MaxPerCore"), -1, "must be greater than or equal to 0")},
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.Min"), -1, "must be greater than or equal to 0")},
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPEstablishedTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeConntrackConfiguration.TCPCloseWaitTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
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
	}

	for _, testCase := range testCases {
		errs := validateKubeProxyConntrackConfiguration(testCase.config, newPath.Child("KubeConntrackConfiguration"))
		if len(testCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
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

	testCases := map[string]struct {
		mode         kubeproxyconfig.ProxyMode
		expectedErrs field.ErrorList
	}{
		"blank mode should default": {
			mode:         kubeproxyconfig.ProxyMode(""),
			expectedErrs: field.ErrorList{},
		},
		"invalid mode non-existent": {
			mode:         kubeproxyconfig.ProxyMode("non-existing"),
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ProxyMode"), "non-existing", expectedNonExistentErrorMsg)},
		},
	}
	for _, testCase := range testCases {
		errs := validateProxyMode(testCase.mode, newPath)
		if len(testCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %v", testCase.expectedErrs[i], err.Error())
			}
		}
	}
}

func TestValidateClientConnectionConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := map[string]struct {
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
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("Burst"), -5, "must be greater than or equal to 0")},
		},
	}

	for _, testCase := range testCases {
		errs := validateClientConnectionConfiguration(testCase.ccc, newPath)
		if len(testCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
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

	errorCases := map[string]struct {
		ip           string
		expectedErrs field.ErrorList
	}{
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
	}

	for _, errorCase := range errorCases {
		errs := validateHostPort(errorCase.ip, newPath.Child("HealthzBindAddress"))
		if len(errorCase.expectedErrs) != len(errs) {
			t.Fatalf("Expected %d errors, got %d errors: %v", len(errorCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != errorCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", errorCase.expectedErrs[i], err.Error())
			}
		}
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

	testCases := map[string]struct {
		addresses    []string
		expectedErrs field.ErrorList
	}{
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
	}

	for _, testCase := range testCases {
		errs := validateKubeProxyNodePortAddress(testCase.addresses, newPath.Child("NodePortAddresses"))
		if len(testCase.expectedErrs) != len(errs) {
			t.Errorf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
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

	testCases := map[string]struct {
		addresses    []string
		expectedErrs field.ErrorList
	}{
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
	}

	for _, testCase := range testCases {
		errs := validateIPVSExcludeCIDRs(testCase.addresses, newPath.Child("ExcludeCIDRS"))
		if len(testCase.expectedErrs) != len(errs) {
			t.Errorf("Expected %d errors, got %d errors: %v", len(testCase.expectedErrs), len(errs), errs)
		}
		for i, err := range errs {
			if err.Error() != testCase.expectedErrs[i].Error() {
				t.Errorf("Expected error: %s, got %s", testCase.expectedErrs[i], err.Error())
			}
		}
	}
}
