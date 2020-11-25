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
	"runtime"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	componentbaseconfig "k8s.io/component-base/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"

	"k8s.io/utils/pointer"
)

func TestValidateKubeProxyConfiguration(t *testing.T) {
	var proxyMode kubeproxyconfig.ProxyMode
	if runtime.GOOS == "windows" {
		proxyMode = kubeproxyconfig.ProxyModeKernelspace
	} else {
		proxyMode = kubeproxyconfig.ProxyModeIPVS
	}
	successCases := []kubeproxyconfig.KubeProxyConfiguration{
		{
			BindAddress:        "192.168.59.103",
			HealthzBindAddress: "0.0.0.0:10256",
			MetricsBindAddress: "127.0.0.1:10249",
			ClusterCIDR:        "192.168.59.0/24",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
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
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "192.168.59.103",
			HealthzBindAddress: "0.0.0.0:10256",
			MetricsBindAddress: "127.0.0.1:10249",
			ClusterCIDR:        "192.168.59.0/24",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "192.168.59.103",
			HealthzBindAddress: "",
			MetricsBindAddress: "127.0.0.1:10249",
			ClusterCIDR:        "192.168.59.0/24",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "fd00:192:168:59::103",
			HealthzBindAddress: "",
			MetricsBindAddress: "[::1]:10249",
			ClusterCIDR:        "fd00:192:168:59::/64",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "10.10.12.11",
			HealthzBindAddress: "0.0.0.0:12345",
			MetricsBindAddress: "127.0.0.1:10249",
			FeatureGates:       map[string]bool{"IPv6DualStack": true, "EndpointSlice": true},
			ClusterCIDR:        "192.168.59.0/24",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "10.10.12.11",
			HealthzBindAddress: "0.0.0.0:12345",
			MetricsBindAddress: "127.0.0.1:10249",
			FeatureGates:       map[string]bool{"IPv6DualStack": true},
			ClusterCIDR:        "fd00:192:168::/64",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
		{
			BindAddress:        "10.10.12.11",
			HealthzBindAddress: "0.0.0.0:12345",
			MetricsBindAddress: "127.0.0.1:10249",
			FeatureGates:       map[string]bool{"IPv6DualStack": true},
			ClusterCIDR:        "192.168.59.0/24,fd00:192:168::/64",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
		},
	}

	for _, successCase := range successCases {
		if errs := Validate(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		config kubeproxyconfig.KubeProxyConfiguration
		msg    string
	}{
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				// only BindAddress is invalid
				BindAddress:        "10.10.12.11:2000",
				HealthzBindAddress: "0.0.0.0:10256",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "not a valid textual representation of an IP address",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress: "10.10.12.11",
				// only HealthzBindAddress is invalid
				HealthzBindAddress: "0.0.0.0",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be IP:port",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				// only MetricsBindAddress is invalid
				MetricsBindAddress: "127.0.0.1",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be IP:port",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// only ClusterCIDR is invalid
				ClusterCIDR:      "192.168.59.0",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// DualStack ClusterCIDR without feature flag enabled
				FeatureGates:     map[string]bool{"IPv6DualStack": false, "EndpointSlice": false},
				ClusterCIDR:      "192.168.59.0/24,fd00:192:168::/64",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "only one CIDR allowed (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// DualStack ClusterCIDR with feature flag enabled but EndpointSlice is not enabled
				FeatureGates:     map[string]bool{"IPv6DualStack": true, "EndpointSlice": false},
				ClusterCIDR:      "192.168.59.0/24,fd00:192:168::/64",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "EndpointSlice feature flag must be turned on",
		},

		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// DualStack with multiple CIDRs but only one IP family
				FeatureGates:     map[string]bool{"IPv6DualStack": true, "EndpointSlice": true},
				ClusterCIDR:      "192.168.59.0/24,10.0.0.0/16",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// DualStack with an invalid subnet
				FeatureGates:     map[string]bool{"IPv6DualStack": true, "EndpointSlice": true},
				ClusterCIDR:      "192.168.59.0/24,fd00:192:168::/64,a.b.c.d/f",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "only one CIDR allowed or a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				FeatureGates:       map[string]bool{"IPv6DualStack": true, "EndpointSlice": true},
				ClusterCIDR:        "192.168.59.0/24,fd00:192:168::/64,10.0.0.0/16",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "only one CIDR allowed or a valid DualStack CIDR (e.g. 10.100.0.0/16,fde4:8dba:82e1::/48)",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				// only UDPIdleTimeout is invalid
				UDPIdleTimeout:   metav1.Duration{Duration: -1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be greater than 0",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				// only ConfigSyncPeriod is invalid
				ConfigSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be greater than 0",
		},
		{
			config: kubeproxyconfig.KubeProxyConfiguration{
				BindAddress:        "192.168.59.103",
				HealthzBindAddress: "0.0.0.0:10256",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				// not specifying valid period in IPVS mode.
				Mode: kubeproxyconfig.ProxyModeIPVS,
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            pointer.Int32Ptr(1),
					Min:                   pointer.Int32Ptr(1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be greater than 0",
		},
	}

	for _, errorCase := range errorCases {
		if errs := Validate(&errorCase.config); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateKubeProxyIPTablesConfiguration(t *testing.T) {
	valid := int32(5)
	successCases := []kubeproxyconfig.KubeProxyIPTablesConfiguration{
		{
			MasqueradeAll: true,
			SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
			MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
		},
		{
			MasqueradeBit: &valid,
			MasqueradeAll: true,
			SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
			MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
		},
	}
	newPath := field.NewPath("KubeProxyConfiguration")
	for _, successCase := range successCases {
		if errs := validateKubeProxyIPTablesConfiguration(successCase, newPath.Child("KubeProxyIPTablesConfiguration")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	invalid := int32(-10)
	errorCases := []struct {
		config kubeproxyconfig.KubeProxyIPTablesConfiguration
		msg    string
	}{
		{
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			msg: "must be greater than 0",
		},
		{
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: &valid,
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: &invalid,
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			msg: "must be within the range [0, 31]",
		},
		// SyncPeriod must be >= MinSyncPeriod
		{
			config: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: &valid,
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			msg: fmt.Sprintf("must be greater than or equal to %s", newPath.Child("KubeProxyIPTablesConfiguration").Child("MinSyncPeriod").String()),
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateKubeProxyIPTablesConfiguration(errorCase.config, newPath.Child("KubeProxyIPTablesConfiguration")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateKubeProxyIPVSConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := []struct {
		config    kubeproxyconfig.KubeProxyIPVSConfiguration
		expectErr bool
		reason    string
	}{
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			expectErr: true,
			reason:    "SyncPeriod must be greater than 0",
		},
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 0 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectErr: true,
			reason:    "SyncPeriod must be greater than 0",
		},
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			expectErr: true,
			reason:    "MinSyncPeriod must be greater than or equal to 0",
		},
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 1 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectErr: true,
			reason:    "SyncPeriod must be greater than or equal to MinSyncPeriod",
		},
		// SyncPeriod == MinSyncPeriod
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			},
			expectErr: false,
		},
		// SyncPeriod > MinSyncPeriod
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 5 * time.Second},
			},
			expectErr: false,
		},
		// SyncPeriod can be 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 0 * time.Second},
			},
			expectErr: false,
		},
		// IPVS Timeout can be 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 0 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 0 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 0 * time.Second},
			},
			expectErr: false,
		},
		// IPVS Timeout > 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout:    metav1.Duration{Duration: 1 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 2 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 3 * time.Second},
			},
			expectErr: false,
		},
		// TCPTimeout Timeout < 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod: metav1.Duration{Duration: 5 * time.Second},
				TCPTimeout: metav1.Duration{Duration: -1 * time.Second},
			},
			expectErr: true,
			reason:    "TCPTimeout must be greater than or equal to 0",
		},
		// TCPFinTimeout Timeout < 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: -1 * time.Second},
			},
			expectErr: true,
			reason:    "TCPFinTimeout must be greater than or equal to 0",
		},
		// UDPTimeout Timeout < 0
		{
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				SyncPeriod: metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout: metav1.Duration{Duration: -1 * time.Second},
			},
			expectErr: true,
			reason:    "UDPTimeout must be greater than or equal to 0",
		},
	}

	for _, test := range testCases {
		errs := validateKubeProxyIPVSConfiguration(test.config, newPath.Child("KubeProxyIPVSConfiguration"))
		if len(errs) == 0 && test.expectErr {
			t.Errorf("Expect error, got nil, reason: %s", test.reason)
		}
		if len(errs) > 0 && !test.expectErr {
			t.Errorf("Unexpected error: %v", errs)
		}
	}
}

func TestValidateKubeProxyConntrackConfiguration(t *testing.T) {
	successCases := []kubeproxyconfig.KubeProxyConntrackConfiguration{
		{
			MaxPerCore:            pointer.Int32Ptr(1),
			Min:                   pointer.Int32Ptr(1),
			TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
		},
		{
			MaxPerCore:            pointer.Int32Ptr(0),
			Min:                   pointer.Int32Ptr(0),
			TCPEstablishedTimeout: &metav1.Duration{Duration: 0 * time.Second},
			TCPCloseWaitTimeout:   &metav1.Duration{Duration: 0 * time.Second},
		},
	}
	newPath := field.NewPath("KubeProxyConfiguration")
	for _, successCase := range successCases {
		if errs := validateKubeProxyConntrackConfiguration(successCase, newPath.Child("KubeProxyConntrackConfiguration")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		config kubeproxyconfig.KubeProxyConntrackConfiguration
		msg    string
	}{
		{
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(-1),
				Min:                   pointer.Int32Ptr(1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(-1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(3),
				TCPEstablishedTimeout: &metav1.Duration{Duration: -5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            pointer.Int32Ptr(1),
				Min:                   pointer.Int32Ptr(3),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: -5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateKubeProxyConntrackConfiguration(errorCase.config, newPath.Child("KubeProxyConntrackConfiguration")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateProxyMode(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	successCases := []kubeproxyconfig.ProxyMode{
		kubeproxyconfig.ProxyModeUserspace,
		kubeproxyconfig.ProxyMode(""),
	}

	if runtime.GOOS == "windows" {
		successCases = append(successCases, kubeproxyconfig.ProxyModeKernelspace)
	} else {
		successCases = append(successCases, kubeproxyconfig.ProxyModeIPTables, kubeproxyconfig.ProxyModeIPVS)
	}

	for _, successCase := range successCases {
		if errs := validateProxyMode(successCase, newPath.Child("ProxyMode")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		mode kubeproxyconfig.ProxyMode
		msg  string
	}{
		{
			mode: kubeproxyconfig.ProxyMode("non-existing"),
			msg:  "or blank (blank means the",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateProxyMode(errorCase.mode, newPath.Child("ProxyMode")); len(errs) == 0 {
			t.Errorf("expected failure %s for %v", errorCase.msg, errorCase.mode)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateClientConnectionConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []componentbaseconfig.ClientConnectionConfiguration{
		{
			Burst: 0,
		},
		{
			Burst: 5,
		},
	}

	for _, successCase := range successCases {
		if errs := validateClientConnectionConfiguration(successCase, newPath.Child("Burst")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		ccc componentbaseconfig.ClientConnectionConfiguration
		msg string
	}{
		{
			ccc: componentbaseconfig.ClientConnectionConfiguration{Burst: -5},
			msg: "must be greater than or equal to 0",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateClientConnectionConfiguration(errorCase.ccc, newPath.Child("Burst")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
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

	errorCases := []struct {
		ccc string
		msg string
	}{
		{
			ccc: "10.10.10.10",
			msg: "must be IP:port",
		},
		{
			ccc: "123.456.789.10:12345",
			msg: "must be a valid IP",
		},
		{
			ccc: "10.10.10.10:foo",
			msg: "must be a valid port",
		},
		{
			ccc: "10.10.10.10:0",
			msg: "must be a valid port",
		},
		{
			ccc: "10.10.10.10:65536",
			msg: "must be a valid port",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateHostPort(errorCase.ccc, newPath.Child("HealthzBindAddress")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateIPVSSchedulerMethod(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []kubeproxyconfig.IPVSSchedulerMethod{
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

	for _, successCase := range successCases {
		if errs := validateIPVSSchedulerMethod(successCase, newPath.Child("Scheduler")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		mode kubeproxyconfig.IPVSSchedulerMethod
		msg  string
	}{
		{
			mode: kubeproxyconfig.IPVSSchedulerMethod("non-existing"),
			msg:  "blank means the default algorithm method (currently rr)",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateIPVSSchedulerMethod(errorCase.mode, newPath.Child("ProxyMode")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
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
