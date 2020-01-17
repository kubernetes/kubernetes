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
			msg: "must be a valid CIDR block (e.g. 10.100.0.0/16 or FD02::0:0:0/96)",
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

	errorCases := []struct {
		addresses []string
		msg       string
	}{
		{
			addresses: []string{"foo"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"1.2.3"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{""},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"10.20.30.40"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"::1"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"2001:db8:1"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"2001:db8:xyz/64"},
			msg:       "must be a valid IP block",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateKubeProxyNodePortAddress(errorCase.addresses, newPath.Child("NodePortAddresses")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateKubeProxyExcludeCIDRs(t *testing.T) {
	// TODO(rramkumar): This test is a copy of TestValidateKubeProxyNodePortAddress.
	// Maybe some code can be shared?
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

	errorCases := []struct {
		addresses []string
		msg       string
	}{
		{
			addresses: []string{"foo"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"1.2.3"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{""},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"10.20.30.40"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"::1"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"2001:db8:1"},
			msg:       "must be a valid IP block",
		},
		{
			addresses: []string{"2001:db8:xyz/64"},
			msg:       "must be a valid IP block",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateIPVSExcludeCIDRs(errorCase.addresses, newPath.Child("ExcludeCIDRs")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}
