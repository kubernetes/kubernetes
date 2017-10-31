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
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
)

func TestValidateKubeProxyConfiguration(t *testing.T) {
	successCases := []componentconfig.KubeProxyConfiguration{
		{
			BindAddress:        "192.168.59.103",
			HealthzBindAddress: "0.0.0.0:10256",
			MetricsBindAddress: "127.0.0.1:10249",
			ClusterCIDR:        "192.168.59.0/24",
			UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
			ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
			IPTables: componentconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			Conntrack: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(2),
				MaxPerCore: int32(1),
				Min:        int32(1),
				TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
			},
		},
	}

	for _, successCase := range successCases {
		if errs := Validate(&successCase); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		config componentconfig.KubeProxyConfiguration
		msg    string
	}{
		{
			config: componentconfig.KubeProxyConfiguration{
				// only BindAddress is invalid
				BindAddress:        "10.10.12.11:2000",
				HealthzBindAddress: "0.0.0.0:10256",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "not a valid textual representation of an IP address",
		},
		{
			config: componentconfig.KubeProxyConfiguration{
				BindAddress: "10.10.12.11",
				// only HealthzBindAddress is invalid
				HealthzBindAddress: "0.0.0.0",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be IP:port",
		},
		{
			config: componentconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				// only MetricsBindAddress is invalid
				MetricsBindAddress: "127.0.0.1",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod:   metav1.Duration{Duration: 1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be IP:port",
		},
		{
			config: componentconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				// only ClusterCIDR is invalid
				ClusterCIDR:      "192.168.59.0",
				UDPIdleTimeout:   metav1.Duration{Duration: 1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be a valid CIDR block (e.g. 10.100.0.0/16)",
		},
		{
			config: componentconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				// only UDPIdleTimeout is invalid
				UDPIdleTimeout:   metav1.Duration{Duration: -1 * time.Second},
				ConfigSyncPeriod: metav1.Duration{Duration: 1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
				},
			},
			msg: "must be greater than 0",
		},
		{
			config: componentconfig.KubeProxyConfiguration{
				BindAddress:        "10.10.12.11",
				HealthzBindAddress: "0.0.0.0:12345",
				MetricsBindAddress: "127.0.0.1:10249",
				ClusterCIDR:        "192.168.59.0/24",
				UDPIdleTimeout:     metav1.Duration{Duration: 1 * time.Second},
				// only ConfigSyncPeriod is invalid
				ConfigSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
				IPTables: componentconfig.KubeProxyIPTablesConfiguration{
					MasqueradeAll: true,
					SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
					MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
				},
				Conntrack: componentconfig.KubeProxyConntrackConfiguration{
					Max:        int32(2),
					MaxPerCore: int32(1),
					Min:        int32(1),
					TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
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
	successCases := []componentconfig.KubeProxyIPTablesConfiguration{
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
		config componentconfig.KubeProxyIPTablesConfiguration
		msg    string
	}{
		{
			config: componentconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: -5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			msg: "must be greater than 0",
		},
		{
			config: componentconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: &valid,
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: -1 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: componentconfig.KubeProxyIPTablesConfiguration{
				MasqueradeBit: &invalid,
				MasqueradeAll: true,
				SyncPeriod:    metav1.Duration{Duration: 5 * time.Second},
				MinSyncPeriod: metav1.Duration{Duration: 2 * time.Second},
			},
			msg: "must be within the range [0, 31]",
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

func TestValidateKubeProxyConntrackConfiguration(t *testing.T) {
	successCases := []componentconfig.KubeProxyConntrackConfiguration{
		{
			Max:        int32(2),
			MaxPerCore: int32(1),
			Min:        int32(1),
			TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
			TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
		},
		{
			Max:        0,
			MaxPerCore: 0,
			Min:        0,
			TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
			TCPCloseWaitTimeout:   metav1.Duration{Duration: 60 * time.Second},
		},
	}
	newPath := field.NewPath("KubeProxyConfiguration")
	for _, successCase := range successCases {
		if errs := validateKubeProxyConntrackConfiguration(successCase, newPath.Child("KubeProxyConntrackConfiguration")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		config componentconfig.KubeProxyConntrackConfiguration
		msg    string
	}{
		{
			config: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(-1),
				MaxPerCore: int32(1),
				Min:        int32(1),
				TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(2),
				MaxPerCore: int32(-1),
				Min:        int32(1),
				TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(2),
				MaxPerCore: int32(1),
				Min:        int32(-1),
				TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than or equal to 0",
		},
		{
			config: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(4),
				MaxPerCore: int32(1),
				Min:        int32(3),
				TCPEstablishedTimeout: metav1.Duration{Duration: -5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 5 * time.Second},
			},
			msg: "must be greater than 0",
		},
		{
			config: componentconfig.KubeProxyConntrackConfiguration{
				Max:        int32(4),
				MaxPerCore: int32(1),
				Min:        int32(3),
				TCPEstablishedTimeout: metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   metav1.Duration{Duration: -5 * time.Second},
			},
			msg: "must be greater than 0",
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

	successCases := []componentconfig.ProxyMode{
		componentconfig.ProxyModeUserspace,
		componentconfig.ProxyModeIPTables,
		componentconfig.ProxyModeIPVS,
		componentconfig.ProxyMode(""),
	}

	for _, successCase := range successCases {
		if errs := validateProxyMode(successCase, newPath.Child("ProxyMode")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		mode componentconfig.ProxyMode
		msg  string
	}{
		{
			mode: componentconfig.ProxyMode("non-existing"),
			msg:  "or blank (blank means the best-available proxy (currently iptables)",
		},
	}

	for _, errorCase := range errorCases {
		if errs := validateProxyMode(errorCase.mode, newPath.Child("ProxyMode")); len(errs) == 0 {
			t.Errorf("expected failure for %s", errorCase.msg)
		} else if !strings.Contains(errs[0].Error(), errorCase.msg) {
			t.Errorf("unexpected error: %v, expected: %s", errs[0], errorCase.msg)
		}
	}
}

func TestValidateClientConnectionConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	successCases := []componentconfig.ClientConnectionConfiguration{
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
		ccc componentconfig.ClientConnectionConfiguration
		msg string
	}{
		{
			ccc: componentconfig.ClientConnectionConfiguration{Burst: -5},
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

	successCases := []componentconfig.IPVSSchedulerMethod{
		componentconfig.RoundRobin,
		componentconfig.WeightedRoundRobin,
		componentconfig.LeastConnection,
		componentconfig.WeightedLeastConnection,
		componentconfig.LocalityBasedLeastConnection,
		componentconfig.LocalityBasedLeastConnectionWithReplication,
		componentconfig.SourceHashing,
		componentconfig.DestinationHashing,
		componentconfig.ShortestExpectedDelay,
		componentconfig.NeverQueue,
		"",
	}

	for _, successCase := range successCases {
		if errs := validateIPVSSchedulerMethod(successCase, newPath.Child("Scheduler")); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}

	errorCases := []struct {
		mode componentconfig.IPVSSchedulerMethod
		msg  string
	}{
		{
			mode: componentconfig.IPVSSchedulerMethod("non-existing"),
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
