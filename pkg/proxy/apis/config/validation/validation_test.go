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
			HealthzBindAddresses: []string{"0.0.0.0/0"},
			HealthzBindPort:      10256,
			MetricsBindAddresses: []string{"127.0.0.0/8"},
			MetricsBindPort:      10249,
			SyncPeriod:           metav1.Duration{Duration: 5 * time.Second},
			MinSyncPeriod:        metav1.Duration{Duration: 2 * time.Second},
			ConfigSyncPeriod:     metav1.Duration{Duration: 1 * time.Second},
			Linux: kubeproxyconfig.KubeProxyLinuxConfiguration{
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            ptr.To[int32](1),
					Min:                   ptr.To[int32](1),
					TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				},
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
				}
			},
		},
		{
			name: "ipv6 only cluster",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.NodeIPOverride = []string{"fd00:192:168:59::103"}
				config.HealthzBindAddresses = []string{"::/0"}
				config.MetricsBindAddresses = []string{"::1/128"}
			},
		},
		{
			name: "dual stack cluster",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.NodeIPOverride = []string{"192.168.59.103", "fd00:192:168:59::103"}
				config.HealthzBindAddresses = []string{"0.0.0.0/0", "::/0"}
				config.MetricsBindAddresses = []string{"127.0.0.0/8", "::1/128"}
			},
		},
		{
			name: "custom healthz host port",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddresses = []string{"192.168.0.0/16"}
				config.HealthzBindPort = 54321
			},
		},
		{
			name: "custom metrics host port",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MetricsBindAddresses = []string{"172.16.0.0/16"}
				config.MetricsBindPort = 3306
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
			name: "invalid NodeIPOverride",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.NodeIPOverride = []string{"10.10.12.11:2000"}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("NodeIPOverride").Index(0), "10.10.12.11:2000", "must be a valid IP (e.g. 10.100.0.0 or fde4:8dba:82e1::)")},
		},
		{
			name: "invalid HealthzBindAddresses",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindAddresses = []string{"0.0.0.0"}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindAddresses").Index(0), "0.0.0.0", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		{
			name: "invalid HealthzBindPort",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.HealthzBindPort = 70000
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("HealthzBindPort"), 70000, "must be a valid port")},
		},
		{
			name: "invalid MetricsBindAddresses",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MetricsBindAddresses = []string{"127.0.0.1"}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MetricsBindAddresses").Index(0), "127.0.0.1", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		{
			name: "invalid MetricsBindPort",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MetricsBindPort = -250
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MetricsBindPort"), -250, "must be a valid port")},
		},
		{
			name: "SyncPeriod must be > 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.SyncPeriod = metav1.Duration{Duration: -5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("SyncPeriod"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than 0"),
				field.Invalid(newPath.Child("SyncPeriod"), metav1.Duration{Duration: 2 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.MinSyncPeriod")},
		},
		{
			name: "MinSyncPeriod must be > 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.MinSyncPeriod = metav1.Duration{Duration: -1 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("MinSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "SyncPeriod must be >= MinSyncPeriod",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.SyncPeriod = metav1.Duration{Duration: 1 * time.Second}
				config.MinSyncPeriod = metav1.Duration{Duration: 5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("SyncPeriod"), metav1.Duration{Duration: 5 * time.Second}, "must be greater than or equal to KubeProxyConfiguration.MinSyncPeriod")},
		},
		{
			name: "ConfigSyncPeriod must be > 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyConfiguration) {
				config.ConfigSyncPeriod = metav1.Duration{Duration: -1 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("ConfigSyncPeriod"), metav1.Duration{Duration: -1 * time.Second}, "must be greater than 0")},
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

func TestValidateIPVSTimeout(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	testCases := []struct {
		name         string
		config       kubeproxyconfig.KubeProxyIPVSConfiguration
		expectedErrs field.ErrorList
	}{
		{
			name: "IPVS Timeout can be 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				TCPTimeout:    metav1.Duration{Duration: 0 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 0 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 0 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "IPVS Timeout > 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
				TCPTimeout:    metav1.Duration{Duration: 1 * time.Second},
				TCPFinTimeout: metav1.Duration{Duration: 2 * time.Second},
				UDPTimeout:    metav1.Duration{Duration: 3 * time.Second},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "TCP,TCPFin,UDP Timeouts < 0",
			config: kubeproxyconfig.KubeProxyIPVSConfiguration{
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

func TestValidateKubeProxyLinuxConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")
	getBaseConfig := func() *kubeproxyconfig.KubeProxyLinuxConfiguration {
		return &kubeproxyconfig.KubeProxyLinuxConfiguration{
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore:            ptr.To[int32](1),
				Min:                   ptr.To[int32](1),
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 5 * time.Second},
				UDPTimeout:            metav1.Duration{Duration: 5 * time.Second},
				UDPStreamTimeout:      metav1.Duration{Duration: 5 * time.Second},
			},
		}
	}

	testCases := []struct {
		name             string
		mutateConfigFunc func(*kubeproxyconfig.KubeProxyLinuxConfiguration)
		expectedErrs     field.ErrorList
	}{
		{
			name: "invalid positive oom score",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.OOMScoreAdj = ptr.To[int32](9000)
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.OOMScoreAdj"), 9000, "must be within the range [-1000, 1000]")},
		},
		{
			name: "invalid negative oom score",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.OOMScoreAdj = ptr.To[int32](-9000)
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.OOMScoreAdj"), -9000, "must be within the range [-1000, 1000]")},
		},
		{
			name: "valid oom score and conntrack timeouts",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.OOMScoreAdj = ptr.To[int32](900)
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "valid duration equal to 0 second conntrack timeout",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.OOMScoreAdj = ptr.To[int32](-900)
				config.Conntrack.TCPEstablishedTimeout = &metav1.Duration{Duration: 0 * time.Second}
				config.Conntrack.TCPCloseWaitTimeout = &metav1.Duration{Duration: 0 * time.Second}
				config.Conntrack.UDPTimeout = metav1.Duration{Duration: 0 * time.Second}
				config.Conntrack.UDPStreamTimeout = metav1.Duration{Duration: 0 * time.Second}
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "invalid Conntrack.MaxPerCore < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.MaxPerCore = ptr.To[int32](-1)
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.MaxPerCore"), -1, "must be greater than or equal to 0")},
		},
		{
			name: "invalid Conntrack.Min < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.Min = ptr.To[int32](-1)
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.Min"), -1, "must be greater than or equal to 0")},
		},
		{
			name: "invalid Conntrack.TCPEstablishedTimeout < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.TCPEstablishedTimeout = &metav1.Duration{Duration: -5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.TCPEstablishedTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid Conntrack.TCPCloseWaitTimeout < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.TCPCloseWaitTimeout = &metav1.Duration{Duration: -5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.TCPCloseWaitTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid Conntrack.UDPTimeout < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.UDPTimeout = metav1.Duration{Duration: -5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.UDPTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
		{
			name: "invalid Conntrack.UDPStreamTimeout < 0",
			mutateConfigFunc: func(config *kubeproxyconfig.KubeProxyLinuxConfiguration) {
				config.Conntrack.UDPStreamTimeout = metav1.Duration{Duration: -5 * time.Second}
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("KubeProxyLinuxConfiguration.KubeProxyConntrackConfiguration.UDPStreamTimeout"), metav1.Duration{Duration: -5 * time.Second}, "must be greater than or equal to 0")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			config := getBaseConfig()
			tc.mutateConfigFunc(config)
			errs := validateKubeProxyLinuxConfiguration(*config, newPath.Child("KubeProxyLinuxConfiguration"))
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

func TestValidateDetectLocalConfiguration(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name           string
		mode           kubeproxyconfig.LocalMode
		config         kubeproxyconfig.DetectLocalConfiguration
		configHardFail *bool
		expectedErrs   field.ErrorList
	}{
		{
			name: "valid interface name prefix",
			mode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
			config: kubeproxyconfig.DetectLocalConfiguration{
				InterfaceNamePrefix: "vethabcde",
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "valid bridge interface",
			mode: kubeproxyconfig.LocalModeBridgeInterface,
			config: kubeproxyconfig.DetectLocalConfiguration{
				BridgeInterface: "avz",
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "interfacePrefix is empty",
			mode: kubeproxyconfig.LocalModeInterfaceNamePrefix,
			config: kubeproxyconfig.DetectLocalConfiguration{
				InterfaceNamePrefix: "",
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DetectLocal").Child("InterfacePrefix"), "", "must not be empty")},
		},
		{
			name: "bridgeInterfaceName is empty",
			mode: kubeproxyconfig.LocalModeBridgeInterface,
			config: kubeproxyconfig.DetectLocalConfiguration{
				InterfaceNamePrefix: "eth0", // we won't care about prefix since mode is not prefix
			},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DetectLocal").Child("InterfaceName"), "", "must not be empty")},
		},
		{
			name: "valid cluster cidr",
			mode: kubeproxyconfig.LocalModeClusterCIDR,
			config: kubeproxyconfig.DetectLocalConfiguration{
				ClusterCIDRs: []string{"192.168.59.0/24", "fd00:192:168::/64"},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "empty cluster cidrs with cluster cidr mode",
			mode: kubeproxyconfig.LocalModeClusterCIDR,
			config: kubeproxyconfig.DetectLocalConfiguration{
				ClusterCIDRs: []string{},
			},
			expectedErrs: field.ErrorList{},
		},
		{
			name: "empty cluster cidrs with cluster cidr mode and config hard fail true",
			mode: kubeproxyconfig.LocalModeClusterCIDR,
			config: kubeproxyconfig.DetectLocalConfiguration{
				ClusterCIDRs: []string{},
			},
			configHardFail: ptr.To(true),
			expectedErrs:   field.ErrorList{field.Invalid(newPath.Child("DetectLocal").Child("ClusterCIDRs"), []string{}, "must contain at least one CIDR")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateDetectLocalConfiguration(tc.mode, tc.config, tc.configHardFail, newPath.Child("DetectLocal"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateMasqueradeBit(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name          string
		masqueradeBit *int32
		expectedErrs  field.ErrorList
	}{
		{
			name:          "valid custom MasqueradeBit",
			masqueradeBit: ptr.To[int32](5),
			expectedErrs:  field.ErrorList{},
		},
		{
			name:          "masqueradeBit cannot be < 0",
			masqueradeBit: ptr.To[int32](-10),
			expectedErrs:  field.ErrorList{field.Invalid(newPath.Child("BackendConfiguration.MasqueradeBit"), -10, "must be within the range [0, 31]")},
		},
		{
			name:          "masqueradeBit cannot be > 31",
			masqueradeBit: ptr.To[int32](32),
			expectedErrs:  field.ErrorList{field.Invalid(newPath.Child("BackendConfiguration.MasqueradeBit"), 32, "must be within the range [0, 31]")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateMasqueradeBit(tc.masqueradeBit, newPath.Child("BackendConfiguration").Child("MasqueradeBit"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateDualStackCIDRStrings(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name         string
		cidrStrings  []string
		expectedErrs field.ErrorList
	}{
		{
			name:         "empty cidr string",
			cidrStrings:  []string{},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackCIDRList"), []string{}, "must contain at least one CIDR")},
		},
		{
			name:         "single ipv4 cidr",
			cidrStrings:  []string{"192.168.0.0/16"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "single ipv6 cidr",
			cidrStrings:  []string{"fd00:10:96::/112"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "dual stack cidr pair",
			cidrStrings:  []string{"172.16.200.0/24", "fde4:8dba:82e1::/48"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "multiple ipv4 cidrs",
			cidrStrings:  []string{"10.100.0.0/16", "192.168.0.0/16", "fde4:8dba:82e1::/48"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackCIDRList"), []string{"10.100.0.0/16", "192.168.0.0/16", "fde4:8dba:82e1::/48"}, "must be a either a single CIDR or dual-stack pair of CIDRs (e.g. [10.100.0.0/16, fde4:8dba:82e1::/48]")},
		},
		{
			name:         "multiple ipv6 cidrs",
			cidrStrings:  []string{"fd00:10:96::/112", "fde4:8dba:82e1::/48"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackCIDRList"), []string{"fd00:10:96::/112", "fde4:8dba:82e1::/48"}, "must be a either a single CIDR or dual-stack pair of CIDRs (e.g. [10.100.0.0/16, fde4:8dba:82e1::/48]")},
		},
		{
			name:         "malformed ipv6 cidr",
			cidrStrings:  []string{"fd00:10:96::", "192.168.0.0/16"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackCIDRList").Index(0), "fd00:10:96::", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
		{
			name:         "malformed ipv4 cidr",
			cidrStrings:  []string{"fde4:8dba:82e1::/48", "172.16.200.0:24"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackCIDRList").Index(1), "172.16.200.0:24", "must be a valid CIDR block (e.g. 10.100.0.0/16 or fde4:8dba:82e1::/48)")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateDualStackCIDRStrings(tc.cidrStrings, newPath.Child("DualStackCIDRList"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}

func TestValidateDualStackIPStrings(t *testing.T) {
	newPath := field.NewPath("KubeProxyConfiguration")

	testCases := []struct {
		name         string
		ipStrings    []string
		expectedErrs field.ErrorList
	}{
		{
			name:         "empty cidr string",
			ipStrings:    []string{},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackIPList"), []string{}, "must contain at least one IP")},
		},
		{
			name:         "single ipv4",
			ipStrings:    []string{"192.168.0.0"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "single ipv6",
			ipStrings:    []string{"fd00:10:96::"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "dual stack ip pair",
			ipStrings:    []string{"172.16.200.0", "fde4:8dba:82e1::"},
			expectedErrs: field.ErrorList{},
		},
		{
			name:         "multiple ipv4",
			ipStrings:    []string{"10.100.0.0", "192.168.0.0", "fde4:8dba:82e1::"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackIPList"), []string{"10.100.0.0", "192.168.0.0", "fde4:8dba:82e1::"}, "must be a either a single IP or dual-stack pair of IPs (e.g. [10.100.0.0, fde4:8dba:82e1::])")},
		},
		{
			name:         "multiple ipv6",
			ipStrings:    []string{"fd00:10:96::", "fde4:8dba:82e1::"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackIPList"), []string{"fd00:10:96::", "fde4:8dba:82e1::"}, "must be a either a single IP or dual-stack pair of IPs (e.g. [10.100.0.0, fde4:8dba:82e1::])")},
		},
		{
			name:         "malformed ipv6 cidr",
			ipStrings:    []string{"[fd00:10:96::]:54321", "192.168.0.0"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackIPList").Index(0), "[fd00:10:96::]:54321", "must be a valid IP (e.g. 10.100.0.0 or fde4:8dba:82e1::)")},
		},
		{
			name:         "malformed ipv4 cidr",
			ipStrings:    []string{"fde4:8dba:82e1::", "172.16.200.0:3306"},
			expectedErrs: field.ErrorList{field.Invalid(newPath.Child("DualStackIPList").Index(1), "172.16.200.0:3306", "must be a valid IP (e.g. 10.100.0.0 or fde4:8dba:82e1::)")},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := validateDualStackIPStrings(tc.ipStrings, newPath.Child("DualStackIPList"))
			assert.Equalf(t, len(tc.expectedErrs), len(errs),
				"expected %d errors, got %d errors: %v", len(tc.expectedErrs), len(errs), errs,
			)
			for i, err := range errs {
				assert.Equal(t, tc.expectedErrs[i].Error(), err.Error())
			}
		})
	}
}
