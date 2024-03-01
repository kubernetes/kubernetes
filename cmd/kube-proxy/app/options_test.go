/*
Copyright 2024 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	componentbaseconfig "k8s.io/component-base/config"
	logsapi "k8s.io/component-base/logs/api/v1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/utils/ptr"
)

// TestLoadConfigV1Alpha1 tests proper operation of loadConfig()
func TestLoadConfigV1Alpha1(t *testing.T) {

	yamlTemplate := `apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: %s
clientConnection:
  acceptContentTypes: "abc"
  burst: 100
  contentType: content-type
  kubeconfig: "/path/to/kubeconfig"
  qps: 7
clusterCIDR: "%s"
configSyncPeriod: 15s
conntrack:
  maxPerCore: 2
  min: 1
  tcpCloseWaitTimeout: 10s
  tcpEstablishedTimeout: 20s
healthzBindAddress: "%s"
hostnameOverride: "foo"
iptables:
  masqueradeAll: true
  masqueradeBit: 17
  minSyncPeriod: 10s
  syncPeriod: 60s
  localhostNodePorts: true
ipvs:
  minSyncPeriod: 10s
  syncPeriod: 60s
  excludeCIDRs:
    - "10.20.30.40/16"
    - "fd00:1::0/64"
nftables:
  masqueradeAll: true
  masqueradeBit: 18
  minSyncPeriod: 10s
  syncPeriod: 60s
kind: KubeProxyConfiguration
metricsBindAddress: "%s"
mode: "%s"
oomScoreAdj: 17
portRange: "2-7"
detectLocalMode: "ClusterCIDR"
detectLocal:
  bridgeInterface: "cbr0"
  interfaceNamePrefix: "veth"
nodePortAddresses:
  - "10.20.30.40/16"
  - "fd00:1::0/64"
`

	testCases := []struct {
		name                         string
		mode                         string
		bindAddress                  string
		clusterCIDR                  string
		healthzBindAddress           string
		metricsBindAddress           string
		extraConfig                  string
		expectedHealthzBindAddresses []string
		expectedHealthzPort          int32
		expectedMetricsBindAddresses []string
		expectedMetricsPort          int32
	}{
		{
			name:                         "iptables mode, IPv4 all-zeros bind address",
			mode:                         "iptables",
			bindAddress:                  "0.0.0.0",
			clusterCIDR:                  "1.2.3.0/24",
			healthzBindAddress:           "1.2.3.4:12345",
			metricsBindAddress:           "2.3.4.5:23456",
			expectedHealthzBindAddresses: []string{"1.2.3.4/32"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"2.3.4.5/32"},
			expectedMetricsPort:          int32(23456),
		},
		{
			name:                         "iptables mode, non-zeros IPv4 config",
			mode:                         "iptables",
			bindAddress:                  "9.8.7.6",
			clusterCIDR:                  "1.2.3.0/24",
			healthzBindAddress:           "1.2.3.4:12345",
			metricsBindAddress:           "2.3.4.5:23456",
			expectedHealthzBindAddresses: []string{"1.2.3.4/32"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"2.3.4.5/32"},
			expectedMetricsPort:          int32(23456),
		},
		{
			// Test for 'bindAddress: "::"' (IPv6 all-zeros) in kube-proxy
			// config file. The user will need to put quotes around '::' since
			// 'bindAddress: ::' is invalid yaml syntax.
			name:                         "iptables mode, IPv6 \"::\" bind address",
			mode:                         "iptables",
			bindAddress:                  "\"::\"",
			clusterCIDR:                  "fd00:1::0/64",
			healthzBindAddress:           "[fd00:1::5]:12345",
			metricsBindAddress:           "[fd00:2::5]:23456",
			expectedHealthzBindAddresses: []string{"fd00:1::5/128"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"fd00:2::5/128"},
			expectedMetricsPort:          int32(23456),
		},
		{
			// Test for 'bindAddress: "[::]"' (IPv6 all-zeros in brackets)
			// in kube-proxy config file. The user will need to use
			// surrounding quotes here since 'bindAddress: [::]' is invalid
			// yaml syntax.
			name:                         "iptables mode, IPv6 \"[::]\" bind address",
			mode:                         "iptables",
			bindAddress:                  "\"[::]\"",
			clusterCIDR:                  "fd00:1::0/64",
			healthzBindAddress:           "[fd00:1::5]:12345",
			metricsBindAddress:           "[fd00:2::5]:23456",
			expectedHealthzBindAddresses: []string{"fd00:1::5/128"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"fd00:2::5/128"},
			expectedMetricsPort:          int32(23456),
		},
		{
			// Test for 'bindAddress: ::0' (another form of IPv6 all-zeros).
			// No surrounding quotes are required around '::0'.
			name:                         "iptables mode, IPv6 ::0 bind address",
			mode:                         "iptables",
			bindAddress:                  "::0",
			clusterCIDR:                  "fd00:1::0/64",
			healthzBindAddress:           "[fd00:1::5]:12345",
			metricsBindAddress:           "[fd00:2::5]:23456",
			expectedHealthzBindAddresses: []string{"fd00:1::5/128"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"fd00:2::5/128"},
			expectedMetricsPort:          int32(23456),
		},
		{
			name:                         "ipvs mode, IPv6 config",
			mode:                         "ipvs",
			bindAddress:                  "2001:db8::1",
			clusterCIDR:                  "fd00:1::0/64",
			healthzBindAddress:           "[fd00:1::5]:12345",
			metricsBindAddress:           "[fd00:2::5]:23456",
			expectedHealthzBindAddresses: []string{"fd00:1::5/128"},
			expectedHealthzPort:          int32(12345),
			expectedMetricsBindAddresses: []string{"fd00:2::5/128"},
			expectedMetricsPort:          int32(23456),
		},
		{
			// Test for unknown field within config.
			// For v1alpha1 a lenient path is implemented and will throw a
			// strict decoding warning instead of failing to load
			name:                         "unknown field",
			mode:                         "iptables",
			bindAddress:                  "9.8.7.6",
			clusterCIDR:                  "1.2.3.0/24",
			healthzBindAddress:           "4.3.2.1:54321",
			metricsBindAddress:           "21.31.41.51:3306",
			extraConfig:                  "foo: bar",
			expectedHealthzBindAddresses: []string{"4.3.2.1/32"},
			expectedHealthzPort:          int32(54321),
			expectedMetricsBindAddresses: []string{"21.31.41.51/32"},
			expectedMetricsPort:          int32(3306),
		},
		{
			// Test for duplicate field within config.
			// For v1alpha1 a lenient path is implemented and will throw a
			// strict decoding warning instead of failing to load
			name:                         "duplicate field",
			mode:                         "iptables",
			bindAddress:                  "9.8.7.6",
			clusterCIDR:                  "1.2.3.0/24",
			healthzBindAddress:           "12.23.34.45:8080",
			metricsBindAddress:           "35.45.55.65:9090",
			extraConfig:                  "bindAddress: 9.8.7.6",
			expectedHealthzBindAddresses: []string{"12.23.34.45/32"},
			expectedHealthzPort:          int32(8080),
			expectedMetricsBindAddresses: []string{"35.45.55.65/32"},
			expectedMetricsPort:          int32(9090),
		},
		{
			// Test for unspecified healthz and metrics address.
			// conversion from v1alpha1 to internal should add /0 prefix for
			// unspecified address
			name:                         "unspecified healthz and metrics address",
			mode:                         "iptables",
			bindAddress:                  "9.8.7.6",
			clusterCIDR:                  "1.2.3.0/24",
			healthzBindAddress:           "0.0.0.0:45678",
			metricsBindAddress:           "[::]:56789",
			extraConfig:                  "bindAddress: 9.8.7.6",
			expectedHealthzBindAddresses: []string{"0.0.0.0/0"},
			expectedHealthzPort:          int32(45678),
			expectedMetricsBindAddresses: []string{"::/0"},
			expectedMetricsPort:          int32(56789),
		},
	}

	for _, tc := range testCases {
		expBindAddr := tc.bindAddress
		if tc.bindAddress[0] == '"' {
			// Surrounding double quotes will get stripped by the yaml parser.
			expBindAddr = expBindAddr[1 : len(tc.bindAddress)-1]
		}
		expected := &kubeproxyconfig.KubeProxyConfiguration{
			NodeIPOverride: []string{expBindAddr},
			ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
				AcceptContentTypes: "abc",
				Burst:              100,
				ContentType:        "content-type",
				Kubeconfig:         "/path/to/kubeconfig",
				QPS:                7,
			},

			MinSyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
			SyncPeriod:       metav1.Duration{Duration: 60 * time.Second},
			ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
			Linux: kubeproxyconfig.KubeProxyLinuxConfiguration{
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            ptr.To[int32](2),
					Min:                   ptr.To[int32](1),
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 10 * time.Second},
					TCPEstablishedTimeout: &metav1.Duration{Duration: 20 * time.Second},
				},
				MasqueradeAll: true,
				OOMScoreAdj:   ptr.To[int32](17),
			},
			FeatureGates:         map[string]bool{},
			HealthzBindAddresses: tc.expectedHealthzBindAddresses,
			HealthzBindPort:      tc.expectedHealthzPort,
			HostnameOverride:     "foo",
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{

				MasqueradeBit:      ptr.To[int32](17),
				LocalhostNodePorts: ptr.To(true),
			},
			IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
				MasqueradeBit: ptr.To[int32](17),
				ExcludeCIDRs:  []string{"10.20.30.40/16", "fd00:1::0/64"},
			},
			NFTables: kubeproxyconfig.KubeProxyNFTablesConfiguration{
				MasqueradeBit: ptr.To[int32](18),
			},
			MetricsBindAddresses: tc.expectedMetricsBindAddresses,
			MetricsBindPort:      tc.expectedMetricsPort,
			Mode:                 kubeproxyconfig.ProxyMode(tc.mode),
			NodePortAddresses:    []string{"10.20.30.40/16", "fd00:1::0/64"},
			DetectLocalMode:      kubeproxyconfig.LocalModeClusterCIDR,
			DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
				BridgeInterface:     "cbr0",
				ClusterCIDRs:        strings.Split(tc.clusterCIDR, ","),
				InterfaceNamePrefix: "veth",
			},
			Logging: logsapi.LoggingConfiguration{
				Format:         "text",
				FlushFrequency: logsapi.TimeOrMetaDuration{Duration: metav1.Duration{Duration: 5 * time.Second}, SerializeAsString: true},
			},
		}

		options := NewOptions()

		baseYAML := fmt.Sprintf(
			yamlTemplate, tc.bindAddress, tc.clusterCIDR,
			tc.healthzBindAddress, tc.metricsBindAddress, tc.mode)

		// Append additional configuration to the base yaml template
		yaml := fmt.Sprintf("%s\n%s", baseYAML, tc.extraConfig)

		config, err := options.loadConfig([]byte(yaml))

		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)

		if diff := cmp.Diff(config, expected); diff != "" {
			t.Fatalf("unexpected config for %s, diff = %s", tc.name, diff)
		}
	}
}

// TestLoadConfigV1Alpha2 tests proper operation of loadConfig()
func TestLoadConfigV1Alpha2(t *testing.T) {

	yamlTemplate := `apiVersion: kubeproxy.config.k8s.io/v1alpha2
clientConnection:
  acceptContentTypes: "abc"
  burst: 100
  contentType: content-type
  kubeconfig: "/path/to/kubeconfig"
  qps: 7
configSyncPeriod: 15s
healthzBindAddresses: %s
healthzBindPort: %d
hostnameOverride: "foo"
iptables:
  masqueradeBit: 12
  localhostNodePorts: false
ipvs:
  excludeCIDRs:
    - "10.20.30.40/16"
    - "fd00:1::0/64"
  masqueradeBit: 15
nftables:
  masqueradeBit: 18
kind: KubeProxyConfiguration
linux:
  conntrack:
    maxPerCore: 2
    min: 1
    tcpCloseWaitTimeout: 10s
    tcpEstablishedTimeout: 20s
  masqueradeAll: false
  oomScoreAdj: 17
metricsBindAddresses: %s
metricsBindPort: %d
minSyncPeriod: 10s
mode: "%s"
detectLocalMode: "ClusterCIDR"
detectLocal:
  bridgeInterface: "cbr0"
  clusterCIDRs: %s
  interfaceNamePrefix: "veth"
nodePortAddresses:
  - "10.20.30.40/16"
  - "fd00:1::0/64"
syncPeriod: 60s
`

	testCases := []struct {
		name                 string
		mode                 string
		clusterCIDRs         []string
		healthzBindAddresses []string
		healthzBindPort      int32
		metricsBindAddresses []string
		metricsBindPort      int32
		extraConfig          string
	}{
		{
			name:                 "iptables mode, IPv4 family",
			mode:                 "iptables",
			clusterCIDRs:         []string{"10.244.0.0/16"},
			healthzBindAddresses: []string{"192.168.11.0/24"},
			healthzBindPort:      10256,
			metricsBindAddresses: []string{"172.16.120.0/24"},
			metricsBindPort:      10249,
		},
		{
			name:                 "ipvs mode, IPv6 family",
			mode:                 "ipvs",
			clusterCIDRs:         []string{"fd00:1::0/64"},
			healthzBindAddresses: []string{"fd00:1::5/48"},
			healthzBindPort:      54321,
			metricsBindAddresses: []string{"fd00:2::5/48"},
			metricsBindPort:      98765,
		},
		{
			name:                 "nftables mode, dual stack",
			mode:                 "nftables",
			clusterCIDRs:         []string{"10.244.0.0/16", "fd00:1::0/64"},
			healthzBindAddresses: []string{"192.168.11.0/24", "fd00:1::5/48"},
			healthzBindPort:      54321,
			metricsBindAddresses: []string{"172.16.120.0/24", "fd00:2::5/48"},
			metricsBindPort:      98765,
		},
		{
			name:                 "dual stack unspecified addresses",
			mode:                 "nftables",
			clusterCIDRs:         []string{"10.244.0.0/16", "fd00:1::0/64"},
			healthzBindAddresses: []string{"0.0.0.0/0", "::/0"},
			healthzBindPort:      54321,
			metricsBindAddresses: []string{"127.0.0.0/8", "::1/128"},
			metricsBindPort:      98765,
		},
	}

	for _, tc := range testCases {
		expected := &kubeproxyconfig.KubeProxyConfiguration{
			ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
				AcceptContentTypes: "abc",
				Burst:              100,
				ContentType:        "content-type",
				Kubeconfig:         "/path/to/kubeconfig",
				QPS:                7,
			},

			MinSyncPeriod:    metav1.Duration{Duration: 10 * time.Second},
			SyncPeriod:       metav1.Duration{Duration: 60 * time.Second},
			ConfigHardFail:   ptr.To(true),
			ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
			Linux: kubeproxyconfig.KubeProxyLinuxConfiguration{
				Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
					MaxPerCore:            ptr.To[int32](2),
					Min:                   ptr.To[int32](1),
					TCPCloseWaitTimeout:   &metav1.Duration{Duration: 10 * time.Second},
					TCPEstablishedTimeout: &metav1.Duration{Duration: 20 * time.Second},
				},
				MasqueradeAll: false,
				OOMScoreAdj:   ptr.To[int32](17),
			},
			FeatureGates:         map[string]bool{},
			HealthzBindAddresses: tc.healthzBindAddresses,
			HealthzBindPort:      tc.healthzBindPort,
			HostnameOverride:     "foo",
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{

				MasqueradeBit:      ptr.To[int32](12),
				LocalhostNodePorts: ptr.To(false),
			},
			IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
				MasqueradeBit: ptr.To[int32](15),
				ExcludeCIDRs:  []string{"10.20.30.40/16", "fd00:1::0/64"},
			},
			NFTables: kubeproxyconfig.KubeProxyNFTablesConfiguration{
				MasqueradeBit: ptr.To[int32](18),
			},
			MetricsBindAddresses: tc.metricsBindAddresses,
			MetricsBindPort:      tc.metricsBindPort,
			Mode:                 kubeproxyconfig.ProxyMode(tc.mode),
			NodePortAddresses:    []string{"10.20.30.40/16", "fd00:1::0/64"},
			DetectLocalMode:      kubeproxyconfig.LocalModeClusterCIDR,
			DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
				BridgeInterface:     "cbr0",
				ClusterCIDRs:        tc.clusterCIDRs,
				InterfaceNamePrefix: "veth",
			},
			Logging: logsapi.LoggingConfiguration{
				Format:         "text",
				FlushFrequency: logsapi.TimeOrMetaDuration{Duration: metav1.Duration{Duration: 5 * time.Second}, SerializeAsString: true},
			},
		}

		options := NewOptions()

		getYamlRepr := func(list []string) string {
			output := "["
			for i, item := range list {
				output += fmt.Sprintf(`"%s"`, item)
				if i != len(list)-1 {
					output += ","
				}
			}
			output += "]"
			return output
		}

		baseYAML := fmt.Sprintf(
			yamlTemplate,
			getYamlRepr(tc.healthzBindAddresses), tc.healthzBindPort,
			getYamlRepr(tc.metricsBindAddresses), tc.metricsBindPort,
			tc.mode, getYamlRepr(tc.clusterCIDRs),
		)

		// Append additional configuration to the base yaml template
		yaml := fmt.Sprintf("%s\n%s", baseYAML, tc.extraConfig)
		config, err := options.loadConfig([]byte(yaml))
		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)
		if diff := cmp.Diff(config, expected); diff != "" {
			t.Fatalf("unexpected config for %s, diff = %s", tc.name, diff)
		}

	}
}

// TestLoadConfigFailures tests failure modes for loadConfig()
func TestLoadConfigFailures(t *testing.T) {
	yamlTemplate := `apiVersion: kubeproxy.config.k8s.io/%s
kind: %s
mode: iptables
`

	testCases := []struct {
		name                 string
		kind                 string
		extraConfig          string
		version              string
		expErr               string
		expStrictDecodingErr bool
	}{
		{
			name:        "decode error v1alpha1",
			version:     "v1alpha1",
			extraConfig: "Twas bryllyg, and ye slythy toves",
			expErr:      "could not find expected ':'",
		},
		{
			name:        "decode error v1alpha2",
			version:     "v1alpha2",
			extraConfig: "lorem ipsum dolor sit amet",
			expErr:      "could not find expected ':'",
		},
		{
			name:    "bad config type v1alpha1",
			version: "v1alpha1",
			kind:    "KubeSchedulerConfiguration",
			expErr:  "no kind",
		},
		{
			name:    "bad config type v1alpha2",
			version: "v1alpha2",
			kind:    "KubeletConfiguration",
			expErr:  "no kind",
		},
		{
			name:        "missing quotes around :: bindAddress",
			version:     "v1alpha1",
			extraConfig: "bindAddress: ::",
			expErr:      "mapping values are not allowed in this context",
		},
		{
			name:                 "duplicate fields in v1alpha2",
			version:              "v1alpha2",
			extraConfig:          "mode: ipvs",
			expErr:               "key \"mode\" already set in map",
			expStrictDecodingErr: true,
		},
		{
			name:                 "unknown field in v1alpha2",
			version:              "v1alpha2",
			extraConfig:          "foo: bar",
			expErr:               "unknown field \"foo\"",
			expStrictDecodingErr: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()

			kind := "KubeProxyConfiguration"
			if tc.kind != "" {
				kind = tc.kind
			}

			config := fmt.Sprintf(yamlTemplate, tc.version, kind)
			if tc.extraConfig != "" {
				config += "\n" + tc.extraConfig
			}

			_, err := options.loadConfig([]byte(config))
			assert.Error(t, err)
			assert.ErrorContains(t, err, tc.expErr)

			if tc.expStrictDecodingErr {
				assert.True(t, runtime.IsStrictDecodingError(errors.Unwrap(err)))
			}
		})
	}
}

// TestProcessMaintainedFlags tests processing maintained flags.
func TestProcessMaintainedFlags(t *testing.T) {
	testCases := []struct {
		name                   string
		hostnameOverrideFlag   string
		nodeIPOverrideFlag     []string
		expectedHostname       string
		expectedNodeIPOverride []string
		expectError            bool
	}{
		{
			name:             "values from config file",
			expectedHostname: "foo",
			expectError:      false,
		},
		{
			name:                   "values from flags",
			hostnameOverrideFlag:   "  bar ",
			nodeIPOverrideFlag:     []string{"172.16.10.25"},
			expectedHostname:       "bar",
			expectedNodeIPOverride: []string{"172.16.10.25"},
			expectError:            false,
		},
		{
			name:                 "Hostname is space",
			hostnameOverrideFlag: "   ",
			expectError:          true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			options.config = &kubeproxyconfig.KubeProxyConfiguration{
				HostnameOverride: "foo",
			}

			options.hostnameOverride = tc.hostnameOverrideFlag
			options.nodeIPOverride = tc.nodeIPOverrideFlag

			err := options.processMaintainedFlags()
			if tc.expectError {
				if err == nil {
					t.Fatalf("should error for this case %s", tc.name)
				}
			} else {
				assert.NoError(t, err, "unexpected error %v", err)
				if tc.expectedHostname != options.config.HostnameOverride {
					t.Fatalf("expected hostname: %s, but got: %s", tc.expectedHostname, options.config.HostnameOverride)
				}

				if !reflect.DeepEqual(tc.nodeIPOverrideFlag, options.config.NodeIPOverride) {
					t.Fatalf("expected nodeIPOverrid: %s, but got: %s", tc.expectedNodeIPOverride, options.config.NodeIPOverride)
				}

			}
		})
	}
}

// TestProcessMaintainedFlags tests processing maintained flags.
func TestProcessIncompatibleFlags(t *testing.T) {
	testCases := []struct {
		name     string
		flags    []string
		validate func(*kubeproxyconfig.KubeProxyConfiguration) bool
	}{
		{
			name: "iptables configuration",
			flags: []string{
				"--iptables-sync-period=36s",
				"--iptables-min-sync-period=3s",
				"--proxy-mode=iptables",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				syncPeriod := metav1.Duration{Duration: 36 * time.Second}
				minSyncPeriod := metav1.Duration{Duration: 3 * time.Second}
				if config.SyncPeriod == syncPeriod &&
					config.MinSyncPeriod == minSyncPeriod {
					return true
				}
				return false
			},
		},
		{
			name: "ipvs configuration",
			flags: []string{
				"--ipvs-sync-period=16s",
				"--ipvs-min-sync-period=7s",
				"--proxy-mode=ipvs",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				syncPeriod := metav1.Duration{Duration: 16 * time.Second}
				minSyncPeriod := metav1.Duration{Duration: 7 * time.Second}
				if config.SyncPeriod == syncPeriod &&
					config.MinSyncPeriod == minSyncPeriod {
					return true
				}
				return false
			},
		},
		{
			name: "metrics and healthz address ipv4",
			flags: []string{
				"--healthz-bind-address=0.0.0.0:54321",
				"--metrics-bind-address=127.0.0.1:3306",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				if reflect.DeepEqual(config.HealthzBindAddresses, []string{"0.0.0.0/0"}) &&
					reflect.DeepEqual(config.MetricsBindAddresses, []string{"127.0.0.1/32"}) &&
					config.HealthzBindPort == 54321 && config.MetricsBindPort == 3306 {
					return true
				}
				return false
			},
		},
		{
			name: "metrics and healthz address ipv6",
			flags: []string{
				"--healthz-bind-address=[::]:9090",
				"--metrics-bind-address=[::1]:8080",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				if reflect.DeepEqual(config.HealthzBindAddresses, []string{"::/0"}) &&
					reflect.DeepEqual(config.MetricsBindAddresses, []string{"::1/128"}) &&
					config.HealthzBindPort == 9090 && config.MetricsBindPort == 8080 {
					return true
				}
				return false
			},
		},
		{
			name: "bind address",
			flags: []string{
				"--bind-address=0.0.0.0",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				return reflect.DeepEqual(config.NodeIPOverride, []string{"0.0.0.0"})
			},
		},
		{
			name: "cluster cidr",
			flags: []string{
				"--cluster-cidr=2002:0:0:1234::/64,10.0.0.0/14",
			},
			validate: func(config *kubeproxyconfig.KubeProxyConfiguration) bool {
				return reflect.DeepEqual(config.DetectLocal.ClusterCIDRs, []string{"2002:0:0:1234::/64", "10.0.0.0/14"})
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			fs := new(pflag.FlagSet)
			options.AddFlags(fs)
			require.NoError(t, fs.Parse(tc.flags))

			options.processIncompatibleFlags(fs)
			require.True(t, tc.validate(options.config))
		})
	}
}

// TestOptionsComplete checks that command line flags are combined with a
// config properly.
func TestOptionsComplete(t *testing.T) {
	header := `apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
`

	// Determine default config (depends on platform defaults).
	o := NewOptions()
	require.NoError(t, o.Complete(new(pflag.FlagSet)))
	expected := o.config

	config := header + `logging:
  format: json
  flushFrequency: 1s
  verbosity: 10
  vmodule:
  - filePattern: foo.go
    verbosity: 6
  - filePattern: bar.go
    verbosity: 8
`
	expectedLoggingConfig := logsapi.LoggingConfiguration{
		Format:         "json",
		FlushFrequency: logsapi.TimeOrMetaDuration{Duration: metav1.Duration{Duration: time.Second}, SerializeAsString: true},
		Verbosity:      10,
		VModule: []logsapi.VModuleItem{
			{
				FilePattern: "foo.go",
				Verbosity:   6,
			},
			{
				FilePattern: "bar.go",
				Verbosity:   8,
			},
		},
		Options: logsapi.FormatOptions{
			JSON: logsapi.JSONOptions{
				OutputRoutingOptions: logsapi.OutputRoutingOptions{
					InfoBufferSize: resource.QuantityValue{Quantity: resource.MustParse("0")},
				},
			},
			Text: logsapi.TextOptions{
				OutputRoutingOptions: logsapi.OutputRoutingOptions{
					InfoBufferSize: resource.QuantityValue{Quantity: resource.MustParse("0")},
				},
			},
		},
	}

	for name, tc := range map[string]struct {
		config   string
		flags    []string
		expected *kubeproxyconfig.KubeProxyConfiguration
	}{
		"empty": {
			expected: expected,
		},
		"empty-config": {
			config:   header,
			expected: expected,
		},
		"logging-config": {
			config: config,
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				c := expected.DeepCopy()
				c.Logging = *expectedLoggingConfig.DeepCopy()
				return c
			}(),
		},
		"flags": {
			flags: []string{
				"-v=7",
				"--vmodule", "goo.go=8",
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				c := expected.DeepCopy()
				c.Logging.Verbosity = 7
				c.Logging.VModule = append(c.Logging.VModule, logsapi.VModuleItem{
					FilePattern: "goo.go",
					Verbosity:   8,
				})
				return c
			}(),
		},
		"both": {
			config: config,
			flags: []string{
				"-v=7",
				"--vmodule", "goo.go=8",
				"--ipvs-scheduler", "some-scheduler", // Overwritten by config.
			},
			expected: func() *kubeproxyconfig.KubeProxyConfiguration {
				c := expected.DeepCopy()
				c.Logging = *expectedLoggingConfig.DeepCopy()
				// Flag wins.
				c.Logging.Verbosity = 7
				// Flag and config get merged with command line flags first.
				c.Logging.VModule = append([]logsapi.VModuleItem{
					{
						FilePattern: "goo.go",
						Verbosity:   8,
					},
				}, c.Logging.VModule...)
				return c
			}(),
		},
	} {
		t.Run(name, func(t *testing.T) {
			options := NewOptions()
			fs := new(pflag.FlagSet)
			options.AddFlags(fs)
			flags := tc.flags
			if len(tc.config) > 0 {
				tmp := t.TempDir()
				configFile := path.Join(tmp, "kube-proxy.conf")
				require.NoError(t, os.WriteFile(configFile, []byte(tc.config), 0666))
				flags = append(flags, "--config", configFile)
			}
			require.NoError(t, fs.Parse(flags))
			require.NoError(t, options.Complete(fs))
			assert.Equal(t, tc.expected, options.config)
		})
	}
}
