/*
Copyright 2015 The Kubernetes Authors.

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
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	apimachineryconfig "k8s.io/apimachinery/pkg/apis/config"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	api "k8s.io/kubernetes/pkg/apis/core"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/util/configz"
	utilpointer "k8s.io/utils/pointer"
)

type fakeNodeInterface struct {
	node api.Node
}

func (fake *fakeNodeInterface) Get(hostname string, options metav1.GetOptions) (*api.Node, error) {
	return &fake.node, nil
}

type fakeIPTablesVersioner struct {
	version string // what to return
	err     error  // what to return
}

func (fake *fakeIPTablesVersioner) GetVersion() (string, error) {
	return fake.version, fake.err
}

func (fake *fakeIPTablesVersioner) IsCompatible() error {
	return fake.err
}

type fakeIPSetVersioner struct {
	version string // what to return
	err     error  // what to return
}

func (fake *fakeIPSetVersioner) GetVersion() (string, error) {
	return fake.version, fake.err
}

type fakeKernelCompatTester struct {
	ok bool
}

func (fake *fakeKernelCompatTester) IsCompatible() error {
	if !fake.ok {
		return fmt.Errorf("error")
	}
	return nil
}

// fakeKernelHandler implements KernelHandler.
type fakeKernelHandler struct {
	modules []string
}

func (fake *fakeKernelHandler) GetModules() ([]string, error) {
	return fake.modules, nil
}

// This test verifies that NewProxyServer does not crash when CleanupAndExit is true.
func TestProxyServerWithCleanupAndExit(t *testing.T) {
	// Each bind address below is a separate test case
	bindAddresses := []string{
		"0.0.0.0",
		"::",
	}
	for _, addr := range bindAddresses {
		options := NewOptions()

		options.config = &kubeproxyconfig.KubeProxyConfiguration{
			BindAddress: addr,
		}
		options.CleanupAndExit = true

		proxyserver, err := NewProxyServer(options)

		assert.Nil(t, err, "unexpected error in NewProxyServer, addr: %s", addr)
		assert.NotNil(t, proxyserver, "nil proxy server obj, addr: %s", addr)
		assert.NotNil(t, proxyserver.IptInterface, "nil iptables intf, addr: %s", addr)
		assert.True(t, proxyserver.CleanupAndExit, "false CleanupAndExit, addr: %s", addr)

		// Clean up config for next test case
		configz.Delete(kubeproxyconfig.GroupName)
	}
}

func TestGetConntrackMax(t *testing.T) {
	ncores := runtime.NumCPU()
	testCases := []struct {
		min        int32
		max        int32
		maxPerCore int32
		expected   int
		err        string
	}{
		{
			expected: 0,
		},
		{
			max:      12345,
			expected: 12345,
		},
		{
			max:        12345,
			maxPerCore: 67890,
			expected:   -1,
			err:        "mutually exclusive",
		},
		{
			maxPerCore: 67890, // use this if Max is 0
			min:        1,     // avoid 0 default
			expected:   67890 * ncores,
		},
		{
			maxPerCore: 1, // ensure that Min is considered
			min:        123456,
			expected:   123456,
		},
		{
			maxPerCore: 0, // leave system setting
			min:        123456,
			expected:   0,
		},
	}

	for i, tc := range testCases {
		cfg := kubeproxyconfig.KubeProxyConntrackConfiguration{
			Min:        utilpointer.Int32Ptr(tc.min),
			Max:        utilpointer.Int32Ptr(tc.max),
			MaxPerCore: utilpointer.Int32Ptr(tc.maxPerCore),
		}
		x, e := getConntrackMax(cfg)
		if e != nil {
			if tc.err == "" {
				t.Errorf("[%d] unexpected error: %v", i, e)
			} else if !strings.Contains(e.Error(), tc.err) {
				t.Errorf("[%d] expected an error containing %q: %v", i, tc.err, e)
			}
		} else if x != tc.expected {
			t.Errorf("[%d] expected %d, got %d", i, tc.expected, x)
		}
	}
}

// TestLoadConfig tests proper operation of loadConfig()
func TestLoadConfig(t *testing.T) {

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
  max: 4
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
ipvs:
  minSyncPeriod: 10s
  syncPeriod: 60s
  excludeCIDRs:
    - "10.20.30.40/16"
    - "fd00:1::0/64"
kind: KubeProxyConfiguration
metricsBindAddress: "%s"
mode: "%s"
oomScoreAdj: 17
portRange: "2-7"
resourceContainer: /foo
udpIdleTimeout: 123ms
nodePortAddresses:
  - "10.20.30.40/16"
  - "fd00:1::0/64"
`

	testCases := []struct {
		name               string
		mode               string
		bindAddress        string
		clusterCIDR        string
		healthzBindAddress string
		metricsBindAddress string
	}{
		{
			name:               "iptables mode, IPv4 all-zeros bind address",
			mode:               "iptables",
			bindAddress:        "0.0.0.0",
			clusterCIDR:        "1.2.3.0/24",
			healthzBindAddress: "1.2.3.4:12345",
			metricsBindAddress: "2.3.4.5:23456",
		},
		{
			name:               "iptables mode, non-zeros IPv4 config",
			mode:               "iptables",
			bindAddress:        "9.8.7.6",
			clusterCIDR:        "1.2.3.0/24",
			healthzBindAddress: "1.2.3.4:12345",
			metricsBindAddress: "2.3.4.5:23456",
		},
		{
			// Test for 'bindAddress: "::"' (IPv6 all-zeros) in kube-proxy
			// config file. The user will need to put quotes around '::' since
			// 'bindAddress: ::' is invalid yaml syntax.
			name:               "iptables mode, IPv6 \"::\" bind address",
			mode:               "iptables",
			bindAddress:        "\"::\"",
			clusterCIDR:        "fd00:1::0/64",
			healthzBindAddress: "[fd00:1::5]:12345",
			metricsBindAddress: "[fd00:2::5]:23456",
		},
		{
			// Test for 'bindAddress: "[::]"' (IPv6 all-zeros in brackets)
			// in kube-proxy config file. The user will need to use
			// surrounding quotes here since 'bindAddress: [::]' is invalid
			// yaml syntax.
			name:               "iptables mode, IPv6 \"[::]\" bind address",
			mode:               "iptables",
			bindAddress:        "\"[::]\"",
			clusterCIDR:        "fd00:1::0/64",
			healthzBindAddress: "[fd00:1::5]:12345",
			metricsBindAddress: "[fd00:2::5]:23456",
		},
		{
			// Test for 'bindAddress: ::0' (another form of IPv6 all-zeros).
			// No surrounding quotes are required around '::0'.
			name:               "iptables mode, IPv6 ::0 bind address",
			mode:               "iptables",
			bindAddress:        "::0",
			clusterCIDR:        "fd00:1::0/64",
			healthzBindAddress: "[fd00:1::5]:12345",
			metricsBindAddress: "[fd00:2::5]:23456",
		},
		{
			name:               "ipvs mode, IPv6 config",
			mode:               "ipvs",
			bindAddress:        "2001:db8::1",
			clusterCIDR:        "fd00:1::0/64",
			healthzBindAddress: "[fd00:1::5]:12345",
			metricsBindAddress: "[fd00:2::5]:23456",
		},
	}

	for _, tc := range testCases {
		expBindAddr := tc.bindAddress
		if tc.bindAddress[0] == '"' {
			// Surrounding double quotes will get stripped by the yaml parser.
			expBindAddr = expBindAddr[1 : len(tc.bindAddress)-1]
		}
		expected := &kubeproxyconfig.KubeProxyConfiguration{
			BindAddress: expBindAddr,
			ClientConnection: apimachineryconfig.ClientConnectionConfiguration{
				AcceptContentTypes: "abc",
				Burst:              100,
				ContentType:        "content-type",
				Kubeconfig:         "/path/to/kubeconfig",
				QPS:                7,
			},
			ClusterCIDR:      tc.clusterCIDR,
			ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
				Max:                   utilpointer.Int32Ptr(4),
				MaxPerCore:            utilpointer.Int32Ptr(2),
				Min:                   utilpointer.Int32Ptr(1),
				TCPCloseWaitTimeout:   &metav1.Duration{Duration: 10 * time.Second},
				TCPEstablishedTimeout: &metav1.Duration{Duration: 20 * time.Second},
			},
			FeatureGates:       map[string]bool{},
			HealthzBindAddress: tc.healthzBindAddress,
			HostnameOverride:   "foo",
			IPTables: kubeproxyconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				MasqueradeBit: utilpointer.Int32Ptr(17),
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
				SyncPeriod:    metav1.Duration{Duration: 60 * time.Second},
			},
			IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
				SyncPeriod:    metav1.Duration{Duration: 60 * time.Second},
				ExcludeCIDRs:  []string{"10.20.30.40/16", "fd00:1::0/64"},
			},
			MetricsBindAddress: tc.metricsBindAddress,
			Mode:               kubeproxyconfig.ProxyMode(tc.mode),
			OOMScoreAdj:        utilpointer.Int32Ptr(17),
			PortRange:          "2-7",
			ResourceContainer:  "/foo",
			UDPIdleTimeout:     metav1.Duration{Duration: 123 * time.Millisecond},
			NodePortAddresses:  []string{"10.20.30.40/16", "fd00:1::0/64"},
		}

		options := NewOptions()

		yaml := fmt.Sprintf(
			yamlTemplate, tc.bindAddress, tc.clusterCIDR,
			tc.healthzBindAddress, tc.metricsBindAddress, tc.mode)
		config, err := options.loadConfig([]byte(yaml))
		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)
		if !reflect.DeepEqual(expected, config) {
			t.Fatalf("unexpected config for %s, diff = %s", tc.name, diff.ObjectDiff(config, expected))
		}
	}
}

// TestLoadConfigFailures tests failure modes for loadConfig()
func TestLoadConfigFailures(t *testing.T) {
	testCases := []struct {
		name   string
		config string
		expErr string
	}{
		{
			name:   "Decode error test",
			config: "Twas bryllyg, and ye slythy toves",
			expErr: "could not find expected ':'",
		},
		{
			name:   "Bad config type test",
			config: "kind: KubeSchedulerConfiguration",
			expErr: "no kind",
		},
		{
			name:   "Missing quotes around :: bindAddress",
			config: "bindAddress: ::",
			expErr: "mapping values are not allowed in this context",
		},
	}
	version := "apiVersion: kubeproxy.config.k8s.io/v1alpha1"
	for _, tc := range testCases {
		options := NewOptions()
		config := fmt.Sprintf("%s\n%s", version, tc.config)
		_, err := options.loadConfig([]byte(config))
		if assert.Error(t, err, tc.name) {
			assert.Contains(t, err.Error(), tc.expErr, tc.name)
		}
	}
}

// TestProcessHostnameOverrideFlag tests processing hostname-override arg
func TestProcessHostnameOverrideFlag(t *testing.T) {
	testCases := []struct {
		name                 string
		hostnameOverrideFlag string
		expectedHostname     string
	}{
		{
			name:                 "Hostname from config file",
			hostnameOverrideFlag: "",
			expectedHostname:     "foo",
		},
		{
			name:                 "Hostname from flag",
			hostnameOverrideFlag: "  bar ",
			expectedHostname:     "bar",
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			options.config = &kubeproxyconfig.KubeProxyConfiguration{
				HostnameOverride: "foo",
			}

			options.hostnameOverride = tc.hostnameOverrideFlag

			err := options.processHostnameOverrideFlag()
			assert.NoError(t, err, "unexpected error %v", err)
			if tc.expectedHostname != options.config.HostnameOverride {
				t.Fatalf("expected hostname: %s, but got: %s", tc.expectedHostname, options.config.HostnameOverride)
			}
		})
	}
}
