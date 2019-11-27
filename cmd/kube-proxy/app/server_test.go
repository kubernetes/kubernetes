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
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	utilpointer "k8s.io/utils/pointer"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	componentbaseconfig "k8s.io/component-base/config"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/pkg/util/configz"
)

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

		// Clean up config for next test case
		configz.Delete(kubeproxyconfig.GroupName)
	}
}

func TestGetConntrackMax(t *testing.T) {
	ncores := runtime.NumCPU()
	testCases := []struct {
		min        int32
		maxPerCore int32
		expected   int
		err        string
	}{
		{
			expected: 0,
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
		extraConfig        string
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
		{
			// Test for unknown field within config.
			// For v1alpha1 a lenient path is implemented and will throw a
			// strict decoding warning instead of failing to load
			name:               "unknown field",
			mode:               "iptables",
			bindAddress:        "9.8.7.6",
			clusterCIDR:        "1.2.3.0/24",
			healthzBindAddress: "1.2.3.4:12345",
			metricsBindAddress: "2.3.4.5:23456",
			extraConfig:        "foo: bar",
		},
		{
			// Test for duplicate field within config.
			// For v1alpha1 a lenient path is implemented and will throw a
			// strict decoding warning instead of failing to load
			name:               "duplicate field",
			mode:               "iptables",
			bindAddress:        "9.8.7.6",
			clusterCIDR:        "1.2.3.0/24",
			healthzBindAddress: "1.2.3.4:12345",
			metricsBindAddress: "2.3.4.5:23456",
			extraConfig:        "bindAddress: 9.8.7.6",
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
			ClientConnection: componentbaseconfig.ClientConnectionConfiguration{
				AcceptContentTypes: "abc",
				Burst:              100,
				ContentType:        "content-type",
				Kubeconfig:         "/path/to/kubeconfig",
				QPS:                7,
			},
			ClusterCIDR:      tc.clusterCIDR,
			ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
			Conntrack: kubeproxyconfig.KubeProxyConntrackConfiguration{
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
			UDPIdleTimeout:     metav1.Duration{Duration: 123 * time.Millisecond},
			NodePortAddresses:  []string{"10.20.30.40/16", "fd00:1::0/64"},
		}

		options := NewOptions()

		baseYAML := fmt.Sprintf(
			yamlTemplate, tc.bindAddress, tc.clusterCIDR,
			tc.healthzBindAddress, tc.metricsBindAddress, tc.mode)

		// Append additional configuration to the base yaml template
		yaml := fmt.Sprintf("%s\n%s", baseYAML, tc.extraConfig)

		config, err := options.loadConfig([]byte(yaml))

		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)

		if !reflect.DeepEqual(expected, config) {
			t.Fatalf("unexpected config for %s, diff = %s", tc.name, diff.ObjectDiff(config, expected))
		}
	}
}

// TestLoadConfigFailures tests failure modes for loadConfig()
func TestLoadConfigFailures(t *testing.T) {
	// TODO(phenixblue): Uncomment below template when v1alpha2+ of kube-proxy config is
	// released with strict decoding. These associated tests will fail with
	// the lenient codec and only one config API version.
	/*
			yamlTemplate := `bindAddress: 0.0.0.0
		clusterCIDR: "1.2.3.0/24"
		configSyncPeriod: 15s
		kind: KubeProxyConfiguration`
	*/

	testCases := []struct {
		name    string
		config  string
		expErr  string
		checkFn func(err error) bool
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
		// TODO(phenixblue): Uncomment below tests when v1alpha2+ of kube-proxy config is
		// released with strict decoding. These tests will fail with the
		// lenient codec and only one config API version.
		/*
			{
				name:    "Duplicate fields",
				config:  fmt.Sprintf("%s\nbindAddress: 1.2.3.4", yamlTemplate),
				checkFn: kuberuntime.IsStrictDecodingError,
			},
			{
				name:    "Unknown field",
				config:  fmt.Sprintf("%s\nfoo: bar", yamlTemplate),
				checkFn: kuberuntime.IsStrictDecodingError,
			},
		*/
	}

	version := "apiVersion: kubeproxy.config.k8s.io/v1alpha1"
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			config := fmt.Sprintf("%s\n%s", version, tc.config)
			_, err := options.loadConfig([]byte(config))

			if assert.Error(t, err, tc.name) {
				if tc.expErr != "" {
					assert.Contains(t, err.Error(), tc.expErr)
				}
				if tc.checkFn != nil {
					assert.True(t, tc.checkFn(err), tc.name)
				}
			}
		})
	}
}

// TestProcessHostnameOverrideFlag tests processing hostname-override arg
func TestProcessHostnameOverrideFlag(t *testing.T) {
	testCases := []struct {
		name                 string
		hostnameOverrideFlag string
		expectedHostname     string
		expectError          bool
	}{
		{
			name:                 "Hostname from config file",
			hostnameOverrideFlag: "",
			expectedHostname:     "foo",
			expectError:          false,
		},
		{
			name:                 "Hostname from flag",
			hostnameOverrideFlag: "  bar ",
			expectedHostname:     "bar",
			expectError:          false,
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

			err := options.processHostnameOverrideFlag()
			if tc.expectError {
				if err == nil {
					t.Fatalf("should error for this case %s", tc.name)
				}
			} else {
				assert.NoError(t, err, "unexpected error %v", err)
				if tc.expectedHostname != options.config.HostnameOverride {
					t.Fatalf("expected hostname: %s, but got: %s", tc.expectedHostname, options.config.HostnameOverride)
				}
			}
		})
	}
}

func TestConfigChange(t *testing.T) {
	setUp := func() (*os.File, string, error) {
		tempDir, err := ioutil.TempDir("", "kubeproxy-config-change")
		if err != nil {
			return nil, "", fmt.Errorf("unable to create temporary directory: %v", err)
		}
		fullPath := filepath.Join(tempDir, "kube-proxy-config")
		file, err := os.Create(fullPath)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when creating temp file: %v", err)
		}

		_, err = file.WriteString(`apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
clientConnection:
  acceptContentTypes: ""
  burst: 10
  contentType: application/vnd.kubernetes.protobuf
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
  qps: 5
clusterCIDR: 10.244.0.0/16
configSyncPeriod: 15m0s
conntrack:
  maxPerCore: 32768
  min: 131072
  tcpCloseWaitTimeout: 1h0m0s
  tcpEstablishedTimeout: 24h0m0s
enableProfiling: false
healthzBindAddress: 0.0.0.0:10256
hostnameOverride: ""
iptables:
  masqueradeAll: false
  masqueradeBit: 14
  minSyncPeriod: 0s
  syncPeriod: 30s
ipvs:
  excludeCIDRs: null
  minSyncPeriod: 0s
  scheduler: ""
  syncPeriod: 30s
kind: KubeProxyConfiguration
metricsBindAddress: 127.0.0.1:10249
mode: ""
nodePortAddresses: null
oomScoreAdj: -999
portRange: ""
udpIdleTimeout: 250ms`)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when writing content to temp kube-proxy config file: %v", err)
		}

		return file, tempDir, nil
	}

	tearDown := func(file *os.File, tempDir string) {
		file.Close()
		os.RemoveAll(tempDir)
	}

	testCases := []struct {
		name        string
		proxyServer proxyRun
		append      bool
		expectedErr string
	}{
		{
			name:        "update config file",
			proxyServer: new(fakeProxyServerLongRun),
			append:      true,
			expectedErr: "content of the proxy server's configuration file was updated",
		},
		{
			name:        "fake error",
			proxyServer: new(fakeProxyServerError),
			expectedErr: "mocking error from ProxyServer.Run()",
		},
	}

	for _, tc := range testCases {
		file, tempDir, err := setUp()
		if err != nil {
			t.Fatalf("unexpected error when setting up environment: %v", err)
		}

		opt := NewOptions()
		opt.ConfigFile = file.Name()
		err = opt.Complete()
		if err != nil {
			t.Fatal(err)
		}
		opt.proxyServer = tc.proxyServer

		errCh := make(chan error)
		go func() {
			errCh <- opt.runLoop()
		}()

		if tc.append {
			file.WriteString("append fake content")
		}

		select {
		case err := <-errCh:
			if err != nil {
				if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("[%s] Expected error containing %v, got %v", tc.name, tc.expectedErr, err)
				}
			}
		case <-time.After(10 * time.Second):
			t.Errorf("[%s] Timeout: unable to get any events or internal timeout.", tc.name)
		}
		tearDown(file, tempDir)
	}
}

type fakeProxyServerLongRun struct{}

// Run runs the specified ProxyServer.
func (s *fakeProxyServerLongRun) Run() error {
	for {
		time.Sleep(2 * time.Second)
	}
}

// CleanupAndExit runs in the specified ProxyServer.
func (s *fakeProxyServerLongRun) CleanupAndExit() error {
	return nil
}

type fakeProxyServerError struct{}

// Run runs the specified ProxyServer.
func (s *fakeProxyServerError) Run() error {
	for {
		time.Sleep(2 * time.Second)
		return fmt.Errorf("mocking error from ProxyServer.Run()")
	}
}

// CleanupAndExit runs in the specified ProxyServer.
func (s *fakeProxyServerError) CleanupAndExit() error {
	return errors.New("mocking error from ProxyServer.CleanupAndExit()")
}

func TestAddressFromDeprecatedFlags(t *testing.T) {
	testCases := []struct {
		name               string
		healthzPort        int32
		healthzBindAddress string
		metricsPort        int32
		metricsBindAddress string
		expHealthz         string
		expMetrics         string
	}{
		{
			name:               "IPv4 bind address",
			healthzBindAddress: "1.2.3.4",
			healthzPort:        12345,
			metricsBindAddress: "2.3.4.5",
			metricsPort:        23456,
			expHealthz:         "1.2.3.4:12345",
			expMetrics:         "2.3.4.5:23456",
		},
		{
			name:               "IPv4 bind address has port",
			healthzBindAddress: "1.2.3.4:12345",
			healthzPort:        23456,
			metricsBindAddress: "2.3.4.5:12345",
			metricsPort:        23456,
			expHealthz:         "1.2.3.4:12345",
			expMetrics:         "2.3.4.5:12345",
		},
		{
			name:               "IPv6 bind address",
			healthzBindAddress: "fd00:1::5",
			healthzPort:        12345,
			metricsBindAddress: "fd00:1::6",
			metricsPort:        23456,
			expHealthz:         "[fd00:1::5]:12345",
			expMetrics:         "[fd00:1::6]:23456",
		},
		{
			name:               "IPv6 bind address has port",
			healthzBindAddress: "[fd00:1::5]:12345",
			healthzPort:        56789,
			metricsBindAddress: "[fd00:1::6]:56789",
			metricsPort:        12345,
			expHealthz:         "[fd00:1::5]:12345",
			expMetrics:         "[fd00:1::6]:56789",
		},
		{
			name:               "Invalid IPv6 Config",
			healthzBindAddress: "[fd00:1::5]",
			healthzPort:        12345,
			metricsBindAddress: "[fd00:1::6]",
			metricsPort:        56789,
			expHealthz:         "[fd00:1::5]",
			expMetrics:         "[fd00:1::6]",
		},
	}

	for i := range testCases {
		gotHealthz := addressFromDeprecatedFlags(testCases[i].healthzBindAddress, testCases[i].healthzPort)
		gotMetrics := addressFromDeprecatedFlags(testCases[i].metricsBindAddress, testCases[i].metricsPort)

		errFn := func(name, except, got string) {
			t.Errorf("case %s: expected %v, got %v", name, except, got)
		}

		if gotHealthz != testCases[i].expHealthz {
			errFn(testCases[i].name, testCases[i].expHealthz, gotHealthz)
		}

		if gotMetrics != testCases[i].expMetrics {
			errFn(testCases[i].name, testCases[i].expMetrics, gotMetrics)
		}

	}
}
