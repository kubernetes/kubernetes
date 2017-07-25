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
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sRuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/util/configz"
	"k8s.io/kubernetes/pkg/util/iptables"
	utilpointer "k8s.io/kubernetes/pkg/util/pointer"
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

type fakeKernelCompatTester struct {
	ok bool
}

func (fake *fakeKernelCompatTester) IsCompatible() error {
	if !fake.ok {
		return fmt.Errorf("error")
	}
	return nil
}

func Test_getProxyMode(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("skipping on non-Linux")
	}
	var cases = []struct {
		flag            string
		annotationKey   string
		annotationVal   string
		iptablesVersion string
		kernelCompat    bool
		iptablesError   error
		expected        string
	}{
		{ // flag says userspace
			flag:     "userspace",
			expected: proxyModeUserspace,
		},
		{ // flag says iptables, error detecting version
			flag:          "iptables",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // flag says iptables, version too low
			flag:            "iptables",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel not compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // flag says iptables, version ok, kernel is compatible
			flag:            "iptables",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
		{ // detect, error
			flag:          "",
			iptablesError: fmt.Errorf("oops!"),
			expected:      proxyModeUserspace,
		},
		{ // detect, version too low
			flag:            "",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel not compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // detect, version ok, kernel is compatible
			flag:            "",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPTables,
		},
	}
	for i, c := range cases {
		versioner := &fakeIPTablesVersioner{c.iptablesVersion, c.iptablesError}
		kcompater := &fakeKernelCompatTester{c.kernelCompat}
		r := getProxyMode(c.flag, versioner, kcompater)
		if r != c.expected {
			t.Errorf("Case[%d] Expected %q, got %q", i, c.expected, r)
		}
	}
}

// TestNewOptionsFailures tests failure modes for NewOptions()
func TestNewOptionsFailures(t *testing.T) {

	// Create a fake scheme builder that generates an error
	errString := fmt.Sprintf("Simulated error")
	genError := func(scheme *k8sRuntime.Scheme) error {
		return errors.New(errString)
	}
	fakeSchemeBuilder := k8sRuntime.NewSchemeBuilder(genError)

	simulatedErrorTest := func(target string) {
		var addToScheme *func(s *k8sRuntime.Scheme) error
		if target == "componentconfig" {
			addToScheme = &componentconfig.AddToScheme
		} else {
			addToScheme = &v1alpha1.AddToScheme
		}
		restoreValue := *addToScheme
		restore := func() {
			*addToScheme = restoreValue
		}
		defer restore()
		*addToScheme = fakeSchemeBuilder.AddToScheme
		_, err := NewOptions()
		assert.Error(t, err, fmt.Sprintf("Simulated error in component %s", target))
	}

	// Simulate errors in calls to AddToScheme()
	faultTargets := []string{"componentconfig", "v1alpha1"}
	for _, target := range faultTargets {
		simulatedErrorTest(target)
	}
}

// This test verifies that NewProxyServer does not crash when CleanupAndExit is true.
func TestProxyServerWithCleanupAndExit(t *testing.T) {
	// Each bind address below is a separate test case
	bindAddresses := []string{
		"0.0.0.0",
		"2001:db8::1",
	}
	for _, addr := range bindAddresses {
		options, err := NewOptions()
		if err != nil {
			t.Fatalf("Unexpected error with address %s: %v", addr, err)
		}

		options.config = &componentconfig.KubeProxyConfiguration{
			BindAddress: addr,
		}
		options.CleanupAndExit = true

		proxyserver, err := NewProxyServer(options.config, options.CleanupAndExit, options.scheme, options.master)

		assert.Nil(t, err, "unexpected error in NewProxyServer, addr: %s", addr)
		assert.NotNil(t, proxyserver, "nil proxy server obj, addr: %s", addr)
		assert.NotNil(t, proxyserver.IptInterface, "nil iptables intf, addr: %s", addr)
		assert.True(t, proxyserver.CleanupAndExit, "false CleanupAndExit, addr: %s", addr)

		// Clean up config for next test case
		configz.Delete("componentconfig")
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
		cfg := componentconfig.KubeProxyConntrackConfiguration{
			Min:        tc.min,
			Max:        tc.max,
			MaxPerCore: tc.maxPerCore,
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

	yamlTemplate := `apiVersion: componentconfig/v1alpha1
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
featureGates: "all"
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
kind: KubeProxyConfiguration
metricsBindAddress: "%s"
mode: "iptables"
oomScoreAdj: 17
portRange: "2-7"
resourceContainer: /foo
udpTimeoutMilliseconds: 123ms
`

	testCases := []struct {
		name               string
		bindAddress        string
		clusterCIDR        string
		healthzBindAddress string
		metricsBindAddress string
	}{
		{
			name:               "IPv4 config",
			bindAddress:        "9.8.7.6",
			clusterCIDR:        "1.2.3.0/24",
			healthzBindAddress: "1.2.3.4:12345",
			metricsBindAddress: "2.3.4.5:23456",
		},
		{
			name:               "IPv6 config",
			bindAddress:        "2001:db8::1",
			clusterCIDR:        "fd00:1::0/64",
			healthzBindAddress: "[fd00:1::5]:12345",
			metricsBindAddress: "[fd00:2::5]:23456",
		},
	}

	for _, tc := range testCases {
		expected := &componentconfig.KubeProxyConfiguration{
			BindAddress: tc.bindAddress,
			ClientConnection: componentconfig.ClientConnectionConfiguration{
				AcceptContentTypes: "abc",
				Burst:              100,
				ContentType:        "content-type",
				KubeConfigFile:     "/path/to/kubeconfig",
				QPS:                7,
			},
			ClusterCIDR:      tc.clusterCIDR,
			ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
			Conntrack: componentconfig.KubeProxyConntrackConfiguration{
				Max:                   4,
				MaxPerCore:            2,
				Min:                   1,
				TCPCloseWaitTimeout:   metav1.Duration{Duration: 10 * time.Second},
				TCPEstablishedTimeout: metav1.Duration{Duration: 20 * time.Second},
			},
			FeatureGates:       "all",
			HealthzBindAddress: tc.healthzBindAddress,
			HostnameOverride:   "foo",
			IPTables: componentconfig.KubeProxyIPTablesConfiguration{
				MasqueradeAll: true,
				MasqueradeBit: utilpointer.Int32Ptr(17),
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
				SyncPeriod:    metav1.Duration{Duration: 60 * time.Second},
			},
			IPVS: componentconfig.KubeProxyIPVSConfiguration{
				MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
				SyncPeriod:    metav1.Duration{Duration: 60 * time.Second},
			},
			MetricsBindAddress: tc.metricsBindAddress,
			Mode:               "iptables",
			// TODO: IPVS
			OOMScoreAdj:       utilpointer.Int32Ptr(17),
			PortRange:         "2-7",
			ResourceContainer: "/foo",
			UDPIdleTimeout:    metav1.Duration{Duration: 123 * time.Millisecond},
		}

		options, err := NewOptions()
		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)

		yaml := fmt.Sprintf(
			yamlTemplate, tc.bindAddress, tc.clusterCIDR,
			tc.healthzBindAddress, tc.metricsBindAddress)
		config, err := options.loadConfig([]byte(yaml))
		assert.NoError(t, err, "unexpected error for %s: %v", tc.name, err)
		if !reflect.DeepEqual(expected, config) {
			t.Fatalf("unexpected config for %s test, diff = %s", tc.name, diff.ObjectDiff(config, expected))
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
			expErr: "unexpected config type",
		},
	}
	version := "apiVersion: componentconfig/v1alpha1"
	for _, tc := range testCases {
		options, _ := NewOptions()
		config := fmt.Sprintf("%s\n%s", version, tc.config)
		_, err := options.loadConfig([]byte(config))
		if assert.Error(t, err, tc.name) {
			assert.Contains(t, err.Error(), tc.expErr, tc.name)
		}
	}
}
