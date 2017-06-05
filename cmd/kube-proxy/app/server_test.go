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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/iptables"
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
		{ // flag says ipvs, but iptable version too low
			flag:            "ipvs",
			iptablesVersion: "0.0.0",
			expected:        proxyModeUserspace,
		},
		{ // flag says ipvs, version ok, kernel not compatible
			flag:            "ipvs",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    false,
			expected:        proxyModeUserspace,
		},
		{ // flag says ipvs, version ok, kernel is compatible
			flag:            "ipvs",
			iptablesVersion: iptables.MinCheckVersion,
			kernelCompat:    true,
			expected:        proxyModeIPVS,
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

// This test verifies that Proxy Server does not crash when CleanupAndExit is true.
func TestProxyServerWithCleanupAndExit(t *testing.T) {
	options, err := NewOptions()
	if err != nil {
		t.Fatal(err)
	}

	options.config = &componentconfig.KubeProxyConfiguration{
		BindAddress: "0.0.0.0",
	}
	options.CleanupAndExit = true

	proxyserver, err := NewProxyServer(options.config, options.CleanupAndExit, options.scheme, options.master)

	assert.Nil(t, err)
	assert.NotNil(t, proxyserver)
	assert.NotNil(t, proxyserver.IptInterface)
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

func TestLoadConfig(t *testing.T) {
	yaml := `apiVersion: componentconfig/v1alpha1
bindAddress: 9.8.7.6
clientConnection:
  acceptContentTypes: "abc"
  burst: 100
  contentType: content-type
  kubeconfig: "/path/to/kubeconfig"
  qps: 7
clusterCIDR: "1.2.3.0/24"
configSyncPeriod: 15s
conntrack:
  max: 4
  maxPerCore: 2
  min: 1
  tcpCloseWaitTimeout: 10s
  tcpEstablishedTimeout: 20s
featureGates: "all"
healthzBindAddress: 1.2.3.4:12345
hostnameOverride: "foo"
iptables:
  masqueradeAll: true
  masqueradeBit: 17
  minSyncPeriod: 10s
  syncPeriod: 60s
kind: KubeProxyConfiguration
metricsBindAddress: 2.3.4.5:23456
mode: "iptables"
oomScoreAdj: 17
portRange: "2-7"
resourceContainer: /foo
udpTimeoutMilliseconds: 123ms
`

	expected := &componentconfig.KubeProxyConfiguration{
		BindAddress: "9.8.7.6",
		ClientConnection: componentconfig.ClientConnectionConfiguration{
			AcceptContentTypes: "abc",
			Burst:              100,
			ContentType:        "content-type",
			KubeConfigFile:     "/path/to/kubeconfig",
			QPS:                7,
		},
		ClusterCIDR:      "1.2.3.0/24",
		ConfigSyncPeriod: metav1.Duration{Duration: 15 * time.Second},
		Conntrack: componentconfig.KubeProxyConntrackConfiguration{
			Max:                   4,
			MaxPerCore:            2,
			Min:                   1,
			TCPCloseWaitTimeout:   metav1.Duration{Duration: 10 * time.Second},
			TCPEstablishedTimeout: metav1.Duration{Duration: 20 * time.Second},
		},
		FeatureGates:       "all",
		HealthzBindAddress: "1.2.3.4:12345",
		HostnameOverride:   "foo",
		IPTables: componentconfig.KubeProxyIPTablesConfiguration{
			MasqueradeAll: true,
			MasqueradeBit: util.Int32Ptr(17),
			MinSyncPeriod: metav1.Duration{Duration: 10 * time.Second},
			SyncPeriod:    metav1.Duration{Duration: 60 * time.Second},
		},
		MetricsBindAddress: "2.3.4.5:23456",
		Mode:               "iptables",
		OOMScoreAdj:        util.Int32Ptr(17),
		PortRange:          "2-7",
		ResourceContainer:  "/foo",
		UDPIdleTimeout:     metav1.Duration{Duration: 123 * time.Millisecond},
	}

	options, err := NewOptions()
	assert.NoError(t, err)

	config, err := options.loadConfig([]byte(yaml))
	assert.NoError(t, err)
	if !reflect.DeepEqual(expected, config) {
		t.Fatalf("unexpected config, diff = %s", diff.ObjectDiff(config, expected))
	}
}
