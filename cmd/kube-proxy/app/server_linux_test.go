//go:build linux
// +build linux

/*
Copyright 2018 The Kubernetes Authors.

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
	"net"
	"os"
	"path/filepath"
	"reflect"
	goruntime "runtime"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientgotesting "k8s.io/client-go/testing"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyutiliptables "k8s.io/kubernetes/pkg/proxy/util/iptables"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
	"k8s.io/utils/ptr"
)

func Test_platformApplyDefaults(t *testing.T) {
	testCases := []struct {
		name                string
		mode                proxyconfigapi.ProxyMode
		expectedMode        proxyconfigapi.ProxyMode
		detectLocal         proxyconfigapi.LocalMode
		expectedDetectLocal proxyconfigapi.LocalMode
	}{
		{
			name:                "defaults",
			mode:                "",
			expectedMode:        proxyconfigapi.ProxyModeIPTables,
			detectLocal:         "",
			expectedDetectLocal: proxyconfigapi.LocalModeClusterCIDR,
		},
		{
			name:                "explicit",
			mode:                proxyconfigapi.ProxyModeIPTables,
			expectedMode:        proxyconfigapi.ProxyModeIPTables,
			detectLocal:         proxyconfigapi.LocalModeClusterCIDR,
			expectedDetectLocal: proxyconfigapi.LocalModeClusterCIDR,
		},
		{
			name:                "override mode",
			mode:                "ipvs",
			expectedMode:        proxyconfigapi.ProxyModeIPVS,
			detectLocal:         "",
			expectedDetectLocal: proxyconfigapi.LocalModeClusterCIDR,
		},
		{
			name:                "override detect-local",
			mode:                "",
			expectedMode:        proxyconfigapi.ProxyModeIPTables,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: proxyconfigapi.LocalModeNodeCIDR,
		},
		{
			name:                "override both",
			mode:                "ipvs",
			expectedMode:        proxyconfigapi.ProxyModeIPVS,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: proxyconfigapi.LocalModeNodeCIDR,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			config := &proxyconfigapi.KubeProxyConfiguration{
				Mode:            tc.mode,
				DetectLocalMode: tc.detectLocal,
			}

			options.platformApplyDefaults(config)
			if config.Mode != tc.expectedMode {
				t.Fatalf("expected mode: %s, but got: %s", tc.expectedMode, config.Mode)
			}
			if config.DetectLocalMode != tc.expectedDetectLocal {
				t.Fatalf("expected detect-local: %s, but got: %s", tc.expectedDetectLocal, config.DetectLocalMode)
			}
		})
	}
}

func Test_getLocalDetector(t *testing.T) {
	cases := []struct {
		name         string
		mode         proxyconfigapi.LocalMode
		config       *proxyconfigapi.KubeProxyConfiguration
		family       v1.IPFamily
		expected     proxyutiliptables.LocalTrafficDetector
		nodePodCIDRs []string
		errExpected  bool
	}{
		// LocalModeClusterCIDR
		{
			name:        "LocalModeClusterCIDR, IPv4 cluster",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14")),
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, IPv6 cluster",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			family:      v1.IPv6Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002:0:0:1234::/64")),
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, IPv6 cluster with IPv6 config",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			family:      v1.IPv6Protocol,
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, IPv4 cluster with IPv6 config",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			family:      v1.IPv4Protocol,
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, IPv4 kube-proxy in dual-stack IPv6-primary cluster",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64,10.0.0.0/14"},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14")),
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, no ClusterCIDR",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			family:      v1.IPv4Protocol,
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		// LocalModeNodeCIDR
		{
			name:         "LocalModeNodeCIDR, IPv4 cluster",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			family:       v1.IPv4Protocol,
			expected:     resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24")),
			nodePodCIDRs: []string{"10.0.0.0/24"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, IPv6 cluster",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			family:       v1.IPv6Protocol,
			expected:     resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96")),
			nodePodCIDRs: []string{"2002::1234:abcd:ffff:0:0/96"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, IPv6 cluster with IPv4 config",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			family:       v1.IPv6Protocol,
			expected:     proxyutiliptables.NewNoOpLocalDetector(),
			nodePodCIDRs: []string{"10.0.0.0/24"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, IPv4 cluster with IPv6 config",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			family:       v1.IPv4Protocol,
			expected:     proxyutiliptables.NewNoOpLocalDetector(),
			nodePodCIDRs: []string{"2002::1234:abcd:ffff:0:0/96"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, IPv6 kube-proxy in dual-stack IPv4-primary cluster",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14,2002:0:0:1234::/64"},
			family:       v1.IPv6Protocol,
			expected:     resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96")),
			nodePodCIDRs: []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, no PodCIDRs",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			family:       v1.IPv4Protocol,
			expected:     proxyutiliptables.NewNoOpLocalDetector(),
			nodePodCIDRs: []string{},
			errExpected:  false,
		},
		// unknown mode
		{
			name:        "unknown LocalMode",
			mode:        proxyconfigapi.LocalMode("abcd"),
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			family:      v1.IPv4Protocol,
			expected:    proxyutiliptables.NewNoOpLocalDetector(),
			errExpected: false,
		},
		// LocalModeBridgeInterface
		{
			name: "LocalModeBrideInterface",
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "eth"},
			},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByBridgeInterface("eth")),
			errExpected: false,
		},
		{
			name: "LocalModeBridgeInterface, strange bridge name",
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "1234567890123456789"},
			},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByBridgeInterface("1234567890123456789")),
			errExpected: false,
		},
		// LocalModeInterfaceNamePrefix
		{
			name: "LocalModeInterfaceNamePrefix",
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "eth"},
			},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("eth")),
			errExpected: false,
		},
		{
			name: "LocalModeInterfaceNamePrefix, strange interface name",
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "1234567890123456789"},
			},
			family:      v1.IPv4Protocol,
			expected:    resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("1234567890123456789")),
			errExpected: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			r, err := getLocalDetector(logger, c.family, c.mode, c.config, c.nodePodCIDRs)
			if c.errExpected {
				if err == nil {
					t.Errorf("Expected error, but succeeded with %v", r)
				}
				return
			}
			if err != nil {
				t.Errorf("Error resolving detect-local: %v", err)
				return
			}
			if !reflect.DeepEqual(r, c.expected) {
				t.Errorf("Unexpected detect-local implementation, expected: %q, got: %q", c.expected, r)
			}
		})
	}
}

func Test_getDualStackLocalDetectorTuple(t *testing.T) {
	cases := []struct {
		name         string
		mode         proxyconfigapi.LocalMode
		config       *proxyconfigapi.KubeProxyConfiguration
		expected     [2]proxyutiliptables.LocalTrafficDetector
		nodePodCIDRs []string
		errExpected  bool
	}{
		// LocalModeClusterCIDR
		{
			name:   "LocalModeClusterCIDR, dual-stack IPv4-primary cluster",
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14,2002:0:0:1234::/64"},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14"))(
				proxyutiliptables.NewDetectLocalByCIDR("2002:0:0:1234::/64")),
			errExpected: false,
		},
		{
			name:   "LocalModeClusterCIDR, dual-stack IPv6-primary cluster",
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64,10.0.0.0/14"},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14"))(
				proxyutiliptables.NewDetectLocalByCIDR("2002:0:0:1234::/64")),
			errExpected: false,
		},
		{
			name:   "LocalModeClusterCIDR, single-stack IPv4 cluster",
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/14")),
				proxyutiliptables.NewNoOpLocalDetector()},
			errExpected: false,
		},
		{
			name:   "LocalModeClusterCIDR, single-stack IPv6 cluster",
			mode:   proxyconfigapi.LocalModeClusterCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				proxyutiliptables.NewNoOpLocalDetector(),
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002:0:0:1234::/64"))},
			errExpected: false,
		},
		{
			name:        "LocalModeClusterCIDR, no ClusterCIDR",
			mode:        proxyconfigapi.LocalModeClusterCIDR,
			config:      &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			expected:    [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			errExpected: false,
		},
		// LocalModeNodeCIDR
		{
			name:   "LocalModeNodeCIDR, dual-stack IPv4-primary cluster",
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14,2002:0:0:1234::/64"},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24"))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96")),
			nodePodCIDRs: []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			errExpected:  false,
		},
		{
			name:   "LocalModeNodeCIDR, dual-stack IPv6-primary cluster",
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64,10.0.0.0/14"},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24"))(
				proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96")),
			nodePodCIDRs: []string{"2002::1234:abcd:ffff:0:0/96", "10.0.0.0/24"},
			errExpected:  false,
		},
		{
			name:   "LocalModeNodeCIDR, single-stack IPv4 cluster",
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "10.0.0.0/14"},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("10.0.0.0/24")),
				proxyutiliptables.NewNoOpLocalDetector()},
			nodePodCIDRs: []string{"10.0.0.0/24"},
			errExpected:  false,
		},
		{
			name:   "LocalModeNodeCIDR, single-stack IPv6 cluster",
			mode:   proxyconfigapi.LocalModeNodeCIDR,
			config: &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: "2002:0:0:1234::/64"},
			expected: [2]proxyutiliptables.LocalTrafficDetector{
				proxyutiliptables.NewNoOpLocalDetector(),
				resolveLocalDetector(t)(proxyutiliptables.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"))},
			nodePodCIDRs: []string{"2002::1234:abcd:ffff:0:0/96"},
			errExpected:  false,
		},
		{
			name:         "LocalModeNodeCIDR, no PodCIDRs",
			mode:         proxyconfigapi.LocalModeNodeCIDR,
			config:       &proxyconfigapi.KubeProxyConfiguration{ClusterCIDR: ""},
			expected:     [2]proxyutiliptables.LocalTrafficDetector{proxyutiliptables.NewNoOpLocalDetector(), proxyutiliptables.NewNoOpLocalDetector()},
			nodePodCIDRs: []string{},
			errExpected:  false,
		},
		// LocalModeBridgeInterface
		{
			name: "LocalModeBridgeInterface",
			mode: proxyconfigapi.LocalModeBridgeInterface,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{BridgeInterface: "eth"},
			},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByBridgeInterface("eth"))(
				proxyutiliptables.NewDetectLocalByBridgeInterface("eth")),
			errExpected: false,
		},
		// LocalModeInterfaceNamePrefix
		{
			name: "LocalModeInterfaceNamePrefix",
			mode: proxyconfigapi.LocalModeInterfaceNamePrefix,
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{InterfaceNamePrefix: "veth"},
			},
			expected: resolveDualStackLocalDetectors(t)(
				proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("veth"))(
				proxyutiliptables.NewDetectLocalByInterfaceNamePrefix("veth")),
			errExpected: false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			r, err := getDualStackLocalDetectorTuple(logger, c.mode, c.config, c.nodePodCIDRs)
			if c.errExpected {
				if err == nil {
					t.Errorf("Expected error, but succeeded with %q", r)
				}
				return
			}
			if err != nil {
				t.Errorf("Error resolving detect-local: %v", err)
				return
			}
			if !reflect.DeepEqual(r, c.expected) {
				t.Errorf("Unexpected detect-local implementation, expected: %q, got: %q", c.expected, r)
			}
		})
	}
}

func makeNodeWithPodCIDRs(cidrs ...string) *v1.Node {
	if len(cidrs) == 0 {
		return &v1.Node{}
	}
	return &v1.Node{
		Spec: v1.NodeSpec{
			PodCIDR:  cidrs[0],
			PodCIDRs: cidrs,
		},
	}
}

func resolveLocalDetector(t *testing.T) func(proxyutiliptables.LocalTrafficDetector, error) proxyutiliptables.LocalTrafficDetector {
	return func(localDetector proxyutiliptables.LocalTrafficDetector, err error) proxyutiliptables.LocalTrafficDetector {
		t.Helper()
		if err != nil {
			t.Fatalf("Error resolving detect-local: %v", err)
		}
		return localDetector
	}
}

func resolveDualStackLocalDetectors(t *testing.T) func(localDetector proxyutiliptables.LocalTrafficDetector, err1 error) func(proxyutiliptables.LocalTrafficDetector, error) [2]proxyutiliptables.LocalTrafficDetector {
	return func(localDetector proxyutiliptables.LocalTrafficDetector, err error) func(proxyutiliptables.LocalTrafficDetector, error) [2]proxyutiliptables.LocalTrafficDetector {
		t.Helper()
		if err != nil {
			t.Fatalf("Error resolving dual stack detect-local: %v", err)
		}
		return func(otherLocalDetector proxyutiliptables.LocalTrafficDetector, err1 error) [2]proxyutiliptables.LocalTrafficDetector {
			t.Helper()
			if err1 != nil {
				t.Fatalf("Error resolving dual stack detect-local: %v", err)
			}
			return [2]proxyutiliptables.LocalTrafficDetector{localDetector, otherLocalDetector}
		}
	}
}

func TestConfigChange(t *testing.T) {
	setUp := func() (*os.File, string, error) {
		tempDir, err := os.MkdirTemp("", "kubeproxy-config-change")
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
bindAddressHardFail: false
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
detectLocalMode: "BridgeInterface"`)
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
		err = opt.Complete(new(pflag.FlagSet))
		if err != nil {
			t.Fatal(err)
		}
		opt.proxyServer = tc.proxyServer

		errCh := make(chan error, 1)
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

func Test_waitForPodCIDR(t *testing.T) {
	expected := []string{"192.168.0.0/24", "fd00:1:2::/64"}
	nodeName := "test-node"
	oldNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            nodeName,
			ResourceVersion: "1000",
		},
		Spec: v1.NodeSpec{
			PodCIDR:  "10.0.0.0/24",
			PodCIDRs: []string{"10.0.0.0/24", "2001:db2:1/64"},
		},
	}
	node := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:            nodeName,
			ResourceVersion: "1",
		},
	}
	updatedNode := node.DeepCopy()
	updatedNode.Spec.PodCIDRs = expected
	updatedNode.Spec.PodCIDR = expected[0]

	// start with the new node
	client := clientsetfake.NewSimpleClientset()
	client.AddReactor("list", "nodes", func(action clientgotesting.Action) (handled bool, ret runtime.Object, err error) {
		obj := &v1.NodeList{}
		return true, obj, nil
	})
	fakeWatch := watch.NewFake()
	client.PrependWatchReactor("nodes", clientgotesting.DefaultWatchReactor(fakeWatch, nil))

	go func() {
		fakeWatch.Add(node)
		// receive a delete event for the old node
		fakeWatch.Delete(oldNode)
		// set the PodCIDRs on the new node
		fakeWatch.Modify(updatedNode)
	}()
	got, err := waitForPodCIDR(client, node.Name)
	if err != nil {
		t.Errorf("waitForPodCIDR() unexpected error %v", err)
		return
	}
	if !reflect.DeepEqual(got.Spec.PodCIDRs, expected) {
		t.Errorf("waitForPodCIDR() got %v expected to be %v ", got.Spec.PodCIDRs, expected)
	}
}

func TestGetConntrackMax(t *testing.T) {
	ncores := goruntime.NumCPU()
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
		cfg := proxyconfigapi.KubeProxyConntrackConfiguration{
			Min:        ptr.To(tc.min),
			MaxPerCore: ptr.To(tc.maxPerCore),
		}
		logger, _ := ktesting.NewTestContext(t)
		x, e := getConntrackMax(logger, cfg)
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

func TestProxyServer_platformSetup(t *testing.T) {
	tests := []struct {
		name         string
		node         *v1.Node
		config       *proxyconfigapi.KubeProxyConfiguration
		wantPodCIDRs []string
	}{
		{
			name:         "LocalModeNodeCIDR store the node PodCIDRs obtained",
			node:         makeNodeWithPodCIDRs("10.0.0.0/24"),
			config:       &proxyconfigapi.KubeProxyConfiguration{DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR},
			wantPodCIDRs: []string{"10.0.0.0/24"},
		},
		{
			name:         "LocalModeNodeCIDR store the node PodCIDRs obtained dual stack",
			node:         makeNodeWithPodCIDRs("10.0.0.0/24", "2001:db2:1/64"),
			config:       &proxyconfigapi.KubeProxyConfiguration{DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR},
			wantPodCIDRs: []string{"10.0.0.0/24", "2001:db2:1/64"},
		},
		{
			name:   "LocalModeClusterCIDR does not get the node PodCIDRs",
			node:   makeNodeWithPodCIDRs("10.0.0.0/24", "2001:db2:1/64"),
			config: &proxyconfigapi.KubeProxyConfiguration{DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset(tt.node)
			s := &ProxyServer{
				Config:   tt.config,
				Client:   client,
				Hostname: "nodename",
				NodeIPs: map[v1.IPFamily]net.IP{
					v1.IPv4Protocol: netutils.ParseIPSloppy("127.0.0.1"),
					v1.IPv6Protocol: net.IPv6zero,
				},
			}
			err := s.platformSetup()
			if err != nil {
				t.Errorf("ProxyServer.createProxier() error = %v", err)
				return
			}
			if !reflect.DeepEqual(s.podCIDRs, tt.wantPodCIDRs) {
				t.Errorf("Expected PodCIDRs %v got %v", tt.wantPodCIDRs, s.podCIDRs)
			}

		})
	}
}
