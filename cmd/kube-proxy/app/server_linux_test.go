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
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	goruntime "runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/pflag"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
	"k8s.io/kubernetes/test/utils/ktesting"
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

func Test_getLocalDetectors(t *testing.T) {
	cases := []struct {
		name            string
		config          *proxyconfigapi.KubeProxyConfiguration
		primaryIPFamily v1.IPFamily
		nodePodCIDRs    []string
		expected        map[v1.IPFamily]proxyutil.LocalTrafficDetector
	}{
		// LocalModeClusterCIDR
		{
			name: "LocalModeClusterCIDR, single-stack IPv4 cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv6 cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv6 cluster with single-stack IPv4 config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			// This will output a warning that there is no IPv6 CIDR but it
			// will still use the provided IPv4 CIDR for IPv4.
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeClusterCIDR, single-stack IPv4 cluster with single-stack IPv6 config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			// This will output a warning that there is no IPv4 CIDR but it
			// will still use the provided IPv6 CIDR for IPv6.
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, dual-stack IPv4-primary cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, dual-stack IPv6-primary cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, IPv4-primary kube-proxy / IPv6-primary config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/14"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002:0:0:1234::/64"),
			},
		},
		{
			name: "LocalModeClusterCIDR, no ClusterCIDR",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeClusterCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{""},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		// LocalModeNodeCIDR
		{
			name: "LocalModeNodeCIDR, single-stack IPv4 cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24"},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv6 cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv6 cluster with single-stack IPv4 config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24"},
			// This will output a warning that there is no IPv6 CIDR but it
			// will still use the provided IPv4 CIDR for IPv4.
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		{
			name: "LocalModeNodeCIDR, single-stack IPv4 cluster with single-stack IPv6 config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96"},
			// This will output a warning that there is no IPv4 CIDR but it
			// will still use the provided IPv6 CIDR for IPv6.
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, dual-stack IPv4-primary cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, dual-stack IPv6-primary cluster",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"2002:0:0:1234::/64", "10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"2002::1234:abcd:ffff:0:0/96", "10.0.0.0/24"},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, IPv6-primary kube-proxy / IPv4-primary config",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14", "2002:0:0:1234::/64"},
				},
			},
			primaryIPFamily: v1.IPv6Protocol,
			nodePodCIDRs:    []string{"10.0.0.0/24", "2002::1234:abcd:ffff:0:0/96"},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByCIDR("10.0.0.0/24"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByCIDR("2002::1234:abcd:ffff:0:0/96"),
			},
		},
		{
			name: "LocalModeNodeCIDR, no PodCIDRs",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeNodeCIDR,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{""},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			nodePodCIDRs:    []string{},
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		// unknown mode
		{
			name: "unknown LocalMode",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalMode("abcd"),
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					ClusterCIDRs: []string{"10.0.0.0/14"},
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewNoOpLocalDetector(),
				v1.IPv6Protocol: proxyutil.NewNoOpLocalDetector(),
			},
		},
		// LocalModeBridgeInterface
		{
			name: "LocalModeBridgeInterface",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeBridgeInterface,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					BridgeInterface: "eth",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByBridgeInterface("eth"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByBridgeInterface("eth"),
			},
		},
		{
			name: "LocalModeBridgeInterface, strange bridge name",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeBridgeInterface,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					BridgeInterface: "1234567890123456789",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByBridgeInterface("1234567890123456789"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByBridgeInterface("1234567890123456789"),
			},
		},
		// LocalModeInterfaceNamePrefix
		{
			name: "LocalModeInterfaceNamePrefix",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeInterfaceNamePrefix,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					InterfaceNamePrefix: "eth",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByInterfaceNamePrefix("eth"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByInterfaceNamePrefix("eth"),
			},
		},
		{
			name: "LocalModeInterfaceNamePrefix, strange interface name",
			config: &proxyconfigapi.KubeProxyConfiguration{
				DetectLocalMode: proxyconfigapi.LocalModeInterfaceNamePrefix,
				DetectLocal: proxyconfigapi.DetectLocalConfiguration{
					InterfaceNamePrefix: "1234567890123456789",
				},
			},
			primaryIPFamily: v1.IPv4Protocol,
			expected: map[v1.IPFamily]proxyutil.LocalTrafficDetector{
				v1.IPv4Protocol: proxyutil.NewDetectLocalByInterfaceNamePrefix("1234567890123456789"),
				v1.IPv6Protocol: proxyutil.NewDetectLocalByInterfaceNamePrefix("1234567890123456789"),
			},
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			r := getLocalDetectors(logger, c.primaryIPFamily, c.config, c.nodePodCIDRs)
			if !reflect.DeepEqual(r, c.expected) {
				t.Errorf("Unexpected detect-local implementation, expected: %q, got: %q", c.expected, r)
			}
		})
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
		_, ctx := ktesting.NewTestContext(t)
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
			errCh <- opt.runLoop(ctx)
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
		_, ctx := ktesting.NewTestContext(t)
		x, e := getConntrackMax(ctx, cfg)
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

type fakeConntracker struct {
	called []string
	err    error
}

// SetMax value is calculated based on the number of CPUs by getConntrackMax()
func (fc *fakeConntracker) SetMax(ctx context.Context, max int) error {
	fc.called = append(fc.called, "SetMax")
	return fc.err
}
func (fc *fakeConntracker) SetTCPEstablishedTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPEstablishedTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetTCPCloseWaitTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPCloseWaitTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetTCPBeLiberal(ctx context.Context, value int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPBeLiberal(%d)", value))
	return fc.err
}
func (fc *fakeConntracker) SetUDPTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetUDPTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetUDPStreamTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetUDPStreamTimeout(%d)", seconds))
	return fc.err
}

func TestSetupConntrack(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tests := []struct {
		name         string
		config       proxyconfigapi.KubeProxyConntrackConfiguration
		expect       []string
		conntrackErr error
		wantErr      bool
	}{
		{
			name:   "do nothing if conntrack config is empty",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{},
			expect: nil,
		},
		{
			name: "SetMax is called if conntrack.maxPerCore is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			expect: []string{"SetMax"},
		},
		{
			name: "SetMax is not called if conntrack.maxPerCore is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(0)),
			},
			expect: nil,
		},
		{
			name: "SetTCPEstablishedTimeout is called if conntrack.tcpEstablishedTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPEstablishedTimeout(5)"},
		},
		{
			name: "SetTCPEstablishedTimeout is not called if conntrack.tcpEstablishedTimeout is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPCloseWaitTimeout is called if conntrack.tcpCloseWaitTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPCloseWaitTimeout(5)"},
		},
		{
			name: "SetTCPCloseWaitTimeout is not called if conntrack.tcpCloseWaitTimeout is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPBeLiberal is called if conntrack.tcpBeLiberal is true",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPBeLiberal: true,
			},
			expect: []string{"SetTCPBeLiberal(1)"},
		},
		{
			name: "SetTCPBeLiberal is not called if conntrack.tcpBeLiberal is false",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPBeLiberal: false,
			},
			expect: nil,
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPTimeout(5)"},
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is zero",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPStreamTimeout(5)"},
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is zero",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "an error is returned if conntrack.SetTCPEstablishedTimeout fails",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect:       []string{"SetTCPEstablishedTimeout(5)"},
			conntrackErr: errors.New("random error"),
			wantErr:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fc := &fakeConntracker{err: test.conntrackErr}
			s := &ProxyServer{
				Config: &proxyconfigapi.KubeProxyConfiguration{
					Linux: proxyconfigapi.KubeProxyLinuxConfiguration{
						Conntrack: test.config,
					},
				},
			}
			err := s.setupConntrack(ctx, fc)
			if test.wantErr && err == nil {
				t.Errorf("Test %q: Expected error, got nil", test.name)
			}
			if !test.wantErr && err != nil {
				t.Errorf("Test %q: Expected no error, got %v", test.name, err)
			}
			if !cmp.Equal(fc.called, test.expect) {
				t.Errorf("Test %q: Expected conntrack calls: %v, got: %v", test.name, test.expect, fc.called)
			}
		})
	}
}
