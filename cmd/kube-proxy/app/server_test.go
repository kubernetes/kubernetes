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
	"context"
	"errors"
	"fmt"
	"net"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/utils/ktesting"
	netutils "k8s.io/utils/net"
)

type fakeProxyServerLongRun struct{}

// Run runs the specified ProxyServer.
func (s *fakeProxyServerLongRun) Run(ctx context.Context) error {
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
func (s *fakeProxyServerError) Run(ctx context.Context) error {
	for {
		time.Sleep(2 * time.Second)
		return fmt.Errorf("mocking error from ProxyServer.Run()")
	}
}

// CleanupAndExit runs in the specified ProxyServer.
func (s *fakeProxyServerError) CleanupAndExit() error {
	return errors.New("mocking error from ProxyServer.CleanupAndExit()")
}

func Test_detectNodeIPs(t *testing.T) {
	cases := []struct {
		name           string
		rawNodeIPs     []net.IP
		bindAddress    string
		expectedFamily v1.IPFamily
		expectedIPv4   string
		expectedIPv6   string
	}{
		{
			name:           "Bind address IPv4 unicast address and no Node object",
			rawNodeIPs:     nil,
			bindAddress:    "10.0.0.1",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "10.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address IPv6 unicast address and no Node object",
			rawNodeIPs:     nil,
			bindAddress:    "fd00:4321::2",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:4321::2",
		},
		{
			name:           "No Valid IP found and no bind address",
			rawNodeIPs:     nil,
			bindAddress:    "",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "No Valid IP found and unspecified bind address",
			rawNodeIPs:     nil,
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address 0.0.0.0 and node with IPv4 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "192.168.1.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address :: and node with IPv4 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("192.168.1.1")},
			bindAddress:    "::",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "192.168.1.1",
			expectedIPv6:   "::1",
		},
		{
			name:           "Bind address 0.0.0.0 and node with IPv6 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("fd00:1234::1")},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:1234::1",
		},
		{
			name:           "Bind address :: and node with IPv6 InternalIP set",
			rawNodeIPs:     []net.IP{netutils.ParseIPSloppy("fd00:1234::1")},
			bindAddress:    "::",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "fd00:1234::1",
		},
		{
			name: "Dual stack, primary IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "::",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, primary IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "0.0.0.0",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "80.80.80.80",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "80.80.80.80",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "2001:db8::555",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "2001:db8::555",
		},
		{
			name: "Dual stack, override primary family, IPv4",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("2001:db8::2"),
				netutils.ParseIPSloppy("90.90.90.90"),
			},
			bindAddress:    "127.0.0.1",
			expectedFamily: v1.IPv4Protocol,
			expectedIPv4:   "127.0.0.1",
			expectedIPv6:   "2001:db8::2",
		},
		{
			name: "Dual stack, override primary family, IPv6",
			rawNodeIPs: []net.IP{
				netutils.ParseIPSloppy("90.90.90.90"),
				netutils.ParseIPSloppy("2001:db8::2"),
			},
			bindAddress:    "::1",
			expectedFamily: v1.IPv6Protocol,
			expectedIPv4:   "90.90.90.90",
			expectedIPv6:   "::1",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			primaryFamily, ips := detectNodeIPs(ctx, c.rawNodeIPs, c.bindAddress)
			if primaryFamily != c.expectedFamily {
				t.Errorf("Expected family %q got %q", c.expectedFamily, primaryFamily)
			}
			if ips[v1.IPv4Protocol].String() != c.expectedIPv4 {
				t.Errorf("Expected IPv4 %q got %q", c.expectedIPv4, ips[v1.IPv4Protocol].String())
			}
			if ips[v1.IPv6Protocol].String() != c.expectedIPv6 {
				t.Errorf("Expected IPv6 %q got %q", c.expectedIPv6, ips[v1.IPv6Protocol].String())
			}
		})
	}
}

func Test_checkBadConfig(t *testing.T) {
	cases := []struct {
		name  string
		proxy *ProxyServer
		err   bool
	}{
		{
			name: "single-stack NodePortAddresses with single-stack config",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"10.0.0.0/8"},
					},
					NodePortAddresses: []string{"192.168.0.0/24"},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			err: false,
		},
		{
			name: "dual-stack NodePortAddresses with dual-stack config",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"10.0.0.0/8", "fd09::/64"},
					},
					NodePortAddresses: []string{"192.168.0.0/24", "fd03::/64"},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			err: false,
		},
		{
			name: "empty NodePortAddresses",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					NodePortAddresses: []string{},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			err: true,
		},
		{
			name: "single-stack NodePortAddresses with dual-stack config",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"10.0.0.0/8", "fd09::/64"},
					},
					NodePortAddresses: []string{"192.168.0.0/24"},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			err: true,
		},
		{
			name: "wrong-single-stack NodePortAddresses",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"fd09::/64"},
					},
					NodePortAddresses: []string{"192.168.0.0/24"},
				},
				PrimaryIPFamily: v1.IPv6Protocol,
			},
			err: true,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err := checkBadConfig(c.proxy)
			if err != nil && !c.err {
				t.Errorf("unexpected error: %v", err)
			} else if err == nil && c.err {
				t.Errorf("unexpected lack of error")
			}
		})
	}
}

func Test_checkBadIPConfig(t *testing.T) {
	cases := []struct {
		name    string
		proxy   *ProxyServer
		ssErr   bool
		ssFatal bool
		dsErr   bool
		dsFatal bool
	}{
		{
			name: "empty config",
			proxy: &ProxyServer{
				Config:          &kubeproxyconfig.KubeProxyConfiguration{},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},

		{
			name: "ok single-stack clusterCIDR",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"10.0.0.0/8"},
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok dual-stack clusterCIDR",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"10.0.0.0/8", "fd01:2345::/64"},
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok reversed dual-stack clusterCIDR",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"fd01:2345::/64", "10.0.0.0/8"},
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "wrong-family clusterCIDR",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"fd01:2345::/64"},
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr:   true,
			ssFatal: false,
			dsErr:   true,
			dsFatal: false,
		},
		{
			name: "wrong-family clusterCIDR when using ClusterCIDR LocalDetector",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocal: kubeproxyconfig.DetectLocalConfiguration{
						ClusterCIDRs: []string{"fd01:2345::/64"},
					},
					DetectLocalMode: kubeproxyconfig.LocalModeClusterCIDR,
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr:   true,
			ssFatal: true,
			dsErr:   true,
			dsFatal: false,
		},

		{
			name: "ok single-stack node.spec.podCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				},
				PrimaryIPFamily: v1.IPv4Protocol,
				podCIDRs:        []string{"10.0.0.0/8"},
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok dual-stack node.spec.podCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				},
				PrimaryIPFamily: v1.IPv4Protocol,
				podCIDRs:        []string{"10.0.0.0/8", "fd01:2345::/64"},
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok reversed dual-stack node.spec.podCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				},
				PrimaryIPFamily: v1.IPv4Protocol,
				podCIDRs:        []string{"fd01:2345::/64", "10.0.0.0/8"},
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "wrong-family node.spec.podCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					DetectLocalMode: kubeproxyconfig.LocalModeNodeCIDR,
				},
				PrimaryIPFamily: v1.IPv4Protocol,
				podCIDRs:        []string{"fd01:2345::/64"},
			},
			ssErr:   true,
			ssFatal: true,
			dsErr:   true,
			dsFatal: true,
		},

		{
			name: "ok winkernel.sourceVip",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					Winkernel: kubeproxyconfig.KubeProxyWinkernelConfiguration{
						SourceVip: "10.0.0.1",
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "wrong family winkernel.sourceVip",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					Winkernel: kubeproxyconfig.KubeProxyWinkernelConfiguration{
						SourceVip: "fd01:2345::1",
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr:   true,
			ssFatal: false,
			dsErr:   true,
			dsFatal: false,
		},

		{
			name: "ok IPv4 metricsBindAddress",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					MetricsBindAddress: "10.0.0.1:9999",
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok IPv6 metricsBindAddress",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					MetricsBindAddress: "[fd01:2345::1]:9999",
				},
				PrimaryIPFamily: v1.IPv6Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "ok unspecified wrong-family metricsBindAddress",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					MetricsBindAddress: "0.0.0.0:9999",
				},
				PrimaryIPFamily: v1.IPv6Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "wrong family metricsBindAddress",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					MetricsBindAddress: "10.0.0.1:9999",
				},
				PrimaryIPFamily: v1.IPv6Protocol,
			},
			ssErr:   true,
			ssFatal: false,
			dsErr:   false,
		},

		{
			name: "ok ipvs.excludeCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
						ExcludeCIDRs: []string{"10.0.0.0/8"},
					},
				},
				PrimaryIPFamily: v1.IPv4Protocol,
			},
			ssErr: false,
			dsErr: false,
		},
		{
			name: "wrong family ipvs.excludeCIDRs",
			proxy: &ProxyServer{
				Config: &kubeproxyconfig.KubeProxyConfiguration{
					IPVS: kubeproxyconfig.KubeProxyIPVSConfiguration{
						ExcludeCIDRs: []string{"10.0.0.0/8", "192.168.0.0/24"},
					},
				},
				PrimaryIPFamily: v1.IPv6Protocol,
			},
			ssErr:   true,
			ssFatal: false,
			dsErr:   false,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			err, fatal := checkBadIPConfig(c.proxy, false)
			if err != nil && !c.ssErr {
				t.Errorf("unexpected error in single-stack case: %v", err)
			} else if err == nil && c.ssErr {
				t.Errorf("unexpected lack of error in single-stack case")
			} else if fatal != c.ssFatal {
				t.Errorf("expected fatal=%v, got %v", c.ssFatal, fatal)
			}

			err, fatal = checkBadIPConfig(c.proxy, true)
			if err != nil && !c.dsErr {
				t.Errorf("unexpected error in dual-stack case: %v", err)
			} else if err == nil && c.dsErr {
				t.Errorf("unexpected lack of error in dual-stack case")
			} else if fatal != c.dsFatal {
				t.Errorf("expected fatal=%v, got %v", c.dsFatal, fatal)
			}
		})
	}
}
