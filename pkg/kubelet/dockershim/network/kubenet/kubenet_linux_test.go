// +build !dockerless

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

package kubenet

import (
	"fmt"
	"net"
	"strings"
	"testing"

	"github.com/containernetworking/cni/libcni"
	"github.com/containernetworking/cni/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	utilsets "k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	mockcni "k8s.io/kubernetes/pkg/kubelet/dockershim/network/cni/testing"
	nettest "k8s.io/kubernetes/pkg/kubelet/dockershim/network/testing"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	ipttest "k8s.io/kubernetes/pkg/util/iptables/testing"
	sysctltest "k8s.io/kubernetes/pkg/util/sysctl/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

// test it fulfills the NetworkPlugin interface
var _ network.NetworkPlugin = &kubenetNetworkPlugin{}

func newFakeKubenetPlugin(initMap map[kubecontainer.ContainerID]utilsets.String, execer exec.Interface, host network.Host) *kubenetNetworkPlugin {
	return &kubenetNetworkPlugin{
		podIPs: initMap,
		execer: execer,
		mtu:    1460,
		host:   host,
	}
}

func TestGetPodNetworkStatus(t *testing.T) {
	podIPMap := make(map[kubecontainer.ContainerID]utilsets.String)
	podIPMap[kubecontainer.ContainerID{ID: "1"}] = utilsets.NewString("10.245.0.2")
	podIPMap[kubecontainer.ContainerID{ID: "2"}] = utilsets.NewString("10.245.0.3")
	podIPMap[kubecontainer.ContainerID{ID: "3"}] = utilsets.NewString("10.245.0.4", "2000::")
	podIPMap[kubecontainer.ContainerID{ID: "4"}] = utilsets.NewString("2000::2")

	testCases := []struct {
		id          string
		expectError bool
		expectIP    utilsets.String
	}{
		//in podCIDR map
		{
			id:          "1",
			expectError: false,
			expectIP:    utilsets.NewString("10.245.0.2"),
		},
		{
			id:          "2",
			expectError: false,
			expectIP:    utilsets.NewString("10.245.0.3"),
		},
		{
			id:          "3",
			expectError: false,
			expectIP:    utilsets.NewString("10.245.0.4", "2000::"),
		},
		{
			id:          "4",
			expectError: false,
			expectIP:    utilsets.NewString("2000::2"),
		},

		//not in podIP map
		{
			id:          "does-not-exist-map",
			expectError: true,
			expectIP:    nil,
		},
		//TODO: add test cases for retrieving ip inside container network namespace
	}

	fakeCmds := make([]fakeexec.FakeCommandAction, 0)
	for _, t := range testCases {
		// the fake commands return the IP from the given index, or an error
		fCmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) {
					ips, ok := podIPMap[kubecontainer.ContainerID{ID: t.id}]
					if !ok {
						return nil, nil, fmt.Errorf("Pod IP %q not found", t.id)
					}
					ipsList := ips.UnsortedList()
					return []byte(ipsList[0]), nil, nil
				},
			},
		}
		fakeCmds = append(fakeCmds, func(cmd string, args ...string) exec.Cmd {
			return fakeexec.InitFakeCmd(&fCmd, cmd, args...)
		})
	}
	fexec := fakeexec.FakeExec{
		CommandScript: fakeCmds,
		LookPathFunc: func(file string) (string, error) {
			return fmt.Sprintf("/fake-bin/%s", file), nil
		},
	}

	fhost := nettest.NewFakeHost(nil)
	fakeKubenet := newFakeKubenetPlugin(podIPMap, &fexec, fhost)

	for i, tc := range testCases {
		out, err := fakeKubenet.GetPodNetworkStatus("", "", kubecontainer.ContainerID{ID: tc.id})
		if tc.expectError {
			if err == nil {
				t.Errorf("Test case %d expects error but got none", i)
			}
			continue
		} else {
			if err != nil {
				t.Errorf("Test case %d expects error but got error: %v", i, err)
			}
		}
		seen := make(map[string]bool)
		allExpected := tc.expectIP.UnsortedList()
		for _, expectedIP := range allExpected {
			for _, outIP := range out.IPs {
				if expectedIP == outIP.String() {
					seen[expectedIP] = true
					break
				}
			}
		}
		if len(tc.expectIP) != len(seen) {
			t.Errorf("Test case %d expects ip %s but got %s", i, tc.expectIP, out.IP.String())
		}

	}
}

// TestTeardownCallsShaper tests that a `TearDown` call does call
// `shaper.Reset`
func TestTeardownCallsShaper(t *testing.T) {
	fexec := &fakeexec.FakeExec{
		CommandScript: []fakeexec.FakeCommandAction{},
		LookPathFunc: func(file string) (string, error) {
			return fmt.Sprintf("/fake-bin/%s", file), nil
		},
	}
	fhost := nettest.NewFakeHost(nil)
	fshaper := &bandwidth.FakeShaper{}
	mockcni := &mockcni.MockCNI{}
	ips := make(map[kubecontainer.ContainerID]utilsets.String)
	kubenet := newFakeKubenetPlugin(ips, fexec, fhost)
	kubenet.loConfig = &libcni.NetworkConfig{
		Network: &types.NetConf{
			Name: "loopback-fake",
			Type: "loopback",
		},
	}
	kubenet.cniConfig = mockcni
	kubenet.iptables = ipttest.NewFake()
	kubenet.bandwidthShaper = fshaper

	mockcni.On("DelNetwork", mock.AnythingOfType("*context.timerCtx"), mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

	details := make(map[string]interface{})
	details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = "10.0.0.1/24"
	kubenet.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)

	existingContainerID := kubecontainer.BuildContainerID("docker", "123")
	kubenet.podIPs[existingContainerID] = utilsets.NewString("10.0.0.1")

	if err := kubenet.TearDownPod("namespace", "name", existingContainerID); err != nil {
		t.Fatalf("Unexpected error in TearDownPod: %v", err)
	}
	assert.Equal(t, []string{"10.0.0.1/32"}, fshaper.ResetCIDRs, "shaper.Reset should have been called")

	mockcni.AssertExpectations(t)
}

// TestInit tests that a `Init` call with an MTU sets the MTU
func TestInit_MTU(t *testing.T) {
	var fakeCmds []fakeexec.FakeCommandAction
	{
		// modprobe br-netfilter
		fCmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeAction{
				func() ([]byte, []byte, error) {
					return make([]byte, 0), nil, nil
				},
			},
		}
		fakeCmds = append(fakeCmds, func(cmd string, args ...string) exec.Cmd {
			return fakeexec.InitFakeCmd(&fCmd, cmd, args...)
		})
	}

	fexec := &fakeexec.FakeExec{
		CommandScript: fakeCmds,
		LookPathFunc: func(file string) (string, error) {
			return fmt.Sprintf("/fake-bin/%s", file), nil
		},
	}

	fhost := nettest.NewFakeHost(nil)
	ips := make(map[kubecontainer.ContainerID]utilsets.String)
	kubenet := newFakeKubenetPlugin(ips, fexec, fhost)
	kubenet.iptables = ipttest.NewFake()

	sysctl := sysctltest.NewFake()
	sysctl.Settings["net/bridge/bridge-nf-call-iptables"] = 0
	kubenet.sysctl = sysctl

	if err := kubenet.Init(nettest.NewFakeHost(nil), kubeletconfig.HairpinNone, "10.0.0.0/8", 1234); err != nil {
		t.Fatalf("Unexpected error in Init: %v", err)
	}
	assert.Equal(t, 1234, kubenet.mtu, "kubenet.mtu should have been set")
	assert.Equal(t, 1, sysctl.Settings["net/bridge/bridge-nf-call-iptables"], "net/bridge/bridge-nf-call-iptables sysctl should have been set")
}

// TestInvocationWithoutRuntime invokes the plugin without a runtime.
// This is how kubenet is invoked from the cri.
func TestTearDownWithoutRuntime(t *testing.T) {
	testCases := []struct {
		podCIDR         []string
		expectedPodCIDR []string
		ip              string
	}{
		{
			podCIDR:         []string{"10.0.0.0/24"},
			expectedPodCIDR: []string{"10.0.0.0/24"},
			ip:              "10.0.0.1",
		},
		{
			podCIDR:         []string{"10.0.0.1/24"},
			expectedPodCIDR: []string{"10.0.0.0/24"},
			ip:              "10.0.0.1",
		},
		{
			podCIDR:         []string{"2001:beef::/48"},
			expectedPodCIDR: []string{"2001:beef::/48"},
			ip:              "2001:beef::1",
		},
		{
			podCIDR:         []string{"2001:beef::1/48"},
			expectedPodCIDR: []string{"2001:beef::/48"},
			ip:              "2001:beef::1",
		},
	}
	for _, tc := range testCases {

		fhost := nettest.NewFakeHost(nil)
		fhost.Legacy = false
		mockcni := &mockcni.MockCNI{}

		fexec := &fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{},
			LookPathFunc: func(file string) (string, error) {
				return fmt.Sprintf("/fake-bin/%s", file), nil
			},
		}

		ips := make(map[kubecontainer.ContainerID]utilsets.String)
		kubenet := newFakeKubenetPlugin(ips, fexec, fhost)
		kubenet.loConfig = &libcni.NetworkConfig{
			Network: &types.NetConf{
				Name: "loopback-fake",
				Type: "loopback",
			},
		}
		kubenet.cniConfig = mockcni
		kubenet.iptables = ipttest.NewFake()

		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = strings.Join(tc.podCIDR, ",")
		kubenet.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)

		if len(kubenet.podCIDRs) != len(tc.podCIDR) {
			t.Errorf("generated podCidr: %q, expecting: %q are not of the same length", kubenet.podCIDRs, tc.podCIDR)
			continue
		}
		for idx := range tc.podCIDR {
			if kubenet.podCIDRs[idx].String() != tc.expectedPodCIDR[idx] {
				t.Errorf("generated podCidr: %q, expecting: %q", kubenet.podCIDRs[idx].String(), tc.expectedPodCIDR[idx])
			}
		}

		existingContainerID := kubecontainer.BuildContainerID("docker", "123")
		kubenet.podIPs[existingContainerID] = utilsets.NewString(tc.ip)

		mockcni.On("DelNetwork", mock.AnythingOfType("*context.timerCtx"), mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

		if err := kubenet.TearDownPod("namespace", "name", existingContainerID); err != nil {
			t.Fatalf("Unexpected error in TearDownPod: %v", err)
		}
		// Assert that the CNI DelNetwork made it through and we didn't crash
		// without a runtime.
		mockcni.AssertExpectations(t)
	}
}

func TestGetRoutesConfig(t *testing.T) {
	for _, test := range []struct {
		cidrs  []string
		routes string
	}{
		{
			cidrs:  []string{"10.0.0.1/24"},
			routes: `{"dst": "0.0.0.0/0"}`,
		},
		{
			cidrs:  []string{"2001:4860:4860::8888/32"},
			routes: `{"dst": "::/0"}`,
		},
		{
			cidrs:  []string{"2001:4860:4860::8888/32", "10.0.0.1/24"},
			routes: `{"dst": "0.0.0.0/0"},{"dst": "::/0"}`,
		},
	} {
		var cidrs []*net.IPNet
		for _, c := range test.cidrs {
			_, cidr, err := net.ParseCIDR(c)
			assert.NoError(t, err)
			cidrs = append(cidrs, cidr)
		}
		fakeKubenet := &kubenetNetworkPlugin{podCIDRs: cidrs}
		assert.Equal(t, test.routes, fakeKubenet.getRoutesConfig())
	}
}

func TestGetRangesConfig(t *testing.T) {
	for _, test := range []struct {
		cidrs  []string
		ranges string
	}{
		{
			cidrs: []string{"10.0.0.0/24"},
			ranges: `
[{
"subnet": "10.0.0.0/24"
}]`,
		},
		{
			cidrs: []string{"2001:4860::/32"},
			ranges: `
[{
"subnet": "2001:4860::/32"
}]`,
		},
		{
			cidrs: []string{"10.0.0.0/24", "2001:4860::/32"},
			ranges: `
[{
"subnet": "10.0.0.0/24"
}],
[{
"subnet": "2001:4860::/32"
}]`,
		},
	} {
		var cidrs []*net.IPNet
		for _, c := range test.cidrs {
			_, cidr, err := net.ParseCIDR(c)
			assert.NoError(t, err)
			cidrs = append(cidrs, cidr)
		}
		fakeKubenet := &kubenetNetworkPlugin{podCIDRs: cidrs}
		assert.Equal(t, test.ranges, fakeKubenet.getRangesConfig())
	}
}

//TODO: add unit test for each implementation of network plugin interface
