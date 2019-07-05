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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	"strings"

	"testing"

	utilsets "k8s.io/apimachinery/pkg/util/sets"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/network/cni/testing"
	hostporttest "k8s.io/kubernetes/pkg/kubelet/dockershim/network/hostport/testing"
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
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				func() ([]byte, error) {
					ips, ok := podIPMap[kubecontainer.ContainerID{ID: t.id}]
					if !ok {
						return nil, fmt.Errorf("Pod IP %q not found", t.id)
					}
					ipsList := ips.UnsortedList()
					return []byte(ipsList[0]), nil
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
	mockcni := &mock_cni.MockCNI{}
	ips := make(map[kubecontainer.ContainerID]utilsets.String)
	kubenet := newFakeKubenetPlugin(ips, fexec, fhost)
	kubenet.cniConfig = mockcni
	kubenet.iptables = ipttest.NewFake()
	kubenet.bandwidthShaper = fshaper
	kubenet.hostportSyncer = hostporttest.NewFakeHostportSyncer()

	mockcni.On("DelNetwork", mock.AnythingOfType("*context.emptyCtx"), mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

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
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				func() ([]byte, error) {
					return make([]byte, 0), nil
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
		ip              string
		expectedGateway []string
	}{
		{
			podCIDR:         []string{"10.0.0.1/24"},
			ip:              "10.0.0.1",
			expectedGateway: []string{"10.0.0.1"},
		},
		{
			podCIDR:         []string{"2001:beef::1/48"},
			ip:              "2001:beef::1",
			expectedGateway: []string{"2001:beef::1"},
		},
	}
	for _, tc := range testCases {

		fhost := nettest.NewFakeHost(nil)
		fhost.Legacy = false
		mockcni := &mock_cni.MockCNI{}

		fexec := &fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{},
			LookPathFunc: func(file string) (string, error) {
				return fmt.Sprintf("/fake-bin/%s", file), nil
			},
		}

		ips := make(map[kubecontainer.ContainerID]utilsets.String)
		kubenet := newFakeKubenetPlugin(ips, fexec, fhost)
		kubenet.cniConfig = mockcni
		kubenet.iptables = ipttest.NewFake()

		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = strings.Join(tc.podCIDR, ",")
		kubenet.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)

		if len(kubenet.podGateways) != len(tc.expectedGateway) {
			t.Errorf("generated gateway: %q, expecting: %q are not of the same length", kubenet.podGateways, tc.expectedGateway)
			continue
		}

		for idx := range tc.expectedGateway {
			if kubenet.podGateways[idx].String() != tc.expectedGateway[idx] {
				t.Errorf("generated gateway: %q, expecting: %q", kubenet.podGateways[idx].String(), tc.expectedGateway[idx])

			}
		}

		if len(kubenet.podCIDRs) != len(tc.podCIDR) {
			t.Errorf("generated podCidr: %q, expecting: %q are not of the same length", kubenet.podCIDRs, tc.podCIDR)
			continue
		}
		for idx := range tc.podCIDR {
			if kubenet.podCIDRs[idx].String() != tc.podCIDR[idx] {
				t.Errorf("generated podCidr: %q, expecting: %q", kubenet.podCIDRs[idx].String(), tc.podCIDR[idx])
			}
		}

		existingContainerID := kubecontainer.BuildContainerID("docker", "123")
		kubenet.podIPs[existingContainerID] = utilsets.NewString(tc.ip)

		mockcni.On("DelNetwork", mock.AnythingOfType("*context.emptyCtx"), mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

		if err := kubenet.TearDownPod("namespace", "name", existingContainerID); err != nil {
			t.Fatalf("Unexpected error in TearDownPod: %v", err)
		}
		// Assert that the CNI DelNetwork made it through and we didn't crash
		// without a runtime.
		mockcni.AssertExpectations(t)
	}
}

//TODO: add unit test for each implementation of network plugin interface
