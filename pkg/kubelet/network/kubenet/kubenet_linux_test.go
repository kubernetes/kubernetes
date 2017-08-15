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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"

	"testing"

	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/network/cni/testing"
	hostporttest "k8s.io/kubernetes/pkg/kubelet/network/hostport/testing"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	ipttest "k8s.io/kubernetes/pkg/util/iptables/testing"
	sysctltest "k8s.io/kubernetes/pkg/util/sysctl/testing"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
)

// test it fulfills the NetworkPlugin interface
var _ network.NetworkPlugin = &kubenetNetworkPlugin{}

func newFakeKubenetPlugin(initMap map[kubecontainer.ContainerID]string, execer exec.Interface, host network.Host) *kubenetNetworkPlugin {
	return &kubenetNetworkPlugin{
		podIPs: initMap,
		execer: execer,
		mtu:    1460,
		host:   host,
	}
}

func TestGetPodNetworkStatus(t *testing.T) {
	podIPMap := make(map[kubecontainer.ContainerID]string)
	podIPMap[kubecontainer.ContainerID{ID: "1"}] = "10.245.0.2"
	podIPMap[kubecontainer.ContainerID{ID: "2"}] = "10.245.0.3"

	testCases := []struct {
		id          string
		expectError bool
		expectIP    string
	}{
		//in podCIDR map
		{
			"1",
			false,
			"10.245.0.2",
		},
		{
			"2",
			false,
			"10.245.0.3",
		},
		//not in podCIDR map
		{
			"3",
			true,
			"",
		},
		//TODO: add test cases for retrieving ip inside container network namespace
	}

	fakeCmds := make([]fakeexec.FakeCommandAction, 0)
	for _, t := range testCases {
		// the fake commands return the IP from the given index, or an error
		fCmd := fakeexec.FakeCmd{
			CombinedOutputScript: []fakeexec.FakeCombinedOutputAction{
				func() ([]byte, error) {
					ip, ok := podIPMap[kubecontainer.ContainerID{ID: t.id}]
					if !ok {
						return nil, fmt.Errorf("Pod IP %q not found", t.id)
					}
					return []byte(ip), nil
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
		if tc.expectIP != out.IP.String() {
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
	kubenet := newFakeKubenetPlugin(map[kubecontainer.ContainerID]string{}, fexec, fhost)
	kubenet.cniConfig = mockcni
	kubenet.iptables = ipttest.NewFake()
	kubenet.bandwidthShaper = fshaper
	kubenet.hostportSyncer = hostporttest.NewFakeHostportSyncer()

	mockcni.On("DelNetwork", mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

	details := make(map[string]interface{})
	details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = "10.0.0.1/24"
	kubenet.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)

	existingContainerID := kubecontainer.BuildContainerID("docker", "123")
	kubenet.podIPs[existingContainerID] = "10.0.0.1"

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
	kubenet := newFakeKubenetPlugin(map[kubecontainer.ContainerID]string{}, fexec, fhost)
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

func TestGenerateMacAddress(t *testing.T) {
	testCases := []struct {
		ip          net.IP
		expectedMAC string
	}{
		{
			ip:          net.ParseIP("10.0.0.2"),
			expectedMAC: privateMACPrefix + ":0a:00:00:02",
		},
		{
			ip:          net.ParseIP("10.250.0.244"),
			expectedMAC: privateMACPrefix + ":0a:fa:00:f4",
		},
		{
			ip:          net.ParseIP("172.17.0.2"),
			expectedMAC: privateMACPrefix + ":ac:11:00:02",
		},
	}

	for _, tc := range testCases {
		mac, err := generateHardwareAddr(tc.ip)
		if err != nil {
			t.Errorf("Did not expect error: %v", err)
		}
		if mac.String() != tc.expectedMAC {
			t.Errorf("generated mac: %q, expecting: %q", mac.String(), tc.expectedMAC)
		}
	}
}

// TestInvocationWithoutRuntime invokes the plugin without a runtime.
// This is how kubenet is invoked from the cri.
func TestTearDownWithoutRuntime(t *testing.T) {
	testCases := []struct {
		podCIDR         string
		ip              string
		expectedGateway string
	}{
		{
			podCIDR:         "10.0.0.1/24",
			ip:              "10.0.0.1",
			expectedGateway: "10.0.0.1",
		},
		{
			podCIDR:         "2001:beef::1/48",
			ip:              "2001:beef::1",
			expectedGateway: "2001:beef::1",
		},
	}
	for _, tc := range testCases {
		fhost := nettest.NewFakeHost(nil)
		fhost.Legacy = false
		fhost.Runtime = nil
		mockcni := &mock_cni.MockCNI{}

		fexec := &fakeexec.FakeExec{
			CommandScript: []fakeexec.FakeCommandAction{},
			LookPathFunc: func(file string) (string, error) {
				return fmt.Sprintf("/fake-bin/%s", file), nil
			},
		}

		kubenet := newFakeKubenetPlugin(map[kubecontainer.ContainerID]string{}, fexec, fhost)
		kubenet.cniConfig = mockcni
		kubenet.iptables = ipttest.NewFake()

		details := make(map[string]interface{})
		details[network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE_DETAIL_CIDR] = tc.podCIDR
		kubenet.Event(network.NET_PLUGIN_EVENT_POD_CIDR_CHANGE, details)

		if kubenet.gateway.String() != tc.expectedGateway {
			t.Errorf("generated gateway: %q, expecting: %q", kubenet.gateway.String(), tc.expectedGateway)
		}
		if kubenet.podCidr != tc.podCIDR {
			t.Errorf("generated podCidr: %q, expecting: %q", kubenet.podCidr, tc.podCIDR)
		}
		existingContainerID := kubecontainer.BuildContainerID("docker", "123")
		kubenet.podIPs[existingContainerID] = tc.ip

		mockcni.On("DelNetwork", mock.AnythingOfType("*libcni.NetworkConfig"), mock.AnythingOfType("*libcni.RuntimeConf")).Return(nil)

		if err := kubenet.TearDownPod("namespace", "name", existingContainerID); err != nil {
			t.Fatalf("Unexpected error in TearDownPod: %v", err)
		}
		// Assert that the CNI DelNetwork made it through and we didn't crash
		// without a runtime.
		mockcni.AssertExpectations(t)
	}
}

//TODO: add unit test for each implementation of network plugin interface
