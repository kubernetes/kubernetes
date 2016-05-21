/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/util/exec"
	"testing"
)

func newFakeKubenetPlugin(initMap map[kubecontainer.ContainerID]string, execer exec.Interface, host network.Host) network.NetworkPlugin {
	return &kubenetNetworkPlugin{
		podCIDRs: initMap,
		execer:   execer,
		MTU:      1460,
		host:     host,
	}
}

func TestGetPodNetworkStatus(t *testing.T) {
	podIPMap := make(map[kubecontainer.ContainerID]string)
	podIPMap[kubecontainer.ContainerID{ID: "1"}] = "10.245.0.2/32"
	podIPMap[kubecontainer.ContainerID{ID: "2"}] = "10.245.0.3/32"

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

	fakeCmds := make([]exec.FakeCommandAction, 0)
	for _, t := range testCases {
		// the fake commands return the IP from the given index, or an error
		fCmd := exec.FakeCmd{
			CombinedOutputScript: []exec.FakeCombinedOutputAction{
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
			return exec.InitFakeCmd(&fCmd, cmd, args...)
		})
	}
	fexec := exec.FakeExec{
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

//TODO: add unit test for each implementation of network plugin interface
