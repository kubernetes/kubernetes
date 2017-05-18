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

package util

import (
	"fmt"
	"net"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/kubernetes/pkg/util/exec"
)

func newFakeServiceInfo(service proxy.ServicePortName, ip net.IP, port int, protocol api.Protocol, onlyNodeLocalEndpoints bool) *ServiceInfo {
	return &ServiceInfo{
		SessionAffinityType:    api.ServiceAffinityNone,
		ClusterIP:              ip,
		Port:                   port,
		Protocol:               protocol,
		OnlyNodeLocalEndpoints: onlyNodeLocalEndpoints,
	}
}

func TestExecConntrackTool(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := [][]string{
		{"-L", "-p", "udp"},
		{"-D", "-p", "udp", "-d", "10.0.240.1"},
		{"-D", "-p", "udp", "--orig-dst", "10.240.0.2", "--dst-nat", "10.0.10.2"},
	}

	expectErr := []bool{false, false, true}

	for i := range testCases {
		err := ExecConntrackTool(&fexec, testCases[i]...)

		if expectErr[i] {
			if err == nil {
				t.Errorf("expected err, got %v", err)
			}
		} else {
			if err != nil {
				t.Errorf("expected success, got %v", err)
			}
		}

		execCmd := strings.Join(fcmd.CombinedOutputLog[i], " ")
		expectCmd := fmt.Sprintf("%s %s", "conntrack", strings.Join(testCases[i], " "))

		if execCmd != expectCmd {
			t.Errorf("expect execute command: %s, but got: %s", expectCmd, execCmd)
		}
	}
}

func TestDeleteServiceConnections(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	testCases := [][]string{
		{
			"10.240.0.3",
			"10.240.0.5",
		},
		{
			"10.240.0.4",
		},
	}

	svcCount := 0
	for i := range testCases {
		DeleteServiceConnections(&fexec, testCases[i])
		for _, ip := range testCases[i] {
			expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s -p udp", ip)
			execCommand := strings.Join(fcmd.CombinedOutputLog[svcCount], " ")
			if expectCommand != execCommand {
				t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
			}
			svcCount += 1
		}
		if svcCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", svcCount, fexec.CommandCalls)
		}
	}
}

func TestDeleteEndpointConnections(t *testing.T) {
	fcmd := exec.FakeCmd{
		CombinedOutputScript: []exec.FakeCombinedOutputAction{
			func() ([]byte, error) { return []byte("1 flow entries have been deleted"), nil },
			func() ([]byte, error) {
				return []byte(""), fmt.Errorf("conntrack v1.4.2 (conntrack-tools): 0 flow entries have been deleted.")
			},
		},
	}
	fexec := exec.FakeExec{
		CommandScript: []exec.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
			func(cmd string, args ...string) exec.Cmd { return exec.InitFakeCmd(&fcmd, cmd, args...) },
		},
		LookPathFunc: func(cmd string) (string, error) { return cmd, nil },
	}

	serviceMap := make(map[proxy.ServicePortName]*ServiceInfo)
	svc1 := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns1", Name: "svc1"}, Port: "p80"}
	svc2 := proxy.ServicePortName{NamespacedName: types.NamespacedName{Namespace: "ns1", Name: "svc2"}, Port: "p80"}
	serviceMap[svc1] = newFakeServiceInfo(svc1, net.IPv4(10, 20, 30, 40), 80, api.ProtocolUDP, false)
	serviceMap[svc2] = newFakeServiceInfo(svc1, net.IPv4(10, 20, 30, 41), 80, api.ProtocolTCP, false)

	testCases := []EndpointServicePair{
		{
			Endpoint:        "10.240.0.3:80",
			ServicePortName: svc1,
		},
		{
			Endpoint:        "10.240.0.4:80",
			ServicePortName: svc1,
		},
		{
			Endpoint:        "10.240.0.5:80",
			ServicePortName: svc2,
		},
	}

	expectCommandExecCount := 0
	for i := range testCases {
		input := map[EndpointServicePair]bool{testCases[i]: true}
		DeleteEndpointConnections(&fexec, serviceMap, input)
		svcInfo := serviceMap[testCases[i].ServicePortName]
		if svcInfo.Protocol == api.ProtocolUDP {
			svcIp := svcInfo.ClusterIP.String()
			EndpointIp := strings.Split(testCases[i].Endpoint, ":")[0]
			expectCommand := fmt.Sprintf("conntrack -D --orig-dst %s --dst-nat %s -p udp", svcIp, EndpointIp)
			execCommand := strings.Join(fcmd.CombinedOutputLog[expectCommandExecCount], " ")
			if expectCommand != execCommand {
				t.Errorf("Exepect comand: %s, but executed %s", expectCommand, execCommand)
			}
			expectCommandExecCount += 1
		}

		if expectCommandExecCount != fexec.CommandCalls {
			t.Errorf("Exepect comand executed %d times, but got %d", expectCommandExecCount, fexec.CommandCalls)
		}
	}
}
