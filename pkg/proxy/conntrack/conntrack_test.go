/*
Copyright 2023 The Kubernetes Authors.

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

package conntrack

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/proxy"
	"k8s.io/utils/exec"
	fakeexec "k8s.io/utils/exec/testing"
	"strings"
	"testing"
)

const TestNamespace = "test-namespace"

// newServiceInfo returns a new proxy.ServicePort which abstracts a serviceInfo.
func newServiceInfo(port *v1.ServicePort, service *v1.Service, baseInfo *proxy.BaseServicePortInfo) proxy.ServicePort {
	return baseInfo
}

// makeService returns v1.Service object for the given parameters.
func makeService(name, clusterIP, lbIP, extIP string, protocol v1.Protocol, port, targetPort, nodePort int) *v1.Service {
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   TestNamespace,
			Annotations: map[string]string{},
		},
		Spec: v1.ServiceSpec{
			ClusterIP: clusterIP,
			Ports: []v1.ServicePort{
				{
					Name:       "test-port",
					Protocol:   protocol,
					Port:       int32(port),
					TargetPort: intstr.FromInt(targetPort),
					NodePort:   int32(nodePort),
				},
			},
		},
		Status: v1.ServiceStatus{},
	}

	if extIP != "" {
		svc.Spec.ExternalIPs = []string{extIP}
	}

	if lbIP != "" {
		svc.Status.LoadBalancer.Ingress = []v1.LoadBalancerIngress{{IP: lbIP}}
	}

	return svc
}

func TestConntrack_CleanUpIPv4(t *testing.T) {

	svcs := []*v1.Service{
		makeService("svc0", "10.96.10.10", "", "", v1.ProtocolTCP,
			80, 5000, 0),
		makeService("svc1", "10.96.10.11", "", "", v1.ProtocolTCP,
			81, 5001, 0),
		makeService("svc2", "10.96.10.12", "", "", v1.ProtocolTCP,
			82, 5002, 32002),
		makeService("svc3", "10.96.10.13", "", "", v1.ProtocolUDP,
			83, 5003, 0),
		makeService("svc4", "10.96.10.14", "", "", v1.ProtocolUDP,
			84, 5004, 32004),
		makeService("svc5", "10.96.10.15", "2.3.4.5", "", v1.ProtocolUDP,
			85, 5005, 0),
		makeService("svc6", "10.96.10.16", "", "8.7.6.5", v1.ProtocolUDP,
			86, 5006, 0),
		makeService("svc7", "10.96.10.17", "", "", v1.ProtocolSCTP,
			87, 5007, 0),
		makeService("svc8", "10.96.10.18", "", "", v1.ProtocolSCTP,
			87, 5008, 32008),
		makeService("svc9", "10.96.10.19", "2.3.4.9", "", v1.ProtocolSCTP,
			89, 5009, 0),
	}

	svcPortMap := make(proxy.ServicePortMap)
	//endpointMap := make(proxy.EndpointsMap)

	serviceChangeTracker := proxy.NewServiceChangeTracker(newServiceInfo, v1.IPv4Protocol, events.NewFakeRecorder(100), nil)
	//endpointChangeTracker := proxy.NewEndpointChangeTracker("test-host", newEndpointInfo, v1.IPv4Protocol, events.NewFakeRecorder(100), nil)

	// add all services to tracker
	for _, svc := range svcs {
		serviceChangeTracker.Update(nil, svc)
	}
	// update servicePortMap
	svcPortMap.Update(serviceChangeTracker)

	serviceUpdateResult := proxy.UpdateServiceMapResult{
		UDPStaleClusterIP: make(sets.String).Insert("10.96.10.13", "10.96.10.14"),
	}

	endpointUpdateResult := proxy.UpdateEndpointMapResult{
		StaleEndpoints: []proxy.ServiceEndpoint{
			{
				Endpoint: "10.1.1.10",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc1"},
					Port:           "test-port",
					Protocol:       v1.ProtocolTCP,
				},
			},
			{
				Endpoint: "10.1.1.13",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc3"},
					Port:           "test-port",
					Protocol:       v1.ProtocolUDP,
				},
			},
			{
				Endpoint: "10.1.1.17",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc7"},
					Port:           "test-port",
					Protocol:       v1.ProtocolSCTP,
				},
			},
			{
				Endpoint: "10.1.1.18",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc8"},
					Port:           "test-port",
					Protocol:       v1.ProtocolSCTP,
				},
			},
		},
		StaleServiceNames: []proxy.ServicePortName{
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc2"},
				Port:           "test-port",
				Protocol:       v1.ProtocolTCP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc4"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc5"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc6"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc9"},
				Port:           "test-port",
				Protocol:       v1.ProtocolSCTP,
			},
		},
	}

	// setup up FakeExec
	cmdFunc := func() ([]byte, []byte, error) { return nil, nil, nil }
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: make([]fakeexec.FakeAction, 0),
	}

	execFunc := func(cmd string, args ...string) exec.Cmd {
		return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
	}
	fexec := &fakeexec.FakeExec{
		CommandScript: make([]fakeexec.FakeCommandAction, 0),
		LookPathFunc:  func(cmd string) (string, error) { return cmd, nil },
	}
	for i := 0; i < 100; i++ {
		fcmd.CombinedOutputScript = append(fcmd.CombinedOutputScript, cmdFunc)
		fexec.CommandScript = append(fexec.CommandScript, execFunc)
	}

	// setup up Conntrack
	ct := NewConntrack(fexec)
	ct.CleanServiceConnections(svcPortMap, serviceUpdateResult, endpointUpdateResult)
	ct.CleanEndpointConnections(svcPortMap, endpointUpdateResult)

	executedCommands := make(sets.Set[string], 0)
	for i := 0; i < len(fcmd.CombinedOutputLog); i++ {
		executedCommands.Insert(strings.Join(fcmd.CombinedOutputLog[i], " "))
	}

	expectedCommands := make(sets.Set[string])
	expectedCommands.Insert(
		"conntrack -D --orig-dst 10.96.10.13 -p udp",
		"conntrack -D --orig-dst 10.96.10.14 -p udp",
		"conntrack -D --orig-dst 10.96.10.15 -p udp",
		"conntrack -D --orig-dst 10.96.10.16 -p udp",
		"conntrack -D --orig-dst 10.96.10.19 -p udp",

		"conntrack -D --orig-dst 2.3.4.5 -p udp",
		"conntrack -D --orig-dst 2.3.4.9 -p udp",

		"conntrack -D --orig-dst 8.7.6.5 -p udp",

		"conntrack -D -p udp --dport 32004",

		"conntrack -D -p sctp --dport 32008 --dst-nat 10.1.1.18",

		"conntrack -D --orig-dst 10.96.10.13 --dst-nat 10.1.1.13 -p udp",
		"conntrack -D --orig-dst 10.96.10.17 --dst-nat 10.1.1.17 -p sctp",
		"conntrack -D --orig-dst 10.96.10.18 --dst-nat 10.1.1.18 -p sctp",
	)

	for _, cmd := range expectedCommands.Difference(executedCommands).UnsortedList() {
		t.Errorf("conntrack command '%s' expected but not executed", cmd)
	}

	for _, cmd := range executedCommands.Difference(expectedCommands).UnsortedList() {
		t.Errorf("conntrack command '%s' not expecetd but executed", cmd)
	}
}

func TestConntrack_CleanUpIPv6(t *testing.T) {
	// setup up Conntrack and FakeExec
	svcs := []*v1.Service{
		makeService("svc0", "fd00:1230::20", "", "", v1.ProtocolTCP,
			80, 5000, 0),
		makeService("svc1", "fd00:1231::20", "", "", v1.ProtocolTCP,
			81, 5001, 0),
		makeService("svc2", "fd00:1232::20", "", "", v1.ProtocolTCP,
			82, 5002, 32002),
		makeService("svc3", "fd00:1233::20", "", "", v1.ProtocolUDP,
			83, 5003, 0),
		makeService("svc4", "fd00:1234::20", "", "", v1.ProtocolUDP,
			84, 5004, 32004),
		makeService("svc5", "fd00:1235::20", "fd00:7775::20", "", v1.ProtocolUDP,
			85, 5005, 0),
		makeService("svc6", "fd00:1236::20", "", "fd00:4446::20", v1.ProtocolUDP,
			86, 5006, 0),
		makeService("svc7", "fd00:1237::20", "", "", v1.ProtocolSCTP,
			87, 5007, 0),
		makeService("svc8", "fd00:1238::20", "", "", v1.ProtocolSCTP,
			87, 5008, 32008),
		makeService("svc9", "fd00:1239::20", "fd00:7779::20", "", v1.ProtocolSCTP,
			89, 5009, 0),
	}

	svcPortMap := make(proxy.ServicePortMap)
	//endpointMap := make(proxy.EndpointsMap)

	serviceChangeTracker := proxy.NewServiceChangeTracker(newServiceInfo, v1.IPv6Protocol, events.NewFakeRecorder(100), nil)
	//endpointChangeTracker := proxy.NewEndpointChangeTracker("test-host", newEndpointInfo, v1.IPv4Protocol, events.NewFakeRecorder(100), nil)

	for _, svc := range svcs {
		serviceChangeTracker.Update(nil, svc)
	}
	svcPortMap.Update(serviceChangeTracker)

	serviceUpdateResult := proxy.UpdateServiceMapResult{
		UDPStaleClusterIP: sets.String{
			"fd00:1233::20": sets.Empty{},
			"fd00:1234::20": sets.Empty{},
		},
	}

	endpointUpdateResult := proxy.UpdateEndpointMapResult{
		StaleEndpoints: []proxy.ServiceEndpoint{
			{
				Endpoint: "fd00:3210::20",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc1"},
					Port:           "test-port",
					Protocol:       v1.ProtocolTCP,
				},
			},
			{
				Endpoint: "fd00:3213::20",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc3"},
					Port:           "test-port",
					Protocol:       v1.ProtocolUDP,
				},
			},
			{
				Endpoint: "fd00:3217::20",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc7"},
					Port:           "test-port",
					Protocol:       v1.ProtocolSCTP,
				},
			},
			{
				Endpoint: "fd00:3218::20",
				ServicePortName: proxy.ServicePortName{
					NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc8"},
					Port:           "test-port",
					Protocol:       v1.ProtocolSCTP,
				},
			},
		},
		StaleServiceNames: []proxy.ServicePortName{
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc2"},
				Port:           "test-port",
				Protocol:       v1.ProtocolTCP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc4"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc5"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc6"},
				Port:           "test-port",
				Protocol:       v1.ProtocolUDP,
			},
			{
				NamespacedName: types.NamespacedName{Namespace: TestNamespace, Name: "svc9"},
				Port:           "test-port",
				Protocol:       v1.ProtocolSCTP,
			},
		},
	}

	// setup up FakeExec
	cmdFunc := func() ([]byte, []byte, error) { return nil, nil, nil }
	fcmd := fakeexec.FakeCmd{
		CombinedOutputScript: make([]fakeexec.FakeAction, 0),
	}

	execFunc := func(cmd string, args ...string) exec.Cmd {
		return fakeexec.InitFakeCmd(&fcmd, cmd, args...)
	}
	fexec := &fakeexec.FakeExec{
		CommandScript: make([]fakeexec.FakeCommandAction, 0),
		LookPathFunc:  func(cmd string) (string, error) { return cmd, nil },
	}
	for i := 0; i < 100; i++ {
		fcmd.CombinedOutputScript = append(fcmd.CombinedOutputScript, cmdFunc)
		fexec.CommandScript = append(fexec.CommandScript, execFunc)
	}

	ct := NewConntrack(fexec)
	ct.CleanServiceConnections(svcPortMap, serviceUpdateResult, endpointUpdateResult)
	ct.CleanEndpointConnections(svcPortMap, endpointUpdateResult)

	executedCommands := make(sets.Set[string], 0)
	for i := 0; i < len(fcmd.CombinedOutputLog); i++ {
		executedCommands.Insert(strings.Join(fcmd.CombinedOutputLog[i], " "))
	}

	expectedCommands := make(sets.Set[string])
	expectedCommands.Insert(
		"conntrack -D --orig-dst fd00:1233::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:1234::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:1235::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:1236::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:1239::20 -p udp -f ipv6",

		"conntrack -D --orig-dst fd00:7775::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:7779::20 -p udp -f ipv6",

		"conntrack -D --orig-dst fd00:4446::20 -p udp -f ipv6",

		"conntrack -D -p sctp --dport 32008 --dst-nat fd00:3218::20 -f ipv6",

		"conntrack -D -p udp --dport 32004 -f ipv6",

		"conntrack -D --orig-dst fd00:1233::20 --dst-nat fd00:3213::20 -p udp -f ipv6",
		"conntrack -D --orig-dst fd00:1237::20 --dst-nat fd00:3217::20 -p sctp -f ipv6",
		"conntrack -D --orig-dst fd00:1238::20 --dst-nat fd00:3218::20 -p sctp -f ipv6",
	)

	for _, cmd := range expectedCommands.Difference(executedCommands).UnsortedList() {
		t.Errorf("conntrack command '%s' expected but not executed", cmd)
	}

	for _, cmd := range executedCommands.Difference(expectedCommands).UnsortedList() {
		t.Errorf("conntrack command '%s' not expecetd but executed", cmd)
	}
}
