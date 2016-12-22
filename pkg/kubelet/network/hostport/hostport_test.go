/*
Copyright 2016 The Kubernetes Authors.

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

package hostport

import (
	"fmt"
	"net"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

type fakeSocket struct {
	port     int32
	protocol string
	closed   bool
}

func (f *fakeSocket) Close() error {
	if f.closed {
		return fmt.Errorf("Socket %q.%s already closed!", f.port, f.protocol)
	}
	f.closed = true
	return nil
}

func openFakeSocket(hp *hostport) (closeable, error) {
	return &fakeSocket{hp.port, hp.protocol, false}, nil
}

type ruleMatch struct {
	hostport int
	chain    string
	match    string
}

func TestOpenPodHostports(t *testing.T) {
	fakeIPTables := NewFakeIPTables()

	h := &handler{
		hostPortMap: make(map[hostport]closeable),
		iptables:    fakeIPTables,
		portOpener:  openFakeSocket,
	}

	tests := []struct {
		pod     *v1.Pod
		ip      string
		matches []*ruleMatch
	}{
		// New pod that we are going to add
		{
			&v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name:      "test-pod",
					Namespace: v1.NamespaceDefault,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Ports: []v1.ContainerPort{{
							HostPort:      4567,
							ContainerPort: 80,
							Protocol:      v1.ProtocolTCP,
						}, {
							HostPort:      5678,
							ContainerPort: 81,
							Protocol:      v1.ProtocolUDP,
						}},
					}},
				},
			},
			"10.1.1.2",
			[]*ruleMatch{
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"test-pod_default hostport 4567\" -m tcp -p tcp --dport 4567",
				},
				{
					4567,
					"",
					"-m comment --comment \"test-pod_default hostport 4567\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ",
				},
				{
					4567,
					"",
					"-m comment --comment \"test-pod_default hostport 4567\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:80",
				},
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"test-pod_default hostport 5678\" -m udp -p udp --dport 5678",
				},
				{
					5678,
					"",
					"-m comment --comment \"test-pod_default hostport 5678\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ",
				},
				{
					5678,
					"",
					"-m comment --comment \"test-pod_default hostport 5678\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:81",
				},
			},
		},
		// Already running pod
		{
			&v1.Pod{
				ObjectMeta: v1.ObjectMeta{
					Name:      "another-test-pod",
					Namespace: v1.NamespaceDefault,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{{
						Ports: []v1.ContainerPort{{
							HostPort:      123,
							ContainerPort: 654,
							Protocol:      v1.ProtocolTCP,
						}},
					}},
				},
			},
			"10.1.1.5",
			[]*ruleMatch{
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"another-test-pod_default hostport 123\" -m tcp -p tcp --dport 123",
				},
				{
					123,
					"",
					"-m comment --comment \"another-test-pod_default hostport 123\" -s 10.1.1.5/32 -j KUBE-MARK-MASQ",
				},
				{
					123,
					"",
					"-m comment --comment \"another-test-pod_default hostport 123\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.5:654",
				},
			},
		},
	}

	activePods := make([]*ActivePod, 0)

	// Fill in any match rules missing chain names
	for _, test := range tests {
		for _, match := range test.matches {
			if match.hostport >= 0 {
				found := false
				for _, c := range test.pod.Spec.Containers {
					for _, cp := range c.Ports {
						if int(cp.HostPort) == match.hostport {
							match.chain = string(hostportChainName(cp, kubecontainer.GetPodFullName(test.pod)))
							found = true
							break
						}
					}
				}
				if !found {
					t.Fatalf("Failed to find ContainerPort for match %d/'%s'", match.hostport, match.match)
				}
			}
		}
		activePods = append(activePods, &ActivePod{
			Pod: test.pod,
			IP:  net.ParseIP(test.ip),
		})
	}

	// Already running pod's host port
	hp := hostport{
		tests[1].pod.Spec.Containers[0].Ports[0].HostPort,
		strings.ToLower(string(tests[1].pod.Spec.Containers[0].Ports[0].Protocol)),
	}
	h.hostPortMap[hp] = &fakeSocket{
		tests[1].pod.Spec.Containers[0].Ports[0].HostPort,
		strings.ToLower(string(tests[1].pod.Spec.Containers[0].Ports[0].Protocol)),
		false,
	}

	err := h.OpenPodHostportsAndSync(&ActivePod{Pod: tests[0].pod, IP: net.ParseIP(tests[0].ip)}, "br0", activePods)
	if err != nil {
		t.Fatalf("Failed to OpenPodHostportsAndSync: %v", err)
	}

	// Generic rules
	genericRules := []*ruleMatch{
		{-1, "POSTROUTING", "-m comment --comment \"SNAT for localhost access to hostports\" -o br0 -s 127.0.0.0/8 -j MASQUERADE"},
		{-1, "PREROUTING", "-m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS"},
		{-1, "OUTPUT", "-m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS"},
	}

	for _, rule := range genericRules {
		_, chain, err := fakeIPTables.getChain(utiliptables.TableNAT, utiliptables.Chain(rule.chain))
		if err != nil {
			t.Fatalf("Expected NAT chain %s did not exist", rule.chain)
		}
		if !matchRule(chain, rule.match) {
			t.Fatalf("Expected %s chain rule match '%s' not found", rule.chain, rule.match)
		}
	}

	// Pod rules
	for _, test := range tests {
		for _, match := range test.matches {
			// Ensure chain exists
			_, chain, err := fakeIPTables.getChain(utiliptables.TableNAT, utiliptables.Chain(match.chain))
			if err != nil {
				t.Fatalf("Expected NAT chain %s did not exist", match.chain)
			}
			if !matchRule(chain, match.match) {
				t.Fatalf("Expected NAT chain %s rule containing '%s' not found", match.chain, match.match)
			}
		}
	}

	// Socket
	hostPortMap := map[hostport]closeable{
		hostport{123, "tcp"}:  &fakeSocket{123, "tcp", false},
		hostport{4567, "tcp"}: &fakeSocket{4567, "tcp", false},
		hostport{5678, "udp"}: &fakeSocket{5678, "udp", false},
	}
	if !reflect.DeepEqual(hostPortMap, h.hostPortMap) {
		t.Fatalf("Mismatch in expected hostPortMap. Expected '%v', got '%v'", hostPortMap, h.hostPortMap)
	}
}

func matchRule(chain *fakeChain, match string) bool {
	for _, rule := range chain.rules {
		if strings.Contains(rule, match) {
			return true
		}
	}
	return false
}
