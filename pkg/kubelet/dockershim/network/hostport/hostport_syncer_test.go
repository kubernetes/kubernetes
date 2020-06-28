// +build !dockerless

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
	"net"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
)

type ruleMatch struct {
	hostport int
	chain    string
	match    string
}

func TestOpenPodHostports(t *testing.T) {
	fakeIPTables := NewFakeIPTables()
	fakeOpener := NewFakeSocketManager()

	h := &hostportSyncer{
		hostPortMap: make(map[hostport]closeable),
		iptables:    fakeIPTables,
		portOpener:  fakeOpener.openFakeSocket,
	}

	tests := []struct {
		mapping *PodPortMapping
		matches []*ruleMatch
	}{
		// New pod that we are going to add
		{
			&PodPortMapping{
				Name:        "test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("10.1.1.2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      4567,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      5678,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
				},
			},
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
			&PodPortMapping{
				Name:        "another-test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("10.1.1.5"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      123,
						ContainerPort: 654,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
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
		// IPv6 pod
		{
			&PodPortMapping{
				Name:        "ipv6-test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("2001:dead::5"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      123,
						ContainerPort: 654,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			[]*ruleMatch{},
		},
	}

	activePodPortMapping := make([]*PodPortMapping, 0)

	// Fill in any match rules missing chain names
	for _, test := range tests {
		for _, match := range test.matches {
			if match.hostport >= 0 {
				found := false
				for _, pm := range test.mapping.PortMappings {
					if int(pm.HostPort) == match.hostport {
						match.chain = string(hostportChainName(pm, getPodFullName(test.mapping)))
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("Failed to find ContainerPort for match %d/'%s'", match.hostport, match.match)
				}
			}
		}
		activePodPortMapping = append(activePodPortMapping, test.mapping)

	}

	// Already running pod's host port
	hp := hostport{
		tests[1].mapping.PortMappings[0].HostPort,
		strings.ToLower(string(tests[1].mapping.PortMappings[0].Protocol)),
	}
	h.hostPortMap[hp] = &fakeSocket{
		tests[1].mapping.PortMappings[0].HostPort,
		strings.ToLower(string(tests[1].mapping.PortMappings[0].Protocol)),
		false,
	}

	err := h.OpenPodHostportsAndSync(tests[0].mapping, "br0", activePodPortMapping)
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
		{123, "tcp"}:  &fakeSocket{123, "tcp", false},
		{4567, "tcp"}: &fakeSocket{4567, "tcp", false},
		{5678, "udp"}: &fakeSocket{5678, "udp", false},
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

func TestOpenPodHostportsIPv6(t *testing.T) {
	fakeIPTables := NewFakeIPTables()
	fakeIPTables.protocol = utiliptables.ProtocolIPv6
	fakeOpener := NewFakeSocketManager()

	h := &hostportSyncer{
		hostPortMap: make(map[hostport]closeable),
		iptables:    fakeIPTables,
		portOpener:  fakeOpener.openFakeSocket,
	}

	tests := []struct {
		mapping *PodPortMapping
		matches []*ruleMatch
	}{
		// New pod that we are going to add
		{
			&PodPortMapping{
				Name:        "test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("2001:beef::2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      4567,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      5678,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
				},
			},
			[]*ruleMatch{
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"test-pod_default hostport 4567\" -m tcp -p tcp --dport 4567",
				},
				{
					4567,
					"",
					"-m comment --comment \"test-pod_default hostport 4567\" -s 2001:beef::2/32 -j KUBE-MARK-MASQ",
				},
				{
					4567,
					"",
					"-m comment --comment \"test-pod_default hostport 4567\" -m tcp -p tcp -j DNAT --to-destination [2001:beef::2]:80",
				},
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"test-pod_default hostport 5678\" -m udp -p udp --dport 5678",
				},
				{
					5678,
					"",
					"-m comment --comment \"test-pod_default hostport 5678\" -s 2001:beef::2/32 -j KUBE-MARK-MASQ",
				},
				{
					5678,
					"",
					"-m comment --comment \"test-pod_default hostport 5678\" -m udp -p udp -j DNAT --to-destination [2001:beef::2]:81",
				},
			},
		},
		// Already running pod
		{
			&PodPortMapping{
				Name:        "another-test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("2001:beef::5"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      123,
						ContainerPort: 654,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			[]*ruleMatch{
				{
					-1,
					"KUBE-HOSTPORTS",
					"-m comment --comment \"another-test-pod_default hostport 123\" -m tcp -p tcp --dport 123",
				},
				{
					123,
					"",
					"-m comment --comment \"another-test-pod_default hostport 123\" -s 2001:beef::5/32 -j KUBE-MARK-MASQ",
				},
				{
					123,
					"",
					"-m comment --comment \"another-test-pod_default hostport 123\" -m tcp -p tcp -j DNAT --to-destination [2001:beef::5]:654",
				},
			},
		},
		// IPv4 pod
		{
			&PodPortMapping{
				Name:        "ipv4-test-pod",
				Namespace:   v1.NamespaceDefault,
				IP:          net.ParseIP("192.168.2.5"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      123,
						ContainerPort: 654,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			[]*ruleMatch{},
		},
	}

	activePodPortMapping := make([]*PodPortMapping, 0)

	// Fill in any match rules missing chain names
	for _, test := range tests {
		for _, match := range test.matches {
			if match.hostport >= 0 {
				found := false
				for _, pm := range test.mapping.PortMappings {
					if int(pm.HostPort) == match.hostport {
						match.chain = string(hostportChainName(pm, getPodFullName(test.mapping)))
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("Failed to find ContainerPort for match %d/'%s'", match.hostport, match.match)
				}
			}
		}
		activePodPortMapping = append(activePodPortMapping, test.mapping)

	}

	// Already running pod's host port
	hp := hostport{
		tests[1].mapping.PortMappings[0].HostPort,
		strings.ToLower(string(tests[1].mapping.PortMappings[0].Protocol)),
	}
	h.hostPortMap[hp] = &fakeSocket{
		tests[1].mapping.PortMappings[0].HostPort,
		strings.ToLower(string(tests[1].mapping.PortMappings[0].Protocol)),
		false,
	}

	err := h.OpenPodHostportsAndSync(tests[0].mapping, "br0", activePodPortMapping)
	if err != nil {
		t.Fatalf("Failed to OpenPodHostportsAndSync: %v", err)
	}

	// Generic rules
	genericRules := []*ruleMatch{
		{-1, "POSTROUTING", "-m comment --comment \"SNAT for localhost access to hostports\" -o br0 -s ::1/128 -j MASQUERADE"},
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
		{123, "tcp"}:  &fakeSocket{123, "tcp", false},
		{4567, "tcp"}: &fakeSocket{4567, "tcp", false},
		{5678, "udp"}: &fakeSocket{5678, "udp", false},
	}
	if !reflect.DeepEqual(hostPortMap, h.hostPortMap) {
		t.Fatalf("Mismatch in expected hostPortMap. Expected '%v', got '%v'", hostPortMap, h.hostPortMap)
	}
}

func TestHostportChainName(t *testing.T) {
	m := make(map[string]int)
	chain := hostportChainName(&PortMapping{HostPort: 57119, Protocol: "TCP", ContainerPort: 57119}, "testrdma-2")
	m[string(chain)] = 1
	chain = hostportChainName(&PortMapping{HostPort: 55429, Protocol: "TCP", ContainerPort: 55429}, "testrdma-2")
	m[string(chain)] = 1
	chain = hostportChainName(&PortMapping{HostPort: 56833, Protocol: "TCP", ContainerPort: 56833}, "testrdma-2")
	m[string(chain)] = 1
	if len(m) != 3 {
		t.Fatal(m)
	}
}

func TestHostPortSyncerRemoveLegacyRules(t *testing.T) {
	iptables := NewFakeIPTables()
	legacyRules := [][]string{
		{"-A", "KUBE-HOSTPORTS", "-m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-5N7UH5JAXCVP5UJR"},
		{"-A", "KUBE-HOSTPORTS", "-m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp --dport 8081 -j KUBE-HP-7THKRFSEH4GIIXK7"},
		{"-A", "KUBE-HOSTPORTS", "-m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp --dport 8080 -j KUBE-HP-4YVONL46AKYWSKS3"},
		{"-A", "OUTPUT", "-m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS"},
		{"-A", "PREROUTING", "-m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS"},
		{"-A", "POSTROUTING", "-m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s 127.0.0.0/8 -j MASQUERADE"},
		{"-A", "KUBE-HP-4YVONL46AKYWSKS3", "-m comment --comment \"pod1_ns1 hostport 8080\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ"},
		{"-A", "KUBE-HP-4YVONL46AKYWSKS3", "-m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:80"},
		{"-A", "KUBE-HP-7THKRFSEH4GIIXK7", "-m comment --comment \"pod1_ns1 hostport 8081\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ"},
		{"-A", "KUBE-HP-7THKRFSEH4GIIXK7", "-m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:81"},
		{"-A", "KUBE-HP-5N7UH5JAXCVP5UJR", "-m comment --comment \"pod3_ns1 hostport 8443\" -s 10.1.1.4/32 -j KUBE-MARK-MASQ"},
		{"-A", "KUBE-HP-5N7UH5JAXCVP5UJR", "-m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.4:443"},
	}
	for _, rule := range legacyRules {
		_, err := iptables.EnsureChain(utiliptables.TableNAT, utiliptables.Chain(rule[1]))
		assert.NoError(t, err)
		_, err = iptables.ensureRule(utiliptables.RulePosition(rule[0]), utiliptables.TableNAT, utiliptables.Chain(rule[1]), rule[2])
		assert.NoError(t, err)
	}
	portOpener := NewFakeSocketManager()
	h := &hostportSyncer{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  portOpener.openFakeSocket,
	}
	// check preserve pod3's rules and remove pod1's rules
	pod3PortMapping := &PodPortMapping{
		Name:        "pod3",
		Namespace:   "ns1",
		IP:          net.ParseIP("10.1.1.4"),
		HostNetwork: false,
		PortMappings: []*PortMapping{
			{
				HostPort:      8443,
				ContainerPort: 443,
				Protocol:      v1.ProtocolTCP,
			},
		},
	}
	h.SyncHostports("cbr0", []*PodPortMapping{pod3PortMapping})

	newChainName := string(hostportChainName(pod3PortMapping.PortMappings[0], getPodFullName(pod3PortMapping)))
	expectRules := [][]string{
		{"KUBE-HOSTPORTS", "-m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j " + newChainName},
		{newChainName, "-m comment --comment \"pod3_ns1 hostport 8443\" -s 10.1.1.4/32 -j KUBE-MARK-MASQ"},
		{newChainName, "-m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.4:443"},
	}

	natTable, ok := iptables.tables[string(utiliptables.TableNAT)]
	assert.True(t, ok)
	// check pod1's rules in KUBE-HOSTPORTS chain should be cleaned up
	hostportChain, ok := natTable.chains["KUBE-HOSTPORTS"]
	assert.True(t, ok, string(hostportChain.name))
	assert.Equal(t, 1, len(hostportChain.rules), "%v", hostportChain.rules)

	// check pod3's rules left
	assert.Equal(t, expectRules[0][1], hostportChain.rules[0])
	chain, ok := natTable.chains[newChainName]
	assert.True(t, ok)
	assert.Equal(t, 2, len(chain.rules))
	assert.Equal(t, expectRules[1][1], chain.rules[0])
	assert.Equal(t, expectRules[2][1], chain.rules[1])

	// check legacy KUBE-HP-* chains should be deleted
	for _, name := range []string{"KUBE-HP-4YVONL46AKYWSKS3", "KUBE-HP-7THKRFSEH4GIIXK7", "KUBE-HP-5N7UH5JAXCVP5UJR"} {
		_, ok := natTable.chains[name]
		assert.False(t, ok)
	}
}
