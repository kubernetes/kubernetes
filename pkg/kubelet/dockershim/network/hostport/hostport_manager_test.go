//go:build !dockerless
// +build !dockerless

/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"k8s.io/utils/exec"
	netutils "k8s.io/utils/net"
)

func TestOpenCloseHostports(t *testing.T) {
	openPortCases := []struct {
		podPortMapping *PodPortMapping
		expectError    bool
	}{
		// no portmaps
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n0",
			},
			false,
		},
		// allocate port 80/TCP, 8080/TCP and 443/TCP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n1",
				PortMappings: []*PortMapping{
					{HostPort: 80, Protocol: v1.ProtocolTCP},
					{HostPort: 8080, Protocol: v1.ProtocolTCP},
					{HostPort: 443, Protocol: v1.ProtocolTCP},
				},
			},
			false,
		},
		// fail to allocate port previously allocated 80/TCP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n2",
				PortMappings: []*PortMapping{
					{HostPort: 80, Protocol: v1.ProtocolTCP},
				},
			},
			true,
		},
		// fail to allocate port previously allocated 8080/TCP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n3",
				PortMappings: []*PortMapping{
					{HostPort: 8081, Protocol: v1.ProtocolTCP},
					{HostPort: 8080, Protocol: v1.ProtocolTCP},
				},
			},
			true,
		},
		// allocate port 8081/TCP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n3",
				PortMappings: []*PortMapping{
					{HostPort: 8081, Protocol: v1.ProtocolTCP},
				},
			},
			false,
		},
		// allocate port 7777/SCTP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n4",
				PortMappings: []*PortMapping{
					{HostPort: 7777, Protocol: v1.ProtocolSCTP},
				},
			},
			false,
		},
		// same HostPort different HostIP
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n5",
				PortMappings: []*PortMapping{
					{HostPort: 8888, Protocol: v1.ProtocolUDP, HostIP: "127.0.0.1"},
					{HostPort: 8888, Protocol: v1.ProtocolUDP, HostIP: "127.0.0.2"},
				},
			},
			false,
		},
		// same HostPort different protocol
		{
			&PodPortMapping{
				Namespace: "ns1",
				Name:      "n6",
				PortMappings: []*PortMapping{
					{HostPort: 9999, Protocol: v1.ProtocolTCP},
					{HostPort: 9999, Protocol: v1.ProtocolUDP},
				},
			},
			false,
		},
	}

	iptables := NewFakeIPTables()
	iptables.protocol = utiliptables.ProtocolIPv4
	portOpener := NewFakeSocketManager()
	manager := &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  portOpener.openFakeSocket,
		execer:      exec.New(),
	}

	// open all hostports defined in the test cases
	for _, tc := range openPortCases {
		mapping, err := manager.openHostports(tc.podPortMapping)
		for hostport, socket := range mapping {
			manager.hostPortMap[hostport] = socket
		}
		if tc.expectError {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
		// SCTP ports are not allocated
		countSctp := 0
		for _, pm := range tc.podPortMapping.PortMappings {
			if pm.Protocol == v1.ProtocolSCTP {
				countSctp++
			}
		}
		assert.EqualValues(t, len(mapping), len(tc.podPortMapping.PortMappings)-countSctp)
	}

	// We have following ports open: 80/TCP, 443/TCP, 8080/TCP, 8081/TCP,
	// 127.0.0.1:8888/TCP, 127.0.0.2:8888/TCP, 9999/TCP and 9999/UDP open now.
	assert.EqualValues(t, len(manager.hostPortMap), 8)
	closePortCases := []struct {
		portMappings []*PortMapping
		expectError  bool
	}{
		{
			portMappings: nil,
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 80, Protocol: v1.ProtocolTCP},
				{HostPort: 8080, Protocol: v1.ProtocolTCP},
				{HostPort: 443, Protocol: v1.ProtocolTCP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 80, Protocol: v1.ProtocolTCP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 8081, Protocol: v1.ProtocolTCP},
				{HostPort: 8080, Protocol: v1.ProtocolTCP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 8081, Protocol: v1.ProtocolTCP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 7070, Protocol: v1.ProtocolTCP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 7777, Protocol: v1.ProtocolSCTP},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 8888, Protocol: v1.ProtocolUDP, HostIP: "127.0.0.1"},
				{HostPort: 8888, Protocol: v1.ProtocolUDP, HostIP: "127.0.0.2"},
			},
		},
		{
			portMappings: []*PortMapping{
				{HostPort: 9999, Protocol: v1.ProtocolTCP},
				{HostPort: 9999, Protocol: v1.ProtocolUDP},
			},
		},
	}

	// close all the hostports opened in previous step
	for _, tc := range closePortCases {
		err := manager.closeHostports(tc.portMappings)
		if tc.expectError {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
	}
	// assert all elements in hostPortMap were cleared
	assert.Zero(t, len(manager.hostPortMap))
}

func TestHostportManager(t *testing.T) {
	iptables := NewFakeIPTables()
	iptables.protocol = utiliptables.ProtocolIPv4
	portOpener := NewFakeSocketManager()
	manager := &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  portOpener.openFakeSocket,
		execer:      exec.New(),
	}
	testCases := []struct {
		mapping     *PodPortMapping
		expectError bool
	}{
		// open HostPorts 8080/TCP, 8081/UDP and 8083/SCTP
		{
			mapping: &PodPortMapping{
				Name:        "pod1",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("10.1.1.2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8080,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      8081,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
					{
						HostPort:      8083,
						ContainerPort: 83,
						Protocol:      v1.ProtocolSCTP,
					},
				},
			},
			expectError: false,
		},
		// fail to open HostPort due to conflict 8083/SCTP
		{
			mapping: &PodPortMapping{
				Name:        "pod2",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("10.1.1.3"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8082,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      8081,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
					{
						HostPort:      8083,
						ContainerPort: 83,
						Protocol:      v1.ProtocolSCTP,
					},
				},
			},
			expectError: true,
		},
		// open port 443
		{
			mapping: &PodPortMapping{
				Name:        "pod3",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("10.1.1.4"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8443,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			expectError: false,
		},
		// fail to open HostPort 8443 already allocated
		{
			mapping: &PodPortMapping{
				Name:        "pod3",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("192.168.12.12"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8443,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			expectError: true,
		},
		// skip HostPort with PodIP and HostIP using different families
		{
			mapping: &PodPortMapping{
				Name:        "pod4",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("2001:beef::2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8444,
						ContainerPort: 444,
						Protocol:      v1.ProtocolTCP,
						HostIP:        "192.168.1.1",
					},
				},
			},
			expectError: false,
		},

		// open same HostPort on different IP
		{
			mapping: &PodPortMapping{
				Name:        "pod5",
				Namespace:   "ns5",
				IP:          netutils.ParseIPSloppy("10.1.1.5"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8888,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
						HostIP:        "127.0.0.2",
					},
					{
						HostPort:      8888,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
						HostIP:        "127.0.0.1",
					},
				},
			},
			expectError: false,
		},
		// open same HostPort on different
		{
			mapping: &PodPortMapping{
				Name:        "pod6",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("10.1.1.2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      9999,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      9999,
						ContainerPort: 443,
						Protocol:      v1.ProtocolUDP,
					},
				},
			},
			expectError: false,
		},
	}

	// Add Hostports
	for _, tc := range testCases {
		err := manager.Add("id", tc.mapping, "cbr0")
		if tc.expectError {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
	}

	// Check port opened
	expectedPorts := []hostport{
		{IPv4, "", 8080, "tcp"},
		{IPv4, "", 8081, "udp"},
		{IPv4, "", 8443, "tcp"},
		{IPv4, "127.0.0.1", 8888, "tcp"},
		{IPv4, "127.0.0.2", 8888, "tcp"},
		{IPv4, "", 9999, "tcp"},
		{IPv4, "", 9999, "udp"},
	}
	openedPorts := make(map[hostport]bool)
	for hp, port := range portOpener.mem {
		if !port.closed {
			openedPorts[hp] = true
		}
	}
	assert.EqualValues(t, len(openedPorts), len(expectedPorts))
	for _, hp := range expectedPorts {
		_, ok := openedPorts[hp]
		assert.EqualValues(t, true, ok)
	}

	// Check Iptables-save result after adding hostports
	raw := bytes.NewBuffer(nil)
	err := iptables.SaveInto(utiliptables.TableNAT, raw)
	assert.NoError(t, err)

	lines := strings.Split(raw.String(), "\n")
	expectedLines := map[string]bool{
		`*nat`:                              true,
		`:KUBE-HOSTPORTS - [0:0]`:           true,
		`:OUTPUT - [0:0]`:                   true,
		`:PREROUTING - [0:0]`:               true,
		`:POSTROUTING - [0:0]`:              true,
		`:KUBE-HP-IJHALPHTORMHHPPK - [0:0]`: true,
		`:KUBE-HP-63UPIDJXVRSZGSUZ - [0:0]`: true,
		`:KUBE-HP-WFBOALXEP42XEMJK - [0:0]`: true,
		`:KUBE-HP-XU6AWMMJYOZOFTFZ - [0:0]`: true,
		`:KUBE-HP-TUKTZ736U5JD5UTK - [0:0]`: true,
		`:KUBE-HP-CAAJ45HDITK7ARGM - [0:0]`: true,
		`:KUBE-HP-WFUNFVXVDLD5ZVXN - [0:0]`: true,
		`:KUBE-HP-4MFWH2F2NAOMYD6A - [0:0]`: true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-WFBOALXEP42XEMJK":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp --dport 8081 -j KUBE-HP-63UPIDJXVRSZGSUZ":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp --dport 8080 -j KUBE-HP-IJHALPHTORMHHPPK":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8083\" -m sctp -p sctp --dport 8083 -j KUBE-HP-XU6AWMMJYOZOFTFZ":                      true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod5_ns5 hostport 8888\" -m tcp -p tcp --dport 8888 -j KUBE-HP-TUKTZ736U5JD5UTK":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod5_ns5 hostport 8888\" -m tcp -p tcp --dport 8888 -j KUBE-HP-CAAJ45HDITK7ARGM":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod6_ns1 hostport 9999\" -m udp -p udp --dport 9999 -j KUBE-HP-4MFWH2F2NAOMYD6A":                        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod6_ns1 hostport 9999\" -m tcp -p tcp --dport 9999 -j KUBE-HP-WFUNFVXVDLD5ZVXN":                        true,
		"-A OUTPUT -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                                         true,
		"-A PREROUTING -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                                     true,
		"-A POSTROUTING -m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s 127.0.0.0/8 -j MASQUERADE":                             true,
		"-A KUBE-HP-IJHALPHTORMHHPPK -m comment --comment \"pod1_ns1 hostport 8080\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-IJHALPHTORMHHPPK -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:80":                  true,
		"-A KUBE-HP-63UPIDJXVRSZGSUZ -m comment --comment \"pod1_ns1 hostport 8081\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-63UPIDJXVRSZGSUZ -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:81":                  true,
		"-A KUBE-HP-XU6AWMMJYOZOFTFZ -m comment --comment \"pod1_ns1 hostport 8083\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-XU6AWMMJYOZOFTFZ -m comment --comment \"pod1_ns1 hostport 8083\" -m sctp -p sctp -j DNAT --to-destination 10.1.1.2:83":                true,
		"-A KUBE-HP-WFBOALXEP42XEMJK -m comment --comment \"pod3_ns1 hostport 8443\" -s 10.1.1.4/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-WFBOALXEP42XEMJK -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.4:443":                 true,
		"-A KUBE-HP-TUKTZ736U5JD5UTK -m comment --comment \"pod5_ns5 hostport 8888\" -s 10.1.1.5/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-TUKTZ736U5JD5UTK -m comment --comment \"pod5_ns5 hostport 8888\" -m tcp -p tcp -d 127.0.0.1/32 -j DNAT --to-destination 10.1.1.5:443": true,
		"-A KUBE-HP-CAAJ45HDITK7ARGM -m comment --comment \"pod5_ns5 hostport 8888\" -s 10.1.1.5/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-CAAJ45HDITK7ARGM -m comment --comment \"pod5_ns5 hostport 8888\" -m tcp -p tcp -d 127.0.0.2/32 -j DNAT --to-destination 10.1.1.5:443": true,
		"-A KUBE-HP-WFUNFVXVDLD5ZVXN -m comment --comment \"pod6_ns1 hostport 9999\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-WFUNFVXVDLD5ZVXN -m comment --comment \"pod6_ns1 hostport 9999\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:443":                 true,
		"-A KUBE-HP-4MFWH2F2NAOMYD6A -m comment --comment \"pod6_ns1 hostport 9999\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                                    true,
		"-A KUBE-HP-4MFWH2F2NAOMYD6A -m comment --comment \"pod6_ns1 hostport 9999\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:443":                 true,
		`COMMIT`: true,
	}
	for _, line := range lines {
		t.Logf("Line: %s", line)
		if len(strings.TrimSpace(line)) > 0 {
			_, ok := expectedLines[strings.TrimSpace(line)]
			assert.EqualValues(t, true, ok)
		}
	}

	// Remove all added hostports
	for _, tc := range testCases {
		if !tc.expectError {
			err := manager.Remove("id", tc.mapping)
			assert.NoError(t, err)
		}
	}

	// Check Iptables-save result after deleting hostports
	raw.Reset()
	err = iptables.SaveInto(utiliptables.TableNAT, raw)
	assert.NoError(t, err)
	lines = strings.Split(raw.String(), "\n")
	remainingChains := make(map[string]bool)
	for _, line := range lines {
		if strings.HasPrefix(line, ":") {
			remainingChains[strings.TrimSpace(line)] = true
		}
	}
	expectDeletedChains := []string{
		"KUBE-HP-4YVONL46AKYWSKS3", "KUBE-HP-7THKRFSEH4GIIXK7", "KUBE-HP-5N7UH5JAXCVP5UJR",
		"KUBE-HP-TUKTZ736U5JD5UTK", "KUBE-HP-CAAJ45HDITK7ARGM", "KUBE-HP-WFUNFVXVDLD5ZVXN", "KUBE-HP-4MFWH2F2NAOMYD6A",
	}
	for _, chain := range expectDeletedChains {
		_, ok := remainingChains[chain]
		assert.EqualValues(t, false, ok)
	}

	// check if all ports are closed
	for _, port := range portOpener.mem {
		assert.EqualValues(t, true, port.closed)
	}
	// Clear all elements in hostPortMap
	assert.Zero(t, len(manager.hostPortMap))
}

func TestGetHostportChain(t *testing.T) {
	m := make(map[string]int)
	chain := getHostportChain("testrdma-2", &PortMapping{HostPort: 57119, Protocol: "TCP", ContainerPort: 57119})
	m[string(chain)] = 1
	chain = getHostportChain("testrdma-2", &PortMapping{HostPort: 55429, Protocol: "TCP", ContainerPort: 55429})
	m[string(chain)] = 1
	chain = getHostportChain("testrdma-2", &PortMapping{HostPort: 56833, Protocol: "TCP", ContainerPort: 56833})
	m[string(chain)] = 1
	if len(m) != 3 {
		t.Fatal(m)
	}
}

func TestHostportManagerIPv6(t *testing.T) {
	iptables := NewFakeIPTables()
	iptables.protocol = utiliptables.ProtocolIPv6
	portOpener := NewFakeSocketManager()
	manager := &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  portOpener.openFakeSocket,
		execer:      exec.New(),
	}
	testCases := []struct {
		mapping     *PodPortMapping
		expectError bool
	}{
		{
			mapping: &PodPortMapping{
				Name:        "pod1",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("2001:beef::2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8080,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      8081,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
					{
						HostPort:      8083,
						ContainerPort: 83,
						Protocol:      v1.ProtocolSCTP,
					},
				},
			},
			expectError: false,
		},
		{
			mapping: &PodPortMapping{
				Name:        "pod2",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("2001:beef::3"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8082,
						ContainerPort: 80,
						Protocol:      v1.ProtocolTCP,
					},
					{
						HostPort:      8081,
						ContainerPort: 81,
						Protocol:      v1.ProtocolUDP,
					},
					{
						HostPort:      8083,
						ContainerPort: 83,
						Protocol:      v1.ProtocolSCTP,
					},
				},
			},
			expectError: true,
		},
		{
			mapping: &PodPortMapping{
				Name:        "pod3",
				Namespace:   "ns1",
				IP:          netutils.ParseIPSloppy("2001:beef::4"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8443,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			expectError: false,
		},
		{
			mapping: &PodPortMapping{
				Name:        "pod4",
				Namespace:   "ns2",
				IP:          netutils.ParseIPSloppy("192.168.2.2"),
				HostNetwork: false,
				PortMappings: []*PortMapping{
					{
						HostPort:      8443,
						ContainerPort: 443,
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			expectError: true,
		},
	}

	// Add Hostports
	for _, tc := range testCases {
		err := manager.Add("id", tc.mapping, "cbr0")
		if tc.expectError {
			assert.Error(t, err)
			continue
		}
		assert.NoError(t, err)
	}

	// Check port opened
	expectedPorts := []hostport{{IPv6, "", 8080, "tcp"}, {IPv6, "", 8081, "udp"}, {IPv6, "", 8443, "tcp"}}
	openedPorts := make(map[hostport]bool)
	for hp, port := range portOpener.mem {
		if !port.closed {
			openedPorts[hp] = true
		}
	}
	assert.EqualValues(t, len(openedPorts), len(expectedPorts))
	for _, hp := range expectedPorts {
		_, ok := openedPorts[hp]
		assert.EqualValues(t, true, ok)
	}

	// Check Iptables-save result after adding hostports
	raw := bytes.NewBuffer(nil)
	err := iptables.SaveInto(utiliptables.TableNAT, raw)
	assert.NoError(t, err)

	lines := strings.Split(raw.String(), "\n")
	expectedLines := map[string]bool{
		`*nat`:                              true,
		`:KUBE-HOSTPORTS - [0:0]`:           true,
		`:OUTPUT - [0:0]`:                   true,
		`:PREROUTING - [0:0]`:               true,
		`:POSTROUTING - [0:0]`:              true,
		`:KUBE-HP-IJHALPHTORMHHPPK - [0:0]`: true,
		`:KUBE-HP-63UPIDJXVRSZGSUZ - [0:0]`: true,
		`:KUBE-HP-WFBOALXEP42XEMJK - [0:0]`: true,
		`:KUBE-HP-XU6AWMMJYOZOFTFZ - [0:0]`: true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-WFBOALXEP42XEMJK":               true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp --dport 8081 -j KUBE-HP-63UPIDJXVRSZGSUZ":               true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp --dport 8080 -j KUBE-HP-IJHALPHTORMHHPPK":               true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8083\" -m sctp -p sctp --dport 8083 -j KUBE-HP-XU6AWMMJYOZOFTFZ":             true,
		"-A OUTPUT -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                                true,
		"-A PREROUTING -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                            true,
		"-A POSTROUTING -m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s ::1/128 -j MASQUERADE":                        true,
		"-A KUBE-HP-IJHALPHTORMHHPPK -m comment --comment \"pod1_ns1 hostport 8080\" -s 2001:beef::2/32 -j KUBE-MARK-MASQ":                       true,
		"-A KUBE-HP-IJHALPHTORMHHPPK -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp -j DNAT --to-destination [2001:beef::2]:80":   true,
		"-A KUBE-HP-63UPIDJXVRSZGSUZ -m comment --comment \"pod1_ns1 hostport 8081\" -s 2001:beef::2/32 -j KUBE-MARK-MASQ":                       true,
		"-A KUBE-HP-63UPIDJXVRSZGSUZ -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp -j DNAT --to-destination [2001:beef::2]:81":   true,
		"-A KUBE-HP-XU6AWMMJYOZOFTFZ -m comment --comment \"pod1_ns1 hostport 8083\" -s 2001:beef::2/32 -j KUBE-MARK-MASQ":                       true,
		"-A KUBE-HP-XU6AWMMJYOZOFTFZ -m comment --comment \"pod1_ns1 hostport 8083\" -m sctp -p sctp -j DNAT --to-destination [2001:beef::2]:83": true,
		"-A KUBE-HP-WFBOALXEP42XEMJK -m comment --comment \"pod3_ns1 hostport 8443\" -s 2001:beef::4/32 -j KUBE-MARK-MASQ":                       true,
		"-A KUBE-HP-WFBOALXEP42XEMJK -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination [2001:beef::4]:443":  true,
		`COMMIT`: true,
	}
	for _, line := range lines {
		if len(strings.TrimSpace(line)) > 0 {
			_, ok := expectedLines[strings.TrimSpace(line)]
			assert.EqualValues(t, true, ok)
		}
	}

	// Remove all added hostports
	for _, tc := range testCases {
		if !tc.expectError {
			err := manager.Remove("id", tc.mapping)
			assert.NoError(t, err)
		}
	}

	// Check Iptables-save result after deleting hostports
	raw.Reset()
	err = iptables.SaveInto(utiliptables.TableNAT, raw)
	assert.NoError(t, err)
	lines = strings.Split(raw.String(), "\n")
	remainingChains := make(map[string]bool)
	for _, line := range lines {
		if strings.HasPrefix(line, ":") {
			remainingChains[strings.TrimSpace(line)] = true
		}
	}
	expectDeletedChains := []string{"KUBE-HP-4YVONL46AKYWSKS3", "KUBE-HP-7THKRFSEH4GIIXK7", "KUBE-HP-5N7UH5JAXCVP5UJR"}
	for _, chain := range expectDeletedChains {
		_, ok := remainingChains[chain]
		assert.EqualValues(t, false, ok)
	}

	// check if all ports are closed
	for _, port := range portOpener.mem {
		assert.EqualValues(t, true, port.closed)
	}
}
