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
	"net"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/api/core/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
	"strings"
)

func NewFakeHostportManager() HostPortManager {
	return &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    NewFakeIPTables(),
		portOpener:  NewFakeSocketManager().openFakeSocket,
	}
}

func TestHostportManager(t *testing.T) {
	iptables := NewFakeIPTables()
	portOpener := NewFakeSocketManager()
	manager := &hostportManager{
		hostPortMap: make(map[hostport]closeable),
		iptables:    iptables,
		portOpener:  portOpener.openFakeSocket,
	}

	testCases := []struct {
		mapping     *PodPortMapping
		expectError bool
	}{
		{
			mapping: &PodPortMapping{
				Name:        "pod1",
				Namespace:   "ns1",
				IP:          net.ParseIP("10.1.1.2"),
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
				},
			},
			expectError: false,
		},
		{
			mapping: &PodPortMapping{
				Name:        "pod2",
				Namespace:   "ns1",
				IP:          net.ParseIP("10.1.1.3"),
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
				},
			},
			expectError: true,
		},
		{
			mapping: &PodPortMapping{
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
	expectedPorts := []hostport{{8080, "tcp"}, {8081, "udp"}, {8443, "tcp"}}
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

	lines := strings.Split(string(raw.Bytes()), "\n")
	expectedLines := map[string]bool{
		`*nat`: true,
		`:KUBE-HOSTPORTS - [0:0]`:                                                                                                         true,
		`:OUTPUT - [0:0]`:                                                                                                                 true,
		`:PREROUTING - [0:0]`:                                                                                                             true,
		`:POSTROUTING - [0:0]`:                                                                                                            true,
		`:KUBE-HP-4YVONL46AKYWSKS3 - [0:0]`:                                                                                               true,
		`:KUBE-HP-7THKRFSEH4GIIXK7 - [0:0]`:                                                                                               true,
		`:KUBE-HP-5N7UH5JAXCVP5UJR - [0:0]`:                                                                                               true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-5N7UH5JAXCVP5UJR":        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp --dport 8081 -j KUBE-HP-7THKRFSEH4GIIXK7":        true,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp --dport 8080 -j KUBE-HP-4YVONL46AKYWSKS3":        true,
		"-A OUTPUT -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                         true,
		"-A PREROUTING -m comment --comment \"kube hostport portals\" -m addrtype --dst-type LOCAL -j KUBE-HOSTPORTS":                     true,
		"-A POSTROUTING -m comment --comment \"SNAT for localhost access to hostports\" -o cbr0 -s 127.0.0.0/8 -j MASQUERADE":             true,
		"-A KUBE-HP-4YVONL46AKYWSKS3 -m comment --comment \"pod1_ns1 hostport 8080\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                    true,
		"-A KUBE-HP-4YVONL46AKYWSKS3 -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:80":  true,
		"-A KUBE-HP-7THKRFSEH4GIIXK7 -m comment --comment \"pod1_ns1 hostport 8081\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ":                    true,
		"-A KUBE-HP-7THKRFSEH4GIIXK7 -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:81":  true,
		"-A KUBE-HP-5N7UH5JAXCVP5UJR -m comment --comment \"pod3_ns1 hostport 8443\" -s 10.1.1.4/32 -j KUBE-MARK-MASQ":                    true,
		"-A KUBE-HP-5N7UH5JAXCVP5UJR -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.4:443": true,
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
	lines = strings.Split(string(raw.Bytes()), "\n")
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
