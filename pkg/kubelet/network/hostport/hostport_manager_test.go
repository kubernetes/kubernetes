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
	"net"
	"testing"

	"strings"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	utiliptables "k8s.io/kubernetes/pkg/util/iptables"
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
		err := manager.Add("id", tc.mapping)
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
	raw, err := iptables.Save(utiliptables.TableNAT)
	assert.NoError(t, err)

	linesList := strings.Split(string(raw), "\n")
	for i, line := range linesList {
		linesList[i] = strings.TrimSpace(line)
	}
	lines := sets.NewString(linesList...)

	expectedLines := sets.NewString(
		`*nat`,
		`:KUBE-HOSTPORTS - [0:0]`,
		`:KUBE-LOCALDEST - [0:0]`,
		`:OUTPUT - [0:0]`,
		`:PREROUTING - [0:0]`,
		`:KUBE-HP-4YVONL46AKYWSKS3 - [0:0]`,
		`:KUBE-HP-7THKRFSEH4GIIXK7 - [0:0]`,
		`:KUBE-HP-5N7UH5JAXCVP5UJR - [0:0]`,
		"-A KUBE-HOSTPORTS -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp --dport 8443 -j KUBE-HP-5N7UH5JAXCVP5UJR",
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp --dport 8081 -j KUBE-HP-7THKRFSEH4GIIXK7",
		"-A KUBE-HOSTPORTS -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp --dport 8080 -j KUBE-HP-4YVONL46AKYWSKS3",
		"-A KUBE-LOCALDEST -m comment --comment \"maybe kube hostport\" -d 127.0.0.1/32 -j KUBE-HOSTPORTS",
		"-A OUTPUT -m comment --comment \"maybe kube hostport\" -m addrtype --dst-type LOCAL -j KUBE-LOCALDEST",
		"-A PREROUTING -m comment --comment \"maybe kube hostport\" -m addrtype --dst-type LOCAL -j KUBE-LOCALDEST",
		"-A KUBE-HP-4YVONL46AKYWSKS3 -m comment --comment \"pod1_ns1 hostport 8080\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ",
		"-A KUBE-HP-4YVONL46AKYWSKS3 -m comment --comment \"pod1_ns1 hostport 8080\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.2:80",
		"-A KUBE-HP-7THKRFSEH4GIIXK7 -m comment --comment \"pod1_ns1 hostport 8081\" -s 10.1.1.2/32 -j KUBE-MARK-MASQ",
		"-A KUBE-HP-7THKRFSEH4GIIXK7 -m comment --comment \"pod1_ns1 hostport 8081\" -m udp -p udp -j DNAT --to-destination 10.1.1.2:81",
		"-A KUBE-HP-5N7UH5JAXCVP5UJR -m comment --comment \"pod3_ns1 hostport 8443\" -s 10.1.1.4/32 -j KUBE-MARK-MASQ",
		"-A KUBE-HP-5N7UH5JAXCVP5UJR -m comment --comment \"pod3_ns1 hostport 8443\" -m tcp -p tcp -j DNAT --to-destination 10.1.1.4:443",
		`COMMIT`)
	for line := range expectedLines {
		_, ok := lines[line]
		assert.EqualValues(t, true, ok, "line %q not found in output", line)
	}

	// Remove all added hostports
	for _, tc := range testCases {
		if !tc.expectError {
			err := manager.Remove("id", tc.mapping)
			assert.NoError(t, err)
		}
	}

	// Check Iptables-save result after deleting hostports
	raw, err = iptables.Save(utiliptables.TableNAT)
	assert.NoError(t, err)
	linesList = strings.Split(string(raw), "\n")
	for i, line := range linesList {
		linesList[i] = strings.TrimSpace(line)
	}
	lines = sets.NewString(linesList...)
	remainingChains := sets.NewString()
	for line := range lines {
		if strings.HasPrefix(line, ":") {
			remainingChains.Insert(line)
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
