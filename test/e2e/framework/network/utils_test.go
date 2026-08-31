/*
Copyright The Kubernetes Authors.

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

package network

import (
	"testing"

	"github.com/onsi/ginkgo/v2"
)

func TestSetEndpointPorts(t *testing.T) {
	podNetwork := &NetworkingTestConfig{}
	podNetwork.setEndpointPorts()

	if podNetwork.EndpointHTTPPort != endpointHTTPPort ||
		podNetwork.EndpointUDPPort != endpointUDPPort ||
		podNetwork.EndpointSCTPPort != endpointSCTPPort {
		t.Errorf("pod network endpoints should keep the well-known ports, got http=%d udp=%d sctp=%d",
			podNetwork.EndpointHTTPPort, podNetwork.EndpointUDPPort, podNetwork.EndpointSCTPPort)
	}

	hostNetwork := &NetworkingTestConfig{EndpointsHostNetwork: true}
	hostNetwork.setEndpointPorts()

	wantHTTP, wantUDP, wantSCTP := hostNetworkEndpointPorts(ginkgo.GinkgoParallelProcess())
	if hostNetwork.EndpointHTTPPort != wantHTTP ||
		hostNetwork.EndpointUDPPort != wantUDP ||
		hostNetwork.EndpointSCTPPort != wantSCTP {
		t.Errorf("host network endpoints should take the ports reserved for this process, want http=%d udp=%d sctp=%d, got http=%d udp=%d sctp=%d",
			wantHTTP, wantUDP, wantSCTP,
			hostNetwork.EndpointHTTPPort, hostNetwork.EndpointUDPPort, hostNetwork.EndpointSCTPPort)
	}
}

// TestHostNetworkEndpointPortsAreUnique covers the property that keeps host
// network endpoints from answering dials meant for someone else: the endpoint
// pods of one config listen on every node, so no two configs that can run at the
// same time may claim a port, and none of them may claim a pod network port.
// xref: https://issues.k8s.io/131370
func TestHostNetworkEndpointPortsAreUnique(t *testing.T) {
	// The start of the default service-node-port-range. Host network endpoints
	// have to stay clear of it, and of the ephemeral range above it, or the
	// node may already be using the port they try to bind.
	const nodePortRangeStart = 30000
	// Far wider than the parallelism any job runs with, so that the range still
	// holds if jobs get wider.
	const parallelProcesses = 256

	podNetworkPorts := map[int]string{
		endpointHTTPPort: "http",
		endpointUDPPort:  "udp",
		endpointSCTPPort: "sctp",
	}
	owner := make(map[int]int, parallelProcesses*hostNetworkEndpointPortRange)
	for process := 1; process <= parallelProcesses; process++ {
		httpPort, udpPort, sctpPort := hostNetworkEndpointPorts(process)
		for _, port := range []int{httpPort, udpPort, sctpPort} {
			if name, shared := podNetworkPorts[port]; shared {
				t.Errorf("process %d claims pod network %s port %d", process, name, port)
			}
			if previous, taken := owner[port]; taken {
				t.Errorf("process %d claims port %d, already held by process %d", process, port, previous)
			}
			if port >= nodePortRangeStart {
				t.Errorf("process %d claims port %d, which is inside the default NodePort range", process, port)
			}
			owner[port] = process
		}
	}
}
