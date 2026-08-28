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

import "testing"

func TestSetEndpointPorts(t *testing.T) {
	podNetwork := &NetworkingTestConfig{}
	podNetwork.setEndpointPorts()

	hostNetwork := &NetworkingTestConfig{EndpointsHostNetwork: true}
	hostNetwork.setEndpointPorts()

	if podNetwork.EndpointHTTPPort != EndpointHTTPPort ||
		podNetwork.EndpointUDPPort != EndpointUDPPort ||
		podNetwork.EndpointSCTPPort != EndpointSCTPPort {
		t.Errorf("pod network endpoints should keep the default ports, got http=%d udp=%d sctp=%d",
			podNetwork.EndpointHTTPPort, podNetwork.EndpointUDPPort, podNetwork.EndpointSCTPPort)
	}

	// Host network endpoints bind their listeners in the node's network
	// namespace, so any port they share with pod network endpoints lets one
	// test config answer a concurrently running config's dials.
	// xref: https://issues.k8s.io/131370
	podNetworkPorts := map[int]string{
		podNetwork.EndpointHTTPPort: "http",
		podNetwork.EndpointUDPPort:  "udp",
		podNetwork.EndpointSCTPPort: "sctp",
	}
	for _, hostPort := range []int{hostNetwork.EndpointHTTPPort, hostNetwork.EndpointUDPPort, hostNetwork.EndpointSCTPPort} {
		if name, shared := podNetworkPorts[hostPort]; shared {
			t.Errorf("host network endpoints must not reuse pod network endpoint port %d (%s)", hostPort, name)
		}
	}
}
