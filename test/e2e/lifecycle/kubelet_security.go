/*
Copyright 2018 The Kubernetes Authors.

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

package lifecycle

import (
	"fmt"
	"net"
	"net/http"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("Ports Security Check [Feature:KubeletSecurity]", func() {
	f := framework.NewDefaultFramework("kubelet-security")

	var node *v1.Node
	var nodeName string

	BeforeEach(func() {
		nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		Expect(len(nodes.Items)).NotTo(BeZero())
		node = &nodes.Items[0]
		nodeName = node.Name
	})

	// make sure kubelet readonly (10255) and cadvisor (4194) ports are disabled via API server proxy
	It(fmt.Sprintf("should not be able to proxy to the readonly kubelet port %v using proxy subresource", ports.KubeletReadOnlyPort), func() {
		result, err := framework.NodeProxyRequest(f.ClientSet, nodeName, "pods/", ports.KubeletReadOnlyPort)
		Expect(err).NotTo(HaveOccurred())

		var statusCode int
		result.StatusCode(&statusCode)
		Expect(statusCode).NotTo(Equal(http.StatusOK))
	})
	It("should not be able to proxy to cadvisor port 4194 using proxy subresource", func() {
		result, err := framework.NodeProxyRequest(f.ClientSet, nodeName, "containers/", 4194)
		Expect(err).NotTo(HaveOccurred())

		var statusCode int
		result.StatusCode(&statusCode)
		Expect(statusCode).NotTo(Equal(http.StatusOK))
	})

	// make sure kubelet readonly (10255) and cadvisor (4194) ports are closed on the public IP address
	disabledPorts := []int{ports.KubeletReadOnlyPort, 4194}
	for _, port := range disabledPorts {
		It(fmt.Sprintf("should not have port %d open on its all public IP addresses", port), func() {
			portClosedTest(f, node, port)
		})
	}
})

// checks whether the target port is closed
func portClosedTest(f *framework.Framework, pickNode *v1.Node, port int) {
	nodeAddrs := framework.GetNodeAddresses(pickNode, v1.NodeExternalIP)
	Expect(len(nodeAddrs)).NotTo(BeZero())

	for _, addr := range nodeAddrs {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", addr, port), 1*time.Minute)
		if err == nil {
			conn.Close()
			framework.Failf("port %d is not disabled", port)
		}
	}
}
