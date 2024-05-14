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

package gcp

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Ports Security Check", feature.KubeletSecurity, func() {
	f := framework.NewDefaultFramework("kubelet-security")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var node *v1.Node
	var nodeName string

	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		var err error
		node, err = e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		nodeName = node.Name
	})

	// make sure kubelet readonly (10255) and cadvisor (4194) ports are disabled via API server proxy
	ginkgo.It(fmt.Sprintf("should not be able to proxy to the readonly kubelet port %v using proxy subresource", ports.KubeletReadOnlyPort), func(ctx context.Context) {
		result, err := e2ekubelet.ProxyRequest(ctx, f.ClientSet, nodeName, "pods/", ports.KubeletReadOnlyPort)
		framework.ExpectNoError(err)

		var statusCode int
		result.StatusCode(&statusCode)
		gomega.Expect(statusCode).ToNot(gomega.Equal(http.StatusOK))
	})
	ginkgo.It("should not be able to proxy to cadvisor port 4194 using proxy subresource", func(ctx context.Context) {
		result, err := e2ekubelet.ProxyRequest(ctx, f.ClientSet, nodeName, "containers/", 4194)
		framework.ExpectNoError(err)

		var statusCode int
		result.StatusCode(&statusCode)
		gomega.Expect(statusCode).ToNot(gomega.Equal(http.StatusOK))
	})

	// make sure kubelet readonly (10255) and cadvisor (4194) ports are closed on the public IP address
	disabledPorts := []int{ports.KubeletReadOnlyPort, 4194}
	for _, port := range disabledPorts {
		port := port
		ginkgo.It(fmt.Sprintf("should not have port %d open on its all public IP addresses", port), func(ctx context.Context) {
			portClosedTest(f, node, port)
		})
	}
})

// checks whether the target port is closed
func portClosedTest(f *framework.Framework, pickNode *v1.Node, port int) {
	nodeAddrs := e2enode.GetAddresses(pickNode, v1.NodeExternalIP)
	gomega.Expect(nodeAddrs).ToNot(gomega.BeEmpty())

	for _, addr := range nodeAddrs {
		conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", addr, port), 1*time.Minute)
		if err == nil {
			conn.Close()
			framework.Failf("port %d is not disabled", port)
		}
	}
}
