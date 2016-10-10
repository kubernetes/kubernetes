/*
Copyright 2014 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"net/http"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("Networking", func() {
	var svcname = "nettest"
	f := framework.NewDefaultFramework(svcname)

	BeforeEach(func() {
		// Assert basic external connectivity.
		// Since this is not really a test of kubernetes in any way, we
		// leave it as a pre-test assertion, rather than a Ginko test.
		By("Executing a successful http request from the external internet")
		resp, err := http.Get("http://google.com")
		if err != nil {
			framework.Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			framework.Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}
	})

	It("should provide Internet connection for containers [Conformance]", func() {
		By("Running container which tries to wget google.com")
		framework.ExpectNoError(framework.CheckConnectivityToHost(f, "", "wget-test", "google.com", 30))
	})

	// First test because it has no dependencies on variables created later on.
	It("should provide unchanging, static URL paths for kubernetes api services [Conformance]", func() {
		tests := []struct {
			path string
		}{
			{path: "/healthz"},
			{path: "/api"},
			{path: "/apis"},
			{path: "/logs"},
			{path: "/metrics"},
			{path: "/swaggerapi"},
			{path: "/version"},
			// TODO: test proxy links here
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := f.Client.RESTClient.Get().
				AbsPath(test.path).
				DoRaw()
			if err != nil {
				framework.Failf("Failed: %v\nBody: %s", err, string(data))
			}
		}
	})

	It("should check kube-proxy urls", func() {
		// TODO: this is overkill we just need the host networking pod
		// to hit kube-proxy urls.
		config := framework.NewNetworkingTestConfig(f)

		By("checking kube-proxy URLs")
		config.GetSelfURL("/healthz", "ok")
		config.GetSelfURL("/proxyMode", "iptables") // the default
	})

	// TODO: Remove [Slow] when this has had enough bake time to prove presubmit worthiness.
	framework.KubeDescribe("Granular Checks: Services [Slow]", func() {

		It("should function for pod-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.ExternalAddrs[0], config.NodeHttpPort))
			config.DialFromTestContainer("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for pod-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.ExternalAddrs[0], config.NodeUdpPort))
			config.DialFromTestContainer("udp", config.NodeIP, config.NodeUdpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for node-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (config.clusterIP)", config.NodeIP, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromNode("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHttpPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for node-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (config.clusterIP)", config.NodeIP, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromNode("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUdpPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUdpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for endpoint-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromEndpointContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.NodeIP, config.NodeHttpPort))
			config.DialFromEndpointContainer("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for endpoint-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromEndpointContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.NodeIP, config.NodeUdpPort))
			config.DialFromEndpointContainer("udp", config.NodeIP, config.NodeUdpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should update endpoints: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNetProxyPod()

			By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
		})

		It("should update endpoints: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNetProxyPod()

			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		It("should update nodePort: http [Slow]", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHttpPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNodePortService()

			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHttpPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, config.MaxTries, sets.NewString())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		It("should update nodePort: udp [Slow]", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUdpPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNodePortService()

			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUdpPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUdpPort, config.MaxTries, config.MaxTries, sets.NewString())
		})
		// TODO: Test sessionAffinity #31712
	})
})
