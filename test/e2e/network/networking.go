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

package network

import (
	"fmt"
	"net/http"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Networking", func() {
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

	It("should provide Internet connection for containers [Feature:Networking-IPv4]", func() {
		By("Running container which tries to ping 8.8.8.8")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "ping-test", "8.8.8.8", framework.IPv4PingCommand, 30))
	})

	It("should provide Internet connection for containers [Feature:Networking-IPv6][Experimental]", func() {
		By("Running container which tries to ping google.com")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "ping-test", "google.com", framework.IPv6PingCommand, 30))
	})

	// First test because it has no dependencies on variables created later on.
	It("should provide unchanging, static URL paths for kubernetes api services", func() {
		tests := []struct {
			path string
		}{
			{path: "/healthz"},
			{path: "/api"},
			{path: "/apis"},
			{path: "/metrics"},
			{path: "/swaggerapi"},
			{path: "/version"},
			// TODO: test proxy links here
		}
		if !framework.ProviderIs("gke", "skeleton") {
			tests = append(tests, struct{ path string }{path: "/logs"})
		}
		for _, test := range tests {
			By(fmt.Sprintf("testing: %s", test.path))
			data, err := f.ClientSet.CoreV1().RESTClient().Get().
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
		config.GetSelfURL(ports.ProxyHealthzPort, "/healthz", "200 OK")
		// Verify /healthz returns the proper content.
		config.GetSelfURL(ports.ProxyHealthzPort, "/healthz", "lastUpdated")
		// Verify /proxyMode returns http status code 200.
		config.GetSelfURLStatusCode(ports.ProxyStatusPort, "/proxyMode", "200")
	})

	// TODO: Remove [Slow] when this has had enough bake time to prove presubmit worthiness.
	Describe("Granular Checks: Services [Slow]", func() {

		It("should function for pod-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHttpPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.NodeIP, config.NodeHttpPort))
			config.DialFromTestContainer("http", config.NodeIP, config.NodeHttpPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		It("should function for pod-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUdpPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries, 0, config.EndpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.NodeIP, config.NodeUdpPort))
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

		It("should function for client IP based session affinity: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHttpPort))
			updateSessionAffinity := func(svc *v1.Service) {
				svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			}
			_, err := framework.UpdateService(f.ClientSet, config.NodePortService.Namespace, config.NodePortService.Name, updateSessionAffinity)
			if err != nil {
				framework.Failf("Failed to update service session affinity, error: %v", err)
			}

			// Fetch first endpoints when visiting service
			firstEndpoints, err := config.GetEndpointsFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries)
			if err != nil {
				framework.Failf("Unable to get endpoints from test container: %v", err)
			}
			// Check if first endpoints are equal to endpoints which are fetched later
			for i := 0; i < framework.SessionAffinityChecks; i++ {
				eps, err := config.GetEndpointsFromTestContainer("http", config.ClusterIP, framework.ClusterHttpPort, config.MaxTries)
				if err != nil {
					framework.Failf("Unable to get endpoints from test container: %v", err)
				}
				if !eps.Equal(firstEndpoints) {
					framework.Failf("Expect endpoints: %v, got: %v", firstEndpoints, eps)
				}
			}
		})

		It("should function for client IP based session affinity: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUdpPort))
			updateSessionAffinity := func(svc *v1.Service) {
				svc.Spec.SessionAffinity = v1.ServiceAffinityClientIP
			}
			_, err := framework.UpdateService(f.ClientSet, config.NodePortService.Namespace, config.NodePortService.Name, updateSessionAffinity)
			if err != nil {
				framework.Failf("Failed to update service session affinity, error: %v", err)
			}

			// Fetch first endpoints when visiting service
			firstEndpoints, err := config.GetEndpointsFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries)
			if err != nil {
				framework.Failf("Unable to get endpoints from test containers: %v", err)
			}
			// Check if first endpoints are equal to endpoints which are fetched later
			for i := 0; i < framework.SessionAffinityChecks; i++ {
				eps, err := config.GetEndpointsFromTestContainer("udp", config.ClusterIP, framework.ClusterUdpPort, config.MaxTries)
				if err != nil {
					framework.Failf("Unable to get endpoints from test containers: %v", err)
				}
				if !eps.Equal(firstEndpoints) {
					framework.Failf("Expect endpoints: %v, got: %v", firstEndpoints, eps)
				}
			}
		})
	})
})
