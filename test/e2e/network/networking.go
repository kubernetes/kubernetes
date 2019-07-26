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

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Networking", func() {
	var svcname = "nettest"
	f := framework.NewDefaultFramework(svcname)

	ginkgo.BeforeEach(func() {
		// Assert basic external connectivity.
		// Since this is not really a test of kubernetes in any way, we
		// leave it as a pre-test assertion, rather than a Ginko test.
		ginkgo.By("Executing a successful http request from the external internet")
		resp, err := http.Get("http://google.com")
		if err != nil {
			e2elog.Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			e2elog.Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}
	})

	ginkgo.It("should provide Internet connection for containers [Feature:Networking-IPv4]", func() {
		ginkgo.By("Running container which tries to ping 8.8.8.8")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "ping-test", "8.8.8.8", framework.IPv4PingCommand, 30))
	})

	ginkgo.It("should provide Internet connection for containers [Feature:Networking-IPv6][Experimental]", func() {
		ginkgo.By("Running container which tries to ping 2001:4860:4860::8888")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "ping-test", "2001:4860:4860::8888", framework.IPv6PingCommand, 30))
	})

	// First test because it has no dependencies on variables created later on.
	ginkgo.It("should provide unchanging, static URL paths for kubernetes api services", func() {
		tests := []struct {
			path string
		}{
			{path: "/healthz"},
			{path: "/api"},
			{path: "/apis"},
			{path: "/metrics"},
			{path: "/openapi/v2"},
			{path: "/version"},
			// TODO: test proxy links here
		}
		if !framework.ProviderIs("gke", "skeleton") {
			tests = append(tests, struct{ path string }{path: "/logs"})
		}
		for _, test := range tests {
			ginkgo.By(fmt.Sprintf("testing: %s", test.path))
			data, err := f.ClientSet.CoreV1().RESTClient().Get().
				AbsPath(test.path).
				DoRaw()
			if err != nil {
				e2elog.Failf("ginkgo.Failed: %v\nBody: %s", err, string(data))
			}
		}
	})

	ginkgo.It("should check kube-proxy urls", func() {
		// TODO: this is overkill we just need the host networking pod
		// to hit kube-proxy urls.
		config := framework.NewNetworkingTestConfig(f)

		ginkgo.By("checking kube-proxy URLs")
		config.GetSelfURL(ports.ProxyHealthzPort, "/healthz", "200 OK")
		// Verify /healthz returns the proper content.
		config.GetSelfURL(ports.ProxyHealthzPort, "/healthz", "lastUpdated")
		// Verify /proxyMode returns http status code 200.
		config.GetSelfURLStatusCode(ports.ProxyStatusPort, "/proxyMode", "200")
	})

	ginkgo.Describe("Granular Checks: Services", func() {

		ginkgo.It("should function for pod-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHTTPPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.NodeIP, config.NodeHTTPPort))
			config.DialFromTestContainer("http", config.NodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should function for pod-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUDPPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (nodeIP)", config.TestContainerPod.Name, config.NodeIP, config.NodeUDPPort))
			config.DialFromTestContainer("udp", config.NodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should function for node-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (config.clusterIP)", config.NodeIP, config.ClusterIP, framework.ClusterHTTPPort))
			config.DialFromNode("http", config.ClusterIP, framework.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHTTPPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should function for node-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (config.clusterIP)", config.NodeIP, config.ClusterIP, framework.ClusterUDPPort))
			config.DialFromNode("udp", config.ClusterIP, framework.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUDPPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should function for endpoint-Service: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.ClusterIP, framework.ClusterHTTPPort))
			config.DialFromEndpointContainer("http", config.ClusterIP, framework.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.NodeIP, config.NodeHTTPPort))
			config.DialFromEndpointContainer("http", config.NodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should function for endpoint-Service: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (config.clusterIP)", config.EndpointPods[0].Name, config.ClusterIP, framework.ClusterUDPPort))
			config.DialFromEndpointContainer("udp", config.ClusterIP, framework.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (nodeIP)", config.EndpointPods[0].Name, config.NodeIP, config.NodeUDPPort))
			config.DialFromEndpointContainer("udp", config.NodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())
		})

		ginkgo.It("should update endpoints: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHTTPPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHTTPPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNetProxyPod()

			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterHTTPPort))
			config.DialFromTestContainer("http", config.ClusterIP, framework.ClusterHTTPPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
		})

		ginkgo.It("should update endpoints: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUDPPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUDPPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNetProxyPod()

			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v (config.clusterIP)", config.TestContainerPod.Name, config.ClusterIP, framework.ClusterUDPPort))
			config.DialFromTestContainer("udp", config.ClusterIP, framework.ClusterUDPPort, config.MaxTries, config.MaxTries, config.EndpointHostnames())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		ginkgo.It("should update nodePort: http [Slow]", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHTTPPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHTTPPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNodePortService()

			ginkgo.By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeHTTPPort))
			config.DialFromNode("http", config.NodeIP, config.NodeHTTPPort, config.MaxTries, config.MaxTries, sets.NewString())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		ginkgo.It("should update nodePort: udp [Slow]", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUDPPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUDPPort, config.MaxTries, 0, config.EndpointHostnames())

			config.DeleteNodePortService()

			ginkgo.By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", config.NodeIP, config.NodeIP, config.NodeUDPPort))
			config.DialFromNode("udp", config.NodeIP, config.NodeUDPPort, config.MaxTries, config.MaxTries, sets.NewString())
		})

		ginkgo.It("should function for client IP based session affinity: http", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIP, framework.ClusterHTTPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer("http", config.SessionAffinityService.Spec.ClusterIP, framework.ClusterHTTPPort, framework.SessionAffinityChecks)
			if err != nil {
				e2elog.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				e2elog.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				e2elog.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})

		ginkgo.It("should function for client IP based session affinity: udp", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIP, framework.ClusterUDPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer("udp", config.SessionAffinityService.Spec.ClusterIP, framework.ClusterUDPPort, framework.SessionAffinityChecks)
			if err != nil {
				e2elog.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				e2elog.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				e2elog.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})
	})
})
