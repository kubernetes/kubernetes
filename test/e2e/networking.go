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
	"time"

	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Networking", func() {
	var svcname = "nettest"
	f := framework.NewDefaultFramework(svcname)

	BeforeEach(func() {
		//Assert basic external connectivity.
		//Since this is not really a test of kubernetes in any way, we
		//leave it as a pre-test assertion, rather than a Ginko test.
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
		config := &NetworkingTestConfig{f: f}

		// TODO: this is overkill we just need the host networking pod
		// to hit kube-proxy urls.
		config.setup()

		By("checking kube-proxy URLs")
		config.getSelfURL("/healthz", "ok")
		config.getSelfURL("/proxyMode", "iptables") // the default
	})

	framework.KubeDescribe("Granular Checks: Pods", func() {
		var config *NetworkingTestConfig
		var tries int

		BeforeEach(func() {
			config = &NetworkingTestConfig{f: f}
			By("Performing setup")
			config.setup()
			epCount := len(config.endpointPods)
			tries = epCount*epCount + testTries
		})
		AfterEach(func() {
			By("Performing cleanup")
			config.cleanup()
		})

		// Try to hit all endpoints through a test container, retry 5 times,
		// expect exactly one unique hostname. Each of these endpoints reports
		// its own hostname.
		It("should function for intra-pod communication: http [Conformance]", func() {
			for _, endpointPod := range config.endpointPods {
				config.dialFromTestContainer("http", endpointPod.Status.PodIP, endpointHttpPort, tries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for intra-pod communication: udp [Conformance]", func() {
			for _, endpointPod := range config.endpointPods {
				config.dialFromTestContainer("udp", endpointPod.Status.PodIP, endpointUdpPort, tries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for node-pod communication: http [Conformance]", func() {
			for _, endpointPod := range config.endpointPods {
				config.dialFromNode("http", endpointPod.Status.PodIP, endpointHttpPort, tries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for node-pod communication: udp [Conformance]", func() {
			for _, endpointPod := range config.endpointPods {
				config.dialFromNode("udp", endpointPod.Status.PodIP, endpointUdpPort, tries, 0, sets.NewString(endpointPod.Name))
			}
		})
	})

	// TODO: Remove [Slow] when this has had enough bake time to prove presubmit worthiness.
	framework.KubeDescribe("Granular Checks: Service [Slow]", func() {
		var config *NetworkingTestConfig
		var tries int
		var clusterIP, node1_IP string

		BeforeEach(func() {
			config = &NetworkingTestConfig{f: f}
			By("Performing setup")
			config.setup()
			epCount := len(config.endpointPods)
			tries = epCount*epCount + testTries
			clusterIP = config.nodePortService.Spec.ClusterIP
			node1_IP = config.externalAddrs[0]
		})

		AfterEach(func() {
			By("Performing cleanup")
			config.cleanup()
		})

		It("should function for pod-Service: http", func() {
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterHttpPort))
			config.dialFromTestContainer("http", clusterIP, clusterHttpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(http) %v --> %v:%v (nodeIP)", config.testContainerPod.Name, config.externalAddrs[0], nodeHttpPort))
			config.dialFromTestContainer("http", node1_IP, nodeHttpPort, tries, 0, config.endpointHostnames())
		})

		It("should function for pod-Service: udp", func() {
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterUdpPort))
			config.dialFromTestContainer("udp", clusterIP, clusterUdpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (nodeIP)", config.testContainerPod.Name, config.externalAddrs[0], nodeUdpPort))
			config.dialFromTestContainer("udp", node1_IP, nodeUdpPort, tries, 0, config.endpointHostnames())
		})

		It("should function for node-Service: http", func() {
			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (clusterIP)", node1_IP, clusterIP, clusterHttpPort))
			config.dialFromNode("http", clusterIP, clusterHttpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeHttpPort))
			config.dialFromNode("http", node1_IP, nodeHttpPort, tries, 0, config.endpointHostnames())
		})

		It("should function for node-Service: udp", func() {
			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (clusterIP)", node1_IP, clusterIP, clusterUdpPort))
			config.dialFromNode("udp", clusterIP, clusterUdpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeUdpPort))
			config.dialFromNode("udp", node1_IP, nodeUdpPort, tries, 0, config.endpointHostnames())
		})

		It("should function for endpoint-Service: http", func() {
			By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (clusterIP)", config.endpointPods[0].Name, clusterIP, clusterHttpPort))
			config.dialFromEndpointContainer("http", clusterIP, clusterHttpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(http) %v (endpoint) --> %v:%v (nodeIP)", config.endpointPods[0].Name, node1_IP, nodeHttpPort))
			config.dialFromEndpointContainer("http", node1_IP, nodeHttpPort, tries, 0, config.endpointHostnames())
		})

		It("should function for endpoint-Service: udp", func() {
			By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (clusterIP)", config.endpointPods[0].Name, clusterIP, clusterUdpPort))
			config.dialFromEndpointContainer("udp", clusterIP, clusterUdpPort, tries, 0, config.endpointHostnames())

			By(fmt.Sprintf("dialing(udp) %v (endpoint) --> %v:%v (nodeIP)", config.endpointPods[0].Name, node1_IP, nodeUdpPort))
			config.dialFromEndpointContainer("udp", node1_IP, nodeUdpPort, tries, 0, config.endpointHostnames())
		})

		It("should update endpoints: http", func() {
			By(fmt.Sprintf("dialing(http) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterHttpPort))
			config.dialFromTestContainer("http", clusterIP, clusterHttpPort, tries, 0, config.endpointHostnames())

			config.deleteNetProxyPod()

			By(fmt.Sprintf("dialing(http) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterHttpPort))
			config.dialFromTestContainer("http", clusterIP, clusterHttpPort, tries, tries, config.endpointHostnames())
		})

		It("should update endpoints: udp", func() {
			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterUdpPort))
			config.dialFromTestContainer("udp", clusterIP, clusterUdpPort, tries, 0, config.endpointHostnames())

			config.deleteNetProxyPod()

			By(fmt.Sprintf("dialing(udp) %v --> %v:%v (clusterIP)", config.testContainerPod.Name, clusterIP, clusterUdpPort))
			config.dialFromTestContainer("udp", clusterIP, clusterUdpPort, tries, tries, config.endpointHostnames())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		It("should update nodePort: http [Slow]", func() {
			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeHttpPort))
			config.dialFromNode("http", node1_IP, nodeHttpPort, tries, 0, config.endpointHostnames())

			config.deleteNodePortService()

			By(fmt.Sprintf("dialing(http) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeHttpPort))
			config.dialFromNode("http", node1_IP, nodeHttpPort, tries, tries, sets.NewString())
		})

		// Slow because we confirm that the nodePort doesn't serve traffic, which requires a period of polling.
		It("should update nodePort: udp [Slow]", func() {
			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeUdpPort))
			config.dialFromNode("udp", node1_IP, nodeUdpPort, tries, 0, config.endpointHostnames())

			config.deleteNodePortService()

			By(fmt.Sprintf("dialing(udp) %v (node) --> %v:%v (nodeIP)", node1_IP, node1_IP, nodeUdpPort))
			config.dialFromNode("udp", node1_IP, nodeUdpPort, tries, tries, sets.NewString())
		})

	})
})
