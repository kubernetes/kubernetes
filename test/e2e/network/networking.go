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
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	utilwait "k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"

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
			framework.Failf("Unable to connect/talk to the internet: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			framework.Failf("Unexpected error code, expected 200, got, %v (%v)", resp.StatusCode, resp)
		}
	})

	ginkgo.It("should provide Internet connection for containers [Feature:Networking-IPv4]", func() {
		ginkgo.By("Running container which tries to connect to 8.8.8.8")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "connectivity-test", "8.8.8.8", 53, 30))
	})

	ginkgo.It("should provide Internet connection for containers [Feature:Networking-IPv6][Experimental][LinuxOnly]", func() {
		// IPv6 is not supported on Windows.
		framework.SkipIfNodeOSDistroIs("windows")
		ginkgo.By("Running container which tries to connect to 2001:4860:4860::8888")
		framework.ExpectNoError(
			framework.CheckConnectivityToHost(f, "", "connectivity-test", "2001:4860:4860::8888", 53, 30))
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
				framework.Failf("ginkgo.Failed: %v\nBody: %s", err, string(data))
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

		// [LinuxOnly]: Windows does not support session affinity.
		ginkgo.It("should function for client IP based session affinity: http [LinuxOnly]", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(http) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIP, framework.ClusterHTTPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer("http", config.SessionAffinityService.Spec.ClusterIP, framework.ClusterHTTPPort, framework.SessionAffinityChecks)
			if err != nil {
				framework.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				framework.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				framework.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})

		// [LinuxOnly]: Windows does not support session affinity.
		ginkgo.It("should function for client IP based session affinity: udp [LinuxOnly]", func() {
			config := framework.NewNetworkingTestConfig(f)
			ginkgo.By(fmt.Sprintf("dialing(udp) %v --> %v:%v", config.TestContainerPod.Name, config.SessionAffinityService.Spec.ClusterIP, framework.ClusterUDPPort))

			// Check if number of endpoints returned are exactly one.
			eps, err := config.GetEndpointsFromTestContainer("udp", config.SessionAffinityService.Spec.ClusterIP, framework.ClusterUDPPort, framework.SessionAffinityChecks)
			if err != nil {
				framework.Failf("ginkgo.Failed to get endpoints from test container, error: %v", err)
			}
			if len(eps) == 0 {
				framework.Failf("Unexpected no endpoints return")
			}
			if len(eps) > 1 {
				framework.Failf("Unexpected endpoints return: %v, expect 1 endpoints", eps)
			}
		})
	})

	ginkgo.It("should recreate its iptables rules if they are deleted [Disruptive]", func() {
		framework.SkipUnlessProviderIs(framework.ProvidersWithSSH...)
		framework.SkipUnlessSSHKeyPresent()

		hosts, err := e2essh.NodeSSHHosts(f.ClientSet)
		framework.ExpectNoError(err, "failed to find external/internal IPs for every node")
		if len(hosts) == 0 {
			framework.Failf("No ssh-able nodes")
		}
		host := hosts[0]

		ns := f.Namespace.Name
		numPods, servicePort := 3, defaultServeHostnameServicePort
		svc := "iptables-flush-test"

		defer func() {
			framework.ExpectNoError(e2eservice.StopServeHostnameService(f.ClientSet, ns, svc))
		}()
		podNames, svcIP, err := e2eservice.StartServeHostnameService(f.ClientSet, getServeHostnameService(svc), ns, numPods)
		framework.ExpectNoError(err, "failed to create replication controller with service: %s in the namespace: %s", svc, ns)

		// Ideally we want to reload the system firewall, but we don't necessarily
		// know how to do that on this system ("firewall-cmd --reload"? "systemctl
		// restart iptables"?). So instead we just manually delete all "KUBE-"
		// chains.

		ginkgo.By("dumping iptables rules on a node")
		result, err := e2essh.SSH("sudo iptables-save", host, framework.TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
			framework.Failf("couldn't dump iptable rules: %v", err)
		}

		// All the commands that delete rules have to come before all the commands
		// that delete chains, since the chains can't be deleted while there are
		// still rules referencing them.
		var deleteRuleCmds, deleteChainCmds []string
		table := ""
		for _, line := range strings.Split(result.Stdout, "\n") {
			if strings.HasPrefix(line, "*") {
				table = line[1:]
			} else if table == "" {
				continue
			}

			// Delete jumps from non-KUBE chains to KUBE chains
			if !strings.HasPrefix(line, "-A KUBE-") && strings.Contains(line, "-j KUBE-") {
				deleteRuleCmds = append(deleteRuleCmds, fmt.Sprintf("sudo iptables -t %s -D %s || true", table, line[3:]))
			}
			// Flush and delete all KUBE chains
			if strings.HasPrefix(line, ":KUBE-") {
				chain := strings.Split(line, " ")[0][1:]
				deleteRuleCmds = append(deleteRuleCmds, fmt.Sprintf("sudo iptables -t %s -F %s || true", table, chain))
				deleteChainCmds = append(deleteChainCmds, fmt.Sprintf("sudo iptables -t %s -X %s || true", table, chain))
			}
		}
		cmd := strings.Join(append(deleteRuleCmds, deleteChainCmds...), "\n")

		ginkgo.By("deleting all KUBE-* iptables chains")
		result, err = e2essh.SSH(cmd, host, framework.TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
			framework.Failf("couldn't delete iptable rules: %v", err)
		}

		ginkgo.By("verifying that kube-proxy rules are eventually recreated")
		framework.ExpectNoError(e2eservice.VerifyServeHostnameServiceUp(f.ClientSet, ns, host, podNames, svcIP, servicePort))

		ginkgo.By("verifying that kubelet rules are eventually recreated")
		err = utilwait.PollImmediate(framework.Poll, framework.RestartNodeReadyAgainTimeout, func() (bool, error) {
			result, err = e2essh.SSH("sudo iptables-save -t nat", host, framework.TestContext.Provider)
			if err != nil || result.Code != 0 {
				e2essh.LogResult(result)
				return false, err
			}

			if strings.Contains(result.Stdout, "\n-A KUBE-MARK-DROP ") {
				return true, nil
			}
			return false, nil
		})
		framework.ExpectNoError(err, "kubelet did not recreate its iptables rules")
	})
})
