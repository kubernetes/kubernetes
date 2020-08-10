/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enetwork "k8s.io/kubernetes/test/e2e/framework/network"
)

var _ = ginkgo.Describe("[sig-network] Networking", func() {
	f := framework.NewDefaultFramework("pod-network-test")

	ginkgo.Describe("Granular Checks: Pods", func() {

		checkNodeConnectivity := func(config *e2enetwork.NetworkingTestConfig, protocol string, port int) {
			errors := []error{}
			for _, endpointPod := range config.EndpointPods {
				if err := config.DialFromTestContainer(protocol, endpointPod.Status.PodIP, port, config.MaxTries, 0, sets.NewString(endpointPod.Name)); err != nil {
					errors = append(errors, err)
					framework.Logf("Warning: Test failure (%v) will occur due to %v", len(errors)+1, err) // convenient error message for diagnosis... how many pods failed, and on what hosts?
				} else {
					framework.Logf("Was able to reach %v on %v ", endpointPod.Status.PodIP, endpointPod.Status.HostIP)
				}
			}
			if len(errors) > 0 {
				framework.Logf("Pod polling failure summary:")
				for _, e := range errors {
					framework.Logf("%v", e)
				}
				framework.Failf("Failed due to %v errors polling %v pods", len(errors), len(config.EndpointPods))
			}
		}

		// Try to hit all endpoints through a test container, retry 5 times,
		// expect exactly one unique hostname. Each of these endpoints reports
		// its own hostname.
		/*
			Release: v1.9, v1.18
			Testname: Networking, intra pod http
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		framework.ConformanceIt("should function for intra-pod communication: http [NodeConformance]", func() {
			config := e2enetwork.NewCoreNetworkingTestConfig(f, false)
			checkNodeConnectivity(config, "http", e2enetwork.EndpointHTTPPort)
		})

		/*
			Release: v1.9, v1.18
			Testname: Networking, intra pod udp
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a udp port on the each of service proxy endpoints in the cluster and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		framework.ConformanceIt("should function for intra-pod communication: udp [NodeConformance]", func() {
			config := e2enetwork.NewCoreNetworkingTestConfig(f, false)
			checkNodeConnectivity(config, "udp", e2enetwork.EndpointUDPPort)
		})

		/*
			Release: v1.9
			Testname: Networking, intra pod http, from node
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster using a http post(protocol=tcp)  and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
			This test is marked LinuxOnly since HostNetwork is not supported on other platforms like Windows.
		*/
		framework.ConformanceIt("should function for node-pod communication: http [LinuxOnly] [NodeConformance]", func() {
			config := e2enetwork.NewCoreNetworkingTestConfig(f, true)
			checkNodeConnectivity(config, "http", e2enetwork.EndpointHTTPPort)
		})

		/*
			Release: v1.9
			Testname: Networking, intra pod http, from node
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster using a http post(protocol=udp)  and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
			This test is marked LinuxOnly since HostNetwork is not supported on other platforms like Windows.
		*/
		framework.ConformanceIt("should function for node-pod communication: udp [LinuxOnly] [NodeConformance]", func() {
			config := e2enetwork.NewCoreNetworkingTestConfig(f, true)
			checkNodeConnectivity(config, "udp", e2enetwork.EndpointUDPPort)
		})
	})
})
