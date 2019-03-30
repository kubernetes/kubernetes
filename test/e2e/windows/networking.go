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

package windows

import (
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

// NOTE(claudiub): Spawning Pods With HostNetwork enabled is not currently supported by Windows Kubelet.
// TODO(claudiub): Remove this test suite once this PR merges:
// https://github.com/kubernetes/kubernetes/pull/69525
var _ = ginkgo.Describe("[sig-network] [sig-windows] Networking", func() {
	f := framework.NewDefaultFramework("pod-network-test")

	ginkgo.BeforeEach(func() {
		// NOTE(claudiub): These tests are Windows specific.
		framework.SkipUnlessNodeOSDistroIs("windows")
	})

	ginkgo.Describe("Granular Checks: Pods", func() {

		// Try to hit all endpoints through a test container, retry 5 times,
		// expect exactly one unique hostname. Each of these endpoints reports
		// its own hostname.
		/*
			Release : v1.14
			Testname: Networking, intra pod http
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		ginkgo.It("should function for intra-pod communication: http", func() {
			config := framework.NewCoreNetworkingTestConfig(f, false)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("http", endpointPod.Status.PodIP, framework.EndpointHTTPPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			Release : v1.14
			Testname: Networking, intra pod udp
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a udp port on the each of service proxy endpoints in the cluster and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		ginkgo.It("should function for intra-pod communication: udp", func() {
			config := framework.NewCoreNetworkingTestConfig(f, false)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("udp", endpointPod.Status.PodIP, framework.EndpointUDPPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			Release : v1.14
			Testname: Networking, intra pod http, from node
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster using a http post(protocol=tcp)  and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		ginkgo.It("should function for node-pod communication: http", func() {
			config := framework.NewCoreNetworkingTestConfig(f, false)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("http", endpointPod.Status.PodIP, framework.EndpointHTTPPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			Release : v1.14
			Testname: Networking, intra pod http, from node
			Description: Create a hostexec pod that is capable of curl to netcat commands. Create a test Pod that will act as a webserver front end exposing ports 8080 for tcp and 8081 for udp. The netserver service proxies are created on specified number of nodes.
			The kubectl exec on the webserver container MUST reach a http port on the each of service proxy endpoints in the cluster using a http post(protocol=udp)  and the request MUST be successful. Container will execute curl command to reach the service port within specified max retry limit and MUST result in reporting unique hostnames.
		*/
		ginkgo.It("should function for node-pod communication: udp", func() {
			config := framework.NewCoreNetworkingTestConfig(f, false)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("udp", endpointPod.Status.PodIP, framework.EndpointUDPPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})
	})
})
