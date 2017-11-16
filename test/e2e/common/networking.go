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
	. "github.com/onsi/ginkgo"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = Describe("[sig-network] Networking", func() {
	f := framework.NewDefaultFramework("pod-network-test")

	Describe("Granular Checks: Pods", func() {

		// Try to hit all endpoints through a test container, retry 5 times,
		// expect exactly one unique hostname. Each of these endpoints reports
		// its own hostname.
		/*
			    Testname: networking-intra-pod-http
			    Description: Try to hit test endpoints from a test container and make
				sure each of them can report a unique hostname.
		*/
		framework.ConformanceIt("should function for intra-pod communication: http ", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("http", endpointPod.Status.PodIP, framework.EndpointHttpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			    Testname: networking-intra-pod-udp
			    Description: Try to hit test endpoints from a test container using udp
				and make sure each of them can report a unique hostname.
		*/
		framework.ConformanceIt("should function for intra-pod communication: udp ", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("udp", endpointPod.Status.PodIP, framework.EndpointUdpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			    Testname: networking-node-pod-http
			    Description: Try to hit test endpoints from the pod and make sure each
				of them can report a unique hostname.
		*/
		framework.ConformanceIt("should function for node-pod communication: http ", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("http", endpointPod.Status.PodIP, framework.EndpointHttpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		/*
			    Testname: networking-node-pod-udp
			    Description: Try to hit test endpoints from the pod using udp and make sure
				each of them can report a unique hostname.
		*/
		framework.ConformanceIt("should function for node-pod communication: udp ", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("udp", endpointPod.Status.PodIP, framework.EndpointUdpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})
	})
})
