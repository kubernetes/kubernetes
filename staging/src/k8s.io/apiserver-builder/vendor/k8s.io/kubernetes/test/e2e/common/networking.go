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

var _ = framework.KubeDescribe("Networking", func() {
	f := framework.NewDefaultFramework("pod-network-test")

	framework.KubeDescribe("Granular Checks: Pods", func() {

		// Try to hit all endpoints through a test container, retry 5 times,
		// expect exactly one unique hostname. Each of these endpoints reports
		// its own hostname.
		It("should function for intra-pod communication: http [Conformance]", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("http", endpointPod.Status.PodIP, framework.EndpointHttpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for intra-pod communication: udp [Conformance]", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromTestContainer("udp", endpointPod.Status.PodIP, framework.EndpointUdpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for node-pod communication: http [Conformance]", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("http", endpointPod.Status.PodIP, framework.EndpointHttpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})

		It("should function for node-pod communication: udp [Conformance]", func() {
			config := framework.NewCoreNetworkingTestConfig(f)
			for _, endpointPod := range config.EndpointPods {
				config.DialFromNode("udp", endpointPod.Status.PodIP, framework.EndpointUdpPort, config.MaxTries, 0, sets.NewString(endpointPod.Name))
			}
		})
	})
})
