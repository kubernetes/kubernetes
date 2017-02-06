/*
Copyright 2017 The Kubernetes Authors.

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

package e2e_node

import (
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Port forwarding", func() {
	f := framework.NewDefaultFramework("port-forwarding")
	var command framework.PortForwardCommand

	BeforeEach(func() {
		command = framework.MakePortForwardCommand(f)
	})

	Context("With a server listening on 0.0.0.0", func() {
		Context("that expects a client request", func() {
			It("should support a client that connects, sends no data, and disconnects", func() {
				framework.DoTestMustConnectSendNothing(command, "0.0.0.0", f)
			})
			It("should support a client that connects, sends data, and disconnects", func() {
				framework.DoTestMustConnectSendDisconnect(command, "0.0.0.0", f)
			})
		})

		Context("that expects no client request", func() {
			It("should support a client that connects, sends data, and disconnects", func() {
				framework.DoTestConnectSendDisconnect(command, "0.0.0.0", f)
			})
		})

		It("should support forwarding over websockets", func() {
			framework.DoTestOverWebSockets("0.0.0.0", f)
		})
	})

	Context("With a server listening on localhost", func() {
		Context("that expects a client request", func() {
			It("should support a client that connects, sends no data, and disconnects [Conformance]", func() {
				framework.DoTestMustConnectSendNothing(command, "localhost", f)
			})
			It("should support a client that connects, sends data, and disconnects [Conformance]", func() {
				framework.DoTestMustConnectSendDisconnect(command, "localhost", f)
			})
		})

		Context("that expects no client request", func() {
			It("should support a client that connects, sends data, and disconnects [Conformance]", func() {
				framework.DoTestConnectSendDisconnect(command, "localhost", f)
			})
		})

		It("should support forwarding over websockets", func() {
			framework.DoTestOverWebSockets("localhost", f)
		})
	})
})
