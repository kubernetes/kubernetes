/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

var _ = Describe("Container Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&client.Config{Host: *apiServerAddress})
	})

	Describe("container conformance blackbox test", func() {
		Context("when running a container that terminates", func() {
			var terminateCase ConformanceContainer
			BeforeEach(func() {
				terminateCase = ConformanceContainer{
					Container: api.Container{
						Image:           "gcc.io/google_testcontainers/busybox",
						Name:            "busybox",
						Command:         []string{"echo", "'Hello World'"},
						ImagePullPolicy: api.PullIfNotPresent,
					},
					Status: "terminated",
					Client: cl,
				}
			})
			It("it should start successfully [Conformance]", func() {
				err := terminateCase.Create()
				Expect(err).NotTo(HaveOccurred())

				Eventually(func() bool {
					return terminateCase.IsStarted()
				}, time.Minute*1, time.Second*30).Should(BeTrue())
			})
			It("it should report its status as 'terminated' [Conformance]", func() {
				ccontainer, err := terminateCase.Get()
				Expect(err).NotTo(HaveOccurred())
				Expect(ccontainer).Should(CContainerEqual(terminateCase))
			})
			It("it should be possible to delete [Conformance]", func() {
				err := terminateCase.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
		Context("when running a container with invalid image", func() {
			var invalidImageCase ConformanceContainer
			BeforeEach(func() {
				invalidImageCase = ConformanceContainer{
					Container: api.Container{
						Image:           "foo.com/foo/foo",
						Name:            "foo",
						Command:         []string{"foo", "'Should not work'"},
						ImagePullPolicy: api.PullIfNotPresent,
					},
					Status: "waiting",
					Client: cl,
				}
			})
			It("it should not start successfully [Conformance]", func() {
				err := invalidImageCase.Create()
				Expect(err).NotTo(HaveOccurred())

				Eventually(func() bool {
					return invalidImageCase.IsStarted()
				}, time.Minute*1, time.Second*30).ShouldNot(BeTrue())
			})
			It("it should report its status as 'waiting' [Conformance]", func() {
				ccontainer, err := invalidImageCase.Get()
				Expect(err).NotTo(HaveOccurred())
				Expect(ccontainer).Should(CContainerEqual(invalidImageCase))
			})
			It("it should be possible to delete [Conformance]", func() {
				err := invalidImageCase.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})
