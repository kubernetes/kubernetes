/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	retryTimeout = time.Minute * 4
	pollInterval = time.Second * 5
)

var _ = Describe("Container Runtime Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})

	Describe("container conformance blackbox test", func() {
		Context("when running a container that terminates", func() {
			var containerCase ConformanceContainer

			BeforeEach(func() {
				containerCase = ConformanceContainer{
					Container: api.Container{
						Image:           "gcr.io/google_containers/busybox",
						Name:            "busybox",
						Command:         []string{"sh", "-c", "env"},
						ImagePullPolicy: api.PullIfNotPresent,
					},
					Client:   cl,
					Phase:    api.PodSucceeded,
					NodeName: *nodeName,
				}
				err := containerCase.Create()
				Expect(err).NotTo(HaveOccurred())
			})

			It("it should report its phase as 'succeeded' and get a same container [Conformance]", func() {
				var container ConformanceContainer
				var err error
				Eventually(func() api.PodPhase {
					container, err = containerCase.Get()
					return container.Phase
				}, retryTimeout, pollInterval).Should(Equal(api.PodSucceeded))
				Expect(err).NotTo(HaveOccurred())
				Expect(container).Should(CContainerEqual(containerCase))
			})

			It("it should be possible to delete [Conformance]", func() {
				err := containerCase.Delete()
				Expect(err).NotTo(HaveOccurred())
				Eventually(func() bool {
					isPresent, err := containerCase.Present()
					return err == nil && !isPresent
				}, retryTimeout, pollInterval).Should(BeTrue())
			})

			AfterEach(func() {
				containerCase.Delete()
				Eventually(func() bool {
					isPresent, err := containerCase.Present()
					return err == nil && !isPresent
				}, retryTimeout, pollInterval).Should(BeTrue())
			})

		})
		Context("when running a container with invalid image", func() {
			var containerCase ConformanceContainer
			BeforeEach(func() {
				containerCase = ConformanceContainer{
					Container: api.Container{
						Image:           "foo.com/foo/foo",
						Name:            "foo",
						Command:         []string{"foo", "'Should not work'"},
						ImagePullPolicy: api.PullIfNotPresent,
					},
					Client:   cl,
					Phase:    api.PodPending,
					NodeName: *nodeName,
				}
				err := containerCase.Create()
				Expect(err).NotTo(HaveOccurred())
			})

			It("it should report its phase as 'pending' and get a same container [Conformance]", func() {
				var container ConformanceContainer
				var err error
				Consistently(func() api.PodPhase {
					container, err = containerCase.Get()
					return container.Phase
				}, retryTimeout, pollInterval).Should(Equal(api.PodPending))
				Expect(err).NotTo(HaveOccurred())
				Expect(container).Should(CContainerEqual(containerCase))
			})

			It("it should be possible to delete [Conformance]", func() {
				err := containerCase.Delete()
				Expect(err).NotTo(HaveOccurred())
				Eventually(func() bool {
					isPresent, err := containerCase.Present()
					return err == nil && !isPresent
				}, retryTimeout, pollInterval).Should(BeTrue())
			})

			AfterEach(func() {
				containerCase.Delete()
				Eventually(func() bool {
					isPresent, err := containerCase.Present()
					return err == nil && !isPresent
				}, retryTimeout, pollInterval).Should(BeTrue())
			})
		})
	})
})
