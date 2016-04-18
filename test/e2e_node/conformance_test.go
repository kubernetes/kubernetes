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
	retryTimeout      = time.Minute * 4
	pollInterval      = time.Second * 5
	imageRetryTimeout = time.Minute * 2
	imagePullInterval = time.Second * 15
)

var _ = Describe("Container Conformance Test", func() {
	var cl *client.Client

	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&restclient.Config{Host: *apiServerAddress})
	})

	Describe("container conformance blackbox test", func() {
		Context("when testing images that exist", func() {
			var conformImages []ConformanceImage
			conformImageTags := []string{
				"gcr.io/google_containers/node-conformance:v1",
				"gcr.io/google_containers/node-conformance:v2",
				"gcr.io/google_containers/node-conformance:v3",
				"gcr.io/google_containers/node-conformance:v4",
			}
			// TODO(random-liu): Each It should be independent, we should not let them depend on
			// each other.
			It("should pull successfully [Conformance]", func() {
				for _, imageTag := range conformImageTags {
					image, _ := NewConformanceImage("docker", imageTag)
					conformImages = append(conformImages, image)
				}
				for _, image := range conformImages {
					// Pulling images from gcr.io is flaky, so retry failures
					Eventually(func() error {
						return image.Pull()
					}, imageRetryTimeout, imagePullInterval).ShouldNot(HaveOccurred())
					present, err := image.Present()
					Expect(err).ShouldNot(HaveOccurred())
					Expect(present).Should(BeTrue())
				}
			})
			It("should list pulled images [Conformance]", func() {
				image, _ := NewConformanceImage("docker", "")
				tags, _ := image.List()
				for _, tag := range conformImageTags {
					Expect(tags).To(ContainElement(tag))
				}
			})
			It("should remove successfully [Conformance]", func() {
				for _, image := range conformImages {
					err := image.Remove()
					Expect(err).NotTo(HaveOccurred())
					present, err := image.Present()
					Expect(err).NotTo(HaveOccurred())
					Expect(present).To(BeFalse())
				}
			})
		})
		Context("when testing image that does not exist", func() {
			It("should not pull successfully [Conformance]", func() {
				invalidImageTags := []string{
					// nonexistent image registry
					"foo.com/foo/foo",
					// nonexistent image
					"gcr.io/google_containers/not_exist",
					// TODO(random-liu): Add test for image pulling credential
				}
				for _, invalidImageTag := range invalidImageTags {
					invalidImage, _ := NewConformanceImage("docker", invalidImageTag)
					By("start pulling image")
					err := invalidImage.Pull()
					Expect(err).Should(HaveOccurred())

					By("check image present")
					present, err := invalidImage.Present()
					Expect(err).ShouldNot(HaveOccurred())
					Expect(present).To(BeFalse())

					By("listing image")
					tags, err := invalidImage.List()
					Expect(err).ShouldNot(HaveOccurred())
					Expect(tags).NotTo(ContainElement(invalidImage))

					By("removing image")
					err = invalidImage.Remove()
					Expect(err).Should(HaveOccurred())
				}
			})
		})
		Context("when running a container that terminates", func() {
			var terminateCase ConformanceContainer
			It("should run successfully to completion [Conformance]", func() {
				terminateCase = ConformanceContainer{
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
				err := terminateCase.Create()
				Expect(err).NotTo(HaveOccurred())

				// TODO: Check that the container enters running state by sleeping in the container #23309
				Eventually(func() (api.PodPhase, error) {
					pod, err := terminateCase.Get()
					return pod.Phase, err
				}, retryTimeout, pollInterval).Should(Equal(terminateCase.Phase))
			})
			It("should report its phase as 'succeeded' [Conformance]", func() {
				ccontainer, err := terminateCase.Get()
				Expect(err).NotTo(HaveOccurred())
				Expect(ccontainer).Should(CContainerEqual(terminateCase))
			})
			It("should be possible to delete [Conformance]", func() {
				err := terminateCase.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
		Context("when running a container with invalid image", func() {
			var invalidImageCase ConformanceContainer
			It("should not start successfully [Conformance]", func() {
				invalidImageCase = ConformanceContainer{
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
				err := invalidImageCase.Create()
				Expect(err).NotTo(HaveOccurred())
				Eventually(func() (api.PodPhase, error) {
					pod, err := invalidImageCase.Get()
					return pod.Phase, err
				}, retryTimeout, pollInterval).Should(Equal(invalidImageCase.Phase))
			})
			It("should report its phase as 'pending' [Conformance]", func() {
				ccontainer, err := invalidImageCase.Get()
				Expect(err).NotTo(HaveOccurred())
				Expect(ccontainer).Should(CContainerEqual(invalidImageCase))
			})
			It("should be possible to delete [Conformance]", func() {
				err := invalidImageCase.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})
