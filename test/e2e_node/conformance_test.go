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
	imageWaitTimeout  = time.Minute * 2
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
				"gcr.io/google_containers/busybox:1.24",
				"gcr.io/google_containers/mounttest:0.2",
				"gcr.io/google_containers/nettest:1.7",
				"gcr.io/google_containers/nginx:1.7.9",
			}
			It("it should pull successfully [Conformance]", func() {
				for _, imageTag := range conformImageTags {
					image, _ := NewConformanceImage("docker", imageTag)
					conformImages = append(conformImages, image)
				}
				for _, image := range conformImages {
					// Pulling images from gcr.io is flaky, so retry failures
					Eventually(func() error {
						return image.Pull()
					}, imageWaitTimeout, imagePullInterval).ShouldNot(HaveOccurred())
				}
			})
			It("it should list pulled images [Conformance]", func() {
				image, _ := NewConformanceImage("docker", "")
				tags, _ := image.List()
				for _, tag := range conformImageTags {
					Expect(tags).To(ContainElement(tag))
				}
			})
			It("it should remove successfully [Conformance]", func() {
				for _, image := range conformImages {
					if err := image.Remove(); err != nil {
						Expect(err).NotTo(HaveOccurred())
						break
					}
				}
			})
		})
		Context("when testing image that does not exist", func() {
			var invalidImage ConformanceImage
			var invalidImageTag string
			It("it should not pull successfully [Conformance]", func() {
				invalidImageTag = "foo.com/foo/foo"
				invalidImage, _ = NewConformanceImage("docker", invalidImageTag)
				err := invalidImage.Pull()
				Expect(err).To(HaveOccurred())
			})
			It("it should not list pulled images [Conformance]", func() {
				image, _ := NewConformanceImage("docker", "")
				tags, _ := image.List()
				Expect(tags).NotTo(ContainElement(invalidImageTag))
			})
			It("it should not remove successfully [Conformance]", func() {
				err := invalidImage.Remove()
				Expect(err).To(HaveOccurred())
			})
		})
		Context("when running a container that terminates", func() {
			It("should be able to create, inspect and delete it [Conformance]", func() {
				c := NewConformanceContainer(cl, *nodeName, api.Container{
					Image:           "gcr.io/google_containers/busybox:1.24",
					Name:            "terminate-container",
					Command:         []string{"sh", "-c", "env"},
					ImagePullPolicy: api.PullIfNotPresent,
				})

				By("create the container")
				defer c.Delete()
				err := c.Create()
				Expect(err).NotTo(HaveOccurred())

				By("wait up to 2m for the container to become  terminated")
				// TODO: Check that the container enters running state by sleeping in the container #23309
				c.Wait(2*time.Minute, func(s *api.ContainerStatus) bool { return s.State.Terminated != nil })

				By("check the container status")
				status, err := c.Status()
				Expect(err).NotTo(HaveOccurred())
				Expect(isContainerSucceed(status)).Should(BeTrue())

				By("delete the container")
				err = c.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
			It("should report termination message if TerminationMessagePath is set [Conformance]", func() {
				terminationMessage := "DONE"
				terminationMessagePath := "/dev/termination-log"
				c := NewConformanceContainer(cl, *nodeName, api.Container{
					Image:   "gcr.io/google_containers/busybox:1.24",
					Name:    "termination-message-container",
					Command: []string{"/bin/sh", "-c"},
					Args:    []string{"/bin/echo -n " + terminationMessage + " > " + terminationMessagePath},
					TerminationMessagePath: terminationMessagePath,
					ImagePullPolicy:        api.PullIfNotPresent,
				})

				By("create the container")
				defer c.Delete()
				err := c.Create()
				Expect(err).NotTo(HaveOccurred())

				By("wait up to 2m for the container to become terminated")
				c.Wait(2*time.Minute, func(s *api.ContainerStatus) bool { return s.State.Terminated != nil })

				By("check the termination message")
				status, err := c.Status()
				Expect(err).NotTo(HaveOccurred())
				Expect(isContainerSucceed(status)).Should(BeTrue())
				Expect(status.State.Terminated.Message).Should(Equal(terminationMessage))

				By("delete the container")
				err = c.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
		Context("when running a container with invalid image", func() {
			It("it should not start successfully [Conformance]", func() {
				c := NewConformanceContainer(cl, *nodeName, api.Container{
					Image:           "foo.com/foo/foo",
					Name:            "invalid-image-container",
					Command:         []string{"foo", "'Should not work'"},
					ImagePullPolicy: api.PullIfNotPresent,
				})

				By("create the container")
				defer c.Delete()
				err := c.Create()
				Expect(err).NotTo(HaveOccurred())

				By("wait up to 2m for the container to become waiting")
				c.Wait(2*time.Minute, func(s *api.ContainerStatus) bool { return s.State.Waiting != nil })

				By("wait for 20 seconds to make sure the container is always waiting")
				c.Always(20*time.Second, func(s *api.ContainerStatus) bool { return s.State.Waiting != nil })

				By("delete the container")
				err = c.Delete()
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})
