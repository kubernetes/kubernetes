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

package e2e_cri

import (
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	"k8s.io/kubernetes/test/e2e_cri/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	// image name for test image api
	testImageName = "busybox"

	// name-tagged reference for latest test image
	latestTestImageRef = testImageName + ":latest"

	// Digested reference for test image
	busyboxDigestRef = testImageName + "@sha256:a59906e33509d14c036c8678d687bd4eec81ed7c4b8ce907b888c607f6a1e0e6"
)

var _ = framework.KubeDescribe("Test CRI image manager", func() {
	f := framework.NewDefaultCRIFramework("CRI-image-manager-test")

	var c internalapi.ImageManagerService

	BeforeEach(func() {
		c = f.CRIClient.CRIImageClient
	})

	It("public image with tag should be pulled and removed", func() {
		testPullPublicImage(c, latestTestImageRef)
	})

	It("public image without tag should be pulled and removed", func() {
		testPullPublicImage(c, testImageName)
	})

	It("public image with digest should be pulled and removed", func() {
		testPullPublicImage(c, busyboxDigestRef)
	})

	It("image status get image fields should not be empty", func() {
		pullPublicImageOrFail(c, latestTestImageRef)

		defer removeImage(c, latestTestImageRef)

		status := imageStatusOrFail(c, latestTestImageRef)
		Expect(status.Id).NotTo(BeNil(), "image Id should not be nil")
		Expect(len(status.RepoTags)).NotTo(Equal(0), "should have repoTags in image")
		Expect(status.Size_).NotTo(BeNil(), "image Size should not be nil")
	})

	It("ListImage should get exactly 3 image in the result list", func() {
		// different tags refer to different images
		testImageList := []string{
			"busybox:1",
			"busybox:musl",
			"busybox:glibc",
		}

		pullImageList(c, testImageList)

		defer removeImageList(c, testImageList)

		images := listImagesOrFail(c, testImageName)
		Expect(len(images)).To(Equal(3), "should have exactly three images in list")
	})

	It("ListImage should get exactly 3 repoTags in the result image", func() {
		// different tags refer to the same image
		testImageList := []string{
			"busybox:1",
			"busybox:1.25",
			"busybox:uclibc",
		}

		pullImageList(c, testImageList)

		defer removeImageList(c, testImageList)

		images := listImagesOrFail(c, testImageName)

		Expect(len(images)).To(Equal(1), "should have only one image in list")
		Expect(len(images[0].RepoTags)).To(Equal(3), "should have three repoTags in image")
		Expect(images[0].RepoTags).To(ContainElement("busybox:1"), "should have the repoTag named busybox:1")
		Expect(images[0].RepoTags).To(ContainElement("busybox:1.25"), "should have the repoTag named busybox:1.25")
		Expect(images[0].RepoTags).To(ContainElement("busybox:uclibc"), "should have the repoTag named busybox:uclibc")
	})
})
