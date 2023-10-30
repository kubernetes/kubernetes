/*
Copyright 2024 The Kubernetes Authors.

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

package e2enode

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

// var _ = SIGDescribe("ImageGarbageCollect [Serial][NodeFeature:GarbageCollect]", func() {
var _ = SIGDescribe("ImageGarbageCollect", func() {
	f := framework.NewDefaultFramework("image-garbage-collect-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var is internalapi.ImageManagerService
	ginkgo.BeforeEach(func() {
		var err error
		_, is, err = getCRIClient()
		framework.ExpectNoError(err)
	})
	ginkgo.AfterEach(func() {
		// framework.ExpectNoError(PrePullAllImages())
	})
	ginkgo.Context("when ImageMaximumGCAge is set", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.ImageMaximumGCAge = metav1.Duration{Duration: time.Duration(time.Second * 30)}
			initialConfig.ImageMinimumGCAge = metav1.Duration{Duration: time.Duration(time.Second * 1)}
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates[string(kubefeatures.ImageMaximumGCAge)] = true
		})
		ginkgo.It("should GC unused images", func(ctx context.Context) {
			pod := innocentPod()
			e2epod.NewPodClient(f).CreateBatch(ctx, []*v1.Pod{pod})

			_, err := is.PullImage(context.Background(), &runtimeapi.ImageSpec{Image: agnhostImage}, nil, nil)
			framework.ExpectNoError(err)

			allImages, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
			framework.ExpectNoError(err)

			e2epod.NewPodClient(f).DeleteSync(ctx, pod.ObjectMeta.Name, metav1.DeleteOptions{}, e2epod.DefaultPodDeletionTimeout)

			time.Sleep(time.Minute * 2)

			// All images but one is unused, so expect them to be garbage collected
			gcdImageList, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
			gomega.Expect(len(allImages)).To(gomega.BeNumerically(">", len(gcdImageList)))
		})
	})
})
