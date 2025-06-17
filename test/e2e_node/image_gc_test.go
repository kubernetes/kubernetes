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
	"fmt"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internalapi "k8s.io/cri-api/pkg/apis"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	// Kubelet GC's on a frequency of once every 5 minutes
	// Add a little leeway to give it time.
	checkGCUntil time.Duration = 6 * time.Minute
	checkGCFreq  time.Duration = 30 * time.Second
)

var _ = SIGDescribe("ImageGarbageCollect", framework.WithSerial(), feature.GarbageCollect, func() {
	f := framework.NewDefaultFramework("image-garbage-collect-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var is internalapi.ImageManagerService
	ginkgo.BeforeEach(func() {
		var err error
		_, is, err = getCRIClient()
		framework.ExpectNoError(err)
	})
	ginkgo.AfterEach(func(ctx context.Context) {
		framework.ExpectNoError(PrePullAllImages(ctx))
	})
	ginkgo.Context("when ImageMaximumGCAge is set", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.ImageMaximumGCAge = metav1.Duration{Duration: time.Duration(time.Minute * 1)}
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

			e2epod.NewPodClient(f).DeleteSync(ctx, pod.ObjectMeta.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			// Even though the image gc max timing is less, we are bound by the kubelet's
			// ImageGCPeriod, which is hardcoded to 5 minutes.
			gomega.Eventually(ctx, func() int {
				gcdImageList, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
				framework.ExpectNoError(err)
				return len(gcdImageList)
			}, checkGCUntil, checkGCFreq).Should(gomega.BeNumerically("<", len(allImages)))
		})
		ginkgo.It("should not GC unused images prematurely", func(ctx context.Context) {
			pod := innocentPod()
			e2epod.NewPodClient(f).CreateBatch(ctx, []*v1.Pod{pod})

			_, err := is.PullImage(context.Background(), &runtimeapi.ImageSpec{Image: agnhostImage}, nil, nil)
			framework.ExpectNoError(err)

			allImages, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
			framework.ExpectNoError(err)

			e2epod.NewPodClient(f).DeleteSync(ctx, pod.ObjectMeta.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			restartKubelet(ctx, true)

			// Wait until the maxAge of the image after the kubelet is restarted to ensure it doesn't
			// GC too early.
			gomega.Consistently(ctx, func() int {
				gcdImageList, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
				framework.ExpectNoError(err)
				return len(gcdImageList)
			}, 50*time.Second, 10*time.Second).Should(gomega.Equal(len(allImages)))

			// Even though the image gc max timing is less, we are bound by the kubelet's
			// ImageGCPeriod, which is hardcoded to 5 minutes.
			gomega.Eventually(ctx, func() int {
				gcdImageList, err := is.ListImages(context.Background(), &runtimeapi.ImageFilter{})
				framework.ExpectNoError(err)
				return len(gcdImageList)
			}, checkGCUntil, checkGCFreq).Should(gomega.BeNumerically("<", len(allImages)))
		})
	})
	ginkgo.Context("Concurrent garbage collection stability", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.EvictionHard = map[string]string{
				"imagefs.available": "15%",
			}
			initialConfig.EvictionMinimumReclaim = map[string]string{
				"imagefs.available": "1GB",
			}
			// Configure aggressive GC thresholds
			initialConfig.ImageGCHighThresholdPercent = 50
			initialConfig.ImageGCLowThresholdPercent = 40
			initialConfig.ImageMinimumGCAge = metav1.Duration{Duration: 0}
		})

		ginkgo.It("should handle concurrent image deletions without crashing under DiskPressure", func(ctx context.Context) {
			resp, err := is.ImageFsInfo(ctx)
			framework.ExpectNoError(err)
			gomega.Expect(resp.ImageFilesystems).NotTo(gomega.BeEmpty())
			gomega.Expect(resp.ImageFilesystems[0].FsId).NotTo(gomega.BeNil())
			diskToPressure := filepath.Dir(resp.ImageFilesystems[0].FsId.Mountpoint)
			ginkgo.By(fmt.Sprintf("Got imageFs directory: %s", diskToPressure))

			var imagesLenBeforeGC int
			gomega.Eventually(ctx, func(ctx context.Context) error {
				images, err := is.ListImages(ctx, &runtimeapi.ImageFilter{})
				if err != nil {
					return err
				}
				imagesLenBeforeGC = len(images)
				return nil
			}, 1*time.Minute, 5*time.Second).Should(gomega.Succeed())
			ginkgo.By(fmt.Sprintf("Number of images found before GC: %d", imagesLenBeforeGC))

			const numPods = 15
			image := imageutils.GetE2EImage(imageutils.Agnhost)
			var pods []*v1.Pod

			for i := 0; i < numPods; i++ {
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:      fmt.Sprintf("gc-stress-pod-%d", i),
						Namespace: f.Namespace.Name,
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:    fmt.Sprintf("container-%d", i),
							Image:   image,
							Command: []string{"/bin/sh", "-c", "sleep 3600"},
						}},
					},
				}
				pods = append(pods, pod)
			}

			e2epod.NewPodClient(f).CreateBatch(ctx, pods)
			for _, pod := range pods {
				e2epod.NewPodClient(f).DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
			}

			kubeletHealthy := true
			watchdogCtx, watchdogCancel := context.WithCancel(ctx)
			defer watchdogCancel()
			startHealthMonitor(watchdogCtx, &kubeletHealthy)

			sizeOfPressure := "8000" // 8GB - proven effective size
			ginkgo.By(fmt.Sprintf("Inducing disk pressure on %s with size %s MB", diskToPressure, sizeOfPressure))
			gomega.Expect(runDDOnFilesystem(diskToPressure, sizeOfPressure)).Should(gomega.Succeed())

			ginkgo.By("Waiting for NodeDiskPressure condition")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				if hasNodeCondition(ctx, f, v1.NodeDiskPressure) {
					return nil
				}
				return fmt.Errorf("NodeDiskPressure condition not active")
			}, 5*time.Minute, 5*time.Second).Should(gomega.Succeed())

			ginkgo.By("Waiting for garbage collection to complete")
			gomega.Eventually(ctx, func() bool {
				currentResp, err := is.ListImages(ctx, &runtimeapi.ImageFilter{})
				if err != nil {
					return false
				}
				return len(currentResp) < imagesLenBeforeGC
			}, 5*time.Minute, 30*time.Second).Should(gomega.BeTrueBecause("GC should reduce image count"))

			gomega.Expect(kubeletHealthy).To(gomega.BeTrueBecause("Kubelet should remain healthy"))

			// 9. Cleanup (from split_disk_test)
			ginkgo.By("Removing disk pressure")
			gomega.Expect(removeDiskPressure(diskToPressure)).Should(gomega.Succeed())

			// 10. Verify pressure cleared (from split_disk_test)
			ginkgo.By("Waiting for DiskPressure to clear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				if hasNodeCondition(ctx, f, v1.NodeDiskPressure) {
					return fmt.Errorf("DiskPressure still active")
				}
				return nil
			}, 2*time.Minute, 5*time.Second).Should(gomega.Succeed())
		})
	})
})

func startHealthMonitor(ctx context.Context, healthy *bool) {
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if !kubeletHealthCheck(kubeletHealthCheckURL) {
					*healthy = false
					return
				}
			}
		}
	}()
}
