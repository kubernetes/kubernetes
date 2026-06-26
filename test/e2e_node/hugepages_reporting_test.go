/*
Copyright The Kubernetes Authors.

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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

// HugepageAwareMemoryReporting verifies that the HugepageAwareEviction feature
// gate controls whether hugepage capacity is subtracted from memory.available
// in node stats reporting.
var _ = SIGDescribe("HugepageAwareMemoryReporting", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.HugePages, func() {
	f := framework.NewDefaultFramework("hugepage-aware-reporting-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	hugepageCount := 32 // 32 x 2Mi = 64Mi
	hugepages := map[string]int{hugepagesResourceName2Mi: hugepageCount}
	expectedHugepageBytes := uint64(hugepageCount) * 2 * 1024 * 1024
	margin := uint64(50 * 1024 * 1024)

	getAvailableMemory := func(ctx context.Context) uint64 {
		var available uint64
		gomega.Eventually(ctx, func() error {
			summary, err := getNodeSummary(ctx)
			if err != nil {
				return err
			}
			if summary == nil || summary.Node.Memory == nil || summary.Node.Memory.AvailableBytes == nil {
				return fmt.Errorf("memory stats not yet available")
			}
			available = *summary.Node.Memory.AvailableBytes
			return nil
		}).WithTimeout(time.Minute).WithPolling(2 * time.Second).Should(gomega.Succeed())
		return available
	}

	getNodeMemoryCapacity := func(ctx context.Context) int64 {
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		cap := node.Status.Capacity[v1.ResourceMemory]
		return cap.Value()
	}

	ginkgo.Context("with feature gate enabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates["HugepageAwareEviction"] = true
		})

		ginkgo.It("should subtract hugepage capacity from memory.available", func(ctx context.Context) {
			if !isHugePageAvailable(hugepagesSize2M) {
				e2eskipper.Skipf("skipping: 2Mi hugepages not supported on this node")
			}

			capacityBefore := getNodeMemoryCapacity(ctx)
			availableBefore := getAvailableMemory(ctx)

			ginkgo.By(fmt.Sprintf("Allocating %d x 2Mi hugepages", hugepageCount))
			setHugepages(ctx, hugepages)
			defer releaseHugepages(ctx, hugepages)

			ginkgo.By("Restarting kubelet to pick up hugepage allocation")
			restartKubelet(ctx, true)
			waitForHugepages(f, ctx, hugepages)

			capacityAfter := getNodeMemoryCapacity(ctx)
			gomega.Expect(capacityAfter).To(gomega.Equal(capacityBefore),
				"node memory capacity must not change")

			availableAfter := getAvailableMemory(ctx)

			drop := uint64(0)
			if availableBefore > availableAfter {
				drop = availableBefore - availableAfter
			}

			framework.Logf("enabled: availableBefore=%d availableAfter=%d drop=%d hugepages=%d",
				availableBefore, availableAfter, drop, expectedHugepageBytes)

			gomega.Expect(drop).To(
				gomega.BeNumerically(">=", expectedHugepageBytes-margin),
				"memory.available should decrease by at least the hugepage reservation")
		})
	})

	ginkgo.Context("with feature gate disabled", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
			initialConfig.FeatureGates["HugepageAwareEviction"] = false
		})

		ginkgo.It("should include hugepage capacity in memory.available", func(ctx context.Context) {
			if !isHugePageAvailable(hugepagesSize2M) {
				e2eskipper.Skipf("skipping: 2Mi hugepages not supported on this node")
			}

			availableBefore := getAvailableMemory(ctx)

			ginkgo.By(fmt.Sprintf("Allocating %d x 2Mi hugepages", hugepageCount))
			setHugepages(ctx, hugepages)
			defer releaseHugepages(ctx, hugepages)

			ginkgo.By("Restarting kubelet to pick up hugepage allocation")
			restartKubelet(ctx, true)
			waitForHugepages(f, ctx, hugepages)

			availableAfter := getAvailableMemory(ctx)

			framework.Logf("disabled: availableBefore=%d availableAfter=%d hugepages=%d",
				availableBefore, availableAfter, expectedHugepageBytes)

			// With the feature gate disabled, hugepage reservation is NOT subtracted
			// from memory.available. The cgroup-based calculation still counts hugepage
			// memory as available, so the drop should be much smaller than the
			// hugepage reservation.
			drop := uint64(0)
			if availableBefore > availableAfter {
				drop = availableBefore - availableAfter
			}
			gomega.Expect(drop).To(
				gomega.BeNumerically("<", expectedHugepageBytes-margin),
				"memory.available should NOT decrease by the hugepage reservation when feature gate is disabled")
		})
	})
})
