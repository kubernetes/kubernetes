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
	"strings"
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
	// The kubelet computes memory.available as the node capacity minus the
	// working set, from one cgroup sample, and subtracts the hugepage capacity
	// when the HugepageAwareEviction gate is enabled. Each test checks that
	// identity on a single /stats/summary sample. A before and after comparison
	// across the kubelet restart flakes on the memory the restart itself moves.
	// identityEpsilon only absorbs skew between the node capacity and the root
	// cgroup limit, and stays far below the 64Mi reservation.
	identityEpsilon := uint64(16 * 1024 * 1024)

	// getMemorySample returns AvailableBytes and WorkingSetBytes from one
	// /stats/summary response, so both come from the same cgroup sample.
	getMemorySample := func(ctx context.Context) (uint64, uint64) {
		var available, workingSet uint64
		gomega.Eventually(ctx, func() error {
			summary, err := getNodeSummary(ctx)
			if err != nil {
				return err
			}
			if summary == nil || summary.Node.Memory == nil ||
				summary.Node.Memory.AvailableBytes == nil || summary.Node.Memory.WorkingSetBytes == nil {
				return fmt.Errorf("memory stats not yet available")
			}
			available = *summary.Node.Memory.AvailableBytes
			workingSet = *summary.Node.Memory.WorkingSetBytes
			return nil
		}).WithTimeout(time.Minute).WithPolling(2 * time.Second).Should(gomega.Succeed())
		return available, workingSet
	}

	getNodeMemoryCapacity := func(ctx context.Context) int64 {
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		cap := node.Status.Capacity[v1.ResourceMemory]
		return cap.Value()
	}

	// getNodeHugepageCapacity returns the sum of all hugepages-* resources in
	// the node capacity. The kubelet subtracts this same sum, so the identity
	// must include hugepages the node already had, for example preallocated
	// 1Gi pages, not only the 2Mi pages this test sets.
	getNodeHugepageCapacity := func(ctx context.Context) uint64 {
		node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		var total uint64
		for name, quantity := range node.Status.Capacity {
			if strings.HasPrefix(string(name), v1.ResourceHugePagesPrefix) {
				total += uint64(quantity.Value())
			}
		}
		return total
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

			ginkgo.By(fmt.Sprintf("Allocating %d x 2Mi hugepages", hugepageCount))
			setHugepages(ctx, hugepages)
			defer releaseHugepages(ctx, hugepages)

			ginkgo.By("Restarting kubelet to pick up hugepage allocation")
			restartKubelet(ctx, true)
			waitForHugepages(f, ctx, hugepages)

			capacityAfter := getNodeMemoryCapacity(ctx)
			gomega.Expect(capacityAfter).To(gomega.Equal(capacityBefore),
				"node memory capacity must not change")

			hugepageBytes := getNodeHugepageCapacity(ctx)
			gomega.Expect(hugepageBytes).To(gomega.BeNumerically(">=", expectedHugepageBytes),
				"node capacity must report at least the hugepage reservation")

			available, workingSet := getMemorySample(ctx)

			framework.Logf("enabled: capacity=%d available=%d workingSet=%d hugepages=%d",
				capacityAfter, available, workingSet, hugepageBytes)

			gomega.Expect(available+workingSet+hugepageBytes).To(
				gomega.BeNumerically("~", capacityAfter, identityEpsilon),
				"memory.available should decrease by the hugepage reservation")
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

			ginkgo.By(fmt.Sprintf("Allocating %d x 2Mi hugepages", hugepageCount))
			setHugepages(ctx, hugepages)
			defer releaseHugepages(ctx, hugepages)

			ginkgo.By("Restarting kubelet to pick up hugepage allocation")
			restartKubelet(ctx, true)
			waitForHugepages(f, ctx, hugepages)

			capacity := getNodeMemoryCapacity(ctx)
			available, workingSet := getMemorySample(ctx)

			framework.Logf("disabled: capacity=%d available=%d workingSet=%d hugepages=%d",
				capacity, available, workingSet, expectedHugepageBytes)

			// With the feature gate disabled, the hugepage reservation is NOT
			// subtracted from memory.available.
			gomega.Expect(available+workingSet).To(
				gomega.BeNumerically("~", capacity, identityEpsilon),
				"memory.available should NOT decrease by the hugepage reservation when feature gate is disabled")
		})
	})
})
