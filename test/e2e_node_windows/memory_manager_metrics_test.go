//go:build windows

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

package e2enodewindows

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

// On Windows the memory manager runs the "BestEffort" policy (there is no
// cpuset.mems equivalent, so memory placement follows the CPU manager's NUMA
// decision on a best-effort basis). The BestEffort policy delegates allocation
// to the same logic as the static policy, so it still increments the
// kubelet_memory_manager_pinning_* counters. These are plain counters (not
// label vectors), so unlike the topology manager numa_node series they are
// created at registration time and read 0 on a fresh restart.
var _ = SIGWindowsDescribe(feature.MemoryManager, feature.Windows, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("memorymanager-metrics-windows")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("when querying /metrics", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			// Switching the memory manager policy (None <-> BestEffort) invalidates
			// the persisted state checkpoint, so the kubelet refuses to start unless
			// the memory_manager_state file is removed first. Clear it on the way in.
			updateWindowsKubeletConfigClearState(ctx, f, buildWindowsMemoryManagerKubeletConfig(oldCfg))
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				testPod = nil
			}
			// Restoring the original (None) policy again changes the memory manager
			// policy, which likewise requires clearing the state checkpoint.
			updateWindowsKubeletConfigClearState(ctx, f, oldCfg)
		})

		ginkgo.It("should report zero pinning counters after a fresh restart", func(ctx context.Context) {
			// We restarted the kubelet in BeforeEach and, being [Serial], no other
			// pods are running, so the pinning counters must be zero.
			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should report pinning failures when the memory manager allocation is known to fail", func(ctx context.Context) {
			// The memory manager only records a pinning error when it makes its OWN
			// NUMA-placement decision and that Allocate fails. That path is reached
			// only by a Guaranteed pod with a FRACTIONAL CPU request: it gets no
			// exclusive CPUs, so the memory manager does not follow the CPU manager's
			// node and instead selects/extends NUMA nodes itself. An integer CPU
			// request would take the follow-CPU path, which binds best-effort and
			// never errors (the pod would just be rejected later by the node-fit
			// predicate). The oversized memory request fits no NUMA node, so Allocate
			// fails and both the pinning request and error counters increment.
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedMemoryManagerPodWindows("memmngr-fail", "500m", "1000Gi"))

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})

		ginkgo.It("should not report any pinning failures when the memory manager allocation is expected to succeed", func(ctx context.Context) {
			// A guaranteed pod with a small memory request is admitted and pinned,
			// so the request counter increments but the error counter stays zero.
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedMemoryManagerPodWindows("memmngr-ok", "1", "64Mi"))

			matchResourceMetrics := gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
				"kubelet_memory_manager_pinning_requests_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(1),
				}),
				"kubelet_memory_manager_pinning_errors_total": gstruct.MatchAllElements(nodeID, gstruct.Elements{
					"": timelessSample(0),
				}),
			})

			gomega.Eventually(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
			gomega.Consistently(ctx, getWindowsKubeletMetrics, 1*time.Minute, 15*time.Second).Should(matchResourceMetrics)
		})
	})
})
