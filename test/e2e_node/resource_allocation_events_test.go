/*
Copyright 2023 The Kubernetes Authors.

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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Resource Allocation Events", framework.WithSerial(), feature.TopologyManager, feature.CPUManager, feature.MemoryManager, feature.DeviceManager, func() {
	f := framework.NewDefaultFramework("resource-allocation-events")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("with topology and cpumanager configured", func() {
		var oldCfg *kubeletconfig.KubeletConfiguration
		var testPod *v1.Pod
		var cpusNumPerNUMA, coresNumPerNUMA, numaNodes, threadsPerCore int

		ginkgo.BeforeEach(func(ctx context.Context) {
			var err error
			if oldCfg == nil {
				oldCfg, err = getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
			}

			numaNodes = detectNUMANodes()
			if numaNodes < minNumaNodes {
				e2eskipper.Skipf("this test is intended to be run on a system with at least %d NUMA cells", minNumaNodes)
			}
			coresNumPerNUMA = detectCoresPerSocket() // this is not completely correct but holds true on amd64 nowadays
			if coresNumPerNUMA < minCoreCount {
				e2eskipper.Skipf("this test is intended to be run on a system with at least %d cores per socket", minCoreCount)
			}
			threadsPerCore = detectThreadPerCore()
			cpusNumPerNUMA = coresNumPerNUMA * threadsPerCore

			// It is safe to assume that the CPUs are distributed equally across
			// NUMA nodes and therefore number of CPUs on all NUMA nodes are same
			// so we just check the CPUs on the first NUMA node

			framework.Logf("numaNodes on the system %d", numaNodes)
			framework.Logf("Cores per NUMA on the system %d", coresNumPerNUMA)
			framework.Logf("Threads per Core on the system %d", threadsPerCore)
			framework.Logf("CPUs per NUMA on the system %d", cpusNumPerNUMA)

			policy := topologymanager.PolicyRestricted // events will be emitted anyway
			scope := podScopeTopology                  // not relevant

			newCfg, _ := configureTopologyManagerInKubelet(oldCfg, policy, scope, nil, 0)
			updateKubeletConfig(ctx, f, newCfg, true)

			gomega.Eventually(ctx, func(ctx context.Context) bool {
				_, ready := getLocalTestNode(ctx, f)
				return ready
			}).WithPolling(framework.Poll).WithTimeout(5 * time.Minute).Should(gomega.BeTrueBecause("local node not ready"))
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			if testPod != nil {
				deletePodSyncByName(ctx, f, testPod.Name)
			}
			updateKubeletConfig(ctx, f, oldCfg, true)
		})

		ginkgo.It("should emit event when resource allocation is successful", func(ctx context.Context) {
			ginkgo.By("Creating the test pod which will be admitted")
			testPod = e2epod.NewPodClient(f).CreateSync(ctx, makeGuaranteedCPUExclusiveSleeperPod("pin-alloc-good", threadsPerCore))

			eventSelector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      testPod.Name,
				"involvedObject.namespace": testPod.Namespace,
				"reason":                   kubeletevents.AllocatedResources,
			}.AsSelector().String()
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, testPod.Namespace, eventSelector, "", framework.PodEventTimeout))
		})

		ginkgo.It("should emit event when resource allocation fails", func(ctx context.Context) {
			ginkgo.By("Creating the test pod which will be fail admission")
			// +1 to make sure the requested cpus can't be aligned on a single node
			testPod = e2epod.NewPodClient(f).Create(ctx, makeGuaranteedCPUExclusiveSleeperPod("pin-alloc-fail", cpusNumPerNUMA+1))

			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, testPod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase != v1.PodPending {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)
			testPod, err = e2epod.NewPodClient(f).Get(ctx, testPod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			if testPod.Status.Phase != v1.PodFailed {
				framework.Failf("pod %s not failed: %v", testPod.Name, testPod.Status)
			}

			eventSelector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      testPod.Name,
				"involvedObject.namespace": testPod.Namespace,
				"reason":                   kubeletevents.FailedAllocationCPU,
			}.AsSelector().String()
			framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, testPod.Namespace, eventSelector, "container", framework.PodEventTimeout))
		})

	})
})
