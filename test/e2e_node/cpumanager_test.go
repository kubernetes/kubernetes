/*
Copyright 2025 The Kubernetes Authors.

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

/*
 * this is a rewrite of the cpumanager e2e_node test.
 * we will move testcases from cpu_manager_test.go to cpumanager_test.go.
 * Full details in the tracking issue: https://github.com/kubernetes/kubernetes/issues/129884
 */

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

const (
	defaultCFSPeriod = "100000"
)

// this is ugly, but pratical
var (
	e2enodeRuntimeName     string
	e2enodeCgroupV2Enabled bool
	e2enodeCgroupDriver    string
)

/*
   - Serial:
   because the test updates kubelet configuration.

   - Ordered:
   Each spec (It block) need to run with a kubelet configuration in place. At minimum, we need
   the non-default cpumanager static policy, then we have the cpumanager options and so forth.
   The simplest solution is to set the kubelet explicitly each time, but this will cause a kubelet restart
   each time, which takes longer and makes the flow intrinsically more fragile (so more flakes are more likely).
   Using Ordered allows us to use BeforeAll/AfterAll, and most notably to reuse the kubelet config in a batch
   of specs (It blocks). Each it block will still set its kubelet config preconditions, but with a sensible
   test arrangement, many of these preconditions will devolve into noop.
   Arguably, this decision increases the coupling among specs, leaving room for subtle ordering bugs.
   There's no argue the ginkgo spec randomization would help, but the tradeoff here is between
   lane complexity/fragility (reconfiguring the kubelet is not bulletproof yet) and accepting this risk.
   If in the future we decide to pivot to make each spec fully independent, little changes will be needed.
   Finally, worth pointing out that the previous cpumanager e2e test incarnation implemented the same
   concept in a more convoluted way with function helpers, so arguably using Ordered and making it
   explicit is already an improvement.
*/

/*
 * Extending the cpumanager test suite
 * TL;DRs: THE MOST IMPORTANT:
 * Please keep the test hierarchy very flat.
 * Nesting more than 2 contexts total from SIGDescribe root is likely to be a smell.
 * The problem with deep nesting is the interaction between BeforeEach blocks and the increased scope of variables.
 * The Ideal layout would be
 *   SIGDescribe # top level, unique
 *     Context   # you can add more context, but please try hard to avoid nesting them
 *       It      # add it blocks freely. Feel free to start new *non-nested* contexts as you see fit
 *     Context
 *       It
 *       It
 * Exception: if you need to add the same labels to quite a few (say, 3+) It blocks, you can add a **empty** context
 * The most important thing is to avoid long chain of beforeeach/aftereach and > 2 level of context nesting.
 * So a **empty** context only to group labels is acceptable:
 *   SIGDescribe
 *     Context(label1, label2) # avoid beforeeach/aftereach and variables here
 *       Context
 *         It
 *         It
 *     Context
 *       It
 * Final rule of thumb: if the nesting of the context description starts to read awkward or funny or stop making sense
 * if read as english sentence, then the nesting is likely too deep.
 */
var _ = SIGDescribe("CPU Manager", ginkgo.Ordered, framework.WithSerial(), feature.CPUManager, func() {
	f := framework.NewDefaultFramework("cpumanager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// original kubeletconfig before the context start, to be restored
	var oldCfg *kubeletconfig.KubeletConfiguration
	var localNode *v1.Node
	var onlineCPUs cpuset.CPUSet
	var smtLevel int
	// tracks all the pods created by a It() block. Best would be a namespace per It block
	// TODO: move to a namespace per It block?
	var podMap map[string]*v1.Pod

	ginkgo.BeforeAll(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)

		onlineCPUs, err = getOnlineCPUs() // this should not change at all, at least during this suite lifetime
		framework.ExpectNoError(err)
		framework.Logf("Online CPUs: %s", onlineCPUs)

		smtLevel = smtLevelFromSysFS() // this should not change at all, at least during this suite lifetime
		framework.Logf("SMT level: %d", smtLevel)

		e2enodeCgroupV2Enabled = IsCgroup2UnifiedMode()
		framework.Logf("cgroup V2 enabled: %v", e2enodeCgroupV2Enabled)

		e2enodeCgroupDriver = oldCfg.CgroupDriver
		framework.Logf("cgroup driver: %s", e2enodeCgroupDriver)

		runtime, _, err := getCRIClient()
		framework.ExpectNoError(err, "Failed to get CRI client")

		version, err := runtime.Version(context.Background(), "")
		framework.ExpectNoError(err, "Failed to get runtime version")

		e2enodeRuntimeName = version.GetRuntimeName()
		framework.Logf("runtime: %s", e2enodeRuntimeName)
	})

	ginkgo.AfterAll(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		podMap = make(map[string]*v1.Pod)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		deletePodsAsync(ctx, f, podMap)
	})

	ginkgo.When("running non-guaranteed pods tests", ginkgo.Label("non-guaranteed", "reserved-cpus"), func() {
		ginkgo.It("should let the container access all the online CPUs without a reserved CPUs set", func(ctx context.Context) {
			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: cpuset.CPUSet{},
			}))

			pod := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "100m",
					cpuLimit:   "200m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			gomega.Expect(pod).To(HaveContainerCPUsEqualTo("non-gu-container", onlineCPUs))
		})

		ginkgo.It("should let the container access all the online CPUs when using a reserved CPUs set", func(ctx context.Context) {
			reservedCPUs := cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.

			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "100m",
					cpuLimit:   "200m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			gomega.Expect(pod).To(HaveContainerCPUsEqualTo("non-gu-container", onlineCPUs))
		})

		ginkgo.It("should let the container access all the online non-exclusively-allocated CPUs when using a reserved CPUs set", ginkgo.Label("guaranteed", "exclusive-cpus"), func(ctx context.Context) {
			cpuCount := 1
			reservedCPUs := cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.

			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			podGu := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container",
					cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
					cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
				},
			})
			ginkgo.By("creating the guaranteed test pod")
			podGu = e2epod.NewPodClient(f).CreateSync(ctx, podGu)
			podMap[string(podGu.UID)] = podGu

			podBu := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "200m",
					cpuLimit:   "300m",
				},
			})
			ginkgo.By("creating the burstable test pod")
			podBu = e2epod.NewPodClient(f).CreateSync(ctx, podBu)
			podMap[string(podBu.UID)] = podBu

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(podGu).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(podGu).To(HaveContainerCPUsASubsetOf("gu-container", onlineCPUs))
			gomega.Expect(podGu).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))

			exclusiveCPUs, err := getContainerAllowedCPUs(podGu, "gu-container")
			framework.ExpectNoError(err, "cannot get exclusive CPUs for pod %s/%s", podGu.Namespace, podGu.Name)
			expectedSharedCPUs := onlineCPUs.Difference(exclusiveCPUs)
			gomega.Expect(podBu).To(HaveContainerCPUsEqualTo("non-gu-container", expectedSharedCPUs))
		})
	})

	ginkgo.When("running guaranteed pod tests", ginkgo.Label("guaranteed", "exclusive-cpus"), func() {
		ginkgo.It("should allocate exclusively a CPU to a 1-container pod", func(ctx context.Context) {
			cpuCount := 1
			reservedCPUs := cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node. -- note this snippet is repeated multiple time over the tests

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container",
					cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
					cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))
		})

		// we don't use a separate group (gingo.When) with BeforeEach to factor out the tests because each
		// test need to check for the amount of CPUs it needs.

		ginkgo.It("should allocate exclusively a even number of CPUs to a 1-container pod", func(ctx context.Context) {
			cpuCount := 2
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container",
					cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
					cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))
			// TODO: this is probably too strict but it is the closest of the old test did
			gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings("gu-container"))
		})

		ginkgo.It("should allocate exclusively a odd number of CPUs to a 1-container pod", func(ctx context.Context) {
			cpuCount := 3
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container",
					cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
					cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))
			// TODO: this is probably too strict but it is the closest of the old test did
			toleration := 1
			gomega.Expect(pod).To(HaveContainerCPUsQuasiThreadSiblings("gu-container", toleration))
		})

		ginkgo.It("should allocate exclusively CPUs to a multi-container pod (1+2)", func(ctx context.Context) {
			cpuCount := 3 // total
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container-1",
					cpuRequest: "1000m",
					cpuLimit:   "1000m",
				},
				{
					ctnName:    "gu-container-2",
					cpuRequest: "2000m",
					cpuLimit:   "2000m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-1", 1))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-1", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-1", reservedCPUs))

			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-2", 2))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-2", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-2", reservedCPUs))
			// TODO: this is probably too strict but it is the closest of the old test did
			gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings("gu-container-2"))
		})

		ginkgo.It("should allocate exclusively CPUs to a multi-container pod (3+2)", func(ctx context.Context) {
			cpuCount := 5 // total
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container-1",
					cpuRequest: "3000m",
					cpuLimit:   "3000m",
				},
				{
					ctnName:    "gu-container-2",
					cpuRequest: "2000m",
					cpuLimit:   "2000m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-1", 3))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-1", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-1", reservedCPUs))
			toleration := 1
			gomega.Expect(pod).To(HaveContainerCPUsQuasiThreadSiblings("gu-container-1", toleration))

			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-2", 2))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-2", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-2", reservedCPUs))
			gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings("gu-container-2"))
		})

		ginkgo.It("should allocate exclusively CPUs to a multi-container pod (4+2)", func(ctx context.Context) {
			cpuCount := 6 // total
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container-1",
					cpuRequest: "4000m",
					cpuLimit:   "4000m",
				},
				{
					ctnName:    "gu-container-2",
					cpuRequest: "2000m",
					cpuLimit:   "2000m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-1", 4))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-1", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-1", reservedCPUs))
			gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings("gu-container-1"))

			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-2", 2))
			gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-2", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-2", reservedCPUs))
			gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings("gu-container-2"))
		})

		ginkgo.It("should allocate exclusively a CPU to multiple 1-container pods", func(ctx context.Context) {
			cpuCount := 4 // total
			reservedCPUs := cpuset.New(0)

			skipIfNotEnoughAllocatableCPUs(ctx, f, cpuCount+reservedCPUs.Size(), onlineCPUs)

			_ = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: reservedCPUs, // Not really needed for the tests but helps to make a more precise check
			}))

			pod1 := makeCPUManagerPod("gu-pod-1", []ctnAttribute{
				{
					ctnName:    "gu-container-1",
					cpuRequest: "2000m",
					cpuLimit:   "2000m",
				},
			})
			ginkgo.By("creating the test pod 1")
			pod1 = e2epod.NewPodClient(f).CreateSync(ctx, pod1)
			podMap[string(pod1.UID)] = pod1

			pod2 := makeCPUManagerPod("gu-pod-2", []ctnAttribute{
				{
					ctnName:    "gu-container-2",
					cpuRequest: "2000m",
					cpuLimit:   "2000m",
				},
			})
			ginkgo.By("creating the test pod 2")
			pod2 = e2epod.NewPodClient(f).CreateSync(ctx, pod2)
			podMap[string(pod2.UID)] = pod2

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod1).To(HaveContainerCPUsCount("gu-container-1", 2))
			gomega.Expect(pod1).To(HaveContainerCPUsASubsetOf("gu-container-1", onlineCPUs))
			gomega.Expect(pod1).ToNot(HaveContainerCPUsOverlapWith("gu-container-1", reservedCPUs))
			gomega.Expect(pod1).To(HaveContainerCPUsThreadSiblings("gu-container-1"))

			gomega.Expect(pod2).To(HaveContainerCPUsCount("gu-container-2", 2))
			gomega.Expect(pod2).To(HaveContainerCPUsASubsetOf("gu-container-2", onlineCPUs))
			gomega.Expect(pod2).ToNot(HaveContainerCPUsOverlapWith("gu-container-2", reservedCPUs))
			gomega.Expect(pod2).To(HaveContainerCPUsThreadSiblings("gu-container-2"))
		})
	})

	ginkgo.When("running with strict CPU reservation", ginkgo.Label("strict-cpu-reservation"), func() {
		var reservedCPUs cpuset.CPUSet

		ginkgo.BeforeEach(func(ctx context.Context) {
			reservedCPUs = cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.
		})

		ginkgo.It("should let the container access all the online CPUs without a reserved CPUs set", func(ctx context.Context) {
			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      cpuset.CPUSet{},
				enableCPUManagerOptions: true,
				options: map[string]string{
					cpumanager.StrictCPUReservationOption: "true",
				},
			}))

			pod := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "100m",
					cpuLimit:   "200m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// cpumanager will always reserve at least 1 cpu. In this case we don't set which, and if we treat the cpumanager
			// as black box (which we very much should) we can't predict which one. So we can only assert that *A* cpu is not
			// usable because is reserved.
			gomega.Expect(pod).To(HaveContainerCPUsCount("non-gu-container", onlineCPUs.Size()-1))
		})

		ginkgo.It("should let the container access all the online CPUs minus the reserved CPUs set when enabled", func(ctx context.Context) {
			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      reservedCPUs, // Not really needed for the tests but helps to make a more precise check
				enableCPUManagerOptions: true,
				options: map[string]string{
					cpumanager.StrictCPUReservationOption: "true",
				},
			}))

			pod := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "100m",
					cpuLimit:   "200m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			gomega.Expect(pod).To(HaveContainerCPUsEqualTo("non-gu-container", onlineCPUs.Difference(reservedCPUs)))
		})

		ginkgo.It("should let the container access all the online non-exclusively-allocated CPUs minus the reserved CPUs set when enabled", func(ctx context.Context) {
			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      reservedCPUs, // Not really needed for the tests but helps to make a more precise check
				enableCPUManagerOptions: true,
				options: map[string]string{
					cpumanager.StrictCPUReservationOption: "true",
				},
			}))

			cpuCount := 1

			podGu := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container",
					cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
					cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
				},
			})
			ginkgo.By("creating the guaranteed test pod")
			podGu = e2epod.NewPodClient(f).CreateSync(ctx, podGu)
			podMap[string(podGu.UID)] = podGu

			podBu := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "200m",
					cpuLimit:   "300m",
				},
			})
			ginkgo.By("creating the burstable test pod")
			podBu = e2epod.NewPodClient(f).CreateSync(ctx, podBu)
			podMap[string(podBu.UID)] = podBu

			ginkgo.By("checking if the expected cpuset was assigned")

			usableCPUs := onlineCPUs.Difference(reservedCPUs)

			// any full CPU is fine - we cannot nor we should predict which one, though
			gomega.Expect(podGu).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(podGu).To(HaveContainerCPUsASubsetOf("gu-container", usableCPUs))
			gomega.Expect(podGu).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))

			exclusiveCPUs, err := getContainerAllowedCPUs(podGu, "gu-container")
			framework.ExpectNoError(err, "cannot get exclusive CPUs for pod %s/%s", podGu.Namespace, podGu.Name)
			expectedSharedCPUs := usableCPUs.Difference(exclusiveCPUs)
			gomega.Expect(podBu).To(HaveContainerCPUsEqualTo("non-gu-container", expectedSharedCPUs))
		})
	})

	ginkgo.When("running with SMT Alignment", ginkgo.Label("smt-alignment"), func() {
		var cpuDetails nodeCPUDetails

		ginkgo.BeforeEach(func(ctx context.Context) {
			// strict SMT alignment is trivially verified and granted on non-SMT systems
			if smtLevel < minSMTLevel {
				e2eskipper.Skipf("Skipping CPU Manager %q tests since SMT disabled", cpumanager.FullPCPUsOnlyOption)
			}

			reservedCPUs := cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.

			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				reservedSystemCPUs:      reservedCPUs,
				enableCPUManagerOptions: true,
				options: map[string]string{
					cpumanager.FullPCPUsOnlyOption: "true",
				},
			}))
			cpuDetails = cpuDetailsFromNode(localNode)
		})

		ginkgo.It("should reject workload asking non-SMT-multiple of cpus", func(ctx context.Context) {
			// our tests want to allocate a full core, so we need at last 2*2=4 virtual cpus
			if cpuDetails.Allocatable < int64(smtLevel) {
				e2eskipper.Skipf("Skipping CPU Manager %q tests since the CPU capacity < %d", cpumanager.FullPCPUsOnlyOption, smtLevel)
			}

			ginkgo.By("creating the testing pod")
			// negative test: try to run a container whose requests aren't a multiple of SMT level, expect a rejection
			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container-neg",
					cpuRequest: "1000m",
					cpuLimit:   "1000m",
				},
			})
			ginkgo.By("creating the test pod")
			// CreateSync would wait for pod to become Ready - which will never happen if production code works as intended!
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("ensuring the testing pod is in failed state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
				if pod.Status.Phase != v1.PodPending {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the testing pod is failed for the expected reason")
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod).To(BeAPodInPhase(v1.PodFailed))
			gomega.Expect(pod).To(HaveStatusReasonMatchingRegex(`SMT.*Alignment.*Error`))
		})

		ginkgo.It("should admit workload asking SMT-multiple of cpus", func(ctx context.Context) {
			// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
			// 1. are core siblings
			// 2. take a full core
			// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
			//          this means on more-than-2-way SMT systems this test will prove nothing

			if cpuDetails.Allocatable < int64(smtLevel) {
				e2eskipper.Skipf("required %d allocatable CPUs found %d", smtLevel, cpuDetails.Allocatable)
			}

			cpuRequest := fmt.Sprintf("%d000m", smtLevel)
			ginkgo.By(fmt.Sprintf("creating the testing pod cpuRequest=%v", cpuRequest))
			pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
				{
					ctnName:    "gu-container-pos",
					cpuRequest: cpuRequest,
					cpuLimit:   cpuRequest,
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("validating each container in the testing pod")
			for _, cnt := range pod.Spec.Containers {
				ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))

				gomega.Expect(pod).To(HaveContainerCPUsAlignedTo(cnt.Name, smtLevel))
				gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings(cnt.Name))
			}
		})
	})

	ginkgo.When("checking the compatibility between options", func() {
		// please avoid nesting `BeforeEach` as much as possible. Ideally avoid completely.
		ginkgo.Context("SMT Alignment and strict CPU reservation", ginkgo.Label("smt-alignment", "strict-cpu-reservation"), func() {
			var reservedCPUs cpuset.CPUSet

			ginkgo.BeforeEach(func(ctx context.Context) {
				// strict SMT alignment is trivially verified and granted on non-SMT systems
				if smtLevel < minSMTLevel {
					e2eskipper.Skipf("Skipping CPU Manager %q tests since SMT disabled", cpumanager.FullPCPUsOnlyOption)
				}
				reservedCPUs = cpuset.New(0)
			})

			ginkgo.It("should reject workload asking non-SMT-multiple of cpus", func(ctx context.Context) {
				// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
				// check what we do have in the node.

				localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      reservedCPUs,
					enableCPUManagerOptions: true,
					options: map[string]string{
						cpumanager.FullPCPUsOnlyOption:        "true",
						cpumanager.StrictCPUReservationOption: "true",
					},
				}))
				cpuDetails := cpuDetailsFromNode(localNode)

				// our tests want to allocate a full core, so we need at last 2*2=4 virtual cpus
				if cpuDetails.Allocatable < int64(smtLevel) {
					e2eskipper.Skipf("Skipping CPU Manager %q tests since the CPU capacity < %d", cpumanager.FullPCPUsOnlyOption, smtLevel)
				}

				// negative test: try to run a container whose requests aren't a multiple of SMT level, expect a rejection
				pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
					{
						ctnName:    "gu-container-neg",
						cpuRequest: "1000m",
						cpuLimit:   "1000m",
					},
				})
				ginkgo.By("creating the testing pod")
				// CreateSync would wait for pod to become Ready - which will never happen if production code works as intended!
				pod = e2epod.NewPodClient(f).Create(ctx, pod)
				podMap[string(pod.UID)] = pod

				ginkgo.By("ensuring the testing pod is in failed state")
				err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
					if pod.Status.Phase != v1.PodPending {
						return true, nil
					}
					return false, nil
				})
				framework.ExpectNoError(err)

				ginkgo.By("ensuring the testing pod is failed for the expected reason")
				pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
				framework.ExpectNoError(err)
				gomega.Expect(pod).To(BeAPodInPhase(v1.PodFailed))
				gomega.Expect(pod).To(HaveStatusReasonMatchingRegex(`SMT.*Alignment.*Error`))
			})

			ginkgo.It("should admit workload asking SMT-multiple of cpus", func(ctx context.Context) {
				// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
				// check what we do have in the node.

				localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      reservedCPUs,
					enableCPUManagerOptions: true,
					options: map[string]string{
						cpumanager.FullPCPUsOnlyOption:        "true",
						cpumanager.StrictCPUReservationOption: "true",
					},
				}))
				cpuDetails := cpuDetailsFromNode(localNode)

				// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
				// 1. are core siblings
				// 2. take a full core
				// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
				//          this means on more-than-2-way SMT systems this test will prove nothing

				if cpuDetails.Allocatable < int64(smtLevel) {
					e2eskipper.Skipf("required %d allocatable CPUs found %d", smtLevel, cpuDetails.Allocatable)
				}

				cpuCount := smtLevel
				cpuRequest := fmt.Sprintf("%d000m", cpuCount)
				ginkgo.By(fmt.Sprintf("creating the testing pod cpuRequest=%v", cpuRequest))
				pod := makeCPUManagerPod("gu-pod", []ctnAttribute{
					{
						ctnName:    "gu-container-x",
						cpuRequest: cpuRequest,
						cpuLimit:   cpuRequest,
					},
				})
				ginkgo.By("creating the testing pod")
				pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
				podMap[string(pod.UID)] = pod

				usableCPUs := onlineCPUs.Difference(reservedCPUs)

				gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container-x", cpuCount))
				gomega.Expect(pod).To(HaveContainerCPUsASubsetOf("gu-container-x", usableCPUs))
				gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container-x", reservedCPUs))

				ginkgo.By("validating each container in the testing pod")
				for _, cnt := range pod.Spec.Containers {
					ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))

					gomega.Expect(pod).To(HaveContainerCPUsAlignedTo(cnt.Name, smtLevel))
					gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings(cnt.Name))
				}
			})
		})

		// please avoid nesting `BeforeEach` as much as possible. Ideally avoid completely.
		ginkgo.Context("SMT Alignment and distribution across NUMA", ginkgo.Label("smt-alignment", "distribute-cpus-across-numa"), func() {
			var reservedCPUs cpuset.CPUSet

			ginkgo.BeforeEach(func(ctx context.Context) {
				// strict SMT alignment is trivially verified and granted on non-SMT systems
				if smtLevel < minSMTLevel {
					e2eskipper.Skipf("Skipping CPU Manager %q tests since SMT disabled", cpumanager.FullPCPUsOnlyOption)
				}
				reservedCPUs = cpuset.New(0)
			})

			ginkgo.It("should assign packed CPUs with distribute-cpus-across-numa disabled and pcpu-only policy options enabled", func(ctx context.Context) {
				localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      reservedCPUs,
					enableCPUManagerOptions: true,
					options: map[string]string{
						cpumanager.FullPCPUsOnlyOption:            "true",
						cpumanager.DistributeCPUsAcrossNUMAOption: "false",
					},
				}))
				cpuDetails := cpuDetailsFromNode(localNode)

				// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
				// 1. are core siblings
				// 2. take a full core
				// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
				//          this means on more-than-2-way SMT systems this test will prove nothing

				minCPUCount := int64(smtLevel * minCPUCapacity)
				if cpuDetails.Allocatable < minCPUCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < %d", minCPUCount)
				}

				ctnAttrs := []ctnAttribute{
					{
						ctnName:    "test-gu-container-distribute-cpus-across-numa-disabled",
						cpuRequest: "2000m",
						cpuLimit:   "2000m",
					},
				}
				pod := makeCPUManagerPod("test-pod-distribute-cpus-across-numa-disabled", ctnAttrs)
				ginkgo.By("creating the test pod")
				pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
				podMap[string(pod.UID)] = pod

				ginkgo.By("validating each container in the testing pod")
				for _, cnt := range pod.Spec.Containers {
					ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))

					gomega.Expect(pod).To(HaveContainerCPUsAlignedTo(cnt.Name, smtLevel))
					gomega.Expect(pod).To(HaveContainerCPUsThreadSiblings(cnt.Name))
				}
			})

			ginkgo.It("should assign CPUs distributed across NUMA with distribute-cpus-across-numa and pcpu-only policy options enabled", func(ctx context.Context) {
				reservedCPUs := cpuset.New(0)

				// this test is intended to be run on a multi-node NUMA system and
				// a system with at least 4 cores per socket, hostcheck skips test
				// if above requirements are not satisfied
				numaNodeNum, _, _, cpusNumPerNUMA := hostCheck()

				localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:              string(cpumanager.PolicyStatic),
					reservedSystemCPUs:      reservedCPUs,
					enableCPUManagerOptions: true,
					options: map[string]string{
						cpumanager.FullPCPUsOnlyOption:            "true",
						cpumanager.DistributeCPUsAcrossNUMAOption: "true",
					},
				}))
				cpuDetails := cpuDetailsFromNode(localNode)

				// our tests want to allocate a full core, so we need at least 2*2=4 virtual cpus
				minCPUCount := int64(smtLevel * minCPUCapacity)
				if cpuDetails.Allocatable < minCPUCount {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < %d", minCPUCount)
				}

				// 'distribute-cpus-across-numa' policy option ensures that CPU allocations are evenly distributed
				//  across NUMA nodes in cases where more than one NUMA node is required to satisfy the allocation.
				// So, we want to ensure that the CPU Request exceeds the number of CPUs that can fit within a single
				// NUMA node. We have to pick cpuRequest such that:
				// 1. CPURequest > cpusNumPerNUMA
				// 2. Not occupy all the CPUs on the node ande leave room for reserved CPU
				// 3. CPURequest is a multiple if number of NUMA nodes to allow equal CPU distribution across NUMA nodes
				//
				// In summary: cpusNumPerNUMA < CPURequest < ((cpusNumPerNuma * numaNodeNum) - reservedCPUscount)
				// Considering all these constraints we select: CPURequest= (cpusNumPerNUMA-smtLevel)*numaNodeNum

				cpuReq := (cpusNumPerNUMA - smtLevel) * numaNodeNum
				ctnAttrs := []ctnAttribute{
					{
						ctnName:    "test-gu-container-distribute-cpus-across-numa",
						cpuRequest: fmt.Sprintf("%d", cpuReq),
						cpuLimit:   fmt.Sprintf("%d", cpuReq),
					},
				}
				pod := makeCPUManagerPod("test-pod-distribute-cpus-across-numa", ctnAttrs)
				ginkgo.By("creating the test pod")
				pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
				podMap[string(pod.UID)] = pod

				for _, cnt := range pod.Spec.Containers {
					ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))

					gomega.Expect(pod).To(HaveContainerCPUsAlignedTo(cnt.Name, smtLevel))
					cpus, err := getContainerAllowedCPUs(pod, cnt.Name)
					framework.ExpectNoError(err, "cannot get cpus allocated to pod %s/%s cnt %s", pod.Namespace, pod.Name, cnt.Name)

					siblingsCPUs := makeThreadSiblingCPUSet(cpus)
					gomega.Expect(pod).To(HaveContainerCPUsEqualTo(cnt.Name, siblingsCPUs))

					// We expect a perfectly even spilit i.e. equal distribution across NUMA Node as the CPU Request is 4*smtLevel*numaNodeNum.
					expectedSpread := cpus.Size() / numaNodeNum
					gomega.Expect(cpus).To(BeDistributedCPUs(expectedSpread))
				}
			})
		})
	})

	ginkgo.When("checking the CFS quota management", ginkgo.Label("cfs-quota"), func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			if !e2enodeCgroupV2Enabled {
				e2eskipper.Skipf("Skipping since CgroupV2 not used")
			}

			// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
			//          this means on more-than-2-way SMT systems this test will prove nothing
			reservedCPUs := cpuset.New(0)
			localNode = updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:                       string(cpumanager.PolicyStatic),
				reservedSystemCPUs:               reservedCPUs,
				disableCPUQuotaWithExclusiveCPUs: true,
			}))
		})

		ginkgo.It("should enforce for best-effort pod", func(ctx context.Context) {
			ctnName := "be-container"
			pod := makeCPUManagerBEPod("be-pod", []ctnAttribute{
				{
					ctnName:    ctnName,
					ctnCommand: "sleep 1d",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("max"))
			gomega.Expect(pod).To(HaveContainerQuota(ctnName, "max"))
		})

		ginkgo.It("should disable for guaranteed pod with exclusive CPUs assigned", func(ctx context.Context) {
			ctnName := "gu-container-cfsquota-disabled"
			pod := makeCPUManagerPod("gu-pod-cfsquota-off", []ctnAttribute{
				{
					ctnName:    ctnName,
					cpuRequest: "1",
					cpuLimit:   "1",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("max"))
			gomega.Expect(pod).To(HaveContainerQuota(ctnName, "max"))
		})

		ginkgo.It("should enforce for guaranteed pod", func(ctx context.Context) {
			ctnName := "gu-container-cfsquota-enabled"
			pod := makeCPUManagerPod("gu-pod-cfs-quota-on", []ctnAttribute{
				{
					ctnName:    ctnName,
					cpuRequest: "500m",
					cpuLimit:   "500m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("50000"))
			gomega.Expect(pod).To(HaveContainerQuota(ctnName, "50000"))
		})

		ginkgo.It("should enforce for burstable pod", func(ctx context.Context) {
			ctnName := "bu-container-cfsquota-enabled"
			pod := makeCPUManagerPod("bu-pod-cfs-quota-on", []ctnAttribute{
				{
					ctnName:    ctnName,
					cpuRequest: "100m",
					cpuLimit:   "500m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("50000"))
			gomega.Expect(pod).To(HaveContainerQuota(ctnName, "50000"))
		})

		ginkgo.It("should not enforce with multiple containers without exclusive CPUs", func(ctx context.Context) {
			cpuDetails := cpuDetailsFromNode(localNode)
			if cpuDetails.Allocatable < int64(2) {
				e2eskipper.Skipf("Skipping because needs %d allocatable CPUs, detected %d", 2, cpuDetails.Allocatable)
			}

			pod := makeCPUManagerPod("gu-pod-multicontainer", []ctnAttribute{
				{
					ctnName:    "gu-container-non-int-values-1",
					cpuRequest: "100m",
					cpuLimit:   "500m",
				},
				{
					ctnName:    "gu-container-non-int-values-2",
					cpuRequest: "300m",
					cpuLimit:   "1200m",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("170000"))
			gomega.Expect(pod).To(HaveContainerQuota("gu-container-non-int-values-1", "50000"))
			gomega.Expect(pod).To(HaveContainerQuota("gu-container-non-int-values-2", "120000"))
		})

		ginkgo.It("should not enforce with multiple containers only in the container with exclusive CPUs", func(ctx context.Context) {
			cpuDetails := cpuDetailsFromNode(localNode)
			if cpuDetails.Allocatable < int64(2) {
				e2eskipper.Skipf("Skipping because needs %d allocatable CPUs, detected %d", 2, cpuDetails.Allocatable)
			}

			pod := makeCPUManagerPod("gu-pod-multicontainer-mixed", []ctnAttribute{
				{
					ctnName:    "gu-container-non-int-values",
					cpuRequest: "500m",
					cpuLimit:   "500m",
				},
				{
					ctnName:    "gu-container-int-values",
					cpuRequest: "1",
					cpuLimit:   "1",
				},
			})
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			gomega.Expect(pod).To(HaveSandboxQuota("max"))
			gomega.Expect(pod).To(HaveContainerQuota("gu-container-non-int-values", "50000"))
			gomega.Expect(pod).To(HaveContainerQuota("gu-container-int-values", "max"))
		})
	})
})

// Matching helpers

func HaveStatusReasonMatchingRegex(expr string) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		re, err := regexp.Compile(expr)
		if err != nil {
			return false, err
		}
		return re.MatchString(actual.Status.Reason), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} reason {{.Actual.Status.Reason}} does not match regexp {{.Data}}", expr)
}

type msgData struct {
	Name           string
	CurrentCPUs    string
	ExpectedCPUs   string
	MismatchedCPUs string
	Count          int
	Aligned        int
	CurrentQuota   string
	ExpectedQuota  string
}

func HaveContainerCPUsCount(ctnName string, val int) types.GomegaMatcher {
	md := &msgData{
		Name:  ctnName,
		Count: val,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		return cpus.Size() == val, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not matching expected count <{{.Data.Count}}> for container {{.Data.Name}}", md)
}

func HaveContainerCPUsAlignedTo(ctnName string, val int) types.GomegaMatcher {
	md := &msgData{
		Name:    ctnName,
		Aligned: val,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		return cpus.Size()%val == 0, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not aligned to value <{{.Data.Aligned}}> for container {{.Data.Name}}", md)
}

func HaveContainerCPUsOverlapWith(ctnName string, ref cpuset.CPUSet) types.GomegaMatcher {
	md := &msgData{
		Name:         ctnName,
		ExpectedCPUs: ref.String(),
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		sharedCPUs := cpus.Intersection(ref)
		return sharedCPUs.Size() > 0, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> overlapping with expected CPUs <{{.Data.ExpectedCPUs}}> for container {{.Data.Name}}", md)
}

func HaveContainerCPUsASubsetOf(ctnName string, ref cpuset.CPUSet) types.GomegaMatcher {
	md := &msgData{
		Name:         ctnName,
		ExpectedCPUs: ref.String(),
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		return cpus.IsSubsetOf(ref), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not a subset of expected CPUs <{{.Data.ExpectedCPUs}}> for container {{.Data.Name}}", md)
}

func HaveContainerCPUsEqualTo(ctnName string, expectedCPUs cpuset.CPUSet) types.GomegaMatcher {
	md := &msgData{
		Name:         ctnName,
		ExpectedCPUs: expectedCPUs.String(),
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		return cpus.Equals(expectedCPUs), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not matching the expected value <{{.Data.ExpectedCPUs}}> for container {{.Data.Name}}", md)
}

func HaveSandboxQuota(expectedQuota string) types.GomegaMatcher {
	md := &msgData{
		ExpectedQuota: expectedQuota,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		md.Name = klog.KObj(actual).String()
		quota, err := getSandboxCFSQuota(actual)
		md.CurrentQuota = quota
		if err != nil {
			framework.Logf("getSandboxCFSQuota() failed: %v", err)
			return false, err
		}
		re, err := regexp.Compile(fmt.Sprintf("^%s %s$", expectedQuota, defaultCFSPeriod))
		if err != nil {
			return false, err
		}
		return re.MatchString(quota), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has quota <{{.Data.CurrentQuota}}> not matching expected value <{{.Data.ExpectedQuota}}>", md)
}

func HaveContainerQuota(ctnName, expectedQuota string) types.GomegaMatcher {
	md := &msgData{
		Name:          ctnName,
		ExpectedQuota: expectedQuota,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		quota, err := getContainerCFSQuota(actual, ctnName)
		md.CurrentQuota = quota
		if err != nil {
			framework.Logf("getContainerCFSQuota(%s) failed: %v", ctnName, err)
			return false, err
		}
		re, err := regexp.Compile(fmt.Sprintf("^%s %s$", expectedQuota, defaultCFSPeriod))
		if err != nil {
			return false, err
		}
		return re.MatchString(quota), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has quota <{{.Data.CurrentQuota}}> not matching expected value <{{.Data.ExpectedQuota}}> for container {{.Data.Name}}", md)
}

func HaveContainerCPUsThreadSiblings(ctnName string) types.GomegaMatcher {
	md := &msgData{
		Name: ctnName,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		expectedCPUs := makeThreadSiblingCPUSet(cpus)
		md.ExpectedCPUs = expectedCPUs.String()
		return cpus.Equals(expectedCPUs), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not all thread sibling pairs (would be <{{.Data.ExpectedCPUs}}>) for container {{.Data.Name}}", md)
}

func HaveContainerCPUsQuasiThreadSiblings(ctnName string, toleration int) types.GomegaMatcher {
	md := &msgData{
		Name:  ctnName,
		Count: toleration,
	}
	return gcustom.MakeMatcher(func(actual *v1.Pod) (bool, error) {
		cpus, err := getContainerAllowedCPUs(actual, ctnName)
		md.CurrentCPUs = cpus.String()
		if err != nil {
			framework.Logf("getContainerAllowedCPUs(%s) failed: %v", ctnName, err)
			return false, err
		}
		// this is by construction >= cpus (extreme case: cpus is made by all non-thread-siblings)
		expectedCPUs := makeThreadSiblingCPUSet(cpus)
		md.ExpectedCPUs = expectedCPUs.String()
		mismatchedCPUs := expectedCPUs.Difference(cpus)
		md.MismatchedCPUs = mismatchedCPUs.String()
		return mismatchedCPUs.Size() <= toleration, nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} has allowed CPUs <{{.Data.CurrentCPUs}}> not all thread sibling pairs (would be <{{.Data.ExpectedCPUs}}> mismatched <{{.Data.MismatchedCPUs}}> toleration <{{.Data.Count}}>) for container {{.Data.Name}}", md)
}

// Other helpers

func skipIfNotEnoughAllocatableCPUs(ctx context.Context, f *framework.Framework, cpuReq int, onlineCPUs cpuset.CPUSet) {
	localNode := getLocalNode(ctx, f)
	cpuAllocQty := localNode.Status.Allocatable[v1.ResourceCPU]
	cpuAlloc := cpuAllocQty.Value()
	if cpuAlloc < int64(cpuReq) {
		ginkgo.Skip(fmt.Sprintf("Skipping CPU Manager tests since the CPU allocatable %d < CPU request %d (online %d)", cpuAlloc, cpuReq, onlineCPUs.Size()), 1)
	}
}

func getContainerAllowedCPUs(pod *v1.Pod, ctnName string) (cpuset.CPUSet, error) {
	cgPath, err := makeCgroupPathForContainer(pod, ctnName)
	if err != nil {
		return cpuset.CPUSet{}, err
	}
	framework.Logf("pod %s/%s cnt %s qos=%s path %q", pod.Namespace, pod.Name, ctnName, pod.Status.QOSClass, cgPath)
	data, err := os.ReadFile(filepath.Join(cgPath, "cpuset.cpus.effective"))
	if err != nil {
		return cpuset.CPUSet{}, err
	}
	cpus := strings.TrimSpace(string(data))
	framework.Logf("pod %s/%s cnt %s cpuset %q", pod.Namespace, pod.Name, ctnName, cpus)
	return cpuset.Parse(cpus)
}

func getSandboxCFSQuota(pod *v1.Pod) (string, error) {
	cgPath := filepath.Join(makeCgroupPathForPod(pod), "cpu.max")
	data, err := os.ReadFile(cgPath)
	if err != nil {
		return "", err
	}
	quota := strings.TrimSpace(string(data))
	framework.Logf("pod %s/%s qos=%s path %q quota %q", pod.Namespace, pod.Name, pod.Status.QOSClass, cgPath, quota)
	return quota, nil
}

func getContainerCFSQuota(pod *v1.Pod, ctnName string) (string, error) {
	cgPath, err := makeCgroupPathForContainer(pod, ctnName)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(filepath.Join(cgPath, "cpu.max"))
	if err != nil {
		return "", err
	}
	quota := strings.TrimSpace(string(data))
	framework.Logf("pod %s/%s qos=%s cnt %s path %q quota %q", pod.Namespace, pod.Name, pod.Status.QOSClass, ctnName, cgPath, quota)
	return quota, nil
}

const (
	kubeCgroupRoot = "/sys/fs/cgroup"
)

// example path (systemd):
// /sys/fs/cgroup/ kubepods.slice/kubepods-burstable.slice/kubepods-burstable-pod0b7632a2_a56e_4278_987a_22de18008dbe.slice/ crio-conmon-0bc5eac79e3ae7a0c2651f14722aa10fa333eb2325c2ca97da33aa284cda81b0.scope

func makeCgroupPathForPod(pod *v1.Pod) string {
	components := []string{defaultNodeAllocatableCgroup}
	if pod.Status.QOSClass != v1.PodQOSGuaranteed {
		components = append(components, strings.ToLower(string(pod.Status.QOSClass)))
	}
	components = append(components, "pod"+string(pod.UID))

	cgroupName := cm.NewCgroupName(cm.RootCgroupName, components...)
	cgroupFsName := ""
	// it's quite ugly to use a global, but it saves us to pass a parameter all across the stack many times
	if e2enodeCgroupDriver == "systemd" {
		cgroupFsName = cgroupName.ToSystemd()
	} else {
		cgroupFsName = cgroupName.ToCgroupfs()
	}

	return filepath.Join(kubeCgroupRoot, cgroupFsName)
}

func makeCgroupPathForContainer(pod *v1.Pod, ctnName string) (string, error) {
	cntSt := findContainerStatusByName(pod, ctnName)
	if cntSt == nil {
		return "", fmt.Errorf("cannot find status for container %q", ctnName)
	}
	cntID, err := parseContainerID(cntSt.ContainerID)
	if err != nil {
		return "", err
	}
	cntPath := ""
	if e2enodeCgroupDriver == "systemd" {
		cntPath = containerCgroupPathPrefixFromDriver(e2enodeRuntimeName) + "-" + cntID + ".scope"
	} else {
		cntPath = cntID
	}

	return filepath.Join(makeCgroupPathForPod(pod), cntPath), nil
}

func containerCgroupPathPrefixFromDriver(runtimeName string) string {
	if runtimeName == "cri-o" {
		return "cri-o"
	}
	return "cri-containerd"
}

func parseContainerID(fullID string) (string, error) {
	_, cntID, found := strings.Cut(fullID, "://")
	if !found {
		return "", fmt.Errorf("unsupported containerID: %q", fullID)
	}
	// TODO: should we validate the kind?
	return cntID, nil
}

func findContainerStatusByName(pod *v1.Pod, ctnName string) *v1.ContainerStatus {
	for idx := range pod.Status.ContainerStatuses {
		cntSt := &pod.Status.ContainerStatuses[idx] // shortcat
		if cntSt.Name == ctnName {
			return cntSt
		}
	}
	return nil
}

func makeThreadSiblingCPUSet(cpus cpuset.CPUSet) cpuset.CPUSet {
	siblingsCPUs := cpuset.New()
	for _, cpuID := range cpus.UnsortedList() {
		siblingsCPUs = siblingsCPUs.Union(cpuSiblingListFromSysFS(int64(cpuID)))
	}
	return siblingsCPUs
}

func updateKubeletConfigIfNeeded(ctx context.Context, f *framework.Framework, desiredCfg *kubeletconfig.KubeletConfiguration) *v1.Node {
	curCfg, err := getCurrentKubeletConfig(ctx)
	framework.ExpectNoError(err)

	if equalKubeletConfiguration(curCfg, desiredCfg) {
		framework.Logf("Kubelet configuration already compliant, nothing to do")
		return getLocalNode(ctx, f)
	}

	framework.Logf("Updating Kubelet configuration")
	updateKubeletConfig(ctx, f, desiredCfg, true)
	framework.Logf("Updated Kubelet configuration")

	return getLocalNode(ctx, f)
}

func equalKubeletConfiguration(cfgA, cfgB *kubeletconfig.KubeletConfiguration) bool {
	cfgA = cfgA.DeepCopy()
	cfgB = cfgB.DeepCopy()
	// we care only about the payload, force metadata to be uniform
	cfgA.TypeMeta = metav1.TypeMeta{}
	cfgB.TypeMeta = metav1.TypeMeta{}
	return reflect.DeepEqual(cfgA, cfgB)
}

type nodeCPUDetails struct {
	Capacity    int64
	Allocatable int64
	Reserved    int64
}

func cpuDetailsFromNode(node *v1.Node) nodeCPUDetails {
	localNodeCap := node.Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := node.Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.DeepCopy()
	cpuRes.Sub(cpuAlloc)
	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)
	return nodeCPUDetails{
		Capacity:    cpuCap.Value(),
		Allocatable: cpuCap.Value() - cpuRes.Value(),
		Reserved:    cpuRes.Value(),
	}
}

// smtLevelFromSysFS returns the number of symmetrical multi-thread (SMT) execution units the processor provides.
// The most common value on x86_64 is 2 (2 virtual threads/cores per physical core), that would be smtLevel == 2.
// The following are all synonyms: threadsPerCore, smtLevel
// Note: can't find a good enough yet not overly long name, "threadSiblingCount", "smtLevel", "threadsPerCore" are all questionable.
func smtLevelFromSysFS() int {
	cpuID := int64(0) // this is just the most likely cpu to be present in a random system. No special meaning besides this.
	cpus := cpuSiblingListFromSysFS(cpuID)
	return cpus.Size()
}

func cpuSiblingListFromSysFS(cpuID int64) cpuset.CPUSet {
	data, err := os.ReadFile(fmt.Sprintf("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpuID))
	framework.ExpectNoError(err)
	// how many thread sibling you have = SMT level
	// example: 2-way SMT means 2 threads sibling for each thread
	cpus, err := cpuset.Parse(strings.TrimSpace(string(data)))
	framework.ExpectNoError(err)
	return cpus
}

func makeCPUManagerBEPod(podName string, ctnAttributes []ctnAttribute) *v1.Pod {
	var containers []v1.Container
	for _, ctnAttr := range ctnAttributes {
		ctn := v1.Container{
			Name:    ctnAttr.ctnName,
			Image:   busyboxImage,
			Command: []string{"sh", "-c", ctnAttr.ctnCommand},
			VolumeMounts: []v1.VolumeMount{
				{
					Name:      "sysfscgroup",
					MountPath: "/sysfscgroup",
				},
				{
					Name:      "podinfo",
					MountPath: "/podinfo",
				},
			},
		}
		containers = append(containers, ctn)
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers:    containers,
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
					},
				},
				{
					Name: "podinfo",
					VolumeSource: v1.VolumeSource{
						DownwardAPI: &v1.DownwardAPIVolumeSource{
							Items: []v1.DownwardAPIVolumeFile{
								{
									Path: "uid",
									FieldRef: &v1.ObjectFieldSelector{
										APIVersion: "v1",
										FieldPath:  "metadata.uid",
									},
								},
							},
						},
					},
				},
			},
		},
	}
}
