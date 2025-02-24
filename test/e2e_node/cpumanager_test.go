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
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gcustom"
	"github.com/onsi/gomega/types"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
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

// **Important!** Please keep the test hierarchy very flat.
// Nesting more than 2 contexts total from SIGDescribe root is likely to be a smell.
// we need a master container which is Ordered so we can use BeforeAll and AfterAll.
// we use BeforeAll/AfterAll to maximize the chances we clean up properly.
var _ = SIGDescribe("CPU Manager", ginkgo.Ordered, framework.WithSerial(), feature.CPUManager, func() {
	f := framework.NewDefaultFramework("cpumanager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// original kubeletconfig before the context start, to be restored
	var oldCfg *kubeletconfig.KubeletConfiguration
	var localNode *corev1.Node
	var onlineCPUs cpuset.CPUSet
	var smtLevel int
	// tracks all the pods created by a It() block. Best would be a namespace per It block
	// TODO: move to a namespace per It block?
	var podMap map[string]*corev1.Pod

	ginkgo.BeforeAll(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)

		onlineCPUs, err = getOnlineCPUs() // this should not change at all, at least during this suite lifetime
		framework.ExpectNoError(err)
		framework.Logf("Online CPUs: %s", onlineCPUs)

		smtLevel = smtLevelFromSysFS() // this should not change at all, at least during this suite lifetime
		framework.Logf("SMT level %d", smtLevel)
	})

	ginkgo.AfterAll(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		podMap = make(map[string]*corev1.Pod)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		deletePodsAsync(ctx, f, podMap)
	})

	ginkgo.When("running non-guaranteed pods tests", func() {
		ginkgo.It("should let the container access all the online CPUs with reserved CPUs set", func(ctx context.Context) {
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
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			cnt := &pod.Spec.Containers[0] // shortcut
			cpus := getContainerAllowedCPUs(ctx, f, pod, cnt)
			gomega.Expect(cpus).To(EqualCPUSet(onlineCPUs))
		})

		ginkgo.It("should let the container access all the online CPUs with reserved CPUs set", func(ctx context.Context) {
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
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			cnt := &pod.Spec.Containers[0] // shortcut
			ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))
			cpus := getContainerAllowedCPUs(ctx, f, pod, cnt)
			gomega.Expect(cpus).To(EqualCPUSet(onlineCPUs))
		})

		ginkgo.It("should let the container access all the online non-exclusively-allocated CPUs with reserved CPUs set", func(ctx context.Context) {
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
			podGu = e2epod.NewPodClient(f).CreateSync(ctx, podGu)
			podMap[string(podGu.UID)] = podGu

			podBu := makeCPUManagerPod("non-gu-pod", []ctnAttribute{
				{
					ctnName:    "non-gu-container",
					cpuRequest: "200m",
					cpuLimit:   "300m",
				},
			})
			podBu = e2epod.NewPodClient(f).CreateSync(ctx, podBu)
			podMap[string(podBu.UID)] = podBu

			ginkgo.By("checking if the expected cpuset was assigned")

			exclusiveCPUs := getContainerAllowedCPUs(ctx, f, podGu, &podGu.Spec.Containers[0])
			// any full CPU is fine - we cannot nor we should predict which one, though
			gomega.Expect(exclusiveCPUs).To(BeACPUSetOfSize(cpuCount))
			gomega.Expect(exclusiveCPUs).To(OverlapWith(onlineCPUs))
			gomega.Expect(exclusiveCPUs).ToNot(OverlapWith(reservedCPUs))

			sharedCPUs := getContainerAllowedCPUs(ctx, f, podBu, &podBu.Spec.Containers[0])
			expectedSharedCPUs := onlineCPUs.Difference(exclusiveCPUs)
			gomega.Expect(sharedCPUs).To(EqualCPUSet(expectedSharedCPUs))
		})
	})

	ginkgo.When("running guaranteed pod tests", func() {
		ginkgo.It("should allocate exclusively a CPU to a 1-container pod", func(ctx context.Context) {
			cpuCount := 1
			reservedCPUs := cpuset.New(0)
			// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
			// check what we do have in the node.

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
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			cnt := &pod.Spec.Containers[0] // shortcut
			cpus := getContainerAllowedCPUs(ctx, f, pod, cnt)
			// any full CPU is fine - we cannot nor we should predict which one, though
			gomega.Expect(cpus).To(BeACPUSetOfSize(cpuCount))
			gomega.Expect(cpus).To(OverlapWith(onlineCPUs))
			gomega.Expect(cpus).ToNot(OverlapWith(reservedCPUs))
		})
	})

	ginkgo.When("running the SMT Alignment tests", func() {
		ginkgo.BeforeEach(func(ctx context.Context) {
			// strict SMT alignment is trivially verified and granted on non-SMT systems
			if smtLevel < 2 {
				e2eskipper.Skipf("Skipping CPU Manager %q tests since SMT disabled", cpumanager.FullPCPUsOnlyOption)
			}
		})

		ginkgo.It("should reject workload asking non-SMT-multiple of cpus", func(ctx context.Context) {
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
			cpuDetails := cpuDetailsFromNode(localNode)

			// our tests want to allocate a full core, so we need at last 2*2=4 virtual cpus
			if cpuDetails.Allocatable < int64(smtLevel*2) {
				e2eskipper.Skipf("Skipping CPU Manager %q tests since the CPU capacity < 4", cpumanager.FullPCPUsOnlyOption)
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
			// CreateSync would wait for pod to become Ready - which will never happen if production code works as intended!
			pod = e2epod.NewPodClient(f).Create(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("ensuring the testing pod is in failed state")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *corev1.Pod) (bool, error) {
				if pod.Status.Phase != corev1.PodPending {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err)

			ginkgo.By("ensuring the testing pod is failed for the expected reason")
			pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod).To(BeAPodInPhase(corev1.PodFailed))
			gomega.Expect(pod).To(HaveStatusReasonMatchingRegex(`SMT.*Alignment.*Error`))
		})

		ginkgo.It("should admit workload asking SMT-multiple of cpus", func(ctx context.Context) {
			// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
			// 1. are core siblings
			// 2. take a full core
			// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
			//          this means on more-than-2-way SMT systems this test will prove nothing
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
			cpuDetails := cpuDetailsFromNode(localNode)
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

			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("validating each container in the testing pod")
			for _, cnt := range pod.Spec.Containers {
				ginkgo.By(fmt.Sprintf("validating the container %s on pod %s", cnt.Name, pod.Name))
				cpus := getContainerAllowedCPUs(ctx, f, pod, &cnt)

				framework.Logf("validating cpus: %v", cpus)
				gomega.Expect(cpus).To(BeAMultipleOf(smtLevel))

				siblingsCPUs := makeThreadSiblingCPUSet(cpus)
				framework.Logf("pod %q container %q: siblings cpus: %v", pod.Name, cnt.Name, siblingsCPUs)
				gomega.Expect(cpus).To(EqualCPUSet(siblingsCPUs))
			}
		})
	})
})

// Matching helpers

func HaveStatusReasonMatchingRegex(expr string) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual *corev1.Pod) (bool, error) {
		re, err := regexp.Compile(expr)
		if err != nil {
			return false, err
		}
		return re.MatchString(actual.Status.Reason), nil
	}).WithTemplate("Pod {{.Actual.Namespace}}/{{.Actual.Name}} UID {{.Actual.UID}} reason {{.Actual.Status.Reason}} does not match regexp {{.Data}}", expr)
}

func EqualCPUSet(ref cpuset.CPUSet) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual cpuset.CPUSet) (bool, error) {
		return actual.Equals(ref), nil
	}).WithTemplate("CPUs {{.Actual}} not equal to {{.Data}}", ref)
}

func BeACPUSetOfSize(size int) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual cpuset.CPUSet) (bool, error) {
		return actual.Size() == size, nil
	}).WithTemplate("CPUSet {{.Actual}} size different from expected {{.Data}}", size)
}

func BeAMultipleOf(smtLevel int) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual cpuset.CPUSet) (bool, error) {
		return actual.Size()%smtLevel == 0, nil
	}).WithTemplate("CPUSet {{.Actual}} size not a multiple of {{.Data}}", smtLevel)
}

func OverlapWith(ref cpuset.CPUSet) types.GomegaMatcher {
	return gcustom.MakeMatcher(func(actual cpuset.CPUSet) (bool, error) {
		sharedCPUs := actual.Intersection(ref)
		return sharedCPUs.Size() > 0, nil
		// TODO: report shared CPUs in the error message?
	}).WithTemplate("CPUSet {{.Actual}} {{.To}} overlaps with {{.Data}} on CPUs", ref)
}

// Other helpers

func getContainerAllowedCPUs(ctx context.Context, f *framework.Framework, pod *corev1.Pod, cnt *corev1.Container) cpuset.CPUSet {
	// TODO: depends on the container command line, which is a hidden and unnecessary dep. We should access cgroups from inside the container.

	logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

	framework.Logf("got pod logs: %v", logs)
	cpus, err := cpuset.Parse(strings.TrimSpace(logs))
	framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)
	return cpus
}

func makeThreadSiblingCPUSet(cpus cpuset.CPUSet) cpuset.CPUSet {
	siblingsCPUs := cpuset.New()
	for _, cpuID := range cpus.UnsortedList() {
		siblingsCPUs = siblingsCPUs.Union(cpuSiblingListFromSysFS(int64(cpuID)))
	}
	return siblingsCPUs
}

func updateKubeletConfigIfNeeded(ctx context.Context, f *framework.Framework, desiredCfg *kubeletconfig.KubeletConfiguration) *corev1.Node {
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

func cpuDetailsFromNode(node *corev1.Node) nodeCPUDetails {
	localNodeCap := node.Status.Capacity
	cpuCap := localNodeCap[corev1.ResourceCPU]
	localNodeAlloc := node.Status.Allocatable
	cpuAlloc := localNodeAlloc[corev1.ResourceCPU]
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
