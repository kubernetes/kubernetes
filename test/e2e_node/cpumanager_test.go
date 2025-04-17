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

		ginkgo.It("should let the container access all the online non-exclusively-allocated CPUs when using a reserved CPUs set", Label("guaranteed", "exclusive-cpus"), func(ctx context.Context) {
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
			gomega.Expect(podGu).To(HaveContainerCPUsOverlapWith("gu-container", onlineCPUs))
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
			ginkgo.By("creating the test pod")
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podMap[string(pod.UID)] = pod

			ginkgo.By("checking if the expected cpuset was assigned")

			// we cannot nor we should predict which CPUs the container gets
			gomega.Expect(pod).To(HaveContainerCPUsCount("gu-container", cpuCount))
			gomega.Expect(pod).To(HaveContainerCPUsOverlapWith("gu-container", onlineCPUs))
			gomega.Expect(pod).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))
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
			gomega.Expect(podGu).To(HaveContainerCPUsOverlapWith("gu-container", usableCPUs))
			gomega.Expect(podGu).ToNot(HaveContainerCPUsOverlapWith("gu-container", reservedCPUs))

			exclusiveCPUs, err := getContainerAllowedCPUs(podGu, "gu-container")
			framework.ExpectNoError(err, "cannot get exclusive CPUs for pod %s/%s", podGu.Namespace, podGu.Name)
			expectedSharedCPUs := usableCPUs.Difference(exclusiveCPUs)
			gomega.Expect(podBu).To(HaveContainerCPUsEqualTo("non-gu-container", expectedSharedCPUs))
		})
	})

	// TODO full-cpus-only && strict-cpu-reservation

	ginkgo.When("running with SMT Alignment", ginkgo.Label("smt-alignment"), func() {
		var cpuDetails nodeCPUDetails

		ginkgo.BeforeEach(func(ctx context.Context) {
			// strict SMT alignment is trivially verified and granted on non-SMT systems
			if smtLevel < 2 {
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
				cpus, err := getContainerAllowedCPUs(pod, cnt.Name)
				framework.ExpectNoError(err, "cannot get cpus allocated to pod %s/%s cnt %s", pod.Namespace, pod.Name, cnt.Name)

				siblingsCPUs := makeThreadSiblingCPUSet(cpus)
				gomega.Expect(pod).To(HaveContainerCPUsEqualTo(cnt.Name, siblingsCPUs))
			}
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
	Name          string
	CurrentCPUs   string
	ExpectedCPUs  string
	Count         int
	Aligned       int
	CurrentQuota  string
	ExpectedQuota string
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

// Other helpers

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
