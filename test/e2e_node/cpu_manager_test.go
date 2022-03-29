/*
Copyright 2017 The Kubernetes Authors.

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
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	cpumanagerstate "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/types"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

// Helper for makeCPUManagerPod().
type ctnAttribute struct {
	ctnName    string
	cpuRequest string
	cpuLimit   string
}

// makeCPUMangerPod returns a pod with the provided ctnAttributes.
func makeCPUManagerPod(podName string, ctnAttributes []ctnAttribute) *v1.Pod {
	var containers []v1.Container
	for _, ctnAttr := range ctnAttributes {
		cpusetCmd := fmt.Sprintf("grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d")
		ctn := v1.Container{
			Name:  ctnAttr.ctnName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpuRequest),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpuLimit),
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
			},
			Command: []string{"sh", "-c", cpusetCmd},
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
		},
	}
}

func deletePodSyncByName(f *framework.Framework, podName string) {
	gp := int64(0)
	delOpts := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}
	f.PodClient().DeleteSync(podName, delOpts, framework.DefaultPodDeletionTimeout)
}

func deletePods(f *framework.Framework, podNames []string) {
	for _, podName := range podNames {
		deletePodSyncByName(f, podName)
	}
}

func getLocalNodeCPUDetails(f *framework.Framework) (cpuCapVal int64, cpuAllocVal int64, cpuResVal int64) {
	localNodeCap := getLocalNode(f).Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := getLocalNode(f).Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.DeepCopy()
	cpuRes.Sub(cpuAlloc)

	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)

	return cpuCap.Value(), cpuCap.Value() - cpuRes.Value(), cpuRes.Value()
}

func waitForContainerRemoval(containerName, podName, podNS string) {
	rs, _, err := getCRIClient()
	framework.ExpectNoError(err)
	gomega.Eventually(func() bool {
		containers, err := rs.ListContainers(&runtimeapi.ContainerFilter{
			LabelSelector: map[string]string{
				types.KubernetesPodNameLabel:       podName,
				types.KubernetesPodNamespaceLabel:  podNS,
				types.KubernetesContainerNameLabel: containerName,
			},
		})
		if err != nil {
			return false
		}
		return len(containers) == 0
	}, 2*time.Minute, 1*time.Second).Should(gomega.BeTrue())
}

func waitForStateFileCleanedUp() {
	gomega.Eventually(func() bool {
		restoredState, err := cpumanagerstate.NewCheckpointState("/var/lib/kubelet", "cpu_manager_state", "static", nil)
		framework.ExpectNoError(err, "failed to create testing cpumanager state instance")
		assignments := restoredState.GetCPUAssignments()
		if len(assignments) == 0 {
			return true
		}
		return false
	}, 2*time.Minute, 1*time.Second).Should(gomega.BeTrue())
}

func isHTEnabled() bool {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"Thread(s) per core:\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	threadsPerCore, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return threadsPerCore > 1
}

func isMultiNUMA() bool {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu | grep \"NUMA node(s):\" | cut -d \":\" -f 2").Output()
	framework.ExpectNoError(err)

	numaNodes, err := strconv.Atoi(strings.TrimSpace(string(outData)))
	framework.ExpectNoError(err)

	return numaNodes > 1
}

func getSMTLevel() int {
	cpuID := 0 // this is just the most likely cpu to be present in a random system. No special meaning besides this.
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuID)).Output()
	framework.ExpectNoError(err)
	// how many thread sibling you have = SMT level
	// example: 2-way SMT means 2 threads sibling for each thread
	cpus, err := cpuset.Parse(strings.TrimSpace(string(out)))
	framework.ExpectNoError(err)
	return cpus.Size()
}

func getCPUSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

func getCoreSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/core_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

type cpuManagerKubeletArguments struct {
	policyName              string
	enableCPUManager        bool
	enableCPUManagerOptions bool
	reservedSystemCPUs      cpuset.CPUSet
	options                 map[string]string
}

func configureCPUManagerInKubelet(oldCfg *kubeletconfig.KubeletConfiguration, kubeletArguments *cpuManagerKubeletArguments) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}

	newCfg.FeatureGates["CPUManager"] = kubeletArguments.enableCPUManager

	newCfg.FeatureGates["CPUManagerPolicyOptions"] = kubeletArguments.enableCPUManagerOptions
	newCfg.FeatureGates["CPUManagerPolicyBetaOptions"] = kubeletArguments.enableCPUManagerOptions
	newCfg.FeatureGates["CPUManagerPolicyAlphaOptions"] = kubeletArguments.enableCPUManagerOptions

	newCfg.CPUManagerPolicy = kubeletArguments.policyName
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	if kubeletArguments.options != nil {
		newCfg.CPUManagerPolicyOptions = kubeletArguments.options
	}

	if kubeletArguments.reservedSystemCPUs.Size() > 0 {
		cpus := kubeletArguments.reservedSystemCPUs.String()
		framework.Logf("configureCPUManagerInKubelet: using reservedSystemCPUs=%q", cpus)
		newCfg.ReservedSystemCPUs = cpus
	} else {
		// The Kubelet panics if either kube-reserved or system-reserved is not set
		// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
		// kubelet doesn't panic.
		if newCfg.KubeReserved == nil {
			newCfg.KubeReserved = map[string]string{}
		}

		if _, ok := newCfg.KubeReserved["cpu"]; !ok {
			newCfg.KubeReserved["cpu"] = "200m"
		}
	}

	return newCfg
}

func runGuPodTest(f *framework.Framework, cpuCount int) {
	var pod *v1.Pod

	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
			cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
		},
	}
	pod = makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	// any full CPU is fine - we cannot nor we should predict which one, though
	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.Logf("got pod logs: %v", logs)
		cpus, err := cpuset.Parse(strings.TrimSpace(logs))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.ExpectEqual(cpus.Size(), cpuCount, "expected cpu set size == %d, got %q", cpuCount, cpus.String())
	}

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForAllContainerRemoval(pod.Name, pod.Namespace)
}

func runNonGuPodTest(f *framework.Framework, cpuCap int64) {
	var ctnAttrs []ctnAttribute
	var err error
	var pod *v1.Pod
	var expAllowedCPUsListRegex string

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "100m",
			cpuLimit:   "200m",
		},
	}
	pod = makeCPUManagerPod("non-gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	expAllowedCPUsListRegex = fmt.Sprintf("^0-%d\n$", cpuCap-1)
	// on the single CPU node the only possible value is 0
	if cpuCap == 1 {
		expAllowedCPUsListRegex = "^0\n$"
	}
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
}

func runMultipleGuNonGuPods(f *framework.Framework, cpuCap int64, cpuAlloc int64) {
	var cpuListString, expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1 int
	var cset cpuset.CPUSet
	var err error
	var ctnAttrs []ctnAttribute
	var pod1, pod2 *v1.Pod

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod1 = makeCPUManagerPod("gu-pod", ctnAttrs)
	pod1 = f.PodClient().CreateSync(pod1)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "200m",
			cpuLimit:   "300m",
		},
	}
	pod2 = makeCPUManagerPod("non-gu-pod", ctnAttrs)
	pod2 = f.PodClient().CreateSync(pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1 = 1
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		cpu1 = cpuList[1]
	} else if isMultiNUMA() {
		cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
		if len(cpuList) > 1 {
			cpu1 = cpuList[1]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = f.PodClient().MatchContainerOutput(pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	cpuListString = "0"
	if cpuAlloc > 2 {
		cset = cpuset.MustParse(fmt.Sprintf("0-%d", cpuCap-1))
		cpuListString = fmt.Sprintf("%s", cset.Difference(cpuset.NewCPUSet(cpu1)))
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = f.PodClient().MatchContainerOutput(pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)
	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)
}

func runMultipleCPUGuPod(f *framework.Framework) {
	var cpuListString, expAllowedCPUsListRegex string
	var cpuList []int
	var cset cpuset.CPUSet
	var err error
	var ctnAttrs []ctnAttribute
	var pod *v1.Pod

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: "2000m",
			cpuLimit:   "2000m",
		},
	}
	pod = makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpuListString = "1-2"
	if isMultiNUMA() {
		cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
		if len(cpuList) > 1 {
			cset = cpuset.MustParse(getCPUSiblingList(int64(cpuList[1])))
			if !isHTEnabled() && len(cpuList) > 2 {
				cset = cpuset.MustParse(fmt.Sprintf("%d-%d", cpuList[1], cpuList[2]))
			}
			cpuListString = fmt.Sprintf("%s", cset)
		}
	} else if isHTEnabled() {
		cpuListString = "2-3"
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cset = cpuset.MustParse(getCPUSiblingList(1))
			cpuListString = fmt.Sprintf("%s", cset)
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
}

func runMultipleCPUContainersGuPod(f *framework.Framework) {
	var expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1, cpu2 int
	var err error
	var ctnAttrs []ctnAttribute
	var pod *v1.Pod
	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container1",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
		{
			ctnName:    "gu-container2",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod = makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
		if isMultiNUMA() {
			cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
			if len(cpuList) > 1 {
				cpu2 = cpuList[1]
			}
		}
	} else if isMultiNUMA() {
		cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
		if len(cpuList) > 2 {
			cpu1, cpu2 = cpuList[1], cpuList[2]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d|%d\n$", cpu1, cpu2)
	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[1].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[1].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod.Name})
	waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
	waitForContainerRemoval(pod.Spec.Containers[1].Name, pod.Name, pod.Namespace)
}

func runMultipleGuPods(f *framework.Framework) {
	var expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1, cpu2 int
	var err error
	var ctnAttrs []ctnAttribute
	var pod1, pod2 *v1.Pod

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container1",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod1 = makeCPUManagerPod("gu-pod1", ctnAttrs)
	pod1 = f.PodClient().CreateSync(pod1)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container2",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod2 = makeCPUManagerPod("gu-pod2", ctnAttrs)
	pod2 = f.PodClient().CreateSync(pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
		if isMultiNUMA() {
			cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
			if len(cpuList) > 1 {
				cpu2 = cpuList[1]
			}
		}
	} else if isMultiNUMA() {
		cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
		if len(cpuList) > 2 {
			cpu1, cpu2 = cpuList[1], cpuList[2]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = f.PodClient().MatchContainerOutput(pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu2)
	err = f.PodClient().MatchContainerOutput(pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)
	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)
}

func runCPUManagerTests(f *framework.Framework) {
	var cpuCap, cpuAlloc int64
	var oldCfg *kubeletconfig.KubeletConfiguration
	var expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1 int
	var err error
	var ctnAttrs []ctnAttribute
	var pod *v1.Pod

	ginkgo.BeforeEach(func() {
		var err error
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig()
			framework.ExpectNoError(err)
		}
	})

	ginkgo.It("should assign CPUs as expected based on the Pod spec", func() {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(f)

		// Skip CPU Manager tests altogether if the CPU capacity < 2.
		if cpuCap < 2 {
			e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < 2")
		}

		// Enable CPU Manager in the kubelet.
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			enableCPUManager:   true,
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(f, newCfg, true)

		ginkgo.By("running a non-Gu pod")
		runNonGuPodTest(f, cpuCap)

		ginkgo.By("running a Gu pod")
		runGuPodTest(f, 1)

		ginkgo.By("running multiple Gu and non-Gu pods")
		runMultipleGuNonGuPods(f, cpuCap, cpuAlloc)

		// Skip rest of the tests if CPU capacity < 3.
		if cpuCap < 3 {
			e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
		}

		ginkgo.By("running a Gu pod requesting multiple CPUs")
		runMultipleCPUGuPod(f)

		ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
		runMultipleCPUContainersGuPod(f)

		ginkgo.By("running multiple Gu pods")
		runMultipleGuPods(f)

		ginkgo.By("test for automatically remove inactive pods from cpumanager state file.")
		// First running a Gu Pod,
		// second disable cpu manager in kubelet,
		// then delete the Gu Pod,
		// then enable cpu manager in kubelet,
		// at last wait for the reconcile process cleaned up the state file, if the assignments map is empty,
		// it proves that the automatic cleanup in the reconcile process is in effect.
		ginkgo.By("running a Gu pod for test remove")
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container-testremove",
				cpuRequest: "1000m",
				cpuLimit:   "1000m",
			},
		}
		pod = makeCPUManagerPod("gu-pod-testremove", ctnAttrs)
		pod = f.PodClient().CreateSync(pod)

		ginkgo.By("checking if the expected cpuset was assigned")
		cpu1 = 1
		if isHTEnabled() {
			cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
			cpu1 = cpuList[1]
		} else if isMultiNUMA() {
			cpuList = cpuset.MustParse(getCoreSiblingList(0)).ToSlice()
			if len(cpuList) > 1 {
				cpu1 = cpuList[1]
			}
		}
		expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		ginkgo.By("disable cpu manager in kubelet")
		newCfg = configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			enableCPUManager:   false,
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(f, newCfg, false)

		ginkgo.By("by deleting the pod and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

		ginkgo.By("enable cpu manager in kubelet without delete state file")
		newCfg = configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			enableCPUManager:   true,
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(f, newCfg, false)

		ginkgo.By("wait for the deleted pod to be cleaned up from the state file")
		waitForStateFileCleanedUp()
		ginkgo.By("the deleted pod has already been deleted from the state file")
	})

	ginkgo.It("should assign CPUs as expected with enhanced policy based on strict SMT alignment", func() {
		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(f)
		smtLevel := getSMTLevel()

		// strict SMT alignment is trivially verified and granted on non-SMT systems
		if smtLevel < 2 {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since SMT disabled", fullCPUsOnlyOpt)
		}

		// our tests want to allocate a full core, so we need at last 2*2=4 virtual cpus
		if cpuAlloc < int64(smtLevel*2) {
			e2eskipper.Skipf("Skipping CPU Manager %s tests since the CPU capacity < 4", fullCPUsOnlyOpt)
		}

		framework.Logf("SMT level %d", smtLevel)

		// TODO: we assume the first available CPUID is 0, which is pretty fair, but we should probably
		// check what we do have in the node.
		cpuPolicyOptions := map[string]string{
			cpumanager.FullPCPUsOnlyOption: "true",
		}
		newCfg := configureCPUManagerInKubelet(oldCfg,
			&cpuManagerKubeletArguments{
				policyName:              string(cpumanager.PolicyStatic),
				enableCPUManager:        true,
				reservedSystemCPUs:      cpuset.NewCPUSet(0),
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(f, newCfg, true)

		// the order between negative and positive doesn't really matter
		runSMTAlignmentNegativeTests(f)
		runSMTAlignmentPositiveTests(f, smtLevel)
	})

	ginkgo.AfterEach(func() {
		updateKubeletConfig(f, oldCfg, true)
	})
}

func runSMTAlignmentNegativeTests(f *framework.Framework) {
	// negative test: try to run a container whose requests aren't a multiple of SMT level, expect a rejection
	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container-neg",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod := makeCPUManagerPod("gu-pod", ctnAttrs)
	// CreateSync would wait for pod to become Ready - which will never happen if production code works as intended!
	pod = f.PodClient().Create(pod)

	err := e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
		if pod.Status.Phase != v1.PodPending {
			return true, nil
		}
		return false, nil
	})
	framework.ExpectNoError(err)
	pod, err = f.PodClient().Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if pod.Status.Phase != v1.PodFailed {
		framework.Failf("pod %s not failed: %v", pod.Name, pod.Status)
	}
	if !isSMTAlignmentError(pod) {
		framework.Failf("pod %s failed for wrong reason: %q", pod.Name, pod.Status.Reason)
	}

	deletePodSyncByName(f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(pod.Name, pod.Namespace)
}

func runSMTAlignmentPositiveTests(f *framework.Framework, smtLevel int) {
	// positive test: try to run a container whose requests are a multiple of SMT level, check allocated cores
	// 1. are core siblings
	// 2. take a full core
	// WARNING: this assumes 2-way SMT systems - we don't know how to access other SMT levels.
	//          this means on more-than-2-way SMT systems this test will prove nothing
	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container-pos",
			cpuRequest: "2000m",
			cpuLimit:   "2000m",
		},
	}
	pod := makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = f.PodClient().CreateSync(pod)

	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.Logf("got pod logs: %v", logs)
		cpus, err := cpuset.Parse(strings.TrimSpace(logs))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)

		validateSMTAlignment(cpus, smtLevel, pod, &cnt)
	}

	deletePodSyncByName(f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(pod.Name, pod.Namespace)
}

func validateSMTAlignment(cpus cpuset.CPUSet, smtLevel int, pod *v1.Pod, cnt *v1.Container) {
	framework.Logf("validating cpus: %v", cpus)

	if cpus.Size()%smtLevel != 0 {
		framework.Failf("pod %q cnt %q received non-smt-multiple cpuset %v (SMT level %d)", pod.Name, cnt.Name, cpus, smtLevel)
	}

	// now check all the given cpus are thread siblings.
	// to do so the easiest way is to rebuild the expected set of siblings from all the cpus we got.
	// if the expected set matches the given set, the given set was good.
	b := cpuset.NewBuilder()
	for _, cpuID := range cpus.ToSliceNoSort() {
		threadSiblings, err := cpuset.Parse(strings.TrimSpace(getCPUSiblingList(int64(cpuID))))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)
		b.Add(threadSiblings.ToSliceNoSort()...)
	}
	siblingsCPUs := b.Result()

	framework.Logf("siblings cpus: %v", siblingsCPUs)
	if !siblingsCPUs.Equals(cpus) {
		framework.Failf("pod %q cnt %q received non-smt-aligned cpuset %v (expected %v)", pod.Name, cnt.Name, cpus, siblingsCPUs)
	}
}

func isSMTAlignmentError(pod *v1.Pod) bool {
	re := regexp.MustCompile(`SMT.*Alignment.*Error`)
	return re.MatchString(pod.Status.Reason)
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("CPU Manager [Serial] [Feature:CPUManager]", func() {
	f := framework.NewDefaultFramework("cpu-manager-test")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})
})
