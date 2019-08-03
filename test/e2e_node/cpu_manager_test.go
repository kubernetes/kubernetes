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

package e2e_node

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	cpumanagerstate "k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
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
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(ctnAttr.cpuRequest),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
				},
				Limits: v1.ResourceList{
					v1.ResourceName(v1.ResourceCPU):    resource.MustParse(ctnAttr.cpuLimit),
					v1.ResourceName(v1.ResourceMemory): resource.MustParse("100Mi"),
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

func deletePods(f *framework.Framework, podNames []string) {
	for _, podName := range podNames {
		gp := int64(0)
		delOpts := metav1.DeleteOptions{
			GracePeriodSeconds: &gp,
		}
		f.PodClient().DeleteSync(podName, &delOpts, framework.DefaultPodDeletionTimeout)
	}
}

func getLocalNodeCPUDetails(f *framework.Framework) (cpuCapVal int64, cpuAllocVal int64, cpuResVal int64) {
	localNodeCap := getLocalNode(f).Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := getLocalNode(f).Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.Copy()
	cpuRes.Sub(cpuAlloc)

	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)

	return cpuCap.Value(), (cpuCap.Value() - cpuRes.Value()), cpuRes.Value()
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
		restoredState, err := cpumanagerstate.NewCheckpointState("/var/lib/kubelet", "cpu_manager_state", "static")
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

func getCPUSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

func deleteStateFile() {
	err := exec.Command("/bin/sh", "-c", "rm -f /var/lib/kubelet/cpu_manager_state").Run()
	framework.ExpectNoError(err, "error deleting state file")
}

func setOldKubeletConfig(f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration) {
	// Delete the CPU Manager state file so that the old Kubelet configuration
	// can take effect.i
	deleteStateFile()

	if oldCfg != nil {
		framework.ExpectNoError(setKubeletConfiguration(f, oldCfg))
	}
}

func disableCPUManagerInKubelet(f *framework.Framework) (oldCfg *kubeletconfig.KubeletConfiguration) {
	// Disable CPU Manager in Kubelet.
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}
	newCfg.FeatureGates["CPUManager"] = false

	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return oldCfg
}

func enableCPUManagerInKubelet(f *framework.Framework, cleanStateFile bool) (oldCfg *kubeletconfig.KubeletConfiguration) {
	// Enable CPU Manager in Kubelet with static policy.
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	} else {
		newCfg.FeatureGates["CPUManager"] = true
	}

	// After graduation of the CPU Manager feature to Beta, the CPU Manager
	// "none" policy is ON by default. But when we set the CPU Manager policy to
	// "static" in this test and the Kubelet is restarted so that "static"
	// policy can take effect, there will always be a conflict with the state
	// checkpointed in the disk (i.e., the policy checkpointed in the disk will
	// be "none" whereas we are trying to restart Kubelet with "static"
	// policy). Therefore, we delete the state file so that we can proceed
	// with the tests.
	// Only delete the state file at the begin of the tests.
	if cleanStateFile {
		deleteStateFile()
	}

	// Set the CPU Manager policy to static.
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)

	// Set the CPU Manager reconcile period to 1 second.
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	// The Kubelet panics if either kube-reserved or system-reserved is not set
	// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
	// kubelet doesn't panic.
	if newCfg.KubeReserved == nil {
		newCfg.KubeReserved = map[string]string{}
	}

	if _, ok := newCfg.KubeReserved["cpu"]; !ok {
		newCfg.KubeReserved["cpu"] = "200m"
	}
	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())

	return oldCfg
}

func runCPUManagerTests(f *framework.Framework) {
	var cpuCap, cpuAlloc int64
	var oldCfg *kubeletconfig.KubeletConfiguration
	var cpuListString, expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1, cpu2 int
	var cset cpuset.CPUSet
	var err error
	var ctnAttrs []ctnAttribute
	var pod, pod1, pod2 *v1.Pod

	ginkgo.It("should assign CPUs as expected based on the Pod spec", func() {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(f)

		// Skip CPU Manager tests altogether if the CPU capacity < 2.
		if cpuCap < 2 {
			framework.Skipf("Skipping CPU Manager tests since the CPU capacity < 2")
		}

		// Enable CPU Manager in the kubelet.
		oldCfg = enableCPUManagerInKubelet(f, true)

		ginkgo.By("running a non-Gu pod")
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
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		ginkgo.By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

		ginkgo.By("running a Gu pod")
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container",
				cpuRequest: "1000m",
				cpuLimit:   "1000m",
			},
		}
		pod = makeCPUManagerPod("gu-pod", ctnAttrs)
		pod = f.PodClient().CreateSync(pod)

		ginkgo.By("checking if the expected cpuset was assigned")
		cpu1 = 1
		if isHTEnabled() {
			cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
			cpu1 = cpuList[1]
		}
		expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		ginkgo.By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

		ginkgo.By("running multiple Gu and non-Gu pods")
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

		// Skip rest of the tests if CPU capacity < 3.
		if cpuCap < 3 {
			framework.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
		}

		ginkgo.By("running a Gu pod requesting multiple CPUs")
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
		if isHTEnabled() {
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

		ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
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
		}

		expAllowedCPUsListRegex = fmt.Sprintf("^%d|%d\n$", cpu1, cpu2)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[1].Name, pod.Name)

		ginkgo.By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
		waitForContainerRemoval(pod.Spec.Containers[1].Name, pod.Name, pod.Namespace)

		ginkgo.By("running multiple Gu pods")
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
		}
		expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		ginkgo.By("disable cpu manager in kubelet")
		disableCPUManagerInKubelet(f)

		ginkgo.By("by deleting the pod and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)

		ginkgo.By("enable cpu manager in kubelet without delete state file")
		enableCPUManagerInKubelet(f, false)

		ginkgo.By("wait for the deleted pod to be cleaned up from the state file")
		waitForStateFileCleanedUp()
		ginkgo.By("the deleted pod has already been deleted from the state file")

		setOldKubeletConfig(f, oldCfg)
	})
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("CPU Manager [Serial] [Feature:CPUManager][NodeAlphaFeature:CPUManager]", func() {
	f := framework.NewDefaultFramework("cpu-manager-test")

	ginkgo.Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})
})
