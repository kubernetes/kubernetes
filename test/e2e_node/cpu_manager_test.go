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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	cpuManagerFeatureGate = "CPUManager=true"
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

// TODO(balajismaniam): Make this func generic to all container runtimes.
func waitForContainerRemoval(ctnPartName string) {
	Eventually(func() bool {
		err := exec.Command("/bin/sh", "-c", fmt.Sprintf("if [ -n \"$(docker ps -a | grep -i %s)\" ]; then exit 1; fi", ctnPartName)).Run()
		if err != nil {
			return false
		}
		return true
	}, 2*time.Minute, 1*time.Second).Should(BeTrue())
}

func isHTEnabled() bool {
	err := exec.Command("/bin/sh", "-c", "if [[ $(lscpu | grep \"Thread(s) per core:\" | cut -c24) != \"2\" ]]; then exit 1; fi").Run()
	if err != nil {
		return false
	}
	return true
}

func getCPUSiblingList(cpuRes int64) string {
	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat /sys/devices/system/cpu/cpu%d/topology/thread_siblings_list | tr -d \"\n\r\"", cpuRes)).Output()
	framework.ExpectNoError(err)
	return string(out)
}

func setOldKubeletConfig(f *framework.Framework, oldCfg *kubeletconfig.KubeletConfiguration) {
	if oldCfg != nil {
		framework.ExpectNoError(setKubeletConfiguration(f, oldCfg))
	}
}

func enableCPUManagerInKubelet(f *framework.Framework) (oldCfg *kubeletconfig.KubeletConfiguration) {
	// Run only if the container runtime is Docker.
	// TODO(balajismaniam): Make this test generic to all container runtimes.
	framework.RunIfContainerRuntimeIs("docker")

	// Enable CPU Manager in Kubelet with static policy.
	oldCfg, err := getCurrentKubeletConfig()
	framework.ExpectNoError(err)
	clone, err := scheme.Scheme.DeepCopy(oldCfg)
	framework.ExpectNoError(err)
	newCfg := clone.(*kubeletconfig.KubeletConfiguration)

	// Enable CPU Manager using feature gate.
	if newCfg.FeatureGates != "" {
		newCfg.FeatureGates = fmt.Sprintf("%s,%s", cpuManagerFeatureGate, newCfg.FeatureGates)
	} else {
		newCfg.FeatureGates = cpuManagerFeatureGate
	}

	// Set the CPU Manager policy to static.
	newCfg.CPUManagerPolicy = string(cpumanager.PolicyStatic)

	// Set the CPU Manager reconcile period to 1 second.
	newCfg.CPUManagerReconcilePeriod = metav1.Duration{Duration: 1 * time.Second}

	// The Kubelet panics if either kube-reserved or system-reserved is not set
	// when CPU Manager is enabled. Set cpu in kube-reserved > 0 so that
	// kubelet doesn't panic.
	if _, ok := newCfg.KubeReserved["cpu"]; !ok {
		newCfg.KubeReserved["cpu"] = "200m"
	}
	// Update the Kubelet configuration.
	framework.ExpectNoError(setKubeletConfiguration(f, newCfg))

	// Wait for the Kubelet to be ready.
	Eventually(func() bool {
		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		return len(nodeList.Items) == 1
	}, time.Minute, time.Second).Should(BeTrue())

	return oldCfg
}

func runCPUManagerTests(f *framework.Framework) {
	var cpuCap, cpuAlloc, cpuRes int64
	var oldCfg *kubeletconfig.KubeletConfiguration
	var cpuListString, expAllowedCPUsListRegex string
	var cpuList []int
	var cpu1, cpu2 int
	var cset cpuset.CPUSet
	var err error
	var ctnAttrs []ctnAttribute
	var pod, pod1, pod2 *v1.Pod

	It("should assign CPUs as expected based on the Pod spec", func() {
		oldCfg = enableCPUManagerInKubelet(f)

		cpuCap, cpuAlloc, cpuRes = getLocalNodeCPUDetails(f)

		// Skip CPU Manager tests if the number of allocatable CPUs < 1.
		if cpuAlloc < 1 {
			framework.Skipf("Skipping CPU Manager tests since the number of allocatable CPUs < 1")
		}

		By("running a non-Gu pod")
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "non-gu-container",
				cpuRequest: "100m",
				cpuLimit:   "200m",
			},
		}
		pod = makeCPUManagerPod("non-gu-pod", ctnAttrs)
		pod = f.PodClient().CreateSync(pod)

		By("checking if the expected cpuset was assigned")
		expAllowedCPUsListRegex = fmt.Sprintf("^0-%d\n$", cpuCap-1)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod.Spec.Containers[0].Name, pod.Name))

		By("running a Gu pod")
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container",
				cpuRequest: "1000m",
				cpuLimit:   "1000m",
			},
		}
		pod = makeCPUManagerPod("gu-pod", ctnAttrs)
		pod = f.PodClient().CreateSync(pod)

		By("checking if the expected cpuset was assigned")
		cpu1 = 1
		if isHTEnabled() {
			cpuList = cpuset.MustParse(getCPUSiblingList(0)).ToSlice()
			cpu1 = cpuList[1]
		}
		expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
		err = f.PodClient().MatchContainerOutput(pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod.Spec.Containers[0].Name, pod.Name))

		By("running multiple Gu and non-Gu pods")
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

		By("checking if the expected cpuset was assigned")
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

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod1.Name, pod2.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod1.Spec.Containers[0].Name, pod1.Name))
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod2.Spec.Containers[0].Name, pod2.Name))

		// Skip rest of the tests if the number of allocatable CPUs < 2.
		if cpuAlloc < 2 {
			framework.Skipf("Skipping rest of the CPU Manager tests since the number of allocatable CPUs < 2")
		}

		By("running a Gu pod requesting multiple CPUs")
		ctnAttrs = []ctnAttribute{
			{
				ctnName:    "gu-container",
				cpuRequest: "2000m",
				cpuLimit:   "2000m",
			},
		}
		pod = makeCPUManagerPod("gu-pod", ctnAttrs)
		pod = f.PodClient().CreateSync(pod)

		By("checking if the expected cpuset was assigned")
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

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod.Spec.Containers[0].Name, pod.Name))

		By("running a Gu pod with multiple containers requesting integer CPUs")
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

		By("checking if the expected cpuset was assigned")
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

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod.Spec.Containers[0].Name, pod.Name))
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod.Spec.Containers[1].Name, pod.Name))

		By("running multiple Gu pods")
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

		By("checking if the expected cpuset was assigned")
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

		By("by deleting the pods and waiting for container removal")
		deletePods(f, []string{pod1.Name, pod2.Name})
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod1.Spec.Containers[0].Name, pod1.Name))
		waitForContainerRemoval(fmt.Sprintf("%s_%s", pod2.Spec.Containers[0].Name, pod2.Name))

		setOldKubeletConfig(f, oldCfg)
	})
}

// Serial because the test updates kubelet configuration.
var _ = framework.KubeDescribe("CPU Manager [Serial]", func() {
	f := framework.NewDefaultFramework("cpu-manager-test")

	Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})
})
