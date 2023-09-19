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
	"bufio"
	"context"
	"fmt"
	"os/exec"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	"k8s.io/kubelet/pkg/types"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

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
	memRequest string
	memLimit   string
	command    string
}

func (ca ctnAttribute) Clone() ctnAttribute {
	return ctnAttribute{
		ctnName:    ca.ctnName,
		cpuRequest: ca.cpuRequest,
		cpuLimit:   ca.cpuLimit,
		memRequest: ca.memRequest,
		memLimit:   ca.memLimit,
		command:    ca.command,
	}
}

func (ca ctnAttribute) WithDefaults() ctnAttribute {
	cmd := "grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d"
	memRequest := "100Mi"
	memLimit := "100Mi"

	ret := ca.Clone()

	if ret.command == "" {
		ret.command = cmd
	}
	if ret.memRequest == "" {
		ret.memRequest = memRequest
	}
	if ret.memLimit == "" {
		ret.memLimit = memLimit
	}

	return ret
}

// xref: https://kubernetes.io/docs/concepts/architecture/cgroups/#check-cgroup-version
const cgroupGetFSTypeCmd = `/bin/stat -fc %T /sys/fs/cgroup/`

func cgroupVersionFromMountInfo(minfo string) (string, error) {
	scanner := bufio.NewScanner(strings.NewReader(minfo))
	for scanner.Scan() {
		line := scanner.Text()
		if err := scanner.Err(); err != nil {
			return "", err
		}
		// intentional catchall for both v1 and v2
		if !strings.HasPrefix(line, "cgroup") {
			continue
		}

		line = strings.TrimSpace(line)
		framework.Logf("matched cgroup mountinfo line: [%s]", line)

		items := strings.Fields(line)
		// man 5 fstab: "The third field (fs_vfstype). This field describes the type of the filesystem [...]"
		return cgroupVersionFromMountVFSType(items[2])
	}
	return "", fmt.Errorf("unrecognized cgroup FS type from mountInfo")
}

func cgroupVersionFromMountVFSType(fsVFSType string) (string, error) {
	switch fsVFSType {
	case "cgroup":
		return "v1", nil
	case "cgroup2":
		return "v2", nil
	default:
		return "", fmt.Errorf("unsupported type: %s", fsVFSType)
	}
}

func cgroupVersionFromCgroupFSType(fsType string) (string, error) {
	switch fsType {
	case "tmpfs":
		return "v1", nil
	case "cgroup2fs":
		return "v2", nil
	default:
		return "", fmt.Errorf("unsupported type: %s", fsType)
	}
}

func cgroupVersionFromHostData(fsType, mountInfo string) (string, error) {
	// preferred because recommended in the docs
	ver, err := cgroupVersionFromCgroupFSType(fsType)
	if err == nil {
		return ver, nil
	}
	framework.Logf("failed to detect cgroup version from fsType, trying from mountInfo (err=%v)", err)
	return cgroupVersionFromMountInfo(mountInfo)
}

func cgroupCpusetPathFromCgroupVersion(ver string) (string, error) {
	switch ver {
	case "v1":
		return "/sys/fs/cgroup/cpuset/cpuset.effective_cpus", nil
	case "v2":
		return "/sys/fs/cgroup/cpuset.cpus.effective", nil
	default:
		return "", fmt.Errorf("unsupported version: %v", ver)
	}
}

// makeCPUMangerPod returns a pod with the provided ctnAttributes.
func makeCPUManagerPod(podName string, ctnAttributes []ctnAttribute) *v1.Pod {
	var containers []v1.Container
	for idx := range ctnAttributes {
		ctnAttr := ctnAttributes[idx].WithDefaults()

		ctn := v1.Container{
			Name:  ctnAttr.ctnName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpuRequest),
					v1.ResourceMemory: resource.MustParse(ctnAttr.memRequest),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpuLimit),
					v1.ResourceMemory: resource.MustParse(ctnAttr.memLimit),
				},
			},
			Command: []string{"sh", "-c", ctnAttr.command},
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

func deletePodSyncByName(ctx context.Context, f *framework.Framework, podName string) {
	gp := int64(0)
	delOpts := metav1.DeleteOptions{
		GracePeriodSeconds: &gp,
	}
	e2epod.NewPodClient(f).DeleteSync(ctx, podName, delOpts, e2epod.DefaultPodDeletionTimeout)
}

func deletePods(ctx context.Context, f *framework.Framework, podNames []string) {
	for _, podName := range podNames {
		deletePodSyncByName(ctx, f, podName)
	}
}

func getLocalNodeCPUDetails(ctx context.Context, f *framework.Framework) (cpuCapVal int64, cpuAllocVal int64, cpuResVal int64) {
	localNodeCap := getLocalNode(ctx, f).Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := getLocalNode(ctx, f).Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.DeepCopy()
	cpuRes.Sub(cpuAlloc)

	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)

	return cpuCap.Value(), cpuCap.Value() - cpuRes.Value(), cpuRes.Value()
}

func waitForContainerRemoval(ctx context.Context, containerName, podName, podNS string) {
	rs, _, err := getCRIClient()
	framework.ExpectNoError(err)
	gomega.Eventually(ctx, func(ctx context.Context) bool {
		containers, err := rs.ListContainers(ctx, &runtimeapi.ContainerFilter{
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
	enableCPUManagerOptions bool
	reservedSystemCPUs      cpuset.CPUSet
	options                 map[string]string
}

func configureCPUManagerInKubelet(oldCfg *kubeletconfig.KubeletConfiguration, kubeletArguments *cpuManagerKubeletArguments) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}

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

func runGuPodTest(ctx context.Context, f *framework.Framework, cpuCount int) {
	var pod *v1.Pod

	ctnAttrs := []ctnAttribute{
		{
			ctnName:    "gu-container",
			cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
			cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
		},
	}
	pod = makeCPUManagerPod("gu-pod", ctnAttrs)
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	// any full CPU is fine - we cannot nor we should predict which one, though
	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.Logf("got pod logs: %v", logs)
		cpus, err := cpuset.Parse(strings.TrimSpace(logs))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.ExpectEqual(cpus.Size(), cpuCount, "expected cpu set size == %d, got %q", cpuCount, cpus.String())
	}

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod.Name})
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func runNonGuPodTest(ctx context.Context, f *framework.Framework, cpuCap int64) {
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
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	expAllowedCPUsListRegex = fmt.Sprintf("^0-%d\n$", cpuCap-1)
	// on the single CPU node the only possible value is 0
	if cpuCap == 1 {
		expAllowedCPUsListRegex = "^0\n$"
	}
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod.Name})
	waitForContainerRemoval(ctx, pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
}

func mustParseCPUSet(s string) cpuset.CPUSet {
	res, err := cpuset.Parse(s)
	framework.ExpectNoError(err)
	return res
}

func runMultipleGuNonGuPods(ctx context.Context, f *framework.Framework, cpuCap int64, cpuAlloc int64) {
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
	pod1 = e2epod.NewPodClient(f).CreateSync(ctx, pod1)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "non-gu-container",
			cpuRequest: "200m",
			cpuLimit:   "300m",
		},
	}
	pod2 = makeCPUManagerPod("non-gu-pod", ctnAttrs)
	pod2 = e2epod.NewPodClient(f).CreateSync(ctx, pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1 = 1
	if isHTEnabled() {
		cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
		cpu1 = cpuList[1]
	} else if isMultiNUMA() {
		cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
		if len(cpuList) > 1 {
			cpu1 = cpuList[1]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	cpuListString = "0"
	if cpuAlloc > 2 {
		cset = mustParseCPUSet(fmt.Sprintf("0-%d", cpuCap-1))
		cpuListString = fmt.Sprintf("%s", cset.Difference(cpuset.New(cpu1)))
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)
	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(ctx, pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(ctx, pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)
}

func runMultipleCPUGuPod(ctx context.Context, f *framework.Framework) {
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
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpuListString = "1-2"
	if isMultiNUMA() {
		cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
		if len(cpuList) > 1 {
			cset = mustParseCPUSet(getCPUSiblingList(int64(cpuList[1])))
			if !isHTEnabled() && len(cpuList) > 2 {
				cset = mustParseCPUSet(fmt.Sprintf("%d-%d", cpuList[1], cpuList[2]))
			}
			cpuListString = fmt.Sprintf("%s", cset)
		}
	} else if isHTEnabled() {
		cpuListString = "2-3"
		cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
		if cpuList[1] != 1 {
			cset = mustParseCPUSet(getCPUSiblingList(1))
			cpuListString = fmt.Sprintf("%s", cset)
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%s\n$", cpuListString)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod.Name})
	waitForContainerRemoval(ctx, pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
}

func runMultipleCPUContainersGuPod(ctx context.Context, f *framework.Framework) {
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
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
		if isMultiNUMA() {
			cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
			if len(cpuList) > 1 {
				cpu2 = cpuList[1]
			}
		}
	} else if isMultiNUMA() {
		cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
		if len(cpuList) > 2 {
			cpu1, cpu2 = cpuList[1], cpuList[2]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d|%d\n$", cpu1, cpu2)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[0].Name, pod.Name)

	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[1].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod.Spec.Containers[1].Name, pod.Name)

	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod.Name})
	waitForContainerRemoval(ctx, pod.Spec.Containers[0].Name, pod.Name, pod.Namespace)
	waitForContainerRemoval(ctx, pod.Spec.Containers[1].Name, pod.Name, pod.Namespace)
}

func runMultipleGuPods(ctx context.Context, f *framework.Framework) {
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
	pod1 = e2epod.NewPodClient(f).CreateSync(ctx, pod1)

	ctnAttrs = []ctnAttribute{
		{
			ctnName:    "gu-container2",
			cpuRequest: "1000m",
			cpuLimit:   "1000m",
		},
	}
	pod2 = makeCPUManagerPod("gu-pod2", ctnAttrs)
	pod2 = e2epod.NewPodClient(f).CreateSync(ctx, pod2)

	ginkgo.By("checking if the expected cpuset was assigned")
	cpu1, cpu2 = 1, 2
	if isHTEnabled() {
		cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
		if cpuList[1] != 1 {
			cpu1, cpu2 = cpuList[1], 1
		}
		if isMultiNUMA() {
			cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
			if len(cpuList) > 1 {
				cpu2 = cpuList[1]
			}
		}
	} else if isMultiNUMA() {
		cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
		if len(cpuList) > 2 {
			cpu1, cpu2 = cpuList[1], cpuList[2]
		}
	}
	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod1.Name, pod1.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod1.Spec.Containers[0].Name, pod1.Name)

	expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu2)
	err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod2.Name, pod2.Spec.Containers[0].Name, expAllowedCPUsListRegex)
	framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
		pod2.Spec.Containers[0].Name, pod2.Name)
	ginkgo.By("by deleting the pods and waiting for container removal")
	deletePods(ctx, f, []string{pod1.Name, pod2.Name})
	waitForContainerRemoval(ctx, pod1.Spec.Containers[0].Name, pod1.Name, pod1.Namespace)
	waitForContainerRemoval(ctx, pod2.Spec.Containers[0].Name, pod2.Name, pod2.Namespace)
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

	ginkgo.BeforeEach(func(ctx context.Context) {
		var err error
		if oldCfg == nil {
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
		}
	})

	ginkgo.It("should assign CPUs as expected based on the Pod spec", func(ctx context.Context) {
		cpuCap, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)

		// Skip CPU Manager tests altogether if the CPU capacity < 2.
		if cpuCap < 2 {
			e2eskipper.Skipf("Skipping CPU Manager tests since the CPU capacity < 2")
		}

		// Enable CPU Manager in the kubelet.
		newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
			policyName:         string(cpumanager.PolicyStatic),
			reservedSystemCPUs: cpuset.CPUSet{},
		})
		updateKubeletConfig(ctx, f, newCfg, true)

		ginkgo.By("running a non-Gu pod")
		runNonGuPodTest(ctx, f, cpuCap)

		ginkgo.By("running a Gu pod")
		runGuPodTest(ctx, f, 1)

		ginkgo.By("running multiple Gu and non-Gu pods")
		runMultipleGuNonGuPods(ctx, f, cpuCap, cpuAlloc)

		// Skip rest of the tests if CPU capacity < 3.
		if cpuCap < 3 {
			e2eskipper.Skipf("Skipping rest of the CPU Manager tests since CPU capacity < 3")
		}

		ginkgo.By("running a Gu pod requesting multiple CPUs")
		runMultipleCPUGuPod(ctx, f)

		ginkgo.By("running a Gu pod with multiple containers requesting integer CPUs")
		runMultipleCPUContainersGuPod(ctx, f)

		ginkgo.By("running multiple Gu pods")
		runMultipleGuPods(ctx, f)

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
		pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

		ginkgo.By("checking if the expected cpuset was assigned")
		cpu1 = 1
		if isHTEnabled() {
			cpuList = mustParseCPUSet(getCPUSiblingList(0)).List()
			cpu1 = cpuList[1]
		} else if isMultiNUMA() {
			cpuList = mustParseCPUSet(getCoreSiblingList(0)).List()
			if len(cpuList) > 1 {
				cpu1 = cpuList[1]
			}
		}
		expAllowedCPUsListRegex = fmt.Sprintf("^%d\n$", cpu1)
		err = e2epod.NewPodClient(f).MatchContainerOutput(ctx, pod.Name, pod.Spec.Containers[0].Name, expAllowedCPUsListRegex)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]",
			pod.Spec.Containers[0].Name, pod.Name)

		deletePodSyncByName(ctx, f, pod.Name)
		// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
		// this is in turn needed because we will have an unavoidable (in the current framework) race with the
		// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
		waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
	})

	ginkgo.It("should assign CPUs as expected with enhanced policy based on strict SMT alignment", func(ctx context.Context) {
		fullCPUsOnlyOpt := fmt.Sprintf("option=%s", cpumanager.FullPCPUsOnlyOption)
		_, cpuAlloc, _ = getLocalNodeCPUDetails(ctx, f)
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
				reservedSystemCPUs:      cpuset.New(0),
				enableCPUManagerOptions: true,
				options:                 cpuPolicyOptions,
			},
		)
		updateKubeletConfig(ctx, f, newCfg, true)

		// the order between negative and positive doesn't really matter
		runSMTAlignmentNegativeTests(ctx, f)
		runSMTAlignmentPositiveTests(ctx, f, smtLevel)
	})

	ginkgo.AfterEach(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})
}

func runSMTAlignmentNegativeTests(ctx context.Context, f *framework.Framework) {
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
	pod = e2epod.NewPodClient(f).Create(ctx, pod)

	err := e2epod.WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "Failed", 30*time.Second, func(pod *v1.Pod) (bool, error) {
		if pod.Status.Phase != v1.PodPending {
			return true, nil
		}
		return false, nil
	})
	framework.ExpectNoError(err)
	pod, err = e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if pod.Status.Phase != v1.PodFailed {
		framework.Failf("pod %s not failed: %v", pod.Name, pod.Status)
	}
	if !isSMTAlignmentError(pod) {
		framework.Failf("pod %s failed for wrong reason: %q", pod.Name, pod.Status.Reason)
	}

	deletePodSyncByName(ctx, f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func runSMTAlignmentPositiveTests(ctx context.Context, f *framework.Framework, smtLevel int) {
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
	pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

	for _, cnt := range pod.Spec.Containers {
		ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

		logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, cnt.Name)
		framework.ExpectNoError(err, "expected log not found in container [%s] of pod [%s]", cnt.Name, pod.Name)

		framework.Logf("got pod logs: %v", logs)
		cpus, err := cpuset.Parse(strings.TrimSpace(logs))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)

		validateSMTAlignment(cpus, smtLevel, pod, &cnt)
	}

	deletePodSyncByName(ctx, f, pod.Name)
	// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
	// this is in turn needed because we will have an unavoidable (in the current framework) race with th
	// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
	waitForAllContainerRemoval(ctx, pod.Name, pod.Namespace)
}

func validateSMTAlignment(cpus cpuset.CPUSet, smtLevel int, pod *v1.Pod, cnt *v1.Container) {
	framework.Logf("validating cpus: %v", cpus)

	if cpus.Size()%smtLevel != 0 {
		framework.Failf("pod %q cnt %q received non-smt-multiple cpuset %v (SMT level %d)", pod.Name, cnt.Name, cpus, smtLevel)
	}

	// now check all the given cpus are thread siblings.
	// to do so the easiest way is to rebuild the expected set of siblings from all the cpus we got.
	// if the expected set matches the given set, the given set was good.
	siblingsCPUs := cpuset.New()
	for _, cpuID := range cpus.UnsortedList() {
		threadSiblings, err := cpuset.Parse(strings.TrimSpace(getCPUSiblingList(int64(cpuID))))
		framework.ExpectNoError(err, "parsing cpuset from logs for [%s] of pod [%s]", cnt.Name, pod.Name)
		siblingsCPUs = siblingsCPUs.Union(threadSiblings)
	}

	framework.Logf("siblings cpus: %v", siblingsCPUs)
	if !siblingsCPUs.Equals(cpus) {
		framework.Failf("pod %q cnt %q received non-smt-aligned cpuset %v (expected %v)", pod.Name, cnt.Name, cpus, siblingsCPUs)
	}
}

func isSMTAlignmentError(pod *v1.Pod) bool {
	re := regexp.MustCompile(`SMT.*Alignment.*Error`)
	return re.MatchString(pod.Status.Reason)
}

func updateKubeletConfigIfNeeded(ctx context.Context, f *framework.Framework, desiredCfg *kubeletconfig.KubeletConfiguration) {
	curCfg, err := getCurrentKubeletConfig(ctx)
	framework.ExpectNoError(err)

	if equalKubeletConfiguration(curCfg, desiredCfg) {
		framework.Logf("Kubelet configuration already compliant, nothing to do")
		return // nothing to do!
	}
	framework.Logf("Updating Kubelet configuration")
	updateKubeletConfig(ctx, f, desiredCfg, true)
}

func equalKubeletConfiguration(cfgA, cfgB *kubeletconfig.KubeletConfiguration) bool {
	cfgA = cfgA.DeepCopy()
	cfgB = cfgB.DeepCopy()
	// we care only about the payload, force metadata to be uniform
	cfgA.TypeMeta = metav1.TypeMeta{}
	cfgB.TypeMeta = metav1.TypeMeta{}
	return reflect.DeepEqual(cfgA, cfgB)
}

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
   If in the future we decide to pivot to full spec independency, little changes will be needed.
   Finally, worth pointing out that the previous cpumanager e2e test incarnation implemented the same
   concept in a more convoluted way with function helpers, so arguably using Ordered and making it
   explicit is already an improvement.
*/

var _ = SIGDescribe("CPU Manager [Serial] [Feature:CPUManager]", func() {
	f := framework.NewDefaultFramework("cpu-manager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("With kubeconfig updated with static CPU Manager policy run the CPU Manager tests", func() {
		runCPUManagerTests(f)
	})

	ginkgo.Context("With kubeconfig lazily updated with the CPU Manager static policy", ginkgo.Ordered, func() {
		var refPod *v1.Pod
		// original kubeletconfig before the context start, to be restored
		var oldCfg *kubeletconfig.KubeletConfiguration

		var mountInfo string
		var cgroupFSType string
		var cgroupVersion string
		var cgroupCpusetPath string

		ginkgo.BeforeAll(func(ctx context.Context) {
			var err error
			oldCfg, err = getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)

			refPod = &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "ref-be-pod",
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Name:    "ref-be-cnt",
							Image:   busyboxImage,
							Command: []string{"/bin/sleep", "1d"},
						},
					},
				},
			}

			refPod = e2epod.NewPodClient(f).CreateSync(ctx, refPod)
			cnt := &refPod.Spec.Containers[0] // shortcut

			// discover node settings.
			// we don't expect this data to change without node reboots, so we assume it's stable during the whole e2e run

			stdout, _, err := execCommandInContainer(f, refPod.Name, cnt.Name, "/bin/sh", "-c", cgroupGetFSTypeCmd)
			framework.ExpectNoError(err)
			cgroupFSType = strings.TrimSpace(stdout)

			stdout, _, err = execCommandInContainer(f, refPod.Name, cnt.Name, "/bin/cat", "/proc/self/mounts")
			framework.ExpectNoError(err)
			mountInfo = strings.TrimSpace(stdout)

			cgroupVersion, err = cgroupVersionFromHostData(cgroupFSType, mountInfo)
			framework.ExpectNoError(err)

			cgroupCpusetPath, err = cgroupCpusetPathFromCgroupVersion(cgroupVersion)
			framework.ExpectNoError(err)

			framework.Logf("detected cgroup version %q path %q for node %q", cgroupVersion, cgroupCpusetPath, refPod.Spec.NodeName)
		})

		ginkgo.AfterAll(func(ctx context.Context) {
			updateKubeletConfig(ctx, f, oldCfg, true)

			if refPod != nil {
				deletePodSyncByName(ctx, f, refPod.Name)
			}
		})

		// TODO: check sidecar containers
		ginkgo.Context("run the CPUManager tests", func() {
			var podMap map[string]*v1.Pod
			var allocatableCPU int64

			ginkgo.BeforeEach(func(ctx context.Context) {
				// track all pods created by a It() block
				podMap = make(map[string]*v1.Pod)

				_, allocatableCPU, _ = getLocalNodeCPUDetails(ctx, f)
				if allocatableCPU < 4 {
					e2eskipper.Skipf("Skipping CPU Manager tests since the CPU allocatable < 4")
				}
			})

			ginkgo.AfterEach(func(ctx context.Context) {
				deletePodsAsync(ctx, f, podMap)
			})

			ginkgo.It("should allocate exclusive CPUs for each container", func(ctx context.Context) {
				reservedCPUs := cpuset.New(0)

				updateKubeletConfigIfNeeded(ctx, f, configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
					policyName:         string(cpumanager.PolicyStatic),
					reservedSystemCPUs: reservedCPUs,
				}))

				cpuCount := 1
				ctnAttrs := []ctnAttribute{
					{
						ctnName:    "gu-container-1cpu-1",
						cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
						cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
						command:    "/bin/sleep 1d", // 1d is functionally forever for this test
					},
					{
						ctnName:    "gu-container-1cpu-2",
						cpuRequest: fmt.Sprintf("%dm", 1000*cpuCount),
						cpuLimit:   fmt.Sprintf("%dm", 1000*cpuCount),
						command:    "/bin/sleep 1d", // 1d is functionally forever for this test
					},
				}
				pod := makeCPUManagerPod("gu-pod-multicnt-1", ctnAttrs)
				pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
				podMap[pod.Name] = pod

				ginkgo.By("checking if the expected cpuset was assigned")

				gomega.Eventually(ctx, func(ctx context.Context) error {
					for _, cnt := range pod.Spec.Containers {
						ginkgo.By(fmt.Sprintf("validating the container %s on Gu pod %s", cnt.Name, pod.Name))

						stdout, _, err := execCommandInContainer(f, pod.Name, cnt.Name, "/bin/cat", cgroupCpusetPath)
						if err != nil {
							return fmt.Errorf("failed to get command output from container [%s] of pod [%s]: %w", cnt.Name, pod.Name, err)
						}

						cpus, err := cpuset.Parse(strings.TrimSpace(stdout))
						if err != nil {
							return fmt.Errorf("parsing cpuset from logs for [%s] of pod [%s]: %w", cnt.Name, pod.Name, err)
						}

						if cpus.Size() != cpuCount {
							return fmt.Errorf("expected cpu set size == %d, got %q", cpuCount, cpus.String())
						}
					}
					return nil
				}).WithPolling(5 * time.Second).WithTimeout(30 * time.Second).ShouldNot(gomega.HaveOccurred())

				ginkgo.By("checking if the shared pool changed as expected")
				// TODO: check the actual set composition in addition to the size?

				gomega.Eventually(ctx, func(ctx context.Context) error {
					node, ok, err := fetchLocalTestNode(ctx, f)
					if err != nil {
						return err
					}
					if !ok {
						return fmt.Errorf("local node not ready yet")
					}
					allocatableCPUQty := node.Status.Allocatable[v1.ResourceCPU]
					allocatableCPU := allocatableCPUQty.Value()
					framework.Logf("node allocatable cpus: %v", allocatableCPU)

					cnt := &refPod.Spec.Containers[0] // shortcut
					stdout, _, err := execCommandInContainer(f, refPod.Name, cnt.Name, "/bin/cat", cgroupCpusetPath)
					if err != nil {
						return fmt.Errorf("failed to get command output from container [%s] of pod [%s]: %w", cnt.Name, refPod.Name, err)
					}

					cpus, err := cpuset.Parse(strings.TrimSpace(stdout))
					if err != nil {
						return err
					}
					framework.Logf("BE container shared cpus: %v [%v]", cpus, stdout)

					totalCPU, err := computeTotalCPULimit(ctnAttrs)
					framework.ExpectNoError(err)

					sharedCPUCount := allocatableCPU - totalCPU
					framework.Logf("computed shared cpus count: %v", sharedCPUCount)

					// as in kube 1.28.1, BE pods can run on reserved cpus, only GU pods cannot.
					detectedSharedCPUCount := int64(cpus.Size() - reservedCPUs.Size())

					if sharedCPUCount != detectedSharedCPUCount {
						return fmt.Errorf("inconsistent accounting, expected %d shared cpus, found %d", sharedCPUCount, detectedSharedCPUCount)
					}
					return nil
				}).WithPolling(5 * time.Second).WithTimeout(30 * time.Second).ShouldNot(gomega.HaveOccurred())
			})
		})
	})
})

func computeTotalCPULimit(ctnAttrs []ctnAttribute) (int64, error) {
	tot, err := resource.ParseQuantity("0")
	if err != nil {
		return 0, err
	}
	for _, ctnAttr := range ctnAttrs {
		val, err := resource.ParseQuantity(ctnAttr.cpuLimit)
		if err != nil {
			return 0, err
		}
		tot.Add(val)
	}
	return tot.Value(), nil
}

func execCommandInContainer(f *framework.Framework, podName, containerName string, cmd ...string) (string, string, error) {
	stdout, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, podName, containerName, cmd...)
	framework.Logf("pod %s container %s stdout=[%s]", podName, containerName, stdout)
	framework.Logf("pod %s container %s stderr=[%s]", podName, containerName, stderr)
	return stdout, stderr, err
}
