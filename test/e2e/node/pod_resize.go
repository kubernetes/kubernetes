/*
Copyright 2021 The Kubernetes Authors.

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

package node

import (
	"context"
	"fmt"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	resourceapi "k8s.io/kubernetes/pkg/api/v1/resource"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"

	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	imageutils "k8s.io/kubernetes/test/utils/image"

	semver "github.com/blang/semver/v4"
	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	CgroupCPUPeriod    string = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	CgroupCPUShares    string = "/sys/fs/cgroup/cpu/cpu.shares"
	CgroupCPUQuota     string = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	CgroupMemLimit     string = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	Cgroupv2MemLimit   string = "/sys/fs/cgroup/memory.max"
	Cgroupv2MemRequest string = "/sys/fs/cgroup/memory.min"
	Cgroupv2CPULimit   string = "/sys/fs/cgroup/cpu.max"
	Cgroupv2CPURequest string = "/sys/fs/cgroup/cpu.weight"
	CpuPeriod          string = "100000"

	PollInterval time.Duration = 2 * time.Second
	PollTimeout  time.Duration = 4 * time.Minute
)

type ContainerResources struct {
	CPUReq, CPULim, MemReq, MemLim, EphStorReq, EphStorLim string
}

type ContainerAllocations struct {
	CPUAlloc, MemAlloc, ephStorAlloc string
}

type TestContainerInfo struct {
	Name         string
	Resources    *ContainerResources
	Allocations  *ContainerAllocations
	CPUPolicy    *v1.ResourceResizeRestartPolicy
	MemPolicy    *v1.ResourceResizeRestartPolicy
	RestartCount int32
}

func isInPlaceResizeSupportedByRuntime(c clientset.Interface, nodeName string) bool {
	//TODO(vinaykul,InPlacePodVerticalScaling): Can we optimize this?
	node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		return false
	}
	re := regexp.MustCompile("containerd://(.*)")
	match := re.FindStringSubmatch(node.Status.NodeInfo.ContainerRuntimeVersion)
	if len(match) != 2 {
		return false
	}
	if ver, verr := semver.ParseTolerant(match[1]); verr == nil {
		if ver.Compare(semver.MustParse("1.6.9")) < 0 {
			return false
		}
		return true
	}
	return false
}

func getTestResourceInfo(tcInfo TestContainerInfo) (v1.ResourceRequirements, v1.ResourceList, []v1.ContainerResizePolicy) {
	var res v1.ResourceRequirements
	var alloc v1.ResourceList
	var resizePol []v1.ContainerResizePolicy

	if tcInfo.Resources != nil {
		var lim, req v1.ResourceList
		if tcInfo.Resources.CPULim != "" || tcInfo.Resources.MemLim != "" || tcInfo.Resources.EphStorLim != "" {
			lim = make(v1.ResourceList)
		}
		if tcInfo.Resources.CPUReq != "" || tcInfo.Resources.MemReq != "" || tcInfo.Resources.EphStorReq != "" {
			req = make(v1.ResourceList)
		}
		if tcInfo.Resources.CPULim != "" {
			lim[v1.ResourceCPU] = resource.MustParse(tcInfo.Resources.CPULim)
		}
		if tcInfo.Resources.MemLim != "" {
			lim[v1.ResourceMemory] = resource.MustParse(tcInfo.Resources.MemLim)
		}
		if tcInfo.Resources.EphStorLim != "" {
			lim[v1.ResourceEphemeralStorage] = resource.MustParse(tcInfo.Resources.EphStorLim)
		}
		if tcInfo.Resources.CPUReq != "" {
			req[v1.ResourceCPU] = resource.MustParse(tcInfo.Resources.CPUReq)
		}
		if tcInfo.Resources.MemReq != "" {
			req[v1.ResourceMemory] = resource.MustParse(tcInfo.Resources.MemReq)
		}
		if tcInfo.Resources.EphStorReq != "" {
			req[v1.ResourceEphemeralStorage] = resource.MustParse(tcInfo.Resources.EphStorReq)
		}
		res = v1.ResourceRequirements{Limits: lim, Requests: req}
	}
	if tcInfo.Allocations != nil {
		alloc = make(v1.ResourceList)
		if tcInfo.Allocations.CPUAlloc != "" {
			alloc[v1.ResourceCPU] = resource.MustParse(tcInfo.Allocations.CPUAlloc)
		}
		if tcInfo.Allocations.MemAlloc != "" {
			alloc[v1.ResourceMemory] = resource.MustParse(tcInfo.Allocations.MemAlloc)
		}
		if tcInfo.Allocations.ephStorAlloc != "" {
			alloc[v1.ResourceEphemeralStorage] = resource.MustParse(tcInfo.Allocations.ephStorAlloc)
		}

	}
	if tcInfo.CPUPolicy != nil {
		cpuPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceCPU, RestartPolicy: *tcInfo.CPUPolicy}
		resizePol = append(resizePol, cpuPol)
	}
	if tcInfo.MemPolicy != nil {
		memPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: *tcInfo.MemPolicy}
		resizePol = append(resizePol, memPol)
	}
	return res, alloc, resizePol
}

func initDefaultResizePolicy(containers []TestContainerInfo) {
	noRestart := v1.NotRequired
	setDefaultPolicy := func(ci *TestContainerInfo) {
		if ci.CPUPolicy == nil {
			ci.CPUPolicy = &noRestart
		}
		if ci.MemPolicy == nil {
			ci.MemPolicy = &noRestart
		}
	}
	for i := range containers {
		setDefaultPolicy(&containers[i])
	}
}

func makeTestContainer(tcInfo TestContainerInfo) (v1.Container, v1.ContainerStatus) {
	res, alloc, resizePol := getTestResourceInfo(tcInfo)
	bTrue := true
	bFalse := false
	userID := int64(1001)
	userName := "ContainerUser"

	var securityContext *v1.SecurityContext

	if framework.NodeOSDistroIs("windows") {
		securityContext = &v1.SecurityContext{
			RunAsNonRoot: &bTrue,
			WindowsOptions: &v1.WindowsSecurityContextOptions{
				RunAsUserName: &userName,
			},
		}
	} else {
		securityContext = &v1.SecurityContext{
			Privileged:               &bFalse,
			AllowPrivilegeEscalation: &bFalse,
			RunAsUser:                &userID,
			RunAsNonRoot:             &bTrue,
			Capabilities: &v1.Capabilities{
				Drop: []v1.Capability{"ALL"},
			},
			SeccompProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeRuntimeDefault,
			},
		}
	}

	tc := v1.Container{
		Name:            tcInfo.Name,
		Image:           imageutils.GetE2EImage(imageutils.BusyBox),
		Command:         []string{"/bin/sh"},
		Args:            []string{"-c", e2epod.InfiniteSleepCommand},
		Resources:       res,
		ResizePolicy:    resizePol,
		SecurityContext: securityContext,
	}

	tcStatus := v1.ContainerStatus{
		Name:               tcInfo.Name,
		AllocatedResources: alloc,
	}
	return tc, tcStatus
}

func makeTestPod(ns, name, timeStamp string, tcInfo []TestContainerInfo) *v1.Pod {
	var testContainers []v1.Container
	var podOS *v1.PodOS

	for _, ci := range tcInfo {
		tc, _ := makeTestContainer(ci)
		testContainers = append(testContainers, tc)
	}

	if framework.NodeOSDistroIs("windows") {
		podOS = &v1.PodOS{Name: v1.OSName("windows")}
	} else {
		podOS = &v1.PodOS{Name: v1.OSName(runtime.GOOS)}
	}

	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels: map[string]string{
				"name": "fooPod",
				"time": timeStamp,
			},
		},
		Spec: v1.PodSpec{
			OS:            podOS,
			Containers:    testContainers,
			RestartPolicy: v1.RestartPolicyOnFailure,
		},
	}
	return pod
}

func verifyPodResizePolicy(pod *v1.Pod, tcInfo []TestContainerInfo) {
	cMap := make(map[string]*v1.Container)
	for i, c := range pod.Spec.Containers {
		cMap[c.Name] = &pod.Spec.Containers[i]
	}
	for _, ci := range tcInfo {
		gomega.Expect(cMap).Should(gomega.HaveKey(ci.Name))
		c := cMap[ci.Name]
		tc, _ := makeTestContainer(ci)
		gomega.Expect(tc.ResizePolicy).To(gomega.Equal(c.ResizePolicy))
	}
}

func verifyPodResources(pod *v1.Pod, tcInfo []TestContainerInfo) {
	cMap := make(map[string]*v1.Container)
	for i, c := range pod.Spec.Containers {
		cMap[c.Name] = &pod.Spec.Containers[i]
	}
	for _, ci := range tcInfo {
		gomega.Expect(cMap).Should(gomega.HaveKey(ci.Name))
		c := cMap[ci.Name]
		tc, _ := makeTestContainer(ci)
		gomega.Expect(tc.Resources).To(gomega.Equal(c.Resources))
	}
}

func verifyPodAllocations(pod *v1.Pod, tcInfo []TestContainerInfo, flagError bool) bool {
	cStatusMap := make(map[string]*v1.ContainerStatus)
	for i, c := range pod.Status.ContainerStatuses {
		cStatusMap[c.Name] = &pod.Status.ContainerStatuses[i]
	}

	for _, ci := range tcInfo {
		gomega.Expect(cStatusMap).Should(gomega.HaveKey(ci.Name))
		cStatus := cStatusMap[ci.Name]
		if ci.Allocations == nil {
			if ci.Resources != nil {
				alloc := &ContainerAllocations{CPUAlloc: ci.Resources.CPUReq, MemAlloc: ci.Resources.MemReq}
				ci.Allocations = alloc
				defer func() {
					ci.Allocations = nil
				}()
			}
		}

		_, tcStatus := makeTestContainer(ci)
		if flagError {
			gomega.Expect(tcStatus.AllocatedResources).To(gomega.Equal(cStatus.AllocatedResources))
		}
		if !cmp.Equal(cStatus.AllocatedResources, tcStatus.AllocatedResources) {
			return false
		}
	}
	return true
}

func verifyPodStatusResources(pod *v1.Pod, tcInfo []TestContainerInfo) {
	csMap := make(map[string]*v1.ContainerStatus)
	for i, c := range pod.Status.ContainerStatuses {
		csMap[c.Name] = &pod.Status.ContainerStatuses[i]
	}
	for _, ci := range tcInfo {
		gomega.Expect(csMap).Should(gomega.HaveKey(ci.Name))
		cs := csMap[ci.Name]
		tc, _ := makeTestContainer(ci)
		gomega.Expect(tc.Resources).To(gomega.Equal(*cs.Resources))
		//gomega.Expect(cs.RestartCount).To(gomega.Equal(ci.RestartCount))
	}
}

func isPodOnCgroupv2Node(pod *v1.Pod) bool {
	// Determine if pod is running on cgroupv2 or cgroupv1 node
	//TODO(vinaykul,InPlacePodVerticalScaling): Is there a better way to determine this?
	cgroupv2File := "/sys/fs/cgroup/cgroup.controllers"
	_, err := e2ekubectl.RunKubectl(pod.Namespace, "exec", pod.Name, "--", "ls", cgroupv2File)
	if err == nil {
		return true
	}
	return false
}

func verifyPodContainersCgroupValues(pod *v1.Pod, tcInfo []TestContainerInfo, flagError bool) bool {
	podOnCgroupv2Node := isPodOnCgroupv2Node(pod)
	cgroupMemLimit := Cgroupv2MemLimit
	cgroupCPULimit := Cgroupv2CPULimit
	cgroupCPURequest := Cgroupv2CPURequest
	if !podOnCgroupv2Node {
		cgroupMemLimit = CgroupMemLimit
		cgroupCPULimit = CgroupCPUQuota
		cgroupCPURequest = CgroupCPUShares
	}
	verifyCgroupValue := func(cName, cgPath, expectedCgValue string) bool {
		cmd := []string{"head", "-n", "1", cgPath}
		framework.Logf("Namespace %s Pod %s Container %s - looking for cgroup value %s in path %s",
			pod.Namespace, pod.Name, cName, expectedCgValue, cgPath)
		cgValue, err := e2epodoutput.LookForStringInPodExecToContainer(pod.Namespace, pod.Name, cName, cmd, expectedCgValue, PollTimeout)
		if flagError {
			framework.ExpectNoError(err, fmt.Sprintf("failed to find expected value '%s' in container cgroup '%s'",
				expectedCgValue, cgPath))
		}
		cgValue = strings.Trim(cgValue, "\n")
		if flagError {
			gomega.Expect(cgValue).Should(gomega.Equal(expectedCgValue), "cgroup value")
		}
		if cgValue != expectedCgValue {
			return false
		}
		return true
	}
	for _, ci := range tcInfo {
		if ci.Resources == nil {
			continue
		}
		tc, _ := makeTestContainer(ci)
		if tc.Resources.Limits != nil || tc.Resources.Requests != nil {
			var cpuShares int64
			var cpuLimitString, memLimitString string
			memLimitInBytes := tc.Resources.Limits.Memory().Value()
			cpuRequest := tc.Resources.Requests.Cpu()
			cpuLimit := tc.Resources.Limits.Cpu()
			if cpuRequest.IsZero() && !cpuLimit.IsZero() {
				cpuShares = int64(kubecm.MilliCPUToShares(cpuLimit.MilliValue()))
			} else {
				cpuShares = int64(kubecm.MilliCPUToShares(cpuRequest.MilliValue()))
			}
			cpuQuota := kubecm.MilliCPUToQuota(cpuLimit.MilliValue(), kubecm.QuotaPeriod)
			if cpuLimit.IsZero() {
				cpuQuota = -1
			}
			cpuLimitString = strconv.FormatInt(cpuQuota, 10)
			if podOnCgroupv2Node {
				if cpuLimitString == "-1" {
					cpuLimitString = "max"
				}
				cpuLimitString = fmt.Sprintf("%s %s", cpuLimitString, CpuPeriod)
			}
			memLimitString = strconv.FormatInt(memLimitInBytes, 10)
			if podOnCgroupv2Node && memLimitString == "0" {
				memLimitString = "max"
			}
			if memLimitString != "0" {
				if !verifyCgroupValue(ci.Name, cgroupMemLimit, memLimitString) {
					return false
				}
			}
			if !verifyCgroupValue(ci.Name, cgroupCPULimit, cpuLimitString) {
				return false
			}
			if podOnCgroupv2Node {
				// convert cgroup v1 cpu.shares value to cgroup v2 cpu.weight value
				cpuShares = int64(1 + ((cpuShares-2)*9999)/262142)
			}
			if !verifyCgroupValue(ci.Name, cgroupCPURequest, strconv.FormatInt(cpuShares, 10)) {
				return false
			}
		}
	}
	return true
}

func waitForPodResizeActuation(c clientset.Interface, podClient *e2epod.PodClient, pod, patchedPod *v1.Pod, expectedContainers []TestContainerInfo) *v1.Pod {

	waitForContainerRestart := func() error {
		var restartContainersExpected []string
		for _, ci := range expectedContainers {
			if ci.RestartCount > 0 {
				restartContainersExpected = append(restartContainersExpected, ci.Name)
			}
		}
		if len(restartContainersExpected) == 0 {
			return nil
		}
		for start := time.Now(); time.Since(start) < PollTimeout; time.Sleep(PollInterval) {
			pod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			restartedContainersCount := 0
			for _, cName := range restartContainersExpected {
				cs, _ := podutil.GetContainerStatus(pod.Status.ContainerStatuses, cName)
				if cs.RestartCount < 1 {
					break
				}
				restartedContainersCount++
			}
			if restartedContainersCount == len(restartContainersExpected) {
				return nil
			}
		}
		return fmt.Errorf("timed out waiting for expected container restart")
	}
	waitPodAllocationsEqualsExpected := func() (*v1.Pod, error) {
		for start := time.Now(); time.Since(start) < PollTimeout; time.Sleep(PollInterval) {
			pod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			if !verifyPodAllocations(pod, expectedContainers, false) {
				continue
			}
			return pod, nil
		}
		return nil, fmt.Errorf("timed out waiting for pod resource allocation values to match expected")
	}
	waitContainerCgroupValuesEqualsExpected := func() error {
		for start := time.Now(); time.Since(start) < PollTimeout; time.Sleep(PollInterval) {
			if !verifyPodContainersCgroupValues(patchedPod, expectedContainers, false) {
				continue
			}
			return nil
		}
		return fmt.Errorf("timed out waiting for container cgroup values to match expected")
	}
	waitPodStatusResourcesEqualSpecResources := func() (*v1.Pod, error) {
		for start := time.Now(); time.Since(start) < PollTimeout; time.Sleep(PollInterval) {
			pod, err := podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			if err != nil {
				return nil, err
			}
			differs := false
			for idx, c := range pod.Spec.Containers {
				if !cmp.Equal(c.Resources, *pod.Status.ContainerStatuses[idx].Resources) {
					differs = true
					break
				}
			}
			if differs {
				continue
			}
			return pod, nil
		}
		return nil, fmt.Errorf("timed out waiting for pod spec resources to match pod status resources")
	}
	rsErr := waitForContainerRestart()
	framework.ExpectNoError(rsErr, "failed waiting for expected container restart")
	// Wait for pod resource allocations to equal expected values after resize
	resizedPod, aErr := waitPodAllocationsEqualsExpected()
	framework.ExpectNoError(aErr, "failed to verify pod resource allocation values equals expected values")
	//TODO(vinaykul,InPlacePodVerticalScaling): Remove this check once base-OS updates to containerd>=1.6.9
	//                containerd needs to add CRI support before Beta (See Node KEP #2273)
	if !isInPlaceResizeSupportedByRuntime(c, pod.Spec.NodeName) {
		// Wait for PodSpec container resources to equal PodStatus container resources indicating resize is complete
		rPod, rErr := waitPodStatusResourcesEqualSpecResources()
		framework.ExpectNoError(rErr, "failed to verify pod spec resources equals pod status resources")

		ginkgo.By("verifying pod status after resize")
		verifyPodStatusResources(rPod, expectedContainers)
	} else if !framework.NodeOSDistroIs("windows") {
		// Wait for container cgroup values to equal expected cgroup values after resize
		// only for containerd versions before 1.6.9
		cErr := waitContainerCgroupValuesEqualsExpected()
		framework.ExpectNoError(cErr, "failed to verify container cgroup values equals expected values")
	}
	return resizedPod
}

func doPodResizeTests() {
	f := framework.NewDefaultFramework("pod-resize")
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	type testCase struct {
		name        string
		containers  []TestContainerInfo
		patchString string
		expected    []TestContainerInfo
	}

	noRestart := v1.NotRequired
	doRestart := v1.RestartContainer
	tests := []testCase{
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & memory",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "500Mi", MemLim: "500Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m","memory":"250Mi"},"limits":{"cpu":"100m","memory":"250Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & decrease memory",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"100Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & increase memory",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"300Mi"},"limits":{"cpu":"50m","memory":"300Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "50m", CPULim: "50m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, three containers (c1, c2, c3) - increase: CPU (c1,c3), memory (c2) ; decrease: CPU (c2), memory (c1,c3)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"140m","memory":"50Mi"},"limits":{"cpu":"140m","memory":"50Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"150m","memory":"240Mi"},"limits":{"cpu":"150m","memory":"240Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"340m","memory":"250Mi"},"limits":{"cpu":"340m","memory":"250Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "140m", CPULim: "140m", MemReq: "50Mi", MemLim: "50Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "150m", CPULim: "150m", MemReq: "240Mi", MemLim: "240Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "340m", CPULim: "340m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"200Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory limits only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory limits only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"600Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "600Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU limits only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "150m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU limits only",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"200m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"400m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase CPU limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease CPU limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase memory limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease memory limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase memory limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease memory limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase CPU limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "300m", MemReq: "100Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease CPU limits",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "300Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - decrease memory request",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", MemReq: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "200m", MemReq: "400Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU (NotRequired) & memory (RestartContainer)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:         "c1",
					Resources:    &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container - decrease CPU (RestartContainer) & memory (NotRequired)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"100Mi"},"limits":{"cpu":"100m","memory":"200Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:         "c1",
					Resources:    &ContainerResources{CPUReq: "50m", CPULim: "100m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - increase c1 resources, no change for c2, decrease c3 resources (no net change for pod)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m","memory":"150Mi"},"limits":{"cpu":"250m","memory":"250Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "150m", CPULim: "250m", MemReq: "150Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - decrease c1 resources, increase c2 resources, no change for c3 (net increase for pod)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"50Mi"},"limits":{"cpu":"150m","memory":"150Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"350m","memory":"350Mi"},"limits":{"cpu":"450m","memory":"450Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "50m", CPULim: "150m", MemReq: "50Mi", MemLim: "150Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:         "c2",
					Resources:    &ContainerResources{CPUReq: "350m", CPULim: "450m", MemReq: "350Mi", MemLim: "450Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - no change for c1, increase c2 resources, decrease c3 (net decrease for pod)",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c2", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"100m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"200Mi"}}}
					]}}`,
			expected: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:         "c2",
					Resources:    &ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
				{
					Name:         "c3",
					Resources:    &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
	}

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := podClient.List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pods")
			gomega.Expect(podList.Items).Should(gomega.HaveLen(1))

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)
			// Check cgroup values only for containerd versions before 1.6.9
			if !isInPlaceResizeSupportedByRuntime(f.ClientSet, newPod.Spec.NodeName) {
				if !framework.NodeOSDistroIs("windows") {
					verifyPodContainersCgroupValues(newPod, tc.containers, true)
				}
			}

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(context.TODO(), newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			framework.ExpectNoError(pErr, "failed to patch pod for resize")

			ginkgo.By("verifying pod patched for resize")
			verifyPodResources(patchedPod, tc.expected)
			verifyPodAllocations(patchedPod, tc.containers, true)

			ginkgo.By("waiting for resize to be actuated")
			resizedPod := waitForPodResizeActuation(f.ClientSet, podClient, newPod, patchedPod, tc.expected)

			// Check cgroup values only for containerd versions before 1.6.9
			if !isInPlaceResizeSupportedByRuntime(f.ClientSet, newPod.Spec.NodeName) {
				ginkgo.By("verifying pod container's cgroup values after resize")
				if !framework.NodeOSDistroIs("windows") {
					verifyPodContainersCgroupValues(resizedPod, tc.expected, true)
				}
			}

			ginkgo.By("verifying pod resources after resize")
			verifyPodResources(resizedPod, tc.expected)

			ginkgo.By("verifying pod allocations after resize")
			verifyPodAllocations(resizedPod, tc.expected, true)

			ginkgo.By("deleting pod")
			err = e2epod.DeletePodWithWait(ctx, f.ClientSet, newPod)
			framework.ExpectNoError(err, "failed to delete pod")
		})
	}
}

func doPodResizeResourceQuotaTests() {
	f := framework.NewDefaultFramework("pod-resize-resource-quota")
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("pod-resize-resource-quota-test", func(ctx context.Context) {
		resourceQuota := v1.ResourceQuota{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "resize-resource-quota",
				Namespace: f.Namespace.Name,
			},
			Spec: v1.ResourceQuotaSpec{
				Hard: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("800m"),
					v1.ResourceMemory: resource.MustParse("800Mi"),
				},
			},
		}
		containers := []TestContainerInfo{
			{
				Name:      "c1",
				Resources: &ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
			},
		}
		patchString := `{"spec":{"containers":[
			{"name":"c1", "resources":{"requests":{"cpu":"400m","memory":"400Mi"},"limits":{"cpu":"400m","memory":"400Mi"}}}
		]}}`
		expected := []TestContainerInfo{
			{
				Name:      "c1",
				Resources: &ContainerResources{CPUReq: "400m", CPULim: "400m", MemReq: "400Mi", MemLim: "400Mi"},
			},
		}
		patchStringExceedCPU := `{"spec":{"containers":[
			{"name":"c1", "resources":{"requests":{"cpu":"600m","memory":"200Mi"},"limits":{"cpu":"600m","memory":"200Mi"}}}
		]}}`
		patchStringExceedMemory := `{"spec":{"containers":[
			{"name":"c1", "resources":{"requests":{"cpu":"250m","memory":"750Mi"},"limits":{"cpu":"250m","memory":"750Mi"}}}
		]}}`

		ginkgo.By("Creating a ResourceQuota")
		_, rqErr := f.ClientSet.CoreV1().ResourceQuotas(f.Namespace.Name).Create(context.TODO(), &resourceQuota, metav1.CreateOptions{})
		framework.ExpectNoError(rqErr, "failed to create resource quota")

		tStamp := strconv.Itoa(time.Now().Nanosecond())
		initDefaultResizePolicy(containers)
		initDefaultResizePolicy(expected)
		testPod1 := makeTestPod(f.Namespace.Name, "testpod1", tStamp, containers)
		testPod2 := makeTestPod(f.Namespace.Name, "testpod2", tStamp, containers)

		ginkgo.By("creating pods")
		newPod1 := podClient.CreateSync(ctx, testPod1)
		newPod2 := podClient.CreateSync(ctx, testPod2)

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		podList, listErr := podClient.List(context.TODO(), options)
		framework.ExpectNoError(listErr, "failed to query for pods")
		gomega.Expect(podList.Items).Should(gomega.HaveLen(2))

		ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
		verifyPodResources(newPod1, containers)

		ginkgo.By("patching pod for resize within resource quota")
		patchedPod, pErr := f.ClientSet.CoreV1().Pods(newPod1.Namespace).Patch(context.TODO(), newPod1.Name,
			types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{})
		framework.ExpectNoError(pErr, "failed to patch pod for resize")

		ginkgo.By("verifying pod patched for resize within resource quota")
		verifyPodResources(patchedPod, expected)
		verifyPodAllocations(patchedPod, containers, true)

		ginkgo.By("waiting for resize to be actuated")
		resizedPod := waitForPodResizeActuation(f.ClientSet, podClient, newPod1, patchedPod, expected)
		if !isInPlaceResizeSupportedByRuntime(f.ClientSet, newPod1.Spec.NodeName) {
			ginkgo.By("verifying pod container's cgroup values after resize")
			if !framework.NodeOSDistroIs("windows") {
				verifyPodContainersCgroupValues(resizedPod, expected, true)
			}
		}

		ginkgo.By("verifying pod resources after resize")
		verifyPodResources(resizedPod, expected)

		ginkgo.By("verifying pod allocations after resize")
		verifyPodAllocations(resizedPod, expected, true)

		ginkgo.By("patching pod for resize with memory exceeding resource quota")
		_, pErrExceedMemory := f.ClientSet.CoreV1().Pods(resizedPod.Namespace).Patch(context.TODO(),
			resizedPod.Name, types.StrategicMergePatchType, []byte(patchStringExceedMemory), metav1.PatchOptions{})
		gomega.Expect(pErrExceedMemory).To(gomega.HaveOccurred(), "exceeded quota: %s, requested: memory=350Mi, used: memory=700Mi, limited: memory=800Mi",
			resourceQuota.Name)

		ginkgo.By("verifying pod patched for resize exceeding memory resource quota remains unchanged")
		patchedPodExceedMemory, pErrEx2 := podClient.Get(context.TODO(), resizedPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(pErrEx2, "failed to get pod post exceed memory resize")
		verifyPodResources(patchedPodExceedMemory, expected)
		verifyPodAllocations(patchedPodExceedMemory, expected, true)

		ginkgo.By(fmt.Sprintf("patching pod %s for resize with CPU exceeding resource quota", resizedPod.Name))
		_, pErrExceedCPU := f.ClientSet.CoreV1().Pods(resizedPod.Namespace).Patch(context.TODO(),
			resizedPod.Name, types.StrategicMergePatchType, []byte(patchStringExceedCPU), metav1.PatchOptions{})
		gomega.Expect(pErrExceedCPU).To(gomega.HaveOccurred(), "exceeded quota: %s, requested: cpu=200m, used: cpu=700m, limited: cpu=800m",
			resourceQuota.Name)

		ginkgo.By("verifying pod patched for resize exceeding CPU resource quota remains unchanged")
		patchedPodExceedCPU, pErrEx1 := podClient.Get(context.TODO(), resizedPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(pErrEx1, "failed to get pod post exceed CPU resize")
		verifyPodResources(patchedPodExceedCPU, expected)
		verifyPodAllocations(patchedPodExceedCPU, expected, true)

		ginkgo.By("deleting pods")
		delErr1 := e2epod.DeletePodWithWait(ctx, f.ClientSet, newPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", newPod1.Name)
		delErr2 := e2epod.DeletePodWithWait(ctx, f.ClientSet, newPod2)
		framework.ExpectNoError(delErr2, "failed to delete pod %s", newPod2.Name)
	})
}

func doPodResizeErrorTests() {
	f := framework.NewDefaultFramework("pod-resize-errors")
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	type testCase struct {
		name        string
		containers  []TestContainerInfo
		patchString string
		patchError  string
		expected    []TestContainerInfo
	}

	tests := []testCase{
		{
			name: "BestEffort pod - try requesting memory, expect error",
			containers: []TestContainerInfo{
				{
					Name: "c1",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			patchError: "Pod QoS is immutable",
			expected: []TestContainerInfo{
				{
					Name: "c1",
				},
			},
		},
	}

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := podClient.List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pods")
			gomega.Expect(podList.Items).Should(gomega.HaveLen(1))

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)
			if !isInPlaceResizeSupportedByRuntime(f.ClientSet, newPod.Spec.NodeName) {
				if !framework.NodeOSDistroIs("windows") {
					verifyPodContainersCgroupValues(newPod, tc.containers, true)
				}
			}

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(context.TODO(), newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				gomega.Expect(pErr).To(gomega.HaveOccurred(), tc.patchError)
				patchedPod = newPod
			}

			if !isInPlaceResizeSupportedByRuntime(f.ClientSet, patchedPod.Spec.NodeName) {
				ginkgo.By("verifying pod container's cgroup values after patch")
				if !framework.NodeOSDistroIs("windows") {
					verifyPodContainersCgroupValues(patchedPod, tc.expected, true)
				}
			}

			ginkgo.By("verifying pod resources after patch")
			verifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod allocations after patch")
			verifyPodAllocations(patchedPod, tc.expected, true)

			ginkgo.By("deleting pod")
			err = e2epod.DeletePodWithWait(ctx, f.ClientSet, newPod)
			framework.ExpectNoError(err, "failed to delete pod")
		})
	}
}

func doPodResizeSchedulerTests() {
	f := framework.NewDefaultFramework("pod-resize-scheduler")
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	ginkgo.It("pod-resize-scheduler-tests", func(ctx context.Context) {
		nodes, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "failed to get running nodes")
		gomega.Expect(nodes.Items).ShouldNot(gomega.BeEmpty())
		framework.Logf("Found %d schedulable nodes", len(nodes.Items))

		//
		// Calculate available CPU. nodeAvailableCPU = nodeAllocatableCPU - sum(podAllocatedCPU)
		//
		getNodeAllocatableAndAvailableMilliCPUValues := func(n *v1.Node) (int64, int64) {
			nodeAllocatableMilliCPU := n.Status.Allocatable.Cpu().MilliValue()
			gomega.Expect(n.Status.Allocatable).ShouldNot(gomega.BeNil(), "allocatable")
			podAllocatedMilliCPU := int64(0)

			// Exclude pods that are in the Succeeded or Failed states
			selector := fmt.Sprintf("spec.nodeName=%s,status.phase!=%v,status.phase!=%v", n.Name, v1.PodSucceeded, v1.PodFailed)
			listOptions := metav1.ListOptions{FieldSelector: selector}
			podList, err := f.ClientSet.CoreV1().Pods(metav1.NamespaceAll).List(context.TODO(), listOptions)

			framework.ExpectNoError(err, "failed to get running pods")
			framework.Logf("Found %d pods on node '%s'", len(podList.Items), n.Name)
			for _, pod := range podList.Items {
				podRequestMilliCPU := resourceapi.GetResourceRequest(&pod, v1.ResourceCPU)
				podAllocatedMilliCPU += podRequestMilliCPU
			}
			nodeAvailableMilliCPU := nodeAllocatableMilliCPU - podAllocatedMilliCPU
			return nodeAllocatableMilliCPU, nodeAvailableMilliCPU
		}

		ginkgo.By("Find node CPU resources available for allocation!")
		node := nodes.Items[0]
		nodeAllocatableMilliCPU, nodeAvailableMilliCPU := getNodeAllocatableAndAvailableMilliCPUValues(&node)
		framework.Logf("Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU, nodeAvailableMilliCPU)

		//
		// Scheduler focussed pod resize E2E test case #1:
		//     1. Create pod1 and pod2 on node such that pod1 has enough CPU to be scheduled, but pod2 does not.
		//     2. Resize pod2 down so that it fits on the node and can be scheduled.
		//     3. Verify that pod2 gets scheduled and comes up and running.
		//
		testPod1CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU/2, resource.DecimalSI)
		testPod2CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU, resource.DecimalSI)
		testPod2CPUQuantityResized := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/2, resource.DecimalSI)
		framework.Logf("TEST1: testPod1 initial CPU request is '%dm'", testPod1CPUQuantity.MilliValue())
		framework.Logf("TEST1: testPod2 initial CPU request is '%dm'", testPod2CPUQuantity.MilliValue())
		framework.Logf("TEST1: testPod2 resized CPU request is '%dm'", testPod2CPUQuantityResized.MilliValue())

		c1 := []TestContainerInfo{
			{
				Name:      "c1",
				Resources: &ContainerResources{CPUReq: testPod1CPUQuantity.String(), CPULim: testPod1CPUQuantity.String()},
			},
		}
		c2 := []TestContainerInfo{
			{
				Name:      "c2",
				Resources: &ContainerResources{CPUReq: testPod2CPUQuantity.String(), CPULim: testPod2CPUQuantity.String()},
			},
		}
		patchTestpod2ToFitNode := fmt.Sprintf(`{
				"spec": {
					"containers": [
						{
							"name":      "c2",
							"resources": {"requests": {"cpu": "%dm"}, "limits": {"cpu": "%dm"}}
						}
					]
				}
			}`, testPod2CPUQuantityResized.MilliValue(), testPod2CPUQuantityResized.MilliValue())

		tStamp := strconv.Itoa(time.Now().Nanosecond())
		initDefaultResizePolicy(c1)
		initDefaultResizePolicy(c2)
		testPod1 := makeTestPod(f.Namespace.Name, "testpod1", tStamp, c1)
		testPod2 := makeTestPod(f.Namespace.Name, "testpod2", tStamp, c2)
		e2epod.SetNodeAffinity(&testPod1.Spec, node.Name)
		e2epod.SetNodeAffinity(&testPod2.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("TEST1: Create pod '%s' that fits the node '%s'", testPod1.Name, node.Name))
		testPod1 = podClient.CreateSync(ctx, testPod1)
		gomega.Expect(testPod1.Status.Phase).To(gomega.Equal(v1.PodRunning))

		ginkgo.By(fmt.Sprintf("TEST1: Create pod '%s' that won't fit node '%s' with pod '%s' on it", testPod2.Name, node.Name, testPod1.Name))
		testPod2 = podClient.Create(ctx, testPod2)
		err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, testPod2.Name, testPod2.Namespace)
		framework.ExpectNoError(err)
		gomega.Expect(testPod2.Status.Phase).To(gomega.Equal(v1.PodPending))

		ginkgo.By(fmt.Sprintf("TEST1: Resize pod '%s' to fit in node '%s'", testPod2.Name, node.Name))
		testPod2, pErr := f.ClientSet.CoreV1().Pods(testPod2.Namespace).Patch(ctx,
			testPod2.Name, types.StrategicMergePatchType, []byte(patchTestpod2ToFitNode), metav1.PatchOptions{})
		framework.ExpectNoError(pErr, "failed to patch pod for resize")

		ginkgo.By(fmt.Sprintf("TEST1: Verify that pod '%s' is running after resize", testPod2.Name))
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, testPod2))

		//
		// Scheduler focussed pod resize E2E test case #2
		//     1. With pod1 + pod2 running on node above, create pod3 that requests more CPU than available, verify pending.
		//     2. Resize pod1 down so that pod3 gets room to be scheduled.
		//     3. Verify that pod3 is scheduled and running.
		//
		nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2 := getNodeAllocatableAndAvailableMilliCPUValues(&node)
		framework.Logf("TEST2: Node '%s': NodeAllocatable MilliCPUs = %dm. MilliCPUs currently available to allocate = %dm.",
			node.Name, nodeAllocatableMilliCPU2, nodeAvailableMilliCPU2)
		testPod3CPUQuantity := resource.NewMilliQuantity(nodeAvailableMilliCPU2+testPod1CPUQuantity.MilliValue()/4, resource.DecimalSI)
		testPod1CPUQuantityResized := resource.NewMilliQuantity(testPod1CPUQuantity.MilliValue()/3, resource.DecimalSI)
		framework.Logf("TEST2: testPod1 MilliCPUs after resize '%dm'", testPod1CPUQuantityResized.MilliValue())

		c3 := []TestContainerInfo{
			{
				Name:      "c3",
				Resources: &ContainerResources{CPUReq: testPod3CPUQuantity.String(), CPULim: testPod3CPUQuantity.String()},
			},
		}
		patchTestpod1ToMakeSpaceForPod3 := fmt.Sprintf(`{
				"spec": {
					"containers": [
						{
							"name":      "c1",
							"resources": {"requests": {"cpu": "%dm"},"limits": {"cpu": "%dm"}}
						}
					]
				}
			}`, testPod1CPUQuantityResized.MilliValue(), testPod1CPUQuantityResized.MilliValue())

		tStamp = strconv.Itoa(time.Now().Nanosecond())
		initDefaultResizePolicy(c3)
		testPod3 := makeTestPod(f.Namespace.Name, "testpod3", tStamp, c3)
		e2epod.SetNodeAffinity(&testPod3.Spec, node.Name)

		ginkgo.By(fmt.Sprintf("TEST2: Create testPod3 '%s' that cannot fit node '%s' due to insufficient CPU.", testPod3.Name, node.Name))
		testPod3 = podClient.Create(ctx, testPod3)
		p3Err := e2epod.WaitForPodNameUnschedulableInNamespace(ctx, f.ClientSet, testPod3.Name, testPod3.Namespace)
		framework.ExpectNoError(p3Err, "failed to create pod3 or pod3 did not become pending!")
		gomega.Expect(testPod3.Status.Phase).To(gomega.Equal(v1.PodPending))

		ginkgo.By(fmt.Sprintf("TEST2: Resize pod '%s' to make enough space for pod '%s'", testPod1.Name, testPod3.Name))
		testPod1, p1Err := f.ClientSet.CoreV1().Pods(testPod1.Namespace).Patch(context.TODO(),
			testPod1.Name, types.StrategicMergePatchType, []byte(patchTestpod1ToMakeSpaceForPod3), metav1.PatchOptions{})
		framework.ExpectNoError(p1Err, "failed to patch pod for resize")

		ginkgo.By(fmt.Sprintf("TEST2: Verify pod '%s' is running after successfully resizing pod '%s'", testPod3.Name, testPod1.Name))
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod1.Name, testPod1.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod2.Name, testPod2.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.Logf("TEST2: Pod '%s' CPU requests '%dm'", testPod3.Name, testPod3.Spec.Containers[0].Resources.Requests.Cpu().MilliValue())
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, f.ClientSet, testPod3))

		ginkgo.By("deleting pods")
		delErr1 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", testPod1.Name)
		delErr2 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod2)
		framework.ExpectNoError(delErr2, "failed to delete pod %s", testPod2.Name)
		delErr3 := e2epod.DeletePodWithWait(ctx, f.ClientSet, testPod3)
		framework.ExpectNoError(delErr3, "failed to delete pod %s", testPod3.Name)
	})
}

var _ = SIGDescribe(framework.WithSerial(), "Pod InPlace Resize Container (scheduler-focused)", feature.InPlacePodVerticalScaling, func() {
	doPodResizeSchedulerTests()
})

var _ = SIGDescribe("Pod InPlace Resize Container", feature.InPlacePodVerticalScaling, func() {
	doPodResizeTests()
	doPodResizeResourceQuotaTests()
	doPodResizeErrorTests()
})
