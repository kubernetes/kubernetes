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
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

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
	CPUPolicy    *v1.ResourceResizePolicy
	MemPolicy    *v1.ResourceResizePolicy
	RestartCount int32
}

func isFeatureGatePostAlpha() bool {
	if fs, found := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()[features.InPlacePodVerticalScaling]; found {
		if fs.PreRelease == featuregate.Alpha {
			return false
		}
	}
	return true
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
		cpuPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceCPU, Policy: *tcInfo.CPUPolicy}
		resizePol = append(resizePol, cpuPol)
	}
	if tcInfo.MemPolicy != nil {
		memPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, Policy: *tcInfo.MemPolicy}
		resizePol = append(resizePol, memPol)
	}
	return res, alloc, resizePol
}

func initDefaultResizePolicy(containers []TestContainerInfo) {
	noRestart := v1.RestartNotRequired
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
	cmd := "trap exit TERM; while true; do sleep 1; done"
	res, alloc, resizePol := getTestResourceInfo(tcInfo)
	bTrue := true
	bFalse := false
	userID := int64(1001)
	tc := v1.Container{
		Name:         tcInfo.Name,
		Image:        imageutils.GetE2EImage(imageutils.BusyBox),
		Command:      []string{"/bin/sh"},
		Args:         []string{"-c", cmd},
		Resources:    res,
		ResizePolicy: resizePol,
		SecurityContext: &v1.SecurityContext{
			Privileged:               &bFalse,
			AllowPrivilegeEscalation: &bFalse,
			RunAsNonRoot:             &bTrue,
			RunAsUser:                &userID,
			Capabilities: &v1.Capabilities{
				Drop: []v1.Capability{"ALL"},
			},
			SeccompProfile: &v1.SeccompProfile{
				Type: v1.SeccompProfileTypeRuntimeDefault,
			},
		},
	}

	tcStatus := v1.ContainerStatus{
		Name:               tcInfo.Name,
		ResourcesAllocated: alloc,
	}
	return tc, tcStatus
}

func makeTestPod(ns, name, timeStamp string, tcInfo []TestContainerInfo) *v1.Pod {
	var testContainers []v1.Container
	for _, ci := range tcInfo {
		tc, _ := makeTestContainer(ci)
		testContainers = append(testContainers, tc)
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
		c, found := cMap[ci.Name]
		gomega.Expect(found == true)
		tc, _ := makeTestContainer(ci)
		framework.ExpectEqual(tc.ResizePolicy, c.ResizePolicy)
	}
}

func verifyPodResources(pod *v1.Pod, tcInfo []TestContainerInfo) {
	cMap := make(map[string]*v1.Container)
	for i, c := range pod.Spec.Containers {
		cMap[c.Name] = &pod.Spec.Containers[i]
	}
	for _, ci := range tcInfo {
		c, found := cMap[ci.Name]
		gomega.Expect(found == true)
		tc, _ := makeTestContainer(ci)
		framework.ExpectEqual(tc.Resources, c.Resources)
	}
}

func verifyPodAllocations(pod *v1.Pod, tcInfo []TestContainerInfo, flagError bool) bool {
	cStatusMap := make(map[string]*v1.ContainerStatus)
	for i, c := range pod.Status.ContainerStatuses {
		cStatusMap[c.Name] = &pod.Status.ContainerStatuses[i]
	}

	for _, ci := range tcInfo {
		cStatus, found := cStatusMap[ci.Name]
		gomega.Expect(found == true)
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
			framework.ExpectEqual(tcStatus.ResourcesAllocated, cStatus.ResourcesAllocated)
		}
		if diff.ObjectDiff(cStatus.ResourcesAllocated, tcStatus.ResourcesAllocated) != "" {
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
		cs, found := csMap[ci.Name]
		gomega.Expect(found == true)
		tc, _ := makeTestContainer(ci)
		framework.ExpectEqual(tc.Resources, *cs.Resources)
		//framework.ExpectEqual(cs.RestartCount, ci.RestartCount)
	}
}

func isPodOnCgroupv2Node(pod *v1.Pod) bool {
	// Determine if pod is running on cgroupv2 or cgroupv1 node
	cgroupv2File := "/sys/fs/cgroup/cgroup.controllers"
	_, err := framework.RunKubectl(pod.Namespace, "exec", pod.Name, "--", "ls", cgroupv2File)
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
		cgValue, err := framework.LookForStringInPodExecToContainer(pod.Namespace, pod.Name, cName, cmd, expectedCgValue, PollTimeout)
		if flagError {
			framework.ExpectNoError(err, "failed to find expected cgroup value in container")
		}
		cgValue = strings.Trim(cgValue, "\n")
		if flagError {
			gomega.Expect(cgValue == expectedCgValue)
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
			if podOnCgroupv2Node && cpuLimitString == "-1" {
				cpuLimitString = "max"
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
			if !verifyCgroupValue(ci.Name, cgroupCPURequest, strconv.FormatInt(cpuShares, 10)) {
				return false
			}
		}
	}
	return true
}

func waitForPodResizeActuation(podClient *framework.PodClient, pod, patchedPod *v1.Pod, expectedContainers []TestContainerInfo) *v1.Pod {

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
			if verifyPodAllocations(pod, expectedContainers, false) == false {
				continue
			}
			return pod, nil
		}
		return nil, fmt.Errorf("timed out waiting for pod resource allocation values to match expected")
	}
	waitContainerCgroupValuesEqualsExpected := func() error {
		for start := time.Now(); time.Since(start) < PollTimeout; time.Sleep(PollInterval) {
			if verifyPodContainersCgroupValues(patchedPod, expectedContainers, false) == false {
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
				if diff.ObjectDiff(c.Resources, *pod.Status.ContainerStatuses[idx].Resources) != "" {
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
	//TODO(vinaykul,InPlacePodVerticalScaling): Remove this check when cgroupv2 support is added
	if !isPodOnCgroupv2Node(pod) {
		// Wait for container cgroup values to equal expected cgroup values after resize
		cErr := waitContainerCgroupValuesEqualsExpected()
		framework.ExpectNoError(cErr, "failed to verify container cgroup values equals expected values")
	}
	//TODO(vinaykul,InPlacePodVerticalScaling): Remove featureGatePostAlpha upon exiting Alpha.
	//                containerd needs to add CRI support before Beta (See Node KEP #2273)
	if isFeatureGatePostAlpha() {
		// Wait for PodSpec container resources to equal PodStatus container resources indicating resize is complete
		rPod, rErr := waitPodStatusResourcesEqualSpecResources()
		framework.ExpectNoError(rErr, "failed to verify pod spec resources equals pod status resources")

		ginkgo.By("verifying pod status after resize")
		verifyPodStatusResources(rPod, expectedContainers)
	}
	return resizedPod
}

func doPodResizeTests() {
	f := framework.NewDefaultFramework("pod-resize")

	type testCase struct {
		name        string
		containers  []TestContainerInfo
		patchString string
		expected    []TestContainerInfo
	}

	noRestart := v1.RestartNotRequired
	doRestart := v1.RestartRequired
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
			name: "Guaranteed QoS pod, one container - increase CPU (RestartNotRequired) & memory (RestartRequired)",
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
			name: "Burstable QoS pod, one container - decrease CPU (RestartRequired) & memory (RestartNotRequired)",
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
		ginkgo.It(tc.name, func() {
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)

			ginkgo.By("creating pod")
			newPod := f.PodClient().CreateSync(testPod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := f.PodClient().List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pods")
			gomega.Expect(len(podList.Items) == 1)

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)
			verifyPodContainersCgroupValues(newPod, tc.containers, true)

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(context.TODO(), newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			framework.ExpectNoError(pErr, "failed to patch pod for resize")

			ginkgo.By("verifying pod patched for resize")
			verifyPodResources(patchedPod, tc.expected)
			verifyPodAllocations(patchedPod, tc.containers, true)

			ginkgo.By("waiting for resize to be actuated")
			resizedPod := waitForPodResizeActuation(f.PodClient(), newPod, patchedPod, tc.expected)

			ginkgo.By("verifying pod container's cgroup values after resize")
			//TODO(vinaykul,InPlacePodVerticalScaling): Remove this check when cgroupv2 support is added
			if !isPodOnCgroupv2Node(resizedPod) {
				verifyPodContainersCgroupValues(resizedPod, tc.expected, true)
			}

			ginkgo.By("verifying pod resources after resize")
			verifyPodResources(resizedPod, tc.expected)

			ginkgo.By("verifying pod allocations after resize")
			verifyPodAllocations(resizedPod, tc.expected, true)

			ginkgo.By("deleting pod")
			err = e2epod.DeletePodWithWait(f.ClientSet, newPod)
			framework.ExpectNoError(err, "failed to delete pod")
		})
	}
}

func doPodResizeResourceQuotaTests() {
	f := framework.NewDefaultFramework("pod-resize-resource-quota")

	ginkgo.It("pod-resize-resource-quota-test", func() {
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
		newPod1 := f.PodClient().CreateSync(testPod1)
		newPod2 := f.PodClient().CreateSync(testPod2)

		ginkgo.By("verifying the pod is in kubernetes")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
		options := metav1.ListOptions{LabelSelector: selector.String()}
		podList, listErr := f.PodClient().List(context.TODO(), options)
		framework.ExpectNoError(listErr, "failed to query for pods")
		gomega.Expect(len(podList.Items) == 2)

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
		resizedPod := waitForPodResizeActuation(f.PodClient(), newPod1, patchedPod, expected)

		ginkgo.By("verifying pod container's cgroup values after resize")
		//TODO(vinaykul,InPlacePodVerticalScaling): Remove this check when cgroupv2 support is added
		if !isPodOnCgroupv2Node(resizedPod) {
			verifyPodContainersCgroupValues(resizedPod, expected, true)
		}

		ginkgo.By("verifying pod resources after resize")
		verifyPodResources(resizedPod, expected)

		ginkgo.By("verifying pod allocations after resize")
		verifyPodAllocations(resizedPod, expected, true)

		ginkgo.By(fmt.Sprintf("patching pod %s for resize with CPU exceeding resource quota", resizedPod.Name))
		_, pErrExceedCPU := f.ClientSet.CoreV1().Pods(resizedPod.Namespace).Patch(context.TODO(),
			resizedPod.Name, types.StrategicMergePatchType, []byte(patchStringExceedCPU), metav1.PatchOptions{})
		framework.ExpectError(pErrExceedCPU, "exceeded quota: %s, requested: cpu=200m, used: cpu=700m, limited: cpu=800m",
			resourceQuota.Name)

		ginkgo.By("verifying pod patched for resize exceeding CPU resource quota remains unchanged")
		patchedPodExceedCPU, pErrEx1 := f.PodClient().Get(context.TODO(), resizedPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(pErrEx1, "failed to get pod post exceed CPU resize")
		verifyPodResources(patchedPodExceedCPU, expected)
		verifyPodAllocations(patchedPodExceedCPU, expected, true)

		ginkgo.By("patching pod for resize with memory exceeding resource quota")
		_, pErrExceedMemory := f.ClientSet.CoreV1().Pods(resizedPod.Namespace).Patch(context.TODO(),
			resizedPod.Name, types.StrategicMergePatchType, []byte(patchStringExceedMemory), metav1.PatchOptions{})
		framework.ExpectError(pErrExceedMemory, "exceeded quota: %s, requested: memory=350Mi, used: memory=700Mi, limited: memory=800Mi",
			resourceQuota.Name)

		ginkgo.By("verifying pod patched for resize exceeding memory resource quota remains unchanged")
		patchedPodExceedMemory, pErrEx2 := f.PodClient().Get(context.TODO(), resizedPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(pErrEx2, "failed to get pod post exceed memory resize")
		verifyPodResources(patchedPodExceedMemory, expected)
		verifyPodAllocations(patchedPodExceedMemory, expected, true)

		ginkgo.By("deleting pods")
		delErr1 := e2epod.DeletePodWithWait(f.ClientSet, newPod1)
		framework.ExpectNoError(delErr1, "failed to delete pod %s", newPod1.Name)
		delErr2 := e2epod.DeletePodWithWait(f.ClientSet, newPod2)
		framework.ExpectNoError(delErr2, "failed to delete pod %s", newPod2.Name)
	})
}

func doPodResizeErrorTests() {
	f := framework.NewDefaultFramework("pod-resize-errors")

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
		ginkgo.It(tc.name, func() {
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)

			ginkgo.By("creating pod")
			newPod := f.PodClient().CreateSync(testPod)

			ginkgo.By("verifying the pod is in kubernetes")
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": tStamp}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			podList, err := f.PodClient().List(context.TODO(), options)
			framework.ExpectNoError(err, "failed to query for pods")
			gomega.Expect(len(podList.Items) == 1)

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)
			verifyPodContainersCgroupValues(newPod, tc.containers, true)

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(context.TODO(), newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				framework.ExpectError(pErr, tc.patchError)
				patchedPod = newPod
			}

			ginkgo.By("verifying pod container's cgroup values after patch")
			//TODO(vinaykul,InPlacePodVerticalScaling): Remove this check when cgroupv2 support is added
			if !isPodOnCgroupv2Node(patchedPod) {
				verifyPodContainersCgroupValues(patchedPod, tc.expected, true)
			}

			ginkgo.By("verifying pod resources after patch")
			verifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod allocations after patch")
			verifyPodAllocations(patchedPod, tc.expected, true)

			ginkgo.By("deleting pod")
			err = e2epod.DeletePodWithWait(f.ClientSet, newPod)
			framework.ExpectNoError(err, "failed to delete pod")
		})
	}
}

var _ = SIGDescribe("Pod InPlace Resize Container [Feature:InPlacePodVerticalScaling]", func() {
	doPodResizeTests()
	doPodResizeResourceQuotaTests()
	doPodResizeErrorTests()
})
