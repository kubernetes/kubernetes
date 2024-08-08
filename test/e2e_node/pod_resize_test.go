/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"

	semver "github.com/blang/semver/v4"
	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	CgroupCPUPeriod            string = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
	CgroupCPUShares            string = "/sys/fs/cgroup/cpu/cpu.shares"
	CgroupCPUQuota             string = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
	CgroupMemLimit             string = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
	Cgroupv2MemLimit           string = "/sys/fs/cgroup/memory.max"
	Cgroupv2MemRequest         string = "/sys/fs/cgroup/memory.min"
	Cgroupv2CPULimit           string = "/sys/fs/cgroup/cpu.max"
	Cgroupv2CPURequest         string = "/sys/fs/cgroup/cpu.weight"
	CPUPeriod                  string = "100000"
	MinContainerRuntimeVersion string = "1.6.9"
)

var (
	podOnCgroupv2Node bool   = IsCgroup2UnifiedMode()
	cgroupMemLimit    string = Cgroupv2MemLimit
	cgroupCPULimit    string = Cgroupv2CPULimit
	cgroupCPURequest  string = Cgroupv2CPURequest
)

type ContainerResources struct {
	CPUReq     string
	CPULim     string
	MemReq     string
	MemLim     string
	EphStorReq string
	EphStorLim string
}

type ContainerAllocations struct {
	CPUAlloc     string
	MemAlloc     string
	ephStorAlloc string
}

type TestContainerInfo struct {
	Name            string
	Resources       *ContainerResources
	Allocations     *ContainerAllocations
	StatusResources *ContainerResources
	CPUPolicy       *v1.ResourceResizeRestartPolicy
	MemPolicy       *v1.ResourceResizeRestartPolicy
	RestartCount    int32
}

func supportsInPlacePodVerticalScaling(ctx context.Context, f *framework.Framework) bool {
	node := getLocalNode(ctx, f)
	re := regexp.MustCompile("containerd://(.*)")
	match := re.FindStringSubmatch(node.Status.NodeInfo.ContainerRuntimeVersion)
	if len(match) != 2 {
		return false
	}
	// TODO(InPlacePodVerticalScaling): Update when RuntimeHandlerFeature for pod resize have been implemented
	if ver, verr := semver.ParseTolerant(match[1]); verr == nil {
		return ver.Compare(semver.MustParse(MinContainerRuntimeVersion)) >= 0
	}
	return false
}

func getTestResourceInfo(tcInfo TestContainerInfo) (v1.ResourceRequirements, v1.ResourceRequirements, v1.ResourceList, []v1.ContainerResizePolicy) {
	var res, statRes v1.ResourceRequirements
	var alloc v1.ResourceList
	var resizePol []v1.ContainerResizePolicy

	createResourceRequirements := func(resources *ContainerResources) v1.ResourceRequirements {
		var lim, req v1.ResourceList
		if resources.CPULim != "" || resources.MemLim != "" || resources.EphStorLim != "" {
			lim = make(v1.ResourceList)
		}
		if resources.CPUReq != "" || resources.MemReq != "" || resources.EphStorReq != "" {
			req = make(v1.ResourceList)
		}
		if resources.CPULim != "" {
			lim[v1.ResourceCPU] = resource.MustParse(resources.CPULim)
		}
		if resources.MemLim != "" {
			lim[v1.ResourceMemory] = resource.MustParse(resources.MemLim)
		}
		if resources.EphStorLim != "" {
			lim[v1.ResourceEphemeralStorage] = resource.MustParse(resources.EphStorLim)
		}
		if resources.CPUReq != "" {
			req[v1.ResourceCPU] = resource.MustParse(resources.CPUReq)
		}
		if resources.MemReq != "" {
			req[v1.ResourceMemory] = resource.MustParse(resources.MemReq)
		}
		if resources.EphStorReq != "" {
			req[v1.ResourceEphemeralStorage] = resource.MustParse(resources.EphStorReq)
		}
		return v1.ResourceRequirements{Limits: lim, Requests: req}

	}

	if tcInfo.Resources != nil {
		res = createResourceRequirements(tcInfo.Resources)
	}
	if tcInfo.StatusResources != nil {
		statRes = createResourceRequirements(tcInfo.StatusResources)
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
	return res, statRes, alloc, resizePol
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
	cmd := "grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d"
	res, statRes, alloc, resizePol := getTestResourceInfo(tcInfo)

	tc := v1.Container{
		Name:         tcInfo.Name,
		Image:        imageutils.GetE2EImage(imageutils.BusyBox),
		Command:      []string{"/bin/sh"},
		Args:         []string{"-c", cmd},
		Resources:    res,
		ResizePolicy: resizePol,
	}

	tcStatus := v1.ContainerStatus{
		Name:               tcInfo.Name,
		AllocatedResources: alloc,
	}
	if tcInfo.StatusResources == nil {
		tcStatus.Resources = &res
	} else {
		tcStatus.Resources = &statRes
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

	podOS = &v1.PodOS{Name: v1.Linux}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels: map[string]string{
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
	ginkgo.GinkgoHelper()
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
	ginkgo.GinkgoHelper()
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

func verifyPodAllocations(pod *v1.Pod, tcInfo []TestContainerInfo) error {
	ginkgo.GinkgoHelper()
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
		if !cmp.Equal(cStatus.AllocatedResources, tcStatus.AllocatedResources) {
			return fmt.Errorf("failed to verify Pod allocations, allocated resources %v not equal to expected %v", cStatus.AllocatedResources, tcStatus.AllocatedResources)
		}
	}
	return nil
}

func verifyPodStatusResources(pod *v1.Pod, tcInfo []TestContainerInfo) {
	ginkgo.GinkgoHelper()
	csMap := make(map[string]*v1.ContainerStatus)
	for i, c := range pod.Status.ContainerStatuses {
		csMap[c.Name] = &pod.Status.ContainerStatuses[i]
	}
	for _, ci := range tcInfo {
		gomega.Expect(csMap).Should(gomega.HaveKey(ci.Name))
		cs := csMap[ci.Name]
		_, tcs := makeTestContainer(ci)
		gomega.Expect(*tcs.Resources).To(gomega.Equal(*cs.Resources))
	}
}

func verifyPodContainersCgroupValues(ctx context.Context, f *framework.Framework, pod *v1.Pod, tcInfo []TestContainerInfo) error {
	ginkgo.GinkgoHelper()
	verifyCgroupValue := func(cName, cgPath, expectedCgValue string) error {
		mycmd := fmt.Sprintf("head -n 1 %s", cgPath)
		cgValue, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, cName, "/bin/sh", "-c", mycmd)
		framework.Logf("Namespace %s Pod %s Container %s - looking for cgroup value %s in path %s",
			pod.Namespace, pod.Name, cName, expectedCgValue, cgPath)
		if err != nil {
			return fmt.Errorf("failed to find expected value '%s' in container cgroup '%s'", expectedCgValue, cgPath)
		}
		cgValue = strings.Trim(cgValue, "\n")
		if cgValue != expectedCgValue {
			return fmt.Errorf("cgroup value '%s' not equal to expected '%s'", cgValue, expectedCgValue)
		}
		return nil
	}
	for _, ci := range tcInfo {
		if ci.Resources == nil {
			continue
		}
		_, tcs := makeTestContainer(ci)
		if tcs.Resources.Limits != nil || tcs.Resources.Requests != nil {
			var expectedCPUShares int64
			var expectedCPULimitString, expectedMemLimitString string
			expectedMemLimitInBytes := tcs.Resources.Limits.Memory().Value()
			cpuRequest := tcs.Resources.Requests.Cpu()
			cpuLimit := tcs.Resources.Limits.Cpu()
			if cpuRequest.IsZero() && !cpuLimit.IsZero() {
				expectedCPUShares = int64(kubecm.MilliCPUToShares(cpuLimit.MilliValue()))
			} else {
				expectedCPUShares = int64(kubecm.MilliCPUToShares(cpuRequest.MilliValue()))
			}
			cpuQuota := kubecm.MilliCPUToQuota(cpuLimit.MilliValue(), kubecm.QuotaPeriod)
			if cpuLimit.IsZero() {
				cpuQuota = -1
			}
			expectedCPULimitString = strconv.FormatInt(cpuQuota, 10)
			expectedMemLimitString = strconv.FormatInt(expectedMemLimitInBytes, 10)
			if podOnCgroupv2Node {
				if expectedCPULimitString == "-1" {
					expectedCPULimitString = "max"
				}
				expectedCPULimitString = fmt.Sprintf("%s %s", expectedCPULimitString, CPUPeriod)
				if expectedMemLimitString == "0" {
					expectedMemLimitString = "max"
				}
				// convert cgroup v1 cpu.shares value to cgroup v2 cpu.weight value
				// https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/2254-cgroup-v2#phase-1-convert-from-cgroups-v1-settings-to-v2
				expectedCPUShares = int64(1 + ((expectedCPUShares-2)*9999)/262142)
			}
			if expectedMemLimitString != "0" {
				err := verifyCgroupValue(ci.Name, cgroupMemLimit, expectedMemLimitString)
				if err != nil {
					return err
				}
			}
			err := verifyCgroupValue(ci.Name, cgroupCPULimit, expectedCPULimitString)
			if err != nil {
				return err
			}
			err = verifyCgroupValue(ci.Name, cgroupCPURequest, strconv.FormatInt(expectedCPUShares, 10))
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func waitForContainerRestart(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, pod *v1.Pod, expectedContainers []TestContainerInfo) error {
	ginkgo.GinkgoHelper()
	var restartContainersExpected []string
	for _, ci := range expectedContainers {
		if ci.RestartCount > 0 {
			restartContainersExpected = append(restartContainersExpected, ci.Name)
		}
	}
	if len(restartContainersExpected) == 0 {
		return nil
	}
	pod, err := podClient.Get(ctx, pod.Name, metav1.GetOptions{})
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
	if restartedContainersCount > len(restartContainersExpected) {
		return fmt.Errorf("more container restarts than expected")
	} else {
		return fmt.Errorf("less container restarts than expected")
	}
}

func waitForPodResizeActuation(ctx context.Context, f *framework.Framework, c clientset.Interface, podClient *e2epod.PodClient, pod, patchedPod *v1.Pod, expectedContainers []TestContainerInfo) *v1.Pod {
	ginkgo.GinkgoHelper()
	var resizedPod *v1.Pod
	var pErr error
	timeouts := framework.NewTimeoutContext()
	// Wait for container restart
	gomega.Eventually(ctx, waitForContainerRestart, timeouts.PodStartShort, timeouts.Poll).
		WithArguments(f, podClient, pod, expectedContainers).
		ShouldNot(gomega.HaveOccurred(), "failed waiting for expected container restart")
		// Verify Pod Containers Cgroup Values
	gomega.Eventually(ctx, verifyPodContainersCgroupValues, timeouts.PodStartShort, timeouts.Poll).
		WithArguments(f, patchedPod, expectedContainers).
		ShouldNot(gomega.HaveOccurred(), "failed to verify container cgroup values to match expected")
	// Wait for pod resource allocations to equal expected values after resize
	gomega.Eventually(ctx, func() error {
		resizedPod, pErr = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		if pErr != nil {
			return pErr
		}
		return verifyPodAllocations(resizedPod, expectedContainers)
	}, timeouts.PodStartShort, timeouts.Poll).
		ShouldNot(gomega.HaveOccurred(), "timed out waiting for pod resource allocation values to match expected")
	return resizedPod
}

func waitForResizeStatusTransition(ctx context.Context, podClient *e2epod.PodClient, pod *v1.Pod, expectedStatus v1.PodResizeStatus) *v1.Pod {
	ginkgo.GinkgoHelper()

	var updatedPod *v1.Pod
	timeouts := framework.NewTimeoutContext()
	// Wait for container restart
	gomega.Eventually(ctx, func() error {
		var err error
		if updatedPod, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{}); err != nil {
			return err
		}
		if updatedPod.Status.Resize != expectedStatus {
			return fmt.Errorf("resize status %v have not been expected %v", updatedPod.Status.Resize, expectedStatus)
		}
		return nil
	}, timeouts.PodStartShort, timeouts.Poll).
		ShouldNot(gomega.HaveOccurred(), "failed waiting for expected container resize status")
	return updatedPod
}

func doPodResizeTests() {
	f := framework.NewDefaultFramework("pod-resize-test")
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

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			ginkgo.By("waiting for the node to be ready", func() {
				if !supportsInPlacePodVerticalScaling(ctx, f) || framework.NodeOSDistroIs("windows") || isRunningOnArm64() {
					e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
				}
			})
			var testPod *v1.Pod
			var patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, allocations are as expected")
			verifyPodResources(newPod, tc.containers)
			ginkgo.By("verifying initial pod resize policy is as expected")
			verifyPodResizePolicy(newPod, tc.containers)

			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, newPod.Namespace, newPod.Name, "Ready", timeouts.PodStartShort, testutils.PodRunningReady)
			framework.ExpectNoError(err, "pod %s/%s did not go running", newPod.Namespace, newPod.Name)
			framework.Logf("pod %s/%s running", newPod.Namespace, newPod.Name)

			ginkgo.By("verifying initial pod status resources")
			verifyPodStatusResources(newPod, tc.containers)

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			framework.ExpectNoError(pErr, "failed to patch pod for resize")

			ginkgo.By("verifying pod patched for resize")
			verifyPodResources(patchedPod, tc.expected)
			gomega.Eventually(ctx, verifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
				WithArguments(patchedPod, tc.containers).
				Should(gomega.BeNil(), "failed to verify Pod allocations for patchedPod")

			ginkgo.By("waiting for resize to be actuated")
			resizedPod := waitForPodResizeActuation(ctx, f, f.ClientSet, podClient, newPod, patchedPod, tc.expected)

			ginkgo.By("verifying pod resources after resize")
			verifyPodResources(resizedPod, tc.expected)

			ginkgo.By("verifying pod allocations after resize")
			gomega.Eventually(ctx, verifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
				WithArguments(resizedPod, tc.expected).
				Should(gomega.BeNil(), "failed to verify Pod allocations for resizedPod")

			ginkgo.By("deleting pod")
			deletePodSyncByName(ctx, f, newPod.Name)
			// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
			// this is in turn needed because we will have an unavoidable (in the current framework) race with the
			// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
			waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)
		})
	}
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

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			ginkgo.By("waiting for the node to be ready", func() {
				if !supportsInPlacePodVerticalScaling(ctx, f) || framework.NodeOSDistroIs("windows") || isRunningOnArm64() {
					e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
				}
			})
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			perr := e2epod.WaitForPodCondition(ctx, f.ClientSet, newPod.Namespace, newPod.Name, "Ready", timeouts.PodStartSlow, testutils.PodRunningReady)
			framework.ExpectNoError(perr, "pod %s/%s did not go running", newPod.Namespace, newPod.Name)
			framework.Logf("pod %s/%s running", newPod.Namespace, newPod.Name)

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{})
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				gomega.Expect(pErr).To(gomega.HaveOccurred(), tc.patchError)
				patchedPod = newPod
			}

			ginkgo.By("verifying pod resources after patch")
			verifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod allocations after patch")
			gomega.Eventually(ctx, verifyPodAllocations, timeouts.PodStartShort, timeouts.Poll).
				WithArguments(patchedPod, tc.expected).
				Should(gomega.BeNil(), "failed to verify Pod allocations for patchedPod")

			deletePodSyncByName(ctx, f, newPod.Name)
			// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
			// this is in turn needed because we will have an unavoidable (in the current framework) race with the
			// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
			waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)
		})
	}
}

func doPodResizeDeferredAndInfeasibleTests() {
	f := framework.NewDefaultFramework("pod-resize-deferred-and-infeasible")
	var podClient *e2epod.PodClient
	var deferredCPU, deferredCPUBurstable, deferredMem, deferredMemBurstable string
	var infeasibleCPU, infeasibleCPUBurstable, infeasibleMem, infeasibleMemBurstable string

	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)

		node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), getNodeName(context.TODO(), f), metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get node information")
		nodeCPU := node.Status.Allocatable[v1.ResourceCPU]
		nodeMemory := node.Status.Allocatable[v1.ResourceMemory]
		deferredCPU = fmt.Sprintf("%dm", int64(float64(nodeCPU.MilliValue())*0.6))
		deferredCPUBurstable = fmt.Sprintf("%dm", int64(float64(nodeCPU.MilliValue())*0.7))
		infeasibleCPU = fmt.Sprintf("%dm", int64(float64(nodeCPU.MilliValue())*1.1))
		infeasibleCPUBurstable = fmt.Sprintf("%dm", int64(float64(nodeCPU.MilliValue())*1.2))
		deferredMem = fmt.Sprintf("%d", int64(float64(nodeMemory.Value())*0.6))
		deferredMemBurstable = fmt.Sprintf("%d", int64(float64(nodeMemory.Value())*0.7))
		infeasibleMem = fmt.Sprintf("%d", int64(float64(nodeMemory.Value())*1.1))
		infeasibleMemBurstable = fmt.Sprintf("%d", int64(float64(nodeMemory.Value())*1.2))
	})

	type recoveryActionType int
	const (
		noAction recoveryActionType = iota
		removeBlocker
		anotherRequest
	)

	type testCase struct {
		name                 string
		containers           []TestContainerInfo
		containerPatches     []v1.Container
		initDelay            bool
		killedContainerName  string
		expected             []TestContainerInfo
		expectedResizeStatus v1.PodResizeStatus
		recoveryAction       recoveryActionType
		recoveryPatches      []v1.Container
		expectedRecovery     []TestContainerInfo
	}

	dummyDefCPU := "101"
	dummyDefCPUB := "102"
	dummyInfCPU := "103"
	dummyInfCPUB := "104"
	dummyDefMem := "105Gi"
	dummyDefMemB := "106Gi"
	dummyInfMem := "107Gi"
	dummyInfMemB := "108Gi"
	qDummyDefCPU := resource.MustParse(dummyDefCPU)
	qDummyDefCPUB := resource.MustParse(dummyDefCPUB)
	qDummyInfCPU := resource.MustParse(dummyInfCPU)
	qDummyInfCPUB := resource.MustParse(dummyInfCPUB)
	qDummyDefMem := resource.MustParse(dummyDefMem)
	qDummyDefMemB := resource.MustParse(dummyDefMemB)
	qDummyInfMem := resource.MustParse(dummyInfMem)
	qDummyInfMemB := resource.MustParse(dummyInfMemB)

	convertResourceQuota := func(dummy resource.Quantity) resource.Quantity {
		switch dummy {
		case qDummyDefCPU:
			return resource.MustParse(deferredCPU)
		case qDummyDefCPUB:
			return resource.MustParse(deferredCPUBurstable)
		case qDummyInfCPU:
			return resource.MustParse(infeasibleCPU)
		case qDummyInfCPUB:
			return resource.MustParse(infeasibleCPUBurstable)
		case qDummyDefMem:
			return resource.MustParse(deferredMem)
		case qDummyDefMemB:
			return resource.MustParse(deferredMemBurstable)
		case qDummyInfMem:
			return resource.MustParse(infeasibleMem)
		case qDummyInfMemB:
			return resource.MustParse(infeasibleMemBurstable)
		default:
			return dummy
		}
	}

	convertResourceString := func(dummy string) string {
		switch dummy {
		case dummyDefCPU:
			return deferredCPU
		case dummyDefCPUB:
			return deferredCPUBurstable
		case dummyInfCPU:
			return infeasibleCPU
		case dummyInfCPUB:
			return infeasibleCPUBurstable
		case dummyDefMem:
			return deferredMem
		case dummyDefMemB:
			return deferredMemBurstable
		case dummyInfMem:
			return infeasibleMem
		case dummyInfMemB:
			return infeasibleMemBurstable
		default:
			return dummy
		}
	}

	convertTestContainerInfo := func(info *TestContainerInfo) {
		r := info.Resources
		if r != nil {
			r.CPUReq = convertResourceString(r.CPUReq)
			r.MemReq = convertResourceString(r.MemReq)
			r.CPULim = convertResourceString(r.CPULim)
			r.MemLim = convertResourceString(r.MemLim)
		}
		a := info.Allocations
		if a != nil {
			a.CPUAlloc = convertResourceString(a.CPUAlloc)
			a.MemAlloc = convertResourceString(a.MemAlloc)
		}
		s := info.StatusResources
		if s != nil {
			s.CPUReq = convertResourceString(s.CPUReq)
			s.MemReq = convertResourceString(s.MemReq)
			s.CPULim = convertResourceString(s.CPULim)
			s.MemLim = convertResourceString(s.MemLim)
		}
	}

	updateTestcase := func(tc *testCase) {
		for i, p := range tc.containerPatches {
			if p.Resources.Requests != nil {
				if q, found := p.Resources.Requests[v1.ResourceCPU]; found {
					tc.containerPatches[i].Resources.Requests[v1.ResourceCPU] = convertResourceQuota(q)
				}
				if q, found := p.Resources.Requests[v1.ResourceMemory]; found {
					tc.containerPatches[i].Resources.Requests[v1.ResourceMemory] = convertResourceQuota(q)
				}
			}
			if p.Resources.Limits != nil {
				if q, found := p.Resources.Limits[v1.ResourceCPU]; found {
					tc.containerPatches[i].Resources.Limits[v1.ResourceCPU] = convertResourceQuota(q)
				}
				if q, found := p.Resources.Limits[v1.ResourceMemory]; found {
					tc.containerPatches[i].Resources.Limits[v1.ResourceMemory] = convertResourceQuota(q)
				}
			}
		}
		for _, c := range tc.expected {
			convertTestContainerInfo(&c)
		}
		for _, c := range tc.expectedRecovery {
			convertTestContainerInfo(&c)
		}
	}

	tests := []testCase{
		{
			name: "Guaranteed QoS pod - try deferred memory resize, expect Deferred, rollback",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod - two containers - try Deferred memory reseize(c1) and acceptable resize(c2), expect Deferred, only c1 rollback",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations:     &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "200Mi"},
					StatusResources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
				},
				{
					Name:        "c2",
					Resources:   &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "250Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod - try infeasible memory resize, expect Infeasible, rollback",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyInfMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyInfMem,
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyInfMem, MemLim: dummyInfMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusInfeasible,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod - two containers - try infeasible memory reseize(c1) and acceptable resize(c2), expect Infeasible, only c1 rollback",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyInfMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyInfMem,
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyInfMem, MemLim: dummyInfMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations:     &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "200Mi"},
					StatusResources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusInfeasible,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("100Mi"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
				},
				{
					Name:        "c2",
					Resources:   &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "250Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod - try deferred cpu resize, expect Deferred, recover by acceptable resize",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: qDummyDefCPU,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: qDummyDefCPU,
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: dummyDefCPU, CPULim: dummyDefCPU},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("150m"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("250m"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "150m", CPULim: "250m"},
					Allocations: &ContainerAllocations{CPUAlloc: "150m"},
				},
			},
		},
		{
			name: "Burstable QoS pod - two containers - try deferred cpu resize(c2) and acceptable resize(c1) at starting the pod, expect Deferred, remove blocker",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "300m"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("150m"),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: qDummyDefCPU,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: qDummyDefCPU,
						},
					},
				},
			},
			initDelay: true,
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "150m", CPULim: "200m"},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: dummyDefCPU, CPULim: dummyDefCPU},
					Allocations:     &ContainerAllocations{CPUAlloc: "200m"},
					StatusResources: &ContainerResources{CPUReq: "200m"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       removeBlocker,
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "150m", CPULim: "200m"},
					Allocations: &ContainerAllocations{CPUAlloc: "150m"},
				},
				{
					Name:        "c2",
					Resources:   &ContainerResources{CPUReq: dummyDefCPU, CPULim: dummyDefCPU},
					Allocations: &ContainerAllocations{CPUAlloc: dummyDefCPU},
				},
			},
		},
		{
			name: "Burstable QoS pod - two containers - try infeasible cpu reseize(c1) and deferred memory resize(c2), expect Infeasible, recover by acceptable resize",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "300Mi"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "150m", CPULim: "300m", MemReq: "100Mi", MemLim: "200Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: qDummyInfCPU,
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: qDummyInfCPUB,
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMemB,
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: dummyInfCPU, CPULim: dummyInfCPUB, MemReq: "200Mi", MemLim: "300Mi"},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "200Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "300Mi"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: "150m", CPULim: "300m", MemReq: dummyDefMem, MemLim: dummyDefMemB},
					Allocations:     &ContainerAllocations{CPUAlloc: "150m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "150m", CPULim: "300m", MemReq: "100Mi", MemLim: "200Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusInfeasible,
			recoveryAction:       anotherRequest,
			recoveryPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("150m"),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("250m"),
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("50Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("150Mi"),
						},
					},
				},
			},
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "150m", CPULim: "250m", MemReq: "200Mi", MemLim: "300Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "150m", MemAlloc: "200Mi"},
				},
				{
					Name:        "c2",
					Resources:   &ContainerResources{CPUReq: "150m", CPULim: "300m", MemReq: "50Mi", MemLim: "150Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "150m", MemAlloc: "50Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod - try deferred memory resize at starting the pod, expect Deferred, remove blocker",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
					},
				},
			},
			initDelay: true,
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       removeBlocker,
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: dummyDefMem},
				},
			},
		},
		{
			name: "Guaranteed QoS pod - two containers - try Deferred memory reseize(c1) and acceptable resize(c2), expect Deferred, remove blocker",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: qDummyDefMem,
						},
					},
				},
				{
					Name: "c2",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
						Limits: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("250Mi"),
						},
					},
				},
			},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations:     &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "200Mi"},
					StatusResources: &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
			recoveryAction:       removeBlocker,
			expectedRecovery: []TestContainerInfo{
				{
					Name:        "c1",
					Resources:   &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: dummyDefMem, MemLim: dummyDefMem},
					Allocations: &ContainerAllocations{CPUAlloc: "100m", MemAlloc: dummyDefMem},
				},
				{
					Name:        "c2",
					Resources:   &ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "250Mi", MemLim: "250Mi"},
					Allocations: &ContainerAllocations{CPUAlloc: "200m", MemAlloc: "250Mi"},
				},
			},
		},
	}

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			ginkgo.By("waiting for the node to be ready", func() {
				if !supportsInPlacePodVerticalScaling(ctx, f) || framework.NodeOSDistroIs("windows") || isRunningOnArm64() {
					e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
				}
			})
			updateTestcase(&tc)

			var blockerPod *v1.Pod
			if tc.expectedResizeStatus == v1.PodResizeStatusDeferred {
				// Create another pod that requets resources in order to cause Deferred.
				ginkgo.By("creating blocker pod for testing deferred resize")
				tStamp := strconv.Itoa(time.Now().Nanosecond())
				blockerPod = makeTestPod(f.Namespace.Name, "blocker-pod", tStamp,
					[]TestContainerInfo{
						{
							Name:      "blocker1",
							Resources: &ContainerResources{CPUReq: deferredCPU, MemReq: deferredMem},
						},
					})
				blockerPod = e2epod.MustMixinRestrictedPodSecurity(blockerPod)
				blockerPod = podClient.CreateSync(ctx, blockerPod)
				defer func() {
					deletePodSyncByName(ctx, f, blockerPod.Name)
				}()
			}

			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			initDefaultResizePolicy(tc.containers)
			initDefaultResizePolicy(tc.expected)
			testPod = makeTestPod(f.Namespace.Name, "testpod", tStamp, tc.containers)
			if tc.initDelay {
				testPod.Spec.InitContainers = []v1.Container{
					{
						Name:    "init-container",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sh"},
						// Set resources not to affect QoS class
						Args: []string{"-c", "sleep 5"},
						Resources: v1.ResourceRequirements{
							Limits: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
							Requests: v1.ResourceList{
								v1.ResourceCPU:    resource.MustParse("100m"),
								v1.ResourceMemory: resource.MustParse("100Mi"),
							},
						},
					},
				}
			}
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			if tc.killedContainerName != "" {
				for i, c := range testPod.Spec.Containers {
					if c.Name != tc.killedContainerName {
						continue
					}
					testPod.Spec.Containers[i].Command = []string{"/bin/sh", "-c", e2epod.InfiniteSleepCommand}
				}
				testPod.Spec.RestartPolicy = v1.RestartPolicyAlways
			}

			ginkgo.By("creating pod")
			var newPod *v1.Pod
			if tc.initDelay {
				newPod = podClient.Create(ctx, testPod)
				// With InitDelay, resize will be requested while init container is running for #126527
			} else {
				newPod = podClient.CreateSync(ctx, testPod)
				perr := e2epod.WaitForPodCondition(ctx, f.ClientSet, newPod.Namespace, newPod.Name, "Ready", timeouts.PodStartSlow, testutils.PodRunningReady)
				framework.ExpectNoError(perr, "pod %s/%s did not go running", newPod.Namespace, newPod.Name)
				framework.Logf("pod %s/%s running", newPod.Namespace, newPod.Name)
			}

			defer func() {
				deletePodSyncByName(ctx, f, newPod.Name)
				// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
				// this is in turn needed because we will have an unavoidable (in the current framework) race with the
				// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
				waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)
			}()

			if !tc.initDelay {
				ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
				verifyPodResources(newPod, tc.containers)
				verifyPodResizePolicy(newPod, tc.containers)

				ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
				verifyPodStatusResources(newPod, tc.containers)
			}

			ginkgo.By("patching pod for resize")
			patch, err := json.Marshal(v1.Pod{Spec: v1.PodSpec{Containers: tc.containerPatches}})
			framework.ExpectNoError(err, "failed to marshal patch")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			framework.ExpectNoError(pErr, "failed to patch pod for resize")

			ginkgo.By("verifying pod resources after patch")
			verifyPodResources(patchedPod, tc.expected)

			if tc.initDelay {
				ginkgo.By("wating for pod to be ready")
				perr := e2epod.WaitForPodCondition(ctx, f.ClientSet, patchedPod.Namespace, patchedPod.Name, "Ready", timeouts.PodStartSlow, testutils.PodRunningReady)
				framework.ExpectNoError(perr, "pod %s/%s did not go ready", patchedPod.Namespace, patchedPod.Name)
				framework.Logf("pod %s/%s running", newPod.Namespace, newPod.Name)
				patchedPod, perr = f.ClientSet.CoreV1().Pods(patchedPod.Namespace).Get(ctx, patchedPod.Name, metav1.GetOptions{})
				framework.ExpectNoError(perr, "failed to get pod %s/%s", patchedPod.Namespace, patchedPod.Name)
			}

			ginkgo.By("verifying pod allocations after patch")
			err = verifyPodAllocations(patchedPod, tc.expected)
			framework.ExpectNoError(err, "failed to verify Pod allocations for patchedPod")

			verify := func(phase string, expectedStatus v1.PodResizeStatus, expected, expectedResources []TestContainerInfo) {
				ginkgo.By(fmt.Sprintf("waiting for pod resize state to transit %s", phase))
				updatedPod := waitForResizeStatusTransition(ctx, podClient, patchedPod, expectedStatus)

				ginkgo.By(fmt.Sprintf("verifying pod resources %s", phase))
				verifyPodResources(updatedPod, expected)
				ginkgo.By(fmt.Sprintf("verifying pod allocations %s", phase))
				err = verifyPodAllocations(updatedPod, expected)
				framework.ExpectNoError(err, "fail")

				ginkgo.By(fmt.Sprintf("verifying pod status resources and cgroup config are as expected %s", phase))
				verifyPodStatusResources(updatedPod, expectedResources)
				err = verifyPodContainersCgroupValues(ctx, f, updatedPod, expectedResources)
			}
			verify("after resize state transtion", tc.expectedResizeStatus, tc.expected, tc.containers)

			if tc.killedContainerName != "" {
				ginkgo.By("killing container")
				// Restart container in order to verify that issue #126033 is fixed.
				_ = e2epod.ExecShellInContainer(f, patchedPod.Name, tc.killedContainerName, "kill 1")

				ginkgo.By("waiting for container to be restarted and verifying pod resources after restart")
				for i, c := range tc.expected {
					if c.Name == tc.killedContainerName {
						tc.expected[i].RestartCount++
					}
				}
				gomega.Eventually(ctx, waitForContainerRestart, timeouts.PodStartShort, timeouts.Poll).
					WithArguments(f, podClient, patchedPod, tc.expected).
					ShouldNot(gomega.HaveOccurred(), "failed waiting for expected container restart")
				verify("after restarting container", tc.expectedResizeStatus, tc.expected, tc.containers)
			}

			if tc.recoveryAction == noAction {
				return
			}

			if tc.recoveryAction == removeBlocker {
				ginkgo.By("delete blocker pod to resume deferred resize")
				deletePodSyncByName(ctx, f, blockerPod.Name)
			} else {
				ginkgo.By("patching pod for recovery")
				patch, err := json.Marshal(v1.Pod{Spec: v1.PodSpec{Containers: tc.recoveryPatches}})
				framework.ExpectNoError(err, "failed to marshal recovery patch")
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(patchedPod.Namespace).Patch(ctx, patchedPod.Name,
					types.StrategicMergePatchType, patch, metav1.PatchOptions{})
				framework.ExpectNoError(pErr, "failed to patch pod for resize")

				ginkgo.By("verifying pod resources after recovery patch")
				verifyPodResources(patchedPod, tc.expectedRecovery)
			}

			verify("after recovery", "", tc.expectedRecovery, tc.expectedRecovery)
		})
	}
}

// NOTE: Pod resize scheduler resource quota tests are out of scope in e2e_node tests,
//       because in e2e_node tests
//          a) scheduler and controller manager is not running by the Node e2e
//          b) api-server in services doesn't start with --enable-admission-plugins=ResourceQuota
//             and is not possible to start it from TEST_ARGS
//       Above tests are performed by doSheduletTests() and doPodResizeResourceQuotaTests()
//       in test/node/pod_resize_test.go

var _ = SIGDescribe("Pod InPlace Resize Container", framework.WithSerial(), feature.InPlacePodVerticalScaling, "[NodeAlphaFeature:InPlacePodVerticalScaling]", func() {
	if !podOnCgroupv2Node {
		cgroupMemLimit = CgroupMemLimit
		cgroupCPULimit = CgroupCPUQuota
		cgroupCPURequest = CgroupCPUShares
	}
	doPodResizeTests()
	doPodResizeErrorTests()
	doPodResizeDeferredAndInfeasibleTests()
})
