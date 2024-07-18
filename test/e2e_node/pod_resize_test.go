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
		if updatedPod, err = podClient.Get(context.TODO(), pod.Name, metav1.GetOptions{}); err != nil {
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

	const (
		cpuDeferred = iota
		cpuInfeasible
		memDeferred
		memInfeasible
	)

	type resizeError struct {
		cIdx    int
		rStatus int
	}

	type testCase struct {
		name                 string
		containers           []TestContainerInfo
		containerPatches     []v1.Container
		patchError           string
		resizeErrors         []resizeError
		killedContainerName  string
		expected             []TestContainerInfo
		expectedResizeStatus v1.PodResizeStatus
	}

	tests := []testCase{
		{
			name: "BestEffort pod - try requesting memory, expect error",
			containers: []TestContainerInfo{
				{
					Name: "c1",
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceMemory: resource.MustParse("400Mi"),
						},
					},
				},
			},
			patchError: "Pod QoS is immutable",
			expected: []TestContainerInfo{
				{
					Name: "c1",
				},
			},
		},
		{
			name: "Guaranteed QoS pod - try deferred memory resize, expect Deferred",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
				},
			},
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: memDeferred}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m"},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
		},
		{
			name: "Guaranteed QoS pod - two containers - try Deferred memory reseize(c1) and acceptable resize(c2), expect Deferred",
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
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: memDeferred}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m"},
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
		},
		{
			name: "Guaranteed QoS pod - try infeasible memory resize, expect Infeasible",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
				},
			},
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: memInfeasible}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m"},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusInfeasible,
		},
		{
			name: "Guaranteed QoS pod - two containers - try infeasible memory reseize(c1) and acceptable resize(c2), expect Infeasible",
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
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: memInfeasible}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "100m", CPULim: "100m"},
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
		},
		{
			name: "Burstable QoS pod - try deferred cpu resize, expect Deferred",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
				},
			},
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: cpuDeferred}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Allocations:     &ContainerAllocations{CPUAlloc: "100m"},
					StatusResources: &ContainerResources{CPUReq: "100m"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
		},
		{
			name: "Burstable QoS pod - two containers - try infeasible cpu reseize(c2) and acceptable resize(c1), expect Deferred",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "200m"},
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
				},
			},
			resizeErrors:        []resizeError{{cIdx: 1, rStatus: cpuDeferred}},
			killedContainerName: "c2",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Resources:       &ContainerResources{CPUReq: "150m"},
					Allocations:     &ContainerAllocations{CPUAlloc: "100m"},
					StatusResources: &ContainerResources{CPUReq: "100m"},
				},
				{
					Name:            "c2",
					Allocations:     &ContainerAllocations{CPUAlloc: "200m"},
					StatusResources: &ContainerResources{CPUReq: "200m"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusDeferred,
		},
		{
			name: "Burstable QoS pod - two containers - try infeasible cpu reseize(c1) and deferred memory resize(c2), expect Infeasible",
			containers: []TestContainerInfo{
				{
					Name:      "c1",
					Resources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
				{
					Name:      "c2",
					Resources: &ContainerResources{CPUReq: "150m", MemReq: "100Mi", MemLim: "200Mi"},
				},
			},
			containerPatches: []v1.Container{
				{
					Name: "c1",
				},
				{
					Name: "c2",
				},
			},
			resizeErrors:        []resizeError{{cIdx: 0, rStatus: cpuInfeasible}, {cIdx: 1, rStatus: memDeferred}},
			killedContainerName: "c1",
			expected: []TestContainerInfo{
				{
					Name:            "c1",
					Allocations:     &ContainerAllocations{CPUAlloc: "100m"},
					StatusResources: &ContainerResources{CPUReq: "100m", CPULim: "200m"},
				},
				{
					Name:            "c2",
					Resources:       &ContainerResources{CPUReq: "150m"},
					Allocations:     &ContainerAllocations{CPUAlloc: "150m", MemAlloc: "100Mi"},
					StatusResources: &ContainerResources{CPUReq: "150m", MemReq: "100Mi", MemLim: "200Mi"},
				},
			},
			expectedResizeStatus: v1.PodResizeStatusInfeasible,
		},
	}

	timeouts := framework.NewTimeoutContext()

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			ginkgo.By("waiting for the node to be ready", func() {
				if !supportsInPlacePodVerticalScaling(ctx, f) || framework.NodeOSDistroIs("windows") {
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
			newPod := podClient.CreateSync(ctx, testPod)

			perr := e2epod.WaitForPodCondition(ctx, f.ClientSet, newPod.Namespace, newPod.Name, "Ready", timeouts.PodStartSlow, testutils.PodRunningReady)
			framework.ExpectNoError(perr, "pod %s/%s did not go running", newPod.Namespace, newPod.Name)
			framework.Logf("pod %s/%s running", newPod.Namespace, newPod.Name)

			defer func() {
				deletePodSyncByName(ctx, f, newPod.Name)
				// we need to wait for all containers to really be gone so cpumanager reconcile loop will not rewrite the cpu_manager_state.
				// this is in turn needed because we will have an unavoidable (in the current framework) race with the
				// reconcile loop which will make our attempt to delete the state file and to restore the old config go haywire
				waitForAllContainerRemoval(ctx, newPod.Name, newPod.Namespace)
			}()

			ginkgo.By("verifying initial pod resources, allocations, and policy are as expected")
			verifyPodResources(newPod, tc.containers)
			verifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			verifyPodStatusResources(newPod, tc.containers)

			needsOtherPod := false
			for _, rError := range tc.resizeErrors {
				cIdx := rError.cIdx
				// Cause Deferred by requesting the max node allocatable resource
				// Cause Infeasible by requesting over max node allocatable resource
				update := func(resourceName v1.ResourceName, nodeAlloc resource.Quantity, isDeferred bool) {
					patch := v1.ResourceRequirements{}
					var newReq resource.Quantity
					if isDeferred {
						reqVal := nodeAlloc.MilliValue()
						for i, c := range newPod.Spec.Containers {
							if i == cIdx {
								continue
							}
							if req, found := tc.containerPatches[i].Resources.Requests[resourceName]; found {
								reqVal -= req.MilliValue()
							} else if req, found := c.Resources.Requests[resourceName]; found {
								reqVal -= req.MilliValue()
							}
						}
						newReq = resource.MustParse(fmt.Sprintf("%dm", reqVal))
					} else {
						newReq = nodeAlloc.DeepCopy()
						_ = newReq.Mul(2)
					}
					patch.Requests = v1.ResourceList{
						resourceName: newReq,
					}

					if newPod.Status.QOSClass == v1.PodQOSGuaranteed {
						patch.Limits = v1.ResourceList{
							resourceName: newReq,
						}
					} else if _, found := newPod.Spec.Containers[cIdx].Resources.Limits[resourceName]; found {
						// Burstable and limit configured
						newLim := newReq.DeepCopy()
						_ = newLim.Mul(2)
						patch.Limits = v1.ResourceList{
							resourceName: newLim,
						}
					}

					tc.containerPatches[cIdx].Resources = patch
				}

				node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, newPod.Spec.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err, "failed to get node information")
				nodeCPU := node.Status.Allocatable[v1.ResourceCPU]
				nodeMemory := node.Status.Allocatable[v1.ResourceMemory]
				switch rError.rStatus {
				case cpuDeferred:
					update(v1.ResourceCPU, nodeCPU, true)
					needsOtherPod = true
				case cpuInfeasible:
					update(v1.ResourceCPU, nodeCPU, false)
				case memDeferred:
					update(v1.ResourceMemory, nodeMemory, true)
					needsOtherPod = true
				case memInfeasible:
					update(v1.ResourceMemory, nodeMemory, false)
				}

				// Update resource spec in expected containers
				if tc.expected[cIdx].Resources == nil {
					tc.expected[cIdx].Resources = &ContainerResources{}
				}
				if rError.rStatus == cpuDeferred || rError.rStatus == cpuInfeasible {
					tc.expected[cIdx].Resources.CPUReq = tc.containerPatches[cIdx].Resources.Requests.Cpu().String()
					if lim, found := tc.containerPatches[cIdx].Resources.Limits[v1.ResourceCPU]; found {
						tc.expected[cIdx].Resources.CPULim = lim.String()
					}
				}
				if rError.rStatus == memDeferred || rError.rStatus == memInfeasible {
					tc.expected[cIdx].Resources.MemReq = tc.containerPatches[cIdx].Resources.Requests.Memory().String()
					if lim, found := tc.containerPatches[cIdx].Resources.Limits[v1.ResourceMemory]; found {
						tc.expected[cIdx].Resources.MemLim = lim.String()
					}
				}
			}

			if needsOtherPod {
				// Create another pod that requets resources in order to cause Deferred.
				tStamp := strconv.Itoa(time.Now().Nanosecond())
				otherPod := makeTestPod(f.Namespace.Name, "anotherpod", tStamp,
					[]TestContainerInfo{
						{
							Name:      "other1",
							Resources: &ContainerResources{CPUReq: "100m", MemReq: "100Mi"},
						},
					})
				otherPod = e2epod.MustMixinRestrictedPodSecurity(otherPod)
				otherPod.Spec.NodeName = testPod.Spec.NodeName
				otherPod = podClient.CreateSync(ctx, otherPod)
				defer func() {
					deletePodSyncByName(ctx, f, otherPod.Name)
				}()
			}

			ginkgo.By("patching pod for resize")
			patch, err := json.Marshal(v1.Pod{Spec: v1.PodSpec{Containers: tc.containerPatches}})
			framework.ExpectNoError(err, "failed to marshal patch")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, patch, metav1.PatchOptions{})
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				gomega.Expect(pErr).To(gomega.HaveOccurred(), tc.patchError)
				patchedPod = newPod
			}

			ginkgo.By("verifying pod resources after patch")
			verifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod allocations after patch")
			err = verifyPodAllocations(patchedPod, tc.expected)
			framework.ExpectNoError(err, "failed to verify Pod allocations for patchedPod")

			if tc.patchError != "" {
				return
			}

			verify := func(phase string) {
				ginkgo.By(fmt.Sprintf("waiting for pod resize state to transit %s", phase))
				updatedPod := waitForResizeStatusTransition(ctx, podClient, patchedPod, tc.expectedResizeStatus)

				ginkgo.By(fmt.Sprintf("verifying pod resources %s", phase))
				verifyPodResources(updatedPod, tc.expected)
				ginkgo.By(fmt.Sprintf("verifying pod allocations %s", phase))
				err = verifyPodAllocations(updatedPod, tc.expected)
				framework.ExpectNoError(err, "fail")

				ginkgo.By(fmt.Sprintf("verifying pod status resources and cgroup config are as expected %s", phase))
				verifyPodStatusResources(updatedPod, tc.containers)
				if !framework.NodeOSDistroIs("windows") {
					err = verifyPodContainersCgroupValues(ctx, f, newPod, tc.containers)
					framework.ExpectNoError(err, "fail")
				}
			}
			verify("after resize state transtion")

			if tc.killedContainerName == "" {
				return
			}

			ginkgo.By("killing container")
			// Restart container in order to verify that issue #126033 is fixed.
			_ = e2epod.ExecShellInContainer(f, patchedPod.Name, tc.killedContainerName, "kill 1")

			ginkgo.By("waiting for container to be restarted and verifying pod resources after restart")
			// TODO: RestartCount is not updated because the pod status is overwritten with an old status
			//       while its resize status is in Infeasible/Deferred (#125205).
			//       Once this issue fixed, we should wait for RestartCount to be updated.
			// As workaround, use sleep to wait for the container to be restarted.
			time.Sleep(5 * time.Second)
			// for i, c := range tc.expected {
			// 	if c.Name == tc.killedContainerName {
			// 		tc.expected[i].RestartCount++
			// 	}
			// }
			// gomega.Eventually(ctx, waitForContainerRestart, timeouts.PodStartShort, timeouts.Poll).
			// 	WithArguments(f, podClient, patchedPod, tc.expected).
			// 	ShouldNot(gomega.HaveOccurred(), "failed waiting for expected container restart")
			verify("after restarting container")
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

var _ = SIGDescribe("Pod InPlace Resize Container", framework.WithSerial(), feature.InPlacePodVerticalScaling, func() {
	if !podOnCgroupv2Node {
		cgroupMemLimit = CgroupMemLimit
		cgroupCPULimit = CgroupCPUQuota
		cgroupCPURequest = CgroupCPUShares
	}
	doPodResizeTests()
	doPodResizeErrorTests()
})
