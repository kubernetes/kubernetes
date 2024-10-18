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

package pod

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
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
	podOnCgroupv2Node *bool
)

type ContainerResources struct {
	CPUReq              string
	CPULim              string
	MemReq              string
	MemLim              string
	EphStorReq          string
	EphStorLim          string
	ExtendedResourceReq string
	ExtendedResourceLim string
}

type ContainerAllocations struct {
	CPUAlloc              string
	MemAlloc              string
	ephStorAlloc          string
	ExtendedResourceAlloc string
}

type ResizableContainerInfo struct {
	Name         string
	Resources    *ContainerResources
	Allocations  *ContainerAllocations
	CPUPolicy    *v1.ResourceResizeRestartPolicy
	MemPolicy    *v1.ResourceResizeRestartPolicy
	RestartCount int32
}

type containerPatch struct {
	Name      string `json:"name"`
	Resources struct {
		Requests struct {
			CPU     string `json:"cpu,omitempty"`
			Memory  string `json:"memory,omitempty"`
			EphStor string `json:"ephemeral-storage,omitempty"`
		} `json:"requests"`
		Limits struct {
			CPU     string `json:"cpu,omitempty"`
			Memory  string `json:"memory,omitempty"`
			EphStor string `json:"ephemeral-storage,omitempty"`
		} `json:"limits"`
	} `json:"resources"`
}

type patchSpec struct {
	Spec struct {
		Containers []containerPatch `json:"containers"`
	} `json:"spec"`
}

func getTestResourceInfo(tcInfo ResizableContainerInfo) (v1.ResourceRequirements, v1.ResourceList, []v1.ContainerResizePolicy) {
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

func InitDefaultResizePolicy(containers []ResizableContainerInfo) {
	noRestart := v1.NotRequired
	setDefaultPolicy := func(ci *ResizableContainerInfo) {
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

func makeResizableContainer(tcInfo ResizableContainerInfo) (v1.Container, v1.ContainerStatus) {
	cmd := "grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d"
	res, alloc, resizePol := getTestResourceInfo(tcInfo)

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
	return tc, tcStatus
}

func MakePodWithResizableContainers(ns, name, timeStamp string, tcInfo []ResizableContainerInfo) *v1.Pod {
	var testContainers []v1.Container

	for _, ci := range tcInfo {
		tc, _ := makeResizableContainer(ci)
		testContainers = append(testContainers, tc)
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels: map[string]string{
				"time": timeStamp,
			},
		},
		Spec: v1.PodSpec{
			OS:            &v1.PodOS{Name: v1.Linux},
			Containers:    testContainers,
			RestartPolicy: v1.RestartPolicyOnFailure,
		},
	}
	return pod
}

func VerifyPodResizePolicy(gotPod *v1.Pod, wantCtrs []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()
	gomega.Expect(gotPod.Spec.Containers).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for i, wantCtr := range wantCtrs {
		gotCtr := &gotPod.Spec.Containers[i]
		ctr, _ := makeResizableContainer(wantCtr)
		gomega.Expect(gotCtr.Name).To(gomega.Equal(ctr.Name))
		gomega.Expect(gotCtr.ResizePolicy).To(gomega.Equal(ctr.ResizePolicy))
	}
}

func VerifyPodResources(gotPod *v1.Pod, wantCtrs []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()
	gomega.Expect(gotPod.Spec.Containers).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for i, wantCtr := range wantCtrs {
		gotCtr := &gotPod.Spec.Containers[i]
		ctr, _ := makeResizableContainer(wantCtr)
		gomega.Expect(gotCtr.Name).To(gomega.Equal(ctr.Name))
		gomega.Expect(gotCtr.Resources).To(gomega.Equal(ctr.Resources))
	}
}

func VerifyPodAllocations(gotPod *v1.Pod, wantCtrs []ResizableContainerInfo) error {
	ginkgo.GinkgoHelper()
	gomega.Expect(gotPod.Status.ContainerStatuses).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for i, wantCtr := range wantCtrs {
		gotCtrStatus := &gotPod.Status.ContainerStatuses[i]
		if wantCtr.Allocations == nil {
			if wantCtr.Resources != nil {
				alloc := &ContainerAllocations{CPUAlloc: wantCtr.Resources.CPUReq, MemAlloc: wantCtr.Resources.MemReq}
				wantCtr.Allocations = alloc
				defer func() {
					wantCtr.Allocations = nil
				}()
			}
		}

		_, ctrStatus := makeResizableContainer(wantCtr)
		gomega.Expect(gotCtrStatus.Name).To(gomega.Equal(ctrStatus.Name))
		if !cmp.Equal(gotCtrStatus.AllocatedResources, ctrStatus.AllocatedResources) {
			return fmt.Errorf("failed to verify Pod allocations, allocated resources not equal to expected")
		}
	}
	return nil
}

func VerifyPodStatusResources(gotPod *v1.Pod, wantCtrs []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()
	gomega.Expect(gotPod.Status.ContainerStatuses).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for i, wantCtr := range wantCtrs {
		gotCtrStatus := &gotPod.Status.ContainerStatuses[i]
		ctr, _ := makeResizableContainer(wantCtr)
		gomega.Expect(gotCtrStatus.Name).To(gomega.Equal(ctr.Name))
		gomega.Expect(ctr.Resources).To(gomega.Equal(*gotCtrStatus.Resources))
	}
}

// isPodOnCgroupv2Node checks whether the pod is running on cgroupv2 node.
// TODO: Deduplicate this function with NPD cluster e2e test:
// https://github.com/kubernetes/kubernetes/blob/2049360379bcc5d6467769cef112e6e492d3d2f0/test/e2e/node/node_problem_detector.go#L369
func isPodOnCgroupv2Node(f *framework.Framework, pod *v1.Pod) bool {
	cmd := "mount -t cgroup2"
	out, _, err := ExecCommandInContainerWithFullOutput(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-c", cmd)
	if err != nil {
		return false
	}
	return len(out) != 0
}

func VerifyPodContainersCgroupValues(ctx context.Context, f *framework.Framework, pod *v1.Pod, tcInfo []ResizableContainerInfo) error {
	ginkgo.GinkgoHelper()
	if podOnCgroupv2Node == nil {
		value := isPodOnCgroupv2Node(f, pod)
		podOnCgroupv2Node = &value
	}
	cgroupMemLimit := Cgroupv2MemLimit
	cgroupCPULimit := Cgroupv2CPULimit
	cgroupCPURequest := Cgroupv2CPURequest
	if !*podOnCgroupv2Node {
		cgroupMemLimit = CgroupMemLimit
		cgroupCPULimit = CgroupCPUQuota
		cgroupCPURequest = CgroupCPUShares
	}
	verifyCgroupValue := func(cName, cgPath, expectedCgValue string) error {
		cmd := fmt.Sprintf("head -n 1 %s", cgPath)
		framework.Logf("Namespace %s Pod %s Container %s - looking for cgroup value %s in path %s",
			pod.Namespace, pod.Name, cName, expectedCgValue, cgPath)
		cgValue, _, err := ExecCommandInContainerWithFullOutput(f, pod.Name, cName, "/bin/sh", "-c", cmd)
		if err != nil {
			return fmt.Errorf("failed to find expected value %q in container cgroup %q", expectedCgValue, cgPath)
		}
		cgValue = strings.Trim(cgValue, "\n")
		if cgValue != expectedCgValue {
			return fmt.Errorf("cgroup value %q not equal to expected %q", cgValue, expectedCgValue)
		}
		return nil
	}
	for _, ci := range tcInfo {
		if ci.Resources == nil {
			continue
		}
		tc, _ := makeResizableContainer(ci)
		if tc.Resources.Limits != nil || tc.Resources.Requests != nil {
			var expectedCPUShares int64
			var expectedCPULimitString, expectedMemLimitString string
			expectedMemLimitInBytes := tc.Resources.Limits.Memory().Value()
			cpuRequest := tc.Resources.Requests.Cpu()
			cpuLimit := tc.Resources.Limits.Cpu()
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
			if *podOnCgroupv2Node {
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

func waitForContainerRestart(ctx context.Context, podClient *PodClient, pod *v1.Pod, expectedContainers []ResizableContainerInfo, initialContainers []ResizableContainerInfo, isRollback bool) error {
	ginkgo.GinkgoHelper()
	var restartContainersExpected []string

	restartContainers := expectedContainers
	// if we're rolling back, extract restart counts from test case "expected" containers
	if isRollback {
		restartContainers = initialContainers
	}

	for _, ci := range restartContainers {
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

func WaitForPodResizeActuation(ctx context.Context, f *framework.Framework, podClient *PodClient, pod, patchedPod *v1.Pod, expectedContainers []ResizableContainerInfo, initialContainers []ResizableContainerInfo, isRollback bool) *v1.Pod {
	ginkgo.GinkgoHelper()
	var resizedPod *v1.Pod
	var pErr error
	timeouts := framework.NewTimeoutContext()
	// Wait for container restart
	gomega.Eventually(ctx, waitForContainerRestart, timeouts.PodStartShort, timeouts.Poll).
		WithArguments(podClient, pod, expectedContainers, initialContainers, isRollback).
		ShouldNot(gomega.HaveOccurred(), "failed waiting for expected container restart")
		// Verify Pod Containers Cgroup Values
	gomega.Eventually(ctx, VerifyPodContainersCgroupValues, timeouts.PodStartShort, timeouts.Poll).
		WithArguments(f, patchedPod, expectedContainers).
		ShouldNot(gomega.HaveOccurred(), "failed to verify container cgroup values to match expected")
	// Wait for pod resource allocations to equal expected values after resize
	gomega.Eventually(ctx, func() error {
		resizedPod, pErr = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
		if pErr != nil {
			return pErr
		}
		return VerifyPodAllocations(resizedPod, expectedContainers)
	}, timeouts.PodStartShort, timeouts.Poll).
		ShouldNot(gomega.HaveOccurred(), "timed out waiting for pod resource allocation values to match expected")
	return resizedPod
}

// ResizeContainerPatch generates a patch string to resize the pod container.
func ResizeContainerPatch(containers []ResizableContainerInfo) (string, error) {
	var patch patchSpec

	for _, container := range containers {
		var cPatch containerPatch
		cPatch.Name = container.Name
		cPatch.Resources.Requests.CPU = container.Resources.CPUReq
		cPatch.Resources.Requests.Memory = container.Resources.MemReq
		cPatch.Resources.Limits.CPU = container.Resources.CPULim
		cPatch.Resources.Limits.Memory = container.Resources.MemLim

		patch.Spec.Containers = append(patch.Spec.Containers, cPatch)
	}

	patchBytes, err := json.Marshal(patch)
	if err != nil {
		return "", err
	}

	return string(patchBytes), nil
}
