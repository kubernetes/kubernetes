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

package podresize

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	helpers "k8s.io/component-helpers/resource"
	"k8s.io/kubectl/pkg/util/podutils"
	kubeqos "k8s.io/kubernetes/pkg/kubelet/qos"
	"k8s.io/kubernetes/test/e2e/common/node/framework/cgroups"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	MinContainerRuntimeVersion string = "1.6.9"
)

type ResizableContainerInfo struct {
	Name          string
	Resources     *cgroups.ContainerResources
	CPUPolicy     *v1.ResourceResizeRestartPolicy
	MemPolicy     *v1.ResourceResizeRestartPolicy
	RestartCount  int32
	RestartPolicy v1.ContainerRestartPolicy
	InitCtr       bool
}

func getTestResizePolicy(tcInfo ResizableContainerInfo) (resizePol []v1.ContainerResizePolicy) {
	if tcInfo.CPUPolicy != nil {
		cpuPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceCPU, RestartPolicy: *tcInfo.CPUPolicy}
		resizePol = append(resizePol, cpuPol)
	}
	if tcInfo.MemPolicy != nil {
		memPol := v1.ContainerResizePolicy{ResourceName: v1.ResourceMemory, RestartPolicy: *tcInfo.MemPolicy}
		resizePol = append(resizePol, memPol)
	}
	return resizePol
}

func makeResizableContainer(tcInfo ResizableContainerInfo) v1.Container {
	cmd := "grep Cpus_allowed_list /proc/self/status | cut -f2 && sleep 1d"
	resizePol := getTestResizePolicy(tcInfo)
	tc := cgroups.MakeContainerWithResources(tcInfo.Name, tcInfo.Resources, cmd)
	tc.ResizePolicy = resizePol
	if tcInfo.RestartPolicy != "" {
		tc.RestartPolicy = &tcInfo.RestartPolicy
	}

	return tc
}

func MakePodWithResizableContainers(ns, name, timeStamp string, tcInfo []ResizableContainerInfo) *v1.Pod {
	testInitContainers, testContainers := separateContainers(tcInfo)

	minGracePeriodSeconds := int64(0)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
			Labels: map[string]string{
				"time": timeStamp,
			},
		},
		Spec: v1.PodSpec{
			OS:                            &v1.PodOS{Name: v1.Linux},
			InitContainers:                testInitContainers,
			Containers:                    testContainers,
			RestartPolicy:                 v1.RestartPolicyOnFailure,
			TerminationGracePeriodSeconds: &minGracePeriodSeconds,
		},
	}
	return pod
}

// separateContainers splits the input into initContainers and normal containers.
func separateContainers(tcInfo []ResizableContainerInfo) ([]v1.Container, []v1.Container) {
	var initContainers, containers []v1.Container

	for _, ci := range tcInfo {
		tc := makeResizableContainer(ci)
		if ci.InitCtr {
			initContainers = append(initContainers, tc)
		} else {
			containers = append(containers, tc)
		}
	}

	return initContainers, containers
}

// separateContainerStatuses splits the input into initContainerStatuses and containerStatuses.
func separateContainerStatuses(tcInfo []ResizableContainerInfo) ([]v1.ContainerStatus, []v1.ContainerStatus) {
	var containerStatuses, initContainerStatuses []v1.ContainerStatus

	for _, ci := range tcInfo {
		ctrStatus := v1.ContainerStatus{
			Name:         ci.Name,
			RestartCount: ci.RestartCount,
		}
		if ci.InitCtr {
			initContainerStatuses = append(initContainerStatuses, ctrStatus)
		} else {
			containerStatuses = append(containerStatuses, ctrStatus)
		}
	}

	return initContainerStatuses, containerStatuses
}

func VerifyPodResizePolicy(gotPod *v1.Pod, wantInfo []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()

	gotCtrs := append(append([]v1.Container{}, gotPod.Spec.Containers...), gotPod.Spec.InitContainers...)
	var wantCtrs []v1.Container
	for _, ci := range wantInfo {
		wantCtrs = append(wantCtrs, makeResizableContainer(ci))
	}
	gomega.Expect(gotCtrs).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for _, wantCtr := range wantCtrs {
		for _, gotCtr := range gotCtrs {
			if wantCtr.Name != gotCtr.Name {
				continue
			}
			gomega.Expect(v1.Container{Name: gotCtr.Name, ResizePolicy: gotCtr.ResizePolicy}).To(gomega.Equal(v1.Container{Name: wantCtr.Name, ResizePolicy: wantCtr.ResizePolicy}))
		}
	}
}

func VerifyPodResources(gotPod *v1.Pod, wantInfo []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()

	gotCtrs := append(append([]v1.Container{}, gotPod.Spec.Containers...), gotPod.Spec.InitContainers...)
	var wantCtrs []v1.Container
	for _, ci := range wantInfo {
		wantCtrs = append(wantCtrs, makeResizableContainer(ci))
	}
	gomega.Expect(gotCtrs).To(gomega.HaveLen(len(wantCtrs)), "number of containers in pod spec should match")
	for _, wantCtr := range wantCtrs {
		for _, gotCtr := range gotCtrs {
			if wantCtr.Name != gotCtr.Name {
				continue
			}
			gomega.Expect(gotCtr.Resources).To(gomega.BeComparableTo(wantCtr.Resources))
		}
	}
}

func VerifyPodStatusResources(gotPod *v1.Pod, wantInfo []ResizableContainerInfo) error {
	ginkgo.GinkgoHelper()

	wantInitCtrs, wantCtrs := separateContainers(wantInfo)
	var errs []error
	if err := verifyPodContainersStatusResources(gotPod.Status.InitContainerStatuses, wantInitCtrs); err != nil {
		errs = append(errs, err)
	}
	if err := verifyPodContainersStatusResources(gotPod.Status.ContainerStatuses, wantCtrs); err != nil {
		errs = append(errs, err)
	}

	return utilerrors.NewAggregate(errs)
}

func verifyPodContainersStatusResources(gotCtrStatuses []v1.ContainerStatus, wantCtrs []v1.Container) error {
	ginkgo.GinkgoHelper()

	var errs []error
	if len(gotCtrStatuses) != len(wantCtrs) {
		return fmt.Errorf("expectation length mismatch: got %d statuses, want %d",
			len(gotCtrStatuses), len(wantCtrs))
	}
	for i, wantCtr := range wantCtrs {
		gotCtrStatus := gotCtrStatuses[i]
		if gotCtrStatus.Name != wantCtr.Name {
			errs = append(errs, fmt.Errorf("container status %d name %q != expected name %q", i, gotCtrStatus.Name, wantCtr.Name))
			continue
		}

		if err := framework.Gomega().Expect(*gotCtrStatus.Resources).To(gomega.BeComparableTo(wantCtr.Resources)); err != nil {
			errs = append(errs, fmt.Errorf("container[%s] status resources mismatch: %w", wantCtr.Name, err))
		}
	}

	return utilerrors.NewAggregate(errs)
}

func VerifyPodContainersCgroupValues(ctx context.Context, f *framework.Framework, pod *v1.Pod, tcInfo []ResizableContainerInfo) error {
	ginkgo.GinkgoHelper()

	onCgroupv2 := cgroups.IsPodOnCgroupv2Node(f, pod)

	var errs []error
	// Verify container cgroup values
	for _, ci := range tcInfo {
		tc := makeResizableContainer(ci)
		errs = append(errs, cgroups.VerifyContainerCgroupValues(f, pod, &tc, onCgroupv2))
	}

	// Verify pod cgroup values
	// TODO: Consider PodLevelResources feature and move this function to `cgroups` package.
	expectedPod := MakePodWithResizableContainers(pod.Namespace, pod.Name, pod.Labels["time"], tcInfo)

	podRequests := helpers.PodRequests(expectedPod, helpers.PodResourcesOptions{
		SkipPodLevelResources: true,
	})

	cpuLimitDeclared := true
	memoryLimitDeclared := true
	podLimits := helpers.PodLimits(expectedPod, helpers.PodResourcesOptions{
		SkipPodLevelResources: true,
		ContainerFn: func(res v1.ResourceList, containerType helpers.ContainerType) {
			if res.Cpu().IsZero() {
				cpuLimitDeclared = false
			}
			if res.Memory().IsZero() {
				memoryLimitDeclared = false
			}
		},
	})
	var podCPULimitMilliValue, podMemoryLimitInBytes int64
	if cpuLimitDeclared {
		podCPULimitMilliValue = podLimits.Cpu().MilliValue()
	}
	if memoryLimitDeclared {
		podMemoryLimitInBytes = podLimits.Memory().Value()
	}

	podResourceInfo := cgroups.BuildPodResourceInfo(podRequests.Cpu().MilliValue(), podCPULimitMilliValue, podMemoryLimitInBytes)
	errs = append(errs, cgroups.VerifyPodCgroups(ctx, f, pod, &podResourceInfo))

	return utilerrors.NewAggregate(errs)
}

func verifyPodRestarts(f *framework.Framework, pod *v1.Pod, wantInfo []ResizableContainerInfo) error {
	ginkgo.GinkgoHelper()

	initCtrStatuses, ctrStatuses := separateContainerStatuses(wantInfo)
	errs := []error{}
	if err := verifyContainerRestarts(f, pod, pod.Status.InitContainerStatuses, initCtrStatuses); err != nil {
		errs = append(errs, err)
	}
	if err := verifyContainerRestarts(f, pod, pod.Status.ContainerStatuses, ctrStatuses); err != nil {
		errs = append(errs, err)
	}

	return utilerrors.NewAggregate(errs)
}

func verifyContainerRestarts(f *framework.Framework, pod *v1.Pod, gotStatuses []v1.ContainerStatus, wantStatuses []v1.ContainerStatus) error {
	ginkgo.GinkgoHelper()

	if len(gotStatuses) != len(wantStatuses) {
		return fmt.Errorf("expectation length mismatch: got %d statuses, want %d",
			len(gotStatuses), len(wantStatuses))
	}

	errs := []error{}
	for i, gotStatus := range gotStatuses {
		if gotStatus.RestartCount != wantStatuses[i].RestartCount {
			errs = append(errs, fmt.Errorf("unexpected number of restarts for container %s: got %d, want %d", gotStatus.Name, gotStatus.RestartCount, wantStatuses[i].RestartCount))
		} else if gotStatus.RestartCount > 0 {
			if err := verifyOomScoreAdj(f, pod, gotStatus.Name); err != nil {
				errs = append(errs, err)
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}

func verifyOomScoreAdj(f *framework.Framework, pod *v1.Pod, containerName string) error {
	container := e2epod.FindContainerInPod(pod, containerName)
	if container == nil {
		return fmt.Errorf("failed to find container %s in pod %s", containerName, pod.Name)
	}

	node, err := f.ClientSet.CoreV1().Nodes().Get(context.Background(), pod.Spec.NodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}

	nodeMemoryCapacity := node.Status.Capacity[v1.ResourceMemory]
	oomScoreAdj := kubeqos.GetContainerOOMScoreAdjust(pod, container, int64(nodeMemoryCapacity.Value()))
	expectedOomScoreAdj := strconv.FormatInt(int64(oomScoreAdj), 10)

	return cgroups.VerifyOomScoreAdjValue(f, pod, container.Name, expectedOomScoreAdj)
}

func WaitForPodResizeActuation(ctx context.Context, f *framework.Framework, podClient *e2epod.PodClient, pod *v1.Pod, expectedContainers []ResizableContainerInfo) *v1.Pod {
	ginkgo.GinkgoHelper()
	// Wait for resize to complete.

	framework.ExpectNoError(framework.Gomega().
		Eventually(ctx, framework.RetryNotFound(framework.GetObject(f.ClientSet.CoreV1().Pods(pod.Namespace).Get, pod.Name, metav1.GetOptions{}))).
		WithTimeout(f.Timeouts.PodStart).
		Should(framework.MakeMatcher(func(pod *v1.Pod) (func() string, error) {
			if helpers.IsPodResizeInfeasible(pod) {
				// This is a terminal resize state
				return func() string {
					return "resize is infeasible"
				}, nil
			}
			// TODO: Replace this check with a combination of checking the status.observedGeneration
			// and the resize status when available.
			if resourceErrs := VerifyPodStatusResources(pod, expectedContainers); resourceErrs != nil {
				return func() string {
					return fmt.Sprintf("container status resources don't match expected: %v", formatErrors(resourceErrs))
				}, nil
			}
			// Wait for kubelet to clear the resize status conditions.
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.PodResizePending || c.Type == v1.PodResizeInProgress {
					return func() string {
						return fmt.Sprintf("resize status %v is still present in the pod status", c)
					}, nil
				}
			}
			// Wait for the pod to be ready.
			if !podutils.IsPodReady(pod) {
				return func() string { return "pod is not ready" }, nil
			}
			return nil, nil
		})),
	)

	resizedPod, err := framework.GetObject(podClient.Get, pod.Name, metav1.GetOptions{})(ctx)
	framework.ExpectNoError(err, "failed to get resized pod")
	return resizedPod
}

func ExpectPodResized(ctx context.Context, f *framework.Framework, resizedPod *v1.Pod, expectedContainers []ResizableContainerInfo) {
	ginkgo.GinkgoHelper()

	// Verify Pod Containers Cgroup Values
	var errs []error
	if cgroupErrs := VerifyPodContainersCgroupValues(ctx, f, resizedPod, expectedContainers); cgroupErrs != nil {
		errs = append(errs, fmt.Errorf("container cgroup values don't match expected: %w", formatErrors(cgroupErrs)))
	}
	if resourceErrs := VerifyPodStatusResources(resizedPod, expectedContainers); resourceErrs != nil {
		errs = append(errs, fmt.Errorf("container status resources don't match expected: %w", formatErrors(resourceErrs)))
	}
	if restartErrs := verifyPodRestarts(f, resizedPod, expectedContainers); restartErrs != nil {
		errs = append(errs, fmt.Errorf("container restart counts don't match expected: %w", formatErrors(restartErrs)))
	}

	// Verify Pod Resize conditions are empty.
	for _, condition := range resizedPod.Status.Conditions {
		if condition.Type == v1.PodResizeInProgress || condition.Type == v1.PodResizePending {
			errs = append(errs, fmt.Errorf("unexpected resize condition type %s found in pod status", condition.Type))
		}
	}

	if len(errs) > 0 {
		resizedPod.ManagedFields = nil // Suppress managed fields in error output.
		framework.ExpectNoError(formatErrors(utilerrors.NewAggregate(errs)),
			"Verifying pod resources resize state. Pod: %s", framework.PrettyPrintJSON(resizedPod))
	}
}

func MakeResizePatch(originalContainers, desiredContainers []ResizableContainerInfo) []byte {
	original, err := json.Marshal(MakePodWithResizableContainers("", "", "", originalContainers))
	framework.ExpectNoError(err)

	desired, err := json.Marshal(MakePodWithResizableContainers("", "", "", desiredContainers))
	framework.ExpectNoError(err)

	patch, err := strategicpatch.CreateTwoWayMergePatch(original, desired, v1.Pod{})
	framework.ExpectNoError(err)

	return patch
}

// UpdateExpectedContainerRestarts updates the RestartCounts in expectedContainers by
// adding them to the existing RestartCounts in the containerStatuses of the provided pod.
// This reduces the flakiness of the RestartCount assertions by grabbing the current
// restart count right before the resize operation, and verify the expected increment (0 or 1)
// rather than the absolute count.
func UpdateExpectedContainerRestarts(ctx context.Context, pod *v1.Pod, expectedContainers []ResizableContainerInfo) []ResizableContainerInfo {
	initialRestarts := make(map[string]int32)
	newExpectedContainers := []ResizableContainerInfo{}
	for _, ctr := range pod.Status.ContainerStatuses {
		initialRestarts[ctr.Name] = ctr.RestartCount
	}
	for i, ctr := range expectedContainers {
		newExpectedContainers = append(newExpectedContainers, expectedContainers[i])
		newExpectedContainers[i].RestartCount += initialRestarts[ctr.Name]
	}
	return newExpectedContainers
}

func formatErrors(err error) error {
	// Put each error on a new line for readability.
	var agg utilerrors.Aggregate
	if !errors.As(err, &agg) {
		return err
	}

	errStrings := make([]string, len(agg.Errors()))
	for i, err := range agg.Errors() {
		errStrings[i] = err.Error()
	}
	return fmt.Errorf("[\n%s\n]", strings.Join(errStrings, ",\n"))
}
