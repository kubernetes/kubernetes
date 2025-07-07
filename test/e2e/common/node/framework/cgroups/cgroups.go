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

package cgroups

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	cgroupFsPath          string = "/sys/fs/cgroup"
	cgroupCPUSharesFile   string = "cpu.shares"
	cgroupCPUQuotaFile    string = "cpu.cfs_quota_us"
	cgroupMemLimitFile    string = "memory.limit_in_bytes"
	cgroupv2CPUWeightFile string = "cpu.weight"
	cgroupv2CPULimitFile  string = "cpu.max"
	cgroupv2MemLimitFile  string = "memory.max"
	cgroupVolumeName      string = "sysfscgroup"
	cgroupMountPath       string = "/sysfscgroup"
)

var (
	// TODO: cgroup version shouldn't be cached as a global for a cluster where v1 and v2 are mixed.
	podOnCgroupv2Node      *bool
	podOnCgroupv2NodeMutex sync.Mutex
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

func (cr *ContainerResources) ResourceRequirements() *v1.ResourceRequirements {
	if cr == nil {
		return nil
	}

	var lim, req v1.ResourceList
	if cr.CPULim != "" || cr.MemLim != "" || cr.EphStorLim != "" {
		lim = make(v1.ResourceList)
	}
	if cr.CPUReq != "" || cr.MemReq != "" || cr.EphStorReq != "" {
		req = make(v1.ResourceList)
	}
	if cr.CPULim != "" {
		lim[v1.ResourceCPU] = resource.MustParse(cr.CPULim)
	}
	if cr.MemLim != "" {
		lim[v1.ResourceMemory] = resource.MustParse(cr.MemLim)
	}
	if cr.EphStorLim != "" {
		lim[v1.ResourceEphemeralStorage] = resource.MustParse(cr.EphStorLim)
	}
	if cr.CPUReq != "" {
		req[v1.ResourceCPU] = resource.MustParse(cr.CPUReq)
	}
	if cr.MemReq != "" {
		req[v1.ResourceMemory] = resource.MustParse(cr.MemReq)
	}
	if cr.EphStorReq != "" {
		req[v1.ResourceEphemeralStorage] = resource.MustParse(cr.EphStorReq)
	}
	return &v1.ResourceRequirements{Limits: lim, Requests: req}
}

func MakeContainerWithResources(name string, r *ContainerResources, command string) v1.Container {
	var resources v1.ResourceRequirements
	if r != nil {
		resources = *r.ResourceRequirements()
	}
	return v1.Container{
		Name:      name,
		Resources: resources,
		Image:     imageutils.GetE2EImage(imageutils.BusyBox),
		Command:   []string{"/bin/sh"},
		Args:      []string{"-c", command},
	}
}

func ConfigureHostPathForPodCgroup(pod *v1.Pod) {
	if pod.Spec.Volumes == nil {
		pod.Spec.Volumes = []v1.Volume{}
	}
	pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{
		Name: cgroupVolumeName,
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: cgroupFsPath},
		},
	})
	firstContainer := &pod.Spec.Containers[0]
	if firstContainer.VolumeMounts == nil {
		firstContainer.VolumeMounts = []v1.VolumeMount{}
	}
	firstContainer.VolumeMounts = append(firstContainer.VolumeMounts, v1.VolumeMount{
		Name:      cgroupVolumeName,
		MountPath: cgroupMountPath,
	})
}

func getPodCgroupPath(f *framework.Framework, pod *v1.Pod, podOnCgroupv2 bool, subsystem string) (string, error) {
	rootPath := cgroupMountPath
	if !podOnCgroupv2 {
		rootPath += "/" + subsystem
	}
	// search path for both systemd driver and cgroupfs driver
	cmd := fmt.Sprintf("find %s -name '*%s*' -o -name '*%s*'", rootPath, strings.ReplaceAll(string(pod.UID), "-", "_"), string(pod.UID))
	framework.Logf("Namespace %s Pod %s - looking for Pod cgroup directory path: %q", f.Namespace, pod.Name, cmd)
	podCgPath, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, pod.Spec.Containers[0].Name, []string{"/bin/sh", "-c", cmd}...)
	if podCgPath == "" {
		// This command may hit 'No such file or directory' for another cgroup if another test running in parallel has deleted a pod.
		// We ignore errors if podCgPath is found.
		if err != nil || len(stderr) > 0 {
			return "", fmt.Errorf("encountered error while running command: %q, \nerr: %w \nstdErr: %q", cmd, err, stderr)
		}
		return "", fmt.Errorf("pod cgroup dirctory not found by command: %q", cmd)
	}
	return podCgPath, nil
}

func getCgroupMemLimitPath(cgPath string, podOnCgroupv2 bool) string {
	if podOnCgroupv2 {
		return fmt.Sprintf("%s/%s", cgPath, cgroupv2MemLimitFile)
	} else {
		return fmt.Sprintf("%s/memory/%s", cgPath, cgroupMemLimitFile)
	}
}

func getCgroupCPULimitPath(cgPath string, podOnCgroupv2 bool) string {
	if podOnCgroupv2 {
		return fmt.Sprintf("%s/%s", cgPath, cgroupv2CPULimitFile)
	} else {
		return fmt.Sprintf("%s/cpu/%s", cgPath, cgroupCPUQuotaFile)
	}
}

func getCgroupCPURequestPath(cgPath string, podOnCgroupv2 bool) string {
	if podOnCgroupv2 {
		return fmt.Sprintf("%s/%s", cgPath, cgroupv2CPUWeightFile)
	} else {
		return fmt.Sprintf("%s/cpu/%s", cgPath, cgroupCPUSharesFile)
	}
}

// TODO: Remove the rounded cpu limit values when https://github.com/opencontainers/runc/issues/4622
// is fixed.
func getCPULimitCgroupExpectations(cpuLimit *resource.Quantity, podOnCgroupV2 bool) []string {
	var expectedCPULimits []string
	milliCPULimit := cpuLimit.MilliValue()

	cpuQuota := kubecm.MilliCPUToQuota(milliCPULimit, kubecm.QuotaPeriod)
	if cpuLimit.IsZero() {
		cpuQuota = -1
	}
	expectedCPULimits = append(expectedCPULimits, getExpectedCPULimitFromCPUQuota(cpuQuota, podOnCgroupV2))

	if milliCPULimit%10 != 0 && cpuQuota != -1 {
		roundedCPULimit := (milliCPULimit/10 + 1) * 10
		cpuQuotaRounded := kubecm.MilliCPUToQuota(roundedCPULimit, kubecm.QuotaPeriod)
		expectedCPULimits = append(expectedCPULimits, getExpectedCPULimitFromCPUQuota(cpuQuotaRounded, podOnCgroupV2))
	}

	return expectedCPULimits
}

func getExpectedCPULimitFromCPUQuota(cpuQuota int64, podOnCgroupV2 bool) string {
	expectedCPULimitString := strconv.FormatInt(cpuQuota, 10)
	if podOnCgroupV2 {
		if expectedCPULimitString == "-1" {
			expectedCPULimitString = "max"
		}
		expectedCPULimitString = fmt.Sprintf("%s %d", expectedCPULimitString, kubecm.QuotaPeriod)
	}
	return expectedCPULimitString
}

func getExpectedMemLimitString(rr *v1.ResourceRequirements, podOnCgroupv2 bool) string {
	expectedMemLimitInBytes := rr.Limits.Memory().Value()
	expectedMemLimitString := strconv.FormatInt(expectedMemLimitInBytes, 10)
	if podOnCgroupv2 && expectedMemLimitString == "0" {
		expectedMemLimitString = "max"
	}
	return expectedMemLimitString
}

func verifyContainerCPUWeight(f *framework.Framework, pod *v1.Pod, containerName string, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	cpuWeightCgPath := getCgroupCPURequestPath(cgroupFsPath, podOnCgroupv2)
	expectedCPUShares := getExpectedCPUShares(expectedResources, podOnCgroupv2)
	if err := VerifyCgroupValue(f, pod, containerName, cpuWeightCgPath, expectedCPUShares...); err != nil {
		return fmt.Errorf("failed to verify cpu request cgroup value: %w", err)
	}
	return nil
}

func VerifyContainerCPULimit(f *framework.Framework, pod *v1.Pod, containerName string, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	cpuLimCgPath := getCgroupCPULimitPath(cgroupFsPath, podOnCgroupv2)
	expectedCPULimits := getCPULimitCgroupExpectations(expectedResources.Limits.Cpu(), podOnCgroupv2)
	if err := VerifyCgroupValue(f, pod, containerName, cpuLimCgPath, expectedCPULimits...); err != nil {
		return fmt.Errorf("failed to verify cpu limit cgroup value: %w", err)
	}
	return nil
}

func VerifyContainerMemoryLimit(f *framework.Framework, pod *v1.Pod, containerName string, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	memLimCgPath := getCgroupMemLimitPath(cgroupFsPath, podOnCgroupv2)
	expectedMemLim := getExpectedMemLimitString(expectedResources, podOnCgroupv2)
	if expectedMemLim == "0" {
		return nil
	}
	if err := VerifyCgroupValue(f, pod, containerName, memLimCgPath, expectedMemLim); err != nil {
		return fmt.Errorf("failed to verify memory limit cgroup value: %w", err)
	}
	return nil
}

func VerifyContainerCgroupValues(f *framework.Framework, pod *v1.Pod, tc *v1.Container, podOnCgroupv2 bool) error {
	var errs []error
	errs = append(errs, VerifyContainerMemoryLimit(f, pod, tc.Name, &tc.Resources, podOnCgroupv2))
	errs = append(errs, VerifyContainerCPULimit(f, pod, tc.Name, &tc.Resources, podOnCgroupv2))
	errs = append(errs, verifyContainerCPUWeight(f, pod, tc.Name, &tc.Resources, podOnCgroupv2))
	return utilerrors.NewAggregate(errs)
}

func verifyPodCPUWeight(f *framework.Framework, pod *v1.Pod, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	podCgPath, err := getPodCgroupPath(f, pod, podOnCgroupv2, "cpu")
	if err != nil {
		if podCgPath, err = getPodCgroupPath(f, pod, podOnCgroupv2, "cpu,cpuacct"); err != nil {
			return err
		}
	}

	var cpuWeightCgPath string
	if podOnCgroupv2 {
		cpuWeightCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupv2CPUWeightFile)
	} else {
		cpuWeightCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupCPUSharesFile)
	}
	expectedCPUShares := getExpectedCPUShares(expectedResources, podOnCgroupv2)
	if err := VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, cpuWeightCgPath, expectedCPUShares...); err != nil {
		return fmt.Errorf("pod cgroup cpu weight verification failed: %w", err)
	}
	return nil
}

func verifyPodCPULimit(f *framework.Framework, pod *v1.Pod, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	podCgPath, err := getPodCgroupPath(f, pod, podOnCgroupv2, "cpu")
	if err != nil {
		if podCgPath, err = getPodCgroupPath(f, pod, podOnCgroupv2, "cpu,cpuacct"); err != nil {
			return err
		}
	}

	var cpuLimCgPath string
	if podOnCgroupv2 {
		cpuLimCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupv2CPULimitFile)
	} else {
		cpuLimCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupCPUQuotaFile)
	}
	expectedCPULimits := getCPULimitCgroupExpectations(expectedResources.Limits.Cpu(), podOnCgroupv2)
	if err := VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, cpuLimCgPath, expectedCPULimits...); err != nil {
		return fmt.Errorf("pod cgroup cpu limit verification failed: %w", err)
	}
	return nil
}

func verifyPodMemoryLimit(f *framework.Framework, pod *v1.Pod, expectedResources *v1.ResourceRequirements, podOnCgroupv2 bool) error {
	podCgPath, err := getPodCgroupPath(f, pod, podOnCgroupv2, "memory")
	if err != nil {
		return err
	}

	var memLimCgPath string
	if podOnCgroupv2 {
		memLimCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupv2MemLimitFile)
	} else {
		memLimCgPath = fmt.Sprintf("%s/%s", podCgPath, cgroupMemLimitFile)
	}
	expectedMemLim := getExpectedMemLimitString(expectedResources, podOnCgroupv2)
	if expectedMemLim == "0" {
		return nil
	}

	if err := VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, memLimCgPath, expectedMemLim); err != nil {
		return fmt.Errorf("pod cgroup memory limit verification failed: %w", err)
	}
	return nil
}

// VerifyPodCgroups verifies pod cgroup is configured on a node as expected.
func VerifyPodCgroups(ctx context.Context, f *framework.Framework, pod *v1.Pod, info *ContainerResources) error {
	ginkgo.GinkgoHelper()

	onCgroupV2 := IsPodOnCgroupv2Node(f, pod)

	// Verify cgroup values
	expectedResources := info.ResourceRequirements()
	var errs []error
	errs = append(errs, verifyPodCPUWeight(f, pod, expectedResources, onCgroupV2))
	errs = append(errs, verifyPodCPULimit(f, pod, expectedResources, onCgroupV2))
	errs = append(errs, verifyPodMemoryLimit(f, pod, expectedResources, onCgroupV2))

	return utilerrors.NewAggregate(errs)
}

func BuildPodResourceInfo(podCPURequestMilliValue, podCPULimitMilliValue, podMemoryLimitInBytes int64) ContainerResources {
	podResourceInfo := ContainerResources{}
	if podCPURequestMilliValue > 0 {
		podResourceInfo.CPUReq = fmt.Sprintf("%dm", podCPURequestMilliValue)
	}
	if podCPULimitMilliValue > 0 {
		podResourceInfo.CPULim = fmt.Sprintf("%dm", podCPULimitMilliValue)
	}
	if podMemoryLimitInBytes > 0 {
		podResourceInfo.MemLim = fmt.Sprintf("%d", podMemoryLimitInBytes)
	}
	return podResourceInfo
}

// VerifyCgroupValue verifies that the given cgroup path has the expected value in
// the specified container of the pod. It execs into the container to retrieve the
// cgroup value, and ensures that the retrieved cgroup value is equivalent to at
// least one of the values in expectedCgValues.
func VerifyCgroupValue(f *framework.Framework, pod *v1.Pod, cName, cgPath string, expectedCgValues ...string) error {
	cmd := fmt.Sprintf("head -n 1 %s", cgPath)
	framework.Logf("Namespace %s Pod %s Container %s - looking for one of the expected cgroup values %s in path %s",
		pod.Namespace, pod.Name, cName, expectedCgValues, cgPath)

	const maxRetries = 3
	var cgValue string
	var err error
	for i := range maxRetries {
		cgValue, _, err = e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, cName, "/bin/sh", "-c", cmd)
		if err == nil {
			cgValue = strings.Trim(cgValue, "\n")
			break
		} else {
			framework.Logf("[Attempt %d of %d] Failed to read cgroup value %q for container %q: %v", i+1, maxRetries, cgPath, cName, err)
		}
	}
	if err != nil {
		return fmt.Errorf("failed to read cgroup value %q for container %q after %d attempts: %w", cgPath, cName, maxRetries, err)
	}

	if err := framework.Gomega().Expect(cgValue).To(gomega.BeElementOf(expectedCgValues)); err != nil {
		return fmt.Errorf("value of cgroup %q for container %q was %q; expected one of %q", cgPath, cName, cgValue, expectedCgValues)
	}

	return nil
}

// VerifyOomScoreAdjValue verifies that oom_score_adj for pid 1 (pidof init/systemd -> app)
// has the expected value in specified container of the pod. It execs into the container,
// reads the oom_score_adj value from procfs, and compares it against the expected value.
func VerifyOomScoreAdjValue(f *framework.Framework, pod *v1.Pod, cName, expectedOomScoreAdj string) error {
	cmd := "cat /proc/1/oom_score_adj"
	framework.Logf("Namespace %s Pod %s Container %s - looking for oom_score_adj value %s",
		pod.Namespace, pod.Name, cName, expectedOomScoreAdj)
	oomScoreAdj, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, cName, "/bin/sh", "-c", cmd)
	if err != nil {
		return fmt.Errorf("failed to find expected value %s for container app process", expectedOomScoreAdj)
	}
	oomScoreAdj = strings.Trim(oomScoreAdj, "\n")
	if oomScoreAdj != expectedOomScoreAdj {
		return fmt.Errorf("oom_score_adj value %s not equal to expected %s", oomScoreAdj, expectedOomScoreAdj)
	}
	return nil
}

// IsPodOnCgroupv2Node checks whether the pod is running on cgroupv2 node.
// TODO: Deduplicate this function with NPD cluster e2e test:
// https://github.com/kubernetes/kubernetes/blob/2049360379bcc5d6467769cef112e6e492d3d2f0/test/e2e/node/node_problem_detector.go#L369
func IsPodOnCgroupv2Node(f *framework.Framework, pod *v1.Pod) (result bool) {
	podOnCgroupv2NodeMutex.Lock()
	defer podOnCgroupv2NodeMutex.Unlock()
	if podOnCgroupv2Node != nil {
		return *podOnCgroupv2Node
	}
	defer func() {
		podOnCgroupv2Node = &result
	}()

	cmd := "mount -t cgroup2"
	out, _, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-c", cmd)
	if err != nil {
		return false
	}
	// Some tests mount host cgroup using HostPath for verifying pod cgroup values.
	// In this case, "<mount path>/unified" is detected by "mount -t cgroup2" if cgroup hybrid mode is configured on the host.
	// So, we need to see if "/sys/fs/cgroup" is contained in the output.
	return strings.Contains(out, "/sys/fs/cgroup")
}
