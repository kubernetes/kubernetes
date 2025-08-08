//go:build linux

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

package node

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	kubecm "k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	cgroupv2CPUWeight string = "cpu.weight"
	cgroupv2CPULimit  string = "cpu.max"
	cgroupv2MemLimit  string = "memory.max"
	cgroupFsPath      string = "/sys/fs/cgroup"
	CPUPeriod         string = "100000"
	mountPath         string = "/sysfscgroup"
)

var (
	cmd = []string{"/bin/sh", "-c", "sleep 1d"}
)

var _ = SIGDescribe("Pod Level Resources", framework.WithSerial(), feature.PodLevelResources, func() {
	f := framework.NewDefaultFramework("pod-level-resources-tests")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		_, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)

		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("not supported on windows -- skipping")
		}

		// skip the test on nodes with cgroupv2 not enabled.
		if !isCgroupv2Node(f, ctx) {
			e2eskipper.Skipf("not supported on cgroupv1 -- skipping")
		}
	})
	podLevelResourcesTests(f)
})

// isCgroupv2Node creates a small pod and check if it is running on a node
// with cgroupv2 enabled.
// TODO: refactor to mark this test with cgroupv2 label, and rather check
// the label in the test job, to tun this test on a node with cgroupv2.
func isCgroupv2Node(f *framework.Framework, ctx context.Context) bool {
	podClient := e2epod.NewPodClient(f)
	cgroupv2Testpod := &v1.Pod{
		ObjectMeta: makeObjectMetadata("cgroupv2-check", f.Namespace.Name),
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:      "cgroupv2-check",
					Image:     imageutils.GetE2EImage(imageutils.BusyBox),
					Command:   cmd,
					Resources: getResourceRequirements(&resourceInfo{CPULim: "1m", MemReq: "1Mi"}),
				},
			},
		},
	}

	pod := podClient.CreateSync(ctx, cgroupv2Testpod)
	defer func() {
		framework.Logf("Deleting %q pod", cgroupv2Testpod.Name)
		delErr := e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
		framework.ExpectNoError(delErr, "failed to delete pod %s", delErr)
	}()

	return e2epod.IsPodOnCgroupv2Node(f, pod)
}

func makeObjectMetadata(name, namespace string) metav1.ObjectMeta {
	return metav1.ObjectMeta{
		Name: "testpod", Namespace: namespace,
		Labels: map[string]string{"time": strconv.Itoa(time.Now().Nanosecond())},
	}
}

type containerInfo struct {
	Name      string
	Resources *resourceInfo
}
type resourceInfo struct {
	CPUReq string
	CPULim string
	MemReq string
	MemLim string
}

func makeContainer(info containerInfo) v1.Container {
	cmd := []string{"/bin/sh", "-c", "sleep 1d"}
	res := getResourceRequirements(info.Resources)
	return v1.Container{
		Name:      info.Name,
		Command:   cmd,
		Resources: res,
		Image:     imageutils.GetE2EImage(imageutils.BusyBox),
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      "sysfscgroup",
				MountPath: mountPath,
			},
		},
	}
}

func getResourceRequirements(info *resourceInfo) v1.ResourceRequirements {
	var res v1.ResourceRequirements
	if info != nil {
		if info.CPUReq != "" || info.MemReq != "" {
			res.Requests = make(v1.ResourceList)
		}
		if info.CPUReq != "" {
			res.Requests[v1.ResourceCPU] = resource.MustParse(info.CPUReq)
		}
		if info.MemReq != "" {
			res.Requests[v1.ResourceMemory] = resource.MustParse(info.MemReq)
		}

		if info.CPULim != "" || info.MemLim != "" {
			res.Limits = make(v1.ResourceList)
		}
		if info.CPULim != "" {
			res.Limits[v1.ResourceCPU] = resource.MustParse(info.CPULim)
		}
		if info.MemLim != "" {
			res.Limits[v1.ResourceMemory] = resource.MustParse(info.MemLim)
		}
	}
	return res
}

func makePod(metadata *metav1.ObjectMeta, podResources *resourceInfo, containers []containerInfo) *v1.Pod {
	var testContainers []v1.Container
	for _, container := range containers {
		testContainers = append(testContainers, makeContainer(container))
	}

	pod := &v1.Pod{
		ObjectMeta: *metadata,

		Spec: v1.PodSpec{
			Containers: testContainers,
			Volumes: []v1.Volume{
				{
					Name: "sysfscgroup",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{Path: cgroupFsPath},
					},
				},
			},
		},
	}

	if podResources != nil {
		res := getResourceRequirements(podResources)
		pod.Spec.Resources = &res
	}

	return pod
}

func verifyPodResources(gotPod v1.Pod, inputInfo, expectedInfo *resourceInfo) {
	ginkgo.GinkgoHelper()
	var expectedResources *v1.ResourceRequirements
	// expectedResources will be nil if pod-level resources are not set in the test
	// case input.
	if inputInfo != nil {
		resourceInfo := getResourceRequirements(expectedInfo)
		expectedResources = &resourceInfo
	}
	gomega.Expect(expectedResources).To(gomega.Equal(gotPod.Spec.Resources))
}

func verifyQoS(gotPod v1.Pod, expectedQoS v1.PodQOSClass) {
	ginkgo.GinkgoHelper()
	gomega.Expect(expectedQoS).To(gomega.Equal(gotPod.Status.QOSClass))
}

// TODO(ndixita): dedup the conversion logic in pod resize test and move to helpers/utils.
func verifyPodCgroups(ctx context.Context, f *framework.Framework, pod *v1.Pod, info *resourceInfo) error {
	ginkgo.GinkgoHelper()
	cmd := fmt.Sprintf("find %s -name '*%s*'", mountPath, strings.ReplaceAll(string(pod.UID), "-", "_"))
	framework.Logf("Namespace %s Pod %s - looking for Pod cgroup directory path: %q", f.Namespace, pod.Name, cmd)
	podCgPath, stderr, err := e2epod.ExecCommandInContainerWithFullOutput(f, pod.Name, pod.Spec.Containers[0].Name, []string{"/bin/sh", "-c", cmd}...)
	if err != nil || len(stderr) > 0 {
		return fmt.Errorf("encountered error while running command: %q, \nerr: %w \nstdErr: %q", cmd, err, stderr)
	}

	expectedResources := getResourceRequirements(info)
	cpuWeightCgPath := fmt.Sprintf("%s/%s", podCgPath, cgroupv2CPUWeight)
	expectedCPUShares := int64(kubecm.MilliCPUToShares(expectedResources.Requests.Cpu().MilliValue()))
	expectedCPUShares = int64(1 + ((expectedCPUShares-2)*9999)/262142)
	// convert cgroup v1 cpu.shares value to cgroup v2 cpu.weight value
	// https://github.com/kubernetes/enhancements/tree/master/keps/sig-node/2254-cgroup-v2#phase-1-convert-from-cgroups-v1-settings-to-v2
	var errs []error
	err = e2epod.VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, cpuWeightCgPath, strconv.FormatInt(expectedCPUShares, 10))
	if err != nil {
		errs = append(errs, fmt.Errorf("failed to verify cpu request cgroup value: %w", err))
	}

	cpuLimCgPath := fmt.Sprintf("%s/%s", podCgPath, cgroupv2CPULimit)
	expectedCPULimits := e2epod.GetCPULimitCgroupExpectations(expectedResources.Limits.Cpu())

	err = e2epod.VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, cpuLimCgPath, expectedCPULimits...)
	if err != nil {
		errs = append(errs, fmt.Errorf("failed to verify cpu limit cgroup value: %w", err))
	}

	memLimCgPath := fmt.Sprintf("%s/%s", podCgPath, cgroupv2MemLimit)
	expectedMemLim := strconv.FormatInt(expectedResources.Limits.Memory().Value(), 10)
	err = e2epod.VerifyCgroupValue(f, pod, pod.Spec.Containers[0].Name, memLimCgPath, expectedMemLim)
	if err != nil {
		errs = append(errs, fmt.Errorf("failed to verify memory limit cgroup value: %w", err))
	}
	return utilerrors.NewAggregate(errs)
}

func podLevelResourcesTests(f *framework.Framework) {
	type expectedPodConfig struct {
		qos v1.PodQOSClass
		// totalPodResources represents the aggregate resource requests
		// and limits for the pod. If pod-level resource specifications
		// are specified, totalPodResources is equal to pod-level resources.
		// Otherwise, it is calculated by aggregating resource requests and
		// limits from all containers within the pod..
		totalPodResources *resourceInfo
	}

	type testCase struct {
		name         string
		podResources *resourceInfo
		containers   []containerInfo
		expected     expectedPodConfig
	}

	tests := []testCase{
		{
			name: "Guaranteed QoS pod with container resources",
			containers: []containerInfo{
				{Name: "c1", Resources: &resourceInfo{CPUReq: "50m", CPULim: "50m", MemReq: "70Mi", MemLim: "70Mi"}},
				{Name: "c2", Resources: &resourceInfo{CPUReq: "70m", CPULim: "70m", MemReq: "50Mi", MemLim: "50Mi"}},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSGuaranteed,
				totalPodResources: &resourceInfo{CPUReq: "120m", CPULim: "120m", MemReq: "120Mi", MemLim: "120Mi"},
			},
		},
		{
			name:         "Guaranteed QoS pod, no container resources",
			podResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			containers:   []containerInfo{{Name: "c1"}, {Name: "c2"}},
			expected: expectedPodConfig{
				qos:               v1.PodQOSGuaranteed,
				totalPodResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			},
		},
		{
			name:         "Guaranteed QoS pod with container resources",
			podResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			containers: []containerInfo{
				{Name: "c1", Resources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"}},
				{Name: "c2", Resources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"}},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSGuaranteed,
				totalPodResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			},
		},
		{
			name:         "Guaranteed QoS pod, 1 container with resources",
			podResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			containers: []containerInfo{
				{Name: "c1", Resources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"}},
				{Name: "c2"},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSGuaranteed,
				totalPodResources: &resourceInfo{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
			},
		},
		{
			name:         "Burstable QoS pod, no container resources",
			podResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			containers: []containerInfo{
				{Name: "c1"},
				{Name: "c2"},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSBurstable,
				totalPodResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			},
		},
		{
			name:         "Burstable QoS pod with container resources",
			podResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			containers: []containerInfo{
				{Name: "c1", Resources: &resourceInfo{CPUReq: "20m", CPULim: "100m", MemReq: "20Mi", MemLim: "100Mi"}},
				{Name: "c2", Resources: &resourceInfo{CPUReq: "30m", CPULim: "100m", MemReq: "30Mi", MemLim: "100Mi"}},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSBurstable,
				totalPodResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			},
		},
		{
			name:         "Burstable QoS pod, 1 container with resources",
			podResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			containers: []containerInfo{
				{Name: "c1", Resources: &resourceInfo{CPUReq: "20m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"}},
				{Name: "c2"},
			},
			expected: expectedPodConfig{
				qos:               v1.PodQOSBurstable,
				totalPodResources: &resourceInfo{CPUReq: "50m", CPULim: "100m", MemReq: "50Mi", MemLim: "100Mi"},
			},
		},
	}

	for _, tc := range tests {
		ginkgo.It(tc.name, func(ctx context.Context) {
			podMetadata := makeObjectMetadata("testpod", f.Namespace.Name)
			testPod := makePod(&podMetadata, tc.podResources, tc.containers)

			ginkgo.By("creating pods")
			podClient := e2epod.NewPodClient(f)
			pod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying pod resources are as expected")
			verifyPodResources(*pod, tc.podResources, tc.expected.totalPodResources)

			ginkgo.By("verifying pod QoS as expected")
			verifyQoS(*pod, tc.expected.qos)

			ginkgo.By("verifying pod cgroup values")
			err := verifyPodCgroups(ctx, f, pod, tc.expected.totalPodResources)
			framework.ExpectNoError(err, "failed to verify pod's cgroup values: %v", err)

			ginkgo.By("verifying containers cgroup limits are same as pod container's cgroup limits")
			err = verifyContainersCgroupLimits(f, pod)
			framework.ExpectNoError(err, "failed to verify containers cgroup values: %v", err)

			ginkgo.By("deleting pods")
			delErr := e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
			framework.ExpectNoError(delErr, "failed to delete pod %s", delErr)
		})
	}
}

func verifyContainersCgroupLimits(f *framework.Framework, pod *v1.Pod) error {
	var errs []error
	for _, container := range pod.Spec.Containers {
		if pod.Spec.Resources != nil && pod.Spec.Resources.Limits.Memory() != nil &&
			container.Resources.Limits.Memory() == nil {
			expectedCgroupMemLimit := strconv.FormatInt(pod.Spec.Resources.Limits.Memory().Value(), 10)
			err := e2epod.VerifyCgroupValue(f, pod, container.Name, fmt.Sprintf("%s/%s", cgroupFsPath, cgroupv2MemLimit), expectedCgroupMemLimit)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to verify memory limit cgroup value: %w", err))
			}
		}

		if pod.Spec.Resources != nil && pod.Spec.Resources.Limits.Cpu() != nil &&
			container.Resources.Limits.Cpu() == nil {
			expectedCPULimits := e2epod.GetCPULimitCgroupExpectations(pod.Spec.Resources.Limits.Cpu())
			err := e2epod.VerifyCgroupValue(f, pod, container.Name, fmt.Sprintf("%s/%s", cgroupFsPath, cgroupv2CPULimit), expectedCPULimits...)
			if err != nil {
				errs = append(errs, fmt.Errorf("failed to verify cpu limit cgroup value: %w", err))
			}
		}
	}
	return utilerrors.NewAggregate(errs)
}
