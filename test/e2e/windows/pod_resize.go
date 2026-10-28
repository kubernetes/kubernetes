/*
Copyright 2026 The Kubernetes Authors.

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

package windows

import (
	"context"
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	helpers "k8s.io/component-helpers/resource"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = sigDescribe("In-place pod resize", framework.WithSerial(), framework.WithFeatureGate(features.InPlacePodVerticalScaling), framework.WithFeatureGate(features.WindowsInPlacePodResize), skipUnlessWindows(func() {
	f := framework.NewDefaultFramework("windows-pod-resize")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should actuate CPU and memory limits without restarting a process-isolated container", func(ctx context.Context) {
		original := []v1.ResourceRequirements{{
			Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("192Mi")},
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1"), v1.ResourceMemory: resource.MustParse("256Mi")},
		}}
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err, "list schedulable nodes")
		if len(nodeList.Items) == 0 {
			e2eskipper.Skipf("requires a ready schedulable Windows node")
		}
		e2eskipper.SkipUnlessFeatureGateEnabled(features.WindowsInPlacePodResize)
		pod := newWindowsResizePod(f.Namespace.Name, "windows-resize", original)
		podClient := e2epod.NewPodClient(f)
		pod = podClient.CreateSync(ctx, pod)
		initialRestarts := restartCount(pod, pod.Spec.Containers[0].Name)
		originalGeneration := pod.Generation

		desired := []v1.ResourceRequirements{{
			Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("750m"), v1.ResourceMemory: resource.MustParse("384Mi")},
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("1500m"), v1.ResourceMemory: resource.MustParse("512Mi")},
		}}
		patch := resizePatch(pod, original, desired)
		ginkgo.By("patching the pod /resize subresource")
		patched, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Patch(ctx, pod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "resize")
		framework.ExpectNoError(err, "patch Windows pod resize")
		gomega.Expect(patched.Generation).To(gomega.BeNumerically(">", originalGeneration))

		ginkgo.By("waiting for kubelet to report the resize as actuated")
		var resized *v1.Pod
		framework.ExpectNoError(framework.Gomega().Eventually(ctx, func(ctx context.Context) (*v1.Pod, error) {
			return f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		}).WithTimeout(f.Timeouts.PodStart).Should(framework.MakeMatcher(func(p *v1.Pod) (func() string, error) {
			resized = p
			if podResizeActuated(p, desired[0]) {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("pod generation %d, observed generation %d, resize status not actuated", p.Generation, p.Status.ObservedGeneration)
			}, nil
		})), "wait for Windows pod resize actuation")

		gomega.Expect(resized.Status.ContainerStatuses).NotTo(gomega.BeEmpty())
		status := resized.Status.ContainerStatuses[0]
		gomega.Expect(status.Resources).NotTo(gomega.BeNil(), "resize status must report container resources")
		gomega.Expect(status.Resources.Limits.Cpu().MilliValue()).To(gomega.Equal(desired[0].Limits.Cpu().MilliValue()))
		gomega.Expect(status.Resources.Limits.Memory().Value()).To(gomega.Equal(desired[0].Limits.Memory().Value()))
		gomega.Expect(status.RestartCount).To(gomega.Equal(initialRestarts), "CPU/memory ResizePolicy NotRequired must not restart the container")
	})
}))

func newWindowsResizePod(namespace, name string, resources []v1.ResourceRequirements) *v1.Pod {
	noRestart := v1.NotRequired
	container := v1.Container{
		Name:      "resize",
		Image:     imageutils.GetPauseImageName(),
		Command:   []string{"cmd", "/c", "ping", "-t", "127.0.0.1"},
		Resources: resources[0],
		ResizePolicy: []v1.ContainerResizePolicy{
			{ResourceName: v1.ResourceCPU, RestartPolicy: noRestart},
			{ResourceName: v1.ResourceMemory, RestartPolicy: noRestart},
		},
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.PodSpec{
			OS:           &v1.PodOS{Name: v1.Windows},
			NodeSelector: map[string]string{"kubernetes.io/os": "windows"},
			Containers:   []v1.Container{container},
		},
	}
}

func resizePatch(pod *v1.Pod, original, desired []v1.ResourceRequirements) []byte {
	before := pod.DeepCopy()
	after := pod.DeepCopy()
	before.Spec.Containers[0].Resources = original[0]
	after.Spec.Containers[0].Resources = desired[0]
	beforeJSON, err := json.Marshal(before)
	framework.ExpectNoError(err)
	afterJSON, err := json.Marshal(after)
	framework.ExpectNoError(err)
	patch, err := strategicpatch.CreateTwoWayMergePatch(beforeJSON, afterJSON, v1.Pod{})
	framework.ExpectNoError(err)
	return patch
}

func podResizeActuated(pod *v1.Pod, desired v1.ResourceRequirements) bool {
	if pod.Status.ObservedGeneration < pod.Generation || helpers.IsPodResizeInfeasible(pod) {
		return false
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == v1.PodResizePending || condition.Type == v1.PodResizeInProgress {
			return false
		}
	}
	if len(pod.Status.ContainerStatuses) == 0 || pod.Status.ContainerStatuses[0].Resources == nil {
		return false
	}
	got := pod.Status.ContainerStatuses[0].Resources
	return got.Limits.Cpu().Cmp(*desired.Limits.Cpu()) == 0 && got.Limits.Memory().Cmp(*desired.Limits.Memory()) == 0
}

func restartCount(pod *v1.Pod, name string) int32 {
	for _, status := range pod.Status.ContainerStatuses {
		if status.Name == name {
			return status.RestartCount
		}
	}
	return -1
}
