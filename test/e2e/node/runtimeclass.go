/*
Copyright 2019 The Kubernetes Authors.

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

	"k8s.io/pod-security-admission/api"

	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eruntimeclass "k8s.io/kubernetes/test/e2e/framework/node/runtimeclass"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/scheduling"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("RuntimeClass", func() {
	f := framework.NewDefaultFramework("runtimeclass")
	f.NamespacePodSecurityLevel = api.LevelPrivileged

	ginkgo.It("should reject a Pod requesting a RuntimeClass with conflicting node selector", func(ctx context.Context) {
		labelFooName := "foo-" + string(uuid.NewUUID())

		scheduling := &nodev1.Scheduling{
			NodeSelector: map[string]string{
				labelFooName: "conflict",
			},
		}

		runtimeClass := newRuntimeClass(f.Namespace.Name, "conflict-runtimeclass")
		runtimeClass.Scheduling = scheduling
		rc, err := f.ClientSet.NodeV1().RuntimeClasses().Create(ctx, runtimeClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create RuntimeClass resource")

		pod := e2eruntimeclass.NewRuntimeClassPod(rc.GetName())
		pod.Spec.NodeSelector = map[string]string{
			labelFooName: "bar",
		}
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		if !apierrors.IsForbidden(err) {
			framework.Failf("expected 'forbidden' as error, got instead: %v", err)
		}
	})

	f.It("should run a Pod requesting a RuntimeClass with scheduling with taints", f.WithSerial(), func(ctx context.Context) {
		labelFooName := "foo-" + string(uuid.NewUUID())
		labelFizzName := "fizz-" + string(uuid.NewUUID())

		nodeName := scheduling.GetNodeThatCanRunPod(ctx, f)
		nodeSelector := map[string]string{
			labelFooName:  "bar",
			labelFizzName: "buzz",
		}
		tolerations := []v1.Toleration{
			{
				Key:      labelFooName,
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectNoSchedule,
			},
		}
		scheduling := &nodev1.Scheduling{
			NodeSelector: nodeSelector,
			Tolerations:  tolerations,
		}

		ginkgo.By("Trying to apply a label on the found node.")
		for key, value := range nodeSelector {
			e2enode.AddOrUpdateLabelOnNode(f.ClientSet, nodeName, key, value)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, nodeName, key, value)
			ginkgo.DeferCleanup(e2enode.RemoveLabelOffNode, f.ClientSet, nodeName, key)
		}

		ginkgo.By("Trying to apply taint on the found node.")
		taint := v1.Taint{
			Key:    labelFooName,
			Value:  "bar",
			Effect: v1.TaintEffectNoSchedule,
		}
		e2enode.AddOrUpdateTaintOnNode(ctx, f.ClientSet, nodeName, taint)
		e2enode.ExpectNodeHasTaint(ctx, f.ClientSet, nodeName, &taint)
		ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, f.ClientSet, nodeName, taint)

		ginkgo.By("Trying to create runtimeclass and pod")
		runtimeClass := newRuntimeClass(f.Namespace.Name, "non-conflict-runtimeclass")
		runtimeClass.Scheduling = scheduling
		rc, err := f.ClientSet.NodeV1().RuntimeClasses().Create(ctx, runtimeClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create RuntimeClass resource")

		pod := e2eruntimeclass.NewRuntimeClassPod(rc.GetName())
		pod.Spec.NodeSelector = map[string]string{
			labelFooName: "bar",
		}
		pod = e2epod.NewPodClient(f).Create(ctx, pod)

		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, f.ClientSet, f.Namespace.Name, pod.Name))

		// check that pod got scheduled on specified node.
		scheduledPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(nodeName).To(gomega.Equal(scheduledPod.Spec.NodeName))
		gomega.Expect(nodeSelector).To(gomega.Equal(pod.Spec.NodeSelector))
		gomega.Expect(pod.Spec.Tolerations).To(gomega.ContainElement(tolerations[0]))
	})

	ginkgo.It("should run a Pod requesting a RuntimeClass with scheduling without taints ", func(ctx context.Context) {
		if err := e2eruntimeclass.NodeSupportsPreconfiguredRuntimeClassHandler(ctx, f); err != nil {
			e2eskipper.Skipf("Skipping test as node does not have E2E runtime class handler preconfigured in container runtime config: %v", err)
		}

		labelFooName := "foo-" + string(uuid.NewUUID())
		labelFizzName := "fizz-" + string(uuid.NewUUID())

		nodeName := scheduling.GetNodeThatCanRunPod(ctx, f)
		nodeSelector := map[string]string{
			labelFooName:  "bar",
			labelFizzName: "buzz",
		}
		scheduling := &nodev1.Scheduling{
			NodeSelector: nodeSelector,
		}

		ginkgo.By("Trying to apply a label on the found node.")
		for key, value := range nodeSelector {
			e2enode.AddOrUpdateLabelOnNode(f.ClientSet, nodeName, key, value)
			e2enode.ExpectNodeHasLabel(ctx, f.ClientSet, nodeName, key, value)
			ginkgo.DeferCleanup(e2enode.RemoveLabelOffNode, f.ClientSet, nodeName, key)
		}

		ginkgo.By("Trying to create runtimeclass and pod")
		runtimeClass := newRuntimeClass(f.Namespace.Name, "non-conflict-runtimeclass")
		runtimeClass.Scheduling = scheduling
		rc, err := f.ClientSet.NodeV1().RuntimeClasses().Create(ctx, runtimeClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create RuntimeClass resource")

		pod := e2eruntimeclass.NewRuntimeClassPod(rc.GetName())
		pod.Spec.NodeSelector = map[string]string{
			labelFooName: "bar",
		}
		pod = e2epod.NewPodClient(f).Create(ctx, pod)

		framework.ExpectNoError(e2epod.WaitForPodNotPending(ctx, f.ClientSet, f.Namespace.Name, pod.Name))

		// check that pod got scheduled on specified node.
		scheduledPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(nodeName).To(gomega.Equal(scheduledPod.Spec.NodeName))
		gomega.Expect(nodeSelector).To(gomega.Equal(pod.Spec.NodeSelector))
	})
})

// newRuntimeClass returns a test runtime class.
func newRuntimeClass(namespace, name string) *nodev1.RuntimeClass {
	uniqueName := fmt.Sprintf("%s-%s", namespace, name)
	return runtimeclasstest.NewRuntimeClass(uniqueName, e2eruntimeclass.PreconfiguredRuntimeClassHandler)
}
