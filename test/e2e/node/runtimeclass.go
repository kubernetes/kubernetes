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

	v1 "k8s.io/api/core/v1"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/scheduling"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

var _ = ginkgo.Describe("[sig-node] RuntimeClass", func() {
	f := framework.NewDefaultFramework("runtimeclass")

	ginkgo.It("should reject a Pod requesting a RuntimeClass with conflicting node selector", func() {
		scheduling := &nodev1beta1.Scheduling{
			NodeSelector: map[string]string{
				"foo": "conflict",
			},
		}

		runtimeClass := newRuntimeClass(f.Namespace.Name, "conflict-runtimeclass", framework.TestContext.ContainerRuntime)
		runtimeClass.Scheduling = scheduling
		rc, err := f.ClientSet.NodeV1beta1().RuntimeClasses().Create(context.TODO(), runtimeClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create RuntimeClass resource")

		pod := e2enode.NewRuntimeClassPod(rc.GetName())
		pod.Spec.NodeSelector = map[string]string{
			"foo": "bar",
		}
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err, "should be forbidden")
		framework.ExpectEqual(apierrors.IsForbidden(err), true, "should be forbidden error")
	})

	ginkgo.It("should run a Pod requesting a RuntimeClass with scheduling [NodeFeature:RuntimeHandler] [Disruptive] ", func() {
		nodeName := scheduling.GetNodeThatCanRunPod(f)
		nodeSelector := map[string]string{
			"foo":  "bar",
			"fizz": "buzz",
		}
		tolerations := []v1.Toleration{
			{
				Key:      "foo",
				Operator: v1.TolerationOpEqual,
				Value:    "bar",
				Effect:   v1.TaintEffectNoSchedule,
			},
		}
		scheduling := &nodev1beta1.Scheduling{
			NodeSelector: nodeSelector,
			Tolerations:  tolerations,
		}

		ginkgo.By("Trying to apply a label on the found node.")
		for key, value := range nodeSelector {
			framework.AddOrUpdateLabelOnNode(f.ClientSet, nodeName, key, value)
			framework.ExpectNodeHasLabel(f.ClientSet, nodeName, key, value)
			defer framework.RemoveLabelOffNode(f.ClientSet, nodeName, key)
		}

		ginkgo.By("Trying to apply taint on the found node.")
		taint := v1.Taint{
			Key:    "foo",
			Value:  "bar",
			Effect: v1.TaintEffectNoSchedule,
		}
		e2enode.AddOrUpdateTaintOnNode(f.ClientSet, nodeName, taint)
		framework.ExpectNodeHasTaint(f.ClientSet, nodeName, &taint)
		defer e2enode.RemoveTaintOffNode(f.ClientSet, nodeName, taint)

		ginkgo.By("Trying to create runtimeclass and pod")
		runtimeClass := newRuntimeClass(f.Namespace.Name, "non-conflict-runtimeclass", framework.TestContext.ContainerRuntime)
		runtimeClass.Scheduling = scheduling
		rc, err := f.ClientSet.NodeV1beta1().RuntimeClasses().Create(context.TODO(), runtimeClass, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create RuntimeClass resource")

		pod := e2enode.NewRuntimeClassPod(rc.GetName())
		pod.Spec.NodeSelector = map[string]string{
			"foo": "bar",
		}
		pod = f.PodClient().Create(pod)

		framework.ExpectNoError(e2epod.WaitForPodNotPending(f.ClientSet, f.Namespace.Name, pod.Name))

		// check that pod got scheduled on specified node.
		scheduledPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(nodeName, scheduledPod.Spec.NodeName)
		framework.ExpectEqual(nodeSelector, pod.Spec.NodeSelector)
		gomega.Expect(pod.Spec.Tolerations).To(gomega.ContainElement(tolerations[0]))
	})
})

// newRuntimeClass returns a test runtime class.
func newRuntimeClass(namespace, name, handler string) *nodev1beta1.RuntimeClass {
	uniqueName := fmt.Sprintf("%s-%s", namespace, name)
	return runtimeclasstest.NewRuntimeClass(uniqueName, e2enode.PreconfiguredRuntimeClassHandler(handler))
}
