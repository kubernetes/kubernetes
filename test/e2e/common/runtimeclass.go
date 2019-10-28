/*
Copyright 2018 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"

	"github.com/onsi/ginkgo"
)

var _ = ginkgo.Describe("[sig-node] RuntimeClass", func() {
	f := framework.NewDefaultFramework("runtimeclass")

	ginkgo.It("should reject a Pod requesting a non-existent RuntimeClass", func() {
		rcName := f.Namespace.Name + "-nonexistent"
		expectPodRejection(f, newRuntimeClassPod(rcName))
	})

	ginkgo.It("should reject a Pod requesting a RuntimeClass with an unconfigured handler", func() {
		handler := f.Namespace.Name + "-handler"
		rcName := createRuntimeClass(f, "unconfigured-handler", handler)
		pod := f.PodClient().Create(newRuntimeClassPod(rcName))
		expectSandboxFailureEvent(f, pod, handler)
	})

	// This test requires that the PreconfiguredRuntimeHandler has already been set up on nodes.
	ginkgo.It("should run a Pod requesting a RuntimeClass with a configured handler [NodeFeature:RuntimeHandler]", func() {
		// The built-in docker runtime does not support configuring runtime handlers.
		handler := framework.PreconfiguredRuntimeClassHandler()

		rcName := createRuntimeClass(f, "preconfigured-handler", handler)
		pod := f.PodClient().Create(newRuntimeClassPod(rcName))
		expectPodSuccess(f, pod)
	})

	ginkgo.It("should reject a Pod requesting a deleted RuntimeClass", func() {
		rcName := createRuntimeClass(f, "delete-me", "runc")
		rcClient := f.ClientSet.NodeV1beta1().RuntimeClasses()

		ginkgo.By("Deleting RuntimeClass "+rcName, func() {
			err := rcClient.Delete(rcName, nil)
			framework.ExpectNoError(err, "failed to delete RuntimeClass %s", rcName)

			ginkgo.By("Waiting for the RuntimeClass to disappear")
			framework.ExpectNoError(wait.PollImmediate(framework.Poll, time.Minute, func() (bool, error) {
				_, err := rcClient.Get(rcName, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return true, nil // done
				}
				if err != nil {
					return true, err // stop wait with error
				}
				return false, nil
			}))
		})

		expectPodRejection(f, newRuntimeClassPod(rcName))
	})
})

// createRuntimeClass generates a RuntimeClass with the desired handler and a "namespaced" name,
// synchronously creates it, and returns the generated name.
func createRuntimeClass(f *framework.Framework, name, handler string) string {
	uniqueName := fmt.Sprintf("%s-%s", f.Namespace.Name, name)
	rc := runtimeclasstest.NewRuntimeClass(uniqueName, handler)
	rc, err := f.ClientSet.NodeV1beta1().RuntimeClasses().Create(rc)
	framework.ExpectNoError(err, "failed to create RuntimeClass resource")
	return rc.GetName()
}

// newRuntimeClassPod generates a test pod with the given runtimeClassName.
func newRuntimeClassPod(runtimeClassName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("test-runtimeclass-%s-", runtimeClassName),
		},
		Spec: v1.PodSpec{
			RuntimeClassName: &runtimeClassName,
			Containers: []v1.Container{{
				Name:    "test",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"true"},
			}},
			RestartPolicy:                v1.RestartPolicyNever,
			AutomountServiceAccountToken: utilpointer.BoolPtr(false),
		},
	}
}

func expectPodRejection(f *framework.Framework, pod *v1.Pod) {
	// The Node E2E doesn't run the RuntimeClass admission controller, so we expect the rejection to
	// happen by the Kubelet.
	if framework.TestContext.NodeE2E {
		pod = f.PodClient().Create(pod)
		expectSandboxFailureEvent(f, pod, fmt.Sprintf("\"%s\" not found", *pod.Spec.RuntimeClassName))
	} else {
		_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
		framework.ExpectError(err, "should be forbidden")
		framework.ExpectEqual(apierrors.IsForbidden(err), true, "should be forbidden error")
	}
}

// expectPodSuccess waits for the given pod to terminate successfully.
func expectPodSuccess(f *framework.Framework, pod *v1.Pod) {
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(
		f.ClientSet, pod.Name, f.Namespace.Name))
}

// expectSandboxFailureEvent polls for an event with reason "FailedCreatePodSandBox" containing the
// expected message string.
func expectSandboxFailureEvent(f *framework.Framework, pod *v1.Pod, msg string) {
	eventSelector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   events.FailedCreatePodSandBox,
	}.AsSelector().String()
	framework.ExpectNoError(e2eevents.WaitTimeoutForEvent(
		f.ClientSet, f.Namespace.Name, eventSelector, msg, framework.PodEventTimeout))
}
