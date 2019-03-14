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

package node

import (
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"

	. "github.com/onsi/ginkgo"
)

var _ = SIGDescribe("RuntimeClass", func() {
	f := framework.NewDefaultFramework("runtimeclass")

	It("should reject a Pod requesting a non-existent RuntimeClass", func() {
		rcName := f.Namespace.Name + "-nonexistent"
		pod := createRuntimeClassPod(f, rcName)
		expectSandboxFailureEvent(f, pod, fmt.Sprintf("\"%s\" not found", rcName))
	})

	It("should reject a Pod requesting a RuntimeClass with an unconfigured handler", func() {
		handler := f.Namespace.Name + "-handler"
		rcName := createRuntimeClass(f, "unconfigured-handler", handler)
		pod := createRuntimeClassPod(f, rcName)
		expectSandboxFailureEvent(f, pod, handler)
	})

	It("should reject a Pod requesting a deleted RuntimeClass", func() {
		rcName := createRuntimeClass(f, "delete-me", "runc")
		rcClient := f.ClientSet.NodeV1beta1().RuntimeClasses()

		By("Deleting RuntimeClass "+rcName, func() {
			err := rcClient.Delete(rcName, nil)
			framework.ExpectNoError(err, "failed to delete RuntimeClass %s", rcName)

			By("Waiting for the RuntimeClass to disappear")
			framework.ExpectNoError(wait.PollImmediate(framework.Poll, time.Minute, func() (bool, error) {
				_, err := rcClient.Get(rcName, metav1.GetOptions{})
				if errors.IsNotFound(err) {
					return true, nil // done
				}
				if err != nil {
					return true, err // stop wait with error
				}
				return false, nil
			}))
		})

		pod := createRuntimeClassPod(f, rcName)
		expectSandboxFailureEvent(f, pod, fmt.Sprintf("\"%s\" not found", rcName))
	})

	// TODO(tallclair): Test an actual configured non-default runtimeHandler.
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

// createRuntimeClass creates a test pod with the given runtimeClassName.
func createRuntimeClassPod(f *framework.Framework, runtimeClassName string) *v1.Pod {
	pod := &v1.Pod{
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
	return f.PodClient().Create(pod)
}

// expectPodSuccess waits for the given pod to terminate successfully.
func expectPodSuccess(f *framework.Framework, pod *v1.Pod) {
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespace(
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
	framework.ExpectNoError(framework.WaitTimeoutForPodEvent(
		f.ClientSet, pod.Name, f.Namespace.Name, eventSelector, msg, framework.PodEventTimeout))
}
