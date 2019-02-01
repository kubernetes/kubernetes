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
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"

	. "github.com/onsi/ginkgo"
)

const runtimeClassCRDName = "runtimeclasses.node.k8s.io"

var _ = SIGDescribe("RuntimeClass [Feature:RuntimeClass]", func() {
	f := framework.NewDefaultFramework("runtimeclass")

	BeforeEach(func() {
		createRuntimeClassCRD(f)
	})

	It("should run a Pod requesting a non-existent RuntimeClass", func() {
		rcName := f.Namespace.Name + "-nonexistent"
		pod := createRuntimeClassPod(f, rcName)
		expectPodSuccess(f, pod)
	})

	It("should run a Pod requesting a RuntimeClass with an empty handler", func() {
		rcName := createRuntimeClass(f, "empty-handler", "")
		pod := createRuntimeClassPod(f, rcName)
		expectPodSuccess(f, pod)
	})

	It("should run a Pod requesting a RuntimeClass with an unconfigured handler", func() {
		handler := f.Namespace.Name + "-handler"
		rcName := createRuntimeClass(f, "unconfigured-handler", handler)
		pod := createRuntimeClassPod(f, rcName)
		expectPodSuccess(f, pod)
	})

	It("should run a Pod requesting a deleted RuntimeClass", func() {
		rcName := createRuntimeClass(f, "delete-me", "")
		rcClient := f.NodeAPIClientSet.NodeV1alpha1().RuntimeClasses()

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
		expectPodSuccess(f, pod)
	})

	It("should recover when the RuntimeClass CRD is deleted [Serial] [Slow]", func() {
		By("Deleting the RuntimeClass CRD", func() {
			crds := f.APIExtensionsClientSet.ApiextensionsV1beta1().CustomResourceDefinitions()
			runtimeClassCRD, err := crds.Get(runtimeClassCRDName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to get RuntimeClass CRD %s", runtimeClassCRDName)
			runtimeClassCRDUID := runtimeClassCRD.GetUID()

			err = crds.Delete(runtimeClassCRDName, nil)
			framework.ExpectNoError(err, "failed to delete RuntimeClass CRD %s", runtimeClassCRDName)

			By(fmt.Sprintf("Waiting for the CRD (%s) to disappear", string(runtimeClassCRDUID)))
			framework.ExpectNoError(wait.PollImmediate(framework.Poll, time.Minute, func() (bool, error) {
				crd, err := crds.Get(runtimeClassCRDName, metav1.GetOptions{})
				if errors.IsNotFound(err) {
					return true, nil // done
				}
				if err != nil {
					return true, err // stop wait with error
				}
				// If the UID changed, that means the addon manager has already recreated it.
				if crd.GetUID() != runtimeClassCRDUID {
					By(fmt.Sprintf("A new RuntimeClass CRD (%s) has been created", string(crd.GetUID())))
					return true, nil
				} else {
					return false, nil
				}
			}))

		})

		By("Recreating the RuntimeClass CRD manually")
		createRuntimeClassCRD(f)

		rcName := createRuntimeClass(f, "valid", "")
		pod := createRuntimeClassPod(f, rcName)

		// Before the pod can be run, the RuntimeClass informer must time out, by which time the Kubelet
		// will probably be in a backoff state, so the pod can take a long time to start.
		framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(
			f.ClientSet, pod.Name, f.Namespace.Name))
	})

	// TODO(tallclair): Test an actual configured non-default runtimeHandler.
})

// createRuntimeClassCRD creates CRD for RuntimeClass.
func createRuntimeClassCRD(f *framework.Framework) {
	crdFilePath := "cluster/addons/runtimeclass/runtimeclass_crd.yaml"
	data := testfiles.ReadOrDie(crdFilePath, Fail)
	var crd apiextensionsv1beta1.CustomResourceDefinition
	framework.ExpectNoError(runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), data, &crd),
		"failed to decode the content of the CRD file")

	crds := f.APIExtensionsClientSet.ApiextensionsV1beta1().CustomResourceDefinitions()
	By("Creating CRD for RuntimeClass")
	_, err := crds.Create(&crd)
	if err != nil {
		if errors.IsAlreadyExists(err) {
			By("The RuntimeClass CRD is already created")
		} else {
			framework.ExpectNoError(err, "Failed to create %s CRD", crd.Name)
		}
	}

	var crdUID string
	By("Waiting up to 5 minutes for the RuntimeClass CRD to be established")
	framework.ExpectNoError(wait.PollImmediate(
		500*time.Millisecond, 5*time.Minute, func() (bool, error) {
			crd, err := crds.Get(crd.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			for _, cond := range crd.Status.Conditions {
				switch cond.Type {
				case apiextensionsv1beta1.Established:
					if cond.Status == apiextensionsv1beta1.ConditionTrue {
						crdUID = string(crd.GetUID())
						return true, nil
					}
				}
			}
			return false, nil
		}))
	By(fmt.Sprintf("The RuntimeClass CRD (%s) is established", crdUID))
}

// createRuntimeClass generates a RuntimeClass with the desired handler and a "namespaced" name,
// synchronously creates it, and returns the generated name.
func createRuntimeClass(f *framework.Framework, name, handler string) string {
	uniqueName := fmt.Sprintf("%s-%s", f.Namespace.Name, name)
	rc := runtimeclasstest.NewRuntimeClass(uniqueName, handler)
	rc, err := f.NodeAPIClientSet.NodeV1alpha1().RuntimeClasses().Create(rc)
	framework.ExpectNoError(err, "failed to create RuntimeClass resource")
	By(fmt.Sprintf("RuntimeClass %s is created", uniqueName))
	return rc.GetName()
}

// createRuntimeClass creates a test pod with the given runtimeClassName.
func createRuntimeClassPod(f *framework.Framework, runtimeClassName string) *v1.Pod {
	By(fmt.Sprintf("Creating pod using RuntimeClass %s", runtimeClassName))
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
