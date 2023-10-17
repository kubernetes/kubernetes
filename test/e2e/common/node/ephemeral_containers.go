/*
Copyright 2021 The Kubernetes Authors.

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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("Ephemeral Containers", framework.WithNodeConformance(), func() {
	f := framework.NewDefaultFramework("ephemeral-containers-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	var podClient *e2epod.PodClient
	ginkgo.BeforeEach(func() {
		podClient = e2epod.NewPodClient(f)
	})

	// Release: 1.25
	// Testname: Ephemeral Container Creation
	// Description: Adding an ephemeral container to pod.spec MUST result in the container running.
	framework.ConformanceIt("will start an ephemeral container in an existing pod", func(ctx context.Context) {
		ginkgo.By("creating a target pod")
		pod := podClient.CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "ephemeral-containers-target-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
					},
				},
			},
		})

		ginkgo.By("adding an ephemeral container")
		ecName := "debugger"
		ec := &v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    ecName,
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: e2epod.GenerateScriptCmd("while true; do echo polo; sleep 2; done"),
				Stdin:   true,
				TTY:     true,
			},
		}
		err := podClient.AddEphemeralContainerSync(ctx, pod, ec, time.Minute)
		framework.ExpectNoError(err, "Failed to patch ephemeral containers in pod %q", e2epod.FormatPod(pod))

		ginkgo.By("checking pod container endpoints")
		// Can't use anything depending on kubectl here because it's not available in the node test environment
		output := e2epod.ExecCommandInContainer(f, pod.Name, ecName, "/bin/echo", "marco")
		gomega.Expect(output).To(gomega.ContainSubstring("marco"))
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod.Namespace, pod.Name, ecName)
		framework.ExpectNoError(err, "Failed to get logs for pod %q ephemeral container %q", e2epod.FormatPod(pod), ecName)
		gomega.Expect(log).To(gomega.ContainSubstring("polo"))
	})

	/*
		Release: v1.28
		Testname: Ephemeral Container, update ephemeral containers
		Description: Adding an ephemeral container to pod.spec MUST result in the container
		running. There MUST now be only one ephemeral container found. Updating the pod with
		another ephemeral container MUST succeed. There MUST now be two ephemeral containers
		found.
	*/
	framework.ConformanceIt("should update the ephemeral containers in an existing pod", func(ctx context.Context) {
		ginkgo.By("creating a target pod")
		pod := podClient.CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "ephemeral-containers-target-pod"},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "test-container-1",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"/bin/sleep"},
						Args:    []string{"10000"},
					},
				},
			},
		})

		ginkgo.By("adding an ephemeral container")
		ecName := "debugger"
		ec := &v1.EphemeralContainer{
			EphemeralContainerCommon: v1.EphemeralContainerCommon{
				Name:    ecName,
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: e2epod.GenerateScriptCmd("while true; do echo polo; sleep 2; done"),
				Stdin:   true,
				TTY:     true,
			},
		}
		err := podClient.AddEphemeralContainerSync(ctx, pod, ec, time.Minute)
		framework.ExpectNoError(err, "Failed to patch ephemeral containers in pod %q", e2epod.FormatPod(pod))

		ginkgo.By("checking pod container endpoints")
		// Can't use anything depending on kubectl here because it's not available in the node test environment
		output := e2epod.ExecCommandInContainer(f, pod.Name, ecName, "/bin/echo", "marco")
		gomega.Expect(output).To(gomega.ContainSubstring("marco"))
		log, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod.Namespace, pod.Name, ecName)
		framework.ExpectNoError(err, "Failed to get logs for pod %q ephemeral container %q", e2epod.FormatPod(pod), ecName)
		gomega.Expect(log).To(gomega.ContainSubstring("polo"))

		ginkgo.By(fmt.Sprintf("checking pod %q has only one ephemeralcontainer", pod.Name))
		podResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
		unstruct, err := f.DynamicClient.Resource(podResource).Namespace(f.Namespace.Name).Get(ctx, "ephemeral-containers-target-pod", metav1.GetOptions{}, "ephemeralcontainers")
		framework.ExpectNoError(err, "can't get ephermalcontainers")
		verifyPod, err := unstructuredToPod(unstruct)
		framework.ExpectNoError(err, "Getting the %q pod's ephemeralcontainers", verifyPod.Name)
		gomega.Expect(verifyPod.Spec.EphemeralContainers).To(gomega.HaveLen(1), "checking ephemeralContainer count")

		ginkgo.By(fmt.Sprintf("adding another ephemeralcontainer to pod %q", pod.Name))
		var podToUpdate *v1.Pod
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			podToUpdate, err = podClient.Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to retrieve pod %s", pod.Name)

			podToUpdate.Spec.EphemeralContainers = append(podToUpdate.Spec.EphemeralContainers, v1.EphemeralContainer{
				EphemeralContainerCommon: v1.EphemeralContainerCommon{
					Name:                     "debugger2",
					Image:                    imageutils.GetE2EImage(imageutils.Agnhost),
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				},
			})
			_, err = podClient.UpdateEphemeralContainers(context.TODO(), pod.Name, podToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "Failed to update ephemeral container.")

		ginkgo.By(fmt.Sprintf("checking pod %q has only two ephemeralcontainers", pod.Name))
		unstruct, err = f.DynamicClient.Resource(podResource).Namespace(f.Namespace.Name).Get(ctx, "ephemeral-containers-target-pod", metav1.GetOptions{}, "ephemeralcontainers")
		framework.ExpectNoError(err, "can't get ephermalcontainers")
		verifyPod, err = unstructuredToPod(unstruct)
		framework.ExpectNoError(err, "Getting the %q pod's ephemeralcontainers", verifyPod.Name)
		gomega.Expect(verifyPod.Spec.EphemeralContainers).To(gomega.HaveLen(2), "checking ephemeralContainer count")
	})
})

func unstructuredToPod(obj *unstructured.Unstructured) (*v1.Pod, error) {
	json, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return nil, err
	}
	p := &v1.Pod{}
	err = runtime.DecodeInto(clientscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), json, p)

	return p, err
}
