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

package podlifecycle

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/utils/image"
)

var _ = ginkgo.Describe("Pod Lifecycle", func() {
	// Initialize the testing framework
	f := framework.NewDefaultFramework("pod-creation")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should create and delete a Pod as expected", func() {
		// Set up the Pod definition
		podSpec := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-creation-test",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "busybox-container",
						Image: image.GetE2EImage(image.BusyBox),
						Command: []string{"sh", "-c", "sleep 3600"},
					},
				},
			},
		}

		// Apply any necessary Pod setup
		preparePod(podSpec)

		// Create the Pod and ensure it's running
		client := e2epod.NewPodClient(f)
		ginkgo.By("Creating the Pod")
		createdPod := client.Create(context.TODO(), podSpec)

		ginkgo.By("Verifying the Pod is running")
		err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, createdPod)
		framework.ExpectNoError(err)

		// Clean up: delete the Pod and verify it’s gone
		ginkgo.By("Cleaning up the Pod")
		err = client.Delete(context.TODO(), createdPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Confirming the Pod has been deleted")
		err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, createdPod.Name, createdPod.Namespace, time.Minute)
		framework.ExpectNoError(err)
	})
})
