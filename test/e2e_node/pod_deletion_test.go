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
    "k8s.io/kubernetes/test/e2e/framework"
    e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var _ = ginkgo.Describe("Pod Deletion Lifecycle", func() {
    // Set up the testing framework with a specific context
    f := framework.NewDefaultFramework("pod-deletion")
    f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

    ginkgo.It("should handle pod deletion gracefully", func() {
        // Define a Pod specification with a simple busybox container
        podSpec := &v1.Pod{
            ObjectMeta: metav1.ObjectMeta{
                Name: "pod-deletion-test",
            },
            Spec: v1.PodSpec{
                Containers: []v1.Container{
                    {
                        Name:  "busybox-container",
                        Image: "busybox",
                        Command: []string{"sh", "-c", "sleep 3600"},
                    },
                },
            },
        }

        // Create the Pod in the test's namespace
        client := e2epod.NewPodClient(f)
        ginkgo.By("Creating the Pod")
        pod := client.Create(context.TODO(), podSpec)

        // Verify the Pod is running
        ginkgo.By("Ensuring the Pod is in Running state")
        err := e2epod.WaitForPodRunningInNamespace(context.TODO(), f.ClientSet, pod)
        framework.ExpectNoError(err)

        // Proceed to delete the Pod with a grace period
        ginkgo.By("Deleting the Pod gracefully")
        gracePeriod := int64(30)
        err = client.Delete(context.TODO(), pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
        framework.ExpectNoError(err)

        // Verify that the Pod has been deleted from the namespace
        ginkgo.By("Verifying the Pod deletion is successful")
        err = e2epod.WaitForPodNotFoundInNamespace(context.TODO(), f.ClientSet, pod.Name, pod.Namespace, time.Minute)
        framework.ExpectNoError(err)
    })
})



