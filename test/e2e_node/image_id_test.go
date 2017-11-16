/*
Copyright 2016 The Kubernetes Authors.

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

package e2e_node

import (
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/davecgh/go-spew/spew"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("ImageID", func() {

	busyBoxImage := "gcr.io/google_containers/busybox@sha256:4bdd623e848417d96127e16037743f0cd8b528c026e9175e22a84f639eca58ff"

	f := framework.NewDefaultFramework("image-id-test")

	It("should be set to the manifest digest (from RepoDigests) when available", func() {
		podDesc := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod-with-repodigest",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{{
					Name:    "test",
					Image:   busyBoxImage,
					Command: []string{"sh"},
				}},
				RestartPolicy: v1.RestartPolicyNever,
			},
		}

		pod := f.PodClient().Create(podDesc)

		framework.ExpectNoError(framework.WaitTimeoutForPodNoLongerRunningInNamespace(
			f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout))
		runningPod, err := f.PodClient().Get(pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		status := runningPod.Status

		if len(status.ContainerStatuses) == 0 {
			framework.Failf("Unexpected pod status; %s", spew.Sdump(status))
			return
		}

		Expect(status.ContainerStatuses[0].ImageID).To(ContainSubstring(busyBoxImage))
	})
})
