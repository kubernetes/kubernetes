/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Security Context", func() {
	f := framework.NewDefaultFramework("security-context-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})

	Context("When creating a container with runAsUser", func() {
		makeUserPod := func(podName, image string, command []string, userid int64) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
							SecurityContext: &v1.SecurityContext{
								RunAsUser: &userid,
							},
						},
					},
				},
			}
		}
		createAndWaitUserPod := func(userid int64) {
			podName := fmt.Sprintf("busybox-user-%d-%s", userid, uuid.NewUUID())
			podClient.Create(makeUserPod(podName,
				busyboxImage,
				[]string{"sh", "-c", fmt.Sprintf("test $(id -u) -eq %d", userid)},
				userid,
			))

			podClient.WaitForSuccess(podName, framework.PodStartTimeout)
		}

		/*
		  Release : v1.12
		  Testname: Security Context: runAsUser (id:65534)
		  Description: Container created with runAsUser option, passing an id (id:65534) uses that
		  given id when running the container.
		*/
		It("should run the container with uid 65534 [NodeConformance]", func() {
			createAndWaitUserPod(65534)
		})

		/*
		  Release : v1.12
		  Testname: Security Context: runAsUser (id:0)
		  Description: Container created with runAsUser option, passing an id (id:0) uses that
		  given id when running the container.
		*/
		It("should run the container with uid 0 [NodeConformance]", func() {
			createAndWaitUserPod(0)
		})
	})

	Context("When creating a pod with readOnlyRootFilesystem", func() {
		makeUserPod := func(podName, image string, command []string, readOnlyRootFilesystem bool) *v1.Pod {
			return &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image:   image,
							Name:    podName,
							Command: command,
							SecurityContext: &v1.SecurityContext{
								ReadOnlyRootFilesystem: &readOnlyRootFilesystem,
							},
						},
					},
				},
			}
		}
		createAndWaitUserPod := func(readOnlyRootFilesystem bool) string {
			podName := fmt.Sprintf("busybox-readonly-%v-%s", readOnlyRootFilesystem, uuid.NewUUID())
			podClient.Create(makeUserPod(podName,
				"busybox",
				[]string{"sh", "-c", "touch checkfile"},
				readOnlyRootFilesystem,
			))

			if readOnlyRootFilesystem {
				podClient.WaitForFailure(podName, framework.PodStartTimeout)
			} else {
				podClient.WaitForSuccess(podName, framework.PodStartTimeout)
			}

			return podName
		}

		/*
		  Release : v1.12
		  Testname: Security Context: readOnlyRootFilesystem=true.
		  Description: when a container has configured readOnlyRootFilesystem to true, write operations are not allowed.
		*/
		It("should run the container with readonly rootfs when readOnlyRootFilesystem=true [NodeConformance]", func() {
			createAndWaitUserPod(true)
		})

		/*
		  Release : v1.12
		  Testname: Security Context: readOnlyRootFilesystem=false.
		  Description: when a container has configured readOnlyRootFilesystem to false, write operations are allowed.
		*/
		It("should run the container with writable rootfs when readOnlyRootFilesystem=false [NodeConformance]", func() {
			createAndWaitUserPod(false)
		})
	})
})
