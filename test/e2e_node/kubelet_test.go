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
	"bytes"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apiUnversioned "k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet", func() {
	f := framework.NewDefaultFramework("kubelet-test")
	var podClient *framework.PodClient
	BeforeEach(func() {
		podClient = f.PodClient()
	})
	Context("when scheduling a busybox command in a pod", func() {
		podName := "busybox-scheduling-" + string(uuid.NewUUID())
		It("it should print the output to logs [Conformance]", func() {
			podClient.CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   "gcr.io/google_containers/busybox:1.24",
							Name:    podName,
							Command: []string{"sh", "-c", "echo 'Hello World' ; sleep 240"},
						},
					},
				},
			})
			Eventually(func() string {
				sinceTime := apiUnversioned.NewTime(time.Now().Add(time.Duration(-1 * time.Hour)))
				rc, err := podClient.GetLogs(podName, &api.PodLogOptions{SinceTime: &sinceTime}).Stream()
				if err != nil {
					return ""
				}
				defer rc.Close()
				buf := new(bytes.Buffer)
				buf.ReadFrom(rc)
				return buf.String()
			}, time.Minute, time.Second*4).Should(Equal("Hello World\n"))
		})
	})
	Context("when scheduling a busybox command that always fails in a pod", func() {
		var podName string

		BeforeEach(func() {
			podName = "bin-false" + string(uuid.NewUUID())
			podClient.Create(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   "gcr.io/google_containers/busybox:1.24",
							Name:    podName,
							Command: []string{"/bin/false"},
						},
					},
				},
			})
		})

		It("should have an error terminated reason", func() {
			Eventually(func() error {
				podData, err := podClient.Get(podName)
				if err != nil {
					return err
				}
				if len(podData.Status.ContainerStatuses) != 1 {
					return fmt.Errorf("expected only one container in the pod %q", podName)
				}
				contTerminatedState := podData.Status.ContainerStatuses[0].State.Terminated
				if contTerminatedState == nil {
					return fmt.Errorf("expected state to be terminated. Got pod status: %+v", podData.Status)
				}
				if contTerminatedState.Reason != "Error" {
					return fmt.Errorf("expected terminated state reason to be error. Got %+v", contTerminatedState)
				}
				return nil
			}, time.Minute, time.Second*4).Should(BeNil())
		})

		It("should be possible to delete", func() {
			err := podClient.Delete(podName, &api.DeleteOptions{})
			Expect(err).To(BeNil(), fmt.Sprintf("Error deleting Pod %v", err))
		})
	})
	Context("when scheduling a read only busybox container", func() {
		podName := "busybox-readonly-fs" + string(uuid.NewUUID())
		It("it should not write to root filesystem [Conformance]", func() {
			isReadOnly := true
			podClient.CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   "gcr.io/google_containers/busybox:1.24",
							Name:    podName,
							Command: []string{"sh", "-c", "echo test > /file; sleep 240"},
							SecurityContext: &api.SecurityContext{
								ReadOnlyRootFilesystem: &isReadOnly,
							},
						},
					},
				},
			})
			Eventually(func() string {
				rc, err := podClient.GetLogs(podName, &api.PodLogOptions{}).Stream()
				if err != nil {
					return ""
				}
				defer rc.Close()
				buf := new(bytes.Buffer)
				buf.ReadFrom(rc)
				return buf.String()
			}, time.Minute, time.Second*4).Should(Equal("sh: can't create /file: Read-only file system\n"))
		})
	})
})
