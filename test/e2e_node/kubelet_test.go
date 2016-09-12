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
		It("it should print the output to logs", func() {
			podClient.CreateSync(&api.Pod{
				ObjectMeta: api.ObjectMeta{
					Name: podName,
				},
				Spec: api.PodSpec{
					// Don't restart the Pod since it is expected to exit
					RestartPolicy: api.RestartPolicyNever,
					Containers: []api.Container{
						{
							Image:   ImageRegistry[busyBoxImage],
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

	Context("when scheduling a read only busybox container", func() {
		podName := "busybox-readonly-fs" + string(uuid.NewUUID())
		It("it should not write to root filesystem", func() {
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
							Image:   ImageRegistry[busyBoxImage],
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
