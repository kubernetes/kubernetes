/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

var _ = Describe("Kubelet", func() {
	var cl *client.Client
	BeforeEach(func() {
		// Setup the apiserver client
		cl = client.NewOrDie(&client.Config{Host: *apiServerAddress})
	})

	Describe("pod scheduling", func() {
		Context("when scheduling a busybox command in a pod", func() {
			It("it should return succes", func() {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "busybox",
						Namespace: api.NamespaceDefault,
					},
					Spec: api.PodSpec{
						// Force the Pod to schedule to the node without a scheduler running
						NodeName: *nodeName,
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:           "gcr.io/google_containers/busybox",
								Name:            "busybox",
								Command:         []string{"echo", "'Hello World'"},
								ImagePullPolicy: api.PullIfNotPresent,
							},
						},
					},
				}
				_, err := cl.Pods(api.NamespaceDefault).Create(pod)
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})

			It("it should print the output to logs", func() {
				errs := Retry(time.Minute, time.Second*4, func() error {
					rc, err := cl.Pods(api.NamespaceDefault).GetLogs("busybox", &api.PodLogOptions{}).Stream()
					if err != nil {
						return err
					}
					defer rc.Close()
					buf := new(bytes.Buffer)
					buf.ReadFrom(rc)
					if buf.String() != "'Hello World'\n" {
						return fmt.Errorf("Expected %s to match 'Hello World'", buf.String())
					}
					return nil
				})
				Expect(errs).To(BeEmpty(), fmt.Sprintf("Failed to get Logs"))
			})

			It("it should be possible to delete", func() {
				err := cl.Pods(api.NamespaceDefault).Delete("busybox", &api.DeleteOptions{})
				Expect(err).To(BeNil(), fmt.Sprintf("Error creating Pod %v", err))
			})
		})
	})
})
