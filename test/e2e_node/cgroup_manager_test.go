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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("Kubelet Cgroup Manager [Skip]", func() {
	f := framework.NewDefaultFramework("kubelet-cgroup-manager")
	Describe("QOS containers", func() {
		Context("On enabling QOS cgroup hierarchy", func() {
			It("Top level QoS containers should have been created", func() {
				// return fast
				if !framework.TestContext.CgroupsPerQOS {
					return
				}
				podName := "qos-pod" + string(uuid.NewUUID())
				contName := "qos-container" + string(uuid.NewUUID())
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name: podName,
					},
					Spec: api.PodSpec{
						// Don't restart the Pod since it is expected to exit
						RestartPolicy: api.RestartPolicyNever,
						Containers: []api.Container{
							{
								Image:   ImageRegistry[busyBoxImage],
								Name:    contName,
								Command: []string{"sh", "-c", "if [ -d /tmp/memory/Burstable ] && [ -d /tmp/memory/BestEffort ]; then exit 0; else exit 1; fi"},
								VolumeMounts: []api.VolumeMount{
									{
										Name:      "sysfscgroup",
										MountPath: "/tmp",
									},
								},
							},
						},
						Volumes: []api.Volume{
							{
								Name: "sysfscgroup",
								VolumeSource: api.VolumeSource{
									HostPath: &api.HostPathVolumeSource{Path: "/sys/fs/cgroup"},
								},
							},
						},
					},
				}
				podClient := f.PodClient()
				podClient.Create(pod)
				err := framework.WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
				Expect(err).NotTo(HaveOccurred())
			})
		})
	})
})
