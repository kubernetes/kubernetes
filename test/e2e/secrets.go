/*
Copyright 2014 Google Inc. All rights reserved.

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

package e2e

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("Secrets", func() {
	var c *client.Client

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
	})

	It("should be consumable from pods", func() {
		ns := api.NamespaceDefault
		name := "secret-test-" + string(util.NewUUID())
		volumeName := "secret-volume"
		volumeMountPath := "/etc/secret-volume"

		secret := &api.Secret{
			ObjectMeta: api.ObjectMeta{
				Namespace: ns,
				Name:      name,
			},
			Data: map[string][]byte{
				"data-1": []byte("value-1\n"),
				"data-2": []byte("value-2\n"),
				"data-3": []byte("value-3\n"),
			},
		}

		By(fmt.Sprintf("Creating secret with name %s", secret.Name))
		defer func() {
			By("Cleaning up the secret")
			if err := c.Secrets(ns).Delete(secret.Name); err != nil {
				Failf("unable to delete secret %v: %v", secret.Name, err)
			}
		}()
		var err error
		if secret, err = c.Secrets(ns).Create(secret); err != nil {
			Failf("unable to create test secret %s: %v", secret.Name, err)
		}

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-secrets-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							Secret: &api.SecretVolumeSource{
								SecretName: name,
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:    "catcont",
						Image:   "gcr.io/google_containers/busybox",
						Command: []string{"sh", "-c", "cat /etc/secret-volume/data-1"},
						VolumeMounts: []api.VolumeMount{
							{
								Name:      volumeName,
								MountPath: volumeMountPath,
								ReadOnly:  true,
							},
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutput("consume secrets", c, pod, []string{
			"value-1",
		})
	})
})
