/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
)

var _ = Describe("Secrets", func() {
	var c *client.Client
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		ns_, err := createTestingNS("secrets", c)
		ns = ns_.Name
		expectNoError(err)
	})

	AfterEach(func() {
		// Clean up the namespace if a non-default one was used
		if ns != api.NamespaceDefault {
			By("Cleaning up the namespace")
			err := c.Namespaces().Delete(ns)
			expectNoError(err)
		}
	})

	It("should be consumable from pods", func() {
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
						Name:  "secret-test",
						Image: "kubernetes/mounttest:0.1",
						Args: []string{
							"--file_content=/etc/secret-volume/data-1",
							"--file_mode=/etc/secret-volume/data-1"},
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

		testContainerOutputInNamespace("consume secrets", c, pod, []string{
			"content of file \"/etc/secret-volume/data-1\": value-1",
			"mode of file \"/etc/secret-volume/data-1\": -r--r--r--",
		}, ns)
	})
})
