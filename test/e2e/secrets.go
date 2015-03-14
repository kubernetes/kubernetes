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
	"strings"
	"time"

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

		By(fmt.Sprintf("Creating a pod to consume secret %v", secret.Name))
		// Make a client pod that verifies that it has the service environment variables.
		clientName := "client-secrets-" + string(util.NewUUID())
		clientPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: clientName,
			},
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: volumeName,
						VolumeSource: api.VolumeSource{
							Secret: &api.SecretVolumeSource{
								Target: api.ObjectReference{
									Kind:      "Secret",
									Namespace: ns,
									Name:      name,
								},
							},
						},
					},
				},
				Containers: []api.Container{
					{
						Name:    "catcont",
						Image:   "busybox",
						Command: []string{"sh", "-c", "cat /etc/secret-volume/data-1; sleep 600"},
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

		defer c.Pods(ns).Delete(clientPod.Name)
		if _, err := c.Pods(ns).Create(clientPod); err != nil {
			Failf("Failed to create pod: %v", err)
		}
		// Wait for client pod to complete.
		expectNoError(waitForPodRunning(c, clientPod.Name))

		// Grab its logs.  Get host first.
		clientPodStatus, err := c.Pods(ns).Get(clientPod.Name)
		if err != nil {
			Failf("Failed to get clientPod to know host: %v", err)
		}
		By(fmt.Sprintf("Trying to get logs from host %s pod %s container %s: %v",
			clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, err))
		var logs []byte
		start := time.Now()

		// Sometimes the actual containers take a second to get started, try to get logs for 60s
		for time.Now().Sub(start) < (60 * time.Second) {
			logs, err = c.Get().
				Prefix("proxy").
				Resource("minions").
				Name(clientPodStatus.Status.Host).
				Suffix("containerLogs", ns, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name).
				Do().
				Raw()
			fmt.Sprintf("clientPod logs:%v\n", string(logs))
			By(fmt.Sprintf("clientPod logs:%v\n", string(logs)))
			if strings.Contains(string(logs), "Internal Error") {
				By(fmt.Sprintf("Failed to get logs from host %s pod %s container %s: %v",
					clientPodStatus.Status.Host, clientPodStatus.Name, clientPodStatus.Spec.Containers[0].Name, string(logs)))
				time.Sleep(5 * time.Second)
				continue
			}
			break
		}

		toFind := []string{
			"value-1",
		}

		for _, m := range toFind {
			Expect(string(logs)).To(ContainSubstring(m), "%q in secret data", m)
		}

		// We could try a wget the service from the client pod.  But services.sh e2e test covers that pretty well.
	})
})
