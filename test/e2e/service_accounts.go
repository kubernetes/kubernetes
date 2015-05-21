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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/serviceaccount"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("ServiceAccounts", func() {
	var c *client.Client
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		expectNoError(err)
		ns_, err := createTestingNS("service-accounts", c)
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

	It("should mount an API token into pods", func() {
		var tokenName string
		var tokenContent string

		// Standard get, update retry loop
		expectNoError(wait.Poll(time.Millisecond*500, time.Second*10, func() (bool, error) {
			By("getting the auto-created API token")
			tokenSelector := fields.SelectorFromSet(map[string]string{client.SecretType: string(api.SecretTypeServiceAccountToken)})
			secrets, err := c.Secrets(ns).List(labels.Everything(), tokenSelector)
			if err != nil {
				return false, err
			}
			if len(secrets.Items) == 0 {
				return false, nil
			}
			if len(secrets.Items) > 1 {
				return false, fmt.Errorf("Expected 1 token secret, got %d", len(secrets.Items))
			}
			tokenName = secrets.Items[0].Name
			tokenContent = string(secrets.Items[0].Data[api.ServiceAccountTokenKey])
			return true, nil
		}))

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-service-account-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "service-account-test",
						Image: "kubernetes/mounttest:0.1",
						Args: []string{
							fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountTokenKey),
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		testContainerOutputInNamespace("consume service account token", c, pod, []string{
			fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountTokenKey, tokenContent),
		}, ns)
	})
})
