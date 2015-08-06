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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"

	. "github.com/onsi/ginkgo"
)

var _ = Describe("ServiceAccounts", func() {
	f := NewFramework("svcaccounts")

	It("should mount an API token into pods", func() {
		var tokenName string
		var tokenContent string
		var rootCAContent string

		// Standard get, update retry loop
		expectNoError(wait.Poll(time.Millisecond*500, time.Second*10, func() (bool, error) {
			By("getting the auto-created API token")
			tokenSelector := fields.SelectorFromSet(map[string]string{client.SecretType: string(api.SecretTypeServiceAccountToken)})
			secrets, err := f.Client.Secrets(f.Namespace.Name).List(labels.Everything(), tokenSelector)
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
			rootCAContent = string(secrets.Items[0].Data[api.ServiceAccountRootCAKey])
			return true, nil
		}))

		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: "pod-service-account-" + string(util.NewUUID()),
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "token-test",
						Image: "gcr.io/google_containers/mounttest:0.2",
						Args: []string{
							fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountTokenKey),
						},
					},
					{
						Name:  "root-ca-test",
						Image: "gcr.io/google_containers/mounttest:0.2",
						Args: []string{
							fmt.Sprintf("--file_content=%s/%s", serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountRootCAKey),
						},
					},
				},
				RestartPolicy: api.RestartPolicyNever,
			},
		}

		f.TestContainerOutput("consume service account token", pod, 0, []string{
			fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountTokenKey, tokenContent),
		})
		f.TestContainerOutput("consume service account root CA", pod, 1, []string{
			fmt.Sprintf(`content of file "%s/%s": %s`, serviceaccount.DefaultAPITokenMountPath, api.ServiceAccountRootCAKey, rootCAContent),
		})
	})
})
