/*
Copyright 2015 The Kubernetes Authors.

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

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	FederatedSecretName = "federated-secret"
)

// Create/delete secret api objects
var _ = framework.KubeDescribe("Federation secrets [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federated-secret")
	clusterClientSet := make(map[string]*release_1_3.Clientset)

	Describe("Secret objects", func() {

		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			clusters := buildClustersOrFail_14(f)
			for _, cluster := range clusters {
				if _, found := clusterClientSet[cluster.Name]; !found {
					clientset := createClientsetForCluster(*cluster, 1, "e2e-test")
					clusterClientSet[cluster.Name] = clientset
				}
			}
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)
			nsName := f.FederationNamespace.Name
			deleteAllTestSecrets(
				f.FederationClientset_1_4.Core().Secrets(nsName).List,
				f.FederationClientset_1_4.Core().Secrets(nsName).Delete,
				false)
			for _, clientset := range clusterClientSet {
				deleteAllTestSecrets(
					clientset.Core().Secrets(nsName).List,
					clientset.Core().Secrets(nsName).Delete,
					false)
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)

			nsName := f.FederationNamespace.Name

			secret := &v1.Secret{
				ObjectMeta: v1.ObjectMeta{
					Name: FederatedSecretName,
				},
			}

			By(fmt.Sprintf("Creating secret %q in namespace %q", secret.Name, nsName))
			_, err := f.FederationClientset_1_4.Core().Secrets(nsName).Create(secret)
			framework.ExpectNoError(err, "Failed to create secret %s", secret.Name)

			// Check subclusters if the secret was created there
			err = wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
				for _, client := range clusterClientSet {
					_, err := client.Core().Secrets(nsName).Get(secret.Name)
					if err != nil && !errors.IsNotFound(err) {
						return false, err
					}
					if err != nil {
						return false, nil
					}
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Not all secrets created")

			By(fmt.Sprintf("Creation of secret %q in namespace %q succeeded.  Deleting secret.", secret.Name, nsName))
			// Cleanup
			deleteAllTestSecrets(
				f.FederationClientset_1_4.Core().Secrets(nsName).List,
				f.FederationClientset_1_4.Core().Secrets(nsName).Delete,
				false)

		})

	})
})

func deleteAllTestSecrets(lister func(api.ListOptions) (*v1.SecretList, error), deleter func(string, *api.DeleteOptions) error, waitForDeletion bool) {
	list, err := lister(api.ListOptions{})
	if err != nil {
		framework.Failf("Failed to get all secrets: %v", err)
		return
	}
	for _, secret := range list.Items {
		if secret.Name == FederatedSecretName {
			err := deleter(secret.Name, &api.DeleteOptions{})
			if err != nil {
				framework.Failf("Failed to set %s for deletion: %v", secret.Name, err)
			}
		}
	}
	if waitForDeletion {
		waitForNoTestSecrets(lister)
	}
}

func waitForNoTestSecrets(lister func(api.ListOptions) (*v1.SecretList, error)) {
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		list, err := lister(api.ListOptions{})
		if err != nil {
			return false, err
		}
		for _, secret := range list.Items {
			if secret.Name == FederatedSecretName {
				return false, nil
			}
		}
		return true, nil
	})
	if err != nil {
		framework.Failf("Secrets not deleted: %v", err)
	}
}
