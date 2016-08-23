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

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/gomega"
)

const (
	FederatedSecretName = "federated-secret"
)

// Create/delete secret api objects
var _ = framework.KubeDescribe("Federation secrets [Feature:Federation]", func() {
	f := framework.NewDefaultFederatedFramework("federated-secret")

	Describe("Secret objects", func() {
		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)

			// Delete registered secrets.
			// This is if a test failed, it should not affect other tests.
			secretList, err := f.FederationClientset_1_4.Core().Secrets(f.Namespace.Name).List(api.ListOptions{})
			Expect(err).NotTo(HaveOccurred())
			for _, secret := range secretList.Items {
				err := f.FederationClientset_1_4.Core().Secrets(f.Namespace.Name).Delete(secret.Name, &api.DeleteOptions{})
				Expect(err).NotTo(HaveOccurred())
			}
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)
			secret := createSecretOrFail(f.FederationClientset_1_4, f.Namespace.Name)
			By(fmt.Sprintf("Creation of secret %q in namespace %q succeeded.  Deleting secret.", secret.Name, f.Namespace.Name))
			// Cleanup
			err := f.FederationClientset_1_4.Core().Secrets(f.Namespace.Name).Delete(secret.Name, &api.DeleteOptions{})
			framework.ExpectNoError(err, "Error deleting secret %q in namespace %q", secret.Name, secret.Namespace)
			By(fmt.Sprintf("Deletion of secret %q in namespace %q succeeded.", secret.Name, f.Namespace.Name))
		})

	})
})

func createSecretOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1.Secret {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to deleteSecretOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}
	By(fmt.Sprintf("Creating federated secret %q in namespace %q", FederatedSecretName, namespace))

	secret := &v1.Secret{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedSecretName,
		},
	}

	By(fmt.Sprintf("Trying to create secret %q in namespace %q", secret.Name, namespace))
	_, err := clientset.Core().Secrets(namespace).Create(secret)
	framework.ExpectNoError(err, "Creating secret %q in namespace %q", secret.Name, namespace)
	By(fmt.Sprintf("Successfully created federated secret %q in namespace %q", FederatedSecretName, namespace))
	return secret
}
