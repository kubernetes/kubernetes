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
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	FederatedSecretName    = "federated-secret"
	FederatedSecretTimeout = 60 * time.Second
	MaxRetries             = 3
)

// Create/delete secret api objects
var _ = framework.KubeDescribe("Federation secrets [Feature:Federation12]", func() {
	var clusters map[string]*cluster // All clusters, keyed by cluster name

	f := framework.NewDefaultFederatedFramework("federated-secret")

	Describe("Secret objects", func() {

		BeforeEach(func() {
			framework.SkipUnlessFederated(f.Client)
			clusters = map[string]*cluster{}
			registerClusters(clusters, UserAgentName, "", f)
		})

		AfterEach(func() {
			framework.SkipUnlessFederated(f.Client)
			unregisterClusters(clusters, f)
		})

		It("should be created and deleted successfully", func() {
			framework.SkipUnlessFederated(f.Client)
			nsName := f.FederationNamespace.Name
			secret := createSecretOrFail(f.FederationClientset_1_4, nsName)

			defer func() { // Cleanup
				By(fmt.Sprintf("Deleting secret %q in namespace %q", secret.Name, nsName))
				err := f.FederationClientset_1_4.Core().Secrets(nsName).Delete(secret.Name, &api.DeleteOptions{})
				framework.ExpectNoError(err, "Error deleting secret %q in namespace %q", secret.Name, nsName)
			}()
			// wait for secret shards being created
			waitForSecretShardsOrFail(nsName, secret, clusters)
			secret = updateSecretOrFail(f.FederationClientset_1_4, nsName)
			waitForSecretShardsUpdatedOrFail(nsName, secret, clusters)
		})
	})
})

func createSecretOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1.Secret {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createSecretOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	secret := &v1.Secret{
		ObjectMeta: v1.ObjectMeta{
			Name: FederatedSecretName,
		},
	}
	By(fmt.Sprintf("Creating secret %q in namespace %q", secret.Name, namespace))
	_, err := clientset.Core().Secrets(namespace).Create(secret)
	framework.ExpectNoError(err, "Failed to create secret %s", secret.Name)
	By(fmt.Sprintf("Successfully created federated secret %q in namespace %q", FederatedSecretName, namespace))
	return secret
}

func updateSecretOrFail(clientset *federation_release_1_4.Clientset, namespace string) *v1.Secret {
	if clientset == nil || len(namespace) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateSecretOrFail: clientset: %v, namespace: %v", clientset, namespace))
	}

	var newSecret *v1.Secret
	for retryCount := 0; retryCount < MaxRetries; retryCount++ {
		secret, err := clientset.Core().Secrets(namespace).Get(FederatedSecretName)
		if err != nil {
			framework.Failf("failed to get secret %q: %v", FederatedSecretName, err)
		}

		// Update one of the data in the secret.
		secret.Data = map[string][]byte{
			"key": []byte("value"),
		}
		newSecret, err = clientset.Core().Secrets(namespace).Update(secret)
		if err == nil {
			return newSecret
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update secret %q: %v", FederatedSecretName, err)
		}
	}
	framework.Failf("too many retries updating secret %q", FederatedSecretName)
	return newSecret
}

func waitForSecretShardsOrFail(namespace string, secret *v1.Secret, clusters map[string]*cluster) {
	framework.Logf("Waiting for secret %q in %d clusters", secret.Name, len(clusters))
	for _, c := range clusters {
		waitForSecretOrFail(c.Clientset, namespace, secret, true, FederatedSecretTimeout)
	}
}

func waitForSecretOrFail(clientset *release_1_3.Clientset, namespace string, secret *v1.Secret, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated secret shard of secret %q in namespace %q from cluster", secret.Name, namespace))
	var clusterSecret *v1.Secret
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterSecret, err := clientset.Core().Secrets(namespace).Get(secret.Name)
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is absent", secret.Name, namespace))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is present", secret.Name, namespace))
			return true, nil // Success
		}
		By(fmt.Sprintf("Secret %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", secret.Name, namespace, clusterSecret != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify secret %q in namespace %q in cluster: Present=%v", secret.Name, namespace, present)

	if present && clusterSecret != nil {
		Expect(util.SecretEquivalent(*clusterSecret, *secret))
	}
}

func waitForSecretShardsUpdatedOrFail(namespace string, secret *v1.Secret, clusters map[string]*cluster) {
	framework.Logf("Waiting for secret %q in %d clusters", secret.Name, len(clusters))
	for _, c := range clusters {
		waitForSecretUpdateOrFail(c.Clientset, namespace, secret, FederatedSecretTimeout)
	}
}

func waitForSecretUpdateOrFail(clientset *release_1_3.Clientset, namespace string, secret *v1.Secret, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated secret shard of secret %q in namespace %q from cluster", secret.Name, namespace))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterSecret, err := clientset.Core().Secrets(namespace).Get(secret.Name)
		if err == nil { // We want it present, and the Get succeeded, so we're all good.
			if util.SecretEquivalent(*clusterSecret, *secret) {
				By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is updated", secret.Name, namespace))
				return true, nil
			} else {
				By(fmt.Sprintf("Expected equal secrets. expected: %+v\nactual: %+v", *secret, *clusterSecret))
			}
			By(fmt.Sprintf("Secret %q in namespace %q in cluster, waiting for secret being updated, trying again in %s (err=%v)", secret.Name, namespace, framework.Poll, err))
			return false, nil
		}
		By(fmt.Sprintf("Secret %q in namespace %q in cluster, waiting for being updated, trying again in %s (err=%v)", secret.Name, namespace, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify secret %q in namespace %q in cluster", secret.Name, namespace)
}
