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

package e2e_federation

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/test/e2e/framework"
	fedframework "k8s.io/kubernetes/test/e2e_federation/framework"
)

const (
	secretNamePrefix       = "e2e-secret-test-"
	FederatedSecretTimeout = 60 * time.Second
	MaxRetries             = 3
)

// Create/delete secret api objects
var _ = framework.KubeDescribe("Federation secrets [Feature:Federation]", func() {
	var clusters map[string]*cluster // All clusters, keyed by cluster name

	f := fedframework.NewDefaultFederatedFramework("federated-secret")

	Describe("Secret objects", func() {

		BeforeEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			clusters, _ = getRegisteredClusters(UserAgentName, f)
		})

		AfterEach(func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			// Delete all secrets.
			nsName := f.FederationNamespace.Name
			deleteAllSecretsOrFail(f.FederationClientset, nsName)
		})

		It("should be created and deleted successfully", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			secret := createSecretOrFail(f.FederationClientset, nsName)
			// wait for secret shards being created
			waitForSecretShardsOrFail(nsName, secret, clusters)
			secret = updateSecretOrFail(f.FederationClientset, nsName, secret.Name)
			waitForSecretShardsUpdatedOrFail(nsName, secret, clusters)
		})

		It("should be deleted from underlying clusters when OrphanDependents is false", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := false
			verifyCascadingDeletionForSecret(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that secrets were deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is true", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			orphanDependents := true
			verifyCascadingDeletionForSecret(f.FederationClientset, clusters, &orphanDependents, nsName)
			By(fmt.Sprintf("Verified that secrets were not deleted from underlying clusters"))
		})

		It("should not be deleted from underlying clusters when OrphanDependents is nil", func() {
			fedframework.SkipUnlessFederated(f.ClientSet)
			nsName := f.FederationNamespace.Name
			verifyCascadingDeletionForSecret(f.FederationClientset, clusters, nil, nsName)
			By(fmt.Sprintf("Verified that secrets were not deleted from underlying clusters"))
		})
	})
})

// deleteAllSecretsOrFail deletes all secrets in the given namespace name.
func deleteAllSecretsOrFail(clientset *fedclientset.Clientset, nsName string) {
	SecretList, err := clientset.Core().Secrets(nsName).List(metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	orphanDependents := false
	for _, Secret := range SecretList.Items {
		deleteSecretOrFail(clientset, nsName, Secret.Name, &orphanDependents)
	}
}

// verifyCascadingDeletionForSecret verifies that secrets are deleted from
// underlying clusters when orphan dependents is false and they are not
// deleted when orphan dependents is true.
func verifyCascadingDeletionForSecret(clientset *fedclientset.Clientset, clusters map[string]*cluster, orphanDependents *bool, nsName string) {
	secret := createSecretOrFail(clientset, nsName)
	secretName := secret.Name
	// Check subclusters if the secret was created there.
	By(fmt.Sprintf("Waiting for secret %s to be created in all underlying clusters", secretName))
	err := wait.Poll(5*time.Second, 2*time.Minute, func() (bool, error) {
		for _, cluster := range clusters {
			_, err := cluster.Core().Secrets(nsName).Get(secretName, metav1.GetOptions{})
			if err != nil {
				if !errors.IsNotFound(err) {
					return false, err
				}
				return false, nil
			}
		}
		return true, nil
	})
	framework.ExpectNoError(err, "Not all secrets created")

	By(fmt.Sprintf("Deleting secret %s", secretName))
	deleteSecretOrFail(clientset, nsName, secretName, orphanDependents)

	By(fmt.Sprintf("Verifying secrets %s in underlying clusters", secretName))
	errMessages := []string{}
	// secret should be present in underlying clusters unless orphanDependents is false.
	shouldExist := orphanDependents == nil || *orphanDependents == true
	for clusterName, clusterClientset := range clusters {
		_, err := clusterClientset.Core().Secrets(nsName).Get(secretName, metav1.GetOptions{})
		if shouldExist && errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("unexpected NotFound error for secret %s in cluster %s, expected secret to exist", secretName, clusterName))
		} else if !shouldExist && !errors.IsNotFound(err) {
			errMessages = append(errMessages, fmt.Sprintf("expected NotFound error for secret %s in cluster %s, got error: %v", secretName, clusterName, err))
		}
	}
	if len(errMessages) != 0 {
		framework.Failf("%s", strings.Join(errMessages, "; "))
	}
}

func createSecretOrFail(clientset *fedclientset.Clientset, nsName string) *v1.Secret {
	if len(nsName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to createSecretOrFail: namespace: %v", nsName))
	}

	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      v1.SimpleNameGenerator.GenerateName(secretNamePrefix),
			Namespace: nsName,
		},
	}
	By(fmt.Sprintf("Creating secret %q in namespace %q", secret.Name, nsName))
	_, err := clientset.Core().Secrets(nsName).Create(secret)
	framework.ExpectNoError(err, "Failed to create secret %s", secret.Name)
	By(fmt.Sprintf("Successfully created federated secret %q in namespace %q", secret.Name, nsName))
	return secret
}

func deleteSecretOrFail(clientset *fedclientset.Clientset, nsName string, secretName string, orphanDependents *bool) {
	By(fmt.Sprintf("Deleting secret %q in namespace %q", secretName, nsName))
	err := clientset.Core().Secrets(nsName).Delete(secretName, &metav1.DeleteOptions{OrphanDependents: orphanDependents})
	if err != nil && !errors.IsNotFound(err) {
		framework.ExpectNoError(err, "Error deleting secret %q in namespace %q", secretName, nsName)
	}

	// Wait for the secret to be deleted.
	err = wait.Poll(5*time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		_, err := clientset.Core().Secrets(nsName).Get(secretName, metav1.GetOptions{})
		if err != nil && errors.IsNotFound(err) {
			return true, nil
		}
		return false, err
	})
	if err != nil {
		framework.Failf("Error in deleting secret %s: %v", secretName, err)
	}
}

func updateSecretOrFail(clientset *fedclientset.Clientset, nsName string, secretName string) *v1.Secret {
	if clientset == nil || len(nsName) == 0 {
		Fail(fmt.Sprintf("Internal error: invalid parameters passed to updateSecretOrFail: clientset: %v, namespace: %v", clientset, nsName))
	}

	var newSecret *v1.Secret
	for retryCount := 0; retryCount < MaxRetries; retryCount++ {
		secret, err := clientset.Core().Secrets(nsName).Get(secretName, metav1.GetOptions{})
		if err != nil {
			framework.Failf("failed to get secret %q: %v", secretName, err)
		}

		// Update one of the data in the secret.
		secret.Data = map[string][]byte{
			"key": []byte("value"),
		}
		newSecret, err = clientset.Core().Secrets(nsName).Update(secret)
		if err == nil {
			return newSecret
		}
		if !errors.IsConflict(err) && !errors.IsServerTimeout(err) {
			framework.Failf("failed to update secret %q: %v", secretName, err)
		}
	}
	framework.Failf("too many retries updating secret %q", secretName)
	return newSecret
}

func waitForSecretShardsOrFail(nsName string, secret *v1.Secret, clusters map[string]*cluster) {
	framework.Logf("Waiting for secret %q in %d clusters", secret.Name, len(clusters))
	for _, c := range clusters {
		waitForSecretOrFail(c.Clientset, nsName, secret, true, FederatedSecretTimeout)
	}
}

func waitForSecretOrFail(clientset *kubeclientset.Clientset, nsName string, secret *v1.Secret, present bool, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated secret shard of secret %q in namespace %q from cluster", secret.Name, nsName))
	var clusterSecret *v1.Secret
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterSecret, err := clientset.Core().Secrets(nsName).Get(secret.Name, metav1.GetOptions{})
		if (!present) && errors.IsNotFound(err) { // We want it gone, and it's gone.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is absent", secret.Name, nsName))
			return true, nil // Success
		}
		if present && err == nil { // We want it present, and the Get succeeded, so we're all good.
			By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is present", secret.Name, nsName))
			return true, nil // Success
		}
		By(fmt.Sprintf("Secret %q in namespace %q in cluster.  Found: %v, waiting for Found: %v, trying again in %s (err=%v)", secret.Name, nsName, clusterSecret != nil && err == nil, present, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify secret %q in namespace %q in cluster: Present=%v", secret.Name, nsName, present)

	if present && clusterSecret != nil {
		Expect(util.SecretEquivalent(*clusterSecret, *secret))
	}
}

func waitForSecretShardsUpdatedOrFail(nsName string, secret *v1.Secret, clusters map[string]*cluster) {
	framework.Logf("Waiting for secret %q in %d clusters", secret.Name, len(clusters))
	for _, c := range clusters {
		waitForSecretUpdateOrFail(c.Clientset, nsName, secret, FederatedSecretTimeout)
	}
}

func waitForSecretUpdateOrFail(clientset *kubeclientset.Clientset, nsName string, secret *v1.Secret, timeout time.Duration) {
	By(fmt.Sprintf("Fetching a federated secret shard of secret %q in namespace %q from cluster", secret.Name, nsName))
	err := wait.PollImmediate(framework.Poll, timeout, func() (bool, error) {
		clusterSecret, err := clientset.Core().Secrets(nsName).Get(secret.Name, metav1.GetOptions{})
		if err == nil { // We want it present, and the Get succeeded, so we're all good.
			if util.SecretEquivalent(*clusterSecret, *secret) {
				By(fmt.Sprintf("Success: shard of federated secret %q in namespace %q in cluster is updated", secret.Name, nsName))
				return true, nil
			} else {
				By(fmt.Sprintf("Expected equal secrets. expected: %+v\nactual: %+v", *secret, *clusterSecret))
			}
			By(fmt.Sprintf("Secret %q in namespace %q in cluster, waiting for secret being updated, trying again in %s (err=%v)", secret.Name, nsName, framework.Poll, err))
			return false, nil
		}
		By(fmt.Sprintf("Secret %q in namespace %q in cluster, waiting for being updated, trying again in %s (err=%v)", secret.Name, nsName, framework.Poll, err))
		return false, nil
	})
	framework.ExpectNoError(err, "Failed to verify secret %q in namespace %q in cluster", secret.Name, nsName)
}
