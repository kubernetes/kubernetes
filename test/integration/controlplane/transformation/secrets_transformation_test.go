/*
Copyright 2017 The Kubernetes Authors.

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

package transformation

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/test/integration/authutil"
	"k8s.io/utils/ptr"
)

const (
	aesGCMPrefix = "k8s:enc:aesgcm:v1:key1:"
	aesCBCPrefix = "k8s:enc:aescbc:v1:key1:"

	aesGCMConfigYAML = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aesgcm:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`

	aesCBCConfigYAML = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - aescbc:
        keys:
        - name: key1
          secret: c2VjcmV0IGlzIHNlY3VyZQ==
`

	identityConfigYAML = `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
  - resources:
    - secrets
    providers:
    - identity: {}
`
)

// TestSecretsShouldBeEnveloped is an integration test between KubeAPI and etcd that checks:
// 1. Secrets are encrypted on write
// 2. Secrets are decrypted on read
// when EncryptionConfiguration is passed to KubeAPI server.
func TestSecretsShouldBeTransformed(t *testing.T) {
	var testCases = []struct {
		transformerConfigContent string
		transformerPrefix        string
		unSealFunc               unSealSecret
	}{
		{aesGCMConfigYAML, aesGCMPrefix, unSealWithGCMTransformer},
		{aesCBCConfigYAML, aesCBCPrefix, unSealWithCBCTransformer},
		// TODO: add secretbox
	}
	for _, tt := range testCases {
		test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: tt.transformerConfigContent})
		if err != nil {
			t.Fatalf("failed to setup test for envelop %s, error was %v", tt.transformerPrefix, err)
			continue
		}
		test.secret, err = test.createSecret(testSecret, testNamespace)
		if err != nil {
			t.Fatalf("Failed to create test secret, error: %v", err)
		}
		test.runResource(test.TContext, tt.unSealFunc, tt.transformerPrefix, "", "v1", "secrets", test.secret.Name, test.secret.Namespace)
		test.cleanUp()
	}
}

type verifier interface {
	verify(t *testing.T, err error)
}

// TestAllowUnsafeMalformedObjectDeletionFeature is an integration test that verifies:
// 1) if the feature AllowUnsafeMalformedObjectDeletion is enabled, a corrupt
// object can be deleted by enabling the delete option
// 'ignoreStoreReadErrorWithClusterBreakingPotential', it triggers the unsafe
// deletion flow that bypasses any precondition checks or finalizer constraints.
// 2) if the feature AllowUnsafeMalformedObjectDeletion is disabled, the delete
// option 'ignoreStoreReadErrorWithClusterBreakingPotential' has no
// impact (normal deletion flow is used)
func TestAllowUnsafeMalformedObjectDeletionFeature(t *testing.T) {
	tests := []struct {
		// whether the feature is enabled
		featureEnabled bool
		// whether encryption broke after the change
		encryptionBrokenFn func(t *testing.T, got apierrors.APIStatus) bool
		// what we expect for GET on the corrupt object, after encryption has
		// broken but before the deletion
		corruptObjGetPreDelete verifier
		// what we expect for DELETE on the corrupt object, after encryption has
		// broken, but without setting the option to ignore store read error
		corrupObjDeletWithoutOption verifier
		// what we expect for DELETE on the corrupt object, after encryption has
		// broken, with the option to ignore store read error enabled
		corrupObjDeleteWithOption verifier
		// what we expect for DELETE on the corrupt object, after encryption has
		// broken, with the option to ignore store read error enabled, and
		// the user has the permission to do unsafe delete
		corrupObjDeleteWithOptionAndPrivilege verifier
		// what we expect for GET on the corrupt object (post deletion)
		corrupObjGetPostDelete verifier
		// what we expect for GET on the corrupt object ( via the
		// informer cache post deletion)
		corrupObjGetFromListerCachePostDelete verifier
	}{
		{
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: StorageError: corrupt object") &&
					strings.Contains(got.Status().Message, "data from the storage is not transformable revision=0: no matching prefix found")
			},
			corruptObjGetPreDelete:      wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeletWithoutOption: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeleteWithOption: wantAPIStatusError{
				reason:          metav1.StatusReasonForbidden,
				messageContains: `not permitted to do "unsafe-delete-ignore-read-errors"`,
			},
			corrupObjDeleteWithOptionAndPrivilege: wantNoError{},
			corrupObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonNotFound},
			// after the unsafe delete, we expect the deleted object to be removed from the cache
			corrupObjGetFromListerCachePostDelete: wantAPIStatusError{reason: metav1.StatusReasonNotFound},
		},
		{
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: no matching prefix found")
			},
			corruptObjGetPreDelete:                wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeletWithoutOption:           wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeleteWithOption:             wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeleteWithOptionAndPrivilege: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			// although the object is corrupt and unreadable from the underlying
			// storage, the cache continues to serve the object
			corrupObjGetFromListerCachePostDelete: wantNoError{},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled)

			test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: aesGCMConfigYAML, reload: true})
			if err != nil {
				t.Fatalf("failed to setup test for envelop %s, error was %v", aesGCMPrefix, err)
			}
			defer test.cleanUp()

			// a) set up a distinct client for the test user with the least
			// privileges, we will grant permission as we progress through the test
			testUser := "croc"
			testUserConfig := restclient.CopyConfig(test.kubeAPIServer.ClientConfig)
			testUserConfig.Impersonate.UserName = testUser
			testUserClient := clientset.NewForConfigOrDie(testUserConfig)
			adminClient := test.restClient

			// b) use the admin client to grant the the test user initial permissions,
			// we are not going to grant 'delete-ignore-read-errors' just yet
			permitUserToDoVerbOnSecret(t, adminClient, testUser, testNamespace, []string{"create", "get", "delete", "update"})

			// the test should not use the admin client going forward
			test.restClient = testUserClient
			defer func() {
				// for any cleanup that requires admin privileges
				test.restClient = adminClient
			}()

			secretCorrupt := "foo-with-unsafe-delete"
			// c) create and delete the secret, we don't expect any error
			_, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretCorrupt)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, got error: %v", err, testNamespace, secretCorrupt)
			}

			// d) re-create the secret
			test.secret, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("Failed to create test secret, error: %v", err)
			}

			// e) update the secret with a finalizer
			withFinalizer := test.secret.DeepCopy()
			withFinalizer.Finalizers = append(withFinalizer.Finalizers, "tes.k8s.io/fake")
			test.secret, err = test.restClient.CoreV1().Secrets(testNamespace).Update(context.Background(), withFinalizer, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to add finalizer to the secret, error: %v", err)
			}

			test.runResource(test.TContext, unSealWithGCMTransformer, aesGCMPrefix, "", "v1", "secrets", test.secret.Name, test.secret.Namespace)

			factory := informers.NewSharedInformerFactoryWithOptions(adminClient, time.Minute, informers.WithNamespace(testNamespace))
			informer := factory.Core().V1().Secrets()

			// the test adds a secret named "final-secret" after the unsafe delete
			// so we can use it as an marker that the informer has caught up
			finalEvent := make(chan struct{})
			finalSecretName := "final-secret"
			_, err = informer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					if obj, err := meta.Accessor(obj); err == nil && obj.GetName() == finalSecretName {
						close(finalEvent)
					}
				},
			})
			if err != nil {
				t.Fatalf("unexpected error from AddEventHandler: %v", err)
			}

			lister := informer.Lister()
			factory.Start(test.Done())
			waitForSyncCtx, waitForSyncCancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
			defer waitForSyncCancel()
			if !cache.WaitForCacheSync(waitForSyncCtx.Done(), informer.Informer().HasSynced) {
				t.Fatalf("caches failed to sync")
			}

			// g) the secret should be readable from the cache
			if _, err := lister.Secrets(testNamespace).Get(secretCorrupt); err != nil {
				t.Errorf("unexpected failure getting the secret from lister cache: %v", err)
			}

			// h) override the config and break decryption of the old resources,
			// the secret created in step b will be undecryptable
			now := time.Now()
			encryptionConf := filepath.Join(test.configDir, encryptionConfigFileName)
			body, _ := ioutil.ReadFile(encryptionConf)
			t.Logf("file before write: %s", body)
			// we replace the existing key with a new key from a different provider
			if err := os.WriteFile(encryptionConf, []byte(aesCBCConfigYAML), 0o644); err != nil {
				t.Fatalf("failed to write encryption config that's going to make decryption fail")
			}
			body, _ = ioutil.ReadFile(encryptionConf)
			t.Logf("file after write: %s", body)

			// i) wait for the breaking changes to take effect
			testCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// TODO: dynamic encryption config reload takes about 1m, so can't use
			// wait.ForeverTestTimeout just yet, investigate and reduce the reload time.
			err = wait.PollUntilContextTimeout(testCtx, 1*time.Second, 2*time.Minute, true, func(ctx context.Context) (done bool, err error) {
				_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(ctx, secretCorrupt, metav1.GetOptions{})
				var got apierrors.APIStatus
				if !errors.As(err, &got) {
					return false, nil
				}
				if done := tc.encryptionBrokenFn(t, got); done {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Fatalf("encryption never broke: %v", err)
			}
			t.Logf("it took %s for the apiserver to reload the encryption config", time.Since(now))

			// j) create a new secret, and then delete it, it should work
			secretNormal := "bar-with-normal-delete"
			_, err = test.createSecret(secretNormal, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretNormal)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretNormal, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretNormal)
			}

			// k) let's try to get the broken secret created in step b, we expect it
			// to fail, the error will vary depending on whether the feature is enabled.
			// but the cache still serves the object
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPreDelete.verify(t, err)
			if _, err := lister.Secrets(testNamespace).Get(secretCorrupt); err != nil {
				t.Errorf("unexpected failure when etting the secret from the cache: %v", err)
			}

			// l) let's try the normal deletion flow, we expect an error
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			tc.corrupObjDeletWithoutOption.verify(t, err)

			// m) make an attempt to delete the corrupt object by enabling the option,
			// on the other hand, we have not granted the 'delete-ignore-read-errors'
			// verb to the user yet, so we expect admission to deny the delete request
			options := metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, options)
			tc.corrupObjDeleteWithOption.verify(t, err)

			// n) grant the test user to do 'unsafe-delete-ignore-read-errors' on secrets
			permitUserToDoVerbOnSecret(t, adminClient, testUser, testNamespace, []string{"unsafe-delete-ignore-read-errors"})

			// o) let's try to do unsafe delete again
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, options)
			tc.corrupObjDeleteWithOptionAndPrivilege.verify(t, err)

			// p) final get should return a NotFound error after the secret has been deleted
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corrupObjGetPostDelete.verify(t, err)

			// q) create the final secret, and wait for the informer
			// to catch up on the ADDED event.
			_, err = test.createSecret(finalSecretName, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, finalSecretName)
			}
			select {
			case <-finalEvent:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timed out waiting for the informer to catch up")
			}

			// r) read the object from the cache
			_, err = lister.Secrets(testNamespace).Get(secretCorrupt)
			tc.corrupObjGetFromListerCachePostDelete.verify(t, err)
		})
	}
}

type wantNoError struct{}

func (want wantNoError) verify(t *testing.T, err error) {
	t.Helper()

	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

type wantAPIStatusError struct {
	reason          metav1.StatusReason
	messageContains string
	more            func(*testing.T, apierrors.APIStatus)
}

func (wantError wantAPIStatusError) verify(t *testing.T, err error) {
	t.Helper()

	switch {
	case err != nil:
		var statusGot apierrors.APIStatus
		if !errors.As(err, &statusGot) {
			t.Errorf("expected an API status error, but got: %v", err)
			return
		}
		if want, got := wantError.reason, statusGot.Status().Reason; want != got {
			t.Errorf("expected API status Reason: %q, but got: %q, err: %#v", want, got, statusGot)
		}
		if want, got := wantError.messageContains, statusGot.Status().Message; !strings.Contains(got, want) {
			t.Errorf("expected API status message to contain: %q, got err: %#v", want, statusGot)
		}
		if wantError.more != nil {
			wantError.more(t, statusGot)
		}
	default:
		t.Errorf("expected error: %v, but got none", err)
	}
}

// TestListCorruptObjects is an integration test that verifies:
// 1) if the feature AllowUnsafeMalformedObjectDeletion is enabled, LIST operation,
// in its error response, should include information that identifies the objects
// that it failed to read from the storage
// 2) if the feature AllowUnsafeMalformedObjectDeletion is disabled, LIST should
// abort as soon as it encounters the first error, to be backward compatible
func TestListCorruptObjects(t *testing.T) {
	// these are the secrets that the test will initially create, and
	// are expected to become unreadable/corrupt after encryption breaks
	secrets := []string{"corrupt-a", "corrupt-b", "corrupt-c"}

	tests := []struct {
		featureEnabled bool
		// secrets that are created before encryption breaks
		secrets []string
		// whether encryption broke after the config change
		encryptionBrokenFn func(t *testing.T, got apierrors.APIStatus) bool
		// what we expect for LIST on the corrupt objects after encryption has broken
		listAfter verifier
	}{
		{
			secrets:        secrets,
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				// the new encryption config does not have the old key, so reading of resources
				// created before the encryption change will fail with 'no matching prefix found'
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: StorageError: corrupt object") &&
					strings.Contains(got.Status().Message, "data from the storage is not transformable revision=0: no matching prefix found")
			},
			listAfter: wantAPIStatusError{
				reason:          metav1.StatusReasonStoreReadError,
				messageContains: "failed to read one or more secrets from the storage",
				more: func(t *testing.T, err apierrors.APIStatus) {
					t.Helper()

					details := err.Status().Details
					if details == nil {
						t.Errorf("expected Details in APIStatus, but got: %#v", err)
						return
					}
					if want, got := len(secrets), len(details.Causes); want != got {
						t.Errorf("expected to have %d in APIStatus, but got: %d", want, got)
					}
					for _, cause := range details.Causes {
						if want, got := metav1.CauseTypeUnexpectedServerResponse, cause.Type; want != got {
							t.Errorf("expected to cause type to be %s, but got: %s", want, got)
						}
					}
					for _, want := range secrets {
						var found bool
						for _, got := range details.Causes {
							if strings.HasSuffix(got.Field, want) {
								found = true
								break
							}
						}
						if !found {
							t.Errorf("want key: %q in the Fields: %#v", want, details.Causes)
						}
					}
				},
			},
		},
		{
			secrets:        secrets,
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				// the new encryption config does not have the old key, so reading of resources
				// created before the encryption change will fail with 'no matching prefix found'
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: no matching prefix found")
			},
			listAfter: wantAPIStatusError{
				reason:          metav1.StatusReasonInternalError,
				messageContains: "unable to transform key",
			},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled)

			test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: aesGCMConfigYAML, reload: true})
			if err != nil {
				t.Fatalf("failed to setup test for envelop %s, error was %v", aesGCMPrefix, err)
			}
			defer test.cleanUp()

			// a) create a number of secrets in the test namespace
			for _, name := range tc.secrets {
				_, err = test.createSecret(name, testNamespace)
				if err != nil {
					t.Fatalf("Failed to create test secret, error: %v", err)
				}
			}

			// b) list the secrets before we break encryption
			result, err := test.restClient.CoreV1().Secrets(testNamespace).List(context.Background(), metav1.ListOptions{})
			if err != nil {
				t.Fatalf("listing secrets failed unexpectedly with: %v", err)
			}
			if want, got := len(tc.secrets), len(result.Items); got < 3 {
				t.Fatalf("expected at least %d secrets, but got: %d", want, got)
			}

			// c) override the config and break decryption of the old resources,
			// the secret created in step a will be undecryptable
			encryptionConf := filepath.Join(test.configDir, encryptionConfigFileName)
			body, _ := ioutil.ReadFile(encryptionConf)
			t.Logf("file before write: %s", body)
			// we replace the existing key with a new key from a different provider
			if err := os.WriteFile(encryptionConf, []byte(aesCBCConfigYAML), 0o644); err != nil {
				t.Fatalf("failed to write encryption config that's going to make decryption fail")
			}
			body, _ = ioutil.ReadFile(encryptionConf)
			t.Logf("file after write: %s", body)

			// d) wait for the breaking changes to take effect
			testCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
			// TODO: dynamic encryption config reload takes about 1m, so can't use
			// wait.ForeverTestTimeout just yet, investigate and reduce the reload time.
			err = wait.PollUntilContextTimeout(testCtx, 1*time.Second, 2*time.Minute, true, func(ctx context.Context) (done bool, err error) {
				_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(ctx, tc.secrets[0], metav1.GetOptions{})

				if err != nil {
					t.Logf("get returned error: %#v message: %s", err, err.Error())
				}

				var got apierrors.APIStatus
				if !errors.As(err, &got) {
					return false, nil
				}
				if done := tc.encryptionBrokenFn(t, got); done {
					return true, nil
				}
				return false, nil
			})
			if err != nil {
				t.Fatalf("encryption never broke: %v", err)
			}

			// TODO: ConsistentListFromCache feature returns the list of objects
			// from cache even though these objects are not readable from the
			// store after encryption has broken; to work around this issue, let's
			// create a new secret and retrieve it from the store to get a more
			// recent ResourceVersion and invoke the list with:
			//   ResourceVersionMatch: Exact
			newSecretName := "new-a"
			_, err = test.createSecret(newSecretName, testNamespace)
			if err != nil {
				t.Fatalf("expected no error while creating the new secret, but got: %d", err)
			}
			newSecret, err := test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), newSecretName, metav1.GetOptions{})
			if err != nil {
				t.Fatalf("expected no error getting the new secret, but got: %d", err)
			}

			// e) list should return expected error
			_, err = test.restClient.CoreV1().Secrets(testNamespace).List(context.Background(), metav1.ListOptions{
				ResourceVersion:      newSecret.ResourceVersion,
				ResourceVersionMatch: metav1.ResourceVersionMatchExact,
			})
			tc.listAfter.verify(t, err)

		})
	}
}

func permitUserToDoVerbOnSecret(t *testing.T, client *clientset.Clientset, user, namespace string, verbs []string) {
	t.Helper()

	name := fmt.Sprintf("%s-can-do-%s-on-secrets-in-%s", user, strings.Join(verbs, "-"), namespace)
	_, err := client.RbacV1().Roles(namespace).Create(context.TODO(), &rbacv1.Role{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Rules: []rbacv1.PolicyRule{
			{
				Verbs:     verbs,
				APIGroups: []string{""},
				Resources: []string{"secrets"},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error while creating role: %s, err: %v", name, err)
	}

	_, err = client.RbacV1().RoleBindings(namespace).Create(context.TODO(), &rbacv1.RoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Subjects: []rbacv1.Subject{
			{
				Kind: rbacv1.UserKind,
				Name: user,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbacv1.GroupName,
			Kind:     "Role",
			Name:     name,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error while creating role binding: %s, err: %v", name, err)
	}

	authutil.WaitForNamedAuthorizationUpdate(t, context.TODO(), client.AuthorizationV1(),
		user, namespace, verbs[0], "", schema.GroupResource{Resource: "secrets"}, true)
}

// Baseline (no enveloping) - use to contrast with enveloping benchmarks.
func BenchmarkBase(b *testing.B) {
	runBenchmark(b, "")
}

// Identity transformer is a NOOP (crypto-wise) - use to contrast with AESGCM and AESCBC benchmark results.
func BenchmarkIdentityWrite(b *testing.B) {
	runBenchmark(b, identityConfigYAML)
}

func BenchmarkAESGCMEnvelopeWrite(b *testing.B) {
	runBenchmark(b, aesGCMConfigYAML)
}

func BenchmarkAESCBCEnvelopeWrite(b *testing.B) {
	runBenchmark(b, aesCBCConfigYAML)
}

func runBenchmark(b *testing.B, transformerConfig string) {
	b.StopTimer()
	test, err := newTransformTest(b, transformTestConfig{transformerConfigYAML: transformerConfig})
	if err != nil {
		b.Fatalf("failed to setup benchmark for config %s, error was %v", transformerConfig, err)
	}
	defer test.cleanUp()

	b.StartTimer()
	test.benchmark(b)
	b.StopTimer()
	test.printMetrics()
}

func unSealWithGCMTransformer(ctx context.Context, cipherText []byte, dataCtx value.Context,
	transformerConfig apiserverv1.ProviderConfiguration) ([]byte, error) {

	block, err := newAESCipher(transformerConfig.AESGCM.Keys[0].Secret)
	if err != nil {
		return nil, fmt.Errorf("failed to create block cipher: %v", err)
	}

	gcmTransformer, err := aestransformer.NewGCMTransformer(block)
	if err != nil {
		return nil, fmt.Errorf("failed to create transformer from block: %v", err)
	}

	clearText, _, err := gcmTransformer.TransformFromStorage(ctx, cipherText, dataCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to decypt secret: %v", err)
	}

	return clearText, nil
}

func unSealWithCBCTransformer(ctx context.Context, cipherText []byte, dataCtx value.Context,
	transformerConfig apiserverv1.ProviderConfiguration) ([]byte, error) {

	block, err := newAESCipher(transformerConfig.AESCBC.Keys[0].Secret)
	if err != nil {
		return nil, err
	}

	cbcTransformer := aestransformer.NewCBCTransformer(block)

	clearText, _, err := cbcTransformer.TransformFromStorage(ctx, cipherText, dataCtx)
	if err != nil {
		return nil, fmt.Errorf("failed to decypt secret: %v", err)
	}

	return clearText, nil
}

func newAESCipher(key string) (cipher.Block, error) {
	k, err := base64.StdEncoding.DecodeString(key)
	if err != nil {
		return nil, fmt.Errorf("failed to decode config secret: %v", err)
	}

	block, err := aes.NewCipher(k)
	if err != nil {
		return nil, fmt.Errorf("failed to create AES cipher: %v", err)
	}

	return block, nil
}
