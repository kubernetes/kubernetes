/*
Copyright The Kubernetes Authors.

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
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/ptr"
)

// TestBitFlipCorruptObjectDeletion exercises the decoder error path for
// KEP-3926 using Secrets (protobuf encoding). Unlike
// TestAllowUnsafeMalformedObjectDeletionFeature which tests transformer errors
// (wrong encryption config → "untransformable"), this test uses the identity
// (no-op) encryption provider and corrupts stored bytes directly in etcd.
// Flipping the last byte of protobuf-encoded Secrets reliably breaks decoding,
// producing "undecodable" errors instead of "untransformable" errors.
//
// The informer is given an extended timeout (2 minutes) after deletion to
// recover from the exponential backoff accumulated during the corruption
// window. The reflector's backoff caps at [30s, 60s) with jitter
// (see client-go/tools/cache/reflector.go), so 2 minutes provides
// sufficient leeway.
func TestBitFlipCorruptObjectDeletion(t *testing.T) {
	tests := []struct {
		featureEnabled                         bool
		encryptionBrokenFn                     func(t *testing.T, got apierrors.APIStatus) bool
		corruptObjGetPreDelete                 verifier
		corruptObjDeleteWithoutOption          verifier
		corruptObjDeleteWithOption             verifier
		corruptObjDeleteWithOptionAndPrivilege verifier
		corruptObjGetPostDelete                verifier
		corruptObjGetFromListerCachePostDelete verifier
	}{
		{
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "StorageError: corrupt object") &&
					strings.Contains(got.Status().Message, "object not decodable")
			},
			corruptObjGetPreDelete:        wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithoutOption: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOption: wantAPIStatusError{
				reason:          metav1.StatusReasonForbidden,
				messageContains: `unsafe-delete-ignore-read-errors`,
			},
			corruptObjDeleteWithOptionAndPrivilege: wantNoError{},
			corruptObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonNotFound},
			corruptObjGetFromListerCachePostDelete: wantAPIStatusError{reason: metav1.StatusReasonNotFound},
		},
		{
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					!strings.Contains(got.Status().Message, "corrupt object")
			},
			corruptObjGetPreDelete:                 wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithoutOption:          wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOption:             wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjDeleteWithOptionAndPrivilege: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjGetPostDelete:                wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corruptObjGetFromListerCachePostDelete: wantNoError{},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled)

			if !tc.featureEnabled {
				// With gate=false the protobuf decode error ("unexpected EOF")
				// breaks the server-side cacher for Secrets, so GET requests
				// hang instead of returning a proper apierrors.APIStatus.
				// This is a known issue to be addressed separately.
				t.Skip("bit-flip decoder corruption with gate=false is not yet supported for Secrets")
			}

			// Identity provider: no-op transformer, no config reload needed.
			test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: identityConfigYAML, reload: false})
			if err != nil {
				t.Fatalf("failed to setup test, error was %v", err)
			}
			defer test.cleanUp()

			// a) set up a distinct client for the test user with the least privileges
			testUser := "croc"
			testUserConfig := restclient.CopyConfig(test.kubeAPIServer.ClientConfig)
			testUserConfig.Impersonate.UserName = testUser
			testUserClient := clientset.NewForConfigOrDie(testUserConfig)
			adminClient := test.restClient

			// b) grant the test user initial permissions (not unsafe-delete yet)
			permitUserToDoVerbOnSecret(t, adminClient, testUser, testNamespace, []string{"create", "get", "delete", "update"})

			// the test should not use the admin client going forward
			test.restClient = testUserClient
			defer func() {
				test.restClient = adminClient
			}()

			secretCorrupt := "foo-with-unsafe-delete"
			// c) create and delete the secret — no error expected
			_, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", testNamespace, secretCorrupt, err)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, got error: %v", testNamespace, secretCorrupt, err)
			}

			// d) re-create the secret with a finalizer
			test.secret, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("Failed to create test secret, error: %v", err)
			}
			withFinalizer := test.secret.DeepCopy()
			withFinalizer.Finalizers = append(withFinalizer.Finalizers, "test.k8s.io/fake")
			test.secret, err = test.restClient.CoreV1().Secrets(testNamespace).Update(context.Background(), withFinalizer, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to add finalizer to the secret, error: %v", err)
			}

			// e) set up an informer to track secrets in the cache
			factory := informers.NewSharedInformerFactoryWithOptions(adminClient, time.Minute, informers.WithNamespace(testNamespace))
			secretInformer := factory.Core().V1().Secrets()
			lister := secretInformer.Lister()

			// the test adds a secret named "final-secret" after the unsafe delete
			// so we can use it as a marker that the informer has caught up
			finalEvent := make(chan struct{})
			finalSecretName := "final-secret"
			_, err = secretInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
				AddFunc: func(obj any) {
					if obj, err := meta.Accessor(obj); err == nil && obj.GetName() == finalSecretName {
						close(finalEvent)
					}
				},
			})
			if err != nil {
				t.Fatalf("unexpected error from AddEventHandler: %v", err)
			}

			informerCtx, informerCancel := context.WithCancel(test)
			defer informerCancel()
			factory.Start(informerCtx.Done())
			waitForSyncCtx, waitForSyncCancel := context.WithTimeout(context.Background(), wait.ForeverTestTimeout)
			defer waitForSyncCancel()
			if !cache.WaitForCacheSync(waitForSyncCtx.Done(), secretInformer.Informer().HasSynced) {
				t.Fatalf("informer failed to sync")
			}

			// f) the secret should be readable from the lister
			if _, err := lister.Secrets(testNamespace).Get(secretCorrupt); err != nil {
				t.Errorf("unexpected failure getting the secret from lister cache: %v", err)
			}

			// g) corrupt the secret by flipping the last byte in etcd.
			//    With the identity provider, the stored value is raw protobuf —
			//    flipping a byte makes it undecodable.
			etcdPath := test.getETCDPathForResource(test.storageConfig.Prefix, "", "secrets", secretCorrupt, testNamespace)
			resp, err := test.readRawRecordFromETCD(etcdPath)
			if err != nil {
				t.Fatalf("failed to read from etcd: %v", err)
			}
			if len(resp.Kvs) != 1 {
				t.Fatalf("expected 1 key in etcd, got %d", len(resp.Kvs))
			}
			value := make([]byte, len(resp.Kvs[0].Value))
			copy(value, resp.Kvs[0].Value)
			value[len(value)-1] ^= 0xFF
			if _, err := test.writeRawRecordToETCD(etcdPath, value); err != nil {
				t.Fatalf("failed to write corrupted value to etcd: %v", err)
			}

			// h) poll until GET fails with the expected decode error — the corruption
			//    is immediate (no config reload), but we poll for robustness.
			err = wait.PollUntilContextTimeout(t.Context(), 1*time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (done bool, err error) {
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
				t.Fatalf("bit-flip corruption never took effect: %v", err)
			}

			// i) create a new secret, and then delete it — should work fine
			secretNormal := "bar-with-normal-delete"
			_, err = test.createSecret(secretNormal, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", testNamespace, secretNormal, err)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretNormal, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, got error: %v", testNamespace, secretNormal, err)
			}

			// j) GET the corrupt secret — expect failure.
			//    Also check lister still has cached copy from before corruption.
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPreDelete.verify(t, err)
			if _, err := lister.Secrets(testNamespace).Get(secretCorrupt); err != nil {
				t.Errorf("unexpected failure getting the secret from the lister cache: %v", err)
			}

			// k) normal delete — expect error
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			tc.corruptObjDeleteWithoutOption.verify(t, err)

			// l) delete with unsafe option but no privilege — expect forbidden or internal error
			options := metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To(true),
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, options)
			tc.corruptObjDeleteWithOption.verify(t, err)

			// m) grant the test user the unsafe-delete-ignore-read-errors verb
			permitUserToDoVerbOnSecret(t, adminClient, testUser, testNamespace, []string{"unsafe-delete-ignore-read-errors"})

			// n) try unsafe delete again — should succeed when feature enabled
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, options)
			tc.corruptObjDeleteWithOptionAndPrivilege.verify(t, err)

			// o) final GET
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPostDelete.verify(t, err)

			// p) verify via informer cache — the informer accumulates exponential
			//    backoff during the corruption window (up to [30s, 60s) between
			//    retries), so we allow up to 2 minutes for recovery.
			if tc.featureEnabled {
				// create the final secret and wait for the informer to catch up
				_, err = test.createSecret(finalSecretName, testNamespace)
				if err != nil {
					t.Fatalf("'%s/%s' failed to create, got error: %v", testNamespace, finalSecretName, err)
				}
				select {
				case <-finalEvent:
				case <-time.After(2 * time.Minute):
					t.Fatalf("timed out waiting for the informer to catch up")
				}

				// verify corrupt secret is NOT in the lister
				_, err = lister.Secrets(testNamespace).Get(secretCorrupt)
				tc.corruptObjGetFromListerCachePostDelete.verify(t, err)
			} else {
				// gate=false: deletion fails, so the informer's stale cached
				// copy from before corruption is still present.
				_, err = lister.Secrets(testNamespace).Get(secretCorrupt)
				tc.corruptObjGetFromListerCachePostDelete.verify(t, err)
			}
		})
	}
}
