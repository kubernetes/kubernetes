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

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
		test, err := newTransformTest(t, tt.transformerConfigContent, false, "", nil, false)
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
		// what we expect for GET on the corrupt object (post deletion)
		corrupObjGetPostDelete verifier
	}{
		{
			featureEnabled: true,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: StorageError: corrupt object") &&
					strings.Contains(got.Status().Message, "data from the storage is not transformable: no matching prefix found")
			},
			corruptObjGetPreDelete:      wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeletWithoutOption: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeleteWithOption:   wantNoError{},
			corrupObjGetPostDelete:      wantAPIStatusError{reason: metav1.StatusReasonNotFound},
		},
		{
			featureEnabled: false,
			encryptionBrokenFn: func(t *testing.T, got apierrors.APIStatus) bool {
				return got.Status().Reason == metav1.StatusReasonInternalError &&
					strings.Contains(got.Status().Message, "Internal error occurred: no matching prefix found")
			},
			corruptObjGetPreDelete:      wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeletWithoutOption: wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjDeleteWithOption:   wantAPIStatusError{reason: metav1.StatusReasonInternalError},
			corrupObjGetPostDelete:      wantAPIStatusError{reason: metav1.StatusReasonInternalError},
		},
	}
	for _, tc := range tests {
		t.Run(fmt.Sprintf("%s/%t", string(genericfeatures.AllowUnsafeMalformedObjectDeletion), tc.featureEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, tc.featureEnabled)

			test, err := newTransformTest(t, aesGCMConfigYAML, true, "", nil, false)
			if err != nil {
				t.Fatalf("failed to setup test for envelop %s, error was %v", aesGCMPrefix, err)
			}
			defer test.cleanUp()

			secretCorrupt := "foo-with-unsafe-delete"
			// a) create and delete the secret, we don't expect any error
			_, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretCorrupt)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to delete, got error: %v", err, testNamespace, secretCorrupt)
			}

			// b) re-create the secret
			test.secret, err = test.createSecret(secretCorrupt, testNamespace)
			if err != nil {
				t.Fatalf("Failed to create test secret, error: %v", err)
			}

			// c) update the secret with a finalizer
			withFinalizer := test.secret.DeepCopy()
			withFinalizer.Finalizers = append(withFinalizer.Finalizers, "tes.k8s.io/fake")
			test.secret, err = test.restClient.CoreV1().Secrets(testNamespace).Update(context.Background(), withFinalizer, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("Failed to add finalizer to the secret, error: %v", err)
			}

			test.runResource(test.TContext, unSealWithGCMTransformer, aesGCMPrefix, "", "v1", "secrets", test.secret.Name, test.secret.Namespace)

			// d) override the config and break decryption of the old resources,
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

			// e) wait for the breaking changes to take effect
			testCtx, cancel := context.WithCancel(context.Background())
			defer cancel()
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

			// f) create a new secret, and then delete it, it should work
			secretNormal := "bar-with-normal-delete"
			_, err = test.createSecret(secretNormal, testNamespace)
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretNormal)
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretNormal, metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("'%s/%s' failed to create, got error: %v", err, testNamespace, secretNormal)
			}

			// g) let's try to get the broken secret created in step b, we expect it
			// to fail, the error will vary depending on whether the feature is enabled
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corruptObjGetPreDelete.verify(t, err)

			// h) let's try the normal deletion flow, we expect an error
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, metav1.DeleteOptions{})
			tc.corrupObjDeletWithoutOption.verify(t, err)

			// i) make an attempt to delete the corrupt object by enabling the option
			options := metav1.DeleteOptions{
				IgnoreStoreReadErrorWithClusterBreakingPotential: ptr.To[bool](true),
			}
			err = test.restClient.CoreV1().Secrets(testNamespace).Delete(context.Background(), secretCorrupt, options)
			tc.corrupObjDeleteWithOption.verify(t, err)

			// j) final get should return a NotFound error after the secret has been deleted
			_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), secretCorrupt, metav1.GetOptions{})
			tc.corrupObjGetPostDelete.verify(t, err)
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
	default:
		t.Errorf("expected error: %v, but got none", err)
	}
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
	test, err := newTransformTest(b, transformerConfig, false, "", nil, false)
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
