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
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
    - aesgcm:
        keys:
        - name: key1
          secret: dXBlcnByZGVscHJkZWxwCg==
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
		test, err := newTransformTest(t, tt.transformerConfigContent, false, "", nil)
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

// TestBrokenTransformations:
//  1. set up etcd encryption and push a few test secrets
//  2. override the encryption config, thus breaking decryption
//  3. create a few more secrets
//  4. check that the original secrets now cannot be read, but the new ones still can
//  5. check that the read error has the expected format
func TestBrokenTransformations(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, genericfeatures.AllowUnsafeMalformedObjectDeletion, true)

	test, err := newTransformTest(t, aesGCMConfigYAML, true, "", nil)
	defer test.cleanUp()
	if err != nil {
		t.Errorf("failed to setup test for envelop %s, error was %v", aesGCMPrefix, err)
		return
	}

	brokenSecrets := sets.New[string]()
	// push a few secrets that we won't be able to read afterwards
	for _, s := range []string{"a", "b", "c"} {
		name := "pre-fail-" + s
		brokenSecrets.Insert(name)
		_, err = test.createSecret(name, testNamespace)
		if err != nil {
			t.Fatalf("Failed to create test secret, error: %v", err)
		}
	}
	// one more secret to use in an encryption test to confirm the config was fine
	test.secret, err = test.createSecret(testSecret, testNamespace)
	if err != nil {
		t.Fatalf("Failed to create test secret, error: %v", err)
	}
	brokenSecrets.Insert(testSecret)
	test.runResource(test.TContext, unSealWithGCMTransformer, aesGCMPrefix, "", "v1", "secrets", test.secret.Name, test.secret.Namespace)

	// override the config and break decryption of the old resources
	encryptionConf := filepath.Join(test.configDir, encryptionConfigFileName)
	if err := os.WriteFile(encryptionConf, []byte(identityConfigYAML), 0o644); err != nil {
		t.Fatalf("failed to write encryption config that's going to make decryption fail")
	}

	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// wait for the breaking changes to take effect
	err = wait.PollUntilContextTimeout(testCtx, 1*time.Second, 2*time.Minute, true, func(ctx context.Context) (done bool, err error) {
		if _, err := test.restClient.CoreV1().Secrets(testNamespace).Get(ctx, testSecret, metav1.GetOptions{}); err == nil || !apierrors.IsStoreReadError(err) {
			return false, nil
		}
		return true, nil
	})

	if err != nil {
		t.Fatalf("encryption never broke: %v", err)
	}

	// push some new secrets that should be readable
	var secretErr error
	for _, s := range []string{"a", "b", "c"} {
		name := "post-fail-" + s
		_, err = test.createSecret(name, testNamespace)
		if err != nil {
			t.Fatalf("Failed to create test secret, error: %v", err)
		}
	}

	checkStoreError := func(secretErr error, brokenSecrets []string) {
		if secretErr == nil {
			t.Errorf("listing secrets in %q should have failed", testNamespace)
		} else if !apierrors.IsStoreReadError(secretErr) {
			t.Errorf("listing secrets in %q failed with unexpected error: %v", testNamespace, secretErr)
		} else {
			var storeErr apierrors.APIStatus
			if !errors.As(secretErr, &storeErr) {
				t.Fatalf("how is this possible? %v", secretErr)
			}

			causes := storeErr.Status().Details.Causes
			if len(causes) != len(brokenSecrets) {
				t.Errorf("expected %d causes, got %d: %v", len(brokenSecrets), len(causes), causes)
			}

			brokenSecretsCopy := sets.New(brokenSecrets...)
			for _, c := range causes {
				splitKey := strings.Split(c.Field, "/")
				brokenSecretsCopy.Delete(splitKey[len(splitKey)-1])
			}
			if brokenSecretsCopy.Len() != 0 {
				t.Errorf("expected to find all broken secrets in the error, but these secrets' keys were missing: %v", brokenSecretsCopy.UnsortedList())
			}
		}
	}

	_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), "post-fail-a", metav1.GetOptions{})
	if err != nil {
		t.Errorf("'%s/%s' secret should be readable but got error: %v", err, testNamespace, "post-fail-a")
	}

	_, err = test.restClient.CoreV1().Secrets(testNamespace).Get(context.Background(), "pre-fail-c", metav1.GetOptions{})
	checkStoreError(err, []string{"pre-fail-c"})

	_, secretErr = test.restClient.CoreV1().Secrets(testNamespace).List(context.Background(), metav1.ListOptions{})
	checkStoreError(secretErr, brokenSecrets.UnsortedList())

	// create a new NS where listing secrets should be fine
	newNS, err := test.restClient.CoreV1().Namespaces().Create(context.Background(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "pokus"}}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("failed to create the ns: %v", err)
	}

	_, err = test.createSecret("somesecret", newNS.Name)
	if err != nil {
		t.Fatalf("failed to create a secret in the clean ns: %v", err)
	}

	secretList, secretErr := test.restClient.CoreV1().Secrets(newNS.Name).List(context.Background(), metav1.ListOptions{})
	if secretErr != nil || len(secretList.Items) != 1 {
		t.Fatalf("listing secrets in the new NS should be fine but an error appeared: %v\nsecretsList: %v", secretErr, secretList)
	}

	_, secretErr = test.restClient.CoreV1().Secrets("").List(context.Background(), metav1.ListOptions{})
	checkStoreError(secretErr, brokenSecrets.UnsortedList())
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
	test, err := newTransformTest(b, transformerConfig, false, "", nil)
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
