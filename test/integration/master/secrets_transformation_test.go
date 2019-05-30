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

package master

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"fmt"
	"testing"

	apiserverconfigv1 "k8s.io/apiserver/pkg/apis/config/v1"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
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
		test, err := newTransformTest(t, tt.transformerConfigContent)
		if err != nil {
			test.cleanUp()
			t.Errorf("failed to setup test for envelop %s, error was %v", tt.transformerPrefix, err)
			continue
		}
		test.secret, err = test.createSecret(testSecret, testNamespace)
		if err != nil {
			t.Fatalf("Failed to create test secret, error: %v", err)
		}
		test.run(tt.unSealFunc, tt.transformerPrefix)
		test.cleanUp()
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
	test, err := newTransformTest(b, transformerConfig)
	defer test.cleanUp()
	if err != nil {
		b.Fatalf("failed to setup benchmark for config %s, error was %v", transformerConfig, err)
	}

	b.StartTimer()
	test.benchmark(b)
	b.StopTimer()
	test.printMetrics()
}

func unSealWithGCMTransformer(cipherText []byte, ctx value.Context,
	transformerConfig apiserverconfigv1.ProviderConfiguration) ([]byte, error) {

	block, err := newAESCipher(transformerConfig.AESGCM.Keys[0].Secret)
	if err != nil {
		return nil, fmt.Errorf("failed to create block cipher: %v", err)
	}

	gcmTransformer := aestransformer.NewGCMTransformer(block)

	clearText, _, err := gcmTransformer.TransformFromStorage(cipherText, ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to decypt secret: %v", err)
	}

	return clearText, nil
}

func unSealWithCBCTransformer(cipherText []byte, ctx value.Context,
	transformerConfig apiserverconfigv1.ProviderConfiguration) ([]byte, error) {

	block, err := newAESCipher(transformerConfig.AESCBC.Keys[0].Secret)
	if err != nil {
		return nil, err
	}

	cbcTransformer := aestransformer.NewCBCTransformer(block)

	clearText, _, err := cbcTransformer.TransformFromStorage(cipherText, ctx)
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
