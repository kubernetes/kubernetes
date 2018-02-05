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
	"bytes"
	"context"
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/ghodss/yaml"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/server/options/encryptionconfig"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	aestransformer "k8s.io/apiserver/pkg/storage/value/encrypt/aes"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	testNamespace = "secret-encryption-test"
	testSecret    = "test-secret"

	encryptionConfigFileName = "encryption.conf"

	aesGCMPrefix = "k8s:enc:aesgcm:v1:key1:"
	aesCBCPrefix = "k8s:enc:aescbc:v1:key1:"

	// Secret Data
	secretKey = "api_key"
	secretVal = "086a7ffc-0225-11e8-ba89-0ed5f89f718b"

	aesGCMConfigYAML = `
kind: EncryptionConfig
apiVersion: v1
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
kind: EncryptionConfig
apiVersion: v1
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
kind: EncryptionConfig
apiVersion: v1
resources:
  - resources:
    - secrets
    providers:
    - identity: {}
`
)

type unSealSecret func(cipherText []byte, ctx value.Context, config encryptionconfig.ProviderConfig) ([]byte, error)

type envelopTest struct {
	logger            kubeapiservertesting.Logger
	storageConfig     *storagebackend.Config
	configDir         string
	transformerConfig string
	kubeAPIServer     kubeapiservertesting.TestServer
	restClient        *kubernetes.Clientset
	ns                *corev1.Namespace
	secret            *corev1.Secret
}

func newEnvelopeTest(l kubeapiservertesting.Logger, transformerConfigYAML string) (*envelopTest, error) {
	e := envelopTest{
		logger:            l,
		transformerConfig: transformerConfigYAML,
		storageConfig:     framework.SharedEtcd(),
	}

	var err error
	if transformerConfigYAML != "" {
		if e.configDir, err = createKubeAPIServerEncryptionConfig(transformerConfigYAML); err != nil {
			return nil, fmt.Errorf("error while creating KubeAPIServer encryption config: %v", err)
		}
	}

	if e.kubeAPIServer, err = kubeapiservertesting.StartTestServer(l, e.getKubeAPIServerEncryptionOptions(), e.storageConfig); err != nil {
		return nil, fmt.Errorf("failed to start KubeAPI server: %v", err)
	}

	if e.restClient, err = kubernetes.NewForConfig(e.kubeAPIServer.ClientConfig); err != nil {
		return nil, fmt.Errorf("error while creating rest client: %v", err)
	}

	if e.ns, err = createTestNamespace(e.restClient, testNamespace); err != nil {
		return nil, err
	}

	if e.secret, err = createTestSecret(e.restClient, testSecret, e.ns.Name); err != nil {
		return nil, err
	}

	return &e, nil
}

func (e *envelopTest) cleanUp() {
	os.RemoveAll(e.configDir)
	e.restClient.CoreV1().Namespaces().Delete(e.ns.Name, metav1.NewDeleteOptions(0))
	e.kubeAPIServer.TearDownFn()
}

func (e *envelopTest) run(unSealSecretFunc unSealSecret, expectedEnvelopePrefix string) {
	response, err := readRawRecordFromETCD(&e.kubeAPIServer, e.getETCDPath())
	if err != nil {
		e.logger.Errorf("failed to read from etcd: %v", err)
		return
	}

	if !bytes.HasPrefix(response.Kvs[0].Value, []byte(expectedEnvelopePrefix)) {
		e.logger.Errorf("expected secret to be enveloped by %s, but got %s",
			expectedEnvelopePrefix, response.Kvs[0].Value)
		return
	}

	// etcd path of the key is used as the authenticated context - need to pass it to decrypt
	ctx := value.DefaultContext([]byte(e.getETCDPath()))
	// Envelope header precedes the payload
	sealedData := response.Kvs[0].Value[len(expectedEnvelopePrefix):]
	transformerConfig, err := parseTransformerConfig(e.transformerConfig)
	if err != nil {
		e.logger.Errorf("failed to parse transformer config: %v", err)
	}
	v, err := unSealSecretFunc(sealedData, ctx, *transformerConfig)
	if err != nil {
		e.logger.Errorf("failed to unseal secret: %v", err)
		return
	}
	if !strings.Contains(string(v), secretVal) {
		e.logger.Errorf("expected %q after decryption, but got %q", secretVal, string(v))
	}

	// Secrets should be un-enveloped on direct reads from Kube API Server.
	s, err := e.restClient.CoreV1().Secrets(testNamespace).Get(testSecret, metav1.GetOptions{})
	if secretVal != string(s.Data[secretKey]) {
		e.logger.Errorf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
	}
}

func (e *envelopTest) benchmark(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := createTestSecret(e.restClient, e.secret.Name+strconv.Itoa(i), e.ns.Name)
		if err != nil {
			b.Fatalf("failed to create a secret: %v", err)
		}
	}
}

func (e *envelopTest) getETCDPath() string {
	return fmt.Sprintf("/%s/secrets/%s/%s", e.storageConfig.Prefix, e.ns.Name, e.secret.Name)
}

func (e *envelopTest) getKubeAPIServerEncryptionOptions() []string {
	if e.transformerConfig != "" {
		return []string{"--experimental-encryption-provider-config", path.Join(e.configDir, encryptionConfigFileName)}
	}

	return nil
}

// TestSecretsShouldBeEnveloped is an integration test between KubeAPI and etcd that checks:
// 1. Secrets are encrypted on write
// 2. Secrets are decrypted on read
// when EncryptionConfig is passed to KubeAPI server.
func TestSecretsShouldBeEnveloped(t *testing.T) {
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
		test, err := newEnvelopeTest(t, tt.transformerConfigContent)
		if err != nil {
			test.cleanUp()
			t.Errorf("failed to setup test for envelop %s, error was %v", tt.transformerPrefix, err)
			continue
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
	test, err := newEnvelopeTest(b, transformerConfig)
	defer test.cleanUp()
	if err != nil {
		b.Fatalf("failed to setup benchmark for config %s, error was %v", transformerConfig, err)
	}

	b.StartTimer()
	test.benchmark(b)
	b.StopTimer()
}

func createKubeAPIServerEncryptionConfig(transformerConfig string) (string, error) {
	tempDir, err := ioutil.TempDir("", "secrets-encryption-test")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %v", err)
	}

	encryptionConfig := path.Join(tempDir, encryptionConfigFileName)

	if err := ioutil.WriteFile(encryptionConfig, []byte(transformerConfig), 0644); err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("error while writing encryption config: %v", err)
	}

	return tempDir, nil
}

func createTestNamespace(client *kubernetes.Clientset, name string) (*corev1.Namespace, error) {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}

	if _, err := client.CoreV1().Namespaces().Create(ns); err != nil {
		return nil, fmt.Errorf("unable to create testing namespace %v", err)
	}

	return ns, nil
}

func createTestSecret(client *kubernetes.Clientset, name, namespace string) (*corev1.Secret, error) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			secretKey: []byte(secretVal),
		},
	}
	if _, err := client.CoreV1().Secrets(secret.Namespace).Create(secret); err != nil {
		return nil, fmt.Errorf("error while writing secret: %v", err)
	}

	return secret, nil
}

func readRawRecordFromETCD(kubeAPIServer *kubeapiservertesting.TestServer, path string) (*clientv3.GetResponse, error) {
	etcdClient, err := integration.GetEtcdKVClient(kubeAPIServer.ServerOpts.Etcd.StorageConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %v", err)
	}
	response, err := etcdClient.Get(context.Background(), path, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve secret from etcd %v", err)
	}

	return response, nil
}

func parseTransformerConfig(configContent string) (*encryptionconfig.ProviderConfig, error) {
	var config encryptionconfig.EncryptionConfig
	err := yaml.Unmarshal([]byte(configContent), &config)
	if err != nil {
		return nil, fmt.Errorf("failed to extract transformer key: %v", err)
	}

	return &config.Resources[0].Providers[0], nil
}

func unSealWithGCMTransformer(cipherText []byte, ctx value.Context,
	transformerConfig encryptionconfig.ProviderConfig) ([]byte, error) {

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
	transformerConfig encryptionconfig.ProviderConfig) ([]byte, error) {

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
