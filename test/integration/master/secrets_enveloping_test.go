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
)

type unSealSecret func(cipherText []byte, ctx value.Context, config encryptionconfig.ProviderConfig) ([]byte, error)

// TestSecretsShouldBeEnveloped is an integration test between KubeAPI and ECTD that checks:
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
		runEnvelopeTest(t, tt.unSealFunc, tt.transformerConfigContent, tt.transformerPrefix)
	}
}

func runEnvelopeTest(t *testing.T, unSealSecretFunc unSealSecret, transformerConfigYAML, expectedEnvelopePrefix string) {
	transformerConfig := parseTransformerConfigOrDie(t, transformerConfigYAML)

	storageConfig := framework.SharedEtcd()
	kubeAPIServer, err := startKubeApiWithEncryption(t, storageConfig, transformerConfigYAML)
	if err != nil {
		t.Error(err)
		return
	}
	defer kubeAPIServer.TearDownFn()

	client, err := kubernetes.NewForConfig(kubeAPIServer.ClientConfig)
	if err != nil {
		t.Fatalf("error while creating client: %v", err)
	}

	ns, err := createTestNamespace(client, testNamespace)
	if err != nil {
		t.Error(err)
		return
	}
	defer func() {
		client.CoreV1().Namespaces().Delete(ns.Name, metav1.NewDeleteOptions(0))
	}()

	_, err = createTestSecret(client, testSecret, ns.Name)
	if err != nil {
		t.Error(err)
		return
	}

	etcdPath := getETCDPath(storageConfig.Prefix)
	response, err := readRawRecordFromETCD(kubeAPIServer, etcdPath)
	if err != nil {
		t.Error(err)
		return
	}

	if !bytes.HasPrefix(response.Kvs[0].Value, []byte(expectedEnvelopePrefix)) {
		t.Errorf("expected secret to be enveloped by %s, but got %s",
			expectedEnvelopePrefix, response.Kvs[0].Value)
		return
	}

	// etcd path of the key is used as authenticated context - need to pass it to decrypt
	ctx := value.DefaultContext([]byte(etcdPath))
	// Envelope header precedes the payload
	sealedData := response.Kvs[0].Value[len(expectedEnvelopePrefix):]
	v, err := unSealSecretFunc(sealedData, ctx, transformerConfig)
	if err != nil {
		t.Error(err)
		return
	}
	if !strings.Contains(string(v), secretVal) {
		t.Errorf("expected %q after decryption, but got %q", secretVal, string(v))
	}

	// Secrets should be un-enveloped on direct reads from Kube API Server.
	s, err := client.CoreV1().Secrets(testNamespace).Get(testSecret, metav1.GetOptions{})
	if secretVal != string(s.Data[secretKey]) {
		t.Errorf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
	}
}

func startKubeApiWithEncryption(t *testing.T, storageConfig *storagebackend.Config,
	transformerConfig string) (*kubeapiservertesting.TestServer, error) {
	tempDir, err := ioutil.TempDir("", "secrets-encryption-test")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(tempDir)

	encryptionConfig, err := ioutil.TempFile(tempDir, "encryption-config")
	if err != nil {
		return nil, fmt.Errorf("error while creating temp file for encryption config %v", err)
	}

	if _, err := encryptionConfig.Write([]byte(transformerConfig)); err != nil {
		return nil, fmt.Errorf("error while writing encryption config: %v", err)
	}

	kubeAPIOptions := []string{"--experimental-encryption-provider-config", encryptionConfig.Name()}
	server, err := kubeapiservertesting.StartTestServer(t, kubeAPIOptions, storageConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to start KubeAPI Server %v", err)
	}

	return &server, nil
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
	// Reading secret directly from etcd - expect data to be enveloped and the payload encrypted.
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

func getETCDPath(prefix string) string {
	return fmt.Sprintf("/%s/secrets/%s/%s", prefix, testNamespace, testSecret)
}

func parseTransformerConfigOrDie(t *testing.T, configContent string) encryptionconfig.ProviderConfig {
	var config encryptionconfig.EncryptionConfig
	err := yaml.Unmarshal([]byte(configContent), &config)
	if err != nil {
		t.Errorf("failed to extract transformer key: %v", err)
	}

	return config.Resources[0].Providers[0]
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
