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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/yaml"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apiserverconfigv1 "k8s.io/apiserver/pkg/apis/config/v1"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	secretKey                = "api_key"
	secretVal                = "086a7ffc-0225-11e8-ba89-0ed5f89f718b"
	encryptionConfigFileName = "encryption.conf"
	testNamespace            = "secret-encryption-test"
	testSecret               = "test-secret"
	metricsPrefix            = "apiserver_storage_"
)

type unSealSecret func(cipherText []byte, ctx value.Context, config apiserverconfigv1.ProviderConfiguration) ([]byte, error)

type transformTest struct {
	logger            kubeapiservertesting.Logger
	storageConfig     *storagebackend.Config
	configDir         string
	transformerConfig string
	kubeAPIServer     kubeapiservertesting.TestServer
	restClient        *kubernetes.Clientset
	ns                *corev1.Namespace
	secret            *corev1.Secret
}

func newTransformTest(l kubeapiservertesting.Logger, transformerConfigYAML string) (*transformTest, error) {
	e := transformTest{
		logger:            l,
		transformerConfig: transformerConfigYAML,
		storageConfig:     framework.SharedEtcd(),
	}

	var err error
	if transformerConfigYAML != "" {
		if e.configDir, err = e.createEncryptionConfig(); err != nil {
			return nil, fmt.Errorf("error while creating KubeAPIServer encryption config: %v", err)
		}
	}

	if e.kubeAPIServer, err = kubeapiservertesting.StartTestServer(l, nil, e.getEncryptionOptions(), e.storageConfig); err != nil {
		return nil, fmt.Errorf("failed to start KubeAPI server: %v", err)
	}

	if e.restClient, err = kubernetes.NewForConfig(e.kubeAPIServer.ClientConfig); err != nil {
		return nil, fmt.Errorf("error while creating rest client: %v", err)
	}

	if e.ns, err = e.createNamespace(testNamespace); err != nil {
		return nil, err
	}

	if e.secret, err = e.createSecret(testSecret, e.ns.Name); err != nil {
		return nil, err
	}

	return &e, nil
}

func (e *transformTest) cleanUp() {
	os.RemoveAll(e.configDir)
	e.restClient.CoreV1().Namespaces().Delete(e.ns.Name, metav1.NewDeleteOptions(0))
	e.kubeAPIServer.TearDownFn()
}

func (e *transformTest) run(unSealSecretFunc unSealSecret, expectedEnvelopePrefix string) {
	response, err := e.readRawRecordFromETCD(e.getETCDPath())
	if err != nil {
		e.logger.Errorf("failed to read from etcd: %v", err)
		return
	}

	if !bytes.HasPrefix(response.Kvs[0].Value, []byte(expectedEnvelopePrefix)) {
		e.logger.Errorf("expected secret to be prefixed with %s, but got %s",
			expectedEnvelopePrefix, response.Kvs[0].Value)
		return
	}

	// etcd path of the key is used as the authenticated context - need to pass it to decrypt
	ctx := value.DefaultContext([]byte(e.getETCDPath()))
	// Envelope header precedes the payload
	sealedData := response.Kvs[0].Value[len(expectedEnvelopePrefix):]
	transformerConfig, err := e.getEncryptionConfig()
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

func (e *transformTest) benchmark(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := e.createSecret(e.secret.Name+strconv.Itoa(i), e.ns.Name)
		if err != nil {
			b.Fatalf("failed to create a secret: %v", err)
		}
	}
}

func (e *transformTest) getETCDPath() string {
	return fmt.Sprintf("/%s/secrets/%s/%s", e.storageConfig.Prefix, e.ns.Name, e.secret.Name)
}

func (e *transformTest) getRawSecretFromETCD() ([]byte, error) {
	secretETCDPath := e.getETCDPath()
	etcdResponse, err := e.readRawRecordFromETCD(secretETCDPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s from etcd: %v", secretETCDPath, err)
	}
	return etcdResponse.Kvs[0].Value, nil
}

func (e *transformTest) getEncryptionOptions() []string {
	if e.transformerConfig != "" {
		return []string{"--encryption-provider-config", path.Join(e.configDir, encryptionConfigFileName)}
	}

	return nil
}

func (e *transformTest) createEncryptionConfig() (string, error) {
	tempDir, err := ioutil.TempDir("", "secrets-encryption-test")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %v", err)
	}

	encryptionConfig := path.Join(tempDir, encryptionConfigFileName)

	if err := ioutil.WriteFile(encryptionConfig, []byte(e.transformerConfig), 0644); err != nil {
		os.RemoveAll(tempDir)
		return "", fmt.Errorf("error while writing encryption config: %v", err)
	}

	return tempDir, nil
}

func (e *transformTest) getEncryptionConfig() (*apiserverconfigv1.ProviderConfiguration, error) {
	var config apiserverconfigv1.EncryptionConfiguration
	err := yaml.Unmarshal([]byte(e.transformerConfig), &config)
	if err != nil {
		return nil, fmt.Errorf("failed to extract transformer key: %v", err)
	}

	return &config.Resources[0].Providers[0], nil
}

func (e *transformTest) createNamespace(name string) (*corev1.Namespace, error) {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}

	if _, err := e.restClient.CoreV1().Namespaces().Create(ns); err != nil {
		return nil, fmt.Errorf("unable to create testing namespace %v", err)
	}

	return ns, nil
}

func (e *transformTest) createSecret(name, namespace string) (*corev1.Secret, error) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			secretKey: []byte(secretVal),
		},
	}
	if _, err := e.restClient.CoreV1().Secrets(secret.Namespace).Create(secret); err != nil {
		return nil, fmt.Errorf("error while writing secret: %v", err)
	}

	return secret, nil
}

func (e *transformTest) readRawRecordFromETCD(path string) (*clientv3.GetResponse, error) {
	_, etcdClient, err := integration.GetEtcdClients(e.kubeAPIServer.ServerOpts.Etcd.StorageConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %v", err)
	}
	response, err := etcdClient.Get(context.Background(), path, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve secret from etcd %v", err)
	}

	return response, nil
}

func (e *transformTest) printMetrics() error {
	e.logger.Logf("Transformation Metrics:")
	metrics, err := prometheus.DefaultGatherer.Gather()
	if err != nil {
		return fmt.Errorf("failed to gather metrics: %s", err)
	}

	for _, mf := range metrics {
		if strings.HasPrefix(*mf.Name, metricsPrefix) {
			e.logger.Logf("%s", *mf.Name)
			for _, metric := range mf.GetMetric() {
				e.logger.Logf("%v", metric)
			}
		}
	}

	return nil
}

func contains(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}
