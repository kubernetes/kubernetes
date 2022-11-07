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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strconv"
	"strings"
	"testing"

	"k8s.io/klog/v2"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/component-base/metrics/legacyregistry"
	"sigs.k8s.io/yaml"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	apiserverconfigv1 "k8s.io/apiserver/pkg/apis/config/v1"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	secretKey                = "api_key"
	secretVal                = "086a7ffc-0225-11e8-ba89-0ed5f89f718b" // Fake value for testing.
	encryptionConfigFileName = "encryption.conf"
	testNamespace            = "secret-encryption-test"
	testSecret               = "test-secret"
	metricsPrefix            = "apiserver_storage_"
	configMapKey             = "foo"
	configMapVal             = "bar"

	// precomputed key and secret for use with AES CBC
	// this looks exactly the same as the AES GCM secret but with a different value
	oldAESCBCKey = "e0/+tts8FS254BZimFZWtUsOCOUDSkvzB72PyimMlkY="
	oldSecret    = "azhzAAoMCgJ2MRIGU2VjcmV0En4KXwoLdGVzdC1zZWNyZXQSABoWc2VjcmV0LWVuY3J5cHRpb24tdGVzdCIAKiQ3MmRmZTVjNC0xNDU2LTQyMzktYjFlZC1hZGZmYTJmMWY3YmEyADgAQggI5Jy/7wUQAHoAEhMKB2FwaV9rZXkSCPCfpJfwn5C8GgZPcGFxdWUaACIA"
	oldSecretVal = "\xf0\x9f\xa4\x97\xf0\x9f\x90\xbc"
)

type unSealSecret func(ctx context.Context, cipherText []byte, dataCtx value.Context, config apiserverconfigv1.ProviderConfiguration) ([]byte, error)

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
	klog.Infof("Started kube-apiserver %v", e.kubeAPIServer.ClientConfig.Host)

	if e.restClient, err = kubernetes.NewForConfig(e.kubeAPIServer.ClientConfig); err != nil {
		return nil, fmt.Errorf("error while creating rest client: %v", err)
	}

	if e.ns, err = e.createNamespace(testNamespace); err != nil {
		return nil, err
	}

	return &e, nil
}

func (e *transformTest) cleanUp() {
	os.RemoveAll(e.configDir)
	e.restClient.CoreV1().Namespaces().Delete(context.TODO(), e.ns.Name, *metav1.NewDeleteOptions(0))
	e.kubeAPIServer.TearDownFn()
}

func (e *transformTest) runResource(l kubeapiservertesting.Logger, unSealSecretFunc unSealSecret, expectedEnvelopePrefix,
	group,
	version,
	resource,
	name,
	namespaceName string,
) {
	response, err := e.readRawRecordFromETCD(e.getETCDPathForResource(e.storageConfig.Prefix, group, resource, name, namespaceName))
	if err != nil {
		l.Errorf("failed to read from etcd: %v", err)
		return
	}

	if !bytes.HasPrefix(response.Kvs[0].Value, []byte(expectedEnvelopePrefix)) {
		l.Errorf("expected data to be prefixed with %s, but got %s",
			expectedEnvelopePrefix, response.Kvs[0].Value)
		return
	}

	// etcd path of the key is used as the authenticated context - need to pass it to decrypt
	ctx := context.Background()
	dataCtx := value.DefaultContext(e.getETCDPathForResource(e.storageConfig.Prefix, group, resource, name, namespaceName))
	// Envelope header precedes the cipherTextPayload
	sealedData := response.Kvs[0].Value[len(expectedEnvelopePrefix):]
	transformerConfig, err := e.getEncryptionConfig()
	if err != nil {
		l.Errorf("failed to parse transformer config: %v", err)
	}
	v, err := unSealSecretFunc(ctx, sealedData, dataCtx, *transformerConfig)
	if err != nil {
		l.Errorf("failed to unseal secret: %v", err)
		return
	}
	if resource == "secrets" {
		if !strings.Contains(string(v), secretVal) {
			l.Errorf("expected %q after decryption, but got %q", secretVal, string(v))
		}
	} else if resource == "configmaps" {
		if !strings.Contains(string(v), configMapVal) {
			l.Errorf("expected %q after decryption, but got %q", configMapVal, string(v))
		}
	} else {
		if !strings.Contains(string(v), name) {
			l.Errorf("expected %q after decryption, but got %q", name, string(v))
		}
	}

	// Data should be un-enveloped on direct reads from Kube API Server.
	if resource == "secrets" {
		s, err := e.restClient.CoreV1().Secrets(testNamespace).Get(context.TODO(), testSecret, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get Secret from %s, err: %v", testNamespace, err)
		}
		if secretVal != string(s.Data[secretKey]) {
			l.Errorf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
		}
	} else if resource == "configmaps" {
		s, err := e.restClient.CoreV1().ConfigMaps(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get ConfigMap from %s, err: %v", namespaceName, err)
		}
		if configMapVal != string(s.Data[configMapKey]) {
			l.Errorf("expected %s from KubeAPI, but got %s", configMapVal, string(s.Data[configMapKey]))
		}
	} else if resource == "pods" {
		p, err := e.restClient.CoreV1().Pods(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get Pod from %s, err: %v", namespaceName, err)
		}
		if p.Name != name {
			l.Errorf("expected %s from KubeAPI, but got %s", name, p.Name)
		}
	} else {
		l.Logf("Get object with dynamic client")
		fooResource := schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
		obj, err := dynamic.NewForConfigOrDie(e.kubeAPIServer.ClientConfig).Resource(fooResource).Namespace(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("Failed to get test instance: %v, name: %s", err, name)
		}
		if obj.GetObjectKind().GroupVersionKind().Group == group && obj.GroupVersionKind().Version == version && obj.GetKind() == resource && obj.GetNamespace() == namespaceName && obj.GetName() != name {
			l.Errorf("expected %s from KubeAPI, but got %s", name, obj.GetName())
		}
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

func (e *transformTest) getETCDPathForResource(storagePrefix, group, resource, name, namespaceName string) string {
	groupResource := resource
	if group != "" {
		groupResource = fmt.Sprintf("%s/%s", group, resource)
	}
	if namespaceName == "" {
		return fmt.Sprintf("/%s/%s/%s", storagePrefix, groupResource, name)
	}
	return fmt.Sprintf("/%s/%s/%s/%s", storagePrefix, groupResource, namespaceName, name)
}

func (e *transformTest) getRawSecretFromETCD() ([]byte, error) {
	secretETCDPath := e.getETCDPathForResource(e.storageConfig.Prefix, "", "secrets", e.secret.Name, e.secret.Namespace)
	etcdResponse, err := e.readRawRecordFromETCD(secretETCDPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s from etcd: %v", secretETCDPath, err)
	}
	return etcdResponse.Kvs[0].Value, nil
}

func (e *transformTest) getEncryptionOptions() []string {
	if e.transformerConfig != "" {
		return []string{
			"--encryption-provider-config", path.Join(e.configDir, encryptionConfigFileName),
			"--disable-admission-plugins", "ServiceAccount"}
	}

	return nil
}

func (e *transformTest) createEncryptionConfig() (string, error) {
	tempDir, err := os.MkdirTemp("", "secrets-encryption-test")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %v", err)
	}

	encryptionConfig := path.Join(tempDir, encryptionConfigFileName)

	if err := os.WriteFile(encryptionConfig, []byte(e.transformerConfig), 0644); err != nil {
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

	if _, err := e.restClient.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{}); err != nil {
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
	if _, err := e.restClient.CoreV1().Secrets(secret.Namespace).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while writing secret: %v", err)
	}

	return secret, nil
}

func (e *transformTest) createConfigMap(name, namespace string) (*corev1.ConfigMap, error) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string]string{
			configMapKey: configMapVal,
		},
	}
	if _, err := e.restClient.CoreV1().ConfigMaps(cm.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while writing configmap: %v", err)
	}

	return cm, nil
}

func gvr(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}

func createResource(client dynamic.Interface, gvr schema.GroupVersionResource, ns string) (*unstructured.Unstructured, error) {
	stubObj, err := getStubObj(gvr)
	if err != nil {
		return nil, err
	}
	return client.Resource(gvr).Namespace(ns).Create(context.TODO(), stubObj, metav1.CreateOptions{})
}

func getStubObj(gvr schema.GroupVersionResource) (*unstructured.Unstructured, error) {
	stub := ""
	if data, ok := etcd.GetEtcdStorageDataForNamespace(testNamespace)[gvr]; ok {
		stub = data.Stub
	}
	if len(stub) == 0 {
		return nil, fmt.Errorf("no stub data for %#v", gvr)
	}

	stubObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := json.Unmarshal([]byte(stub), &stubObj.Object); err != nil {
		return nil, fmt.Errorf("error unmarshaling stub for %#v: %v", gvr, err)
	}
	return stubObj, nil
}

func (e *transformTest) createPod(namespace string, dynamicInterface dynamic.Interface) (*unstructured.Unstructured, error) {
	podGVR := gvr("", "v1", "pods")
	pod, err := createResource(dynamicInterface, podGVR, namespace)
	if err != nil {
		return nil, fmt.Errorf("error while writing pod: %v", err)
	}
	return pod, nil
}

func (e *transformTest) readRawRecordFromETCD(path string) (*clientv3.GetResponse, error) {
	rawClient, etcdClient, err := integration.GetEtcdClients(e.kubeAPIServer.ServerOpts.Etcd.StorageConfig.Transport)
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %v", err)
	}
	// kvClient is a wrapper around rawClient and to avoid leaking goroutines we need to
	// close the client (which we can do by closing rawClient).
	defer rawClient.Close()

	response, err := etcdClient.Get(context.Background(), path, clientv3.WithPrefix())
	if err != nil {
		return nil, fmt.Errorf("failed to retrieve secret from etcd %v", err)
	}

	return response, nil
}

func (e *transformTest) writeRawRecordToETCD(path string, data []byte) (*clientv3.PutResponse, error) {
	rawClient, etcdClient, err := integration.GetEtcdClients(e.kubeAPIServer.ServerOpts.Etcd.StorageConfig.Transport)
	if err != nil {
		return nil, fmt.Errorf("failed to create etcd client: %v", err)
	}
	// kvClient is a wrapper around rawClient and to avoid leaking goroutines we need to
	// close the client (which we can do by closing rawClient).
	defer rawClient.Close()

	response, err := etcdClient.Put(context.Background(), path, string(data))
	if err != nil {
		return nil, fmt.Errorf("failed to write secret to etcd %v", err)
	}

	return response, nil
}

func (e *transformTest) printMetrics() error {
	e.logger.Logf("Transformation Metrics:")
	metrics, err := legacyregistry.DefaultGatherer.Gather()
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
