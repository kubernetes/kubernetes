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
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"sigs.k8s.io/yaml"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverv1 "k8s.io/apiserver/pkg/apis/apiserver/v1"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/klog/v2"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/pointer"
)

const (
	secretKey                = "api_key"
	secretVal                = "086a7ffc-0225-11e8-ba89-0ed5f89f718b" // Fake value for testing.
	encryptionConfigFileName = "encryption.conf"
	metricsPrefix            = "apiserver_storage_"
	configMapKey             = "foo"
	configMapVal             = "bar"

	TestNamespace = "secret-encryption-test"
)

type UnSealSecret func(ctx context.Context, cipherText []byte, dataCtx value.Context, config apiserverv1.ProviderConfiguration) ([]byte, error)

type TransformTest struct {
	Logger            kubeapiservertesting.Logger
	StorageConfig     *storagebackend.Config
	ConfigDir         string
	TransformerConfig string
	KubeAPIServer     kubeapiservertesting.TestServer
	RestClient        *kubernetes.Clientset
	ns                *corev1.Namespace
	Secret            *corev1.Secret
}

func NewTransformTest(l kubeapiservertesting.Logger, transformerConfigYAML string, reload bool, configDir string, storageConfig *storagebackend.Config) (*TransformTest, error) {
	if storageConfig == nil {
		storageConfig = framework.SharedEtcd()
	}
	e := TransformTest{
		Logger:            l,
		TransformerConfig: transformerConfigYAML,
		StorageConfig:     storageConfig,
	}

	var err error
	// create config dir with provided config yaml
	if transformerConfigYAML != "" && configDir == "" {
		if e.ConfigDir, err = e.createEncryptionConfig(); err != nil {
			e.CleanUp()
			return nil, fmt.Errorf("error while creating KubeAPIServer encryption config: %w", err)
		}
	} else {
		// ConfigDir already exists. api-server must be restarting with existing encryption config
		e.ConfigDir = configDir
	}
	configFile := filepath.Join(e.ConfigDir, encryptionConfigFileName)
	_, err = os.ReadFile(configFile)
	if err != nil {
		e.CleanUp()
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	if e.KubeAPIServer, err = kubeapiservertesting.StartTestServer(l, nil, e.getEncryptionOptions(reload), e.StorageConfig); err != nil {
		e.CleanUp()
		return nil, fmt.Errorf("failed to start KubeAPI server: %w", err)
	}
	klog.Infof("Started kube-apiserver %v", e.KubeAPIServer.ClientConfig.Host)

	if e.RestClient, err = kubernetes.NewForConfig(e.KubeAPIServer.ClientConfig); err != nil {
		e.CleanUp()
		return nil, fmt.Errorf("error while creating rest client: %w", err)
	}

	if e.ns, err = e.CreateNamespace(TestNamespace); err != nil {
		e.CleanUp()
		return nil, err
	}

	if transformerConfigYAML != "" && reload {
		// when reloading is enabled, this healthz endpoint is always present
		MustBeHealthy(l, "/kms-providers", "ok", e.KubeAPIServer.ClientConfig)
		MustNotHaveLivez(l, "/kms-providers", "404 page not found", e.KubeAPIServer.ClientConfig)

		// excluding healthz endpoints even if they do not exist should work
		MustBeHealthy(l, "", `warn: some health checks cannot be excluded: no matches for "kms-provider-0","kms-provider-1","kms-provider-2","kms-provider-3"`,
			e.KubeAPIServer.ClientConfig, "kms-provider-0", "kms-provider-1", "kms-provider-2", "kms-provider-3")
	}

	return &e, nil
}

func (e *TransformTest) CleanUp() {
	if e.ConfigDir != "" {
		_ = os.RemoveAll(e.ConfigDir)
	}

	if e.KubeAPIServer.ClientConfig != nil {
		e.ShutdownAPIServer()
	}
}

func (e *TransformTest) ShutdownAPIServer() {
	e.KubeAPIServer.TearDownFn()
}

func (e *TransformTest) RunResource(l kubeapiservertesting.Logger, unSealSecretFunc UnSealSecret, expectedEnvelopePrefix,
	group,
	version,
	resource,
	name,
	namespaceName string,
) {
	response, err := e.ReadRawRecordFromETCD(e.GetETCDPathForResource(e.StorageConfig.Prefix, group, resource, name, namespaceName))
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
	dataCtx := value.DefaultContext(e.GetETCDPathForResource(e.StorageConfig.Prefix, group, resource, name, namespaceName))
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
		s, err := e.RestClient.CoreV1().Secrets(TestNamespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get Secret from %s, err: %v", TestNamespace, err)
		}
		if secretVal != string(s.Data[secretKey]) {
			l.Errorf("expected %s from KubeAPI, but got %s", secretVal, string(s.Data[secretKey]))
		}
	} else if resource == "configmaps" {
		s, err := e.RestClient.CoreV1().ConfigMaps(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get ConfigMap from %s, err: %v", namespaceName, err)
		}
		if configMapVal != string(s.Data[configMapKey]) {
			l.Errorf("expected %s from KubeAPI, but got %s", configMapVal, string(s.Data[configMapKey]))
		}
	} else if resource == "pods" {
		p, err := e.RestClient.CoreV1().Pods(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("failed to get Pod from %s, err: %v", namespaceName, err)
		}
		if p.Name != name {
			l.Errorf("expected %s from KubeAPI, but got %s", name, p.Name)
		}
	} else {
		l.Logf("Get object with dynamic client")
		fooResource := schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
		obj, err := dynamic.NewForConfigOrDie(e.KubeAPIServer.ClientConfig).Resource(fooResource).Namespace(namespaceName).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			l.Fatalf("Failed to get test instance: %v, name: %s", err, name)
		}
		if obj.GetObjectKind().GroupVersionKind().Group == group && obj.GroupVersionKind().Version == version && obj.GetKind() == resource && obj.GetNamespace() == namespaceName && obj.GetName() != name {
			l.Errorf("expected %s from KubeAPI, but got %s", name, obj.GetName())
		}
	}
}

func (e *TransformTest) Benchmark(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_, err := e.CreateSecret(e.Secret.Name+strconv.Itoa(i), e.ns.Name)
		if err != nil {
			b.Fatalf("failed to create a secret: %v", err)
		}
	}
}

func (e *TransformTest) GetETCDPathForResource(storagePrefix, group, resource, name, namespaceName string) string {
	groupResource := resource
	if group != "" {
		groupResource = fmt.Sprintf("%s/%s", group, resource)
	}
	if namespaceName == "" {
		return fmt.Sprintf("/%s/%s/%s", storagePrefix, groupResource, name)
	}
	return fmt.Sprintf("/%s/%s/%s/%s", storagePrefix, groupResource, namespaceName, name)
}

func (e *TransformTest) GetRawSecretFromETCD() ([]byte, error) {
	secretETCDPath := e.GetETCDPathForResource(e.StorageConfig.Prefix, "", "secrets", e.Secret.Name, e.Secret.Namespace)
	etcdResponse, err := e.ReadRawRecordFromETCD(secretETCDPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read %s from etcd: %v", secretETCDPath, err)
	}
	return etcdResponse.Kvs[0].Value, nil
}

func (e *TransformTest) getEncryptionOptions(reload bool) []string {
	if e.TransformerConfig != "" {
		return []string{
			"--encryption-provider-config", filepath.Join(e.ConfigDir, encryptionConfigFileName),
			fmt.Sprintf("--encryption-provider-config-automatic-reload=%v", reload),
			"--disable-admission-plugins", "ServiceAccount"}
	}

	return nil
}

func (e *TransformTest) createEncryptionConfig() (
	filePathForEncryptionConfig string,
	err error,
) {
	tempDir, err := os.MkdirTemp("", "secrets-encryption-test")
	if err != nil {
		return "", fmt.Errorf("failed to create temp directory: %v", err)
	}

	if err = os.WriteFile(filepath.Join(tempDir, encryptionConfigFileName), []byte(e.TransformerConfig), 0644); err != nil {
		os.RemoveAll(tempDir)
		return tempDir, fmt.Errorf("error while writing encryption config: %v", err)
	}

	return tempDir, nil
}

func (e *TransformTest) getEncryptionConfig() (*apiserverv1.ProviderConfiguration, error) {
	var config apiserverv1.EncryptionConfiguration
	err := yaml.Unmarshal([]byte(e.TransformerConfig), &config)
	if err != nil {
		return nil, fmt.Errorf("failed to extract transformer key: %v", err)
	}

	return &config.Resources[0].Providers[0], nil
}

func (e *TransformTest) CreateNamespace(name string) (*corev1.Namespace, error) {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
	}

	if _, err := e.RestClient.CoreV1().Namespaces().Create(context.TODO(), ns, metav1.CreateOptions{}); err != nil {
		if errors.IsAlreadyExists(err) {
			existingNs, err := e.RestClient.CoreV1().Namespaces().Get(context.TODO(), name, metav1.GetOptions{})
			if err != nil {
				return nil, fmt.Errorf("unable to get testing namespace, err: [%v]", err)
			}
			return existingNs, nil
		}
		return nil, fmt.Errorf("unable to create testing namespace, err: [%v]", err)
	}

	return ns, nil
}

func (e *TransformTest) CreateSecret(name, namespace string) (*corev1.Secret, error) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			secretKey: []byte(secretVal),
		},
	}
	if _, err := e.RestClient.CoreV1().Secrets(secret.Namespace).Create(context.TODO(), secret, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while writing secret: %v", err)
	}

	return secret, nil
}

func (e *TransformTest) CreateConfigMap(name, namespace string) (*corev1.ConfigMap, error) {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string]string{
			configMapKey: configMapVal,
		},
	}
	if _, err := e.RestClient.CoreV1().ConfigMaps(cm.Namespace).Create(context.TODO(), cm, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while writing configmap: %v", err)
	}

	return cm, nil
}

// create jobs
func (e *TransformTest) CreateJob(name, namespace string) (*batchv1.Job, error) {
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: batchv1.JobSpec{
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "test",
							Image: "test",
						},
					},
					RestartPolicy: corev1.RestartPolicyNever,
				},
			},
		},
	}
	if _, err := e.RestClient.BatchV1().Jobs(job.Namespace).Create(context.TODO(), job, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while creating job: %v", err)
	}

	return job, nil
}

// create deployment
func (e *TransformTest) CreateDeployment(name, namespace string) (*appsv1.Deployment, error) {
	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: pointer.Int32(2),
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{
					"app": "nginx",
				},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"app": "nginx",
					},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:  "nginx",
							Image: "nginx:1.17",
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									Protocol:      corev1.ProtocolTCP,
									ContainerPort: 80,
								},
							},
						},
					},
				},
			},
		},
	}
	if _, err := e.RestClient.AppsV1().Deployments(deployment.Namespace).Create(context.TODO(), deployment, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("error while creating deployment: %v", err)
	}

	return deployment, nil
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

func inplaceUpdateResource(client dynamic.Interface, gvr schema.GroupVersionResource, ns string, obj *unstructured.Unstructured) (*unstructured.Unstructured, error) {
	return client.Resource(gvr).Namespace(ns).Update(context.TODO(), obj, metav1.UpdateOptions{})
}

func getStubObj(gvr schema.GroupVersionResource) (*unstructured.Unstructured, error) {
	stub := ""
	if data, ok := etcd.GetEtcdStorageDataForNamespace(TestNamespace)[gvr]; ok {
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

func (e *TransformTest) CreatePod(namespace string, dynamicInterface dynamic.Interface) (*unstructured.Unstructured, error) {
	podGVR := gvr("", "v1", "pods")
	pod, err := createResource(dynamicInterface, podGVR, namespace)
	if err != nil {
		return nil, fmt.Errorf("error while writing pod: %v", err)
	}
	return pod, nil
}

func (e *TransformTest) DeletePod(namespace string, dynamicInterface dynamic.Interface) error {
	podGVR := gvr("", "v1", "pods")
	stubObj, err := getStubObj(podGVR)
	if err != nil {
		return err
	}
	return dynamicInterface.Resource(podGVR).Namespace(namespace).Delete(context.TODO(), stubObj.GetName(), metav1.DeleteOptions{})
}

func (e *TransformTest) InPlaceUpdatePod(namespace string, obj *unstructured.Unstructured, dynamicInterface dynamic.Interface) (*unstructured.Unstructured, error) {
	podGVR := gvr("", "v1", "pods")
	pod, err := inplaceUpdateResource(dynamicInterface, podGVR, namespace, obj)
	if err != nil {
		return nil, fmt.Errorf("error while writing pod: %v", err)
	}
	return pod, nil
}

func (e *TransformTest) ReadRawRecordFromETCD(path string) (*clientv3.GetResponse, error) {
	rawClient, etcdClient, err := integration.GetEtcdClients(e.KubeAPIServer.ServerOpts.Etcd.StorageConfig.Transport)
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

func (e *TransformTest) WriteRawRecordToETCD(path string, data []byte) (*clientv3.PutResponse, error) {
	rawClient, etcdClient, err := integration.GetEtcdClients(e.KubeAPIServer.ServerOpts.Etcd.StorageConfig.Transport)
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

func (e *TransformTest) PrintMetrics() error {
	e.Logger.Logf("Transformation Metrics:")
	metrics, err := legacyregistry.DefaultGatherer.Gather()
	if err != nil {
		return fmt.Errorf("failed to gather metrics: %s", err)
	}

	for _, mf := range metrics {
		if strings.HasPrefix(*mf.Name, metricsPrefix) {
			e.Logger.Logf("%s", *mf.Name)
			for _, metric := range mf.GetMetric() {
				e.Logger.Logf("%v", metric)
			}
		}
	}

	return nil
}

func MustBeHealthy(t kubeapiservertesting.Logger, checkName, wantBodyContains string, clientConfig *rest.Config, excludes ...string) {
	t.Helper()
	var restErr error
	pollErr := wait.PollUntilContextTimeout(context.TODO(), 1*time.Second, 5*time.Minute, true, func(ctx context.Context) (bool, error) {
		body, ok, err := getHealthz(checkName, clientConfig, excludes...)
		restErr = err
		if err != nil {
			return false, err
		}
		done := ok && strings.Contains(body, wantBodyContains)
		if !done {
			t.Logf("expected server check %q to be healthy with message %q but it is not: %s", checkName, wantBodyContains, body)
		}
		return done, nil
	})

	if pollErr != nil {
		t.Fatalf("failed to get the expected healthz status of OK for check: %s, error: %v, debug inner error: %v", checkName, pollErr, restErr)
	}
}

func MustBeUnHealthy(t kubeapiservertesting.Logger, checkName, wantBodyContains string, clientConfig *rest.Config, excludes ...string) {
	t.Helper()
	var restErr error
	pollErr := wait.PollUntilContextTimeout(context.TODO(), 1*time.Second, 5*time.Minute, true, func(ctx context.Context) (bool, error) {
		body, ok, err := getHealthz(checkName, clientConfig, excludes...)
		restErr = err
		if err != nil {
			return false, err
		}
		done := !ok && strings.Contains(body, wantBodyContains)
		if !done {
			t.Logf("expected server check %q to be unhealthy with message %q but it is not: %s", checkName, wantBodyContains, body)
		}
		return done, nil
	})

	if pollErr != nil {
		t.Fatalf("failed to get the expected healthz status of !OK for check: %s, error: %v, debug inner error: %v", checkName, pollErr, restErr)
	}
}

func MustNotHaveLivez(t kubeapiservertesting.Logger, checkName, wantBodyContains string, clientConfig *rest.Config, excludes ...string) {
	t.Helper()
	var restErr error
	pollErr := wait.PollUntilContextTimeout(context.TODO(), 1*time.Second, 5*time.Minute, true, func(ctx context.Context) (bool, error) {
		body, ok, err := getLivez(checkName, clientConfig, excludes...)
		restErr = err
		if err != nil {
			return false, err
		}
		done := !ok && strings.Contains(body, wantBodyContains)
		if !done {
			t.Logf("expected server check %q with message %q but it is not: %s", checkName, wantBodyContains, body)
		}
		return done, nil
	})

	if pollErr != nil {
		t.Fatalf("failed to get the expected livez status of !OK for check: %s, error: %v, debug inner error: %v", checkName, pollErr, restErr)
	}
}

func getHealthz(checkName string, clientConfig *rest.Config, excludes ...string) (string, bool, error) {
	client, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return "", false, fmt.Errorf("failed to create a client: %v", err)
	}

	req := client.CoreV1().RESTClient().Get().AbsPath(fmt.Sprintf("/healthz%v", checkName)).Param("verbose", "true")
	for _, exclude := range excludes {
		req.Param("exclude", exclude)
	}
	body, err := req.DoRaw(context.TODO()) // we can still have a response body during an error case
	return string(body), err == nil, nil
}

func getLivez(checkName string, clientConfig *rest.Config, excludes ...string) (string, bool, error) {
	client, err := kubernetes.NewForConfig(clientConfig)
	if err != nil {
		return "", false, fmt.Errorf("failed to create a client: %v", err)
	}

	req := client.CoreV1().RESTClient().Get().AbsPath(fmt.Sprintf("/livez%v", checkName)).Param("verbose", "true")
	for _, exclude := range excludes {
		req.Param("exclude", exclude)
	}
	body, err := req.DoRaw(context.TODO()) // we can still have a response body during an error case
	return string(body), err == nil, nil
}
