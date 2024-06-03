/*
Copyright 2024 The Kubernetes Authors.

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

package controlplane

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	genericcontrolplanetesting "k8s.io/kubernetes/pkg/controlplane/apiserver/samples/generic/server/testing"
	minimalcontrolplanetesting "k8s.io/kubernetes/pkg/controlplane/apiserver/samples/minimal/server/testing"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestGenericControlplaneStartUp(t *testing.T) {
	server, err := genericcontrolplanetesting.StartTestServer(t, genericcontrolplanetesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Create cluster scoped resource: namespace %q", "test")
	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Logf("Create namesapces resource: configmap %q", "config")
	if _, err := client.CoreV1().ConfigMaps("test").Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "config"},
		Data:       map[string]string{"foo": "bar"},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Logf("Create CRD")
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(server.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)
	if _, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "awesome.bears.com", Version: "v1", Resource: "pandas"}).Create(ctx, &unstructured.Unstructured{
		Object: map[string]interface{}{
			"metadata": map[string]interface{}{
				"name": "baobao",
			},
		},
	}, metav1.CreateOptions{}); err != nil {
		t.Error(err)
	}
}

func TestMinimalControlplaneStartUp(t *testing.T) {
	server, err := minimalcontrolplanetesting.StartTestServer(t, minimalcontrolplanetesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)

	client, err := kubernetes.NewForConfig(server.ClientConfig)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Create cluster scoped resource: namespace %q", "test")
	if _, err := client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "test"}}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	t.Logf("Create namesapces resource: configmap %q", "config")
	if _, err := client.CoreV1().ConfigMaps("test").Create(ctx, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "config"},
		Data:       map[string]string{"foo": "bar"},
	}, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
}
