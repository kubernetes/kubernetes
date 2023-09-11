/*
Copyright 2023 The Kubernetes Authors.

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

package apiserver

import (
	"context"
	"embed"
	"path"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/yaml"

	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

//go:embed fixtures-117260/*
var pr117260Fixtures embed.FS

// TestPR117260 is a reproducer for https://github.com/kubernetes/kubernetes/issues/117260
func TestPR117260(t *testing.T) {
	ctx, cancelFn := context.WithCancel(context.Background())
	t.Cleanup(cancelFn)

	server, err := kubeapiservertesting.StartTestServer(t, kubeapiservertesting.NewDefaultTestServerOptions(), nil, framework.SharedEtcd())
	if err != nil {
		t.Fatal(err)
	}
	defer server.TearDownFn()

	t.Log("Creating flux-system namespace")
	client, err := kubernetes.NewForConfig(server.ClientConfig)
	require.NoError(t, err)
	_, err = client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: "flux-system"}}, metav1.CreateOptions{})
	require.NoError(t, err)

	t.Log("Creating CRD")
	bs, err := pr117260Fixtures.ReadFile("fixtures-117260/crd.yaml")
	require.NoError(t, err)

	crd := &apiextensionsv1.CustomResourceDefinition{}
	err = yaml.Unmarshal(bs, crd)
	require.NoError(t, err)

	var apiExtensionClient *apiextensionsclient.Clientset
	apiExtensionClient, err = apiextensionsclient.NewForConfig(server.ClientConfig)
	require.NoError(t, err)
	_, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, crd, metav1.CreateOptions{})
	require.NoError(t, err)

	dynClient, err := dynamic.NewForConfig(server.ClientConfig)
	require.NoError(t, err)
	v1 := schema.GroupVersionResource{Group: "source.toolkit.fluxcd.io", Version: "v1", Resource: "gitrepositories"}
	err = wait.PollUntilContextTimeout(ctx, time.Second, wait.ForeverTestTimeout, true, func(ctx context.Context) (done bool, err error) {
		obj := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "source.toolkit.fluxcd.io/v1",
				"kind":       "GitRepository",
				"metadata": map[string]interface{}{
					"name":      "seed",
					"namespace": "flux-system",
				},
			},
		}
		_, err = dynClient.Resource(v1).Namespace("flux-system").Apply(ctx, "seed", obj, metav1.ApplyOptions{FieldManager: "integation-test"})
		if err != nil {
			t.Logf("Waiting for CRD to be ready: %v", err)
			return false, nil
		}
		return true, nil
	})
	require.NoError(t, err)

	t.Log("Writing CR directly to etcd")
	bs, err = pr117260Fixtures.ReadFile("fixtures-117260/etcd.json")
	require.NoError(t, err)

	key := path.Join("/", server.EtcdStoragePrefix, "source.toolkit.fluxcd.io", "gitrepositories", "flux-system", "flux-monitoring")
	_, err = server.EtcdClient.Put(ctx, key, string(bs))
	require.NoError(t, err)

	var obj *unstructured.Unstructured
	obj, err = dynClient.Resource(v1).Namespace("flux-system").Get(ctx, "flux-monitoring", metav1.GetOptions{})
	require.NoError(t, err)
	t.Logf("Found CR %s/%s through the apiserver as v1beta1", obj.GetNamespace(), obj.GetName())

	t.Log("Applying same CR")
	v1beta2 := schema.GroupVersionResource{Group: "source.toolkit.fluxcd.io", Version: "v1beta2", Resource: "gitrepositories"}
	bs, err = pr117260Fixtures.ReadFile("fixtures-117260/apply.yaml")
	require.NoError(t, err)
	patch := &unstructured.Unstructured{}
	err = yaml.Unmarshal(bs, &patch.Object)
	require.NoError(t, err)
	for i := 0; i < 100; i++ {
		_, err = dynClient.Resource(v1beta2).Namespace("flux-system").Apply(ctx, "flux-monitoring", patch, metav1.ApplyOptions{
			DryRun:       []string{"All"},
			Force:        false,
			FieldManager: "kustomize-controller",
		})
		require.NoError(t, err)
	}
}
