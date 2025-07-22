/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"
	"time"

	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/integration/etcd"
)

func createResources(t *testing.T, test *transformTest,
	group,
	version,
	kind,
	resource,
	name,
	namespace string,
) {
	switch resource {
	case "pods":
		_, err := test.createPod(namespace, dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig))
		if err != nil {
			t.Fatalf("Failed to create test pod, error: %v, name: %s, ns: %s", err, name, namespace)
		}
	case "configmaps":
		_, err := test.createConfigMap(name, namespace)
		if err != nil {
			t.Fatalf("Failed to create test configmap, error: %v, name: %s, ns: %s", err, name, namespace)
		}
	default:
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		t.Cleanup(cancel)

		gvr := schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
		data := etcd.GetEtcdStorageData()[gvr]
		stub := data.Stub
		dynamicClient, obj, err := etcd.JSONToUnstructured(stub, namespace, &meta.RESTMapping{
			Resource:         gvr,
			GroupVersionKind: gvr.GroupVersion().WithKind(kind),
			Scope:            meta.RESTScopeRoot,
		}, dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig))
		if err != nil {
			t.Fatal(err)
		}
		_, err = dynamicClient.Create(ctx, obj, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if _, err := dynamicClient.Get(ctx, obj.GetName(), metav1.GetOptions{}); err != nil {
			t.Fatalf("object should exist: %v", err)
		}
	}
}

func TestEncryptSupportedForAllResourceTypes(t *testing.T) {
	// check resources provided by the three servers that we have wired together
	// - pods and configmaps from KAS
	// - CRDs and CRs from API extensions
	// - API services from aggregator
	encryptionConfig := `
kind: EncryptionConfiguration
apiVersion: apiserver.config.k8s.io/v1
resources:
- resources:
  - pods
  - configmaps
  - customresourcedefinitions.apiextensions.k8s.io
  - pandas.awesome.bears.com
  - apiservices.apiregistration.k8s.io
  providers:
  - aescbc:
      keys:
      - name: key1
        secret: c2VjcmV0IGlzIHNlY3VyZQ==
`
	test, err := newTransformTest(t, transformTestConfig{transformerConfigYAML: encryptionConfig})
	if err != nil {
		t.Fatalf("failed to start Kube API Server with encryptionConfig\n %s, error: %v", encryptionConfig, err)
	}
	t.Cleanup(test.cleanUp)

	// the storage registry for CRs is dynamic so create one to exercise the wiring
	etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig), false, etcd.GetCustomResourceDefinitionData()...)

	for _, tt := range []struct {
		group     string
		version   string
		kind      string
		resource  string
		name      string
		namespace string
	}{
		{"", "v1", "ConfigMap", "configmaps", "cm1", testNamespace},
		{"apiextensions.k8s.io", "v1", "CustomResourceDefinition", "customresourcedefinitions", "pandas.awesome.bears.com", ""},
		{"awesome.bears.com", "v1", "Panda", "pandas", "cr3panda", ""},
		{"apiregistration.k8s.io", "v1", "APIService", "apiservices", "as2.foo.com", ""},
		{"", "v1", "Pod", "pods", "pod1", testNamespace},
	} {
		tt := tt
		t.Run(tt.resource, func(t *testing.T) {
			t.Parallel()

			createResources(t, test, tt.group, tt.version, tt.kind, tt.resource, tt.name, tt.namespace)
			test.runResource(t, unSealWithCBCTransformer, aesCBCPrefix, tt.group, tt.version, tt.resource, tt.name, tt.namespace)
		})
	}
}
