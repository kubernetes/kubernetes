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

package integration

import (
	"bytes"
	"context"
	"fmt"
	"path"
	"testing"

	"github.com/google/uuid"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	cbor "k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestCBORStorageEnablement(t *testing.T) {
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "bars.mygroup.example.com"},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "mygroup.example.com",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name:    "v1beta1",
				Served:  true,
				Storage: true,
				Schema:  fixtures.AllowAllSchema(),
			}},
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "bars",
				Singular: "bar",
				Kind:     "Bar",
				ListKind: "BarList",
			},
			Scope: apiextensionsv1.ClusterScoped,
		},
	}

	etcdPrefix := uuid.New().String()

	func() {
		t.Log("starting server with feature gate disabled")
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, false)
		tearDown, apiExtensionsClientset, dynamicClient, etcdClient, _, err := fixtures.StartDefaultServerWithClientsAndEtcd(t, "--etcd-prefix", etcdPrefix)
		if err != nil {
			t.Fatal(err)
		}
		defer tearDown()

		if _, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionsClientset, dynamicClient); err != nil {
			t.Fatal(err)
		}

		if _, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "bars"}).Create(
			context.TODO(),
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "mygroup.example.com/v1beta1",
					"kind":       "Bar",
					"metadata": map[string]interface{}{
						"name": "test-storage-json",
					},
				}},
			metav1.CreateOptions{},
		); err != nil {
			t.Fatal(err)
		}

		response, err := etcdClient.KV.Get(context.TODO(), path.Join("/", etcdPrefix, crd.Spec.Group, crd.Spec.Names.Plural, "test-storage-json"))
		if err != nil {
			t.Fatal(err)
		}
		if n := len(response.Kvs); n != 1 {
			t.Fatalf("expected 1 kv, got %d", n)
		}
		if err := json.Unmarshal(response.Kvs[0].Value, new(interface{})); err != nil {
			t.Fatalf("failed to decode stored custom resource as json: %v", err)
		}
	}()

	func() {
		t.Log("starting server with feature gate enabled")
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
		tearDown, _, dynamicClient, etcdClient, _, err := fixtures.StartDefaultServerWithClientsAndEtcd(t, "--etcd-prefix", etcdPrefix)
		if err != nil {
			t.Fatal(err)
		}
		defer tearDown()

		if _, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "bars"}).Create(
			context.TODO(),
			&unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "mygroup.example.com/v1beta1",
					"kind":       "Bar",
					"metadata": map[string]interface{}{
						"name": "test-storage-cbor",
					},
				}},
			metav1.CreateOptions{},
		); err != nil {
			t.Fatal(err)
		}

		response, err := etcdClient.KV.Get(context.TODO(), path.Join("/", etcdPrefix, crd.Spec.Group, crd.Spec.Names.Plural, "test-storage-cbor"))
		if err != nil {
			t.Fatal(err)
		}
		if n := len(response.Kvs); n != 1 {
			t.Fatalf("expected 1 kv, got %d", n)
		}
		if !bytes.HasPrefix(response.Kvs[0].Value, []byte{0xd9, 0xd9, 0xf7}) {
			// Check for the encoding of the "self-described CBOR" tag which acts as a
			// "magic number" for distinguishing CBOR from JSON. Valid CBOR data items
			// do not require this prefix, but the Kubernetes CBOR serializer guarantees
			// it.
			t.Fatalf(`stored custom resource lacks required "self-described CBOR" tag (prefix 0x%x)`, response.Kvs[0].Value[:3])
		}
		if err := cbor.Unmarshal(response.Kvs[0].Value, new(interface{})); err != nil {
			t.Fatalf("failed to decode stored custom resource as cbor: %v", err)
		}

		for _, name := range []string{"test-storage-json", "test-storage-cbor"} {
			_, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "bars"}).Get(context.TODO(), name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("failed to get cr %q: %v", name, err)
			}
		}
	}()

	func() {
		t.Log("starting server with feature gate disabled")
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, false)
		tearDown, _, dynamicClient, _, _, err := fixtures.StartDefaultServerWithClientsAndEtcd(t, "--etcd-prefix", etcdPrefix)
		if err != nil {
			t.Fatal(err)
		}
		defer tearDown()

		for _, name := range []string{"test-storage-json", "test-storage-cbor"} {
			_, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "bars"}).Get(context.TODO(), name, metav1.GetOptions{})
			if err != nil {
				t.Errorf("failed to get cr %q: %v", name, err)
			}
		}
	}()

}

func TestCBORServingEnablement(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
	}{
		{name: "enabled", enabled: true},
		{name: "disabled", enabled: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, tc.enabled)

			tearDown, config, _, err := fixtures.StartDefaultServer(t)
			if err != nil {
				t.Fatal(err)
			}
			defer tearDown()

			apiExtensionsClientset, err := apiextensionsclientset.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}
			dynamicClient, err := dynamic.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}

			crd := &apiextensionsv1.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: "foos.mygroup.example.com"},
				Spec: apiextensionsv1.CustomResourceDefinitionSpec{
					Group: "mygroup.example.com",
					Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
						Name:    "v1beta1",
						Served:  true,
						Storage: true,
						Schema:  fixtures.AllowAllSchema(),
						Subresources: &apiextensionsv1.CustomResourceSubresources{
							Status: &apiextensionsv1.CustomResourceSubresourceStatus{},
							Scale: &apiextensionsv1.CustomResourceSubresourceScale{
								SpecReplicasPath:   ".spec.replicas",
								StatusReplicasPath: ".status.replicas",
							},
						},
					}},
					Names: apiextensionsv1.CustomResourceDefinitionNames{
						Plural:   "foos",
						Singular: "foo",
						Kind:     "Foo",
						ListKind: "FooList",
					},
					Scope: apiextensionsv1.ClusterScoped,
				},
			}
			if _, err = fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionsClientset, dynamicClient); err != nil {
				t.Fatal(err)
			}
			cr, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "foos"}).Create(
				context.TODO(),
				&unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "mygroup.example.com/v1beta1",
						"kind":       "Foo",
						"metadata": map[string]interface{}{
							"name": fmt.Sprintf("test-cbor-%s", tc.name),
						},
						"spec": map[string]interface{}{
							"replicas": int64(0),
						},
						"status": map[string]interface{}{
							"replicas": int64(0),
						},
					}},
				metav1.CreateOptions{},
			)
			if err != nil {
				t.Fatal(err)
			}

			config = rest.CopyConfig(config)
			config.NegotiatedSerializer = serializer.NewCodecFactory(runtime.NewScheme()).WithoutConversion()
			config.APIPath = "/apis"
			config.GroupVersion = &schema.GroupVersion{Group: "mygroup.example.com", Version: "v1beta1"}
			restClient, err := rest.RESTClientFor(config)
			if err != nil {
				t.Fatal(err)
			}

			for _, subresource := range []string{"", "status", "scale"} {
				err = restClient.Get().
					Resource(crd.Spec.Names.Plural).
					SubResource(subresource).
					Name(cr.GetName()).
					SetHeader("Accept", "application/cbor").
					Do(context.TODO()).Error()
				switch {
				case tc.enabled && err == nil:
					// ok
				case !tc.enabled && errors.IsNotAcceptable(err):
					// ok
				default:
					t.Errorf("unexpected error on read (subresource %q): %v", subresource, err)
				}
			}

			createBody, err := cbor.Marshal(map[string]interface{}{
				"apiVersion": "mygroup.example.com/v1beta1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": fmt.Sprintf("test-cbor-%s-2", tc.name),
				},
				"spec": map[string]interface{}{
					"replicas": int64(0),
				},
				"status": map[string]interface{}{
					"replicas": int64(0),
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			err = restClient.Post().
				Resource(crd.Spec.Names.Plural).
				SetHeader("Content-Type", "application/cbor").
				Body(createBody).
				Do(context.TODO()).Error()
			switch {
			case tc.enabled && err == nil:
				// ok
			case !tc.enabled && errors.IsUnsupportedMediaType(err):
				// ok
			default:
				t.Errorf("unexpected error on write: %v", err)
			}

			scaleBody, err := cbor.Marshal(map[string]interface{}{
				"apiVersion": "autoscaling/v1",
				"kind":       "Scale",
				"metadata": map[string]interface{}{
					"name": cr.GetName(),
				},
				"spec": map[string]interface{}{
					"replicas": int64(0),
				},
				"status": map[string]interface{}{
					"replicas": int64(0),
				},
			})
			if err != nil {
				t.Fatal(err)
			}
			err = restClient.Put().
				Resource(crd.Spec.Names.Plural).
				SubResource("scale").
				Name(cr.GetName()).
				SetHeader("Content-Type", "application/cbor").
				Body(scaleBody).
				Do(context.TODO()).Error()
			switch {
			case tc.enabled && err == nil:
				// ok
			case !tc.enabled && errors.IsUnsupportedMediaType(err):
				// ok
			default:
				t.Errorf("unexpected error on scale write: %v", err)
			}

			err = retry.RetryOnConflict(retry.DefaultBackoff, func() error {
				latest, err := dynamicClient.Resource(schema.GroupVersionResource{Group: "mygroup.example.com", Version: "v1beta1", Resource: "foos"}).Get(context.TODO(), cr.GetName(), metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}

				statusBody, err := cbor.Marshal(latest.Object)
				if err != nil {
					t.Fatal(err)
				}

				return restClient.Put().
					Resource(crd.Spec.Names.Plural).
					SubResource("status").
					Name(cr.GetName()).
					SetHeader("Content-Type", "application/cbor").
					Body(statusBody).
					Do(context.TODO()).Error()
			})
			switch {
			case tc.enabled && err == nil:
				// ok
			case !tc.enabled && errors.IsUnsupportedMediaType(err):
				// ok
			default:
				t.Fatalf("unexpected error on status write: %v", err)
			}
		})
	}
}
