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
	"context"
	"fmt"
	"testing"

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
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/util/retry"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

func TestCBORServingEnablement(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
	}{
		{name: "enabled", enabled: true},
		{name: "disabled", enabled: false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.TestOnlyFeatureGate, features.TestOnlyCBORServingAndStorage, tc.enabled)

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
