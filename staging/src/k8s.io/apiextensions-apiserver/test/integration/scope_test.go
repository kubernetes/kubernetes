/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
)

func TestHandlerScope(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	for _, scope := range []apiextensionsv1beta1.ResourceScope{apiextensionsv1beta1.ClusterScoped, apiextensionsv1beta1.NamespaceScoped} {
		t.Run(string(scope), func(t *testing.T) {

			crd := &apiextensionsv1beta1.CustomResourceDefinition{
				ObjectMeta: metav1.ObjectMeta{Name: strings.ToLower(string(scope)) + "s.test.apiextensions-apiserver.k8s.io"},
				Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
					Group:   "test.apiextensions-apiserver.k8s.io",
					Version: "v1beta1",
					Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
						Plural:   strings.ToLower(string(scope)) + "s",
						Singular: strings.ToLower(string(scope)),
						Kind:     string(scope),
						ListKind: string(scope) + "List",
					},
					Scope: scope,
				},
			}
			crd.Spec.Subresources = &apiextensionsv1beta1.CustomResourceSubresources{
				Status: &apiextensionsv1beta1.CustomResourceSubresourceStatus{},
				Scale: &apiextensionsv1beta1.CustomResourceSubresourceScale{
					SpecReplicasPath:   ".spec.replicas",
					StatusReplicasPath: ".status.replicas",
				},
			}
			crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, dynamicClient)
			if err != nil {
				t.Fatal(err)
			}

			gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural}

			ns := "test"
			var client dynamic.ResourceInterface = dynamicClient.Resource(gvr)
			var otherScopeClient dynamic.ResourceInterface = dynamicClient.Resource(gvr).Namespace(ns)
			if crd.Spec.Scope == apiextensionsv1beta1.NamespaceScoped {
				client, otherScopeClient = otherScopeClient, client
			}

			name := "bar"
			cr := &unstructured.Unstructured{
				Object: map[string]interface{}{
					"kind":       crd.Spec.Names.Kind,
					"apiVersion": gvr.GroupVersion().String(),
					"metadata": map[string]interface{}{
						"name": name,
					},
				},
			}

			_, err := otherScopeClient.Create(context.TODO(), cr, metav1.CreateOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Create(context.TODO(), cr, metav1.CreateOptions{}, "status")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Create(context.TODO(), cr, metav1.CreateOptions{}, "scale")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = client.Create(context.TODO(), cr, metav1.CreateOptions{})
			assert.NoError(t, err)

			_, err = otherScopeClient.Get(context.TODO(), name, metav1.GetOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Get(context.TODO(), name, metav1.GetOptions{}, "status")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Get(context.TODO(), name, metav1.GetOptions{}, "scale")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Update(context.TODO(), cr, metav1.UpdateOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Update(context.TODO(), cr, metav1.UpdateOptions{}, "status")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Update(context.TODO(), cr, metav1.UpdateOptions{}, "scale")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Patch(context.TODO(), name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Patch(context.TODO(), name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}, "status")
			assert.True(t, apierrors.IsNotFound(err))

			_, err = otherScopeClient.Patch(context.TODO(), name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"test":"1"}}}`), metav1.PatchOptions{}, "scale")
			assert.True(t, apierrors.IsNotFound(err))

			err = otherScopeClient.Delete(context.TODO(), name, metav1.DeleteOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			err = otherScopeClient.Delete(context.TODO(), name, metav1.DeleteOptions{}, "status")
			assert.True(t, apierrors.IsNotFound(err))

			err = otherScopeClient.Delete(context.TODO(), name, metav1.DeleteOptions{}, "scale")
			assert.True(t, apierrors.IsNotFound(err))

			err = otherScopeClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{})
			assert.True(t, apierrors.IsNotFound(err))

			if scope == apiextensionsv1beta1.ClusterScoped {
				_, err = otherScopeClient.List(context.TODO(), metav1.ListOptions{})
				assert.True(t, apierrors.IsNotFound(err))

				_, err = otherScopeClient.Watch(context.TODO(), metav1.ListOptions{})
				assert.True(t, apierrors.IsNotFound(err))
			} else {
				_, err = otherScopeClient.List(context.TODO(), metav1.ListOptions{})
				assert.NoError(t, err)

				w, err := otherScopeClient.Watch(context.TODO(), metav1.ListOptions{})
				assert.NoError(t, err)
				if w != nil {
					w.Stop()
				}
			}
		})
	}
}
