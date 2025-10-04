/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestGracePeriodOnCustomResource(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	require.NoError(t, err)
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	require.NoError(t, err)

	gvrList := fixtures.GetGroupVersionResourcesOfCustomResource(noxuDefinition)
	gvr := gvrList[0]
	noxuClient := dynamicClient.Resource(gvr).Namespace("default")

	name := "foo123"
	instance := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": gvr.GroupVersion().String(),
			"kind":       "WishIHadChosenNoxu",
			"metadata": map[string]interface{}{
				"name":       name,
				"finalizers": []interface{}{"noxu.example.com/finalizer"},
			},
		},
	}

	ctx := context.Background()
	_, err = noxuClient.Create(ctx, instance, metav1.CreateOptions{})
	require.NoError(t, err)

	gracePeriod := int64(30)
	err = noxuClient.Delete(ctx, name, metav1.DeleteOptions{
		GracePeriodSeconds: &gracePeriod,
	})
	require.NoError(t, err)

	time.Sleep(300 * time.Millisecond)

	got, err := noxuClient.Get(ctx, name, metav1.GetOptions{})
	require.NoError(t, err)
	require.NotNil(t, got.GetDeletionGracePeriodSeconds())
	require.Equal(t, gracePeriod, *got.GetDeletionGracePeriodSeconds())
}
