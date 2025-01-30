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
	"fmt"
	"strings"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/dynamic"
)

func TestLimits(t *testing.T) {
	tearDown, config, _, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
	noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	kind := noxuDefinition.Spec.Names.Kind
	apiVersion := noxuDefinition.Spec.Group + "/" + noxuDefinition.Spec.Versions[0].Name

	rest := apiExtensionClient.Discovery().RESTClient()

	// Create YAML over 3MB limit
	t.Run("create YAML over limit", func(t *testing.T) {
		yamlBody := []byte(fmt.Sprintf(`
apiVersion: %s
kind: %s
metadata:
  name: test
values: `+strings.Repeat("[", 3*1024*1024), apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected too large error, got %v", err)
		}
	})

	// Create YAML just under 3MB limit, nested
	t.Run("create YAML doc under limit, nested", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		yamlBody := []byte(fmt.Sprintf(`
	apiVersion: %s
	kind: %s
	metadata:
	  name: test
	values: `+strings.Repeat("[", 3*1024*1024/2-500)+strings.Repeat("]", 3*1024*1024/2-500), apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create YAML just under 3MB limit, not nested
	t.Run("create YAML doc under limit, not nested", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		yamlBody := []byte(fmt.Sprintf(`
		apiVersion: %s
		kind: %s
		metadata:
		  name: test
		values: `+strings.Repeat("[", 3*1024*1024-1000), apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create JSON over 3MB limit
	t.Run("create JSON over limit", func(t *testing.T) {
		jsonBody := []byte(fmt.Sprintf(`{
	"apiVersion": %q,
	"kind": %q,
	"metadata": {
	  "name": "test"
	},
	"values": `+strings.Repeat("[", 3*1024*1024/2)+strings.Repeat("]", 3*1024*1024/2)+"}", apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(jsonBody).
			DoRaw(context.TODO())
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected too large error, got %v", err)
		}
	})

	// Create JSON just under 3MB limit, nested
	t.Run("create JSON doc under limit, nested", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		jsonBody := []byte(fmt.Sprintf(`{
		"apiVersion": %q,
		"kind": %q,
		"metadata": {
		  "name": "test"
		},
		"values": `+strings.Repeat("[", 3*1024*1024/2-500)+strings.Repeat("]", 3*1024*1024/2-500)+"}", apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(jsonBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create JSON just under 3MB limit, not nested
	t.Run("create JSON doc under limit, not nested", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		jsonBody := []byte(fmt.Sprintf(`{
			"apiVersion": %q,
			"kind": %q,
			"metadata": {
			  "name": "test"
			},
			"values": `+strings.Repeat("[", 3*1024*1024-1000)+"}", apiVersion, kind))

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).
			Body(jsonBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create instance to allow patching
	{
		jsonBody := []byte(fmt.Sprintf(`{"apiVersion": %q, "kind": %q, "metadata": {"name": "test"}}`, apiVersion, kind))
		_, err := rest.Post().AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural).Body(jsonBody).DoRaw(context.TODO())
		if err != nil {
			t.Fatalf("error creating object: %v", err)
		}
	}

	t.Run("JSONPatchType nested patch under limit", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		patchBody := []byte(`[{"op":"add","path":"/foo","value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}]`)
		err = rest.Patch(types.JSONPatchType).AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural, "test").
			Body(patchBody).Do(context.TODO()).Error()
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %v", err)
		}
	})
	t.Run("MergePatchType nested patch under limit", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		patchBody := []byte(`{"value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}`)
		err = rest.Patch(types.MergePatchType).AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural, "test").
			Body(patchBody).Do(context.TODO()).Error()
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %v", err)
		}
	})
	t.Run("ApplyPatchType nested patch under limit", func(t *testing.T) {
		if testing.Short() {
			t.Skip("skipping expensive test")
		}
		patchBody := []byte(`{"value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}`)
		err = rest.Patch(types.ApplyPatchType).Param("fieldManager", "test").AbsPath("/apis", noxuDefinition.Spec.Group, noxuDefinition.Spec.Versions[0].Name, noxuDefinition.Spec.Names.Plural, "test").
			Body(patchBody).Do(context.TODO()).Error()
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request err, got %#v", err)
		}
	})
}
