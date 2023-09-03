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

package openapi

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	"k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"
	"k8s.io/kube-openapi/pkg/handler"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"

	"k8s.io/kube-openapi/pkg/validation/spec"
)

func TestBasicAddRemove(t *testing.T) {
	env, ctx := setup(t)
	env.runFunc()
	defer env.cleanFunc()

	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolFooCRD, metav1.CreateOptions{})
	env.pollForPathExists("/apis/stable.example.com/v1/coolfoos")
	s := env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolfoos")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")

	t.Logf("Removing CRD %s", coolFooCRD.Name)
	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Delete(ctx, coolFooCRD.Name, metav1.DeleteOptions{})
	env.pollForPathNotExists("/apis/stable.example.com/v1/coolfoos")
	s = env.fetchOpenAPIOrDie()
	env.expectNoPath(s, "/apis/stable.example.com/v1/coolfoos")
}

func TestTwoCRDsSameGroup(t *testing.T) {
	env, ctx := setup(t)
	env.runFunc()
	defer env.cleanFunc()

	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolFooCRD, metav1.CreateOptions{})
	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolBarCRD, metav1.CreateOptions{})
	env.pollForPathExists("/apis/stable.example.com/v1/coolfoos")
	env.pollForPathExists("/apis/stable.example.com/v1/coolbars")
	s := env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolfoos")
	env.expectPath(s, "/apis/stable.example.com/v1/coolbars")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")
}

func TestCRDMultiVersion(t *testing.T) {
	env, ctx := setup(t)
	env.runFunc()
	defer env.cleanFunc()

	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolMultiVersion, metav1.CreateOptions{})
	env.pollForPathExists("/apis/stable.example.com/v1/coolbars")
	env.pollForPathExists("/apis/stable.example.com/v1beta1/coolbars")
	s := env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolbars")
	env.expectPath(s, "/apis/stable.example.com/v1beta1/coolbars")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")
}

func TestCRDMultiVersionUpdate(t *testing.T) {
	env, ctx := setup(t)
	env.runFunc()
	defer env.cleanFunc()

	crd, _ := env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolMultiVersion, metav1.CreateOptions{})
	env.pollForPathExists("/apis/stable.example.com/v1/coolbars")
	env.pollForPathExists("/apis/stable.example.com/v1beta1/coolbars")
	s := env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolbars")
	env.expectPath(s, "/apis/stable.example.com/v1beta1/coolbars")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")

	t.Log("Removing version v1beta1")
	crd.Spec.Versions = crd.Spec.Versions[1:]
	crd.Generation += 1
	// Generation is updated before storage to etcd. Since we don't have that in the fake client, manually increase it.
	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Update(ctx, crd, metav1.UpdateOptions{})
	env.pollForPathNotExists("/apis/stable.example.com/v1beta1/coolbars")
	s = env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolbars")
	env.expectNoPath(s, "/apis/stable.example.com/v1beta1/coolbars")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")
}

func TestExistingCRDBeforeAPIServerStart(t *testing.T) {
	env, ctx := setup(t)
	defer env.cleanFunc()

	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolFooCRD, metav1.CreateOptions{})
	env.runFunc()
	env.pollForPathExists("/apis/stable.example.com/v1/coolfoos")
	s := env.fetchOpenAPIOrDie()

	env.expectPath(s, "/apis/stable.example.com/v1/coolfoos")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")
}

func TestUpdate(t *testing.T) {
	env, ctx := setup(t)
	env.runFunc()
	defer env.cleanFunc()

	crd, _ := env.Interface.ApiextensionsV1().CustomResourceDefinitions().Create(ctx, coolFooCRD, metav1.CreateOptions{})
	env.pollForPathExists("/apis/stable.example.com/v1/coolfoos")
	s := env.fetchOpenAPIOrDie()
	env.expectPath(s, "/apis/stable.example.com/v1/coolfoos")
	env.expectPath(s, "/apis/apiextensions.k8s.io/v1")

	t.Log("Updating CRD CoolFoo")
	crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["num"] = v1.JSONSchemaProps{Type: "integer", Description: "updated description"}
	crd.Generation += 1
	// Generation is updated before storage to etcd. Since we don't have that in the fake client, manually increase it.

	env.Interface.ApiextensionsV1().CustomResourceDefinitions().Update(ctx, crd, metav1.UpdateOptions{})
	env.pollForCondition(func(s *spec.Swagger) bool {
		return s.Definitions["com.example.stable.v1.CoolFoo"].Properties["num"].Description == crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["num"].Description
	})
	s = env.fetchOpenAPIOrDie()

	// Ensure that description is updated
	if s.Definitions["com.example.stable.v1.CoolFoo"].Properties["num"].Description != crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["num"].Description {
		t.Error("Error: Description not updated")
	}
	env.expectPath(s, "/apis/stable.example.com/v1/coolfoos")
}

var coolFooCRD = &v1.CustomResourceDefinition{
	TypeMeta: metav1.TypeMeta{
		APIVersion: "apiextensions.k8s.io/v1",
		Kind:       "CustomResourceDefinition",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "coolfoo.stable.example.com",
	},
	Spec: v1.CustomResourceDefinitionSpec{
		Group: "stable.example.com",
		Names: v1.CustomResourceDefinitionNames{
			Plural:     "coolfoos",
			Singular:   "coolfoo",
			ShortNames: []string{"foo"},
			Kind:       "CoolFoo",
			ListKind:   "CoolFooList",
		},
		Scope: v1.ClusterScoped,
		Versions: []v1.CustomResourceDefinitionVersion{
			{
				Name:       "v1",
				Served:     true,
				Storage:    true,
				Deprecated: false,
				Subresources: &v1.CustomResourceSubresources{
					// This CRD has a /status subresource
					Status: &v1.CustomResourceSubresourceStatus{},
				},
				Schema: &v1.CustomResourceValidation{
					OpenAPIV3Schema: &v1.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]v1.JSONSchemaProps{"num": {Type: "integer", Description: "description"}},
					},
				},
			},
		},
		Conversion: &v1.CustomResourceConversion{},
	},
	Status: v1.CustomResourceDefinitionStatus{
		Conditions: []v1.CustomResourceDefinitionCondition{
			{
				Type:   v1.Established,
				Status: v1.ConditionTrue,
			},
		},
	},
}

var coolBarCRD = &v1.CustomResourceDefinition{
	TypeMeta: metav1.TypeMeta{
		APIVersion: "apiextensions.k8s.io/v1",
		Kind:       "CustomResourceDefinition",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "coolbar.stable.example.com",
	},
	Spec: v1.CustomResourceDefinitionSpec{
		Group: "stable.example.com",
		Names: v1.CustomResourceDefinitionNames{
			Plural:     "coolbars",
			Singular:   "coolbar",
			ShortNames: []string{"bar"},
			Kind:       "CoolBar",
			ListKind:   "CoolBarList",
		},
		Scope: v1.ClusterScoped,
		Versions: []v1.CustomResourceDefinitionVersion{
			{
				Name:       "v1",
				Served:     true,
				Storage:    true,
				Deprecated: false,
				Subresources: &v1.CustomResourceSubresources{
					// This CRD has a /status subresource
					Status: &v1.CustomResourceSubresourceStatus{},
				},
				Schema: &v1.CustomResourceValidation{
					OpenAPIV3Schema: &v1.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]v1.JSONSchemaProps{"num": {Type: "integer", Description: "description"}},
					},
				},
			},
		},
		Conversion: &v1.CustomResourceConversion{},
	},
	Status: v1.CustomResourceDefinitionStatus{
		Conditions: []v1.CustomResourceDefinitionCondition{
			{
				Type:   v1.Established,
				Status: v1.ConditionTrue,
			},
		},
	},
}

var coolMultiVersion = &v1.CustomResourceDefinition{
	TypeMeta: metav1.TypeMeta{
		APIVersion: "apiextensions.k8s.io/v1",
		Kind:       "CustomResourceDefinition",
	},
	ObjectMeta: metav1.ObjectMeta{
		Name: "coolbar.stable.example.com",
	},
	Spec: v1.CustomResourceDefinitionSpec{
		Group: "stable.example.com",
		Names: v1.CustomResourceDefinitionNames{
			Plural:     "coolbars",
			Singular:   "coolbar",
			ShortNames: []string{"bar"},
			Kind:       "CoolBar",
			ListKind:   "CoolBarList",
		},
		Scope: v1.ClusterScoped,
		Versions: []v1.CustomResourceDefinitionVersion{
			{
				Name:       "v1beta1",
				Served:     true,
				Storage:    true,
				Deprecated: false,
				Subresources: &v1.CustomResourceSubresources{
					// This CRD has a /status subresource
					Status: &v1.CustomResourceSubresourceStatus{},
				},
				Schema: &v1.CustomResourceValidation{
					OpenAPIV3Schema: &v1.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]v1.JSONSchemaProps{"num": {Type: "integer", Description: "description"}},
					},
				},
			},

			{
				Name:       "v1",
				Served:     true,
				Storage:    true,
				Deprecated: false,
				Subresources: &v1.CustomResourceSubresources{
					// This CRD has a /status subresource
					Status: &v1.CustomResourceSubresourceStatus{},
				},
				Schema: &v1.CustomResourceValidation{
					OpenAPIV3Schema: &v1.JSONSchemaProps{
						Type:       "object",
						Properties: map[string]v1.JSONSchemaProps{"test": {Type: "integer", Description: "foo"}},
					},
				},
			},
		},
		Conversion: &v1.CustomResourceConversion{},
	},
	Status: v1.CustomResourceDefinitionStatus{
		Conditions: []v1.CustomResourceDefinitionCondition{
			{
				Type:   v1.Established,
				Status: v1.ConditionTrue,
			},
		},
	},
}

type testEnv struct {
	t *testing.T
	clientset.Interface
	mux       *http.ServeMux
	cleanFunc func()
	runFunc   func()
}

func setup(t *testing.T) (*testEnv, context.Context) {
	env := &testEnv{
		Interface: fake.NewSimpleClientset(),
		t:         t,
	}

	factory := externalversions.NewSharedInformerFactoryWithOptions(
		env.Interface, 30*time.Second)

	c := NewController(factory.Apiextensions().V1().CustomResourceDefinitions())
	ctx, cancel := context.WithCancel(context.Background())

	factory.Start(ctx.Done())
	factory.WaitForCacheSync(ctx.Done())

	env.mux = http.NewServeMux()
	h := handler.NewOpenAPIService(&spec.Swagger{})
	h.RegisterOpenAPIVersionedService("/openapi/v2", env.mux)

	stopCh := make(chan struct{})

	env.runFunc = func() {
		go c.Run(&spec.Swagger{
			SwaggerProps: spec.SwaggerProps{
				Paths: &spec.Paths{
					Paths: map[string]spec.PathItem{
						"/apis/apiextensions.k8s.io/v1": {},
					},
				},
			},
		}, h, stopCh)
	}

	env.cleanFunc = func() {
		cancel()
		close(stopCh)
	}
	return env, ctx
}

func (t *testEnv) pollForCondition(conditionFunc func(*spec.Swagger) bool) {
	wait.Poll(time.Second*1, wait.ForeverTestTimeout, func() (bool, error) {
		openapi := t.fetchOpenAPIOrDie()
		if conditionFunc(openapi) {
			return true, nil
		}
		return false, nil
	})
}

func (t *testEnv) pollForPathExists(path string) {
	wait.Poll(time.Second*1, wait.ForeverTestTimeout, func() (bool, error) {
		openapi := t.fetchOpenAPIOrDie()
		if _, ok := openapi.Paths.Paths[path]; !ok {
			return false, nil
		}
		return true, nil
	})
}

func (t *testEnv) pollForPathNotExists(path string) {
	wait.Poll(time.Second*1, wait.ForeverTestTimeout, func() (bool, error) {
		openapi := t.fetchOpenAPIOrDie()
		if _, ok := openapi.Paths.Paths[path]; ok {
			return false, nil
		}
		return true, nil
	})
}

func (t *testEnv) fetchOpenAPIOrDie() *spec.Swagger {
	server := httptest.NewServer(t.mux)
	defer server.Close()
	client := server.Client()

	req, err := http.NewRequest("GET", server.URL+"/openapi/v2", nil)
	if err != nil {
		t.t.Error(err)
	}
	resp, err := client.Do(req)
	if err != nil {
		t.t.Error(err)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.t.Error(err)
	}
	swagger := &spec.Swagger{}
	if err := swagger.UnmarshalJSON(body); err != nil {
		t.t.Error(err)
	}
	return swagger
}

func (t *testEnv) expectPath(swagger *spec.Swagger, path string) {
	if _, ok := swagger.Paths.Paths[path]; !ok {
		t.t.Errorf("Expected path %s to exist in OpenAPI", path)
	}
}

func (t *testEnv) expectNoPath(swagger *spec.Swagger, path string) {
	if _, ok := swagger.Paths.Paths[path]; ok {
		t.t.Errorf("Expected path %s to not exist in OpenAPI", path)
	}
}
