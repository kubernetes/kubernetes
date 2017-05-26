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

package testserver

import (
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	apiextensionsv1beta1 "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1beta1"
	"k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset"
)

func NewNoxuCustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:     "noxus",
				Singular:   "nonenglishnoxu",
				Kind:       "WishIHadChosenNoxu",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "NoxuItemList",
			},
			Scope: scope,
		},
	}
}

func NewNoxuInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "WishIHadChosenNoxu",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"content": map[string]interface{}{
				"key": "value",
			},
			"num": map[string]interface{}{
				"num1": 9223372036854775807,
				"num2": 1000000,
			},
		},
	}
}

func NewNoxu2CustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus2.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1alpha1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:     "noxus2",
				Singular:   "nonenglishnoxu2",
				Kind:       "WishIHadChosenNoxu2",
				ShortNames: []string{"foo", "bar", "abc", "def"},
				ListKind:   "Noxu2ItemList",
			},
			Scope: scope,
		},
	}
}

func NewCurletCustomResourceDefinition(scope apiextensionsv1beta1.ResourceScope) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "curlets.mygroup.example.com"},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1beta1",
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   "curlets",
				Singular: "curlet",
				Kind:     "Curlet",
				ListKind: "CurletList",
			},
			Scope: scope,
		},
	}
}

func NewCurletInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1beta1",
			"kind":       "Curlet",
			"metadata": map[string]interface{}{
				"namespace": namespace,
				"name":      name,
			},
			"content": map[string]interface{}{
				"key": "value",
			},
		},
	}
}

func CreateNewCustomResourceDefinition(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, clientPool dynamic.ClientPool) (*dynamic.Client, error) {
	_, err := apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Create(crd)
	if err != nil {
		return nil, err
	}

	// wait until the resource appears in discovery
	err = wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
		resourceList, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + crd.Spec.Version)
		if err != nil {
			return false, nil
		}
		for _, resource := range resourceList.APIResources {
			if resource.Name == crd.Spec.Names.Plural {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	dynamicClient, err := clientPool.ClientForGroupVersionResource(schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural})
	if err != nil {
		return nil, err
	}
	return dynamicClient, nil
}

func DeleteCustomResourceDefinition(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) error {
	if err := apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Delete(crd.Name, nil); err != nil {
		return err
	}
	err := wait.PollImmediate(500*time.Millisecond, 30*time.Second, func() (bool, error) {
		groupResource, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(crd.Spec.Group + "/" + crd.Spec.Version)
		if err != nil {
			if errors.IsNotFound(err) {
				return true, nil

			}
			return false, err
		}
		for _, g := range groupResource.APIResources {
			if g.Name == crd.Spec.Names.Plural {
				return false, nil
			}
		}
		return true, nil
	})
	return err
}

func GetCustomResourceDefinition(crd *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1beta1.CustomResourceDefinition, error) {
	return apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Get(crd.Name, metav1.GetOptions{})
}
