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
	apiextensionsv1alpha1 "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1"
	"k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset"
)

func NewNoxuCustomResourceDefinition(scope apiextensionsv1alpha1.ResourceScope) *apiextensionsv1alpha1.CustomResourceDefinition {
	return &apiextensionsv1alpha1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1alpha1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1alpha1",
			Names: apiextensionsv1alpha1.CustomResourceDefinitionNames{
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
			"apiVersion": "mygroup.example.com/v1alpha1",
			"kind":       "WishIHadChosenNoxu",
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

func NewCurletCustomResourceDefinition() *apiextensionsv1alpha1.CustomResourceDefinition {
	return &apiextensionsv1alpha1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: "curlets.mygroup.example.com"},
		Spec: apiextensionsv1alpha1.CustomResourceDefinitionSpec{
			Group:   "mygroup.example.com",
			Version: "v1alpha1",
			Names: apiextensionsv1alpha1.CustomResourceDefinitionNames{
				Plural:   "curlets",
				Singular: "curlet",
				Kind:     "Curlet",
				ListKind: "CurletList",
			},
			Scope: apiextensionsv1alpha1.NamespaceScoped,
		},
	}
}

func NewCurletInstance(namespace, name string) *unstructured.Unstructured {
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "mygroup.example.com/v1alpha1",
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

func CreateNewCustomResourceDefinition(customResourceDefinition *apiextensionsv1alpha1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, clientPool dynamic.ClientPool) (*dynamic.Client, error) {
	_, err := apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Create(customResourceDefinition)
	if err != nil {
		return nil, err
	}

	// wait until the resource appears in discovery
	err = wait.PollImmediate(30*time.Millisecond, 30*time.Second, func() (bool, error) {
		resourceList, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(customResourceDefinition.Spec.Group + "/" + customResourceDefinition.Spec.Version)
		if err != nil {
			return false, nil
		}
		for _, resource := range resourceList.APIResources {
			if resource.Name == customResourceDefinition.Spec.Names.Plural {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	dynamicClient, err := clientPool.ClientForGroupVersionResource(schema.GroupVersionResource{Group: customResourceDefinition.Spec.Group, Version: customResourceDefinition.Spec.Version, Resource: customResourceDefinition.Spec.Names.Plural})
	if err != nil {
		return nil, err
	}
	return dynamicClient, nil
}

func DeleteCustomResourceDefinition(customResource *apiextensionsv1alpha1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) error {
	if err := apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Delete(customResource.Name, nil); err != nil {
		return err
	}
	err := wait.PollImmediate(30*time.Millisecond, 30*time.Second, func() (bool, error) {
		if _, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(customResource.Spec.Group + "/" + customResource.Spec.Version); err != nil {
			if errors.IsNotFound(err) {
				return true, nil
			}
			return false, err
		}
		return false, nil
	})
	return err
}

func GetCustomResourceDefinition(customResource *apiextensionsv1alpha1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1alpha1.CustomResourceDefinition, error) {
	return apiExtensionsClient.Apiextensions().CustomResourceDefinitions().Get(customResource.Name, metav1.GetOptions{})
}
