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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	apiextensionsv1alpha1 "k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1"
	"k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset"
)

func NewNoxuCustomResourceDefinition() *apiextensionsv1alpha1.CustomResource {
	return &apiextensionsv1alpha1.CustomResource{
		ObjectMeta: metav1.ObjectMeta{Name: "noxus.mygroup.example.com"},
		Spec: apiextensionsv1alpha1.CustomResourceSpec{
			Group:   "mygroup.example.com",
			Version: "v1alpha1",
			Names: apiextensionsv1alpha1.CustomResourceNames{
				Plural:   "noxus",
				Singular: "nonenglishnoxu",
				Kind:     "WishIHadChosenNoxu",
				ListKind: "NoxuItemList",
			},
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

func CreateNewCustomResourceDefinition(customResource *apiextensionsv1alpha1.CustomResource, apiExtensionsClient clientset.Interface, clientPool dynamic.ClientPool) (*dynamic.Client, error) {
	_, err := apiExtensionsClient.Apiextensions().CustomResources().Create(customResource)
	if err != nil {
		return nil, err
	}

	// wait until the resource appears in discovery
	err = wait.PollImmediate(30*time.Millisecond, 30*time.Second, func() (bool, error) {
		resourceList, err := apiExtensionsClient.Discovery().ServerResourcesForGroupVersion(customResource.Spec.Group + "/" + customResource.Spec.Version)
		if err != nil {
			return false, nil
		}
		for _, resource := range resourceList.APIResources {
			if resource.Name == customResource.Spec.Names.Plural {
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		return nil, err
	}

	dynamicClient, err := clientPool.ClientForGroupVersionResource(schema.GroupVersionResource{Group: customResource.Spec.Group, Version: customResource.Spec.Version, Resource: customResource.Spec.Names.Plural})
	if err != nil {
		return nil, err
	}
	return dynamicClient, nil
}
