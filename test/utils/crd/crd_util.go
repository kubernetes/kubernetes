/*
Copyright 2018 The Kubernetes Authors.

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

package crd

import (
	"fmt"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	crdclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// CleanCrdFn declares the clean up function needed to remove the CRD
type CleanCrdFn func() error

// TestCrd holds all the pieces needed to test with the CRD
type TestCrd struct {
	APIExtensionClient *crdclientset.Clientset
	Crd                *apiextensionsv1beta1.CustomResourceDefinition
	DynamicClients     map[string]dynamic.ResourceInterface
	CleanUp            CleanCrdFn
}

// Option is a modifier for a CRD object used to customize CreateMultiVersionTestCRD and CreateTestCRD.
type Option func(crd *apiextensionsv1beta1.CustomResourceDefinition)

// CreateMultiVersionTestCRD creates a new CRD specifically for the calling test.
func CreateMultiVersionTestCRD(f *framework.Framework, group string, opts ...Option) (*TestCrd, error) {
	suffix := framework.RandomSuffix()
	name := fmt.Sprintf("e2e-test-%s-%s-crd", f.BaseName, suffix)
	kind := fmt.Sprintf("E2e-test-%s-%s-crd", f.BaseName, suffix)
	testcrd := &TestCrd{}

	// Creating a custom resource definition for use by assorted tests.
	config, err := framework.LoadConfig()
	if err != nil {
		e2elog.Failf("failed to load config: %v", err)
		return nil, err
	}
	apiExtensionClient, err := crdclientset.NewForConfig(config)
	if err != nil {
		e2elog.Failf("failed to initialize apiExtensionClient: %v", err)
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		e2elog.Failf("failed to initialize dynamic client: %v", err)
		return nil, err
	}

	crd := &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: name + "s." + group},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group: group,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   name + "s",
				Singular: name,
				Kind:     kind,
				ListKind: kind + "List",
			},
			Scope: apiextensionsv1beta1.NamespaceScoped,
		},
	}
	for _, opt := range opts {
		opt(crd)
	}
	if len(crd.Spec.Versions) == 0 && len(crd.Spec.Version) == 0 {
		crd.Spec.Version = "v1"
	}

	//create CRD and waits for the resource to be recognized and available.
	crd, err = fixtures.CreateNewCustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
	if err != nil {
		e2elog.Failf("failed to create CustomResourceDefinition: %v", err)
		return nil, err
	}

	resourceClients := map[string]dynamic.ResourceInterface{}
	for _, v := range crd.Spec.Versions {
		if v.Served {
			gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: v.Name, Resource: crd.Spec.Names.Plural}
			resourceClients[v.Name] = dynamicClient.Resource(gvr).Namespace(f.Namespace.Name)
		}
	}

	testcrd.APIExtensionClient = apiExtensionClient
	testcrd.Crd = crd
	testcrd.DynamicClients = resourceClients
	testcrd.CleanUp = func() error {
		err := fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
		if err != nil {
			e2elog.Failf("failed to delete CustomResourceDefinition(%s): %v", name, err)
		}
		return err
	}
	return testcrd, nil
}

// CreateTestCRD creates a new CRD specifically for the calling test.
func CreateTestCRD(f *framework.Framework, opts ...Option) (*TestCrd, error) {
	group := fmt.Sprintf("%s-crd-test.k8s.io", f.BaseName)
	return CreateMultiVersionTestCRD(f, group, append([]Option{func(crd *apiextensionsv1beta1.CustomResourceDefinition) {
		crd.Spec.Versions = []apiextensionsv1beta1.CustomResourceDefinitionVersion{
			{
				Name:    "v1",
				Served:  true,
				Storage: true,
			},
		}
	}}, opts...)...)
}
