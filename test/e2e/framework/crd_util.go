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

package framework

import (
	"fmt"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	crdclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
)

// CleanCrdFn declares the clean up function needed to remove the CRD
type CleanCrdFn func() error

// TestCrd holds all the pieces needed to test with the CRD
type TestCrd struct {
	Name               string
	Kind               string
	ApiGroup           string
	ApiVersion         string
	ApiExtensionClient *crdclientset.Clientset
	Crd                *apiextensionsv1beta1.CustomResourceDefinition
	DynamicClient      dynamic.ResourceInterface
	CleanUp            CleanCrdFn
}

// CreateTestCRD creates a new CRD specifically for the calling test.
func CreateTestCRD(f *Framework) (*TestCrd, error) {
	suffix := randomSuffix()
	name := fmt.Sprintf("e2e-test-%s-%s-crd", f.BaseName, suffix)
	kind := fmt.Sprintf("E2e-test-%s-%s-crd", f.BaseName, suffix)
	group := fmt.Sprintf("%s-crd-test.k8s.io", f.BaseName)
	apiVersion := "v1"
	testcrd := &TestCrd{
		Name:       name,
		Kind:       kind,
		ApiGroup:   group,
		ApiVersion: apiVersion,
	}

	// Creating a custom resource definition for use by assorted tests.
	config, err := LoadConfig()
	if err != nil {
		Failf("failed to load config: %v", err)
		return nil, err
	}
	apiExtensionClient, err := crdclientset.NewForConfig(config)
	if err != nil {
		Failf("failed to initialize apiExtensionClient: %v", err)
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		Failf("failed to initialize dynamic client: %v", err)
		return nil, err
	}

	crd := newCRDForTest(testcrd)

	//create CRD and waits for the resource to be recognized and available.
	crd, err = fixtures.CreateNewCustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
	if err != nil {
		Failf("failed to create CustomResourceDefinition: %v", err)
		return nil, err
	}

	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural}
	resourceClient := dynamicClient.Resource(gvr).Namespace(f.Namespace.Name)

	testcrd.ApiExtensionClient = apiExtensionClient
	testcrd.Crd = crd
	testcrd.DynamicClient = resourceClient
	testcrd.CleanUp = func() error {
		err := fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
		if err != nil {
			Failf("failed to delete CustomResourceDefinition(%s): %v", name, err)
		}
		return err
	}
	return testcrd, nil
}

// newCRDForTest generates a CRD definition for the test
func newCRDForTest(testcrd *TestCrd) *apiextensionsv1beta1.CustomResourceDefinition {
	return &apiextensionsv1beta1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{Name: testcrd.GetMetaName()},
		Spec: apiextensionsv1beta1.CustomResourceDefinitionSpec{
			Group:   testcrd.ApiGroup,
			Version: testcrd.ApiVersion,
			Names: apiextensionsv1beta1.CustomResourceDefinitionNames{
				Plural:   testcrd.GetPluralName(),
				Singular: testcrd.Name,
				Kind:     testcrd.Kind,
				ListKind: testcrd.GetListName(),
			},
			Scope: apiextensionsv1beta1.NamespaceScoped,
		},
	}
}

// GetMetaName returns the metaname for the CRD.
func (c *TestCrd) GetMetaName() string {
	return c.Name + "s." + c.ApiGroup
}

// GetPluralName returns the plural form of the CRD name
func (c *TestCrd) GetPluralName() string {
	return c.Name + "s"
}

// GetListName returns the name for the CRD list resources
func (c *TestCrd) GetListName() string {
	return c.Name + "List"
}
