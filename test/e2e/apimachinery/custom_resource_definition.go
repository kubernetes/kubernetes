/*
Copyright 2016 The Kubernetes Authors.

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

package apimachinery

import (
	"github.com/onsi/ginkgo"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

var crdVersion = utilversion.MustParseSemantic("v1.7.0")

var _ = SIGDescribe("CustomResourceDefinition resources", func() {

	f := framework.NewDefaultFramework("custom-resource-definition")

	ginkgo.Context("Simple CustomResourceDefinition", func() {
		/*
			Release : v1.9
			Testname: Custom Resource Definition, create
			Description: Create a API extension client, define a random custom resource definition, create the custom resource. API server MUST be able to create the custom resource.
		*/
		framework.ConformanceIt("creating/deleting custom resource definition objects works ", func() {

			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")

			randomDefinition := fixtures.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)

			// Create CRD and waits for the resource to be recognized and available.
			randomDefinition, err = fixtures.CreateNewCustomResourceDefinition(randomDefinition, apiExtensionClient, f.DynamicClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")

			defer func() {
				err = fixtures.DeleteCustomResourceDefinition(randomDefinition, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()
		})

		/*
			Release : v1.16
			Testname: Custom Resource Definition, list
			Description: Create a API extension client, define 10 random custom resource definitions and list them using a label selector. API server MUST be able to list the custom resource definitions and delete them via delete collection.
		*/
		ginkgo.It("listing custom resource definition objects works ", func() {
			testListSize := 10
			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")

			// Label the CRDs we create so we can list only them even though they are cluster scoped
			testUUID := string(uuid.NewUUID())

			// Create CRD and wait for the resource to be recognized and available.
			crds := make([]*v1beta1.CustomResourceDefinition, testListSize)
			for i := 0; i < testListSize; i++ {
				crd := fixtures.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)
				crd.Labels = map[string]string{"e2e-list-test-uuid": testUUID}
				crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
				framework.ExpectNoError(err, "creating CustomResourceDefinition")
				crds[i] = crd
			}

			// Create a crd w/o the label to ensure the label selector matching works correctly
			crd := fixtures.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)
			crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")
			defer func() {
				err = fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()

			selectorListOpts := metav1.ListOptions{LabelSelector: "e2e-list-test-uuid=" + testUUID}
			list, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().List(selectorListOpts)
			framework.ExpectNoError(err, "listing CustomResourceDefinitions")
			framework.ExpectEqual(len(list.Items), testListSize)
			for _, actual := range list.Items {
				var expected *v1beta1.CustomResourceDefinition
				for _, e := range crds {
					if e.Name == actual.Name && e.Namespace == actual.Namespace {
						expected = e
					}
				}
				framework.ExpectNotEqual(expected, nil)
				if !equality.Semantic.DeepEqual(actual.Spec, expected.Spec) {
					e2elog.Failf("Expected CustomResourceDefinition in list with name %s to match crd created with same name, but got different specs:\n%s",
						actual.Name, diff.ObjectReflectDiff(expected.Spec, actual.Spec))
				}
			}

			// Use delete collection to remove the CRDs
			err = fixtures.DeleteCustomResourceDefinitions(selectorListOpts, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinitions")
			_, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Get(crd.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "getting remaining CustomResourceDefinition")
		})

		/*
			Release : v1.16
			Testname: Custom Resource Definition, status sub-resource
			Description: Create a API extension client, create a custom resource definition and then read, update and patch its status sub-resource. API server MUST be able to perform the operations against the status sub-resource.
		*/
		ginkgo.It("getting/updating/patching custom resource definition status sub-resource works ", func() {
			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")
			dynamicClient, err := dynamic.NewForConfig(config)
			framework.ExpectNoError(err, "initializing dynamic client")
			gvr := v1beta1.SchemeGroupVersion.WithResource("customresourcedefinitions")
			resourceClient := dynamicClient.Resource(gvr)

			// Create CRD and waits for the resource to be recognized and available.
			crd := fixtures.NewRandomNameCustomResourceDefinition(v1beta1.ClusterScoped)
			crd, err = fixtures.CreateNewCustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")
			defer func() {
				err = fixtures.DeleteCustomResourceDefinition(crd, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()

			var updated *v1beta1.CustomResourceDefinition
			updateCondition := v1beta1.CustomResourceDefinitionCondition{Message: "updated"}
			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				// Use dynamic client to read the status sub-resource since typed client does not expose it.
				u, err := resourceClient.Get(crd.GetName(), metav1.GetOptions{}, "status")
				framework.ExpectNoError(err, "getting CustomResourceDefinition status")
				status := unstructuredToCRD(u)
				if !equality.Semantic.DeepEqual(status.Spec, crd.Spec) {
					e2elog.Failf("Expected CustomResourceDefinition Spec to match status sub-resource Spec, but got:\n%s", diff.ObjectReflectDiff(status.Spec, crd.Spec))
				}
				status.Status.Conditions = append(status.Status.Conditions, updateCondition)
				updated, err = apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().UpdateStatus(status)
				return err
			})
			framework.ExpectNoError(err, "updating CustomResourceDefinition status")
			expectCondition(updated.Status.Conditions, updateCondition)

			patchCondition := v1beta1.CustomResourceDefinitionCondition{Message: "patched"}
			patched, err := apiExtensionClient.ApiextensionsV1beta1().CustomResourceDefinitions().Patch(
				crd.GetName(),
				types.JSONPatchType,
				[]byte(`[{"op": "add", "path": "/status/conditions", "value": [{"message": "patched"}]}]`),
				"status")
			framework.ExpectNoError(err, "patching CustomResourceDefinition status")
			expectCondition(updated.Status.Conditions, updateCondition)
			expectCondition(patched.Status.Conditions, patchCondition)
		})
	})
})

func unstructuredToCRD(obj *unstructured.Unstructured) *v1beta1.CustomResourceDefinition {
	crd := new(v1beta1.CustomResourceDefinition)
	err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, crd)
	framework.ExpectNoError(err, "converting unstructured to CustomResourceDefinition")
	return crd
}

func expectCondition(conditions []v1beta1.CustomResourceDefinitionCondition, expected v1beta1.CustomResourceDefinitionCondition) {
	for _, c := range conditions {
		if equality.Semantic.DeepEqual(c, expected) {
			return
		}
	}
	e2elog.Failf("Condition %#v not found in conditions %#v", expected, conditions)
}
