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
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("CustomResourceDefinition resources [Privileged:ClusterAdmin]", func() {

	f := framework.NewDefaultFramework("custom-resource-definition")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Simple CustomResourceDefinition", func() {
		/*
			Release: v1.9
			Testname: Custom Resource Definition, create
			Description: Create a API extension client and define a random custom resource definition.
			Create the custom resource definition and then delete it. The creation and deletion MUST
			be successful.
		*/
		framework.ConformanceIt("creating/deleting custom resource definition objects works ", func() {

			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")

			randomDefinition := fixtures.NewRandomNameV1CustomResourceDefinition(v1.ClusterScoped)

			// Create CRD and waits for the resource to be recognized and available.
			randomDefinition, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(randomDefinition, apiExtensionClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")

			defer func() {
				err = fixtures.DeleteV1CustomResourceDefinition(randomDefinition, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()
		})

		/*
			Release: v1.16
			Testname: Custom Resource Definition, list
			Description: Create a API extension client, define 10 labeled custom resource definitions and list them using
			a label selector; the list result MUST contain only the labeled custom resource definitions. Delete the labeled
			custom resource definitions via delete collection; the delete MUST be successful and MUST delete only the
			labeled custom resource definitions.
		*/
		framework.ConformanceIt("listing custom resource definition objects works ", func() {
			testListSize := 10
			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")

			// Label the CRDs we create so we can list only them even though they are cluster scoped
			testUUID := string(uuid.NewUUID())

			// Create CRD and wait for the resource to be recognized and available.
			crds := make([]*v1.CustomResourceDefinition, testListSize)
			for i := 0; i < testListSize; i++ {
				crd := fixtures.NewRandomNameV1CustomResourceDefinition(v1.ClusterScoped)
				crd.Labels = map[string]string{"e2e-list-test-uuid": testUUID}
				crd, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
				framework.ExpectNoError(err, "creating CustomResourceDefinition")
				crds[i] = crd
			}

			// Create a crd w/o the label to ensure the label selector matching works correctly
			crd := fixtures.NewRandomNameV1CustomResourceDefinition(v1.ClusterScoped)
			crd, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")
			defer func() {
				err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()

			selectorListOpts := metav1.ListOptions{LabelSelector: "e2e-list-test-uuid=" + testUUID}
			list, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().List(context.TODO(), selectorListOpts)
			framework.ExpectNoError(err, "listing CustomResourceDefinitions")
			framework.ExpectEqual(len(list.Items), testListSize)
			for _, actual := range list.Items {
				var expected *v1.CustomResourceDefinition
				for _, e := range crds {
					if e.Name == actual.Name && e.Namespace == actual.Namespace {
						expected = e
					}
				}
				framework.ExpectNotEqual(expected, nil)
				if !equality.Semantic.DeepEqual(actual.Spec, expected.Spec) {
					framework.Failf("Expected CustomResourceDefinition in list with name %s to match crd created with same name, but got different specs:\n%s",
						actual.Name, diff.ObjectReflectDiff(expected.Spec, actual.Spec))
				}
			}

			// Use delete collection to remove the CRDs
			err = fixtures.DeleteV1CustomResourceDefinitions(selectorListOpts, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinitions")
			_, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), crd.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "getting remaining CustomResourceDefinition")
		})

		/*
			Release: v1.16
			Testname: Custom Resource Definition, status sub-resource
			Description: Create a custom resource definition. Attempt to read, update and patch its status sub-resource;
			all mutating sub-resource operations MUST be visible to subsequent reads.
		*/
		framework.ConformanceIt("getting/updating/patching custom resource definition status sub-resource works ", func() {
			config, err := framework.LoadConfig()
			framework.ExpectNoError(err, "loading config")
			apiExtensionClient, err := clientset.NewForConfig(config)
			framework.ExpectNoError(err, "initializing apiExtensionClient")
			dynamicClient, err := dynamic.NewForConfig(config)
			framework.ExpectNoError(err, "initializing dynamic client")
			gvr := v1.SchemeGroupVersion.WithResource("customresourcedefinitions")
			resourceClient := dynamicClient.Resource(gvr)

			// Create CRD and waits for the resource to be recognized and available.
			crd := fixtures.NewRandomNameV1CustomResourceDefinition(v1.ClusterScoped)
			crd, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")
			defer func() {
				err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()

			var updated *v1.CustomResourceDefinition
			updateCondition := v1.CustomResourceDefinitionCondition{Message: "updated"}
			err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
				// Use dynamic client to read the status sub-resource since typed client does not expose it.
				u, err := resourceClient.Get(context.TODO(), crd.GetName(), metav1.GetOptions{}, "status")
				framework.ExpectNoError(err, "getting CustomResourceDefinition status")
				status := unstructuredToCRD(u)
				if !equality.Semantic.DeepEqual(status.Spec, crd.Spec) {
					framework.Failf("Expected CustomResourceDefinition Spec to match status sub-resource Spec, but got:\n%s", diff.ObjectReflectDiff(status.Spec, crd.Spec))
				}
				status.Status.Conditions = append(status.Status.Conditions, updateCondition)
				updated, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().UpdateStatus(context.TODO(), status, metav1.UpdateOptions{})
				return err
			})
			framework.ExpectNoError(err, "updating CustomResourceDefinition status")
			expectCondition(updated.Status.Conditions, updateCondition)

			patchCondition := v1.CustomResourceDefinitionCondition{Message: "patched"}
			patched, err := apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Patch(context.TODO(), crd.GetName(),
				types.JSONPatchType,
				[]byte(`[{"op": "add", "path": "/status/conditions", "value": [{"message": "patched"}]}]`), metav1.PatchOptions{},
				"status")
			framework.ExpectNoError(err, "patching CustomResourceDefinition status")
			expectCondition(updated.Status.Conditions, updateCondition)
			expectCondition(patched.Status.Conditions, patchCondition)
		})
	})

	/*
		Release: v1.16
		Testname: Custom Resource Definition, discovery
		Description: Fetch /apis, /apis/apiextensions.k8s.io, and /apis/apiextensions.k8s.io/v1 discovery documents,
		and ensure they indicate CustomResourceDefinition apiextensions.k8s.io/v1 resources are available.
	*/
	framework.ConformanceIt("should include custom resource definition resources in discovery documents", func() {
		{
			ginkgo.By("fetching the /apis discovery document")
			apiGroupList := &metav1.APIGroupList{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis").Do(context.TODO()).Into(apiGroupList)
			framework.ExpectNoError(err, "fetching /apis")

			ginkgo.By("finding the apiextensions.k8s.io API group in the /apis discovery document")
			var group *metav1.APIGroup
			for _, g := range apiGroupList.Groups {
				if g.Name == v1.GroupName {
					group = &g
					break
				}
			}
			framework.ExpectNotEqual(group, nil, "apiextensions.k8s.io API group not found in /apis discovery document")

			ginkgo.By("finding the apiextensions.k8s.io/v1 API group/version in the /apis discovery document")
			var version *metav1.GroupVersionForDiscovery
			for _, v := range group.Versions {
				if v.Version == v1.SchemeGroupVersion.Version {
					version = &v
					break
				}
			}
			framework.ExpectNotEqual(version, nil, "apiextensions.k8s.io/v1 API group version not found in /apis discovery document")
		}

		{
			ginkgo.By("fetching the /apis/apiextensions.k8s.io discovery document")
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/apiextensions.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err, "fetching /apis/apiextensions.k8s.io")
			framework.ExpectEqual(group.Name, v1.GroupName, "verifying API group name in /apis/apiextensions.k8s.io discovery document")

			ginkgo.By("finding the apiextensions.k8s.io/v1 API group/version in the /apis/apiextensions.k8s.io discovery document")
			var version *metav1.GroupVersionForDiscovery
			for _, v := range group.Versions {
				if v.Version == v1.SchemeGroupVersion.Version {
					version = &v
					break
				}
			}
			framework.ExpectNotEqual(version, nil, "apiextensions.k8s.io/v1 API group version not found in /apis/apiextensions.k8s.io discovery document")
		}

		{
			ginkgo.By("fetching the /apis/apiextensions.k8s.io/v1 discovery document")
			apiResourceList := &metav1.APIResourceList{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/apiextensions.k8s.io/v1").Do(context.TODO()).Into(apiResourceList)
			framework.ExpectNoError(err, "fetching /apis/apiextensions.k8s.io/v1")
			framework.ExpectEqual(apiResourceList.GroupVersion, v1.SchemeGroupVersion.String(), "verifying API group/version in /apis/apiextensions.k8s.io/v1 discovery document")

			ginkgo.By("finding customresourcedefinitions resources in the /apis/apiextensions.k8s.io/v1 discovery document")
			var crdResource *metav1.APIResource
			for i := range apiResourceList.APIResources {
				if apiResourceList.APIResources[i].Name == "customresourcedefinitions" {
					crdResource = &apiResourceList.APIResources[i]
				}
			}
			framework.ExpectNotEqual(crdResource, nil, "customresourcedefinitions resource not found in /apis/apiextensions.k8s.io/v1 discovery document")
		}
	})

	/*
		Release: v1.17
		Testname: Custom Resource Definition, defaulting
		Description: Create a custom resource definition without default. Create CR. Add default and read CR until
		the default is applied. Create another CR. Remove default, add default for another field and read CR until
		new field is defaulted, but old default stays.
	*/
	framework.ConformanceIt("custom resource defaulting for requests and from storage works ", func() {
		config, err := framework.LoadConfig()
		framework.ExpectNoError(err, "loading config")
		apiExtensionClient, err := clientset.NewForConfig(config)
		framework.ExpectNoError(err, "initializing apiExtensionClient")
		dynamicClient, err := dynamic.NewForConfig(config)
		framework.ExpectNoError(err, "initializing dynamic client")

		// Create CRD without default and waits for the resource to be recognized and available.
		crd := fixtures.NewRandomNameV1CustomResourceDefinition(v1.ClusterScoped)
		if crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties == nil {
			crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties = map[string]v1.JSONSchemaProps{}
		}
		crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["a"] = v1.JSONSchemaProps{Type: "string"}
		crd.Spec.Versions[0].Schema.OpenAPIV3Schema.Properties["b"] = v1.JSONSchemaProps{Type: "string"}
		crd, err = fixtures.CreateNewV1CustomResourceDefinitionWatchUnsafe(crd, apiExtensionClient)
		framework.ExpectNoError(err, "creating CustomResourceDefinition")
		defer func() {
			err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
			framework.ExpectNoError(err, "deleting CustomResourceDefinition")
		}()

		// create CR without default in storage
		name1 := names.SimpleNameGenerator.GenerateName("cr-1")
		gvr := schema.GroupVersionResource{
			Group:    crd.Spec.Group,
			Version:  crd.Spec.Versions[0].Name,
			Resource: crd.Spec.Names.Plural,
		}
		crClient := dynamicClient.Resource(gvr)
		_, err = crClient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name": name1,
			},
		}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating CR")

		// Setting default for a to "A" and waiting for the CR to get defaulted on read
		crd, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Patch(context.TODO(), crd.Name, types.JSONPatchType, []byte(`[
			{"op":"add","path":"/spec/versions/0/schema/openAPIV3Schema/properties/a/default", "value": "A"}
		]`), metav1.PatchOptions{})
		framework.ExpectNoError(err, "setting default for a to \"A\" in schema")

		err = wait.PollImmediate(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
			u1, err := crClient.Get(context.TODO(), name1, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			a, found, err := unstructured.NestedFieldNoCopy(u1.Object, "a")
			if err != nil {
				return false, err
			}
			if !found {
				return false, nil
			}
			if a != "A" {
				return false, fmt.Errorf("expected a:\"A\", but got a:%q", a)
			}
			return true, nil
		})
		framework.ExpectNoError(err, "waiting for CR to be defaulted on read")

		// create CR with default in storage
		name2 := names.SimpleNameGenerator.GenerateName("cr-2")
		u2, err := crClient.Create(context.TODO(), &unstructured.Unstructured{Object: map[string]interface{}{
			"apiVersion": gvr.Group + "/" + gvr.Version,
			"kind":       crd.Spec.Names.Kind,
			"metadata": map[string]interface{}{
				"name": name2,
			},
		}}, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating CR")
		v, found, err := unstructured.NestedFieldNoCopy(u2.Object, "a")
		if !found {
			framework.Failf("field `a` should have been defaulted in %+v", u2.Object)
		}
		framework.ExpectEqual(v, "A", "\"a\" is defaulted to \"A\"")

		// Deleting default for a, adding default "B" for b and waiting for the CR to get defaulted on read for b
		crd, err = apiExtensionClient.ApiextensionsV1().CustomResourceDefinitions().Patch(context.TODO(), crd.Name, types.JSONPatchType, []byte(`[
			{"op":"remove","path":"/spec/versions/0/schema/openAPIV3Schema/properties/a/default"},
			{"op":"add","path":"/spec/versions/0/schema/openAPIV3Schema/properties/b/default", "value": "B"}
		]`), metav1.PatchOptions{})
		framework.ExpectNoError(err, "setting default for b to \"B\" and remove default for a")

		err = wait.PollImmediate(time.Millisecond*100, wait.ForeverTestTimeout, func() (bool, error) {
			u2, err := crClient.Get(context.TODO(), name2, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			b, found, err := unstructured.NestedFieldNoCopy(u2.Object, "b")
			if err != nil {
				return false, err
			}
			if !found {
				return false, nil
			}
			if b != "B" {
				return false, fmt.Errorf("expected b:\"B\", but got b:%q", b)
			}
			a, found, err := unstructured.NestedFieldNoCopy(u2.Object, "a")
			if err != nil {
				return false, err
			}
			if !found {
				return false, fmt.Errorf("expected a:\"A\" to be unchanged, but it was removed")
			}
			if a != "A" {
				return false, fmt.Errorf("expected a:\"A\" to be unchanged, but it changed to %q", a)
			}
			return true, nil
		})
		framework.ExpectNoError(err, "waiting for CR to be defaulted on read for b and a staying the same")
	})

})

func unstructuredToCRD(obj *unstructured.Unstructured) *v1.CustomResourceDefinition {
	crd := new(v1.CustomResourceDefinition)
	err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, crd)
	framework.ExpectNoError(err, "converting unstructured to CustomResourceDefinition")
	return crd
}

func expectCondition(conditions []v1.CustomResourceDefinitionCondition, expected v1.CustomResourceDefinitionCondition) {
	for _, c := range conditions {
		if equality.Semantic.DeepEqual(c, expected) {
			return
		}
	}
	framework.Failf("Condition %#v not found in conditions %#v", expected, conditions)
}
