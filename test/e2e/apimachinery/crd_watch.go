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

package apimachinery

import (
	"fmt"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = SIGDescribe("CustomResourceDefinition Watch", func() {

	f := framework.NewDefaultFramework("crd-watch")

	Context("CustomResourceDefinition Watch", func() {
		/*
			   	   Testname: crd-watch
			   	   Description: Create a Custom Resource Definition and make sure
				   watches observe events on create/delete.
		*/
		It("watch on custom resource definition objects", func() {

			framework.SkipUnlessServerVersionGTE(crdVersion, f.ClientSet.Discovery())

			const (
				watchCRNameA = "name1"
				watchCRNameB = "name2"
			)

			config, err := framework.LoadConfig()
			if err != nil {
				framework.Failf("failed to load config: %v", err)
			}

			apiExtensionClient, err := clientset.NewForConfig(config)
			if err != nil {
				framework.Failf("failed to initialize apiExtensionClient: %v", err)
			}

			noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.ClusterScoped)
			noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, f.DynamicClient)
			if err != nil {
				framework.Failf("failed to create CustomResourceDefinition: %v", err)
			}

			defer func() {
				err = fixtures.DeleteCustomResourceDefinition(noxuDefinition, apiExtensionClient)
				if err != nil {
					framework.Failf("failed to delete CustomResourceDefinition: %v", err)
				}
			}()

			ns := ""
			noxuResourceClient := newNamespacedCustomResourceClient(ns, f.DynamicClient, noxuDefinition)

			watchA, err := watchCRWithName(noxuResourceClient, watchCRNameA)
			Expect(err).NotTo(HaveOccurred(), "failed to watch custom resource: %s", watchCRNameA)

			watchB, err := watchCRWithName(noxuResourceClient, watchCRNameB)
			Expect(err).NotTo(HaveOccurred(), "failed to watch custom resource: %s", watchCRNameB)

			testCrA := fixtures.NewNoxuInstance(ns, watchCRNameA)
			testCrB := fixtures.NewNoxuInstance(ns, watchCRNameB)

			By("Creating first CR ")
			testCrA, err = instantiateCustomResource(testCrA, noxuResourceClient, noxuDefinition)
			Expect(err).NotTo(HaveOccurred(), "failed to instantiate custom resource: %+v", testCrA)
			expectEvent(watchA, watch.Added, testCrA)
			expectNoEvent(watchB, watch.Added, testCrA)

			By("Creating second CR")
			testCrB, err = instantiateCustomResource(testCrB, noxuResourceClient, noxuDefinition)
			Expect(err).NotTo(HaveOccurred(), "failed to instantiate custom resource: %+v", testCrB)
			expectEvent(watchB, watch.Added, testCrB)
			expectNoEvent(watchA, watch.Added, testCrB)

			By("Deleting first CR")
			err = deleteCustomResource(noxuResourceClient, watchCRNameA)
			Expect(err).NotTo(HaveOccurred(), "failed to delete custom resource: %s", watchCRNameA)
			expectEvent(watchA, watch.Deleted, nil)
			expectNoEvent(watchB, watch.Deleted, nil)

			By("Deleting second CR")
			err = deleteCustomResource(noxuResourceClient, watchCRNameB)
			Expect(err).NotTo(HaveOccurred(), "failed to delete custom resource: %s", watchCRNameB)
			expectEvent(watchB, watch.Deleted, nil)
			expectNoEvent(watchA, watch.Deleted, nil)
		})
	})
})

func watchCRWithName(crdResourceClient dynamic.ResourceInterface, name string) (watch.Interface, error) {
	return crdResourceClient.Watch(
		metav1.ListOptions{
			FieldSelector:  "metadata.name=" + name,
			TimeoutSeconds: int64ptr(600),
		},
	)
}

func instantiateCustomResource(instanceToCreate *unstructured.Unstructured, client dynamic.ResourceInterface, definition *apiextensionsv1beta1.CustomResourceDefinition) (*unstructured.Unstructured, error) {
	createdInstance, err := client.Create(instanceToCreate, metav1.CreateOptions{})
	if err != nil {
		return nil, err
	}
	createdObjectMeta, err := meta.Accessor(createdInstance)
	if err != nil {
		return nil, err
	}
	// it should have a UUID
	if len(createdObjectMeta.GetUID()) == 0 {
		return nil, fmt.Errorf("missing uuid: %#v", createdInstance)
	}
	createdTypeMeta, err := meta.TypeAccessor(createdInstance)
	if err != nil {
		return nil, err
	}
	if e, a := definition.Spec.Group+"/"+definition.Spec.Version, createdTypeMeta.GetAPIVersion(); e != a {
		return nil, fmt.Errorf("expected %v, got %v", e, a)
	}
	if e, a := definition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
		return nil, fmt.Errorf("expected %v, got %v", e, a)
	}
	return createdInstance, nil
}

func deleteCustomResource(client dynamic.ResourceInterface, name string) error {
	return client.Delete(name, &metav1.DeleteOptions{})
}

func newNamespacedCustomResourceClient(ns string, client dynamic.Interface, crd *apiextensionsv1beta1.CustomResourceDefinition) dynamic.ResourceInterface {
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural}

	if crd.Spec.Scope != apiextensionsv1beta1.ClusterScoped {
		return client.Resource(gvr).Namespace(ns)
	} else {
		return client.Resource(gvr)
	}

}
