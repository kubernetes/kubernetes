/*
Copyright 2024 The Kubernetes Authors.

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
	"encoding/json"
	"github.com/onsi/gomega"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("CustomResourceFieldSelectors [Privileged:ClusterAdmin]", framework.WithFeatureGate(apiextensionsfeatures.CustomResourceFieldSelectors), func() {

	f := framework.NewDefaultFramework("crd-selectable-fields")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("CustomResourceFieldSelectors", func() {
		var apiExtensionClient *clientset.Clientset
		ginkgo.BeforeEach(func() {
			var err error
			apiExtensionClient, err = clientset.NewForConfig(f.ClientConfig())
			framework.ExpectNoError(err, "initializing apiExtensionClient")
		})

		customResourceClient := func(crd *apiextensionsv1.CustomResourceDefinition) (dynamic.NamespaceableResourceInterface, schema.GroupVersionResource) {
			gvrs := fixtures.GetGroupVersionResourcesOfCustomResource(crd)
			if len(gvrs) != 1 {
				ginkgo.Fail("Expected one version in custom resource definition")
			}
			gvr := gvrs[0]
			return f.DynamicClient.Resource(gvr), gvr
		}

		var schemaWithValidationExpression = unmarshalSchema([]byte(`{
			"type":"object",
			"properties":{
				"spec":{
					"type":"object",
					"properties":{
						"color":{ "type":"string" },
						"quantity":{ "type":"integer" }
					}
				}
			}
		}`))

		/*
			Release: v1.31
			Testname: Custom Resource Definition, list and watch with selectable fields
			Description: Create a Custom Resource Definition with SelectableFields. Create custom resources. Attempt to
			list and watch custom resources with object selectors; the list and watch MUST return only custom resources
			matching the field selector. Delete and update some of the custom resources. Attempt to list and watch the
			custom resources with object selectors; the list and watch MUST return only the custom resources matching
			the object selectors.
		*/
		framework.It("MUST list and watch custom resources matching the field selector", func(ctx context.Context) {
			ginkgo.By("Creating a custom resource definition with selectable fields")
			crd := fixtures.NewRandomNameV1CustomResourceDefinitionWithSchema(apiextensionsv1.NamespaceScoped, schemaWithValidationExpression, false)
			for i := range crd.Spec.Versions {
				crd.Spec.Versions[i].SelectableFields = []apiextensionsv1.SelectableField{
					{JSONPath: ".spec.color"},
					{JSONPath: ".spec.quantity"},
				}
			}
			crd, err := fixtures.CreateNewV1CustomResourceDefinition(crd, apiExtensionClient, f.DynamicClient)
			framework.ExpectNoError(err, "creating CustomResourceDefinition")
			defer func() {
				err = fixtures.DeleteV1CustomResourceDefinition(crd, apiExtensionClient)
				framework.ExpectNoError(err, "deleting CustomResourceDefinition")
			}()

			ginkgo.By("Watching with field selectors")
			crClient, gvr := customResourceClient(crd)

			watchSimpleSelector, err := crClient.Namespace(f.Namespace.Name).Watch(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue"})
			framework.ExpectNoError(err, "watching custom resources with field selector")
			defer func() {
				watchSimpleSelector.Stop()
			}()
			watchCompoundSelector, err := crClient.Namespace(f.Namespace.Name).Watch(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue,spec.quantity=2"})
			framework.ExpectNoError(err, "watching custom resources with field selector")
			defer func() {
				watchCompoundSelector.Stop()
			}()

			ginkgo.By("Creating custom resources")
			toCreate := []map[string]any{
				{
					"color":    "blue",
					"quantity": int64(2),
				},
				{
					"color":    "blue",
					"quantity": int64(3),
				},
				{
					"color": "green",
				},
			}

			crNames := make([]string, len(toCreate))
			for i, spec := range toCreate {
				name := names.SimpleNameGenerator.GenerateName("selectable-field-cr")
				crNames[i] = name
				_, err = crClient.Namespace(f.Namespace.Name).Create(ctx, &unstructured.Unstructured{Object: map[string]interface{}{
					"apiVersion": gvr.Group + "/" + gvr.Version,
					"kind":       crd.Spec.Names.Kind,
					"metadata": map[string]interface{}{
						"name":      name,
						"namespace": f.Namespace.Name,
					},
					"spec": spec,
				}}, metav1.CreateOptions{})
				framework.ExpectNoError(err, "creating custom resource")
			}

			ginkgo.By("Listing custom resources with field selector spec.color=blue")
			list, err := crClient.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[0], crNames[1])))

			ginkgo.By("Listing custom resources with field selector spec.color=blue,spec.quantity=2")
			list, err = crClient.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue,spec.quantity=2"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[0])))

			ginkgo.By("Waiting for watch events to contain custom resources for field selector spec.color=blue")
			gomega.Eventually(ctx, watchAccumulator(watchSimpleSelector)).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for watch events to contain custom resources for field selector spec.color=blue,spec.quantity=2")
			gomega.Eventually(ctx, watchAccumulator(watchCompoundSelector)).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0]))))

			ginkgo.By("Deleting one custom resources to ensure that deletions are observed")
			var gracePeriod int64 = 0
			err = crClient.Namespace(f.Namespace.Name).Delete(ctx, crNames[0], metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod})
			framework.ExpectNoError(err, "deleting custom resource")

			ginkgo.By("Updating one custom resources to ensure that deletions are observed")
			u, err := crClient.Namespace(f.Namespace.Name).Get(ctx, crNames[1], metav1.GetOptions{})
			framework.ExpectNoError(err, "getting custom resource")
			u.Object["spec"].(map[string]any)["color"] = "green"
			_, err = crClient.Namespace(f.Namespace.Name).Update(ctx, u, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "updating custom resource")

			ginkgo.By("Listing custom resources after updates and deletes for field selector spec.color=blue")
			list, err = crClient.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New[string]()))

			ginkgo.By("Listing custom resources after updates and deletes for field selector spec.color=blue,spec.quantity=2")
			list, err = crClient.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "spec.color=blue,spec.quantity=2"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New[string]()))

			ginkgo.By("Waiting for watch events after updates and deletes for field selector spec.color=blue")
			gomega.Eventually(ctx, watchAccumulator(watchSimpleSelector)).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(deletedEvents(sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for watch events after updates and deletes for field selector spec.color=blue,spec.quantity=2")
			gomega.Eventually(ctx, watchAccumulator(watchCompoundSelector)).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(deletedEvents(sets.New(crNames[0]))))
		})
	})
})

func unmarshalSchema(schemaJSON []byte) *apiextensionsv1.JSONSchemaProps {
	var c apiextensionsv1.JSONSchemaProps
	err := json.Unmarshal(schemaJSON, &c)
	framework.ExpectNoError(err, "unmarshalling OpenAPIv3 schema")
	return &c
}

type accumulatedEvents struct {
	added, deleted sets.Set[string]
}

func emptyEvents() *accumulatedEvents {
	return &accumulatedEvents{added: sets.New[string](), deleted: sets.New[string]()}
}

func addedEvents(added sets.Set[string]) *accumulatedEvents {
	return &accumulatedEvents{added: added, deleted: sets.New[string]()}
}

func deletedEvents(deleted sets.Set[string]) *accumulatedEvents {
	return &accumulatedEvents{added: sets.New[string](), deleted: deleted}
}

func watchAccumulator(w watch.Interface) func(ctx context.Context) (*accumulatedEvents, error) {
	result := emptyEvents()
	return func(ctx context.Context) (*accumulatedEvents, error) {
		for {
			select {
			case event := <-w.ResultChan():
				obj, err := meta.Accessor(event.Object)
				framework.ExpectNoError(err, "accessing object name")
				switch event.Type {
				case watch.Added:
					result.added.Insert(obj.GetName())
				case watch.Deleted:
					result.deleted.Insert(obj.GetName())
				}
			default:
				return result, nil
			}
		}
	}
}

func listResultToNames(list *unstructured.UnstructuredList) sets.Set[string] {
	found := sets.New[string]()
	for _, i := range list.Items {
		found.Insert(i.GetName())
	}
	return found
}
