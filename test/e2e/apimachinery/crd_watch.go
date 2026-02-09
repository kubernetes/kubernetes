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
	"context"
	"fmt"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("CustomResourceDefinition Watch [Privileged:ClusterAdmin]", func() {

	f := framework.NewDefaultFramework("crd-watch")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("CustomResourceDefinition Watch", func() {
		/*
			Release: v1.16
			Testname: Custom Resource Definition, watch
			Description: Create a Custom Resource Definition. Attempt to watch it; the watch MUST observe create,
			modify and delete events.
		*/
		framework.ConformanceIt("watch on custom resource definition objects", func(ctx context.Context) {

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

			noxuDefinition := fixtures.NewNoxuV1CustomResourceDefinition(apiextensionsv1.ClusterScoped)
			noxuDefinition, err = fixtures.CreateNewV1CustomResourceDefinition(noxuDefinition, apiExtensionClient, f.DynamicClient)
			if err != nil {
				framework.Failf("failed to create CustomResourceDefinition: %v", err)
			}

			defer func() {
				err = fixtures.DeleteV1CustomResourceDefinition(noxuDefinition, apiExtensionClient)
				if err != nil {
					framework.Failf("failed to delete CustomResourceDefinition: %v", err)
				}
			}()

			ns := ""
			noxuResourceClient, err := newNamespacedCustomResourceClient(ns, f.DynamicClient, noxuDefinition)
			framework.ExpectNoError(err, "creating custom resource client")

			watchA, err := watchCRWithName(ctx, noxuResourceClient, watchCRNameA)
			framework.ExpectNoError(err, "failed to watch custom resource: %s", watchCRNameA)

			watchB, err := watchCRWithName(ctx, noxuResourceClient, watchCRNameB)
			framework.ExpectNoError(err, "failed to watch custom resource: %s", watchCRNameB)

			testCrA := fixtures.NewNoxuInstance(ns, watchCRNameA)
			testCrB := fixtures.NewNoxuInstance(ns, watchCRNameB)

			ginkgo.By("Creating first CR ")
			testCrA, err = instantiateCustomResource(ctx, testCrA, noxuResourceClient, noxuDefinition)
			framework.ExpectNoError(err, "failed to instantiate custom resource: %+v", testCrA)
			expectEvent(watchA, watch.Added, testCrA)
			expectNoEvent(watchB, watch.Added, testCrA)

			ginkgo.By("Creating second CR")
			testCrB, err = instantiateCustomResource(ctx, testCrB, noxuResourceClient, noxuDefinition)
			framework.ExpectNoError(err, "failed to instantiate custom resource: %+v", testCrB)
			expectEvent(watchB, watch.Added, testCrB)
			expectNoEvent(watchA, watch.Added, testCrB)

			ginkgo.By("Modifying first CR")
			err = patchCustomResource(ctx, noxuResourceClient, watchCRNameA)
			framework.ExpectNoError(err, "failed to patch custom resource: %s", watchCRNameA)
			expectEvent(watchA, watch.Modified, nil)
			expectNoEvent(watchB, watch.Modified, nil)

			ginkgo.By("Modifying second CR")
			err = patchCustomResource(ctx, noxuResourceClient, watchCRNameB)
			framework.ExpectNoError(err, "failed to patch custom resource: %s", watchCRNameB)
			expectEvent(watchB, watch.Modified, nil)
			expectNoEvent(watchA, watch.Modified, nil)

			ginkgo.By("Deleting first CR")
			err = deleteCustomResource(ctx, noxuResourceClient, watchCRNameA)
			framework.ExpectNoError(err, "failed to delete custom resource: %s", watchCRNameA)
			expectEvent(watchA, watch.Deleted, nil)
			expectNoEvent(watchB, watch.Deleted, nil)

			ginkgo.By("Deleting second CR")
			err = deleteCustomResource(ctx, noxuResourceClient, watchCRNameB)
			framework.ExpectNoError(err, "failed to delete custom resource: %s", watchCRNameB)
			expectEvent(watchB, watch.Deleted, nil)
			expectNoEvent(watchA, watch.Deleted, nil)
		})
	})
})

func watchCRWithName(ctx context.Context, crdResourceClient dynamic.ResourceInterface, name string) (watch.Interface, error) {
	return crdResourceClient.Watch(
		ctx,
		metav1.ListOptions{
			FieldSelector:  "metadata.name=" + name,
			TimeoutSeconds: ptr.To[int64](600),
		},
	)
}

func instantiateCustomResource(ctx context.Context, instanceToCreate *unstructured.Unstructured, client dynamic.ResourceInterface, definition *apiextensionsv1.CustomResourceDefinition) (*unstructured.Unstructured, error) {
	createdInstance, err := client.Create(ctx, instanceToCreate, metav1.CreateOptions{})
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
	if len(definition.Spec.Versions) != 1 {
		return nil, fmt.Errorf("expected exactly one version, got %v", definition.Spec.Versions)
	}
	if e, a := definition.Spec.Group+"/"+definition.Spec.Versions[0].Name, createdTypeMeta.GetAPIVersion(); e != a {
		return nil, fmt.Errorf("expected %v, got %v", e, a)
	}
	if e, a := definition.Spec.Names.Kind, createdTypeMeta.GetKind(); e != a {
		return nil, fmt.Errorf("expected %v, got %v", e, a)
	}
	return createdInstance, nil
}

func patchCustomResource(ctx context.Context, client dynamic.ResourceInterface, name string) error {
	_, err := client.Patch(
		ctx,
		name,
		types.JSONPatchType,
		[]byte(`[{ "op": "add", "path": "/dummy", "value": "test" }]`),
		metav1.PatchOptions{})
	return err
}

func deleteCustomResource(ctx context.Context, client dynamic.ResourceInterface, name string) error {
	return client.Delete(ctx, name, metav1.DeleteOptions{})
}

func newNamespacedCustomResourceClient(ns string, client dynamic.Interface, crd *apiextensionsv1.CustomResourceDefinition) (dynamic.ResourceInterface, error) {
	if len(crd.Spec.Versions) != 1 {
		return nil, fmt.Errorf("expected exactly one version, got %v", crd.Spec.Versions)
	}
	gvr := schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Versions[0].Name, Resource: crd.Spec.Names.Plural}

	if crd.Spec.Scope != apiextensionsv1.ClusterScoped {
		return client.Resource(gvr).Namespace(ns), nil
	}
	return client.Resource(gvr), nil

}
