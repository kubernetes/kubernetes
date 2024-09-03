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
	"fmt"
	"github.com/onsi/gomega"
	"sync"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
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
	"k8s.io/client-go/dynamic/dynamicinformer"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/crd"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("CustomResourceFieldSelectors [Privileged:ClusterAdmin]", framework.WithFeatureGate(apiextensionsfeatures.CustomResourceFieldSelectors), func() {

	f := framework.NewDefaultFramework("crd-selectable-fields")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("CustomResourceFieldSelectors", func() {
		customResourceClient := func(crd *apiextensionsv1.CustomResourceDefinition, version string) (dynamic.NamespaceableResourceInterface, schema.GroupVersionResource) {
			gvrs := fixtures.GetGroupVersionResourcesOfCustomResource(crd)
			for _, gvr := range gvrs {
				if gvr.Version == version {
					return f.DynamicClient.Resource(gvr), gvr
				}
			}
			ginkgo.Fail(fmt.Sprintf("Expected version '%s' in custom resource definition", version))
			return nil, schema.GroupVersionResource{}
		}

		var apiVersions = []apiextensionsv1.CustomResourceDefinitionVersion{
			{
				Name:    "v1",
				Served:  true,
				Storage: true,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"hostPort": {Type: "string"},
						},
					},
				},
				SelectableFields: []apiextensionsv1.SelectableField{
					{JSONPath: ".hostPort"},
				},
			},
			{
				Name:    "v2",
				Served:  true,
				Storage: false,
				Schema: &apiextensionsv1.CustomResourceValidation{
					OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{
						Type: "object",
						Properties: map[string]apiextensionsv1.JSONSchemaProps{
							"host": {Type: "string"},
							"port": {Type: "string"},
						},
					},
				},
				SelectableFields: []apiextensionsv1.SelectableField{
					{JSONPath: ".host"},
					{JSONPath: ".port"},
				},
			},
		}

		var certCtx *certContext
		servicePort := int32(9443)
		containerPort := int32(9444)

		ginkgo.BeforeEach(func(ctx context.Context) {
			ginkgo.DeferCleanup(cleanCRDWebhookTest, f.ClientSet, f.Namespace.Name)

			ginkgo.By("Setting up server cert")
			certCtx = setupServerCert(f.Namespace.Name, serviceCRDName)
			createAuthReaderRoleBindingForCRDConversion(ctx, f, f.Namespace.Name)

			deployCustomResourceWebhookAndService(ctx, f, imageutils.GetE2EImage(imageutils.Agnhost), certCtx, servicePort, containerPort)
		})

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
			testcrd, err := crd.CreateMultiVersionTestCRD(f, "stable.example.com", func(crd *apiextensionsv1.CustomResourceDefinition) {
				crd.Spec.Versions = apiVersions
				crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
					Strategy: apiextensionsv1.WebhookConverter,
					Webhook: &apiextensionsv1.WebhookConversion{
						ClientConfig: &apiextensionsv1.WebhookClientConfig{
							CABundle: certCtx.signingCert,
							Service: &apiextensionsv1.ServiceReference{
								Namespace: f.Namespace.Name,
								Name:      serviceCRDName,
								Path:      ptr.To("/crdconvert"),
								Port:      ptr.To(servicePort),
							},
						},
						ConversionReviewVersions: []string{"v1", "v1beta1"},
					},
				}
				crd.Spec.PreserveUnknownFields = false
			})
			if err != nil {
				return
			}
			ginkgo.DeferCleanup(testcrd.CleanUp)

			ginkgo.By("Creating a custom resource conversion webhook")
			waitWebhookConversionReady(ctx, f, testcrd.Crd, testcrd.DynamicClients, "v2")
			crd := testcrd.Crd

			ginkgo.By("Watching with field selectors")

			v2Client, v2gvr := customResourceClient(crd, "v2")
			v2hostWatch, err := v2Client.Namespace(f.Namespace.Name).Watch(ctx, metav1.ListOptions{FieldSelector: "host=host1"})
			framework.ExpectNoError(err, "watching custom resources with field selector")
			v2hostWatchAcc := watchAccumulator(v2hostWatch)
			v2hostPortWatch, err := v2Client.Namespace(f.Namespace.Name).Watch(ctx, metav1.ListOptions{FieldSelector: "host=host1,port=80"})
			framework.ExpectNoError(err, "watching custom resources with field selector")
			framework.ExpectNoError(err, "watching custom resources with field selector")
			v2hostPortWatchAcc := watchAccumulator(v2hostPortWatch)

			v1Client, _ := customResourceClient(crd, "v1")
			v1hostPortWatch, err := v1Client.Namespace(f.Namespace.Name).Watch(ctx, metav1.ListOptions{FieldSelector: "hostPort=host1:80"})
			framework.ExpectNoError(err, "watching custom resources with field selector")
			v1hostPortWatchAcc := watchAccumulator(v1hostPortWatch)

			ginkgo.By("Registering informers with field selectors")

			informerCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()

			hostInformer := dynamicinformer.NewFilteredDynamicSharedInformerFactory(f.DynamicClient, 0, f.Namespace.Name, func(opts *metav1.ListOptions) {
				opts.FieldSelector = "host=host1"
			})
			hostInformer.Start(informerCtx.Done())

			hostPortInformer := dynamicinformer.NewFilteredDynamicSharedInformerFactory(f.DynamicClient, 0, f.Namespace.Name, func(opts *metav1.ListOptions) {
				opts.FieldSelector = "host=host1,port=80"
			})
			hostPortInformer.Start(informerCtx.Done())

			v2HostInformer := hostInformer.ForResource(v2gvr).Informer()
			go v2HostInformer.Run(informerCtx.Done())
			v2HostPortInformer := hostPortInformer.ForResource(v2gvr).Informer()
			go v2HostPortInformer.Run(informerCtx.Done())

			v2HostInformerAcc := informerAccumulator(v2HostInformer)
			v2HostPortInformerAcc := informerAccumulator(v2HostPortInformer)

			framework.ExpectNoError(err, "adding event handler")

			ginkgo.By("Creating custom resources")
			toCreate := []map[string]any{
				{
					"host": "host1",
					"port": "80",
				},
				{
					"host": "host1",
					"port": "8080",
				},
				{
					"host": "host2",
				},
			}

			crNames := make([]string, len(toCreate))
			for i, spec := range toCreate {
				name := names.SimpleNameGenerator.GenerateName("selectable-field-cr")
				crNames[i] = name

				obj := map[string]interface{}{
					"apiVersion": v2gvr.Group + "/" + v2gvr.Version,
					"kind":       crd.Spec.Names.Kind,
					"metadata": map[string]interface{}{
						"name":      name,
						"namespace": f.Namespace.Name,
					},
				}
				for k, v := range spec {
					obj[k] = v
				}
				_, err = v2Client.Namespace(f.Namespace.Name).Create(ctx, &unstructured.Unstructured{Object: obj}, metav1.CreateOptions{})
				framework.ExpectNoError(err, "creating custom resource")
			}
			ginkgo.By("Listing v2 custom resources with field selector host=host1")
			list, err := v2Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "host=host1"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[0], crNames[1])))

			ginkgo.By("Listing v2 custom resources with field selector host=host1,port=80")
			list, err = v2Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "host=host1,port=80"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[0])))

			ginkgo.By("Listing v1 custom resources with field selector hostPort=host1:80")
			list, err = v1Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "hostPort=host1:80"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[0])))

			ginkgo.By("Listing v1 custom resources with field selector hostPort=host1:8080")
			list, err = v1Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "hostPort=host1:8080"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New(crNames[1])))

			ginkgo.By("Waiting for watch events to contain v2 custom resources for field selector host=host1")
			gomega.Eventually(ctx, v2hostWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for watch events to contain v2 custom resources for field selector host=host1,port=80")
			gomega.Eventually(ctx, v2hostPortWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0]))))

			ginkgo.By("Waiting for watch events to contain v1 custom resources for field selector hostPort=host1:80")
			gomega.Eventually(ctx, v1hostPortWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0]))))

			ginkgo.By("Waiting for informer events to contain v2 custom resources for field selector host=host1")
			gomega.Eventually(ctx, v2HostInformerAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for informer events to contain v2 custom resources for field selector host=host1,port=80")
			gomega.Eventually(ctx, v2HostPortInformerAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedEvents(sets.New(crNames[0]))))

			ginkgo.By("Deleting one custom resources to ensure that deletions are observed")
			var gracePeriod int64 = 0
			err = v2Client.Namespace(f.Namespace.Name).DeleteCollection(ctx, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriod}, metav1.ListOptions{FieldSelector: "host=host1,port=80"})
			framework.ExpectNoError(err, "deleting custom resource")

			ginkgo.By("Updating one custom resources to ensure that deletions are observed")
			u, err := v2Client.Namespace(f.Namespace.Name).Get(ctx, crNames[1], metav1.GetOptions{})
			framework.ExpectNoError(err, "getting custom resource")
			u.Object["host"] = "host2"
			_, err = v2Client.Namespace(f.Namespace.Name).Update(ctx, u, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "updating custom resource")

			ginkgo.By("Listing v2 custom resources after updates and deletes for field selector host=host1")
			list, err = v2Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "host=host1"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New[string]()))

			ginkgo.By("Listing v2 custom resources after updates and deletes for field selector host=host1,port=80")
			list, err = v2Client.Namespace(f.Namespace.Name).List(ctx, metav1.ListOptions{FieldSelector: "host=host1,port=80"})
			framework.ExpectNoError(err, "listing custom resources with field selector")
			gomega.Expect(listResultToNames(list)).To(gomega.Equal(sets.New[string]()))

			ginkgo.By("Waiting for v2 watch events after updates and deletes for field selector host=host1")
			gomega.Eventually(ctx, v2hostWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedAndDeletedEvents(sets.New(crNames[0], crNames[1]), sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for v2 watch events after updates and deletes for field selector host=host1,port=80")
			gomega.Eventually(ctx, v2hostPortWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedAndDeletedEvents(sets.New(crNames[0]), sets.New(crNames[0]))))

			ginkgo.By("Waiting for v1 watch events after updates and deletes for field selector hostPort=host1:80")
			gomega.Eventually(ctx, v1hostPortWatchAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedAndDeletedEvents(sets.New(crNames[0]), sets.New(crNames[0]))))

			ginkgo.By("Waiting for v2 informer events after updates and deletes for field selector host=host1")
			gomega.Eventually(ctx, v2HostInformerAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedAndDeletedEvents(sets.New(crNames[0], crNames[1]), sets.New(crNames[0], crNames[1]))))

			ginkgo.By("Waiting for v2 informer events after updates and deletes for field selector host=host1,port=80")
			gomega.Eventually(ctx, v2HostPortInformerAcc).WithPolling(5 * time.Millisecond).WithTimeout(30 * time.Second).
				Should(gomega.Equal(addedAndDeletedEvents(sets.New(crNames[0]), sets.New(crNames[0]))))
		})
	})
})

type accumulatedEvents struct {
	added, deleted sets.Set[string]
}

func emptyEvents() *accumulatedEvents {
	return &accumulatedEvents{added: sets.New[string](), deleted: sets.New[string]()}
}

func addedEvents(added sets.Set[string]) *accumulatedEvents {
	return &accumulatedEvents{added: added, deleted: sets.New[string]()}
}

func addedAndDeletedEvents(added, deleted sets.Set[string]) *accumulatedEvents {
	return &accumulatedEvents{added: added, deleted: deleted}
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

func informerAccumulator(informer cache.SharedIndexInformer) func(ctx context.Context) (*accumulatedEvents, error) {
	var lock sync.Mutex
	result := emptyEvents()

	_, err := informer.AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			defer ginkgo.GinkgoRecover()
			lock.Lock()
			defer lock.Unlock()
			result.added.Insert(obj.(*unstructured.Unstructured).GetName())
		},
		UpdateFunc: func(oldObj, newObj any) {
			defer ginkgo.GinkgoRecover()
			// ignoring
		},
		DeleteFunc: func(obj any) {
			defer ginkgo.GinkgoRecover()
			lock.Lock()
			defer lock.Unlock()
			result.deleted.Insert(obj.(*unstructured.Unstructured).GetName())
		},
	})

	return func(ctx context.Context) (*accumulatedEvents, error) {
		if err != nil {
			return nil, err
		}
		lock.Lock()
		defer lock.Unlock()
		return result, nil
	}
}

func listResultToNames(list *unstructured.UnstructuredList) sets.Set[string] {
	found := sets.New[string]()
	for _, i := range list.Items {
		found.Insert(i.GetName())
	}
	return found
}
