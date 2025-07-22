/*
Copyright 2015 The Kubernetes Authors.

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

package network

import (
	"context"
	"encoding/json"
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("Ingress API", func() {
	f := framework.NewDefaultFramework("ingress")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	/*
		Release: v1.19
		Testname: Ingress API
		Description:
		The networking.k8s.io API group MUST exist in the /apis discovery document.
		The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		The ingresses resources MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		The ingresses resource must support create, get, list, watch, update, patch, delete, and deletecollection.
		The ingresses/status resource must support update and patch
	*/

	framework.ConformanceIt("should support creating Ingress API operations", func(ctx context.Context) {
		// Setup
		ns := f.Namespace.Name
		ingVersion := "v1"
		ingClient := f.ClientSet.NetworkingV1().Ingresses(ns)

		prefixPathType := networkingv1.PathTypeImplementationSpecific
		serviceBackend := &networkingv1.IngressServiceBackend{
			Name: "default-backend",
			Port: networkingv1.ServiceBackendPort{
				Name:   "",
				Number: 8080,
			},
		}
		defaultBackend := networkingv1.IngressBackend{
			Service: serviceBackend,
		}

		ingTemplate := &networkingv1.Ingress{
			ObjectMeta: metav1.ObjectMeta{GenerateName: "e2e-example-ing",
				Labels: map[string]string{
					"special-label": f.UniqueName,
				}},
			Spec: networkingv1.IngressSpec{
				DefaultBackend: &defaultBackend,
				Rules: []networkingv1.IngressRule{
					{
						Host: "foo.bar.com",
						IngressRuleValue: networkingv1.IngressRuleValue{
							HTTP: &networkingv1.HTTPIngressRuleValue{
								Paths: []networkingv1.HTTPIngressPath{{
									Path:     "/",
									PathType: &prefixPathType,
									Backend: networkingv1.IngressBackend{
										Service: &networkingv1.IngressServiceBackend{
											Name: "test-backend",
											Port: networkingv1.ServiceBackendPort{
												Number: 8080,
											},
										},
									},
								}},
							},
						},
					},
				},
			},
			Status: networkingv1.IngressStatus{LoadBalancer: networkingv1.IngressLoadBalancerStatus{}},
		}

		ingress1 := ingTemplate.DeepCopy()
		ingress1.Spec.Rules[0].Host = "host1.bar.com"
		ingress2 := ingTemplate.DeepCopy()
		ingress2.Spec.Rules[0].Host = "host2.bar.com"
		ingress3 := ingTemplate.DeepCopy()
		ingress3.Spec.Rules[0].Host = "host3.bar.com"

		// Discovery
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == networkingv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == ingVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected networking API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/networking.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == ingVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected networking API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/networking.k8s.io" + ingVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(networkingv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundIngress := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "ingresses":
					foundIngress = true
				}
			}
			if !foundIngress {
				framework.Failf("expected ingresses, got %#v", resources.APIResources)
			}
		}

		// Ingress resource create/read/update/watch verbs
		ginkgo.By("creating")
		_, err := ingClient.Create(ctx, ingress1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = ingClient.Create(ctx, ingress2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdIngress, err := ingClient.Create(ctx, ingress3, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenIngress, err := ingClient.Get(ctx, createdIngress.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenIngress.UID).To(gomega.Equal(createdIngress.UID))

		ginkgo.By("listing")
		ings, err := ingClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(ings.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		ingWatch, err := ingClient.Watch(ctx, metav1.ListOptions{ResourceVersion: ings.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		// Test cluster-wide list and watch
		clusterIngClient := f.ClientSet.NetworkingV1().Ingresses("")
		ginkgo.By("cluster-wide listing")
		clusterIngs, err := clusterIngClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(clusterIngs.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("cluster-wide watching")
		framework.Logf("starting watch")
		_, err = clusterIngClient.Watch(ctx, metav1.ListOptions{ResourceVersion: ings.ResourceVersion, LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedIngress, err := ingClient.Patch(ctx, createdIngress.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedIngress.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		var ingToUpdate, updatedIngress *networkingv1.Ingress
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			ingToUpdate, err = ingClient.Get(ctx, createdIngress.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			ingToUpdate.Annotations["updated"] = "true"
			updatedIngress, err = ingClient.Update(ctx, ingToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		gomega.Expect(updatedIngress.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-ingWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedIngress, isIngress := evt.Object.(*networkingv1.Ingress)
				if !isIngress {
					framework.Failf("expected Ingress, got %T", evt.Object)
				}
				if watchedIngress.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					ingWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedIngress.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		// /status subresource operations
		ginkgo.By("patching /status")
		lbStatus := networkingv1.IngressLoadBalancerStatus{
			Ingress: []networkingv1.IngressLoadBalancerIngress{{IP: "169.1.1.1"}},
		}
		lbStatusJSON, err := json.Marshal(lbStatus)
		framework.ExpectNoError(err)
		patchedStatus, err := ingClient.Patch(ctx, createdIngress.Name, types.MergePatchType,
			[]byte(`{"metadata":{"annotations":{"patchedstatus":"true"}},"status":{"loadBalancer":`+string(lbStatusJSON)+`}}`),
			metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(patchedStatus.Status.LoadBalancer).To(gomega.Equal(lbStatus), "patched object should have the applied loadBalancer status")
		gomega.Expect(patchedStatus.Annotations).To(gomega.HaveKeyWithValue("patchedstatus", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating /status")
		var statusToUpdate, updatedStatus *networkingv1.Ingress
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			statusToUpdate, err = ingClient.Get(ctx, createdIngress.Name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			statusToUpdate.Status.LoadBalancer = networkingv1.IngressLoadBalancerStatus{
				Ingress: []networkingv1.IngressLoadBalancerIngress{{IP: "169.1.1.2"}},
			}
			updatedStatus, err = ingClient.UpdateStatus(ctx, statusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err)
		gomega.Expect(updatedStatus.Status.LoadBalancer).To(gomega.Equal(statusToUpdate.Status.LoadBalancer), "updated object expected to have updated loadbalancer status %#v, got %#v", statusToUpdate.Status.LoadBalancer, updatedStatus.Status.LoadBalancer)

		ginkgo.By("get /status")
		ingResource := schema.GroupVersionResource{Group: "networking.k8s.io", Version: ingVersion, Resource: "ingresses"}
		gottenStatus, err := f.DynamicClient.Resource(ingResource).Namespace(ns).Get(ctx, createdIngress.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		statusUID, _, err := unstructured.NestedFieldCopy(gottenStatus.Object, "metadata", "uid")
		framework.ExpectNoError(err)
		gomega.Expect(string(createdIngress.UID)).To(gomega.Equal(statusUID), "createdIngress.UID: %v expected to match statusUID: %v ", createdIngress.UID, statusUID)

		// Ingress resource delete operations
		ginkgo.By("deleting")

		expectFinalizer := func(ing *networkingv1.Ingress, msg string) {
			gomega.Expect(ing.DeletionTimestamp).ToNot(gomega.BeNil(), "expected deletionTimestamp, got nil on step: %q, ingress: %+v", msg, ing)
			if len(ing.Finalizers) == 0 {
				framework.Failf("expected finalizers on ingress, got none on step: %q, ingress: %+v", msg, ing)
			}
		}

		err = ingClient.Delete(ctx, createdIngress.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		ing, err := ingClient.Get(ctx, createdIngress.Name, metav1.GetOptions{})
		// If ingress controller does not support finalizers, we expect a 404.  Otherwise we validate finalizer behavior.
		if err == nil {
			expectFinalizer(ing, "deleting createdIngress")
		} else {
			if !apierrors.IsNotFound(err) {
				framework.Failf("expected 404, got %v", err)
			}
		}
		ings, err = ingClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 3 items since some ingresses might not have been deleted yet due to finalizers
		if len(ings.Items) > 3 {
			framework.Fail("filtered list should have <= 3 items")
		}
		// Validate finalizer on the deleted ingress
		for _, ing := range ings.Items {
			if ing.Namespace == createdIngress.Namespace && ing.Name == createdIngress.Name {
				expectFinalizer(&ing, "listing after deleting createdIngress")
			}
		}

		ginkgo.By("deleting a collection")
		err = ingClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		ings, err = ingClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=" + f.UniqueName})
		framework.ExpectNoError(err)
		// Should have <= 3 items since some ingresses might not have been deleted yet due to finalizers
		if len(ings.Items) > 3 {
			framework.Fail("filtered list should have <= 3 items")
		}
		// Validate finalizers
		for _, ing := range ings.Items {
			expectFinalizer(&ing, "deleting ingress collection")
		}
	})
})
