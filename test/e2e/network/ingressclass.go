/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	networkingv1beta1 "k8s.io/api/networking/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("IngressClass [Feature:Ingress]", func() {
	f := framework.NewDefaultFramework("ingressclass")
	var cs clientset.Interface
	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	ginkgo.It("should set default value on new IngressClass [Serial]", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", true, f.UniqueName)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingress, err := createBasicIngress(cs, f.Namespace.Name)
		framework.ExpectNoError(err)

		if ingress.Spec.IngressClassName == nil {
			framework.Failf("Expected IngressClassName to be set by Admission Controller")
		} else if *ingress.Spec.IngressClassName != ingressClass1.Name {
			framework.Failf("Expected IngressClassName to be %s, got %s", ingressClass1.Name, *ingress.Spec.IngressClassName)
		}
	})

	ginkgo.It("should not set default value if no default IngressClass [Serial]", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", false, f.UniqueName)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingress, err := createBasicIngress(cs, f.Namespace.Name)
		framework.ExpectNoError(err)

		if ingress.Spec.IngressClassName != nil {
			framework.Failf("Expected IngressClassName to be nil, got %s", *ingress.Spec.IngressClassName)
		}
	})

	ginkgo.It("should prevent Ingress creation if more than 1 IngressClass marked as default [Serial]", func() {
		ingressClass1, err := createIngressClass(cs, "ingressclass1", true, f.UniqueName)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass1.Name)

		ingressClass2, err := createIngressClass(cs, "ingressclass2", true, f.UniqueName)
		framework.ExpectNoError(err)
		defer deleteIngressClass(cs, ingressClass2.Name)

		// the admission controller may take a few seconds to observe both ingress classes
		expectedErr := "2 default IngressClasses were found, only 1 allowed"
		var lastErr error
		if err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
			defer cs.NetworkingV1().Ingresses(f.Namespace.Name).Delete(context.TODO(), "ingress1", metav1.DeleteOptions{})
			_, err := createBasicIngress(cs, f.Namespace.Name)
			if err == nil {
				return false, nil
			}
			lastErr = err
			return strings.Contains(err.Error(), expectedErr), nil
		}); err != nil {
			framework.Failf("Expected error to contain %s, got %s", expectedErr, lastErr.Error())
		}
	})

})

func createIngressClass(cs clientset.Interface, name string, isDefault bool, uniqueName string) (*networkingv1.IngressClass, error) {
	ingressClass := &networkingv1.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"ingressclass":  uniqueName,
				"special-label": "generic",
			},
		},
		Spec: networkingv1.IngressClassSpec{
			Controller: "example.com/controller",
		},
	}

	if isDefault {
		ingressClass.Annotations = map[string]string{networkingv1beta1.AnnotationIsDefaultIngressClass: "true"}
	}

	return cs.NetworkingV1().IngressClasses().Create(context.TODO(), ingressClass, metav1.CreateOptions{})
}

func createBasicIngress(cs clientset.Interface, namespace string) (*networkingv1.Ingress, error) {
	return cs.NetworkingV1().Ingresses(namespace).Create(context.TODO(), &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name: "ingress1",
		},
		Spec: networkingv1.IngressSpec{
			DefaultBackend: &networkingv1.IngressBackend{
				Service: &networkingv1.IngressServiceBackend{
					Name: "defaultbackend",
					Port: networkingv1.ServiceBackendPort{
						Number: 80,
					},
				},
			},
		},
	}, metav1.CreateOptions{})
}

func deleteIngressClass(cs clientset.Interface, name string) {
	err := cs.NetworkingV1().IngressClasses().Delete(context.TODO(), name, metav1.DeleteOptions{})
	framework.ExpectNoError(err)
}

var _ = SIGDescribe("IngressClass API", func() {
	f := framework.NewDefaultFramework("ingressclass")
	var cs clientset.Interface
	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})
	/*
		Release: v1.19
		Testname: IngressClass API
		Description:
		- The networking.k8s.io API group MUST exist in the /apis discovery document.
		- The networking.k8s.io/v1 API group/version MUST exist in the /apis/networking.k8s.io discovery document.
		- The ingressclasses resource MUST exist in the /apis/networking.k8s.io/v1 discovery document.
		- The ingressclass resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt(" should support creating IngressClass API operations", func() {

		// Setup
		icClient := f.ClientSet.NetworkingV1().IngressClasses()
		icVersion := "v1"

		// Discovery
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == networkingv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == icVersion {
							found = true
							break
						}
					}
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected networking API group/version, got %#v", discoveryGroups.Groups))
		}
		ginkgo.By("getting /apis/networking.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/networking.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == icVersion {
					found = true
					break
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected networking API version, got %#v", group.Versions))
		}

		ginkgo.By("getting /apis/networking.k8s.io" + icVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(networkingv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundIC := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "ingressclasses":
					foundIC = true
				}
			}
			framework.ExpectEqual(foundIC, true, fmt.Sprintf("expected ingressclasses, got %#v", resources.APIResources))
		}

		// IngressClass resource create/read/update/watch verbs
		ginkgo.By("creating")
		ingressClass1, err := createIngressClass(cs, "ingressclass1", false, f.UniqueName)
		framework.ExpectNoError(err)
		_, err = createIngressClass(cs, "ingressclass2", false, f.UniqueName)
		framework.ExpectNoError(err)
		_, err = createIngressClass(cs, "ingressclass3", false, f.UniqueName)
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenIC, err := icClient.Get(context.TODO(), ingressClass1.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gottenIC.UID, ingressClass1.UID)
		framework.ExpectEqual(gottenIC.UID, ingressClass1.UID)

		ginkgo.By("listing")
		ics, err := icClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "special-label=generic"})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(ics.Items), 3, "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		icWatch, err := icClient.Watch(context.TODO(), metav1.ListOptions{ResourceVersion: ics.ResourceVersion, LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedIC, err := icClient.Patch(context.TODO(), ingressClass1.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedIC.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		icToUpdate := patchedIC.DeepCopy()
		icToUpdate.Annotations["updated"] = "true"
		updatedIC, err := icClient.Update(context.TODO(), icToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedIC.Annotations["updated"], "true", "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-icWatch.ResultChan():
				framework.ExpectEqual(ok, true, "watch channel should not close")
				framework.ExpectEqual(evt.Type, watch.Modified)
				watchedIngress, isIngress := evt.Object.(*networkingv1.IngressClass)
				framework.ExpectEqual(isIngress, true, fmt.Sprintf("expected Ingress, got %T", evt.Object))
				if watchedIngress.Annotations["patched"] == "true" {
					framework.Logf("saw patched and updated annotations")
					sawAnnotations = true
					icWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", watchedIngress.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		// IngressClass resource delete operations
		ginkgo.By("deleting")
		err = icClient.Delete(context.TODO(), ingressClass1.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = icClient.Get(context.TODO(), ingressClass1.Name, metav1.GetOptions{})
		framework.ExpectEqual(apierrors.IsNotFound(err), true, fmt.Sprintf("expected 404, got %#v", err))
		ics, err = icClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(ics.Items), 2, "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = icClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		ics, err = icClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(ics.Items), 0, "filtered list should have 0 items")
	})

})
