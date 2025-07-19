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
	"time"

	networkingv1 "k8s.io/api/networking/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/common"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = common.SIGDescribe("IngressClass", feature.Ingress, func() {
	f := framework.NewDefaultFramework("ingressclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var cs clientset.Interface
	ginkgo.BeforeEach(func() {
		cs = f.ClientSet
	})

	f.It("should set default value on new IngressClass", f.WithSerial(), func(ctx context.Context) {
		ingressClass1, err := createIngressClass(ctx, cs, "ingressclass1", true, f.UniqueName)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(deleteIngressClass, cs, ingressClass1.Name)

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		lastFailure := ""

		// the admission controller may take a few seconds to observe the ingress classes
		if err := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
			lastFailure = ""

			ingress, err := createBasicIngress(ctx, cs, f.Namespace.Name)
			if err != nil {
				lastFailure = err.Error()
				return false, err
			}
			defer func() {
				err := cs.NetworkingV1().Ingresses(ingress.Namespace).Delete(ctx, ingress.Name, metav1.DeleteOptions{})
				framework.Logf("%v", err)
			}()

			if ingress.Spec.IngressClassName == nil {
				lastFailure = "Expected IngressClassName to be set by Admission Controller"
				return false, nil
			} else if *ingress.Spec.IngressClassName != ingressClass1.Name {
				lastFailure = fmt.Sprintf("Expected IngressClassName to be %s, got %s", ingressClass1.Name, *ingress.Spec.IngressClassName)
				return false, nil
			}
			return true, nil

		}); err != nil {
			framework.Failf("%v, final err= %v", lastFailure, err)
		}
	})

	f.It("should not set default value if no default IngressClass", f.WithSerial(), func(ctx context.Context) {
		ingressClass1, err := createIngressClass(ctx, cs, "ingressclass1", false, f.UniqueName)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(deleteIngressClass, cs, ingressClass1.Name)

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()
		lastFailure := ""

		// the admission controller may take a few seconds to observe the ingress classes
		if err := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, false, func(ctx context.Context) (bool, error) {
			lastFailure = ""

			ingress, err := createBasicIngress(ctx, cs, f.Namespace.Name)
			if err != nil {
				lastFailure = err.Error()
				return false, err
			}
			defer func() {
				err := cs.NetworkingV1().Ingresses(ingress.Namespace).Delete(ctx, ingress.Name, metav1.DeleteOptions{})
				framework.Logf("%v", err)
			}()

			if ingress.Spec.IngressClassName != nil {
				lastFailure = fmt.Sprintf("Expected IngressClassName to be nil, got %s", *ingress.Spec.IngressClassName)
				return false, nil
			}
			return true, nil

		}); err != nil {
			framework.Failf("%v, final err= %v", lastFailure, err)
		}
	})

	f.It("should choose the one with the later CreationTimestamp, if equal the one with the lower name when two ingressClasses are marked as default", f.WithSerial(), func(ctx context.Context) {
		ingressClass1, err := createIngressClass(ctx, cs, "ingressclass1", true, f.UniqueName)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(deleteIngressClass, cs, ingressClass1.Name)

		ingressClass2, err := createIngressClass(ctx, cs, "ingressclass2", true, f.UniqueName)
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(deleteIngressClass, cs, ingressClass2.Name)

		expectedName := ingressClass1.Name
		if ingressClass2.CreationTimestamp.UnixNano() > ingressClass1.CreationTimestamp.UnixNano() {
			expectedName = ingressClass2.Name
		}

		ctx, cancel := context.WithCancel(ctx)
		defer cancel()

		// the admission controller may take a few seconds to observe both ingress classes
		if err := wait.Poll(time.Second, time.Minute, func() (bool, error) {
			classes, err := cs.NetworkingV1().IngressClasses().List(ctx, metav1.ListOptions{})
			if err != nil {
				return false, nil
			}
			cntDefault := 0
			for _, class := range classes.Items {
				if class.Annotations[networkingv1.AnnotationIsDefaultIngressClass] == "true" {
					cntDefault++
				}
			}
			if cntDefault < 2 {
				return false, nil
			}
			ingress, err := createBasicIngress(ctx, cs, f.Namespace.Name)
			if err != nil {
				return false, nil
			}
			if ingress.Spec.IngressClassName == nil {
				return false, fmt.Errorf("expected IngressClassName to be set by Admission Controller")
			}
			if *ingress.Spec.IngressClassName != expectedName {
				return false, fmt.Errorf("expected ingress class %s but created with %s", expectedName, *ingress.Spec.IngressClassName)
			}
			return true, nil
		}); err != nil {
			framework.Failf("Failed to create ingress when two ingressClasses are marked as default ,got error %v", err)
		}
	})

	f.It("should allow IngressClass to have Namespace-scoped parameters", f.WithSerial(), func(ctx context.Context) {
		ingressClass := &networkingv1.IngressClass{
			ObjectMeta: metav1.ObjectMeta{
				Name: "ingressclass1",
				Labels: map[string]string{
					"ingressclass":  f.UniqueName,
					"special-label": "generic",
				},
			},
			Spec: networkingv1.IngressClassSpec{
				Controller: "example.com/controller",
				Parameters: &networkingv1.IngressClassParametersReference{
					Scope:     ptr.To("Namespace"),
					Namespace: ptr.To("foo-ns"),
					Kind:      "fookind",
					Name:      "fooname",
					APIGroup:  ptr.To("example.com"),
				},
			},
		}
		createdIngressClass, err := cs.NetworkingV1().IngressClasses().Create(ctx, ingressClass, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		ginkgo.DeferCleanup(deleteIngressClass, cs, createdIngressClass.Name)

		if createdIngressClass.Spec.Parameters == nil {
			framework.Failf("Expected IngressClass.spec.parameters to be set")
		}
		scope := ""
		if createdIngressClass.Spec.Parameters.Scope != nil {
			scope = *createdIngressClass.Spec.Parameters.Scope
		}

		if scope != "Namespace" {
			framework.Failf("Expected IngressClass.spec.parameters.scope to be set to 'Namespace', got %v", scope)
		}
	})

})

func createIngressClass(ctx context.Context, cs clientset.Interface, name string, isDefault bool, uniqueName string) (*networkingv1.IngressClass, error) {
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
		ingressClass.Annotations = map[string]string{networkingv1.AnnotationIsDefaultIngressClass: "true"}
	}

	return cs.NetworkingV1().IngressClasses().Create(ctx, ingressClass, metav1.CreateOptions{})
}

func createBasicIngress(ctx context.Context, cs clientset.Interface, namespace string) (*networkingv1.Ingress, error) {
	return cs.NetworkingV1().Ingresses(namespace).Create(ctx, &networkingv1.Ingress{
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

func deleteIngressClass(ctx context.Context, cs clientset.Interface, name string) {
	err := cs.NetworkingV1().IngressClasses().Delete(ctx, name, metav1.DeleteOptions{})
	framework.ExpectNoError(err)
}

var _ = common.SIGDescribe("IngressClass API", func() {
	f := framework.NewDefaultFramework("ingressclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
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
	framework.ConformanceIt("should support creating IngressClass API operations", func(ctx context.Context) {

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
				if version.Version == icVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected networking API version, got %#v", group.Versions)
			}
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
			if !foundIC {
				framework.Failf("expected ingressclasses, got %#v", resources.APIResources)
			}
		}

		// IngressClass resource create/read/update/watch verbs
		ginkgo.By("creating")
		ingressClass1, err := createIngressClass(ctx, cs, "ingressclass1", false, f.UniqueName)
		framework.ExpectNoError(err)
		_, err = createIngressClass(ctx, cs, "ingressclass2", false, f.UniqueName)
		framework.ExpectNoError(err)
		_, err = createIngressClass(ctx, cs, "ingressclass3", false, f.UniqueName)
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenIC, err := icClient.Get(ctx, ingressClass1.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenIC.UID).To(gomega.Equal(ingressClass1.UID))
		gomega.Expect(gottenIC.UID).To(gomega.Equal(ingressClass1.UID))

		ginkgo.By("listing")
		ics, err := icClient.List(ctx, metav1.ListOptions{LabelSelector: "special-label=generic"})
		framework.ExpectNoError(err)
		gomega.Expect(ics.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		icWatch, err := icClient.Watch(ctx, metav1.ListOptions{ResourceVersion: ics.ResourceVersion, LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchedIC, err := icClient.Patch(ctx, ingressClass1.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedIC.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		icToUpdate := patchedIC.DeepCopy()
		icToUpdate.Annotations["updated"] = "true"
		updatedIC, err := icClient.Update(ctx, icToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedIC.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotations := false; !sawAnnotations; {
			select {
			case evt, ok := <-icWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				watchedIngress, isIngress := evt.Object.(*networkingv1.IngressClass)
				if !isIngress {
					framework.Failf("expected Ingress, got %T", evt.Object)
				}
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
		err = icClient.Delete(ctx, ingressClass1.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = icClient.Get(ctx, ingressClass1.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}
		ics, err = icClient.List(ctx, metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(ics.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = icClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		ics, err = icClient.List(ctx, metav1.ListOptions{LabelSelector: "ingressclass=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(ics.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

})
