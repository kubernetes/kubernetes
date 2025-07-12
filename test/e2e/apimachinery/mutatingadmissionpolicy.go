/*
Copyright 2025 The Kubernetes Authors.

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
	admissionregistrationv1alpha1 "k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/utils/ptr"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	admissionregistrationv1beta1 "k8s.io/api/admissionregistration/v1beta1"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("MutatingAdmissionPolicy [Privileged:ClusterAdmin]", feature.MutatingAdmissionPolicy, framework.WithFeatureGate(features.MutatingAdmissionPolicy), func() {
	f := framework.NewDefaultFramework("mutating-admission-policy")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var client clientset.Interface
	ginkgo.BeforeEach(func() {
		var err error
		client, err = clientset.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err, "initializing client")
		framework.ExpectNoError(err, "initializing api-extensions client")
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Make sure the namespace created for the test is labeled to be selected
		// in binding.spec.matchResources.namespaceSelector.matchLabels
		// By containing the tests within the marked namespace, they will not
		// disturb concurrent tests that run in other namespaces.
		labelNamespace(ctx, f, f.Namespace.Name)
	})

	/*
	   Release: v1.33
	   Testname: MutatingAdmissionPolicy
	   Description:
	   The MutatingAdmissionPolicy should mutate a deployment by adding annotations as defined in the policy.
	*/
	framework.It("should mutate a Deployment", func(ctx context.Context) {
		ginkgo.By("creating the policy", func() {
			policy := &admissionregistrationv1beta1.MutatingAdmissionPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name: f.UniqueName + ".policy.example.com",
				},
				Spec: admissionregistrationv1beta1.MutatingAdmissionPolicySpec{
					MatchConstraints: &admissionregistrationv1beta1.MatchResources{
						ResourceRules: []admissionregistrationv1beta1.NamedRuleWithOperations{
							{
								RuleWithOperations: admissionregistrationv1beta1.RuleWithOperations{
									Operations: []admissionregistrationv1beta1.OperationType{"CREATE", "UPDATE"},
									Rule: admissionregistrationv1beta1.Rule{
										APIGroups:   []string{"apps"},
										APIVersions: []string{"v1"},
										Resources:   []string{"deployments"},
									},
								},
							},
						},
						NamespaceSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								f.UniqueName: "true",
							},
						},
					},
					Mutations: []admissionregistrationv1beta1.Mutation{
						{
							PatchType: admissionregistrationv1beta1.PatchTypeApplyConfiguration,
							ApplyConfiguration: &admissionregistrationv1beta1.ApplyConfiguration{
								Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"mutated-by": "admission-policy"
									}
								}
							}`,
							},
						},
					},
					ReinvocationPolicy: admissionregistrationv1beta1.NeverReinvocationPolicy,
				},
			}

			policy, err := client.AdmissionregistrationV1beta1().MutatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().MutatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)

			binding := createMAPBinding(f.UniqueName+".binding.example.com", f.UniqueName, policy.Name)
			binding, err = client.AdmissionregistrationV1beta1().MutatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create binding")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1beta1().MutatingAdmissionPolicyBindings().Delete(ctx, name, metav1.DeleteOptions{})
			}, binding.Name)

			// Test the mutation
			deployment := &appsv1.Deployment{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-deployment",
					Namespace: f.Namespace.Name,
				},
				Spec: appsv1.DeploymentSpec{
					Replicas: ptr.To[int32](2),
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"app": "test",
						},
					},
					Template: v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"app": "test",
							},
						},
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{
									Name:  "nginx",
									Image: "nginx:latest",
								},
							},
						},
					},
				},
			}

			// wait for MAP is in action
			time.Sleep(30 * time.Second)
			deployment, err = client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create deployment")

			gomega.Expect(deployment.ObjectMeta.Annotations).To(gomega.HaveKeyWithValue("mutated-by", "admission-policy"))
		})
	})

	/*
	   Release: v1.33
	   Testname: MutatingAdmissionPolicy API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1alpha1 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The MutatingAdmissionPolicy MUST exist in the
	     /apis/admissionregistration.k8s.io/v1alpha1 discovery document.
	   The mutatingadmisionpolicy resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.It("should support MutatingAdmissionPolicy API operations", func(ctx context.Context) {
		mapVersion := "v1alpha1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == admissionregistrationv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == mapVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicy API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == mapVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicy API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io/" + mapVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1alpha1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundVAP := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicies":
					foundVAP = true
				}
			}
			if !foundVAP {
				framework.Failf("expected mutatingadmissionpolicies, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1alpha1().MutatingAdmissionPolicies()
		labelKey, labelValue := "example-e2e-map-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1alpha1.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-map-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1alpha1.MutatingAdmissionPolicySpec{
				ReinvocationPolicy: admissionregistrationv1alpha1.NeverReinvocationPolicy,
				Mutations: []admissionregistrationv1alpha1.Mutation{
					{
						PatchType: admissionregistrationv1alpha1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1alpha1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"my-foo-annotation": "myAnnotationValue"
									}
								}
							}`,
						},
					},
				},
				MatchConstraints: &admissionregistrationv1alpha1.MatchResources{
					ResourceRules: []admissionregistrationv1alpha1.NamedRuleWithOperations{
						{
							RuleWithOperations: admissionregistrationv1alpha1.RuleWithOperations{
								Operations: []admissionregistrationv1alpha1.OperationType{"CREATE"},
								Rule: admissionregistrationv1alpha1.Rule{
									APIGroups:   []string{"apps"},
									APIVersions: []string{"v1"},
									Resources:   []string{"deployments"},
								},
							},
						},
					},
				},
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		mapCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		mapRead, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapRead.UID).To(gomega.Equal(mapCreated.UID))

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"failurePolicy":"Ignore"}}`)
		mapPatched, err := client.Patch(ctx, mapCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(mapPatched.Spec.FailurePolicy).To(gomega.HaveValue(gomega.Equal(admissionregistrationv1alpha1.Ignore)), "patched object should have the applied spec")

		ginkgo.By("updating")
		var mapUpdated *admissionregistrationv1alpha1.MutatingAdmissionPolicy
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mpolicy, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapToUpdate := mpolicy.DeepCopy()
			mapToUpdate.Annotations["updated"] = "true"
			fail := admissionregistrationv1alpha1.Fail
			mapToUpdate.Spec.FailurePolicy = &fail

			mapUpdated, err = client.Update(ctx, mapToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicy %q", mapCreated.Name)
		gomega.Expect(mapUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(mapUpdated.Spec.FailurePolicy).To(gomega.HaveValue(gomega.Equal(admissionregistrationv1alpha1.Fail)), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				mapWatched, isFS := evt.Object.(*admissionregistrationv1alpha1.MutatingAdmissionPolicy)
				if !isFS {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1alpha1.MutatingAdmissionPolicy{}, evt.Object)
				}
				if mapWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					mapWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", mapWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("deleting")
		err = client.Delete(ctx, mapCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		vapTmp, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
		switch {
		case err == nil && vapTmp.GetDeletionTimestamp() != nil && len(vapTmp.GetFinalizers()) > 0:
			// deletion requested successfully, object is blocked by finalizers
		case err == nil:
			framework.Failf("expected deleted object, got %#v", vapTmp)
		case apierrors.IsNotFound(err):
			// deleted successfully
		default:
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsWithoutFinalizer []admissionregistrationv1alpha1.MutatingAdmissionPolicy
		for _, item := range list.Items {
			if len(item.GetFinalizers()) == 0 {
				itemsWithoutFinalizer = append(itemsWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsWithoutFinalizer).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsColWithoutFinalizer []admissionregistrationv1alpha1.MutatingAdmissionPolicy
		for _, item := range list.Items {
			if !(item.GetDeletionTimestamp() != nil && len(item.GetFinalizers()) > 0) {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
	   Release: v1.33
	   Testname: MutatingadmissionPolicyBinding API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The MutatingadmissionPolicyBinding resources MUST exist in the
	     /apis/admissionregistration.k8s.io/v1 discovery document.
	   The MutatingadmissionPolicyBinding resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.It("should support MutatingAdmissionPolicyBinding API operations", func(ctx context.Context) {
		mapbVersion := "v1alpha1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == admissionregistrationv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == mapbVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicyBinding API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == mapbVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicyBinding API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io/" + mapbVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1alpha1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundVAPB := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicybindings":
					foundVAPB = true
				}
			}
			if !foundVAPB {
				framework.Failf("expected mutatingadmissionpolicybindings, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1alpha1().MutatingAdmissionPolicyBindings()
		labelKey, labelValue := "example-e2e-mapb-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1alpha1.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-mapb-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1alpha1.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		mapbCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		mapbRead, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapbRead.UID).To(gomega.Equal(mapbCreated.UID))

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapbWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}}}`)
		mapbPatched, err := client.Patch(ctx, mapbCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapbPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		var mapbUpdated *admissionregistrationv1alpha1.MutatingAdmissionPolicyBinding
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mapb, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapbToUpdate := mapb.DeepCopy()
			mapbToUpdate.Annotations["updated"] = "true"

			mapbUpdated, err = client.Update(ctx, mapbToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicybinding %q", mapbCreated.Name)
		gomega.Expect(mapbUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapbWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				vapbWatched, isFS := evt.Object.(*admissionregistrationv1alpha1.MutatingAdmissionPolicyBinding)
				if !isFS {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1beta1.MutatingAdmissionPolicyBinding{}, evt.Object)
				}
				if vapbWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					mapbWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", vapbWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}
		ginkgo.By("deleting")
		err = client.Delete(ctx, mapbCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		mapbTmp, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
		switch {
		case err == nil && mapbTmp.GetDeletionTimestamp() != nil && len(mapbTmp.GetFinalizers()) > 0:
			// deletion requested successfully, object is blocked by finalizers
		case err == nil:
			framework.Failf("expected deleted object, got %#v", mapbTmp)
		case apierrors.IsNotFound(err):
			// deleted successfully
		default:
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsWithoutFinalizer []admissionregistrationv1alpha1.MutatingAdmissionPolicyBinding
		for _, item := range list.Items {
			if len(item.GetFinalizers()) == 0 {
				itemsWithoutFinalizer = append(itemsWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsWithoutFinalizer).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsColWithoutFinalizer []admissionregistrationv1alpha1.MutatingAdmissionPolicyBinding
		for _, item := range list.Items {
			if !(item.GetDeletionTimestamp() != nil && len(item.GetFinalizers()) > 0) {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
	   Release: v1.34
	   Testname: MutatingAdmissionPolicy API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1beta11 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The MutatingAdmissionPolicy MUST exist in the
	     /apis/admissionregistration.k8s.io/v1beta1 discovery document.
	   The mutatingadmisionpolicy resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.It("should support MutatingAdmissionPolicy API operations", func(ctx context.Context) {
		mapVersion := "v1beta1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == admissionregistrationv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == mapVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicy API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == mapVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicy API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io/" + mapVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1beta1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundVAP := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicies":
					foundVAP = true
				}
			}
			if !foundVAP {
				framework.Failf("expected mutatingadmissionpolicies, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1beta1().MutatingAdmissionPolicies()
		labelKey, labelValue := "example-e2e-map-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1beta1.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-map-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1beta1.MutatingAdmissionPolicySpec{
				ReinvocationPolicy: admissionregistrationv1beta1.NeverReinvocationPolicy,
				Mutations: []admissionregistrationv1beta1.Mutation{
					{
						PatchType: admissionregistrationv1beta1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1beta1.ApplyConfiguration{
							Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"my-foo-annotation": "myAnnotationValue"
									}
								}
							}`,
						},
					},
				},
				MatchConstraints: &admissionregistrationv1beta1.MatchResources{
					ResourceRules: []admissionregistrationv1beta1.NamedRuleWithOperations{
						{
							RuleWithOperations: admissionregistrationv1beta1.RuleWithOperations{
								Operations: []admissionregistrationv1beta1.OperationType{"CREATE"},
								Rule: admissionregistrationv1beta1.Rule{
									APIGroups:   []string{"apps"},
									APIVersions: []string{"v1"},
									Resources:   []string{"deployments"},
								},
							},
						},
					},
				},
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		mapCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		mapRead, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapRead.UID).To(gomega.Equal(mapCreated.UID))

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"failurePolicy":"Ignore"}}`)
		mapPatched, err := client.Patch(ctx, mapCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(mapPatched.Spec.FailurePolicy).To(gomega.HaveValue(gomega.Equal(admissionregistrationv1beta1.Ignore)), "patched object should have the applied spec")

		ginkgo.By("updating")
		var mapUpdated *admissionregistrationv1beta1.MutatingAdmissionPolicy
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mpolicy, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapToUpdate := mpolicy.DeepCopy()
			mapToUpdate.Annotations["updated"] = "true"
			fail := admissionregistrationv1beta1.Fail
			mapToUpdate.Spec.FailurePolicy = &fail

			mapUpdated, err = client.Update(ctx, mapToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicy %q", mapCreated.Name)
		gomega.Expect(mapUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(mapUpdated.Spec.FailurePolicy).To(gomega.HaveValue(gomega.Equal(admissionregistrationv1beta1.Fail)), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				mapWatched, isFS := evt.Object.(*admissionregistrationv1beta1.MutatingAdmissionPolicy)
				if !isFS {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1beta1.MutatingAdmissionPolicy{}, evt.Object)
				}
				if mapWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					mapWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", mapWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("deleting")
		err = client.Delete(ctx, mapCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		vapTmp, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
		switch {
		case err == nil && vapTmp.GetDeletionTimestamp() != nil && len(vapTmp.GetFinalizers()) > 0:
			// deletion requested successfully, object is blocked by finalizers
		case err == nil:
			framework.Failf("expected deleted object, got %#v", vapTmp)
		case apierrors.IsNotFound(err):
			// deleted successfully
		default:
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsWithoutFinalizer []admissionregistrationv1beta1.MutatingAdmissionPolicy
		for _, item := range list.Items {
			if len(item.GetFinalizers()) == 0 {
				itemsWithoutFinalizer = append(itemsWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsWithoutFinalizer).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsColWithoutFinalizer []admissionregistrationv1beta1.MutatingAdmissionPolicy
		for _, item := range list.Items {
			if item.GetDeletionTimestamp() == nil || len(item.GetFinalizers()) == 0 {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
	   Release: v1.34
	   Testname: MutatingadmissionPolicyBinding API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1beta1 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The MutatingadmissionPolicyBinding resources MUST exist in the
	     /apis/admissionregistration.k8s.io/v1beta1 discovery document.
	   The MutatingadmissionPolicyBinding resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.It("should support MutatingAdmissionPolicyBinding API operations", func(ctx context.Context) {
		mapbVersion := "v1beta1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == admissionregistrationv1.GroupName {
					for _, version := range group.Versions {
						if version.Version == mapbVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicyBinding API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/admissionregistration.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == mapbVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected MutatingAdmissionPolicyBinding API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/admissionregistration.k8s.io/" + mapbVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1beta1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundVAPB := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicybindings":
					foundVAPB = true
				}
			}
			if !foundVAPB {
				framework.Failf("expected mutatingadmissionpolicybindings, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1beta1().MutatingAdmissionPolicyBindings()
		labelKey, labelValue := "example-e2e-mapb-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1beta1.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-mapb-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1beta1.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		mapbCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		mapbRead, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapbRead.UID).To(gomega.Equal(mapbCreated.UID))

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapbWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}}}`)
		mapbPatched, err := client.Patch(ctx, mapbCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapbPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("updating")
		var mapbUpdated *admissionregistrationv1beta1.MutatingAdmissionPolicyBinding
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mapb, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapbToUpdate := mapb.DeepCopy()
			mapbToUpdate.Annotations["updated"] = "true"

			mapbUpdated, err = client.Update(ctx, mapbToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicybinding %q", mapbCreated.Name)
		gomega.Expect(mapbUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapbWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				vapbWatched, isFS := evt.Object.(*admissionregistrationv1beta1.MutatingAdmissionPolicyBinding)
				if !isFS {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1beta1.MutatingAdmissionPolicyBinding{}, evt.Object)
				}
				if vapbWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					mapbWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", vapbWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}
		ginkgo.By("deleting")
		err = client.Delete(ctx, mapbCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		mapbTmp, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
		switch {
		case err == nil && mapbTmp.GetDeletionTimestamp() != nil && len(mapbTmp.GetFinalizers()) > 0:
			// deletion requested successfully, object is blocked by finalizers
		case err == nil:
			framework.Failf("expected deleted object, got %#v", mapbTmp)
		case apierrors.IsNotFound(err):
			// deleted successfully
		default:
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsWithoutFinalizer []admissionregistrationv1beta1.MutatingAdmissionPolicyBinding
		for _, item := range list.Items {
			if len(item.GetFinalizers()) == 0 {
				itemsWithoutFinalizer = append(itemsWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsWithoutFinalizer).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsColWithoutFinalizer []admissionregistrationv1beta1.MutatingAdmissionPolicyBinding
		for _, item := range list.Items {
			if item.GetDeletionTimestamp() == nil || len(item.GetFinalizers()) == 0 {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})
})

func createMAPBinding(bindingName string, uniqueLabel string, policyName string) *admissionregistrationv1beta1.MutatingAdmissionPolicyBinding {
	return &admissionregistrationv1beta1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: bindingName},
		Spec: admissionregistrationv1beta1.MutatingAdmissionPolicyBindingSpec{
			PolicyName: policyName,
			MatchResources: &admissionregistrationv1beta1.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{uniqueLabel: "true"},
				},
			},
		},
	}
}
