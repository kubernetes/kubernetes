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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("MutatingAdmissionPolicy [Privileged:ClusterAdmin]", func() {
	f := framework.NewDefaultFramework("mutating-admission-policy")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var client clientset.Interface

	ginkgo.BeforeEach(func() {
		var err error
		client, err = clientset.NewForConfig(f.ClientConfig())
		framework.ExpectNoError(err, "initializing client")
	})

	ginkgo.BeforeEach(func(ctx context.Context) {
		// Make sure the namespace created for the test is labeled to be selected
		// in binding.spec.matchResources.namespaceSelector.matchLabels
		// By containing the tests within the marked namespace, they will not
		// disturb concurrent tests that run in other namespaces.
		labelNamespace(ctx, f, f.Namespace.Name)
	})

	/*
	   Release: v1.36
	   Testname: MutatingAdmissionPolicy
	   Description:
	   The MutatingAdmissionPolicy should mutate a deployment by adding annotations as defined in the policy.
	*/
	framework.ConformanceIt("should mutate a Deployment with annotations", func(ctx context.Context) {
		ginkgo.By("creating the policy", func() {
			policy := newMutatingAdmissionPolicyBuilder(f.UniqueName+".policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithMutation(admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `
							Object{
								metadata: Object.metadata{
									annotations: {
										"mutated-by": "admission-policy"
									}
								}
							}`,
					},
				}).
				Build()
			policy, err := client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1().MutatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
			binding := createMutatingBinding(f.UniqueName+".binding.example.com", f.UniqueName, policy.Name)
			binding, err = client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy binding")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Delete(ctx, name, metav1.DeleteOptions{})
			}, binding.Name)
		})
		ginkgo.By("waiting until the marker is mutated", func() {
			deployment := mutatingBasicDeployment("marker-deployment", 1)
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				created, err := client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
				if err != nil {
					return false, err
				}
				defer func() {
					if err := client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
						framework.Logf("Error deleting deployment %s: %v", deployment.Name, err)
					}
				}()

				if created.ObjectMeta.Annotations["mutated-by"] == "admission-policy" {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for marker")
		})
	})

	/*
	   Release: v1.36
	   Testname: MutatingAdmissionPolicy
	   Description:
	   The MutatingAdmissionPolicy should mutate a deployment as the expression defined inside the policy.
	*/
	framework.ConformanceIt("should mutate a Deployment", func(ctx context.Context) {
		ginkgo.By("creating the policy", func() {
			policy := newMutatingAdmissionPolicyBuilder(f.UniqueName+".policy.example.com").
				MatchUniqueNamespace(f.UniqueName).
				StartResourceRule().
				MatchResource([]string{"apps"}, []string{"v1"}, []string{"deployments"}).
				EndResourceRule().
				WithMutation(admissionregistrationv1.Mutation{
					PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
					ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
						Expression: `Object{spec: Object.spec{replicas: 1337}}`,
					},
				}).
				Build()
			policy, err := client.AdmissionregistrationV1().MutatingAdmissionPolicies().Create(ctx, policy, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1().MutatingAdmissionPolicies().Delete(ctx, name, metav1.DeleteOptions{})
			}, policy.Name)
			binding := createMutatingBinding(f.UniqueName+".binding.example.com", f.UniqueName, policy.Name)
			binding, err = client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Create(ctx, binding, metav1.CreateOptions{})
			framework.ExpectNoError(err, "create policy binding")
			ginkgo.DeferCleanup(func(ctx context.Context, name string) error {
				return client.AdmissionregistrationV1().MutatingAdmissionPolicyBindings().Delete(ctx, name, metav1.DeleteOptions{})
			}, binding.Name)
		})
		ginkgo.By("waiting until the marker is mutated", func() {
			deployment := mutatingBasicDeployment("marker-deployment", 1)
			err := wait.PollUntilContextCancel(ctx, 100*time.Millisecond, true, func(ctx context.Context) (done bool, err error) {
				created, err := client.AppsV1().Deployments(f.Namespace.Name).Create(ctx, deployment, metav1.CreateOptions{})
				if err != nil {
					return false, err
				}
				defer func() {
					if err := client.AppsV1().Deployments(f.Namespace.Name).Delete(ctx, deployment.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
						framework.Logf("Error deleting deployment %s: %v", deployment.Name, err)
					}
				}()

				if created.Spec.Replicas != nil && *created.Spec.Replicas == 1337 {
					return true, nil
				}
				return false, nil
			})
			framework.ExpectNoError(err, "wait for marker")
		})
	})

	/*
	   Release: v1.36
	   Testname: MutatingAdmissionPolicy API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The mutatingadmissionpolicies resource MUST exist in the
	     /apis/admissionregistration.k8s.io/v1 discovery document.
	   The mutatingadmissionpolicies resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support MutatingAdmissionPolicy API operations", func(ctx context.Context) {
		mapVersion := "v1"
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
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundMAP := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicies":
					foundMAP = true
				}
			}
			if !foundMAP {
				framework.Failf("expected mutatingadmissionpolicies, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1().MutatingAdmissionPolicies()
		labelKey, labelValue := "example-e2e-map-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-map-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
				ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
				Mutations: []admissionregistrationv1.Mutation{
					{
						PatchType: admissionregistrationv1.PatchTypeApplyConfiguration,
						ApplyConfiguration: &admissionregistrationv1.ApplyConfiguration{
							Expression: `Object{spec: Object.spec{replicas: 100}}`,
						},
					},
				},
				MatchConstraints: &admissionregistrationv1.MatchResources{
					ResourceRules: []admissionregistrationv1.NamedRuleWithOperations{
						{
							RuleWithOperations: admissionregistrationv1.RuleWithOperations{
								Operations: []admissionregistrationv1.OperationType{"CREATE"},
								Rule: admissionregistrationv1.Rule{
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
		gomega.Expect(mapRead).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"reinvocationPolicy":"IfNeeded"}}`)
		mapPatched, err := client.Patch(ctx, mapCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(mapPatched.Spec.ReinvocationPolicy).To(gomega.Equal(admissionregistrationv1.IfNeededReinvocationPolicy), "patched object should have the applied spec")
		gomega.Expect(resourceversion.CompareResourceVersion(mapRead.ResourceVersion, mapPatched.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var mapUpdated *admissionregistrationv1.MutatingAdmissionPolicy
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mp, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapToUpdate := mp.DeepCopy()
			mapToUpdate.Annotations["updated"] = "true"
			mapToUpdate.Spec.ReinvocationPolicy = admissionregistrationv1.NeverReinvocationPolicy

			mapUpdated, err = client.Update(ctx, mapToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicy %q", mapCreated.Name)
		gomega.Expect(mapUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(mapUpdated.Spec.ReinvocationPolicy).To(gomega.Equal(admissionregistrationv1.NeverReinvocationPolicy), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				mapWatched, isMAP := evt.Object.(*admissionregistrationv1.MutatingAdmissionPolicy)
				if !isMAP {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1.MutatingAdmissionPolicy{}, evt.Object)
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
		mapTmp, err := client.Get(ctx, mapCreated.Name, metav1.GetOptions{})
		switch {
		case err == nil && mapTmp.GetDeletionTimestamp() != nil && len(mapTmp.GetFinalizers()) > 0:
			// deletion requested successfully, object is blocked by finalizers
		case err == nil:
			framework.Failf("expected deleted object, got %#v", mapTmp)
		case apierrors.IsNotFound(err):
			// deleted successfully
		default:
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		var itemsWithoutFinalizer []admissionregistrationv1.MutatingAdmissionPolicy
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
		var itemsColWithoutFinalizer []admissionregistrationv1.MutatingAdmissionPolicy
		for _, item := range list.Items {
			if !(item.GetDeletionTimestamp() != nil && len(item.GetFinalizers()) > 0) {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
	   Release: v1.36
	   Testname: MutatingAdmissionPolicyBinding API
	   Description:
	   The admissionregistration.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The admissionregistration.k8s.io/v1 API group/version MUST exist
	     in the /apis/admissionregistration.k8s.io discovery document.
	   The MutatingAdmissionPolicyBinding resources MUST exist in the
	     /apis/admissionregistration.k8s.io/v1 discovery document.
	   The MutatingAdmissionPolicyBinding resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support MutatingAdmissionPolicyBinding API operations", func(ctx context.Context) {
		mapbVersion := "v1"
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
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(admissionregistrationv1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundMAPB := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "mutatingadmissionpolicybindings":
					foundMAPB = true
				}
			}
			if !foundMAPB {
				framework.Failf("expected mutatingadmissionpolicybindings, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.AdmissionregistrationV1().MutatingAdmissionPolicyBindings()
		labelKey, labelValue := "example-e2e-mapb-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &admissionregistrationv1.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-mapb-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
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
		gomega.Expect(mapbRead).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		mapbWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"policyName":"new-policy.example.com"}}`)
		mapbPatched, err := client.Patch(ctx, mapbCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(mapbPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(mapbPatched.Spec.PolicyName).To(gomega.Equal("new-policy.example.com"), "patched object should have the applied spec")
		gomega.Expect(resourceversion.CompareResourceVersion(mapbRead.ResourceVersion, mapbPatched.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var mapbUpdated *admissionregistrationv1.MutatingAdmissionPolicyBinding
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			mp, err := client.Get(ctx, mapbCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			mapbToUpdate := mp.DeepCopy()
			mapbToUpdate.Annotations["updated"] = "true"
			mapbToUpdate.Spec.PolicyName = "replicalimit-policy.example.com"

			mapbUpdated, err = client.Update(ctx, mapbToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update mutatingadmissionpolicybinding %q", mapbCreated.Name)
		gomega.Expect(mapbUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(mapbUpdated.Spec.PolicyName).To(gomega.Equal("replicalimit-policy.example.com"), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-mapbWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				mapbWatched, isMAPB := evt.Object.(*admissionregistrationv1.MutatingAdmissionPolicyBinding)
				if !isMAPB {
					framework.Failf("expected an object of type: %T, but got %T", &admissionregistrationv1.MutatingAdmissionPolicyBinding{}, evt.Object)
				}
				if mapbWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					mapbWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", mapbWatched.Annotations)
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
		var itemsWithoutFinalizer []admissionregistrationv1.MutatingAdmissionPolicyBinding
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
		var itemsColWithoutFinalizer []admissionregistrationv1.MutatingAdmissionPolicyBinding
		for _, item := range list.Items {
			if !(item.GetDeletionTimestamp() != nil && len(item.GetFinalizers()) > 0) {
				itemsColWithoutFinalizer = append(itemsColWithoutFinalizer, item)
			}
		}
		framework.ExpectNoError(err)
		gomega.Expect(itemsColWithoutFinalizer).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})
})

func createMutatingBinding(bindingName string, uniqueLabel string, policyName string) *admissionregistrationv1.MutatingAdmissionPolicyBinding {
	return &admissionregistrationv1.MutatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: bindingName},
		Spec: admissionregistrationv1.MutatingAdmissionPolicyBindingSpec{
			PolicyName: policyName,
			MatchResources: &admissionregistrationv1.MatchResources{
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{uniqueLabel: "true"},
				},
			},
		},
	}
}

func mutatingBasicDeployment(name string, replicas int32) *appsv1.Deployment {
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{"app": "nginx"},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "nginx"},
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "nginx"},
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
		}}
}

type mutatingAdmissionPolicyBuilder struct {
	policy *admissionregistrationv1.MutatingAdmissionPolicy
}

type mutatingResourceRuleBuilder struct {
	policyBuilder *mutatingAdmissionPolicyBuilder
	resourceRule  *admissionregistrationv1.NamedRuleWithOperations
}

func newMutatingAdmissionPolicyBuilder(policyName string) *mutatingAdmissionPolicyBuilder {
	return &mutatingAdmissionPolicyBuilder{
		policy: &admissionregistrationv1.MutatingAdmissionPolicy{
			ObjectMeta: metav1.ObjectMeta{Name: policyName},
			Spec: admissionregistrationv1.MutatingAdmissionPolicySpec{
				ReinvocationPolicy: admissionregistrationv1.NeverReinvocationPolicy,
			},
		},
	}
}

func (b *mutatingAdmissionPolicyBuilder) MatchUniqueNamespace(uniqueLabel string) *mutatingAdmissionPolicyBuilder {
	if b.policy.Spec.MatchConstraints == nil {
		b.policy.Spec.MatchConstraints = &admissionregistrationv1.MatchResources{}
	}
	b.policy.Spec.MatchConstraints.NamespaceSelector = &metav1.LabelSelector{
		MatchLabels: map[string]string{
			uniqueLabel: "true",
		},
	}
	return b
}

func (b *mutatingAdmissionPolicyBuilder) StartResourceRule() *mutatingResourceRuleBuilder {
	return &mutatingResourceRuleBuilder{
		policyBuilder: b,
		resourceRule: &admissionregistrationv1.NamedRuleWithOperations{
			RuleWithOperations: admissionregistrationv1.RuleWithOperations{
				Operations: []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update},
				Rule: admissionregistrationv1.Rule{
					APIGroups:   []string{"apps"},
					APIVersions: []string{"v1"},
					Resources:   []string{"deployments"},
				},
			},
		},
	}
}

func (rb *mutatingResourceRuleBuilder) CreateAndUpdate() *mutatingResourceRuleBuilder {
	rb.resourceRule.Operations = []admissionregistrationv1.OperationType{admissionregistrationv1.Create, admissionregistrationv1.Update}
	return rb
}

func (rb *mutatingResourceRuleBuilder) MatchResource(groups []string, versions []string, resources []string) *mutatingResourceRuleBuilder {
	rb.resourceRule.Rule = admissionregistrationv1.Rule{
		APIGroups:   groups,
		APIVersions: versions,
		Resources:   resources,
	}
	return rb
}

func (rb *mutatingResourceRuleBuilder) EndResourceRule() *mutatingAdmissionPolicyBuilder {
	b := rb.policyBuilder
	if b.policy.Spec.MatchConstraints == nil {
		b.policy.Spec.MatchConstraints = &admissionregistrationv1.MatchResources{}
	}
	b.policy.Spec.MatchConstraints.ResourceRules = append(b.policy.Spec.MatchConstraints.ResourceRules, *rb.resourceRule)
	return b
}

func (b *mutatingAdmissionPolicyBuilder) WithMutation(mutation admissionregistrationv1.Mutation) *mutatingAdmissionPolicyBuilder {
	b.policy.Spec.Mutations = append(b.policy.Spec.Mutations, mutation)
	return b
}

func (b *mutatingAdmissionPolicyBuilder) WithVariable(variable admissionregistrationv1.Variable) *mutatingAdmissionPolicyBuilder {
	b.policy.Spec.Variables = append(b.policy.Spec.Variables, variable)
	return b
}

func (b *mutatingAdmissionPolicyBuilder) Build() *admissionregistrationv1.MutatingAdmissionPolicy {
	return b.policy
}
