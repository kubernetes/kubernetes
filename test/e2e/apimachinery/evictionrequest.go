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

	v1 "k8s.io/api/core/v1"
	lifecycleapi "k8s.io/api/lifecycle/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("EvictionRequest API", func() {
	f := framework.NewDefaultFramework("evictionrequest")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
	   Release: v1.37
	   Testname: EvictionRequest API operations
	   Description:
	   The lifecycle.k8s.io API group MUST exist in the /apis discovery document.
	   The lifecycle.k8s.io/v1alpha1 API group/version MUST exist
	     in the /apis/lifecycle.k8s.io discovery document.
	   The evictionrequests and evictionrequests/status resources MUST exist
	     in the /apis/lifecycle.k8s.io/v1alpha1 discovery document.
	   The evictionrequests resource must support create, get, list, watch,
	     update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support EvictionRequest API operations", func(ctx context.Context) {
		erVersion := "v1alpha1"

		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == lifecycleapi.GroupName {
					for _, version := range group.Versions {
						if version.Version == erVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected lifecycle API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/lifecycle.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/lifecycle.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == erVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected lifecycle API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/lifecycle.k8s.io/" + erVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(lifecycleapi.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundER, foundERStatus := false, false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "evictionrequests":
					foundER = true
				case "evictionrequests/status":
					foundERStatus = true
				}
			}
			if !foundER {
				framework.Failf("expected evictionrequests, got %#v", resources.APIResources)
			}
			if !foundERStatus {
				framework.Failf("expected evictionrequests/status, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.LifecycleV1alpha1().EvictionRequests(f.Namespace.Name)
		podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)

		labelKey, labelValue := "example-e2e-er-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		ginkgo.By("creating a target pod for EvictionRequest")
		podName := "e2e-er-target-" + utilrand.String(5)
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "agnhost",
						Image: imageutils.GetE2EImage(imageutils.Agnhost),
					},
				},
			},
		}
		pod, err := podClient.Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		template := &lifecycleapi.EvictionRequest{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-er-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: lifecycleapi.EvictionRequestSpec{
				Target: lifecycleapi.EvictionRequestTarget{
					Pod: &lifecycleapi.EvictionRequestPodReference{
						Name: pod.Name,
						UID:  pod.UID,
					},
				},
				Requester: "e2e-test.example.com/evictionrequest",
				Intent:    lifecycleapi.EvictionRequestIntentEviction,
			},
		}

		ginkgo.By("creating")
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		erCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		erRead, err := client.Get(ctx, erCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(erRead.UID).To(gomega.Equal(erCreated.UID))
		gomega.Expect(erRead).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		erWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}}}`)
		erPatched, err := client.Patch(ctx, erCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(erPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(resourceversion.CompareResourceVersion(erRead.ResourceVersion, erPatched.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var erUpdated *lifecycleapi.EvictionRequest
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			er, err := client.Get(ctx, erCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			erToUpdate := er.DeepCopy()
			erToUpdate.Annotations["updated"] = "true"

			erUpdated, err = client.Update(ctx, erToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update evictionrequest %q", erCreated.Name)
		gomega.Expect(erUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-erWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				erWatched, isER := evt.Object.(*lifecycleapi.EvictionRequest)
				if !isER {
					framework.Failf("expected an object of type: %T, but got %T", &lifecycleapi.EvictionRequest{}, evt.Object)
				}
				if erWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					erWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", erWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("getting /status")
		erStatusRead, err := f.DynamicClient.Resource(lifecycleapi.SchemeGroupVersion.WithResource("evictionrequests")).Namespace(f.Namespace.Name).Get(ctx, erCreated.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(erStatusRead.GetObjectKind().GroupVersionKind()).To(gomega.Equal(lifecycleapi.SchemeGroupVersion.WithKind("EvictionRequest")))
		gomega.Expect(erStatusRead.GetUID()).To(gomega.Equal(erCreated.UID))

		ginkgo.By("updating /status")
		var erStatusUpdated *lifecycleapi.EvictionRequest
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			er, err := client.Get(ctx, erCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			erStatusToUpdate := er.DeepCopy()
			erStatusToUpdate.Status.ObservedGeneration = ptr.To(erStatusToUpdate.Generation)

			erStatusUpdated, err = client.UpdateStatus(ctx, erStatusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update status of evictionrequest %q", erCreated.Name)
		gomega.Expect(erStatusUpdated.Status.ObservedGeneration).To(gomega.Equal(ptr.To(erStatusUpdated.Generation)))

		ginkgo.By("deleting")
		err = client.Delete(ctx, erCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = client.Get(ctx, erCreated.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})
})
