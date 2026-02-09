/*
Copyright 2022 The Kubernetes Authors.

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

package storage

import (
	"context"
	"time"

	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = utils.SIGDescribe("CSIStorageCapacity", func() {
	f := framework.NewDefaultFramework("csistoragecapacity")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.24
		Testname: CSIStorageCapacity API
		Description:
		The storage.k8s.io API group MUST exist in the /apis discovery document.
		The storage.k8s.io/v1 API group/version MUST exist in the /apis/mode.k8s.io discovery document.
		The csistoragecapacities resource MUST exist in the /apis/storage.k8s.io/v1 discovery document.
		The csistoragecapacities resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support CSIStorageCapacities API operations", func(ctx context.Context) {
		// Setup
		cscVersion := "v1"
		cscClient := f.ClientSet.StorageV1().CSIStorageCapacities(f.Namespace.Name)
		cscClientNoNamespace := f.ClientSet.StorageV1().CSIStorageCapacities("")

		// The fictional StorageClass for these objects.
		scName := "e2e.example.com"

		// All CRUD operations in this test are limited to the objects with the label test=f.UniqueName
		newCSIStorageCapacity := func(nameSuffix string) *storagev1.CSIStorageCapacity {
			return &storagev1.CSIStorageCapacity{
				ObjectMeta: metav1.ObjectMeta{
					Name: f.UniqueName + nameSuffix,
					Labels: map[string]string{
						"test": f.UniqueName,
					},
				},
				StorageClassName: scName,
			}
		}
		csc := newCSIStorageCapacity("-csc1")
		csc2 := newCSIStorageCapacity("-csc2")
		csc3 := newCSIStorageCapacity("-csc3")

		// Discovery

		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == storagev1.GroupName {
					for _, version := range group.Versions {
						if version.Version == cscVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected CSIStorageCapacity API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/storage.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/storage.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == cscVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected CSIStorageCapacity API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/storage.k8s.io/" + cscVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(storagev1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			found := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "csistoragecapacities":
					found = true
				}
			}
			if !found {
				framework.Failf("expected csistoragecapacities, got %#v", resources.APIResources)
			}
		}

		// Main resource create/read/update/watch operations

		ginkgo.By("creating")
		createdCSC, err := cscClient.Create(ctx, csc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = cscClient.Create(ctx, csc, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			framework.Failf("expected 409, got %#v", err)
		}
		_, err = cscClient.Create(ctx, csc2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		cscWatch, err := cscClient.Watch(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		cscWatchNoNamespace, err := cscClientNoNamespace.Watch(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)

		// added for a watch
		_, err = cscClient.Create(ctx, csc3, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenCSC, err := cscClient.Get(ctx, csc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(gottenCSC.UID).To(gomega.Equal(createdCSC.UID))
		gomega.Expect(gottenCSC).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing in namespace")
		cscs, err := cscClient.List(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(cscs.Items).To(gomega.HaveLen(3), "filtered list should have 3 items, got: %s", cscs)

		ginkgo.By("listing across namespaces")
		cscs, err = cscClientNoNamespace.List(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		gomega.Expect(cscs.Items).To(gomega.HaveLen(3), "filtered list should have 3 items, got: %s", cscs)

		ginkgo.By("patching")
		patchedCSC, err := cscClient.Patch(ctx, createdCSC.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedCSC.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(resourceversion.CompareResourceVersion(createdCSC.ResourceVersion, patchedCSC.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		csrToUpdate := patchedCSC.DeepCopy()
		csrToUpdate.Annotations["updated"] = "true"
		updatedCSC, err := cscClient.Update(ctx, csrToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(updatedCSC.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")

		expectWatchResult := func(kind string, w watch.Interface) {
			framework.Logf("waiting for watch events with expected annotations %s", kind)
			for sawAdded, sawPatched, sawUpdated := false, false, false; !sawAdded && !sawPatched && !sawUpdated; {
				select {
				case evt, ok := <-w.ResultChan():
					if !ok {
						framework.Failf("%s: watch channel should not close", kind)
					}
					if evt.Type == watch.Modified {
						watchedCSC, isCSC := evt.Object.(*storagev1.CSIStorageCapacity)
						if !isCSC {
							framework.Failf("%s: expected CSC, got %T", kind, evt.Object)
						}
						if watchedCSC.Annotations["patched"] == "true" {
							framework.Logf("%s: saw patched annotations", kind)
							sawPatched = true
						} else if watchedCSC.Annotations["updated"] == "true" {
							framework.Logf("%s: saw updated annotations", kind)
							sawUpdated = true
						} else {
							framework.Logf("%s: missing expected annotations, waiting: %#v", kind, watchedCSC.Annotations)
						}
					} else if evt.Type == watch.Added {
						_, isCSC := evt.Object.(*storagev1.CSIStorageCapacity)
						if !isCSC {
							framework.Failf("%s: expected CSC, got %T", kind, evt.Object)
						}
						sawAdded = true
					}

				case <-time.After(wait.ForeverTestTimeout):
					framework.Failf("%s: timed out waiting for watch event", kind)
				}
			}
			w.Stop()
		}
		expectWatchResult("in namespace", cscWatch)
		expectWatchResult("across namespace", cscWatchNoNamespace)

		// main resource delete operations

		ginkgo.By("deleting")
		err = cscClient.Delete(ctx, createdCSC.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		csc, err = cscClient.Get(ctx, createdCSC.Name, metav1.GetOptions{})
		min := 2
		max := min
		switch {
		case apierrors.IsNotFound(err):
			// Okay, normal case.
		case err != nil:
			// Unexpected error.
			framework.Failf("expected 404, got %#v", err)
		case csc.DeletionTimestamp != nil && len(csc.Finalizers) > 0:
			// Deletion was prevented by a finalizer, but it might
			// still get deleted before we list them below.
			max++
		default:
			framework.Failf("CSIStorageCapacitity should have been deleted or have DeletionTimestamp and Finalizers, but instead got: %s", csc)
		}
		cscs, err = cscClient.List(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		actualLen := len(cscs.Items)
		if actualLen < min || actualLen > max {
			framework.Failf("expected <= %d and >= %d remaining CSIStorageCapacity objects, got %d: %v", max, min, actualLen, cscs.Items)
		}

		ginkgo.By("deleting a collection")
		err = cscClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		cscs, err = cscClient.List(ctx, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		for _, csc := range cscs.Items {
			// Any remaining objects should be marked for deletion
			// and only held back by a Finalizer.
			if csc.DeletionTimestamp == nil || len(csc.Finalizers) == 0 {
				framework.Failf("CSIStorageCapacity should have been deleted or have DeletionTimestamp and Finalizers, but instead got: %s", &csc)
			}
		}
	})
})
