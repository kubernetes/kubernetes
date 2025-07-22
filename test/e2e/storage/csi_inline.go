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
	"fmt"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = utils.SIGDescribe("CSIInlineVolumes", func() {
	f := framework.NewDefaultFramework("csiinlinevolumes")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.26
		Testname: CSIInlineVolumes should support Pods with inline volumes
		Description: Pod resources with CSIVolumeSource should support
		  create, get, list, patch, and delete operations.
	*/
	framework.ConformanceIt("should support CSIVolumeSource in Pod API", func(ctx context.Context) {
		// Create client
		client := f.ClientSet.CoreV1().Pods(f.Namespace.Name)

		// Fake driver name for this API test
		driverName := "e2e.example.com"

		podName := "pod-csi-inline-volumes"
		vol1name := "csi-inline-vol1"
		vol2name := "csi-inline-vol2"
		vol2fstype := "ext4"
		vol2readonly := true

		// Create a simple pod object with 2 CSI inline volumes
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: podName,
			},
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						Name: vol1name,
						VolumeSource: v1.VolumeSource{
							CSI: &v1.CSIVolumeSource{
								Driver: driverName,
							},
						},
					},
					{
						Name: vol2name,
						VolumeSource: v1.VolumeSource{
							CSI: &v1.CSIVolumeSource{
								Driver:   driverName,
								FSType:   &vol2fstype,
								ReadOnly: &vol2readonly,
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    podName,
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "ls /mnt/*"},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      vol1name,
								MountPath: "/mnt/vol1",
							},
							{
								Name:      vol2name,
								MountPath: "/mnt/vol2",
							},
						},
					},
				},
			},
		}

		ginkgo.By("creating")
		createdPod, err := client.Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, pod, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			framework.Failf("expected 409, got %#v", err)
		}

		ginkgo.By("getting")
		retrievedPod, err := client.Get(ctx, podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(retrievedPod.UID).To(gomega.Equal(createdPod.UID))

		ginkgo.By("listing in namespace")
		podList, err := client.List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(podList.Items).To(gomega.HaveLen(1), "list should have 1 items, got: %s", podList)

		ginkgo.By("patching")
		patchedPod, err := client.Patch(ctx, createdPod.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(patchedPod.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")

		ginkgo.By("deleting")
		err = client.Delete(ctx, createdPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		retrievedPod, err = client.Get(ctx, createdPod.Name, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			// Okay, normal case.
		case err != nil:
			framework.Failf("expected 404, got %#v", err)
		case retrievedPod.DeletionTimestamp != nil:
			// Okay, normal case.
		default:
			framework.Failf("Pod should have been deleted or have DeletionTimestamp, but instead got: %s", retrievedPod)
		}
	})

	/*
		Release: v1.28
		Testname: CSIDriver, lifecycle
		Description: Creating two CSIDrivers MUST succeed. Patching a CSIDriver MUST
		succeed with its new label found. Updating a CSIDriver MUST succeed with its
		new label found. Two CSIDrivers MUST be found when listed. Deleting the first
		CSIDriver MUST succeed. Deleting the second CSIDriver via deleteCollection
		MUST succeed.
	*/
	framework.ConformanceIt("should run through the lifecycle of a CSIDriver", func(ctx context.Context) {
		// Create client
		client := f.ClientSet.StorageV1().CSIDrivers()
		defaultFSGroupPolicy := storagev1.ReadWriteOnceWithFSTypeFSGroupPolicy
		csiDriverLabel := map[string]string{"e2e-test": f.UniqueName}
		csiDriverLabelSelector := labels.SelectorFromSet(csiDriverLabel).String()

		// Driver that supports only Ephemeral
		driver1 := &storagev1.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "inline-driver-" + string(uuid.NewUUID()),
				Labels: csiDriverLabel,
			},

			Spec: storagev1.CSIDriverSpec{
				VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecycleEphemeral},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		}

		// Driver that supports both Ephemeral and Persistent
		driver2 := &storagev1.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "inline-driver-" + string(uuid.NewUUID()),
				Labels: csiDriverLabel,
			},

			Spec: storagev1.CSIDriverSpec{
				VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{
					storagev1.VolumeLifecyclePersistent,
					storagev1.VolumeLifecycleEphemeral,
				},
				FSGroupPolicy: &defaultFSGroupPolicy,
			},
		}

		ginkgo.By("Creating two CSIDrivers")
		createdDriver1, err := client.Create(ctx, driver1, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create first CSIDriver")
		createdDriver2, err := client.Create(ctx, driver2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create second CSIDriver")
		_, err = client.Create(ctx, driver1, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			framework.Failf("expected 409, got %#v", err)
		}

		ginkgo.By(fmt.Sprintf("Getting %q & %q", createdDriver1.Name, createdDriver2.Name))
		retrievedDriver1, err := client.Get(ctx, createdDriver1.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get CSIDriver %q", createdDriver1.Name)
		gomega.Expect(retrievedDriver1.UID).To(gomega.Equal(createdDriver1.UID))

		retrievedDriver2, err := client.Get(ctx, createdDriver2.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get CSIDriver %q", createdDriver2.Name)
		gomega.Expect(retrievedDriver2.UID).To(gomega.Equal(createdDriver2.UID))

		ginkgo.By(fmt.Sprintf("Patching the CSIDriver %q", createdDriver2.Name))
		payload := "{\"metadata\":{\"labels\":{\"" + createdDriver2.Name + "\":\"patched\"}}}"
		patchedCSIDriver, err := client.Patch(ctx, createdDriver2.Name, types.StrategicMergePatchType, []byte(payload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch CSIDriver %q", createdDriver2.Name)
		gomega.Expect(patchedCSIDriver.Labels[createdDriver2.Name]).To(gomega.ContainSubstring("patched"), "Checking that patched label has been applied")

		ginkgo.By(fmt.Sprintf("Updating the CSIDriver %q", createdDriver2.Name))
		var updatedCSIDriver *storagev1.CSIDriver

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			csiDriver, err := client.Get(ctx, createdDriver2.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "Unable to get CSIDriver %q", createdDriver2.Name)
			csiDriver.Labels[retrievedDriver2.Name] = "updated"
			updatedCSIDriver, err = client.Update(ctx, csiDriver, metav1.UpdateOptions{})

			return err
		})
		framework.ExpectNoError(err, "failed to update CSIDriver %q", createdDriver2.Name)
		gomega.Expect(updatedCSIDriver.Labels[createdDriver2.Name]).To(gomega.ContainSubstring("updated"), "Checking that updated label has been applied")

		ginkgo.By(fmt.Sprintf("Listing all CSIDrivers with the labelSelector: %q", csiDriverLabelSelector))
		driverList, err := client.List(ctx, metav1.ListOptions{LabelSelector: csiDriverLabelSelector})
		framework.ExpectNoError(err, "Failed to list all CSIDrivers with the labelSelector %q", csiDriverLabelSelector)
		gomega.Expect(driverList.Items).To(gomega.HaveLen(2), "filtered list should have 2 items, got: %s", driverList)

		ginkgo.By(fmt.Sprintf("Deleting CSIDriver %q", createdDriver1.Name))
		err = client.Delete(ctx, createdDriver1.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete CSIDriver %q", createdDriver1.Name)

		ginkgo.By(fmt.Sprintf("Confirm deletion of CSIDriver %q", createdDriver1.Name))
		retrievedDriver, err := client.Get(ctx, createdDriver1.Name, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			// Okay, normal case.
		case err != nil:
			framework.Failf("expected 404, got %#v", err)
		case retrievedDriver.DeletionTimestamp != nil:
			// Okay, normal case.
		default:
			framework.Failf("CSIDriver should have been deleted or have DeletionTimestamp, but instead got: %s", retrievedDriver)
		}

		ginkgo.By(fmt.Sprintf("Deleting CSIDriver %q via DeleteCollection", createdDriver2.Name))
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: createdDriver2.Name + "=updated"})
		framework.ExpectNoError(err, "Failed to delete CSIDriver %q", createdDriver2.Name)

		ginkgo.By(fmt.Sprintf("Confirm deletion of CSIDriver %q", createdDriver2.Name))
		retrievedDriver, err = client.Get(ctx, createdDriver2.Name, metav1.GetOptions{})
		switch {
		case apierrors.IsNotFound(err):
			// Okay, normal case.
		case err != nil:
			framework.Failf("expected 404, got %#v", err)
		case retrievedDriver.DeletionTimestamp != nil:
			// Okay, normal case.
		default:
			framework.Failf("CSIDriver should have been deleted or have DeletionTimestamp, but instead got: %s", retrievedDriver)
		}
	})
})
