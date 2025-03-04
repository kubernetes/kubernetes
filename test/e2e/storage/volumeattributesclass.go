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

package storage

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	VacFinalizer = "kubernetes.io/vac-protection"
)

var _ = utils.SIGDescribe("VolumeAttributesClass", feature.VolumeAttributesClass, framework.WithFeatureGate(features.VolumeAttributesClass), func() {

	f := framework.NewDefaultFramework("csi-volumeattributesclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline
	/*
		Release: v1.29
		Testname: VolumeAttributesClass, lifecycle
		Description: Creating a VolumeAttributesClass MUST succeed. Reading the VolumeAttributesClass MUST
		succeed. Patching the VolumeAttributesClass MUST succeed with its new label found. Deleting
		the VolumeAttributesClass MUST succeed and it MUST be confirmed. Replacement VolumeAttributesClass
		MUST be created. Updating the VolumeAttributesClass MUST succeed with its new label found.
		Deleting the VolumeAttributesClass via deleteCollection MUST succeed and it MUST be confirmed.
	*/
	framework.It("should run through the lifecycle of a VolumeAttributesClass", func(ctx context.Context) {

		vacClient := f.ClientSet.StorageV1beta1().VolumeAttributesClasses()
		var initialVAC, replacementVAC *storagev1beta1.VolumeAttributesClass

		initialVAC = &storagev1beta1.VolumeAttributesClass{
			TypeMeta: metav1.TypeMeta{
				Kind: "VolumeAttributesClass",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-",
			},
			DriverName: "e2e-fake-csi-driver",
			Parameters: map[string]string{
				"foo": "bar",
			},
		}

		ginkgo.By("Creating a VolumeAttributesClass")
		createdVolumeAttributesClass, err := vacClient.Create(ctx, initialVAC, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create the requested VolumeAttributesClass")

		ginkgo.By(fmt.Sprintf("Get VolumeAttributesClass %q", createdVolumeAttributesClass.Name))
		retrievedVolumeAttributesClass, err := vacClient.Get(ctx, createdVolumeAttributesClass.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "failed to get VolumeAttributesClass %q", createdVolumeAttributesClass.Name)

		ginkgo.By(fmt.Sprintf("Patching the VolumeAttributesClass %q", retrievedVolumeAttributesClass.Name))
		payload := "{\"metadata\":{\"labels\":{\"" + retrievedVolumeAttributesClass.Name + "\":\"patched\"}}}"
		patchedVolumeAttributesClass, err := vacClient.Patch(ctx, retrievedVolumeAttributesClass.Name, types.StrategicMergePatchType, []byte(payload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "failed to patch VolumeAttributesClass %q", retrievedVolumeAttributesClass.Name)
		gomega.Expect(patchedVolumeAttributesClass.Labels).To(gomega.HaveKeyWithValue(patchedVolumeAttributesClass.Name, "patched"), "checking that patched label has been applied")

		ginkgo.By(fmt.Sprintf("Delete VolumeAttributesClass %q", patchedVolumeAttributesClass.Name))
		err = vacClient.Delete(ctx, patchedVolumeAttributesClass.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "failed to delete VolumeAttributesClass %q", patchedVolumeAttributesClass.Name)

		ginkgo.By(fmt.Sprintf("Confirm deletion of VolumeAttributesClass %q", patchedVolumeAttributesClass.Name))

		vacSelector := labels.Set{patchedVolumeAttributesClass.Name: "patched"}.AsSelector().String()
		type state struct {
			VolumeAttributesClasses []storagev1beta1.VolumeAttributesClass
		}

		err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
			vacList, err := vacClient.List(ctx, metav1.ListOptions{LabelSelector: vacSelector})
			if err != nil {
				return nil, fmt.Errorf("failed to list VolumeAttributesClass: %w", err)
			}
			return &state{
				VolumeAttributesClasses: vacList.Items,
			}, nil
		})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
			if len(s.VolumeAttributesClasses) == 0 {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("expected VolumeAttributesClass to be deleted, found %q", s.VolumeAttributesClasses[0].Name)
			}, nil
		}))
		framework.ExpectNoError(err, "timeout while waiting to confirm VolumeAttributesClass %q deletion", patchedVolumeAttributesClass.Name)

		ginkgo.By("Create a replacement VolumeAttributesClass")

		replacementVAC = &storagev1beta1.VolumeAttributesClass{
			TypeMeta: metav1.TypeMeta{
				Kind: "VolumeAttributesClass",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-v2-",
			},
			DriverName: "e2e-fake-csi-driver",
			Parameters: map[string]string{
				"foo": "bar",
			},
		}

		replacementVolumeAttributesClass, err := vacClient.Create(ctx, replacementVAC, metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create replacement VolumeAttributesClass")

		ginkgo.By(fmt.Sprintf("Updating VolumeAttributesClass %q", replacementVolumeAttributesClass.Name))
		var updatedVolumeAttributesClass *storagev1beta1.VolumeAttributesClass

		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			vac, err := vacClient.Get(ctx, replacementVolumeAttributesClass.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "unable to get VolumeAttributesClass %q", replacementVolumeAttributesClass.Name)
			vac.Labels = map[string]string{replacementVolumeAttributesClass.Name: "updated"}
			updatedVolumeAttributesClass, err = vacClient.Update(ctx, vac, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update VolumeAttributesClass %q", replacementVolumeAttributesClass.Name)
		gomega.Expect(updatedVolumeAttributesClass.Labels).To(gomega.HaveKeyWithValue(replacementVolumeAttributesClass.Name, "updated"), "checking that updated label has been applied")

		vacSelector = labels.Set{replacementVolumeAttributesClass.Name: "updated"}.AsSelector().String()
		ginkgo.By(fmt.Sprintf("Listing all VolumeAttributesClasses with the labelSelector: %q", vacSelector))
		vacList, err := vacClient.List(ctx, metav1.ListOptions{LabelSelector: vacSelector})
		framework.ExpectNoError(err, "failed to list VolumeAttributesClasses with the labelSelector: %q", vacSelector)
		gomega.Expect(vacList.Items).To(gomega.HaveLen(1))

		ginkgo.By(fmt.Sprintf("Deleting VolumeAttributesClass %q via DeleteCollection", updatedVolumeAttributesClass.Name))
		err = vacClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: vacSelector})
		framework.ExpectNoError(err, "failed to delete VolumeAttributesClass %q", updatedVolumeAttributesClass.Name)

		ginkgo.By(fmt.Sprintf("Confirm deletion of VolumeAttributesClass %q", updatedVolumeAttributesClass.Name))

		err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (*state, error) {
			vacList, err := vacClient.List(ctx, metav1.ListOptions{LabelSelector: vacSelector})
			if err != nil {
				return nil, fmt.Errorf("failed to list VolumeAttributesClass: %w", err)
			}
			return &state{
				VolumeAttributesClasses: vacList.Items,
			}, nil
		})).WithTimeout(30 * time.Second).Should(framework.MakeMatcher(func(s *state) (func() string, error) {
			if len(s.VolumeAttributesClasses) == 0 {
				return nil, nil
			}
			return func() string {
				return fmt.Sprintf("expected VolumeAttributesClass to be deleted, found %q", s.VolumeAttributesClasses[0].Name)
			}, nil
		}))
		framework.ExpectNoError(err, "timeout while waiting to confirm VolumeAttributesClass %q deletion", updatedVolumeAttributesClass.Name)
	})

	framework.It("should be protected by vac-protection finalizer", func(ctx context.Context) {
		vacClient := f.ClientSet.StorageV1beta1().VolumeAttributesClasses()

		initialVAC := &storagev1beta1.VolumeAttributesClass{
			TypeMeta: metav1.TypeMeta{
				Kind: "VolumeAttributesClass",
			},
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-",
			},
			DriverName: "e2e-fake-csi-driver",
			Parameters: map[string]string{
				"foo": "bar",
			},
		}

		ginkgo.By("Creating a VolumeAttributesClass")
		createdVAC, err := vacClient.Create(ctx, initialVAC, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create the requested VolumeAttributesClass")
		defer vacClient.Delete(ctx, createdVAC.Name, metav1.DeleteOptions{})

		ginkgo.By("Verifying that the vac-protection finalizer is set")
		vac, err := vacClient.Get(ctx, createdVAC.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get VolumeAttributesClass")
		framework.Logf("Finalizers are %v", vac.Finalizers)
		gomega.Expect(vac.Finalizers).Should(gomega.ContainElement(VacFinalizer), "Expected finalizer not found")

		// Simulate VAC being in use
		ginkgo.By("Creating a PVC that uses the VolumeAttributesClass")
		pvc := &corev1.PersistentVolumeClaim{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-pvc-",
			},
			Spec: corev1.PersistentVolumeClaimSpec{
				VolumeAttributesClassName: &createdVAC.Name,
				AccessModes:               []corev1.PersistentVolumeAccessMode{corev1.ReadWriteOnce},
				Resources:                 corev1.VolumeResourceRequirements{Requests: corev1.ResourceList{corev1.ResourceStorage: resource.MustParse("1")}},
			},
		}
		pvcClient := f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name)
		createdPVC, err := pvcClient.Create(ctx, pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create PVC using VolumeAttributesClass %q", createdVAC.Name)
		defer pvcClient.Delete(ctx, createdPVC.Name, metav1.DeleteOptions{})

		ginkgo.By("Attempting to delete the VolumeAttributesClass should be stuck")
		err = vacClient.Delete(ctx, createdVAC.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete VolumeAttributesClass %q", createdVAC.Name)

		watcher, err := vacClient.Watch(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("metadata.name=%s", createdVAC.Name),
		})
		framework.ExpectNoError(err, "Failed to set up watcher for VolumeAttributesClass %q", createdVAC.Name)
		defer watcher.Stop()

		finalizerRemoved := false
		timeout := time.After(10 * time.Second)

	watchLoop:
		for {
			select {
			case event := <-watcher.ResultChan():
				if vac, ok := event.Object.(*storagev1beta1.VolumeAttributesClass); ok {
					if len(vac.Finalizers) == 0 {
						framework.Logf("Finalizer removed for %q", vac.Name)
						finalizerRemoved = true
						break watchLoop
					}
					if vac.DeletionTimestamp != nil {
						framework.Logf("DeletionTimestamp is set for %q", vac.Name)
					}
				} else if event.Type == watch.Deleted {
					framework.Failf("The VolumeAttributesClass %q was unexpectedly deleted", createdVAC.Name)
				}
			case <-timeout:
				framework.Logf("Waiting for VolumeAttributesClass %q finalizer to be removed stuck as expected", createdVAC.Name)
				break watchLoop
			}
		}
		gomega.Expect(finalizerRemoved).Should(gomega.BeFalse(), "Finalizer have been removed before deletion")

		ginkgo.By(fmt.Sprintf("Deleting PVC %q to make the vac unused", createdPVC.Name))
		err = pvcClient.Delete(ctx, createdPVC.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "Failed to delete PVC %q", createdPVC.Name)

		ginkgo.By(fmt.Sprintf("Waiting for PVC %q to be fully deleted", createdPVC.Name))
		err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (bool, error) {
			_, err := pvcClient.Get(ctx, createdPVC.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil // PVC is deleted
			}
			return false, nil
		})).WithTimeout(30 * time.Second).Should(gomega.BeTrue())
		framework.ExpectNoError(err, "Timeout while waiting for PVC %q deletion", createdPVC.Name)

		ginkgo.By(fmt.Sprintf("Confirming final deletion of VolumeAttributesClass %q", createdVAC.Name))
		err = framework.Gomega().Eventually(ctx, framework.HandleRetry(func(ctx context.Context) (bool, error) {
			_, err = vacClient.Get(ctx, createdVAC.Name, metav1.GetOptions{})
			if err != nil && apierrors.IsNotFound(err) {
				return true, nil // VAC is deleted
			}
			return false, nil
		})).WithTimeout(30 * time.Second).Should(gomega.BeTrue())

		framework.ExpectNoError(err, "Timeout while waiting to confirm VolumeAttributesClass %q deletion", createdVAC.Name)
	})

})
