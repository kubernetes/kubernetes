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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("Persistent Volume Claim and StorageClass", func() {
	f := framework.NewDefaultFramework("pvc-retroactive-storageclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var (
		client    clientset.Interface
		namespace string
		prefixPVC string
		prefixSC  string
		t         testsuites.StorageClassTest
		pvc       *v1.PersistentVolumeClaim
		err       error
	)

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		namespace = f.Namespace.Name
		prefixPVC = "retro-pvc-"
		prefixSC = "retro"
		t = testsuites.StorageClassTest{
			Timeouts:  f.Timeouts,
			ClaimSize: "1Gi",
		}
	})

	f.Describe("Retroactive StorageClass assignment", framework.WithSerial(), framework.WithDisruptive(), func() {
		ginkgo.It("should assign default SC to PVCs that have no SC set", func(ctx context.Context) {

			// Temporarily set all default storage classes as non-default
			restoreClasses := temporarilyUnsetDefaultClasses(ctx, client)
			defer restoreClasses()

			// Create PVC with nil SC
			pvcObj := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				NamePrefix: prefixPVC,
				ClaimSize:  t.ClaimSize,
				VolumeMode: &t.VolumeMode,
			}, namespace)
			pvc, err = client.CoreV1().PersistentVolumeClaims(pvcObj.Namespace).Create(ctx, pvcObj, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Error creating PVC")
			defer func(pvc *v1.PersistentVolumeClaim) {
				// Remove test PVC
				err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(ctx, pvc.Name, metav1.DeleteOptions{})
				framework.ExpectNoError(err, "Error cleaning up PVC")
			}(pvc)

			// Create custom default SC
			storageClass := testsuites.SetupStorageClass(ctx, client, makeStorageClass(prefixSC))

			// Wait for PVC to get updated with the new default SC
			pvc, err = waitForPVCStorageClass(ctx, client, namespace, pvc.Name, storageClass.Name, f.Timeouts.ClaimBound)
			framework.ExpectNoError(err, "Error updating PVC with the correct storage class")

			// Create PV with specific class
			pv := e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
				NamePrefix:       "pv-",
				StorageClassName: storageClass.Name,
				VolumeMode:       pvc.Spec.VolumeMode,
				PVSource: v1.PersistentVolumeSource{
					HostPath: &v1.HostPathVolumeSource{
						Path: "/tmp/test",
					},
				},
			})
			_, err = e2epv.CreatePV(ctx, client, f.Timeouts, pv)
			framework.ExpectNoError(err, "Error creating pv %v", err)
			ginkgo.DeferCleanup(e2epv.DeletePersistentVolume, client, pv.Name)

			// Verify the PVC is bound and has the new default SC
			claimNames := []string{pvc.Name}
			err = e2epv.WaitForPersistentVolumeClaimsPhase(ctx, v1.ClaimBound, client, namespace, claimNames, 2*time.Second /* Poll */, t.Timeouts.ClaimProvisionShort, false)
			framework.ExpectNoError(err)
			updatedPVC, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(*updatedPVC.Spec.StorageClassName).To(gomega.Equal(storageClass.Name), "Expected PVC %v to have StorageClass %v, but it has StorageClass %v instead", updatedPVC.Name, prefixSC, updatedPVC.Spec.StorageClassName)
			framework.Logf("Success - PersistentVolumeClaim %s got updated retroactively with StorageClass %v", updatedPVC.Name, storageClass.Name)
		})
	})
})

func makeStorageClass(prefixSC string) *storagev1.StorageClass {
	return &storagev1.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: prefixSC,
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "true",
			},
		},
		Provisioner: "fake-1",
	}
}

func temporarilyUnsetDefaultClasses(ctx context.Context, client clientset.Interface) func() {
	classes, err := client.StorageV1().StorageClasses().List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)

	changedClasses := make(map[string]bool)

	for _, sc := range classes.Items {
		if sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] == "true" {
			changedClasses[sc.GetName()] = true
			sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] = "false"
			_, err := client.StorageV1().StorageClasses().Update(ctx, &sc, metav1.UpdateOptions{})
			framework.ExpectNoError(err)
		}
	}

	return func() {
		classes, err = client.StorageV1().StorageClasses().List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, sc := range classes.Items {
			if _, found := changedClasses[sc.GetName()]; found {
				sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] = "true"
				_, err := client.StorageV1().StorageClasses().Update(ctx, &sc, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
		}
	}

}

func waitForPVCStorageClass(ctx context.Context, c clientset.Interface, namespace, pvcName, scName string, timeout time.Duration) (*v1.PersistentVolumeClaim, error) {
	var watchedPVC *v1.PersistentVolumeClaim

	err := wait.PollWithContext(ctx, 1*time.Second, timeout, func(ctx context.Context) (bool, error) {
		var err error
		watchedPVC, err = c.CoreV1().PersistentVolumeClaims(namespace).Get(ctx, pvcName, metav1.GetOptions{})
		if err != nil {
			return true, err
		}

		if watchedPVC.Spec.StorageClassName == nil {
			return false, nil // Poll until PVC has correct SC
		}
		if watchedPVC.Spec.StorageClassName != nil && *watchedPVC.Spec.StorageClassName != scName {
			framework.Logf("PersistentVolumeClaim %s has unexpected StorageClass %v, expected StorageClass is: %v", watchedPVC.Name, *watchedPVC.Spec.StorageClassName, scName)
			return false, nil // Log unexpected SC and continue poll (something might be changing default SC)
		}
		return true, nil // Correct SC name found on PVC
	})

	if err != nil {
		return watchedPVC, fmt.Errorf("error waiting for claim %s to have StorageClass set to %s: %w", pvcName, scName, err)
	}

	return watchedPVC, nil
}
