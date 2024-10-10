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

	"k8s.io/klog/v2"

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
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("Retroactive StorageClass Assignment", func() {
	f := framework.NewDefaultFramework("retroactive-storageclass")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var (
		client    clientset.Interface
		namespace string
	)

	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		namespace = f.Namespace.Name
	})

	f.It("should assign default StorageClass to PVCs retroactively", f.WithDisruptive(), f.WithSerial(), func(ctx context.Context) {
		defaultSCs, err := getDefaultStorageClasses(ctx, client)
		framework.ExpectNoError(err, "Failed to get default StorageClasses")

		defer func() {
			// Restore existing default StorageClasses at the end of the test
			for _, sc := range defaultSCs {
				setStorageClassDefault(ctx, client, sc.Name, "true")
			}
		}()

		// Unset all default StorageClasses
		for _, sc := range defaultSCs {
			klog.InfoS("Unsetting default StorageClass", "StorageClass", sc.Name)
			setStorageClassDefault(ctx, client, sc.Name, "false")
		}

		// Ensure no default StorageClasses exist
		if len(defaultSCs) > 0 {
			err = ensureNoDefaultStorageClasses(ctx, client)
			framework.ExpectNoError(err, "Failed to ensure no default StorageClasses exist")
		}

		// Create a PVC with nil StorageClass and confirm it is in a Pending state
		pvc := createPVC(ctx, client, namespace)
		defer deletePVC(ctx, client, namespace, pvc.Name)
		err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimPending, client, namespace, pvc.Name, time.Second, 30*time.Second)
		framework.ExpectNoError(err, "PVC should be in Pending state when no default StorageClass exists")

		// Create a default StorageClass
		sc := createDefaultStorageClass(ctx, client)
		defer deleteStorageClass(ctx, client, sc.Name)

		// Create PV with the default StorageClass
		pv := createPV(ctx, client, f, sc.Name, pvc.Spec.VolumeMode)
		ginkgo.DeferCleanup(e2epv.DeletePersistentVolume, client, pv.Name)

		// Ensure the PVC is now in a Bound state
		err = e2epv.WaitForPersistentVolumeClaimPhase(ctx, v1.ClaimBound, client, namespace, pvc.Name, time.Second, 60*time.Second)
		framework.ExpectNoError(err, "Error waiting for PVC to become bound")

		// Get the bound PVC to validate StorageClass assignment
		boundPVC, err := client.CoreV1().PersistentVolumeClaims(namespace).Get(ctx, pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "Error getting bound PVC")
		gomega.Expect(boundPVC.Spec.StorageClassName).ToNot(gomega.BeNil(), "Bound PVC should have a StorageClassName")
		gomega.Expect(*boundPVC.Spec.StorageClassName).To(gomega.Equal(sc.Name), "Bound PVC should have the correct StorageClassName")
	})
})

func getDefaultStorageClasses(ctx context.Context, client clientset.Interface) ([]storagev1.StorageClass, error) {
	scList, err := client.StorageV1().StorageClasses().List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var defaultSCs []storagev1.StorageClass
	for _, sc := range scList.Items {
		if sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] == "true" {
			defaultSCs = append(defaultSCs, sc)
		}
	}
	return defaultSCs, nil
}

func setStorageClassDefault(ctx context.Context, client clientset.Interface, scName string, isDefault string) {
	sc, err := client.StorageV1().StorageClasses().Get(ctx, scName, metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get StorageClass %s", scName)

	if sc.Annotations == nil {
		sc.Annotations = make(map[string]string)
	}
	sc.Annotations[storageutil.IsDefaultStorageClassAnnotation] = isDefault

	_, err = client.StorageV1().StorageClasses().Update(ctx, sc, metav1.UpdateOptions{})
	framework.ExpectNoError(err, "Failed to update StorageClass %s", scName)
}

func ensureNoDefaultStorageClasses(ctx context.Context, client clientset.Interface) error {
	return wait.PollUntilContextTimeout(ctx, 5*time.Second, 2*time.Minute, true, func(ctx context.Context) (bool, error) {
		defaultSCs, err := getDefaultStorageClasses(ctx, client)
		if err != nil {
			return false, err
		}
		if len(defaultSCs) == 0 {
			return true, nil
		}
		framework.Logf("Still found %d default StorageClasses, waiting...", len(defaultSCs))
		return false, nil
	})
}

func createPV(ctx context.Context, client clientset.Interface, f *framework.Framework, storageClassName string, volumeMode *v1.PersistentVolumeMode) *v1.PersistentVolume {
	pv := e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
		StorageClassName: storageClassName,
		VolumeMode:       volumeMode,
		PVSource: v1.PersistentVolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/tmp/test",
			},
		},
	})
	pv, err := e2epv.CreatePV(ctx, client, f.Timeouts, pv)
	framework.ExpectNoError(err, "Error creating PV %v", err)
	return pv
}

func createPVC(ctx context.Context, client clientset.Interface, namespace string) *v1.PersistentVolumeClaim {
	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		Name: "test-pvc",
	}

	ginkgo.By("Creating a PVC")
	pvcObj := e2epv.MakePersistentVolumeClaim(pvcConfig, namespace)
	pvc, err := e2epv.CreatePVC(ctx, client, namespace, pvcObj)
	framework.ExpectNoError(err, "Failed to create PVC")
	return pvc
}

func deletePVC(ctx context.Context, client clientset.Interface, namespace, pvcName string) {
	err := client.CoreV1().PersistentVolumeClaims(namespace).Delete(ctx, pvcName, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "Failed to delete PVC %s", pvcName)
}

func createDefaultStorageClass(ctx context.Context, client clientset.Interface) *storagev1.StorageClass {
	sc := &storagev1.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-default-sc",
			Annotations: map[string]string{
				storageutil.IsDefaultStorageClassAnnotation: "true",
			},
		},
		Provisioner: "fake-1",
	}

	createdSC, err := client.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create default StorageClass")
	return createdSC
}
