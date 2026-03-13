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

		ginkgo.DeferCleanup(func(cleanupContext context.Context) {
			// Restore existing default StorageClasses at the end of the test
			for _, sc := range defaultSCs {
				updateDefaultStorageClass(cleanupContext, client, sc.Name, "true")
			}
		})

		// Unset all default StorageClasses
		for _, sc := range defaultSCs {
			klog.InfoS("Unsetting default StorageClass", "StorageClass", sc.Name)
			updateDefaultStorageClass(ctx, client, sc.Name, "false")
		}

		// Ensure no default StorageClasses exist
		if len(defaultSCs) > 0 {
			ensureNoDefaultStorageClasses(ctx, client)
		}

		// Create a PVC with nil StorageClass
		pvc := createPVC(ctx, client, namespace)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.CoreV1().PersistentVolumeClaims(namespace).Delete(ctx, pvc.Name, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err, "Error deleting PVC")
		})

		// Create a default StorageClass
		sc := createDefaultStorageClass(ctx, client)
		ginkgo.DeferCleanup(func(cleanupContext context.Context) {
			deleteStorageClass(cleanupContext, client, sc.Name)
		})

		// Verify that the PVC is assigned the default StorageClass
		gomega.Eventually(ctx, framework.GetObject(client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get, pvc.Name, metav1.GetOptions{})).
			WithPolling(framework.Poll).
			WithTimeout(2*time.Minute).
			Should(gomega.HaveField("Spec.StorageClassName", gomega.Equal(&sc.Name)),
				"failed to wait for PVC to have the storageclass %s", sc.Name)
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

func ensureNoDefaultStorageClasses(ctx context.Context, client clientset.Interface) {
	gomega.Eventually(ctx, func() (int, error) {
		defaultSCs, err := getDefaultStorageClasses(ctx, client)
		if err != nil {
			return 0, err
		}
		return len(defaultSCs), nil
	}).WithPolling(framework.Poll).WithTimeout(2*time.Minute).
		Should(gomega.BeZero(), "Expected no default StorageClasses")
}

func createPVC(ctx context.Context, client clientset.Interface, namespace string) *v1.PersistentVolumeClaim {
	ginkgo.By("Creating a PVC")

	c := e2epv.PersistentVolumeClaimConfig{
		Name: "test-pvc",
	}

	pvcObj := e2epv.MakePersistentVolumeClaim(c, namespace)
	pvc, err := client.CoreV1().PersistentVolumeClaims(pvcObj.Namespace).Create(ctx, pvcObj, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating PVC")

	return pvc
}

func createDefaultStorageClass(ctx context.Context, client clientset.Interface) *storagev1.StorageClass {
	ginkgo.By("Creating a default StorageClass")

	c := &storagev1.StorageClass{
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

	sc, err := client.StorageV1().StorageClasses().Create(ctx, c, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating StorageClass")

	return sc
}
