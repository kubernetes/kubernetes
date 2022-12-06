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

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = utils.SIGDescribe("CSIInlineVolumes", func() {
	f := framework.NewDefaultFramework("csiinlinevolumes")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.26
		Testname: CSIInlineVolumes should support ephemeral CSIDrivers
		Description: CSIDriver resources with ephemeral VolumeLifecycleMode
		  should support create, get, list, and delete operations.
	*/
	framework.ConformanceIt("should support ephemeral VolumeLifecycleMode in CSIDriver API", func() {
		// Create client
		client := f.ClientSet.StorageV1().CSIDrivers()
		defaultFSGroupPolicy := storagev1.ReadWriteOnceWithFSTypeFSGroupPolicy

		// Driver that supports only Ephemeral
		driver1 := &storagev1.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{
				Name: "inline-driver-" + string(uuid.NewUUID()),
				Labels: map[string]string{
					"test": f.UniqueName,
				},
			},

			Spec: storagev1.CSIDriverSpec{
				VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecycleEphemeral},
				FSGroupPolicy:        &defaultFSGroupPolicy,
			},
		}

		// Driver that supports both Ephemeral and Persistent
		driver2 := &storagev1.CSIDriver{
			ObjectMeta: metav1.ObjectMeta{
				Name: "inline-driver-" + string(uuid.NewUUID()),
				Labels: map[string]string{
					"test": f.UniqueName,
				},
			},

			Spec: storagev1.CSIDriverSpec{
				VolumeLifecycleModes: []storagev1.VolumeLifecycleMode{
					storagev1.VolumeLifecyclePersistent,
					storagev1.VolumeLifecycleEphemeral,
				},
				FSGroupPolicy: &defaultFSGroupPolicy,
			},
		}

		ginkgo.By("creating")
		createdDriver1, err := client.Create(context.TODO(), driver1, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		createdDriver2, err := client.Create(context.TODO(), driver2, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(context.TODO(), driver1, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			framework.Failf("expected 409, got %#v", err)
		}

		ginkgo.By("getting")
		retrievedDriver1, err := client.Get(context.TODO(), createdDriver1.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(retrievedDriver1.UID, createdDriver1.UID)
		retrievedDriver2, err := client.Get(context.TODO(), createdDriver2.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(retrievedDriver2.UID, createdDriver2.UID)

		ginkgo.By("listing")
		driverList, err := client.List(context.TODO(), metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(driverList.Items), 2, "filtered list should have 2 items, got: %s", driverList)

		ginkgo.By("deleting")
		for _, driver := range driverList.Items {
			err := client.Delete(context.TODO(), driver.Name, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
			retrievedDriver, err := client.Get(context.TODO(), driver.Name, metav1.GetOptions{})
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
		}
	})

	/*
		Release: v1.26
		Testname: CSIInlineVolumes should support Pods with inline volumes
		Description: Pod resources with CSIVolumeSource should support
		  create, get, list, patch, and delete operations.
	*/
	framework.ConformanceIt("should support CSIVolumeSource in Pod API", func() {
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
		createdPod, err := client.Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(context.TODO(), pod, metav1.CreateOptions{})
		if !apierrors.IsAlreadyExists(err) {
			framework.Failf("expected 409, got %#v", err)
		}

		ginkgo.By("getting")
		retrievedPod, err := client.Get(context.TODO(), podName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(retrievedPod.UID, createdPod.UID)

		ginkgo.By("listing in namespace")
		podList, err := client.List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(podList.Items), 1, "list should have 1 items, got: %s", podList)

		ginkgo.By("patching")
		patchedPod, err := client.Patch(context.TODO(), createdPod.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedPod.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("deleting")
		err = client.Delete(context.TODO(), createdPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		retrievedPod, err = client.Get(context.TODO(), createdPod.Name, metav1.GetOptions{})
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
})
