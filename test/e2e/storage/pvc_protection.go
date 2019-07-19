/*
Copyright 2017 The Kubernetes Authors.

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
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("PVC Protection", func() {
	var (
		client                  clientset.Interface
		nameSpace               string
		err                     error
		pvc                     *v1.PersistentVolumeClaim
		pvcCreatedAndNotDeleted bool
		pod                     *v1.Pod
	)

	f := framework.NewDefaultFramework("pvc-protection")
	ginkgo.BeforeEach(func() {
		client = f.ClientSet
		nameSpace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))

		ginkgo.By("Creating a PVC")
		prefix := "pvc-protection"
		framework.SkipIfNoDefaultStorageClass(client)
		t := testsuites.StorageClassTest{
			ClaimSize: "1Gi",
		}
		pvc = framework.MakePersistentVolumeClaim(framework.PersistentVolumeClaimConfig{
			NamePrefix: prefix,
			ClaimSize:  t.ClaimSize,
			VolumeMode: &t.VolumeMode,
		}, nameSpace)
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err, "Error creating PVC")
		pvcCreatedAndNotDeleted = true

		ginkgo.By("Creating a Pod that becomes Running and therefore is actively using the PVC")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pod, err = framework.CreatePod(client, nameSpace, nil, pvcClaims, false, "")
		framework.ExpectNoError(err, "While creating pod that uses the PVC or waiting for the Pod to become Running")

		ginkgo.By("Waiting for PVC to become Bound")
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, framework.ClaimBindingTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)

		ginkgo.By("Checking that PVC Protection finalizer is set")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While getting PVC status")
		gomega.Expect(slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)).To(gomega.BeTrue(), "PVC Protection finalizer(%v) is not set in %v", volumeutil.PVCProtectionFinalizer, pvc.ObjectMeta.Finalizers)
	})

	ginkgo.AfterEach(func() {
		if pvcCreatedAndNotDeleted {
			framework.DeletePersistentVolumeClaim(client, pvc.Name, nameSpace)
		}
	})

	ginkgo.It("Verify \"immediate\" deletion of a PVC that is not in active use by a pod", func() {
		ginkgo.By("Deleting the pod using the PVC")
		err = framework.DeletePodWithWait(f, client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Deleting the PVC")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PVC")
		framework.WaitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
		pvcCreatedAndNotDeleted = false
	})

	ginkgo.It("Verify that PVC in active use by a pod is not removed immediately", func() {
		ginkgo.By("Deleting the PVC, however, the PVC must not be removed from the system as it's in active use by a pod")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PVC")

		ginkgo.By("Checking that the PVC status is Terminating")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While checking PVC status")
		framework.ExpectNotEqual(pvc.ObjectMeta.DeletionTimestamp, nil)

		ginkgo.By("Deleting the pod that uses the PVC")
		err = framework.DeletePodWithWait(f, client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking that the PVC is automatically removed from the system because it's no longer in active use by a pod")
		framework.WaitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
		pvcCreatedAndNotDeleted = false
	})

	ginkgo.It("Verify that scheduling of a pod that uses PVC that is being deleted fails and the pod becomes Unschedulable", func() {
		ginkgo.By("Deleting the PVC, however, the PVC must not be removed from the system as it's in active use by a pod")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PVC")

		ginkgo.By("Checking that the PVC status is Terminating")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While checking PVC status")
		framework.ExpectNotEqual(pvc.ObjectMeta.DeletionTimestamp, nil)

		ginkgo.By("Creating second Pod whose scheduling fails because it uses a PVC that is being deleted")
		secondPod, err2 := framework.CreateUnschedulablePod(client, nameSpace, nil, []*v1.PersistentVolumeClaim{pvc}, false, "")
		framework.ExpectNoError(err2, "While creating second pod that uses a PVC that is being deleted and that is Unschedulable")

		ginkgo.By("Deleting the second pod that uses the PVC that is being deleted")
		err = framework.DeletePodWithWait(f, client, secondPod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking again that the PVC status is Terminating")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While checking PVC status")
		framework.ExpectNotEqual(pvc.ObjectMeta.DeletionTimestamp, nil)

		ginkgo.By("Deleting the first pod that uses the PVC")
		err = framework.DeletePodWithWait(f, client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking that the PVC is automatically removed from the system because it's no longer in active use by a pod")
		framework.WaitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
		pvcCreatedAndNotDeleted = false
	})
})
