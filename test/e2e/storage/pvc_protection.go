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

	"fmt"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// waitForPersistentVolumeClaimDeleted waits for a PersistentVolumeClaim to be removed from the system until timeout occurs, whichever comes first.
func waitForPersistentVolumeClaimDeleted(c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for PersistentVolumeClaim %s to be removed", timeout, pvcName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		_, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				framework.Logf("Claim %q in namespace %q doesn't exist in the system", pvcName, ns)
				return nil
			}
			framework.Logf("Failed to get claim %q in namespace %q, retrying in %v. Error: %v", pvcName, ns, Poll, err)
		}
	}
	return fmt.Errorf("PersistentVolumeClaim %s is not removed from the system within %v", pvcName, timeout)
}

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
		e2epv.SkipIfNoDefaultStorageClass(client)
		t := testsuites.StorageClassTest{
			ClaimSize: "1Gi",
		}
		pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			NamePrefix: prefix,
			ClaimSize:  t.ClaimSize,
			VolumeMode: &t.VolumeMode,
		}, nameSpace)
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err, "Error creating PVC")
		pvcCreatedAndNotDeleted = true

		ginkgo.By("Creating a Pod that becomes Running and therefore is actively using the PVC")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pod, err = e2epod.CreatePod(client, nameSpace, nil, pvcClaims, false, "")
		framework.ExpectNoError(err, "While creating pod that uses the PVC or waiting for the Pod to become Running")

		ginkgo.By("Waiting for PVC to become Bound")
		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, e2epv.ClaimBindingTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)

		ginkgo.By("Checking that PVC Protection finalizer is set")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While getting PVC status")
		framework.ExpectEqual(slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil), true, "PVC Protection finalizer(%v) is not set in %v", volumeutil.PVCProtectionFinalizer, pvc.ObjectMeta.Finalizers)
	})

	ginkgo.AfterEach(func() {
		if pvcCreatedAndNotDeleted {
			e2epv.DeletePersistentVolumeClaim(client, pvc.Name, nameSpace)
		}
	})

	ginkgo.It("Verify \"immediate\" deletion of a PVC that is not in active use by a pod", func() {
		ginkgo.By("Deleting the pod using the PVC")
		err = e2epod.DeletePodWithWait(client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Deleting the PVC")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		framework.ExpectNoError(err, "Error deleting PVC")
		waitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
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
		err = e2epod.DeletePodWithWait(client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking that the PVC is automatically removed from the system because it's no longer in active use by a pod")
		waitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
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
		secondPod, err2 := e2epod.CreateUnschedulablePod(client, nameSpace, nil, []*v1.PersistentVolumeClaim{pvc}, false, "")
		framework.ExpectNoError(err2, "While creating second pod that uses a PVC that is being deleted and that is Unschedulable")

		ginkgo.By("Deleting the second pod that uses the PVC that is being deleted")
		err = e2epod.DeletePodWithWait(client, secondPod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking again that the PVC status is Terminating")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While checking PVC status")
		framework.ExpectNotEqual(pvc.ObjectMeta.DeletionTimestamp, nil)

		ginkgo.By("Deleting the first pod that uses the PVC")
		err = e2epod.DeletePodWithWait(client, pod)
		framework.ExpectNoError(err, "Error terminating and deleting pod")

		ginkgo.By("Checking that the PVC is automatically removed from the system because it's no longer in active use by a pod")
		waitForPersistentVolumeClaimDeleted(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimDeletingTimeout)
		pvcCreatedAndNotDeleted = false
	})
})
