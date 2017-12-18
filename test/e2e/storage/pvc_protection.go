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
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/slice"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("PVC Protection [Feature:PVCProtection]", func() {
	var (
		client                  clientset.Interface
		nameSpace               string
		err                     error
		pvc                     *v1.PersistentVolumeClaim
		pvcCreatedAndNotDeleted bool
	)

	f := framework.NewDefaultFramework("pvc-protection")
	BeforeEach(func() {
		client = f.ClientSet
		nameSpace = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(client, framework.TestContext.NodeSchedulableTimeout))

		By("Creating a PVC")
		suffix := "pvc-protection"
		defaultSC := getDefaultStorageClassName(client)
		testStorageClass := storageClassTest{
			claimSize: "1Gi",
		}
		pvc = newClaim(testStorageClass, nameSpace, suffix)
		pvc.Spec.StorageClassName = &defaultSC
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating PVC")
		pvcCreatedAndNotDeleted = true

		By("Waiting for PVC to become Bound")
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, nameSpace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)

		By("Checking that PVC Protection finalizer is set")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While getting PVC status")
		Expect(slice.ContainsString(pvc.ObjectMeta.Finalizers, volumeutil.PVCProtectionFinalizer, nil)).To(BeTrue())
	})

	AfterEach(func() {
		if pvcCreatedAndNotDeleted {
			framework.DeletePersistentVolumeClaim(client, pvc.Name, nameSpace)
		}
	})

	It("Verify \"immediate\" deletion of a PVC that is not in active use by a pod", func() {
		By("Deleting the PVC")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred(), "Error deleting PVC")
		waitForPersistentVolumeClaimBeRemoved(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		pvcCreatedAndNotDeleted = false
	})

	It("Verify that PVC in active use by a pod is not removed immediatelly", func() {
		By("Creating a Pod that becomes Running and therefore is actively using the PVC")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pod, err := framework.CreatePod(client, nameSpace, nil, pvcClaims, false, "")
		Expect(err).NotTo(HaveOccurred(), "While creating pod that uses the PVC or waiting for the Pod to become Running")

		By("Deleting the PVC, however, the PVC must not be removed from the system as it's in active use by a pod")
		err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, metav1.NewDeleteOptions(0))
		Expect(err).NotTo(HaveOccurred(), "Error deleting PVC")

		By("Checking that the PVC status is Terminating")
		pvc, err = client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While checking PVC status")
		Expect(pvc.ObjectMeta.DeletionTimestamp).NotTo(Equal(nil))

		By("Deleting the pod that uses the PVC")
		err = framework.DeletePodWithWait(f, client, pod)
		Expect(err).NotTo(HaveOccurred(), "Error terminating and deleting pod")

		By("Checking that the PVC is automatically removed from the system because it's no longer in active use by a pod")
		waitForPersistentVolumeClaimBeRemoved(client, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		pvcCreatedAndNotDeleted = false
	})
})

// waitForPersistentVolumeClaimBeRemoved waits for a PersistentVolumeClaim to be removed from the system until timeout occurs, whichever comes first.
func waitForPersistentVolumeClaimBeRemoved(c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for PersistentVolumeClaim %s to be removed", timeout, pvcName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		_, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				framework.Logf("Claim %q in namespace %q doesn't exist in the system", pvcName, ns)
				return nil
			}
			framework.Logf("Failed to get claim %q in namespace %q, retrying in %v. Error: %v", pvcName, ns, Poll, err)
		}
	}
	return fmt.Errorf("PersistentVolumeClaim %s is not removed from the system within %v", pvcName, timeout)
}
