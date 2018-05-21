/*
Copyright 2018 The Kubernetes Authors.

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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Volume online expand [Feature:ExpandOnlinePersistentVolumes] [Slow]", func() {
	var (
		c           clientset.Interface
		ns          string
		err         error
		pvc         *v1.PersistentVolumeClaim
		resizableSc *storage.StorageClass
		nonresizableSc *storage.StorageClass
	)

	f := framework.NewDefaultFramework("volume-online-expand")
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce", "cinder")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}
		resizableSc, err = createResizableStorageClass(test, ns, "resizing", c)
		Expect(err).NotTo(HaveOccurred(), "Error creating resizable storage class")
		Expect(resizableSc.AllowVolumeExpansion).NotTo(BeNil())
		Expect(*resizableSc.AllowVolumeExpansion).To(BeTrue())

		pvc = newClaim(test, ns, "default")
		pvc.Spec.StorageClassName = &resizableSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc")
	})

	AfterEach(func() {
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace))
		framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(resizableSc.Name, nil))
	})

	It("Verify if editing PVC online allows resize", func() {
		By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		By("Creating a pod with dynamically provisioned volume")
		pod, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		Expect(err).NotTo(HaveOccurred(), "While creating pods for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod)
			Expect(err).NotTo(HaveOccurred(), "while cleaning up pod already deleted in resize test")
		}()

		By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		Expect(err).NotTo(HaveOccurred(), "While updating pvc for more size")
		Expect(pvc).NotTo(BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pvc resize to finish")

		By("Checking for conditions on pvc")
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While fetching pvc after controller resize")

		inProgressConditions := pvc.Status.Conditions
		Expect(len(inProgressConditions)).To(Equal(1), "pvc must have file system resize pending condition")
		Expect(inProgressConditions[0].Type).To(Equal(v1.PersistentVolumeClaimFileSystemResizePending), "pvc must have fs resizing condition")

		By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})

	It("Verify PVC is resized to the latest request when multiple requests sent during resize in-progress ", func() {
		By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		By("Creating a pod with dynamically provisioned volume")
		pod, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		Expect(err).NotTo(HaveOccurred(), "While creating pods for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod)
			Expect(err).NotTo(HaveOccurred(), "while cleaning up pod already deleted in resize test")
		}()

		By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		Expect(err).NotTo(HaveOccurred(), "While updating pvc for more size")
		Expect(pvc).NotTo(BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pvc resize to finish")

		By("Checking for conditions on pvc")
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While fetching pvc after controller resize")

		inProgressConditions := pvc.Status.Conditions
		Expect(len(inProgressConditions)).To(Equal(1), "pvc must have file system resize pending condition")
		Expect(inProgressConditions[0].Type).To(Equal(v1.PersistentVolumeClaimFileSystemResizePending), "pvc must have fs resizing condition")

		By("Passing multiple requests in quick succession")
		newSize1 := resource.MustParse("7Gi")
		pvc, err = expandPVCSize(pvc, newSize1, c)
		newSize2 := resource.MustParse("8Gi")
		pvc, err = expandPVCSize(pvc, newSize2, c)
		Expect(err).NotTo(HaveOccurred(), "While updating pvc for more size")
		Expect(pvc).NotTo(BeNil())

		pvcNewSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcNewSize.Cmp(newSize2) != 0 {
			framework.Failf("error updating pvc size to the latest request %q", pvc.Name)
		}

		By("Waiting for cloudprovider new resize to finish")
		err = waitForControllerVolumeResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pvc resize to finish")

		By("Checking for latest conditions on pvc")
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "While fetching pvc after controller resize")

		latestInProgressConditions := pvc.Status.Conditions
		Expect(len(latestInProgressConditions)).To(Equal(1), "pvc must have file system resize pending condition")
		Expect(latestInProgressConditions[0].Type).To(Equal(v1.PersistentVolumeClaimFileSystemResizePending), "pvc must have fs resizing condition")

		By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})

	It("Try resizing PVC when AllowVolumeExpansion is not set in sc", func() {
		By("Creating storageclass without AllowVolumeExpansion parameter")
		c = f.ClientSet
		ns = f.Namespace.Name
		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}
		nonresizableSc, err = createNonResizableStorageClass(test, ns, "nonresizing", c)
		Expect(err).NotTo(HaveOccurred(), "Error creating non-resizable storage class")

		pvc = newClaim(test, ns, "default")
		pvc.Spec.StorageClassName = &nonresizableSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc")

		defer func() {
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace))
			framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(nonresizableSc.Name, nil))
		}()

		By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		By("Trying to expand current pvc")
		newSize := resource.MustParse("4Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.Logf("got err %v", err)
		Expect(err).To(HaveOccurred())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) == 0 {
			framework.Failf("AllowVolumeExpansion is False, resizing pvc is not allowed %q", pvc.Name)
		}

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})

	It("Try resizing PVC that is in pending state", func() {
		scName := getDefaultStorageClassName(c)
		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}

		By("setting the is-default StorageClass annotation to false")
		verifyDefaultStorageClass(c, scName, true)
		defer updateDefaultStorageClass(c, scName, "true")
		updateDefaultStorageClass(c, scName, "false")

		pvc = newClaim(test, ns, "default")
		pvc.Spec.StorageClassName = &resizableSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc")

		defer func() {
			framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace))
		}()

		// The claim should timeout phase:Pending
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, c, ns, pvc.Name, 2*time.Second, framework.ClaimProvisionShortTimeout)
		Expect(err).To(HaveOccurred())
		framework.Logf(err.Error())
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(pvc.Status.Phase).To(Equal(v1.ClaimPending))

		By("Trying to expand pvc that is in pending state")
		newSize := resource.MustParse("4Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.Logf("got err %v", err)
		Expect(err).To(HaveOccurred())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) == 0 {
			framework.Failf("resizing pvc with pending state is not allowed %q", pvc.Name)
		}

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})

	It("Verify shrinking PVC size is not allowed", func() {
		By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		By("Creating a pod with dynamically provisioned volume")
		pod, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		Expect(err).NotTo(HaveOccurred(), "While creating pods for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod)
			Expect(err).NotTo(HaveOccurred(), "while cleaning up pod already deleted in resize test")
		}()

		By("Trying to shrink current pvc")
		newSize := resource.MustParse("1Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.Logf("got err %v", err)
		Expect(err).To(HaveOccurred())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) == 0 {
			framework.Failf("resizing pvc to the lower size is not allowed %q", pvc.Name)
		}

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})
})

func createNonResizableStorageClass(t storageClassTest, ns string, suffix string, c clientset.Interface) (*storage.StorageClass, error) {
	stKlass := newStorageClass(t, ns, suffix)
	allowExpansion := false
	stKlass.AllowVolumeExpansion = &allowExpansion

	var err error
	stKlass, err = c.StorageV1().StorageClasses().Create(stKlass)
	return stKlass, err
}
