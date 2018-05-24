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
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	resizePollInterval = 2 * time.Second
	// total time to wait for cloudprovider or file system resize to finish
	totalResizeWaitPeriod = 20 * time.Minute
)

var _ = utils.SIGDescribe("Volume expand [Slow]", func() {
	var (
		c           clientset.Interface
		ns          string
		err         error
		pvc         *v1.PersistentVolumeClaim
		resizableSc *storage.StorageClass
	)

	f := framework.NewDefaultFramework("volume-expand")
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce")
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

	It("Verify if editing PVC allows resize", func() {
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

		By("Deleting the previously created pod")
		err = framework.DeletePodWithWait(f, c, pod)
		Expect(err).NotTo(HaveOccurred(), "while deleting pod for resizing")

		By("Creating a new pod with same volume")
		pod2, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		Expect(err).NotTo(HaveOccurred(), "while recreating pod for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod2)
			Expect(err).NotTo(HaveOccurred(), "while cleaning up pod before exiting resizing test")
		}()

		By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})
})

func createResizableStorageClass(t storageClassTest, ns string, suffix string, c clientset.Interface) (*storage.StorageClass, error) {
	stKlass := newStorageClass(t, ns, suffix)
	allowExpansion := true
	stKlass.AllowVolumeExpansion = &allowExpansion

	var err error
	stKlass, err = c.StorageV1().StorageClasses().Create(stKlass)
	return stKlass, err
}

func expandPVCSize(origPVC *v1.PersistentVolumeClaim, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	pvcName := origPVC.Name
	updatedPVC := origPVC.DeepCopy()

	waitErr := wait.PollImmediate(resizePollInterval, 30*time.Second, func() (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for resizing with %v", pvcName, err)
		}

		updatedPVC.Spec.Resources.Requests[v1.ResourceStorage] = size
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Update(updatedPVC)
		if err == nil {
			return true, nil
		}
		framework.Logf("Error updating pvc %s with %v", pvcName, err)
		return false, nil
	})
	return updatedPVC, waitErr
}

func waitForControllerVolumeResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface) error {
	pvName := pvc.Spec.VolumeName
	return wait.PollImmediate(resizePollInterval, totalResizeWaitPeriod, func() (bool, error) {
		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]

		pv, err := c.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pv %q for resizing %v", pvName, err)
		}

		pvSize := pv.Spec.Capacity[v1.ResourceStorage]

		// If pv size is greater or equal to requested size that means controller resize is finished.
		if pvSize.Cmp(pvcSize) >= 0 {
			return true, nil
		}
		return false, nil
	})
}

func waitForFSResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	var updatedPVC *v1.PersistentVolumeClaim
	waitErr := wait.PollImmediate(resizePollInterval, totalResizeWaitPeriod, func() (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status : %v", pvc.Name, err)
		}

		pvcSize := updatedPVC.Spec.Resources.Requests[v1.ResourceStorage]
		pvcStatusSize := updatedPVC.Status.Capacity[v1.ResourceStorage]

		//If pvc's status field size is greater than or equal to pvc's size then done
		if pvcStatusSize.Cmp(pvcSize) >= 0 {
			return true, nil
		}
		return false, nil
	})
	return updatedPVC, waitErr
}
