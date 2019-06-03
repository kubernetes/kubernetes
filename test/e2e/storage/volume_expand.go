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

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	resizePollInterval = 2 * time.Second
	// total time to wait for cloudprovider or file system resize to finish
	totalResizeWaitPeriod = 20 * time.Minute
)

var _ = utils.SIGDescribe("Volume expand", func() {
	var (
		c               clientset.Interface
		ns              string
		err             error
		pvc             *v1.PersistentVolumeClaim
		storageClassVar *storage.StorageClass
	)

	f := framework.NewDefaultFramework("volume-expand")
	ginkgo.BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	setupFunc := func(allowExpansion bool, blockVolume bool) (*v1.PersistentVolumeClaim, *storage.StorageClass, error) {
		test := testsuites.StorageClassTest{
			Name:      "default",
			ClaimSize: "2Gi",
		}
		if allowExpansion {
			test.AllowVolumeExpansion = true
		}
		if blockVolume {
			test.VolumeMode = v1.PersistentVolumeBlock
		}

		sc, err := createStorageClass(test, ns, "resizing", c)
		framework.ExpectNoError(err, "Error creating storage class for resizing")

		tPVC := newClaim(test, ns, "default")
		tPVC.Spec.StorageClassName = &sc.Name
		tPVC, err = c.CoreV1().PersistentVolumeClaims(tPVC.Namespace).Create(tPVC)
		if err != nil {
			return nil, sc, err
		}
		return tPVC, sc, nil
	}

	ginkgo.AfterEach(func() {
		framework.ExpectNoError(framework.DeletePersistentVolumeClaim(c, pvc.Name, pvc.Namespace))
		framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(storageClassVar.Name, nil))
	})

	ginkgo.It("should not allow expansion of pvcs without AllowVolumeExpansion property", func() {
		pvc, storageClassVar, err = setupFunc(false /* allowExpansion */, false /*BlockVolume*/)
		framework.ExpectNoError(err, "Error creating non-expandable PVC")

		gomega.Expect(storageClassVar.AllowVolumeExpansion).To(gomega.BeNil())

		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		gomega.Expect(len(pvs)).To(gomega.Equal(1))

		ginkgo.By("Expanding non-expandable pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.ExpectError(err, "While updating non-expandable PVC")
	})

	ginkgo.It("Verify if editing PVC allows resize", func() {
		pvc, storageClassVar, err = setupFunc(true /* allowExpansion */, false /*BlockVolume*/)
		framework.ExpectNoError(err, "Error creating non-expandable PVC")

		ginkgo.By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		gomega.Expect(len(pvs)).To(gomega.Equal(1))

		ginkgo.By("Creating a pod with dynamically provisioned volume")
		pod, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		framework.ExpectNoError(err, "While creating pods for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod)
			framework.ExpectNoError(err, "while cleaning up pod already deleted in resize test")
		}()

		ginkgo.By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.ExpectNoError(err, "While updating pvc for more size")
		gomega.Expect(pvc).NotTo(gomega.BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		ginkgo.By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c, totalResizeWaitPeriod)
		framework.ExpectNoError(err, "While waiting for pvc resize to finish")

		ginkgo.By("Checking for conditions on pvc")
		pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "While fetching pvc after controller resize")

		inProgressConditions := pvc.Status.Conditions
		gomega.Expect(len(inProgressConditions)).To(gomega.Equal(1), "pvc must have file system resize pending condition")
		gomega.Expect(inProgressConditions[0].Type).To(gomega.Equal(v1.PersistentVolumeClaimFileSystemResizePending), "pvc must have fs resizing condition")

		ginkgo.By("Deleting the previously created pod")
		err = framework.DeletePodWithWait(f, c, pod)
		framework.ExpectNoError(err, "while deleting pod for resizing")

		ginkgo.By("Creating a new pod with same volume")
		pod2, err := framework.CreatePod(c, ns, nil, pvcClaims, false, "")
		framework.ExpectNoError(err, "while recreating pod for resizing")
		defer func() {
			err = framework.DeletePodWithWait(f, c, pod2)
			framework.ExpectNoError(err, "while cleaning up pod before exiting resizing test")
		}()

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		gomega.Expect(len(pvcConditions)).To(gomega.Equal(0), "pvc should not have conditions")
	})

	ginkgo.It("should allow expansion of block volumes", func() {
		pvc, storageClassVar, err = setupFunc(true /*allowExpansion*/, true /*blockVolume*/)

		ginkgo.By("Waiting for pvc to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		gomega.Expect(len(pvs)).To(gomega.Equal(1))

		ginkgo.By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		framework.ExpectNoError(err, "While updating pvc for more size")
		gomega.Expect(pvc).NotTo(gomega.BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		ginkgo.By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c, totalResizeWaitPeriod)
		framework.ExpectNoError(err, "While waiting for pvc resize to finish")

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		gomega.Expect(len(pvcConditions)).To(gomega.Equal(0), "pvc should not have conditions")
	})
})

func createStorageClass(t testsuites.StorageClassTest, ns string, suffix string, c clientset.Interface) (*storage.StorageClass, error) {
	stKlass := newStorageClass(t, ns, suffix)

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
		e2elog.Logf("Error updating pvc %s with %v", pvcName, err)
		return false, nil
	})
	return updatedPVC, waitErr
}

func waitForResizingCondition(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
	waitErr := wait.PollImmediate(resizePollInterval, duration, func() (bool, error) {
		var err error
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status : %v", pvc.Name, err)
		}

		pvcConditions := updatedPVC.Status.Conditions
		if len(pvcConditions) > 0 {
			if pvcConditions[0].Type == v1.PersistentVolumeClaimResizing {
				return true, nil
			}
		}
		return false, nil
	})
	return waitErr
}

func waitForControllerVolumeResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
	pvName := pvc.Spec.VolumeName
	return wait.PollImmediate(resizePollInterval, duration, func() (bool, error) {
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
