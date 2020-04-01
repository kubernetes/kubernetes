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

package testsuites

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

const (
	resizePollInterval = 2 * time.Second
	// total time to wait for cloudprovider or file system resize to finish
	totalResizeWaitPeriod = 10 * time.Minute
	// time to wait for PVC conditions to sync
	pvcConditionSyncPeriod = 2 * time.Minute
)

type volumeExpandTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeExpandTestSuite{}

// InitVolumeExpandTestSuite returns volumeExpandTestSuite that implements TestSuite interface
func InitVolumeExpandTestSuite() TestSuite {
	return &volumeExpandTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "volume-expand",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.BlockVolModeDynamicPV,
				testpatterns.DefaultFsDynamicPVAllowExpansion,
				testpatterns.BlockVolModeDynamicPVAllowExpansion,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

func (v *volumeExpandTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return v.tsInfo
}

func (v *volumeExpandTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (v *volumeExpandTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config        *PerTestConfig
		driverCleanup func()

		resource *VolumeResource
		pod      *v1.Pod
		pod2     *v1.Pod

		intreeOps   opCounts
		migratedOps opCounts
	}
	var l local

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		if !driver.GetDriverInfo().Capabilities[CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", driver.GetDriverInfo().Name)
		}
		if !driver.GetDriverInfo().Capabilities[CapControllerExpansion] {
			e2eskipper.Skipf("Driver %q does not support volume expansion - skipping", driver.GetDriverInfo().Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("volume-expand")

	init := func() {
		l = local{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName)
		testVolumeSizeRange := v.GetTestSuiteInfo().SupportedSizeRange
		l.resource = CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
	}

	cleanup := func() {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.pod2 != nil {
			ginkgo.By("Deleting pod2")
			err := e2epod.DeletePodWithWait(f.ClientSet, l.pod2)
			errs = append(errs, err)
			l.pod2 = nil
		}

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource())
			l.resource = nil
		}

		errs = append(errs, tryFunc(l.driverCleanup))
		l.driverCleanup = nil
		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		validateMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName, l.intreeOps, l.migratedOps)
	}

	if !pattern.AllowExpansion {
		ginkgo.It("should not allow expansion of pvcs without AllowVolumeExpansion property", func() {
			init()
			defer cleanup()

			var err error
			gomega.Expect(l.resource.Sc.AllowVolumeExpansion).To(gomega.BeNil())
			ginkgo.By("Expanding non-expandable pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			_, err = ExpandPVCSize(l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectError(err, "While updating non-expandable PVC")
		})
	} else {
		ginkgo.It("Verify if offline PVC expansion works", func() {
			init()
			defer cleanup()

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epv.SELinuxLabel,
				NodeSelection: l.config.ClientNodeSelection,
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(f.ClientSet, &podConfig, framework.PodStartTimeout)
			defer func() {
				err = e2epod.DeletePodWithWait(f.ClientSet, l.pod)
				framework.ExpectNoError(err, "while cleaning up pod already deleted in resize test")
			}()
			framework.ExpectNoError(err, "While creating pods for resizing")

			ginkgo.By("Deleting the previously created pod")
			err = e2epod.DeletePodWithWait(f.ClientSet, l.pod)
			framework.ExpectNoError(err, "while deleting pod for resizing")

			// We expand the PVC while no pod is using it to ensure offline expansion
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			newPVC, err := ExpandPVCSize(l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			l.resource.Pvc = newPVC
			gomega.Expect(l.resource.Pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", l.resource.Pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(l.resource.Pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Checking for conditions on pvc")
			npvc, err := WaitForPendingFSResizeCondition(l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "While waiting for pvc to have fs resizing condition")
			l.resource.Pvc = npvc

			ginkgo.By("Creating a new pod with same volume")
			podConfig = e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epv.SELinuxLabel,
				NodeSelection: l.config.ClientNodeSelection,
			}
			l.pod2, err = e2epod.CreateSecPodWithNodeSelection(f.ClientSet, &podConfig, framework.PodStartTimeout)
			defer func() {
				err = e2epod.DeletePodWithWait(f.ClientSet, l.pod2)
				framework.ExpectNoError(err, "while cleaning up pod before exiting resizing test")
			}()
			framework.ExpectNoError(err, "while recreating pod for resizing")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.Pvc, err = WaitForFSResize(l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.Pvc.Status.Conditions
			framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
		})

		ginkgo.It("should resize volume when PVC is edited while pod is using it", func() {
			init()
			defer cleanup()

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epv.SELinuxLabel,
				NodeSelection: l.config.ClientNodeSelection,
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(f.ClientSet, &podConfig, framework.PodStartTimeout)
			defer func() {
				err = e2epod.DeletePodWithWait(f.ClientSet, l.pod)
				framework.ExpectNoError(err, "while cleaning up pod already deleted in resize test")
			}()
			framework.ExpectNoError(err, "While creating pods for resizing")

			// We expand the PVC while no pod is using it to ensure offline expansion
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			newPVC, err := ExpandPVCSize(l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			l.resource.Pvc = newPVC
			gomega.Expect(l.resource.Pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", l.resource.Pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(l.resource.Pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.Pvc, err = WaitForFSResize(l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.Pvc.Status.Conditions
			framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
		})

	}
}

// ExpandPVCSize expands PVC size
func ExpandPVCSize(origPVC *v1.PersistentVolumeClaim, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	pvcName := origPVC.Name
	updatedPVC := origPVC.DeepCopy()

	// Retry the update on error, until we hit a timeout.
	// TODO: Determine whether "retry with timeout" is appropriate here. Maybe we should only retry on version conflict.
	var lastUpdateError error
	waitErr := wait.PollImmediate(resizePollInterval, 30*time.Second, func() (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Get(context.TODO(), pvcName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for resizing: %v", pvcName, err)
		}

		updatedPVC.Spec.Resources.Requests[v1.ResourceStorage] = size
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Update(context.TODO(), updatedPVC, metav1.UpdateOptions{})
		if err != nil {
			framework.Logf("Error updating pvc %s: %v", pvcName, err)
			lastUpdateError = err
			return false, nil
		}
		return true, nil
	})
	if waitErr == wait.ErrWaitTimeout {
		return nil, fmt.Errorf("timed out attempting to update PVC size. last update error: %v", lastUpdateError)
	}
	if waitErr != nil {
		return nil, fmt.Errorf("failed to expand PVC size (check logs for error): %v", waitErr)
	}
	return updatedPVC, nil
}

// WaitForResizingCondition waits for the pvc condition to be PersistentVolumeClaimResizing
func WaitForResizingCondition(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
	waitErr := wait.PollImmediate(resizePollInterval, duration, func() (bool, error) {
		var err error
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status: %v", pvc.Name, err)
		}

		pvcConditions := updatedPVC.Status.Conditions
		if len(pvcConditions) > 0 {
			if pvcConditions[0].Type == v1.PersistentVolumeClaimResizing {
				return true, nil
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return fmt.Errorf("error waiting for pvc %q to have resize status: %v", pvc.Name, waitErr)
	}
	return nil
}

// WaitForControllerVolumeResize waits for the controller resize to be finished
func WaitForControllerVolumeResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
	pvName := pvc.Spec.VolumeName
	waitErr := wait.PollImmediate(resizePollInterval, duration, func() (bool, error) {
		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]

		pv, err := c.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
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
	if waitErr != nil {
		return fmt.Errorf("error while waiting for controller resize to finish: %v", waitErr)
	}
	return nil
}

// WaitForPendingFSResizeCondition waits for pvc to have resize condition
func WaitForPendingFSResizeCondition(pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	var updatedPVC *v1.PersistentVolumeClaim
	waitErr := wait.PollImmediate(resizePollInterval, pvcConditionSyncPeriod, func() (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status : %v", pvc.Name, err)
		}

		inProgressConditions := updatedPVC.Status.Conditions
		// if there are no PVC conditions that means no node expansion is necessary
		if len(inProgressConditions) == 0 {
			return true, nil
		}
		conditionType := inProgressConditions[0].Type
		if conditionType == v1.PersistentVolumeClaimFileSystemResizePending {
			return true, nil
		}
		return false, nil
	})
	if waitErr != nil {
		return nil, fmt.Errorf("error waiting for pvc %q to have filesystem resize status: %v", pvc.Name, waitErr)
	}
	return updatedPVC, nil
}

// WaitForFSResize waits for the filesystem in the pv to be resized
func WaitForFSResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	var updatedPVC *v1.PersistentVolumeClaim
	waitErr := wait.PollImmediate(resizePollInterval, totalResizeWaitPeriod, func() (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})

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
	if waitErr != nil {
		return nil, fmt.Errorf("error waiting for pvc %q filesystem resize to finish: %v", pvc.Name, waitErr)
	}
	return updatedPVC, nil
}
