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
	"fmt"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

const (
	resizePollInterval = 2 * time.Second
	// total time to wait for cloudprovider or file system resize to finish
	totalResizeWaitPeriod = 10 * time.Minute
)

type volumeExpandTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeExpandTestSuite{}

// InitVolumeExpandTestSuite returns volumeExpandTestSuite that implements TestSuite interface
func InitVolumeExpandTestSuite() TestSuite {
	return &volumeExpandTestSuite{
		tsInfo: TestSuiteInfo{
			name: "volume-expand",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
				testpatterns.BlockVolModeDynamicPV,
				testpatterns.DefaultFsDynamicPVAllowExpansion,
				testpatterns.BlockVolModeDynamicPVAllowExpansion,
			},
		},
	}
}

func (v *volumeExpandTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return v.tsInfo
}

func (v *volumeExpandTestSuite) skipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (v *volumeExpandTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		resource *genericVolumeTestResource
		pod      *v1.Pod
		pod2     *v1.Pod

		intreeOps   opCounts
		migratedOps opCounts
	}
	var l local

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		if !driver.GetDriverInfo().Capabilities[CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
			framework.Skipf("Driver %q does not support block volume mode - skipping", driver.GetDriverInfo().Name)
		}
		if !driver.GetDriverInfo().Capabilities[CapControllerExpansion] {
			framework.Skipf("Driver %q does not support volume expansion - skipping", driver.GetDriverInfo().Name)
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
		l.config, l.testCleanup = driver.PrepareTest(f)
		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName)
		l.resource = createGenericVolumeTestResource(driver, l.config, pattern)
	}

	cleanup := func() {
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := framework.DeletePodWithWait(f, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "while deleting pod")
			l.pod = nil
		}

		if l.pod2 != nil {
			ginkgo.By("Deleting pod2")
			err := framework.DeletePodWithWait(f, f.ClientSet, l.pod2)
			framework.ExpectNoError(err, "while deleting pod2")
			l.pod2 = nil
		}

		if l.resource != nil {
			l.resource.cleanupResource()
			l.resource = nil
		}

		if l.testCleanup != nil {
			l.testCleanup()
			l.testCleanup = nil
		}

		validateMigrationVolumeOpCounts(f.ClientSet, driver.GetDriverInfo().InTreePluginName, l.intreeOps, l.migratedOps)
	}

	if !pattern.AllowExpansion {
		ginkgo.It("should not allow expansion of pvcs without AllowVolumeExpansion property", func() {
			init()
			defer cleanup()

			var err error
			gomega.Expect(l.resource.sc.AllowVolumeExpansion).To(gomega.BeNil())
			ginkgo.By("Expanding non-expandable pvc")
			currentPvcSize := l.resource.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			e2elog.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			l.resource.pvc, err = ExpandPVCSize(l.resource.pvc, newSize, f.ClientSet)
			framework.ExpectError(err, "While updating non-expandable PVC")
		})
	} else {
		ginkgo.It("Verify if offline PVC expansion works", func() {
			init()
			defer cleanup()

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			l.pod, err = framework.CreateSecPodWithNodeSelection(f.ClientSet, f.Namespace.Name, []*v1.PersistentVolumeClaim{l.resource.pvc}, nil, false, "", false, false, framework.SELinuxLabel, nil, framework.NodeSelection{Name: l.config.ClientNodeName}, framework.PodStartTimeout)
			defer func() {
				err = framework.DeletePodWithWait(f, f.ClientSet, l.pod)
				framework.ExpectNoError(err, "while cleaning up pod already deleted in resize test")
			}()
			framework.ExpectNoError(err, "While creating pods for resizing")

			ginkgo.By("Deleting the previously created pod")
			err = framework.DeletePodWithWait(f, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "while deleting pod for resizing")

			// We expand the PVC while no pod is using it to ensure offline expansion
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			e2elog.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			l.resource.pvc, err = ExpandPVCSize(l.resource.pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			gomega.Expect(l.resource.pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				e2elog.Failf("error updating pvc size %q", l.resource.pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(l.resource.pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Checking for conditions on pvc")
			l.resource.pvc, err = f.ClientSet.CoreV1().PersistentVolumeClaims(f.Namespace.Name).Get(l.resource.pvc.Name, metav1.GetOptions{})
			framework.ExpectNoError(err, "While fetching pvc after controller resize")

			if pattern.VolMode == v1.PersistentVolumeBlock || !l.resource.driver.GetDriverInfo().Capabilities[CapNodeExpansion] {
				pvcConditions := l.resource.pvc.Status.Conditions
				framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
			} else {
				inProgressConditions := l.resource.pvc.Status.Conditions
				framework.ExpectEqual(len(inProgressConditions), 1, "pvc must have file system resize pending condition")
				framework.ExpectEqual(inProgressConditions[0].Type, v1.PersistentVolumeClaimFileSystemResizePending, "pvc must have fs resizing condition")
			}

			ginkgo.By("Creating a new pod with same volume")
			l.pod2, err = framework.CreateSecPodWithNodeSelection(f.ClientSet, f.Namespace.Name, []*v1.PersistentVolumeClaim{l.resource.pvc}, nil, false, "", false, false, framework.SELinuxLabel, nil, framework.NodeSelection{Name: l.config.ClientNodeName}, framework.PodStartTimeout)
			defer func() {
				err = framework.DeletePodWithWait(f, f.ClientSet, l.pod2)
				framework.ExpectNoError(err, "while cleaning up pod before exiting resizing test")
			}()
			framework.ExpectNoError(err, "while recreating pod for resizing")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.pvc, err = WaitForFSResize(l.resource.pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.pvc.Status.Conditions
			framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
		})

		ginkgo.It("should resize volume when PVC is edited while pod is using it", func() {
			init()
			defer cleanup()

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			l.pod, err = framework.CreateSecPodWithNodeSelection(f.ClientSet, f.Namespace.Name, []*v1.PersistentVolumeClaim{l.resource.pvc}, nil, false, "", false, false, framework.SELinuxLabel, nil, framework.NodeSelection{Name: l.config.ClientNodeName}, framework.PodStartTimeout)
			defer func() {
				err = framework.DeletePodWithWait(f, f.ClientSet, l.pod)
				framework.ExpectNoError(err, "while cleaning up pod already deleted in resize test")
			}()
			framework.ExpectNoError(err, "While creating pods for resizing")

			// We expand the PVC while no pod is using it to ensure offline expansion
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(resource.MustParse("1Gi"))
			e2elog.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			l.resource.pvc, err = ExpandPVCSize(l.resource.pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			gomega.Expect(l.resource.pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				e2elog.Failf("error updating pvc size %q", l.resource.pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(l.resource.pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.pvc, err = WaitForFSResize(l.resource.pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.pvc.Status.Conditions
			framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
		})

	}
}

// ExpandPVCSize expands PVC size
func ExpandPVCSize(origPVC *v1.PersistentVolumeClaim, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
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

// WaitForResizingCondition waits for the pvc condition to be PersistentVolumeClaimResizing
func WaitForResizingCondition(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
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

// WaitForControllerVolumeResize waits for the controller resize to be finished
func WaitForControllerVolumeResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
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

// WaitForFSResize waits for the filesystem in the pv to be resized
func WaitForFSResize(pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
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
