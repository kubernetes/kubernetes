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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	resizePollInterval = 2 * time.Second
	// total time to wait for cloudprovider or file system resize to finish
	totalResizeWaitPeriod = 10 * time.Minute

	// resizedPodStartupTimeout defines time we should wait for pod that uses offline
	// resized volume to startup. This time is higher than default PodStartTimeout because
	// typically time to detach and then attach a volume is amortized in this time duration.
	resizedPodStartupTimeout = 10 * time.Minute

	// time to wait for PVC conditions to sync
	pvcConditionSyncPeriod = 2 * time.Minute
)

type volumeExpandTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomVolumeExpandTestSuite returns volumeExpandTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeExpandTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeExpandTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volume-expand",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Gi",
			},
		},
	}
}

// InitVolumeExpandTestSuite returns volumeExpandTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeExpandTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
		storageframework.BlockVolModeDynamicPV,
		storageframework.DefaultFsDynamicPVAllowExpansion,
		storageframework.BlockVolModeDynamicPVAllowExpansion,
		storageframework.NtfsDynamicPV,
		storageframework.NtfsDynamicPVAllowExpansion,
	}
	return InitCustomVolumeExpandTestSuite(patterns)
}

func (v *volumeExpandTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return v.tsInfo
}

func (v *volumeExpandTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	if !driver.GetDriverInfo().Capabilities[storageframework.CapControllerExpansion] {
		e2eskipper.Skipf("Driver %q does not support volume expansion - skipping", driver.GetDriverInfo().Name)
	}
	// Check preconditions.
	if !driver.GetDriverInfo().Capabilities[storageframework.CapBlock] && pattern.VolMode == v1.PersistentVolumeBlock {
		e2eskipper.Skipf("Driver %q does not support block volume mode - skipping", driver.GetDriverInfo().Name)
	}
}

func (v *volumeExpandTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		resource *storageframework.VolumeResource
		pod      *v1.Pod
		pod2     *v1.Pod

		migrationCheck *migrationOpCheck
	}
	var l local

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volume-expand", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	driverSizeRange := driver.GetDriverInfo().SupportedSizeRange
	expandSize := resource.MustParse("1Gi")
	if driverSizeRange.Step != "" {
		expandSize = resource.MustParse(driverSizeRange.Step)
	}

	init := func(ctx context.Context) {
		l = local{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)
		l.migrationCheck = newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), driver.GetDriverInfo().InTreePluginName)
		testVolumeSizeRange := v.GetTestSuiteInfo().SupportedSizeRange
		l.resource = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
	}

	cleanup := func(ctx context.Context) {
		var errs []error
		if l.pod != nil {
			ginkgo.By("Deleting pod")
			err := e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			errs = append(errs, err)
			l.pod = nil
		}

		if l.pod2 != nil {
			ginkgo.By("Deleting pod2")
			err := e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod2)
			errs = append(errs, err)
			l.pod2 = nil
		}

		if l.resource != nil {
			errs = append(errs, l.resource.CleanupResource(ctx))
			l.resource = nil
		}

		framework.ExpectNoError(errors.NewAggregate(errs), "while cleaning up resource")
		l.migrationCheck.validateMigrationVolumeOpCounts(ctx)
	}

	if !pattern.AllowExpansion {
		ginkgo.It("should not allow expansion of pvcs without AllowVolumeExpansion property", func(ctx context.Context) {
			init(ctx)
			ginkgo.DeferCleanup(cleanup)

			var err error
			// create Pod with pvc
			ginkgo.By("Creating a pod with PVC")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epod.GetLinuxLabel(),
				NodeSelection: l.config.ClientNodeSelection,
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "While creating pods for expanding")

			// Waiting for pod to run
			ginkgo.By("Waiting for pod to run")
			err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, f.ClientSet, l.pod.Name, l.pod.Namespace, f.Timeouts.PodStart)
			framework.ExpectNoError(err)

			gomega.Expect(l.resource.Sc.AllowVolumeExpansion).NotTo(gomega.BeNil())
			allowVolumeExpansion := *l.resource.Sc.AllowVolumeExpansion
			gomega.Expect(allowVolumeExpansion).To(gomega.BeFalseBecause("expected AllowVolumeExpansion value to be false"))
			ginkgo.By("Expanding non-expandable pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(expandSize)
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			_, err = ExpandPVCSizeToError(ctx, l.resource.Pvc, newSize, f.ClientSet)
			gomega.Expect(err).To(gomega.MatchError(apierrors.IsForbidden, "While updating non-expandable PVC"))
		})
	} else {
		ginkgo.It("Verify if offline PVC expansion works", func(ctx context.Context) {
			init(ctx)
			ginkgo.DeferCleanup(cleanup)

			if !driver.GetDriverInfo().Capabilities[storageframework.CapOfflineExpansion] {
				e2eskipper.Skipf("Driver %q does not support offline volume expansion - skipping", driver.GetDriverInfo().Name)
			}

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epod.GetLinuxLabel(),
				NodeSelection: l.config.ClientNodeSelection,
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "While creating pods for resizing")

			ginkgo.By("Deleting the previously created pod")
			err = e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "while deleting pod for resizing")

			// We expand the PVC while no pod is using it to ensure offline expansion
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(expandSize)
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			newPVC, err := ExpandPVCSize(ctx, l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			l.resource.Pvc = newPVC
			gomega.Expect(l.resource.Pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", l.resource.Pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(ctx, l.resource.Pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Checking for conditions on pvc")
			npvc, err := WaitForPendingFSResizeCondition(ctx, l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "While waiting for pvc to have fs resizing condition")
			l.resource.Pvc = npvc

			ginkgo.By("Verifying allocatedResources on PVC")
			err = verifyOfflineAllocatedResources(l.resource.Pvc, pvcSize)
			framework.ExpectNoError(err, "While verifying allocatedResources on PVC")

			ginkgo.By("Creating a new pod with same volume")
			podConfig = e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epod.GetLinuxLabel(),
				NodeSelection: l.config.ClientNodeSelection,
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod2, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, resizedPodStartupTimeout)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod2)
			framework.ExpectNoError(err, "while recreating pod for resizing")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.Pvc, err = WaitForFSResize(ctx, l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.Pvc.Status.Conditions
			gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
			err = VerifyRecoveryRelatedFields(l.resource.Pvc)
			framework.ExpectNoError(err, "while verifying recovery related fields")
		})

		ginkgo.It("should resize volume when PVC is edited while pod is using it", func(ctx context.Context) {
			init(ctx)
			ginkgo.DeferCleanup(cleanup)

			if !driver.GetDriverInfo().Capabilities[storageframework.CapOnlineExpansion] {
				e2eskipper.Skipf("Driver %q does not support online volume expansion - skipping", driver.GetDriverInfo().Name)
			}

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epod.GetLinuxLabel(),
				NodeSelection: l.config.ClientNodeSelection,
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "While creating pods for resizing")

			// We expand the PVC while l.pod is using it for online expansion.
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(expandSize)
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			newPVC, err := ExpandPVCSize(ctx, l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			l.resource.Pvc = newPVC
			gomega.Expect(l.resource.Pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", l.resource.Pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(ctx, l.resource.Pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.Pvc, err = WaitForFSResize(ctx, l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.Pvc.Status.Conditions
			gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")

			err = VerifyRecoveryRelatedFields(l.resource.Pvc)
			framework.ExpectNoError(err, "while verifying recovery related fields")
		})

		ginkgo.It("should resize volume when PVC is edited and the pod is re-created on the same node after controller resize is finished", func(ctx context.Context) {
			init(ctx)
			ginkgo.DeferCleanup(cleanup)

			if !driver.GetDriverInfo().Capabilities[storageframework.CapOnlineExpansion] {
				e2eskipper.Skipf("Driver %q does not support online volume expansion - skipping", driver.GetDriverInfo().Name)
			}

			var err error
			ginkgo.By("Creating a pod with dynamically provisioned volume")
			podConfig := e2epod.Config{
				NS:            f.Namespace.Name,
				PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel:  e2epod.GetLinuxLabel(),
				NodeSelection: l.config.ClientNodeSelection,
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "While creating pods for resizing")

			// We expand the PVC while l.pod is using it for online expansion.
			ginkgo.By("Expanding current pvc")
			currentPvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			newSize := currentPvcSize.DeepCopy()
			newSize.Add(expandSize)
			framework.Logf("currentPvcSize %v, newSize %v", currentPvcSize, newSize)
			newPVC, err := ExpandPVCSize(ctx, l.resource.Pvc, newSize, f.ClientSet)
			framework.ExpectNoError(err, "While updating pvc for more size")
			l.resource.Pvc = newPVC
			gomega.Expect(l.resource.Pvc).NotTo(gomega.BeNil())

			pvcSize := l.resource.Pvc.Spec.Resources.Requests[v1.ResourceStorage]
			if pvcSize.Cmp(newSize) != 0 {
				framework.Failf("error updating pvc size %q", l.resource.Pvc.Name)
			}

			ginkgo.By("Waiting for cloudprovider resize to finish")
			err = WaitForControllerVolumeResize(ctx, l.resource.Pvc, f.ClientSet, totalResizeWaitPeriod)
			framework.ExpectNoError(err, "While waiting for pvc resize to finish")

			ginkgo.By("Deleting the pod")
			nodeName := l.pod.Spec.NodeName
			err = e2epod.DeletePodWithWait(ctx, f.ClientSet, l.pod)
			framework.ExpectNoError(err, "while deleting pod for resizing")
			l.pod = nil

			ginkgo.By("Creating a new pod with same volume on the same node")
			podConfig = e2epod.Config{
				NS:           f.Namespace.Name,
				PVCs:         []*v1.PersistentVolumeClaim{l.resource.Pvc},
				SeLinuxLabel: e2epod.GetLinuxLabel(),
				// The reason we use this node selection is because we do not want pod to move to different node when pod is deleted.
				// Keeping pod on same node reproduces the scenario that volume might already be mounted when resize is attempted.
				// We should consider adding a unit test that exercises this better.
				NodeSelection: e2epod.NodeSelection{Name: nodeName},
				ImageID:       e2epod.GetDefaultTestImageID(),
			}
			l.pod2, err = e2epod.CreateSecPodWithNodeSelection(ctx, f.ClientSet, &podConfig, f.Timeouts.PodStart)
			ginkgo.DeferCleanup(e2epod.DeletePodWithWait, f.ClientSet, l.pod2)
			framework.ExpectNoError(err, "while creating pod for resizing")

			ginkgo.By("Waiting for file system resize to finish")
			l.resource.Pvc, err = WaitForFSResize(ctx, l.resource.Pvc, f.ClientSet)
			framework.ExpectNoError(err, "while waiting for fs resize to finish")

			pvcConditions := l.resource.Pvc.Status.Conditions
			gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")

			err = VerifyRecoveryRelatedFields(l.resource.Pvc)
			framework.ExpectNoError(err, "while verifying recovery related fields")
		})
	}
}

// ExpandPVCSize expands PVC size
func ExpandPVCSize(ctx context.Context, origPVC *v1.PersistentVolumeClaim, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	pvcName := origPVC.Name
	updatedPVC := origPVC.DeepCopy()

	// Retry the update on error, until we hit a timeout.
	// TODO: Determine whether "retry with timeout" is appropriate here. Maybe we should only retry on version conflict.
	var lastUpdateError error
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, 30*time.Second, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Get(pollContext, pvcName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for resizing: %w", pvcName, err)
		}

		updatedPVC.Spec.Resources.Requests[v1.ResourceStorage] = size
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Update(pollContext, updatedPVC, metav1.UpdateOptions{})
		if err != nil {
			framework.Logf("Error updating pvc %s: %v", pvcName, err)
			lastUpdateError = err
			return false, nil
		}
		return true, nil
	})
	if wait.Interrupted(waitErr) {
		return nil, fmt.Errorf("timed out attempting to update PVC size. last update error: %w", lastUpdateError)
	}
	if waitErr != nil {
		return nil, fmt.Errorf("failed to expand PVC size (check logs for error): %v", waitErr)
	}
	return updatedPVC, nil
}

func ExpandPVCSizeToError(ctx context.Context, origPVC *v1.PersistentVolumeClaim, size resource.Quantity, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	pvcName := origPVC.Name
	updatedPVC := origPVC.DeepCopy()

	var lastUpdateError error

	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, 30*time.Second, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Get(pollContext, pvcName, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for resizing: %w", pvcName, err)
		}

		updatedPVC.Spec.Resources.Requests[v1.ResourceStorage] = size
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(origPVC.Namespace).Update(pollContext, updatedPVC, metav1.UpdateOptions{})
		if err == nil {
			return false, fmt.Errorf("pvc %s should not be allowed to be updated", pvcName)
		} else {
			lastUpdateError = err
			if apierrors.IsForbidden(err) {
				return true, nil
			}
			framework.Logf("Error updating pvc %s: %v", pvcName, err)
			return false, nil
		}
	})

	if wait.Interrupted(waitErr) {
		return nil, fmt.Errorf("timed out attempting to update PVC size. last update error: %w", lastUpdateError)
	}
	if waitErr != nil {
		return nil, fmt.Errorf("failed to expand PVC size (check logs for error): %w", waitErr)
	}
	return updatedPVC, lastUpdateError
}

// WaitForResizingCondition waits for the pvc condition to be PersistentVolumeClaimResizing
func WaitForResizingCondition(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface, duration time.Duration) error {
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, duration, true, func(ctx context.Context) (bool, error) {
		var err error
		updatedPVC, err := c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status: %w", pvc.Name, err)
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
func WaitForControllerVolumeResize(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface, timeout time.Duration) error {
	pvName := pvc.Spec.VolumeName
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, timeout, true, func(ctx context.Context) (bool, error) {
		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]

		pv, err := c.CoreV1().PersistentVolumes().Get(ctx, pvName, metav1.GetOptions{})
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
func WaitForPendingFSResizeCondition(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	var updatedPVC *v1.PersistentVolumeClaim
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, pvcConditionSyncPeriod, true, func(ctx context.Context) (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(ctx, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status : %w", pvc.Name, err)
		}

		inProgressConditions := updatedPVC.Status.Conditions
		// if there are no PVC conditions that means no node expansion is necessary
		if len(inProgressConditions) == 0 {
			return true, nil
		}
		for _, condition := range inProgressConditions {
			conditionType := condition.Type
			if conditionType == v1.PersistentVolumeClaimFileSystemResizePending {
				return true, nil
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return nil, fmt.Errorf("error waiting for pvc %q to have filesystem resize status: %v", pvc.Name, waitErr)
	}
	return updatedPVC, nil
}

// WaitForFSResize waits for the filesystem in the pv to be resized
func WaitForFSResize(ctx context.Context, pvc *v1.PersistentVolumeClaim, c clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	var updatedPVC *v1.PersistentVolumeClaim
	waitErr := wait.PollUntilContextTimeout(ctx, resizePollInterval, totalResizeWaitPeriod, true, func(pollContext context.Context) (bool, error) {
		var err error
		updatedPVC, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pollContext, pvc.Name, metav1.GetOptions{})

		if err != nil {
			return false, fmt.Errorf("error fetching pvc %q for checking for resize status : %w", pvc.Name, err)
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

func verifyOfflineAllocatedResources(pvc *v1.PersistentVolumeClaim, allocatedSize resource.Quantity) error {
	actualResizeStatus := pvc.Status.AllocatedResourceStatuses[v1.ResourceStorage]
	if !checkControllerExpansionCompleted(pvc) {
		return fmt.Errorf("pvc %q had %s resize status, expected %s", pvc.Name, actualResizeStatus, v1.PersistentVolumeClaimNodeResizePending)
	}

	actualAllocatedSize := pvc.Status.AllocatedResources.Storage()
	if actualAllocatedSize.Cmp(allocatedSize) < 0 {
		return fmt.Errorf("pvc %q had %s allocated size, expected %s", pvc.Name, actualAllocatedSize.String(), allocatedSize.String())
	}
	return nil
}

func checkControllerExpansionCompleted(pvc *v1.PersistentVolumeClaim) bool {
	resizeStatus := pvc.Status.AllocatedResourceStatuses[v1.ResourceStorage]
	// if resizeStatus is empty that means no node expansion is required but still controller expansion is completed
	return (resizeStatus == "" || resizeStatus == v1.PersistentVolumeClaimNodeResizePending)
}

func VerifyRecoveryRelatedFields(pvc *v1.PersistentVolumeClaim) error {
	resizeStatus := pvc.Status.AllocatedResourceStatuses[v1.ResourceStorage]
	if resizeStatus != "" {
		return fmt.Errorf("pvc %q had %s resize status, expected none", pvc.Name, resizeStatus)
	}

	allocatedSize := pvc.Status.AllocatedResources[v1.ResourceStorage]
	requestedSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
	// at this point allocatedSize should be greater than pvc resource request
	if allocatedSize.Cmp(requestedSize) < 0 {
		return fmt.Errorf("pvc %q had %s allocated size, expected %s", pvc.Name, allocatedSize.String(), requestedSize.String())
	}
	return nil
}
