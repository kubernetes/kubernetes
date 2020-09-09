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

package testsuites

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// snapshot CRD api group
const snapshotGroup = "snapshot.storage.k8s.io"

// snapshot CRD api version
const snapshotAPIVersion = "snapshot.storage.k8s.io/v1beta1"

var (
	// SnapshotGVR is GroupVersionResource for volumesnapshots
	SnapshotGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshots"}
	// SnapshotClassGVR is GroupVersionResource for volumesnapshotclasses
	SnapshotClassGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotclasses"}
	// SnapshotContentGVR is GroupVersionResource for volumesnapshotcontents
	SnapshotContentGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotcontents"}
)

type snapshottableTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &snapshottableTestSuite{}

var (
	sDriver SnapshottableTestDriver
	dDriver DynamicPVTestDriver
)

// InitSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
func InitSnapshottableTestSuite() TestSuite {
	return &snapshottableTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "snapshottable",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DynamicSnapshotDelete,
				testpatterns.DynamicSnapshotRetain,
				testpatterns.PreprovisionedSnapshotDelete,
				testpatterns.PreprovisionedSnapshotRetain,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
			FeatureTag: "[Feature:VolumeSnapshotDataSource]",
		},
	}
}

func (s *snapshottableTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *snapshottableTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (s *snapshottableTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	ginkgo.BeforeEach(func() {
		// Check preconditions.
		dInfo := driver.GetDriverInfo()
		ok := false
		sDriver, ok = driver.(SnapshottableTestDriver)
		if !dInfo.Capabilities[CapSnapshotDataSource] || !ok {
			e2eskipper.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
		}
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %q does not support dynamic provisioning - skipping", driver.GetDriverInfo().Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("snapshotting")

	ginkgo.Describe("volume snapshot controller", func() {
		var (
			err           error
			config        *PerTestConfig
			driverCleanup func()
			cleanupSteps  []func()

			cs                  clientset.Interface
			dc                  dynamic.Interface
			pvc                 *v1.PersistentVolumeClaim
			sc                  *storagev1.StorageClass
			claimSize           string
			originalMntTestData string
		)
		init := func() {
			cleanupSteps = make([]func(), 0)
			// init snap class, create a source PV, PVC, Pod
			cs = f.ClientSet
			dc = f.DynamicClient

			// Now do the more expensive test initialization.
			config, driverCleanup = driver.PrepareTest(f)
			cleanupSteps = append(cleanupSteps, driverCleanup)

			var volumeResource *VolumeResource
			cleanupSteps = append(cleanupSteps, func() {
				framework.ExpectNoError(volumeResource.CleanupResource())
			})
			volumeResource = CreateVolumeResource(dDriver, config, pattern, s.GetTestSuiteInfo().SupportedSizeRange)

			pvc = volumeResource.Pvc
			sc = volumeResource.Sc
			claimSize = pvc.Spec.Resources.Requests.Storage().String()

			ginkgo.By("starting a pod to use the claim")
			originalMntTestData = fmt.Sprintf("hello from %s namespace", pvc.GetNamespace())
			command := fmt.Sprintf("echo '%s' > /mnt/test/data", originalMntTestData)

			RunInPodWithVolume(cs, pvc.Namespace, pvc.Name, "pvc-snapshottable-tester", command, config.ClientNodeSelection)

			err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
			framework.ExpectNoError(err)

			ginkgo.By("checking the claim")
			// Get new copy of the claim
			pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			// Get the bound PV
			ginkgo.By("checking the PV")
			_, err = cs.CoreV1().PersistentVolumes().Get(context.TODO(), pvc.Spec.VolumeName, metav1.GetOptions{})
			framework.ExpectNoError(err)
		}

		cleanup := func() {
			// Don't register an AfterEach then a cleanup step because the order
			// of execution will do the AfterEach first then the cleanup step.
			// Also AfterEach cleanup registration is not fine grained enough
			// Adding to the cleanup steps allows you to register cleanup only when it is needed
			// Ideally we could replace this with https://golang.org/pkg/testing/#T.Cleanup

			// Depending on how far the test executed, cleanup accordingly
			// Execute in reverse order, similar to defer stack
			for i := len(cleanupSteps) - 1; i >= 0; i-- {
				err := tryFunc(cleanupSteps[i])
				framework.ExpectNoError(err, "while running cleanup steps")
			}

		}
		ginkgo.BeforeEach(func() {
			init()
		})
		ginkgo.AfterEach(func() {
			cleanup()
		})

		ginkgo.Context("", func() {
			var (
				vs        *unstructured.Unstructured
				vscontent *unstructured.Unstructured
				vsc       *unstructured.Unstructured
			)

			ginkgo.BeforeEach(func() {
				var sr *SnapshotResource
				cleanupSteps = append(cleanupSteps, func() {
					framework.ExpectNoError(sr.CleanupResource())
				})
				sr = CreateSnapshotResource(sDriver, config, pattern, pvc.GetName(), pvc.GetNamespace())
				vs = sr.Vs
				vscontent = sr.Vscontent
				vsc = sr.Vsclass
			})
			ginkgo.It("should check snapshot fields, check restore correctly works after modifying source data, check deletion", func() {
				ginkgo.By("checking the snapshot")
				// Get new copy of the snapshot
				vs, err = dc.Resource(SnapshotGVR).Namespace(vs.GetNamespace()).Get(context.TODO(), vs.GetName(), metav1.GetOptions{})
				framework.ExpectNoError(err)

				// Get the bound snapshotContent
				snapshotStatus := vs.Object["status"].(map[string]interface{})
				snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
				vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
				framework.ExpectNoError(err)

				snapshotContentSpec := vscontent.Object["spec"].(map[string]interface{})
				volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})

				// Check SnapshotContent properties
				ginkgo.By("checking the SnapshotContent")
				// PreprovisionedCreatedSnapshot do not need to set volume snapshot class name
				if pattern.SnapshotType != testpatterns.PreprovisionedCreatedSnapshot {
					framework.ExpectEqual(snapshotContentSpec["volumeSnapshotClassName"], vsc.GetName())
				}
				framework.ExpectEqual(volumeSnapshotRef["name"], vs.GetName())
				framework.ExpectEqual(volumeSnapshotRef["namespace"], vs.GetNamespace())

				ginkgo.By("Modifying source data test")
				var restoredPVC *v1.PersistentVolumeClaim
				var restoredPod *v1.Pod
				modifiedMntTestData := fmt.Sprintf("modified data from %s namespace", pvc.GetNamespace())

				ginkgo.By("modifying the data in the source PVC")

				command := fmt.Sprintf("echo '%s' > /mnt/test/data", modifiedMntTestData)
				RunInPodWithVolume(cs, pvc.Namespace, pvc.Name, "pvc-snapshottable-data-tester", command, config.ClientNodeSelection)

				ginkgo.By("creating a pvc from the snapshot")
				restoredPVC = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        claimSize,
					StorageClassName: &(sc.Name),
				}, config.Framework.Namespace.Name)

				group := "snapshot.storage.k8s.io"

				restoredPVC.Spec.DataSource = &v1.TypedLocalObjectReference{
					APIGroup: &group,
					Kind:     "VolumeSnapshot",
					Name:     vs.GetName(),
				}

				restoredPVC, err = cs.CoreV1().PersistentVolumeClaims(restoredPVC.Namespace).Create(context.TODO(), restoredPVC, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				cleanupSteps = append(cleanupSteps, func() {
					framework.Logf("deleting claim %q/%q", restoredPVC.Namespace, restoredPVC.Name)
					// typically this claim has already been deleted
					err = cs.CoreV1().PersistentVolumeClaims(restoredPVC.Namespace).Delete(context.TODO(), restoredPVC.Name, metav1.DeleteOptions{})
					if err != nil && !apierrors.IsNotFound(err) {
						framework.Failf("Error deleting claim %q. Error: %v", restoredPVC.Name, err)
					}
				})

				ginkgo.By("starting a pod to use the claim")

				restoredPod = StartInPodWithVolume(cs, restoredPVC.Namespace, restoredPVC.Name, "restored-pvc-tester", "sleep 300", config.ClientNodeSelection)
				cleanupSteps = append(cleanupSteps, func() {
					StopPod(cs, restoredPod)
				})
				framework.ExpectNoError(e2epod.WaitForPodRunningInNamespaceSlow(cs, restoredPod.Name, restoredPod.Namespace))

				command = "cat /mnt/test/data"
				actualData, err := utils.PodExec(f, restoredPod, command)
				framework.ExpectNoError(err)
				framework.ExpectEqual(actualData, originalMntTestData)

				ginkgo.By("should delete the VolumeSnapshotContent according to its deletion policy")
				err = DeleteAndWaitSnapshot(dc, vs.GetNamespace(), vs.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err)

				switch pattern.SnapshotDeletionPolicy {
				case testpatterns.DeleteSnapshot:
					ginkgo.By("checking the SnapshotContent has been deleted")
					err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, vscontent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
					framework.ExpectNoError(err)
				case testpatterns.RetainSnapshot:
					ginkgo.By("checking the SnapshotContent has not been deleted")
					err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, vscontent.GetName(), 1*time.Second /* poll */, 30*time.Second /* timeout */)
					framework.ExpectError(err)
				}
			})
		})
	})
}

// WaitForSnapshotReady waits for a VolumeSnapshot to be ready to use or until timeout occurs, whichever comes first.
func WaitForSnapshotReady(c dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for VolumeSnapshot %s to become ready", timeout, snapshotName)

	if successful := utils.WaitUntil(poll, timeout, func() bool {
		snapshot, err := c.Resource(SnapshotGVR).Namespace(ns).Get(context.TODO(), snapshotName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get snapshot %q, retrying in %v. Error: %v", snapshotName, poll, err)
			return false
		}

		status := snapshot.Object["status"]
		if status == nil {
			framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			return false
		}
		value := status.(map[string]interface{})
		if value["readyToUse"] == true {
			framework.Logf("VolumeSnapshot %s found and is ready", snapshotName)
			return true
		}

		framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
		return false
	}); successful {
		return nil
	}

	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}

// DeleteAndWaitSnapshot deletes a VolumeSnapshot and waits for it to be deleted or until timeout occurs, whichever comes first
func DeleteAndWaitSnapshot(dc dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	var err error
	ginkgo.By("deleting the snapshot")
	err = dc.Resource(SnapshotGVR).Namespace(ns).Delete(context.TODO(), snapshotName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}

	ginkgo.By("checking the Snapshot has been deleted")
	err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, ns, snapshotName, poll, timeout)

	return err
}

// SnapshotResource represents a snapshot class, a snapshot and its bound snapshot contents for a specific test case
type SnapshotResource struct {
	Config  *PerTestConfig
	Pattern testpatterns.TestPattern

	Vs        *unstructured.Unstructured
	Vscontent *unstructured.Unstructured
	Vsclass   *unstructured.Unstructured
}

// CreateSnapshotResource creates a snapshot resource for the current test. It knows how to deal with
// different test pattern snapshot provisioning and deletion policy
func CreateSnapshotResource(sDriver SnapshottableTestDriver, config *PerTestConfig, pattern testpatterns.TestPattern, pvcName string, pvcNamespace string) *SnapshotResource {
	var err error
	if pattern.SnapshotType != testpatterns.DynamicCreatedSnapshot && pattern.SnapshotType != testpatterns.PreprovisionedCreatedSnapshot {
		err = fmt.Errorf("SnapshotType must be set to either DynamicCreatedSnapshot or PreprovisionedCreatedSnapshot")
		framework.ExpectNoError(err)
	}
	r := SnapshotResource{
		Config:  config,
		Pattern: pattern,
	}
	dc := r.Config.Framework.DynamicClient

	ginkgo.By("creating a SnapshotClass")
	r.Vsclass = sDriver.GetSnapshotClass(config)
	if r.Vsclass == nil {
		framework.Failf("Failed to get snapshot class based on test config")
	}
	r.Vsclass.Object["deletionPolicy"] = pattern.SnapshotDeletionPolicy.String()

	r.Vsclass, err = dc.Resource(SnapshotClassGVR).Create(context.TODO(), r.Vsclass, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	r.Vsclass, err = dc.Resource(SnapshotClassGVR).Get(context.TODO(), r.Vsclass.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("creating a dynamic VolumeSnapshot")
	// prepare a dynamically provisioned volume snapshot with certain data
	r.Vs = getSnapshot(pvcName, pvcNamespace, r.Vsclass.GetName())

	r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Create(context.TODO(), r.Vs, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = WaitForSnapshotReady(dc, r.Vs.GetNamespace(), r.Vs.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Get(context.TODO(), r.Vs.GetName(), metav1.GetOptions{})

	snapshotStatus := r.Vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	framework.Logf("received snapshotStatus %v", snapshotStatus)
	framework.Logf("snapshotContentName %s", snapshotContentName)
	framework.ExpectNoError(err)

	r.Vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	if pattern.SnapshotType == testpatterns.PreprovisionedCreatedSnapshot {
		// prepare a pre-provisioned VolumeSnapshotContent with certain data
		// Because this could be run with an external CSI driver, we have no way
		// to pre-provision the snapshot as we normally would using their API.
		// We instead dynamically take a snapshot (above step), delete the old snapshot,
		// and create another snapshot using the first snapshot's snapshot handle.

		ginkgo.By("updating the snapshot content deletion policy to retain")
		r.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Retain"

		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), r.Vscontent, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("recording the volume handle and snapshotHandle")
		snapshotHandle := r.Vscontent.Object["status"].(map[string]interface{})["snapshotHandle"].(string)
		framework.Logf("Recording snapshot handle: %s", snapshotHandle)
		csiDriverName := r.Vsclass.Object["driver"].(string)

		// If the deletion policy is retain on vscontent:
		// when vs is deleted vscontent will not be deleted
		// when the vscontent is manually deleted then the underlying snapshot resource will not be deleted.
		// We exploit this to create a snapshot resource from which we can create a preprovisioned snapshot
		ginkgo.By("deleting the snapshot and snapshot content")
		err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Delete(context.TODO(), r.Vs.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot has been deleted")
		err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, r.Vs.GetName(), r.Vs.GetNamespace(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)

		err = dc.Resource(SnapshotContentGVR).Delete(context.TODO(), r.Vscontent.GetName(), metav1.DeleteOptions{})
		if apierrors.IsNotFound(err) {
			err = nil
		}
		framework.ExpectNoError(err)

		ginkgo.By("checking the Snapshot content has been deleted")
		err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, r.Vscontent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot content with the snapshot handle")
		uuid := uuid.NewUUID()

		snapName := getPreProvisionedSnapshotName(uuid)
		snapcontentName := getPreProvisionedSnapshotContentName(uuid)

		r.Vscontent = getPreProvisionedSnapshotContent(snapcontentName, snapName, pvcNamespace, snapshotHandle, pattern.SnapshotDeletionPolicy.String(), csiDriverName)
		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Create(context.TODO(), r.Vscontent, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a snapshot with that snapshot content")
		r.Vs = getPreProvisionedSnapshot(snapName, pvcNamespace, snapcontentName)
		r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Create(context.TODO(), r.Vs, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		err = WaitForSnapshotReady(dc, r.Vs.GetNamespace(), r.Vs.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("getting the snapshot and snapshot content")
		r.Vs, err = dc.Resource(SnapshotGVR).Namespace(r.Vs.GetNamespace()).Get(context.TODO(), r.Vs.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		r.Vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), r.Vscontent.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)
	}
	return &r
}

// CleanupResource cleans up the snapshot resource and ignores not found errors
func (sr *SnapshotResource) CleanupResource() error {
	var err error
	var cleanupErrs []error

	dc := sr.Config.Framework.DynamicClient

	if sr.Vs != nil {
		framework.Logf("deleting snapshot %q/%q", sr.Vs.GetNamespace(), sr.Vs.GetName())

		sr.Vs, err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Get(context.TODO(), sr.Vs.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			snapshotStatus := sr.Vs.Object["status"].(map[string]interface{})
			snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
			framework.Logf("received snapshotStatus %v", snapshotStatus)
			framework.Logf("snapshotContentName %s", snapshotContentName)

			boundVsContent, err := dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
			switch {
			case err == nil:
				if boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
					// The purpose of this block is to prevent physical snapshotContent leaks.
					// We must update the SnapshotContent to have Delete Deletion policy,
					// or else the physical snapshot content will be leaked.
					boundVsContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
					boundVsContent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), boundVsContent, metav1.UpdateOptions{})
					framework.ExpectNoError(err)
				}
				err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, boundVsContent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err)

			case apierrors.IsNotFound(err):
				// the volume snapshot is not bound to snapshot content yet
				err = dc.Resource(SnapshotGVR).Namespace(sr.Vs.GetNamespace()).Delete(context.TODO(), sr.Vs.GetName(), metav1.DeleteOptions{})
				if apierrors.IsNotFound(err) {
					err = nil
				}
				framework.ExpectNoError(err)

				err = utils.WaitForNamespacedGVRDeletion(dc, SnapshotGVR, sr.Vs.GetName(), sr.Vs.GetNamespace(), framework.Poll, framework.SnapshotDeleteTimeout)
				framework.ExpectNoError(err)
			default:
				cleanupErrs = append(cleanupErrs, err)
			}
		case apierrors.IsNotFound(err):
			// Hope that the underlying snapshot content and resource is gone already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}
	if sr.Vscontent != nil {
		framework.Logf("deleting snapshot content %q", sr.Vscontent.GetName())

		sr.Vscontent, err = dc.Resource(SnapshotContentGVR).Get(context.TODO(), sr.Vscontent.GetName(), metav1.GetOptions{})
		switch {
		case err == nil:
			if sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] != "Delete" {
				// The purpose of this block is to prevent physical snapshotContent leaks.
				// We must update the SnapshotContent to have Delete Deletion policy,
				// or else the physical snapshot content will be leaked.
				sr.Vscontent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
				sr.Vscontent, err = dc.Resource(SnapshotContentGVR).Update(context.TODO(), sr.Vscontent, metav1.UpdateOptions{})
				framework.ExpectNoError(err)
			}
			err = dc.Resource(SnapshotContentGVR).Delete(context.TODO(), sr.Vscontent.GetName(), metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				err = nil
			}
			framework.ExpectNoError(err)

			err = utils.WaitForGVRDeletion(dc, SnapshotContentGVR, sr.Vscontent.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
			framework.ExpectNoError(err)
		case apierrors.IsNotFound(err):
			// Hope the underlying physical snapshot resource has been deleted already
		default:
			cleanupErrs = append(cleanupErrs, err)
		}
	}
	if sr.Vsclass != nil {
		framework.Logf("deleting snapshot class %q", sr.Vsclass.GetName())
		// typically this snapshot class has already been deleted
		err = dc.Resource(SnapshotClassGVR).Delete(context.TODO(), sr.Vsclass.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot class %q. Error: %v", sr.Vsclass.GetName(), err)
		}
		err = utils.WaitForGVRDeletion(dc, SnapshotClassGVR, sr.Vsclass.GetName(), framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)
	}
	return utilerrors.NewAggregate(cleanupErrs)
}
