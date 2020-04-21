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
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
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
				testpatterns.DynamicSnapshot,
			},
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
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
		framework.ExpectEqual(pattern.SnapshotType, testpatterns.DynamicCreatedSnapshot)
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

	init := func(l *snapshottableLocal) {
		l.cs = f.ClientSet
		l.dc = f.DynamicClient

		// Now do the more expensive test initialization.
		config, driverCleanup := driver.PrepareTest(f)
		l.config = config
		l.driverCleanup = driverCleanup

		l.sc = dDriver.GetDynamicProvisionStorageClass(config, "")
		if l.sc == nil {
			framework.Failf("This driver should support dynamic provisioning")
		}
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
		l.pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
		}, config.Framework.Namespace.Name)

		framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", l.sc, l.pvc)

		ginkgo.By("creating a StorageClass " + l.sc.Name)
		l.sc, err = l.cs.StorageV1().StorageClasses().Create(context.TODO(), l.sc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		l.cleanupSteps = append(l.cleanupSteps, func() {
			framework.Logf("deleting storage class %s", l.sc.Name)
			framework.ExpectNoError(l.cs.StorageV1().StorageClasses().Delete(context.TODO(), l.sc.Name, metav1.DeleteOptions{}))
		})

		ginkgo.By("creating a claim")
		l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.pvc.Namespace).Create(context.TODO(), l.pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		l.cleanupSteps = append(l.cleanupSteps, func() {
			framework.Logf("deleting claim %q/%q", l.pvc.Namespace, l.pvc.Name)
			// typically this claim has already been deleted
			err = l.cs.CoreV1().PersistentVolumeClaims(l.pvc.Namespace).Delete(context.TODO(), l.pvc.Name, metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting claim %q. Error: %v", l.pvc.Name, err)
			}
		})

		ginkgo.By("starting a pod to use the claim")
		command := "echo 'hello world' > /mnt/test/data"
		l.pod = StartInPodWithVolume(l.cs, l.pvc.Namespace, l.pvc.Name, "pvc-snapshottable-tester", command, config.ClientNodeSelection)
		l.cleanupSteps = append(l.cleanupSteps, func() {
			StopPod(l.cs, l.pod)
		})

		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, l.cs, l.pvc.Namespace, l.pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the claim")
		// Get new copy of the claim
		l.pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.pvc.Namespace).Get(context.TODO(), l.pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound PV
		_, err = l.cs.CoreV1().PersistentVolumes().Get(context.TODO(), l.pvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
	}
	cleanup := func(l *snapshottableLocal) {
		// Depending on how far the test executed, cleanup accordingly
		// Execute in reverse order, similar to defer stack
		for i := len(l.cleanupSteps) - 1; i >= 0; i-- {
			err := tryFunc(l.cleanupSteps[i])
			framework.ExpectNoError(err, "while running cleanup steps")
		}

		// All tests will require these driver cleanup tests
		err := tryFunc(l.driverCleanup)
		l.driverCleanup = nil
		framework.ExpectNoError(err, "while cleaning up driver")
	}

	ginkgo.It("should create snapshot with delete policy [Feature:VolumeSnapshotDataSource]", func() {
		l := &snapshottableLocal{}

		init(l)
		defer cleanup(l)

		TestSnapshottable(l, SnapshotClassTest{
			DeletionPolicy: "Delete",
		})
		TestSnapshotDeleted(l, SnapshotClassTest{
			DeletionPolicy: "Delete",
		})

	})

	ginkgo.It("should not delete snapshot with retain policy [Feature:VolumeSnapshotDataSource]", func() {
		l := &snapshottableLocal{}

		init(l)
		defer cleanup(l)

		TestSnapshottable(l, SnapshotClassTest{
			DeletionPolicy: "Retain",
		})
		TestSnapshotDeleted(l, SnapshotClassTest{
			DeletionPolicy: "Retain",
		})
	})
}

// snapshottableLocal is used to keep the current state of a snapshottable
// test, associated objects, and cleanup steps.
type snapshottableLocal struct {
	config        *PerTestConfig
	driverCleanup func()
	cleanupSteps  []func()

	cs        clientset.Interface
	dc        dynamic.Interface
	pvc       *v1.PersistentVolumeClaim
	sc        *storagev1.StorageClass
	pod       *v1.Pod
	vsc       *unstructured.Unstructured
	vs        *unstructured.Unstructured
	vscontent *unstructured.Unstructured
}

// SnapshotClassTest represents parameters to be used by snapshot tests.
// Not all parameters are used by all tests.
type SnapshotClassTest struct {
	DeletionPolicy string
}

// TestSnapshottable tests volume snapshots based on a given SnapshotClassTest
func TestSnapshottable(l *snapshottableLocal, sct SnapshotClassTest) {
	var err error

	ginkgo.By("creating a SnapshotClass")
	l.vsc = sDriver.GetSnapshotClass(l.config)
	if l.vsc == nil {
		framework.Failf("Failed to get snapshot class based on test config")
	}
	l.vsc.Object["deletionPolicy"] = sct.DeletionPolicy
	l.vsc, err = l.dc.Resource(SnapshotClassGVR).Create(context.TODO(), l.vsc, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting SnapshotClass %s", l.vsc.GetName())
		l.dc.Resource(SnapshotClassGVR).Delete(context.TODO(), l.vsc.GetName(), metav1.DeleteOptions{})
	}()
	l.vsc, err = l.dc.Resource(SnapshotClassGVR).Namespace(l.vsc.GetNamespace()).Get(context.TODO(), l.vsc.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("creating a snapshot")
	l.vs = getSnapshot(l.pvc.Name, l.pvc.Namespace, l.vsc.GetName())

	l.vs, err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Create(context.TODO(), l.vs, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting snapshot %q/%q", l.vs.GetNamespace(), l.vs.GetName())
		// typically this snapshot has already been deleted
		err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Delete(context.TODO(), l.vs.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot %q. Error: %v", l.pvc.Name, err)
		}
	}()
	err = WaitForSnapshotReady(l.dc, l.vs.GetNamespace(), l.vs.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("checking the snapshot")
	// Get new copy of the snapshot
	l.vs, err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Get(context.TODO(), l.vs.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	// Get the bound snapshotContent
	snapshotStatus := l.vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	l.vscontent, err = l.dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	snapshotContentSpec := l.vscontent.Object["spec"].(map[string]interface{})
	volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})

	// Check SnapshotContent properties
	ginkgo.By("checking the SnapshotContent")
	framework.ExpectEqual(snapshotContentSpec["volumeSnapshotClassName"], l.vsc.GetName())
	framework.ExpectEqual(volumeSnapshotRef["name"], l.vs.GetName())
	framework.ExpectEqual(volumeSnapshotRef["namespace"], l.vs.GetNamespace())
}

// TestSnapshotDeleted tests the results of deleting a VolumeSnapshot
// depending on the deletion policy currently set.
func TestSnapshotDeleted(l *snapshottableLocal, sct SnapshotClassTest) {
	var err error

	ginkgo.By("creating a SnapshotClass")
	l.vsc = sDriver.GetSnapshotClass(l.config)
	if l.vsc == nil {
		framework.Failf("Failed to get snapshot class based on test config")
	}
	l.vsc.Object["deletionPolicy"] = sct.DeletionPolicy
	l.vsc, err = l.dc.Resource(SnapshotClassGVR).Create(context.TODO(), l.vsc, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting SnapshotClass %s", l.vsc.GetName())
		l.dc.Resource(SnapshotClassGVR).Delete(context.TODO(), l.vsc.GetName(), metav1.DeleteOptions{})
	}()
	l.vsc, err = l.dc.Resource(SnapshotClassGVR).Namespace(l.vsc.GetNamespace()).Get(context.TODO(), l.vsc.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("creating a snapshot to delete")
	l.vs = getSnapshot(l.pvc.Name, l.pvc.Namespace, l.vsc.GetName())

	l.vs, err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Create(context.TODO(), l.vs, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting snapshot %q/%q", l.vs.GetNamespace(), l.vs.GetName())
		// typically this snapshot has already been deleted
		err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Delete(context.TODO(), l.vs.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot %q. Error: %v", l.pvc.Name, err)
		}
	}()
	err = WaitForSnapshotReady(l.dc, l.vs.GetNamespace(), l.vs.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	ginkgo.By("get the snapshot to delete")
	l.vs, err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Get(context.TODO(), l.vs.GetName(), metav1.GetOptions{})
	framework.ExpectNoError(err)
	snapshotStatus := l.vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	framework.Logf("received snapshotStatus %v", snapshotStatus)
	framework.Logf("snapshotContentName %s", snapshotContentName)
	l.vscontent, err = l.dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("deleting the snapshot")
	err = l.dc.Resource(SnapshotGVR).Namespace(l.vs.GetNamespace()).Delete(context.TODO(), l.vs.GetName(), metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		framework.Failf("Error deleting snapshot %s in namespace %s. Error: %v", l.vs.GetName(), l.vs.GetNamespace(), err)
	}

	ginkgo.By("checking the Snapshot has been deleted")
	err = utils.WaitForGVRDeletion(l.dc, SnapshotGVR, l.sc.Name, framework.Poll, framework.SnapshotDeleteTimeout)
	framework.ExpectNoError(err)

	if sct.DeletionPolicy == "Delete" {
		ginkgo.By("checking the SnapshotContent has been deleted")
		err = utils.WaitForGVRDeletion(l.dc, SnapshotContentGVR, snapshotContentName, framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)
	} else if sct.DeletionPolicy == "Retain" {
		ginkgo.By("checking the SnapshotContent has not been deleted")
		err = utils.WaitForGVRDeletion(l.dc, SnapshotContentGVR, snapshotContentName, 1*time.Second /* poll */, 30*time.Second /* timeout */)
		framework.ExpectError(err) // should fail deletion check

		// The purpose of this block is to prevent physical snapshotContent leaks.
		// We must update the SnapshotContent to have Delete Deletion policy,
		// or else the physical snapshot content will be leaked.
		ginkgo.By("get the latest copy of volume snapshot content")
		snapshotContent, err := l.dc.Resource(SnapshotContentGVR).Get(context.TODO(), l.vscontent.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("updating the the SnapshotContent to have Delete Deletion policy")
		snapshotContent.Object["spec"].(map[string]interface{})["deletionPolicy"] = "Delete"
		l.vscontent, err = l.dc.Resource(SnapshotContentGVR).Update(context.TODO(), snapshotContent, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("manually deleting the SnapshotContent")
		err = l.dc.Resource(SnapshotContentGVR).Delete(context.TODO(), snapshotContent.GetName(), metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting snapshot content %q. Error: %v", snapshotContent.GetName(), err)
		}

		ginkgo.By("checking the SnapshotContent has been deleted")
		err = utils.WaitForGVRDeletion(l.dc, SnapshotContentGVR, snapshotContentName, framework.Poll, framework.SnapshotDeleteTimeout)
		framework.ExpectNoError(err)
	} else {
		framework.Failf("Invalid test config. DeletionPolicy should be either Delete or Retain. DeletionPolicy: %v", sct.DeletionPolicy)
	}
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
