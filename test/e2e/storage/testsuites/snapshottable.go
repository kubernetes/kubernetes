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
	"fmt"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

// snapshot CRD api group
const snapshotGroup = "snapshot.storage.k8s.io"

// snapshot CRD api version
const snapshotAPIVersion = "snapshot.storage.k8s.io/v1beta1"

var (
	snapshotGVR        = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshots"}
	snapshotClassGVR   = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotclasses"}
	snapshotContentGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1beta1", Resource: "volumesnapshotcontents"}
)

type snapshottableTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &snapshottableTestSuite{}

// InitSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
func InitSnapshottableTestSuite() TestSuite {
	return &snapshottableTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "snapshottable",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.DynamicSnapshot,
			},
			SupportedSizeRange: volume.SizeRange{
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
	var (
		sDriver SnapshottableTestDriver
		dDriver DynamicPVTestDriver
	)

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

	ginkgo.It("should create snapshot with defaults [Feature:VolumeSnapshotDataSource]", func() {
		cs := f.ClientSet
		dc := f.DynamicClient

		// Now do the more expensive test initialization.
		config, driverCleanup := driver.PrepareTest(f)
		defer func() {
			err := tryFunc(driverCleanup)
			framework.ExpectNoError(err, "while cleaning up driver")
		}()

		vsc := sDriver.GetSnapshotClass(config)
		class := dDriver.GetDynamicProvisionStorageClass(config, "")
		if class == nil {
			e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
		}
		testVolumeSizeRange := s.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
		pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(class.Name),
		}, config.Framework.Namespace.Name)

		framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", class, pvc)

		ginkgo.By("creating a StorageClass " + class.Name)
		class, err = cs.StorageV1().StorageClasses().Create(class)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(cs.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()

		ginkgo.By("creating a claim")
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting claim %q/%q", pvc.Namespace, pvc.Name)
			// typically this claim has already been deleted
			err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, nil)
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting claim %q. Error: %v", pvc.Name, err)
			}
		}()

		ginkgo.By("starting a pod to use the claim")
		command := "echo 'hello world' > /mnt/test/data"
		pod := StartInPodWithVolume(cs, pvc.Namespace, pvc.Name, "pvc-snapshottable-tester", command, e2epod.NodeSelection{Name: config.ClientNodeName})
		defer StopPod(cs, pod)

		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the claim")
		// Get new copy of the claim
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound PV
		_, err = cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("creating a SnapshotClass")
		vsc, err = dc.Resource(snapshotClassGVR).Create(vsc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting SnapshotClass %s", vsc.GetName())
			framework.ExpectNoError(dc.Resource(snapshotClassGVR).Delete(vsc.GetName(), nil))
		}()

		ginkgo.By("creating a snapshot")
		snapshot := getSnapshot(pvc.Name, pvc.Namespace, vsc.GetName())

		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Create(snapshot, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		defer func() {
			framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
			// typically this snapshot has already been deleted
			err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Delete(snapshot.GetName(), nil)
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting snapshot %q. Error: %v", pvc.Name, err)
			}
		}()
		err = WaitForSnapshotReady(dc, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("checking the snapshot")
		// Get new copy of the snapshot
		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
		framework.ExpectNoError(err)

		// Get the bound snapshotContent
		snapshotStatus := snapshot.Object["status"].(map[string]interface{})
		snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
		snapshotContent, err := dc.Resource(snapshotContentGVR).Get(snapshotContentName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		snapshotContentSpec := snapshotContent.Object["spec"].(map[string]interface{})
		volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})

		// Check SnapshotContent properties
		ginkgo.By("checking the SnapshotContent")
		framework.ExpectEqual(snapshotContentSpec["volumeSnapshotClassName"], vsc.GetName())
		framework.ExpectEqual(volumeSnapshotRef["name"], snapshot.GetName())
		framework.ExpectEqual(volumeSnapshotRef["namespace"], snapshot.GetNamespace())
	})
}

// WaitForSnapshotReady waits for a VolumeSnapshot to be ready to use or until timeout occurs, whichever comes first.
func WaitForSnapshotReady(c dynamic.Interface, ns string, snapshotName string, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for VolumeSnapshot %s to become ready", timeout, snapshotName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		snapshot, err := c.Resource(snapshotGVR).Namespace(ns).Get(snapshotName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get claim %q, retrying in %v. Error: %v", snapshotName, Poll, err)
			continue
		} else {
			status := snapshot.Object["status"]
			if status == nil {
				framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
				continue
			}
			value := status.(map[string]interface{})
			if value["readyToUse"] == true {
				framework.Logf("VolumeSnapshot %s found and is ready after %v", snapshotName, time.Since(start))
				return nil
			} else if value["ready"] == true {
				framework.Logf("VolumeSnapshot %s found and is ready after %v", snapshotName, time.Since(start))
				return nil
			} else {
				framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			}
		}
	}
	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}
