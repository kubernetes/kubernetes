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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

// snapshot CRD api group
const snapshotGroup = "snapshot.storage.k8s.io"

// snapshot CRD api version
const snapshotAPIVersion = "snapshot.storage.k8s.io/v1alpha1"

var (
	snapshotGVR        = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1alpha1", Resource: "volumesnapshots"}
	snapshotClassGVR   = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1alpha1", Resource: "volumesnapshotclasses"}
	snapshotContentGVR = schema.GroupVersionResource{Group: snapshotGroup, Version: "v1alpha1", Resource: "volumesnapshotcontents"}
)

type SnapshotClassTest struct {
	Name           string
	CloudProviders []string
	Snapshotter    string
	Parameters     map[string]string
	NodeName       string
	NodeSelector   map[string]string // NodeSelector for the pod
}

type snapshottableTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &snapshottableTestSuite{}

// InitSnapshottableTestSuite returns snapshottableTestSuite that implements TestSuite interface
func InitSnapshottableTestSuite() TestSuite {
	return &snapshottableTestSuite{
		tsInfo: TestSuiteInfo{
			name: "snapshottable",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DynamicSnapshot,
			},
		},
	}
}

func (s *snapshottableTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *snapshottableTestSuite) defineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		sDriver SnapshottableTestDriver
		dDriver DynamicPVTestDriver
	)

	BeforeEach(func() {
		// Check preconditions.
		Expect(pattern.SnapshotType).To(Equal(testpatterns.DynamicCreatedSnapshot))
		dInfo := driver.GetDriverInfo()
		ok := false
		sDriver, ok = driver.(SnapshottableTestDriver)
		if !dInfo.Capabilities[CapDataSource] || !ok {
			framework.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
		}
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			framework.Skipf("Driver %q does not support dynamic provisioning - skipping", driver.GetDriverInfo().Name)
		}
	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("snapshotting")

	It("should create snapshot with defaults [Feature:VolumeSnapshotDataSource]", func() {
		cs := f.ClientSet
		dc := f.DynamicClient

		// Now do the more expensive test initialization.
		config, testCleanup := driver.PrepareTest(f)
		defer testCleanup()

		vsc := sDriver.GetSnapshotClass(config)
		class := dDriver.GetDynamicProvisionStorageClass(config, "")
		if class == nil {
			framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
		}

		claimSize := dDriver.GetClaimSize()
		pvc := getClaim(claimSize, config.Framework.Namespace.Name)
		pvc.Spec.StorageClassName = &class.Name
		framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", class, pvc)

		By("creating a StorageClass " + class.Name)
		class, err := cs.StorageV1().StorageClasses().Create(class)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(cs.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()

		By("creating a claim")
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting claim %q/%q", pvc.Namespace, pvc.Name)
			// typically this claim has already been deleted
			err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, nil)
			if err != nil && !apierrs.IsNotFound(err) {
				framework.Failf("Error deleting claim %q. Error: %v", pvc.Name, err)
			}
		}()
		err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("checking the claim")
		// Get new copy of the claim
		pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// Get the bound PV
		pv, err := cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		By("creating a SnapshotClass")
		vsc, err = dc.Resource(snapshotClassGVR).Create(vsc, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting SnapshotClass %s", vsc.GetName())
			framework.ExpectNoError(dc.Resource(snapshotClassGVR).Delete(vsc.GetName(), nil))
		}()

		By("creating a snapshot")
		snapshot := getSnapshot(pvc.Name, pvc.Namespace, vsc.GetName())

		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Create(snapshot, metav1.CreateOptions{})
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
			// typically this snapshot has already been deleted
			err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Delete(snapshot.GetName(), nil)
			if err != nil && !apierrs.IsNotFound(err) {
				framework.Failf("Error deleting snapshot %q. Error: %v", pvc.Name, err)
			}
		}()
		err = WaitForSnapshotReady(dc, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
		Expect(err).NotTo(HaveOccurred())

		By("checking the snapshot")
		// Get new copy of the snapshot
		snapshot, err = dc.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		// Get the bound snapshotContent
		snapshotSpec := snapshot.Object["spec"].(map[string]interface{})
		snapshotContentName := snapshotSpec["snapshotContentName"].(string)
		snapshotContent, err := dc.Resource(snapshotContentGVR).Get(snapshotContentName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		snapshotContentSpec := snapshotContent.Object["spec"].(map[string]interface{})
		volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})
		persistentVolumeRef := snapshotContentSpec["persistentVolumeRef"].(map[string]interface{})

		// Check SnapshotContent properties
		By("checking the SnapshotContent")
		Expect(snapshotContentSpec["snapshotClassName"]).To(Equal(vsc.GetName()))
		Expect(volumeSnapshotRef["name"]).To(Equal(snapshot.GetName()))
		Expect(volumeSnapshotRef["namespace"]).To(Equal(snapshot.GetNamespace()))
		Expect(persistentVolumeRef["name"]).To(Equal(pv.Name))
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
				framework.Logf("VolumeSnapshot %s found and is ready", snapshotName, time.Since(start))
				return nil
			} else if value["ready"] == true {
				framework.Logf("VolumeSnapshot %s found and is ready", snapshotName, time.Since(start))
				return nil
			} else {
				framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			}
		}
	}
	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}
