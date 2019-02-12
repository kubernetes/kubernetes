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
	storage "k8s.io/api/storage/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
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
	Name                 string
	CloudProviders       []string
	Snapshotter          string
	Parameters           map[string]string
	NodeName             string
	NodeSelector         map[string]string // NodeSelector for the pod
	SnapshotContentCheck func(snapshotContent *unstructured.Unstructured) error
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

func (s *snapshottableTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver TestDriver) {
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[CapDataSource] {
		framework.Skipf("Driver %q does not support snapshots - skipping", dInfo.Name)
	}
}

func createSnapshottableTestInput(driver TestDriver, pattern testpatterns.TestPattern) (snapshottableTestResource, snapshottableTestInput) {
	// Setup test resource for driver and testpattern
	resource := snapshottableTestResource{}
	resource.setupResource(driver, pattern)

	dInfo := driver.GetDriverInfo()
	input := snapshottableTestInput{
		testCase: SnapshotClassTest{
			NodeName: dInfo.Config.ClientNodeName,
		},
		cs:    dInfo.Config.Framework.ClientSet,
		dc:    dInfo.Config.Framework.DynamicClient,
		pvc:   resource.pvc,
		sc:    resource.sc,
		vsc:   resource.vsc,
		dInfo: dInfo,
	}

	return resource, input
}

func (s *snapshottableTestSuite) execTest(driver TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(s, pattern), func() {
		var (
			resource     snapshottableTestResource
			input        snapshottableTestInput
			needsCleanup bool
		)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(s, driver, pattern)
			needsCleanup = true

			// Create test input
			resource, input = createSnapshottableTestInput(driver, pattern)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		// Ginkgo's "Global Shared Behaviors" require arguments for a shared function
		// to be a single struct and to be passed as a pointer.
		// Please see https://onsi.github.io/ginkgo/#global-shared-behaviors for details.
		testSnapshot(&input)
	})
}

type snapshottableTestResource struct {
	driver    TestDriver
	claimSize string

	sc  *storage.StorageClass
	pvc *v1.PersistentVolumeClaim
	// volume snapshot class
	vsc *unstructured.Unstructured
}

var _ TestResource = &snapshottableTestResource{}

func (s *snapshottableTestResource) setupResource(driver TestDriver, pattern testpatterns.TestPattern) {
	// Setup snapshottableTest resource
	switch pattern.SnapshotType {
	case testpatterns.DynamicCreatedSnapshot:
		if dDriver, ok := driver.(DynamicPVTestDriver); ok {
			s.sc = dDriver.GetDynamicProvisionStorageClass("")
			if s.sc == nil {
				framework.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", driver.GetDriverInfo().Name)
			}
			s.driver = driver
			s.claimSize = dDriver.GetClaimSize()
			s.pvc = getClaim(s.claimSize, driver.GetDriverInfo().Config.Framework.Namespace.Name)
			s.pvc.Spec.StorageClassName = &s.sc.Name
			framework.Logf("In creating storage class object and pvc object for driver - sc: %v, pvc: %v", s.sc, s.pvc)

			if sDriver, ok := driver.(SnapshottableTestDriver); ok {
				s.vsc = sDriver.GetSnapshotClass()
			}
		}

	default:
		framework.Failf("Dynamic Snapshot test doesn't support: %s", pattern.SnapshotType)
	}
}

func (s *snapshottableTestResource) cleanupResource(driver TestDriver, pattern testpatterns.TestPattern) {
}

type snapshottableTestInput struct {
	testCase SnapshotClassTest
	cs       clientset.Interface
	dc       dynamic.Interface
	pvc      *v1.PersistentVolumeClaim
	sc       *storage.StorageClass
	// volume snapshot class
	vsc   *unstructured.Unstructured
	dInfo *DriverInfo
}

func testSnapshot(input *snapshottableTestInput) {
	It("should create snapshot with defaults [Feature:VolumeSnapshotDataSource]", func() {
		TestCreateSnapshot(input.testCase, input.cs, input.dc, input.pvc, input.sc, input.vsc)
	})
}

// TestCreateSnapshot tests dynamic creating snapshot with specified SnapshotClassTest and snapshotClass
func TestCreateSnapshot(
	t SnapshotClassTest,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	claim *v1.PersistentVolumeClaim,
	class *storage.StorageClass,
	snapshotClass *unstructured.Unstructured,
) *unstructured.Unstructured {
	var err error
	if class != nil {
		By("creating a StorageClass " + class.Name)
		class, err = client.StorageV1().StorageClasses().Create(class)
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			framework.ExpectNoError(client.StorageV1().StorageClasses().Delete(class.Name, nil))
		}()
	}

	By("creating a claim")
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(claim)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil)
		if err != nil && !apierrs.IsNotFound(err) {
			framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the claim")
	// Get new copy of the claim
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(claim.Spec.VolumeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	By("creating a SnapshotClass")
	snapshotClass, err = dynamicClient.Resource(snapshotClassGVR).Create(snapshotClass, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		framework.Logf("deleting SnapshotClass %s", snapshotClass.GetName())
		framework.ExpectNoError(dynamicClient.Resource(snapshotClassGVR).Delete(snapshotClass.GetName(), nil))
	}()

	By("creating a snapshot")
	snapshot := getSnapshot(claim.Name, claim.Namespace, snapshotClass.GetName())

	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Create(snapshot, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		framework.Logf("deleting snapshot %q/%q", snapshot.GetNamespace(), snapshot.GetName())
		// typically this snapshot has already been deleted
		err = dynamicClient.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Delete(snapshot.GetName(), nil)
		if err != nil && !apierrs.IsNotFound(err) {
			framework.Failf("Error deleting snapshot %q. Error: %v", claim.Name, err)
		}
	}()
	err = WaitForSnapshotReady(dynamicClient, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the snapshot")
	// Get new copy of the snapshot
	snapshot, err = dynamicClient.Resource(snapshotGVR).Namespace(snapshot.GetNamespace()).Get(snapshot.GetName(), metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Get the bound snapshotContent
	snapshotSpec := snapshot.Object["spec"].(map[string]interface{})
	snapshotContentName := snapshotSpec["snapshotContentName"].(string)
	snapshotContent, err := dynamicClient.Resource(snapshotContentGVR).Get(snapshotContentName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	snapshotContentSpec := snapshotContent.Object["spec"].(map[string]interface{})
	volumeSnapshotRef := snapshotContentSpec["volumeSnapshotRef"].(map[string]interface{})
	persistentVolumeRef := snapshotContentSpec["persistentVolumeRef"].(map[string]interface{})

	// Check SnapshotContent properties
	By("checking the SnapshotContent")
	Expect(snapshotContentSpec["snapshotClassName"]).To(Equal(snapshotClass.GetName()))
	Expect(volumeSnapshotRef["name"]).To(Equal(snapshot.GetName()))
	Expect(volumeSnapshotRef["namespace"]).To(Equal(snapshot.GetNamespace()))
	Expect(persistentVolumeRef["name"]).To(Equal(pv.Name))

	// Run the checker
	if t.SnapshotContentCheck != nil {
		err = t.SnapshotContentCheck(snapshotContent)
		Expect(err).NotTo(HaveOccurred())
	}

	return snapshotContent
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
