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
	"crypto/sha256"
	"fmt"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csiclient "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

type volumeAttachmentTestSuite struct {
	tsInfo TestSuiteInfo
}

var _ TestSuite = &volumeAttachmentTestSuite{}

// InitVolumeAttachmentTestSuite returns volumeAttachmentTestSuite that implements TestSuite interface
func InitVolumeAttachmentTestSuite() TestSuite {
	return &volumeAttachmentTestSuite{
		tsInfo: TestSuiteInfo{
			name:       "volumeAttachment",
			featureTag: "",
			testPatterns: []testpatterns.TestPattern{
				testpatterns.DefaultFsDynamicPV,
			},
		},
	}
}

func (s *volumeAttachmentTestSuite) getTestSuiteInfo() TestSuiteInfo {
	return s.tsInfo
}

func (s *volumeAttachmentTestSuite) skipUnsupportedTest(pattern testpatterns.TestPattern, driver drivers.TestDriver) {
}

func createVolumeAttachmentTestInput(pattern testpatterns.TestPattern, resource genericVolumeTestResource) volumeAttachmentTestInput {
	driver := resource.driver
	dInfo := driver.GetDriverInfo()
	sc := resource.sc

	return volumeAttachmentTestInput{
		f:             dInfo.Framework,
		pvc:           resource.pvc,
		nodeName:      dInfo.Config.ClientNodeName,
		driverName:    dInfo.Name,
		provisioner:   sc.Provisioner,
		canSkipAttach: dInfo.CanSkipAttach,
	}
}

func (s *volumeAttachmentTestSuite) execTest(driver drivers.TestDriver, pattern testpatterns.TestPattern) {
	Context(getTestNameStr(s, pattern), func() {
		var (
			resource     genericVolumeTestResource
			input        volumeAttachmentTestInput
			needsCleanup bool
		)

		BeforeEach(func() {
			needsCleanup = false
			// Skip unsupported tests to avoid unnecessary resource initialization
			skipUnsupportedTest(s, driver, pattern)
			needsCleanup = true

			// Setup test resource for driver and testpattern
			resource = genericVolumeTestResource{}
			resource.setupResource(driver, pattern)

			// Create test input
			input = createVolumeAttachmentTestInput(pattern, resource)
		})

		AfterEach(func() {
			if needsCleanup {
				resource.cleanupResource(driver, pattern)
			}
		})

		testVolumeAttachment(&input)
	})
}

type volumeAttachmentTestInput struct {
	f             *framework.Framework
	pvc           *v1.PersistentVolumeClaim
	nodeName      string
	driverName    string
	provisioner   string
	canSkipAttach bool
}

func testVolumeAttachment(input *volumeAttachmentTestInput) {
	It("should create VolumeAttachment", func() {
		f := input.f
		cs := f.ClientSet
		ns := f.Namespace

		By("Creating pod")
		pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{input.pvc}, false, "", false, false, framework.SELinuxLabel, nil, input.nodeName, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred())

		By("Checking if VolumeAttachment was created for the pod")
		pv := getPVFromPVC(cs, input.pvc)
		if pv.Spec.CSI == nil {
			framework.Skipf("In-tree driver has no volumeAttachment implementation, yet- skipping")
		}
		handle := getVolumeHandle(cs, pv)
		attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, input.provisioner, input.nodeName)))
		attachmentName := fmt.Sprintf("csi-%x", attachmentHash)
		_, err = cs.StorageV1beta1().VolumeAttachments().Get(attachmentName, metav1.GetOptions{})
		if err != nil {
			if errors.IsNotFound(err) {
				framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
			} else {
				framework.ExpectNoError(err, "Failed to find VolumeAttachment")
			}
		}
	})

	It("should skip creating VolumeAttachment for drivers that set attachRequired to false [Feature:CSIDriverRegistry]", func() {
		f := input.f
		cs := f.ClientSet
		csics := f.CSIClientSet
		ns := f.Namespace

		// Decide parameters to test by using canSkipAttach
		if !input.canSkipAttach {
			framework.Skipf("Driver %q does not support skip attach - skipping", input.driverName)
		}

		// Create CSI driver with setting attachRequired to false
		driver := createCSIDriver(csics, input.driverName, false)
		if driver != nil {
			defer csics.CsiV1alpha1().CSIDrivers().Delete(driver.Name, nil)
		}

		By("Creating pod")
		pod, err := framework.CreateSecPodWithNodeName(cs, ns.Name, []*v1.PersistentVolumeClaim{input.pvc}, false, "", false, false, framework.SELinuxLabel, nil, input.nodeName, framework.PodStartTimeout)
		defer func() {
			framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))
		}()
		Expect(err).NotTo(HaveOccurred())

		By("Checking if VolumeAttachment wasn't created for the pod")
		pv := getPVFromPVC(cs, input.pvc)
		if pv.Spec.CSI == nil {
			framework.Skipf("In-tree driver does not implement volumeAttachment - skipping", input.driverName)
		}
		handle := getVolumeHandle(cs, pv)
		attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, input.provisioner, input.nodeName)))
		attachmentName := fmt.Sprintf("csi-%x", attachmentHash)
		_, err = cs.StorageV1beta1().VolumeAttachments().Get(attachmentName, metav1.GetOptions{})
		Expect(err).To(HaveOccurred(), "Unexpected VolumeAttachment found")
	})
}

func createCSIDriver(csics csiclient.Interface, driverName string, attachRequired bool) *csiv1alpha1.CSIDriver {
	By("Creating CSIDriver instance")
	podInfoOnMountVersion := "null"
	driver := &csiv1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: driverName,
		},
		Spec: csiv1alpha1.CSIDriverSpec{
			AttachRequired:        &attachRequired,
			PodInfoOnMountVersion: &podInfoOnMountVersion,
		},
	}
	driver, err := csics.CsiV1alpha1().CSIDrivers().Create(driver)
	framework.ExpectNoError(err, "Failed to create CSIDriver: %v", err)
	return driver
}

func getPVFromPVC(cs clientset.Interface, claim *v1.PersistentVolumeClaim) *v1.PersistentVolume {
	// re-get the claim to the the latest state with bound volume
	claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PVC")
		return nil
	}
	pvName := claim.Spec.VolumeName
	pv, err := cs.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PV")
		return nil
	}
	return pv
}

func getVolumeHandle(cs clientset.Interface, pv *v1.PersistentVolume) string {
	if pv.Spec.CSI == nil {
		Expect(pv.Spec.CSI).NotTo(BeNil())
		return ""
	}
	return pv.Spec.CSI.VolumeHandle
}
