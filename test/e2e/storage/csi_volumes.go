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

package storage

import (
	"context"
	"fmt"
	"regexp"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	csiclient "k8s.io/csi-api/pkg/client/clientset/versioned"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/podlogs"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"crypto/sha256"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// List of testDrivers to be executed in below loop
var csiTestDrivers = []func() drivers.TestDriver{
	drivers.InitHostPathCSIDriver,
	drivers.InitGcePDCSIDriver,
	drivers.InitGcePDExternalCSIDriver,
	drivers.InitHostV0PathCSIDriver,
}

// List of testSuites to be executed in below loop
var csiTestSuites = []func() testsuites.TestSuite{
	testsuites.InitVolumesTestSuite,
	testsuites.InitVolumeIOTestSuite,
	testsuites.InitVolumeModeTestSuite,
	testsuites.InitSubPathTestSuite,
	testsuites.InitProvisioningTestSuite,
}

func csiTunePattern(patterns []testpatterns.TestPattern) []testpatterns.TestPattern {
	tunedPatterns := []testpatterns.TestPattern{}

	for _, pattern := range patterns {
		// Skip inline volume and pre-provsioned PV tests for csi drivers
		if pattern.VolType == testpatterns.InlineVolume || pattern.VolType == testpatterns.PreprovisionedPV {
			continue
		}
		tunedPatterns = append(tunedPatterns, pattern)
	}

	return tunedPatterns
}

// This executes testSuites for csi volumes.
var _ = utils.SIGDescribe("CSI Volumes", func() {
	f := framework.NewDefaultFramework("csi-volumes")

	var (
		cancel context.CancelFunc
		cs     clientset.Interface
		ns     *v1.Namespace
		config framework.VolumeTestConfig
	)

	BeforeEach(func() {
		ctx, c := context.WithCancel(context.Background())
		cancel = c
		cs = f.ClientSet
		ns = f.Namespace
		config = framework.VolumeTestConfig{
			Namespace: ns.Name,
			Prefix:    "csi",
		}
		// Debugging of the following tests heavily depends on the log output
		// of the different containers. Therefore include all of that in log
		// files (when using --report-dir, as in the CI) or the output stream
		// (otherwise).
		to := podlogs.LogOutput{
			StatusWriter: GinkgoWriter,
		}
		if framework.TestContext.ReportDir == "" {
			to.LogWriter = GinkgoWriter
		} else {
			test := CurrentGinkgoTestDescription()
			reg := regexp.MustCompile("[^a-zA-Z0-9_-]+")
			// We end the prefix with a slash to ensure that all logs
			// end up in a directory named after the current test.
			to.LogPathPrefix = framework.TestContext.ReportDir + "/" +
				reg.ReplaceAllString(test.FullTestText, "_") + "/"
		}
		podlogs.CopyAllLogs(ctx, cs, ns.Name, to)

		// pod events are something that the framework already collects itself
		// after a failed test. Logging them live is only useful for interactive
		// debugging, not when we collect reports.
		if framework.TestContext.ReportDir == "" {
			podlogs.WatchPods(ctx, cs, ns.Name, GinkgoWriter)
		}
	})

	AfterEach(func() {
		cancel()
	})

	for _, initDriver := range csiTestDrivers {
		curDriver := initDriver()
		Context(drivers.GetDriverNameWithFeatureTags(curDriver), func() {
			driver := curDriver

			BeforeEach(func() {
				// setupDriver
				drivers.SetCommonDriverParameters(driver, f, config)
				driver.CreateDriver()
			})

			AfterEach(func() {
				// Cleanup driver
				driver.CleanupDriver()
			})

			testsuites.RunTestSuite(f, config, driver, csiTestSuites, csiTunePattern)
		})
	}

	// The CSIDriverRegistry feature gate is needed for this test in Kubernetes 1.12.
	Context("CSI attach test using HostPath driver [Feature:CSIDriverRegistry]", func() {
		var (
			cs     clientset.Interface
			csics  csiclient.Interface
			driver drivers.TestDriver
		)

		BeforeEach(func() {
			cs = f.ClientSet
			csics = f.CSIClientSet
			driver = drivers.InitHostPathCSIDriver()
			drivers.SetCommonDriverParameters(driver, f, config)
			driver.CreateDriver()
		})

		AfterEach(func() {
			driver.CleanupDriver()
		})

		tests := []struct {
			name                   string
			driverAttachable       bool
			driverExists           bool
			expectVolumeAttachment bool
		}{
			{
				name:                   "non-attachable volume does not need VolumeAttachment",
				driverAttachable:       false,
				driverExists:           true,
				expectVolumeAttachment: false,
			},
			{
				name:                   "attachable volume needs VolumeAttachment",
				driverAttachable:       true,
				driverExists:           true,
				expectVolumeAttachment: true,
			},
			{
				name:                   "volume with no CSI driver needs VolumeAttachment",
				driverExists:           false,
				expectVolumeAttachment: true,
			},
		}

		for _, t := range tests {
			test := t
			It(test.name, func() {
				if test.driverExists {
					csiDriver := createCSIDriver(csics, drivers.GetUniqueDriverName(driver), test.driverAttachable)
					if csiDriver != nil {
						defer csics.CsiV1alpha1().CSIDrivers().Delete(csiDriver.Name, nil)
					}
				}

				By("Creating pod")
				var sc *storagev1.StorageClass
				if dDriver, ok := driver.(drivers.DynamicPVTestDriver); ok {
					sc = dDriver.GetDynamicProvisionStorageClass("")
				}
				nodeName := driver.GetDriverInfo().Config.ClientNodeName
				scTest := testsuites.StorageClassTest{
					Name:         driver.GetDriverInfo().Name,
					Provisioner:  sc.Provisioner,
					Parameters:   sc.Parameters,
					ClaimSize:    "1Gi",
					ExpectedSize: "1Gi",
					NodeName:     nodeName,
				}
				class, claim, pod := startPausePod(cs, scTest, ns.Name)
				if class != nil {
					defer cs.StorageV1().StorageClasses().Delete(class.Name, nil)
				}
				if claim != nil {
					defer cs.CoreV1().PersistentVolumeClaims(ns.Name).Delete(claim.Name, nil)
				}
				if pod != nil {
					// Fully delete (=unmount) the pod before deleting CSI driver
					defer framework.DeletePodWithWait(f, cs, pod)
				}
				if pod == nil {
					return
				}

				err := framework.WaitForPodNameRunningInNamespace(cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				By("Checking if VolumeAttachment was created for the pod")
				// Check that VolumeAttachment does not exist
				handle := getVolumeHandle(cs, claim)
				attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, scTest.Provisioner, nodeName)))
				attachmentName := fmt.Sprintf("csi-%x", attachmentHash)
				_, err = cs.StorageV1beta1().VolumeAttachments().Get(attachmentName, metav1.GetOptions{})
				if err != nil {
					if errors.IsNotFound(err) {
						if test.expectVolumeAttachment {
							framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
						}
					} else {
						framework.ExpectNoError(err, "Failed to find VolumeAttachment")
					}
				}
				if !test.expectVolumeAttachment {
					Expect(err).To(HaveOccurred(), "Unexpected VolumeAttachment found")
				}
			})
		}
	})
})

func createCSIDriver(csics csiclient.Interface, name string, attachable bool) *csiv1alpha1.CSIDriver {
	By("Creating CSIDriver instance")
	driver := &csiv1alpha1.CSIDriver{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: csiv1alpha1.CSIDriverSpec{
			AttachRequired: &attachable,
		},
	}
	driver, err := csics.CsiV1alpha1().CSIDrivers().Create(driver)
	framework.ExpectNoError(err, "Failed to create CSIDriver: %v", err)
	return driver
}

func getVolumeHandle(cs clientset.Interface, claim *v1.PersistentVolumeClaim) string {
	// re-get the claim to the latest state with bound volume
	claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PVC")
		return ""
	}
	pvName := claim.Spec.VolumeName
	pv, err := cs.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
	if err != nil {
		framework.ExpectNoError(err, "Cannot get PV")
		return ""
	}
	if pv.Spec.CSI == nil {
		Expect(pv.Spec.CSI).NotTo(BeNil())
		return ""
	}
	return pv.Spec.CSI.VolumeHandle
}

func startPausePod(cs clientset.Interface, t testsuites.StorageClassTest, ns string) (*storagev1.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
	class := newStorageClass(t, ns, "")
	class, err := cs.StorageV1().StorageClasses().Create(class)
	framework.ExpectNoError(err, "Failed to create class : %v", err)
	claim := newClaim(t, ns, "")
	claim.Spec.StorageClassName = &class.Name
	claim, err = cs.CoreV1().PersistentVolumeClaims(ns).Create(claim)
	framework.ExpectNoError(err, "Failed to create claim: %v", err)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "volume-tester",
					Image: imageutils.GetE2EImage(imageutils.Pause),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: claim.Name,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}

	if len(t.NodeName) != 0 {
		pod.Spec.NodeName = t.NodeName
	}
	pod, err = cs.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return class, claim, pod
}
