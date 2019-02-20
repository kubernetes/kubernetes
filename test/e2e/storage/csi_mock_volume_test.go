/*
Copyright 2019 The Kubernetes Authors.

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
	"crypto/sha256"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type cleanupFuncs func()

var _ = utils.SIGDescribe("CSI Mock volumes", func() {
	type mockDriverSetup struct {
		cs           clientset.Interface
		config       *testsuites.PerTestConfig
		testCleanups []cleanupFuncs
		pods         []*v1.Pod
		pvcs         []*v1.PersistentVolumeClaim
		sc           map[string]*storage.StorageClass
		driver       testsuites.TestDriver
		provisioner  string
	}
	var m mockDriverSetup
	var attachable bool
	var deployCRD bool
	var podInfoVersion *string
	var scName string
	f := framework.NewDefaultFramework("csi-mock-volumes")

	init := func() {
		m = mockDriverSetup{cs: f.ClientSet}
		csics := f.CSIClientSet
		var err error

		m.driver = drivers.InitMockCSIDriver(deployCRD, attachable, podInfoVersion)
		config, testCleanup := m.driver.PrepareTest(f)
		m.testCleanups = append(m.testCleanups, testCleanup)
		m.config = config

		if deployCRD {
			err = waitForCSIDriver(csics, m.config.GetUniqueDriverName())
			framework.ExpectNoError(err, "Failed to get CSIDriver : %v", err)
			m.testCleanups = append(m.testCleanups, func() {
				destroyCSIDriver(csics, m.config.GetUniqueDriverName())
			})
		}
	}

	createPod := func() (*storage.StorageClass, *v1.PersistentVolumeClaim, *v1.Pod) {
		By("Creating pod")
		var sc *storagev1.StorageClass
		if dDriver, ok := m.driver.(testsuites.DynamicPVTestDriver); ok {
			sc = dDriver.GetDynamicProvisionStorageClass(m.config, "")
		}
		m.provisioner = sc.Provisioner
		nodeName := m.config.ClientNodeName
		scTest := testsuites.StorageClassTest{
			Name:         m.driver.GetDriverInfo().Name,
			Provisioner:  sc.Provisioner,
			Parameters:   sc.Parameters,
			ClaimSize:    "1Gi",
			ExpectedSize: "1Gi",
		}
		if scName != "" {
			scTest.StorageClassName = scName
		}
		nodeSelection := testsuites.NodeSelection{
			// The mock driver only works when everything runs on a single node.
			Name: nodeName,
		}
		class, claim, pod := startPausePod(f.ClientSet, scTest, nodeSelection, f.Namespace.Name)
		if class != nil {
			m.sc[class.Name] = class
		}
		if claim != nil {
			m.pvcs = append(m.pvcs, claim)
		}
		if pod != nil {
			m.pods = append(m.pods, pod)
		}
		return class, claim, pod
	}

	resetSharedVariables := func() {
		attachable = false
		deployCRD = false
		scName = ""
		podInfoVersion = nil
	}

	cleanup := func() {
		cs := f.ClientSet
		var errs []error
		By("Deleting pod")
		for _, pod := range m.pods {
			errs = append(errs, framework.DeletePodWithWait(f, cs, pod))
		}

		By("Deleting claim")
		for _, claim := range m.pvcs {
			claim, err := cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(claim.Name, metav1.GetOptions{})
			if err == nil {
				cs.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(claim.Name, nil)
				framework.WaitForPersistentVolumeDeleted(cs, claim.Spec.VolumeName, framework.Poll, 2*time.Minute)
			}

		}

		By("Deleting storageclass")
		for _, sc := range m.sc {
			cs.StorageV1().StorageClasses().Delete(sc.Name, nil)
		}

		By("Cleaning up resources")
		for _, cleanupFunc := range m.testCleanups {
			cleanupFunc()
		}

		// reset some of common variables
		resetSharedVariables()
		err := utilerrors.NewAggregate(errs)
		Expect(err).NotTo(HaveOccurred(), "while cleaning up after test")
	}

	// The CSIDriverRegistry feature gate is needed for this test in Kubernetes 1.12.
	Context("CSI attach test using mock driver [Feature:CSIDriverRegistry]", func() {
		tests := []struct {
			name             string
			driverAttachable bool
			deployDriverCRD  bool
		}{
			{
				name:             "should not require VolumeAttach for drivers without attachment",
				driverAttachable: false,
				deployDriverCRD:  true,
			},
			{
				name:             "should require VolumeAttach for drivers with attachment",
				driverAttachable: true,
				deployDriverCRD:  true,
			},
			{
				name:             "should preserve attachment policy when no CSIDriver present",
				driverAttachable: true,
				deployDriverCRD:  false,
			},
		}
		for _, t := range tests {
			It(t.name, func() {
				deployCRD = t.deployDriverCRD
				attachable = t.driverAttachable
				var err error
				init()
				defer cleanup()

				_, claim, pod := createPod()
				if pod == nil {
					return
				}
				err = framework.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)

				By("Checking if VolumeAttachment was created for the pod")
				handle := getVolumeHandle(m.cs, claim)
				attachmentHash := sha256.Sum256([]byte(fmt.Sprintf("%s%s%s", handle, m.provisioner, m.config.ClientNodeName)))
				attachmentName := fmt.Sprintf("csi-%x", attachmentHash)
				_, err = m.cs.StorageV1beta1().VolumeAttachments().Get(attachmentName, metav1.GetOptions{})
				if err != nil {
					if errors.IsNotFound(err) {
						if t.driverAttachable {
							framework.ExpectNoError(err, "Expected VolumeAttachment but none was found")
						}
					} else {
						framework.ExpectNoError(err, "Failed to find VolumeAttachment")
					}
				}
				if !t.driverAttachable {
					Expect(err).To(HaveOccurred(), "Unexpected VolumeAttachment found")
				}
			})

		}
	})

	Context("CSI workload information using mock driver [Feature:CSIDriverRegistry]", func() {
		var (
			err            error
			podInfoV1      = "v1"
			podInfoUnknown = "unknown"
			podInfoEmpty   = ""
		)
		tests := []struct {
			name                  string
			podInfoOnMountVersion *string
			deployDriverCRD       bool
			expectPodInfo         bool
		}{
			{
				name:                  "should not be passed when podInfoOnMountVersion=nil",
				podInfoOnMountVersion: nil,
				deployDriverCRD:       true,
				expectPodInfo:         false,
			},
			{
				name:                  "should be passed when podInfoOnMountVersion=v1",
				podInfoOnMountVersion: &podInfoV1,
				deployDriverCRD:       true,
				expectPodInfo:         true,
			},
			{
				name:                  "should not be passed when podInfoOnMountVersion=<empty string>",
				podInfoOnMountVersion: &podInfoEmpty,
				deployDriverCRD:       true,
				expectPodInfo:         false,
			},
			{
				name:                  "should not be passed when podInfoOnMountVersion=<unknown string>",
				podInfoOnMountVersion: &podInfoUnknown,
				deployDriverCRD:       true,
				expectPodInfo:         false,
			},
			{
				name:            "should not be passed when CSIDriver does not exist",
				deployDriverCRD: false,
				expectPodInfo:   false,
			},
		}
		for _, t := range tests {
			It(t.name, func() {
				deployCRD = t.deployDriverCRD
				scName = "csi-mock-sc-" + f.UniqueName
				init()
				defer cleanup()

				_, _, pod := createPod()
				if pod == nil {
					return
				}
				err = framework.WaitForPodNameRunningInNamespace(m.cs, pod.Name, pod.Namespace)
				framework.ExpectNoError(err, "Failed to start pod: %v", err)
				By("Checking CSI driver logs")

				// The driver is deployed as a statefulset with stable pod names
				driverPodName := "csi-mockplugin-0"
				err = checkPodInfo(m.cs, f.Namespace.Name, driverPodName, "mock", pod, t.expectPodInfo)
				framework.ExpectNoError(err)
			})
		}
	})
})
